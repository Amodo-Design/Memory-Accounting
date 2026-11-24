"""Provenance shared by the two hosts of a throughput run."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .codec_profile import FIELD_BITS, GROUP_ENC, GROUP_RAW, PROFILE, TWEAK


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def benchmark_sources(root):
    # Includes uncommitted experiments and works in a deployed tree without .git.
    files = [p for p in (root / "memory_accounting").rglob("*") if p.suffix in {".py", ".c", ".h", ".cu", ".cpp", ".html", ".metal"}]
    for relative in ("scripts/bench_throughput.py", "scripts/bench_full.py", "pyproject.toml", "uv.lock"):
        path = root / relative
        if path.is_file():
            files.append(path)
    return sorted(files)


def provenance():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in benchmark_sources(root):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(file_digest(path)))
    revision = None
    if (root / ".git").exists():
        try:
            revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                               text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    packages = {}
    for name in ("vllm", "torch", "triton", "transformers", "numpy", "scipy", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    env = ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
           "VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_KV_CACHE_LAYOUT", "POSM_VLLM_ATTN",
           "TORCH_CUDA_ARCH_LIST", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "HF_HUB_OFFLINE")
    cpu = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and ":" in line:
                cpu = line.partition(":")[2].strip()
                break
    return {
        "cpu": cpu, "logical_cpus": os.cpu_count(),
        "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
        "git_revision": revision, "source_sha256": digest.hexdigest(), "packages": packages,
        "environment": {k: os.environ[k] for k in env if k in os.environ},
        "codec": {"profile": PROFILE, "prime": str((1 << FIELD_BITS) - 1), "tweak": TWEAK,
                  "plaintext_words_per_group": GROUP_RAW, "encoded_words_per_group": GROUP_ENC},
    }


def read_verifier_handoff(go_file, manifest, skip_checks):
    contents = Path(go_file).read_text()
    if not contents.strip():
        if skip_checks:
            raise ValueError("unchecked benchmarking requires the verifier's calibrated JSON handoff")
        # Preserve the existing externally managed, touch-based benchmark interface.
        return {"settings_available": False, "reason": "legacy empty go-file"}
    config = json.loads(contents)
    if config.get("schema") != "posm-throughput-verifier-v1":
        raise ValueError("unexpected verifier benchmark handoff format")
    if config.get("manifest_sha256") != file_digest(manifest):
        raise ValueError("verifier handoff belongs to a different exported manifest")
    if config["response_checks"] != (not skip_checks):
        raise ValueError("prover and verifier selected different response-check modes")
    return config
