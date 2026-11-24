"""Measure six Qwen 32B static and dynamic timing conditions, optionally with zero-cheat controls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time

from bench_full import ROOT, VERIFIER_DEFAULTS, check_running, prepare_registered_weights
from memory_accounting.core.wire import CONTROL_POSTURES


POSTURES = "honest,encode_otf@0.01,partial@0.01,other_gpu@0.01,dram_copy@0.01,cpu@0.01"
MODEL_REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"


def wait_for(log, marker, *processes, timeout):
    deadline = time.monotonic() + timeout
    with log.open() as stream:
        while time.monotonic() < deadline:
            if any(marker in line for line in stream):
                return
            check_running(*processes)
            time.sleep(2)
    raise TimeoutError(f"{log}: did not see {marker!r} within {timeout}s")


def run(args):
    postures = POSTURES + ("," + ",".join(CONTROL_POSTURES) if args.controls else "")
    cycle_postures = args.cycle_postures or postures
    offered = "cpu," + postures
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.reuse_prover_bundle:
        if not args.reuse_verifier_run or not args.reuse_image_sha256:
            raise ValueError("reuse of a prover bundle requires a verifier run and an observed image SHA-256")
        (out / "bundle").symlink_to(Path(args.reuse_prover_bundle).resolve(), target_is_directory=True)
    with tarfile.open(out / "benchmark-source.tar.gz", "w:gz") as archive:
        from memory_accounting.core.benchmark import benchmark_sources
        for path in benchmark_sources(ROOT):
            archive.add(path, arcname=path.relative_to(ROOT))
        for name in ("qwen_timing.py", "reproduce_timing.py"):
            archive.add(ROOT / "scripts" / name, arcname="scripts/" + name)
    ssh = ["ssh", "-o", "BatchMode=yes", "--", args.verifier_ssh]

    def remote(cmd):
        return subprocess.check_output([*ssh, cmd], text=True).strip()

    repo = remote("cd " + shlex.quote(args.verifier_repo) + " && pwd -P")
    run_dir = remote("mktemp -d " + shlex.quote(str(Path(repo).parent / "timing.XXXXXXXX")))
    (out / "verifier-run-path.txt").write_text(run_dir + "\n")
    cwd = "cd " + shlex.quote(run_dir) + " && "
    python = ["env", f"PYTHONPATH={repo}", "POSM_VDE_BITS=89", "POSM_VDE_TWEAK=element",
              "POSM_VDE_LAYOUT=packed128",
              str(Path(repo) / ".venv/bin/python")]
    cli = [*python, "-m", "memory_accounting.setup.cli"]
    if args.reuse_verifier_run:
        from memory_accounting.setup.registration import encoding_key, load_receipt

        old = shlex.quote(args.reuse_verifier_run)
        remote(cwd + f"cp {old}/weights.json {old}/seed-receipt.json .")
        for name in ("weights.json", "seed-receipt.json"):
            (out / name).write_text(remote(cwd + "cat " + shlex.quote(name)) + "\n")
        receipt = load_receipt(out / "seed-receipt.json", out / "weights.json")
        seed, key = receipt["filler_seed"], f"{encoding_key(receipt, 0):032x}"
        registration = ["--weight-reference", args.verifier_weight_reference,
                        "--weight-registration", "weights.json", "--weight-seed-receipt", "seed-receipt.json"]
    else:
        seed, key, registration = prepare_registered_weights(
            remote, cwd, cli, args.verifier_weight_reference, out
        )
    print(f"Verifier run: {args.verifier_ssh}:{run_dir}", flush=True)
    print(f"Prover results: {out}", flush=True)
    (out / "plan.json").write_text(json.dumps({
        "model": "Qwen/Qwen2.5-32B-Instruct", "revision": MODEL_REVISION,
        "codec": "p89_element_packed128", "postures": postures.split(","),
        "controls": args.controls,
        "cycle_postures": cycle_postures.split(","), "timing_phases": args.timing_phases,
        "rounds": args.rounds, "slot_seconds": args.slot_seconds,
        "timing_concurrency": args.timing_concurrency,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "skip_figures": args.skip_figures,
        "timing_prompt_tokens": 256, "timing_generated_tokens": 48,
        "pause_between_request_batches_s": 1.5,
        "vllm_max_num_seqs": args.timing_concurrency if args.timing_concurrency >= 256 else "vllm_default",
        "vllm_max_num_batched_tokens": 2048,
        "warmup": args.warmup, "calibrate_per_kind": args.calibrate,
        "block_bytes": 1 << 20, "static_links": 128, "dynamic_links": args.dynamic_links,
        "kernel_timestamps_required": True, "correctness_fraction": .02,
        "share": "seeded contiguous aligned interval in weights; 1% of all challenged static blocks",
        "dynamic_share": "same five postures on one block-aligned 1% span of the encoded KV region, beginning at low page IDs in the final KV store; the active hidden source is refreshed on posture changes and after live KV writes or page resets",
    }, indent=2) + "\n")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), POSM_VDE_BITS="89", POSM_VDE_TWEAK="element",
               POSM_VDE_LAYOUT="packed128",
               HF_HUB_OFFLINE="1")
    prover_cmd = [sys.executable, "-u", "-m", "memory_accounting.prover.vllm_serve",
                  "--model", "Qwen/Qwen2.5-32B-Instruct", "--model-revision", MODEL_REVISION,
                  "--enc-key", key, "--filler-seed", seed, "--bundle", str(out / "bundle"),
                  "--port", "5558", "--dyn-block", "1M", "--filler-gb", "8",
                  "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--verifier-addr", args.verifier_address,
                  "--strategies", offered, "--compact-share-fraction", ".01",
                  "--dynamic-timing-shares",
                  "--timing-workload", "--timing-concurrency", str(args.timing_concurrency),
                  "--request-gap-s", "1.5"]
    if args.timing_phases:
        prover_cmd += ["--timing-phases", args.timing_phases,
                       "--timing-phase-start-file", str(out / "timing-phase-start")]
    if args.reuse_prover_bundle:
        prover_cmd.append("--reuse-bundle")
    timing_defaults = VERIFIER_DEFAULTS.copy()
    scratch_flag = timing_defaults.index("--max-scratch")
    del timing_defaults[scratch_flag:scratch_flag + 2]  # timing intentionally caps filler at 8 GiB
    strategy_flag = timing_defaults.index("--strategies")
    timing_defaults[strategy_flag + 1] = postures  # the UI must permit every measured switch
    timing_defaults[timing_defaults.index("--dyn-n") + 1] = str(args.dynamic_links)
    verifier_args = [*python, "-u", "-m", "memory_accounting.verifier.net", "live", *timing_defaults,
                     "--interval", ".125", "--round-mix", "equal", "--warmup", str(args.warmup),
                     "--calibrate", str(args.calibrate), "--check-fraction", ".02",
                     "--dynamic-check-fraction", ".02", "--claim-F", ".01",
                     "--claim-C", ".999", "--claim-mode", "window", "--claim-drift-us", "0",
                     "--claim-max-fill-s", "4200", "--claim-delta-c", "15.9",
                     "--claim-delta-c-dyn", "16.4", "--ui-port", "8765",
                     "--filler-seed", seed, *registration, "--port", "5558",
                     "--host", args.prover_address, "--manifest", "bundle/manifest.json",
                     "--evidence-dir", "evidence", "--save-baseline", "baseline.json",
                     "--out", "demo_live.jsonl"]
    verifier = None
    with (out / "prover.log").open("w") as plog, (out / "verifier.log").open("w") as vlog:
        prover = subprocess.Popen(prover_cmd, cwd=out, env=env, stdout=plog,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_for(out / "prover.log", "posm responder on", prover, timeout=3600)
            if args.reuse_verifier_run:
                old_manifest = json.loads(remote("cat " + shlex.quote(str(Path(args.reuse_verifier_run) / "bundle/manifest.json"))))
                new_manifest = json.loads((out / "bundle/manifest.json").read_text())
                for manifest in (old_manifest, new_manifest):
                    manifest.pop("created", None)
                    manifest.pop("commit_key", None)
                    for gpu in manifest["gpus"]:
                        gpu.pop("commit_root", None)
                        gpu.pop("image_root", None)
                if old_manifest != new_manifest:
                    raise RuntimeError("reused verifier bundle has incompatible model, seed, or layout")
                if args.reuse_image_sha256:
                    def digest(path):
                        with path.open("rb") as stream:
                            return hashlib.file_digest(stream, "sha256").hexdigest()
                    local_digest = digest(out / "bundle/image_gpu0.npy")
                    remote_digest = remote("sha256sum " + shlex.quote(str(Path(args.reuse_verifier_run) / "bundle/image_gpu0.npy"))).split()[0]
                    if local_digest != args.reuse_image_sha256 or remote_digest != local_digest:
                        raise RuntimeError("reused verifier image bytes do not match the recorded SHA-256")
                else:
                    raise RuntimeError("--reuse-verifier-run requires --reuse-image-sha256")
                remote(cwd + "ln -s " + shlex.quote(str(Path(args.reuse_verifier_run) / "bundle")) + " bundle")
            else:
                remote(cwd + "mkdir bundle")
                subprocess.run(["rsync", "-a", "--protect-args", str(out / "bundle") + "/",
                                f"{args.verifier_ssh}:{run_dir}/bundle/"], check=True)
            verifier_cmd = cwd + "echo $$ > verifier.pid && exec " + shlex.join(verifier_args)
            verifier = subprocess.Popen([*ssh, verifier_cmd], stdout=vlog, stderr=subprocess.STDOUT)
            wait_for(out / "verifier.log", "calibrated:", prover, verifier, timeout=7200)
            if args.timing_phases:
                (out / "timing-phase-start").touch()
            cycle_args = [*python, "-u", str(Path(repo) / "scripts/posture_cycle.py"),
                          "--dir", run_dir, "--name", "qwen_p89_packed128_six", "--rounds", str(args.rounds),
                          "--slot", str(args.slot_seconds), "--postures", cycle_postures,
                          "--url", "http://127.0.0.1:8765/posture"]
            with (out / "cycle.log").open("w") as clog:
                cycle = subprocess.Popen([*ssh, cwd + shlex.join(cycle_args)], stdout=clog,
                                         stderr=subprocess.STDOUT)
                while cycle.poll() is None:
                    check_running(prover, verifier)
                    time.sleep(2)
                if cycle.returncode:
                    raise RuntimeError(f"posture cycle failed with {cycle.returncode}")
        finally:
            if verifier is not None and verifier.poll() is None:
                remote(cwd + 'test ! -s verifier.pid || kill -TERM "$(cat verifier.pid)"')
                verifier.wait(timeout=60)
            # vLLM may leave its EngineCore alive after the driver exits. The
            # child has its own process group, so clean up the whole engine.
            try:
                os.killpg(prover.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                prover.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(prover.pid, signal.SIGKILL)
                prover.wait(timeout=10)
    for name in ("demo_live.jsonl", "baseline.json", "weights.json", "seed-receipt.json"):
        try:
            with (out / name).open("wb") as dest:
                subprocess.run([*ssh, cwd + "cat " + shlex.quote(name)], stdout=dest, check=True)
        except subprocess.CalledProcessError:
            (out / name).unlink(missing_ok=True)
    if (out / "demo_live.jsonl").is_file() and not args.skip_figures:
        with (out / "figure.log").open("w") as flog:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/reproduce_timing.py"), str(out / "demo_live.jsonl"),
                 "--out", str(out / "figures")], stdout=flog, stderr=subprocess.STDOUT,
            )
        if result.returncode:
            print(f"Figure generation returned {result.returncode}; see {out / 'figure.log'}", flush=True)
    print(f"Timing run complete: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verifier-ssh", required=True)
    p.add_argument("--verifier-repo", required=True)
    p.add_argument("--verifier-weight-reference", required=True)
    p.add_argument("--reuse-verifier-run", default=None,
                   help="reuse an existing verifier seed and bundle after checking the new export manifest")
    p.add_argument("--reuse-prover-bundle", default=None,
                   help="diagnostic: skip re-export using an existing local encoded image")
    p.add_argument("--reuse-image-sha256", default=None,
                   help="SHA-256 of the previously checked encoded image, required for verifier reuse")
    p.add_argument("--prover-address", required=True)
    p.add_argument("--verifier-address", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--calibrate", type=int, default=3000)
    p.add_argument("--rounds", type=int, default=23)
    p.add_argument("--slot-seconds", type=float, default=40)
    p.add_argument("--controls", action="store_true", help="measure five zero-cheat simulator controls as well")
    p.add_argument("--cycle-postures", default=None,
                   help="subset of offered postures to cycle; diagnostic runs may repeat honest only")
    p.add_argument("--timing-phases", default=None,
                   help="diagnostic workload phases passed to vllm_serve")
    p.add_argument("--timing-concurrency", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--dynamic-links", type=int, default=64)
    p.add_argument("--skip-figures", action="store_true", help="keep short-check setup fast; plot the full run later")
    args = p.parse_args()
    if args.timing_concurrency < 1:
        p.error("--timing-concurrency must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        p.error("--gpu-memory-utilization must be between zero and one")
    if args.dynamic_links < 1:
        p.error("--dynamic-links must be positive")
    if args.cycle_postures and (not set(args.cycle_postures.split(",")) <=
                                set((POSTURES + ("," + ",".join(CONTROL_POSTURES) if args.controls else "")).split(","))):
        p.error("--cycle-postures must use offered postures")
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    run(args)


if __name__ == "__main__":
    main()
