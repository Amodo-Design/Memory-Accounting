"""Run the full two-host throughput benchmark from the prover, collecting both hosts' settings.

Set POSM_VDE_BITS=89, POSM_VDE_TWEAK=element and POSM_VDE_LAYOUT=packed128
for 11 independently tweaked values per 128-byte storage row.

Use the prover's vLLM Python environment. The verifier needs the same checkout, a CPU
.venv, ssh access from the prover and enough disk for the checked reference bundle.
See docs/benchmark-throughput.md for examples.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROVER_DEFAULTS = ["--model", "Qwen/Qwen2.5-32B-Instruct", "--model-revision", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd", "--batches", "256", "--prompt-len", "256",
                   "--gen-len", "256", "--calls", "3", "--gpu-memory-utilization", "0.75"]
VERIFIER_DEFAULTS = [
    "--seed", "15385", "--n", "128", "--dyn-n", "64", "--check-fraction", "1", "--drain", "--dynamic", "--device", "cpu",
    "--dyn-B", "1M", "--gpu-type", "H200-NVL", "--dyn-encoded", "--max-scratch", "24G", "--strategies", "honest",
    "--deadline-rule", "p999", "--margin", "0.05", "--mode", "monitor", "--warmup", "200", "--calibrate", "1000",
    "--round-mix", "pool-bytes", "--interval", ".5", "--claim-F", ".2", "--claim-C", ".999", "--claim-mode", "window", "--claim-drift-us", "0",
    "--challenge-flags", "alert", "--claim-period-s", "3600", "--claim-max-fill-s", "1800",
    "--evidence-dir", "evidence", "--evidence-max-total", "20G", "--ui-port", "8768",
    "--ui-bind", "127.0.0.1", "--timing", "kernel",
]


BREAKDOWN_STAGES = (
    ("vanilla", "Stock vLLM"),
    ("settings", "Deterministic inference"),
    ("encoded", "Add weight decoding"),
    ("dynamic", "Add dynamic decoding and encoding"),
    ("full", "Add static and whole-pool dynamic challenges"),
)


def write_breakdown(out, selected_stages):
    rows = []
    previous = {}
    for variant, label in BREAKDOWN_STAGES:
        if variant not in selected_stages:
            continue
        result = json.loads((out / f"{variant}.json").read_text())
        for batch in result["batches"]:
            size = batch["batch"]
            rate = batch["tok_s"]
            counts = batch["served"] or {"static": 0, "dynamic": 0}
            gc_calls = [call["diagnostics"]["worker_gc_during_call"]
                        for call in batch["calls"] if call.get("diagnostics")]
            rows.append({
                "stage": variant, "description": label, "profile": result["vde_profile"], "batch": size,
                "generated_tokens": sum(call["tokens"] for call in batch["calls"]),
                "measured_seconds": batch["seconds"], "tokens_per_second": rate,
                "change_from_previous_pct": (100 * (rate / previous[size] - 1)) if size in previous else None,
                "static_responses": counts["static"], "dynamic_responses": counts["dynamic"],
                "worker_gc_collections": sum(gc["collections"] for gc in gc_calls) if gc_calls else None,
                "worker_gc_ms": sum(gc["total_ms"] for gc in gc_calls) if gc_calls else None,
                "worker_gc_objects_collected": sum(gc["objects_collected"] for gc in gc_calls) if gc_calls else None,
            })
            previous[size] = rate
            print(f"{label} (batch {size}): {rate:.1f} generated tokens/s")
    with (out / "breakdown.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_running(*processes):
    for process in processes:
        status = process.poll()
        if status is not None:
            raise RuntimeError(f"benchmark process exited before it was ready (status {status}); see run logs")


def prepare_registered_weights(remote, cwd, cli, reference, out):
    from memory_accounting.setup.registration import encoding_key, load_receipt

    remote(cwd + shlex.join([*cli, "register-stream", "--reference", reference, "--out", "weights.json"]))
    remote(cwd + shlex.join([*cli, "issue-seed", "--registration", "weights.json", "--out", "seed-receipt.json"]))
    for name in ("weights.json", "seed-receipt.json"):
        (out / name).write_text(remote(cwd + shlex.join(["cat", name])) + "\n")
    receipt = load_receipt(out / "seed-receipt.json", out / "weights.json")
    flags = ["--weight-reference", reference, "--weight-registration", "weights.json",
             "--weight-seed-receipt", "seed-receipt.json"]
    return receipt["filler_seed"], f"{encoding_key(receipt, 0):032x}", flags


def run(args):
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Archive the exact sources used, including uncommitted CUDA changes.
    sys.path.insert(0, str(ROOT))
    from memory_accounting.core.benchmark import benchmark_sources
    with tarfile.open(out / "benchmark-source.tar.gz", "w:gz") as archive:
        for path in benchmark_sources(ROOT):
            archive.add(path, arcname=path.relative_to(ROOT))
    ssh = ["ssh", "-o", "BatchMode=yes", "--", args.verifier_ssh]

    def remote(command):
        return subprocess.check_output([*ssh, command], text=True).strip()

    repo = remote("cd " + shlex.quote(args.verifier_repo) + " && pwd -P")
    vpython = str(Path(repo) / ".venv/bin/python")
    venv = ["env", f"PYTHONPATH={repo}", f"POSM_VDE_BITS={os.getenv('POSM_VDE_BITS', '89')}",
            f"POSM_VDE_LAYOUT={os.getenv('POSM_VDE_LAYOUT', 'legacy')}",
            f"POSM_VDE_TWEAK={os.getenv('POSM_VDE_TWEAK', 'group64')}"]
    run_dir = remote("cd " + shlex.quote(repo) + ' && mktemp -d "$PWD/throughput.XXXXXXXX"')
    cwd = "cd " + shlex.quote(run_dir) + " && "
    reference = getattr(args, "verifier_weight_reference", None)
    registration_flags, weight_args = [], []
    if reference:
        cli = [*venv, vpython, "-m", "memory_accounting.setup.cli"]
        seed, key, registration_flags = prepare_registered_weights(remote, cwd, cli, reference, out)
        weight_args = ["--enc-key", key]
    else:
        seed = args.filler_seed or remote(shlex.join([*venv, vpython, "-m", "memory_accounting.setup.cli", "filler-seed"]))
    mode = ["--skip-response-checks"] if args.skip_response_checks else []
    capture_mode = ["--no-record-capture"] if args.no_record_capture else []
    common = ["--filler-seed", seed, "--port", str(args.port), *mode]
    bench = [sys.executable, "-u", str(ROOT / "scripts/bench_throughput.py")]
    workload = [*PROVER_DEFAULTS, *args.prover_args, "--filler-seed", seed, *weight_args]
    proof = ["--registered-stream"] if reference else []
    prover_args = [*bench, "full", *workload, "--port", str(args.port), *mode, *capture_mode, *proof, "--bundle", "bundle", "--go-file", "go.json",
                   "--verifier-addr", args.verifier_address, "--out", "full.json"]
    verifier_defaults = VERIFIER_DEFAULTS.copy()
    if args.skip_response_checks:
        scratch_flag = verifier_defaults.index("--max-scratch")
        del verifier_defaults[scratch_flag:scratch_flag + 2]
    verifier_args = [*venv, vpython, "-u", "-m", "memory_accounting.verifier.net", "live", *verifier_defaults,
                     *args.verifier_args, *common, *registration_flags, "--host", args.prover_address, "--manifest", "bundle/manifest.json",
                     "--benchmark-ready-file", "ready.json", "--out", "challenges.jsonl"]
    print(f"Prover results: {out}\nVerifier files: {args.verifier_ssh}:{run_dir}", flush=True)
    verifier = None
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    if args.breakdown:
        for variant, label in BREAKDOWN_STAGES[:-1]:
            if variant not in args.breakdown_stages:
                continue
            print(f"Running {label} ({variant}.log)", flush=True)
            with (out / f"{variant}.log").open("w") as log:
                stage_env = env.copy()
                if variant == "vanilla":
                    for name in ("VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER"):
                        stage_env.pop(name, None)
                subprocess.run([*bench, variant, *workload, "--out", f"{variant}.json"],
                               cwd=out, env=stage_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / "prover.log").open("w") as plog, (out / "verifier.log").open("w") as vlog:
        prover = subprocess.Popen(prover_args, cwd=out, env=env, stdout=plog, stderr=subprocess.STDOUT)
        try:
            with (out / "prover.log").open() as reader:
                while not any(line.startswith("full: waiting for") for line in reader):
                    check_running(prover)
                    time.sleep(1)
            bundle = out / "bundle"
            files = ["manifest.json"]
            if not args.skip_response_checks or reference:
                for gpu in json.loads((bundle / "manifest.json").read_text())["gpus"]:
                    files.extend([gpu["image_file"], gpu["commit_file"]])
            remote(cwd + "mkdir bundle")
            subprocess.run(["rsync", "-a", "--protect-args", "--files-from=-", str(bundle) + "/",
                            f"{args.verifier_ssh}:{run_dir}/bundle/"],
                           input="\n".join(files) + "\n", text=True, check=True)
            launch = cwd + "echo $$ > verifier.pid && exec " + shlex.join(verifier_args)
            verifier = subprocess.Popen([*ssh, launch], stdout=vlog, stderr=subprocess.STDOUT)
            while True:
                check_running(prover, verifier)
                handoff = remote(cwd + "if test -f ready.json; then cat ready.json; fi")
                if handoff:
                    if args.breakdown:
                        options = json.loads(handoff)["options"]
                        if not options["dynamic"] or options["dyn_live"]:
                            raise ValueError("breakdown requires the whole dynamic pool and no separate live-only chain")
                    break
                time.sleep(1)
            tmp = out / "go.tmp"
            tmp.write_text(handoff)
            tmp.replace(out / "go.json")
            while prover.poll() is None:
                check_running(verifier)
                time.sleep(1)
            if prover.returncode:
                raise subprocess.CalledProcessError(prover.returncode, prover_args)
        finally:
            try:
                if verifier is not None and verifier.poll() is None:
                    remote(cwd + 'test ! -s verifier.pid || kill -TERM "$(cat verifier.pid)"')
                    verifier.wait(timeout=60)
            finally:
                if verifier is not None and verifier.poll() is None:
                    verifier.terminate()
                    verifier.wait()
                if prover.poll() is None:
                    prover.terminate()
                    prover.wait()
    with (out / "challenges.jsonl").open("wb") as log:
        subprocess.run([*ssh, cwd + "cat challenges.jsonl"], stdout=log, check=True)
    if args.breakdown:
        write_breakdown(out, args.breakdown_stages)
    print(out / "full.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verifier-ssh", required=True, help="verifier SSH alias as reached from the prover")
    ap.add_argument("--verifier-repo", default="LVHEM_MACC_SW", help="checkout relative to SSH home or absolute")
    ap.add_argument("--prover-address", required=True, help="prover IP on the challenge network")
    ap.add_argument("--verifier-address", required=True, help="verifier IP on the challenge network")
    ap.add_argument("--out-dir", required=True, help="new results directory on the prover")
    ap.add_argument("--breakdown", action="store_true",
                    help="measure deterministic inference, add weight decoding, add dynamic coding, then add challenges")
    ap.add_argument("--breakdown-stages", default="vanilla,settings,encoded,dynamic,full",
                    help="comma-separated stages to measure with --breakdown; full is always required")
    ap.add_argument("--skip-response-checks", action="store_true")
    ap.add_argument("--no-record-capture", action="store_true",
                    help="diagnostic only: disable evidence capture; requires --skip-response-checks")
    ap.add_argument("--filler-seed", help="hex seed to repeat a run; otherwise minted on the verifier")
    ap.add_argument("--verifier-weight-reference", help="absolute path on the verifier to its trusted loaded-weight reference")
    ap.add_argument("--port", default=5558, type=int)
    ap.add_argument("--prover-args", default="", help="additional quoted bench_throughput arguments")
    ap.add_argument("--verifier-args", default="", help="additional quoted live verifier arguments")
    args = ap.parse_args()
    args.breakdown_stages = [part.strip() for part in args.breakdown_stages.split(",")]
    known_stages = {name for name, _ in BREAKDOWN_STAGES}
    if (not args.breakdown_stages or len(set(args.breakdown_stages)) != len(args.breakdown_stages)
            or set(args.breakdown_stages) - known_stages or "full" not in args.breakdown_stages):
        ap.error("--breakdown-stages must list distinct known stages and include full")
    if not args.breakdown and set(args.breakdown_stages) != known_stages:
        ap.error("--breakdown-stages requires --breakdown")
    if args.verifier_weight_reference and (args.filler_seed or not Path(args.verifier_weight_reference).is_absolute()):
        ap.error("registered setup requires an absolute verifier reference and a fresh filler seed")
    if args.no_record_capture and not args.skip_response_checks:
        ap.error("--no-record-capture requires --skip-response-checks")
    managed = {"--skip-response-checks", "--benchmark-ready-file", "--manifest", "--out", "--go-file",
               "--no-record-capture",
               "--bundle", "--verifier-addr", "--host", "--port", "--filler-seed", "--registered-stream",
               "--weight-reference", "--weight-registration", "--weight-seed-receipt"}
    for attr in ("prover_args", "verifier_args"):
        tokens = shlex.split(getattr(args, attr))
        if any(tok.split("=")[0] in managed for tok in tokens):
            ap.error(f"{attr} contains a setting managed by this runner")
        if args.breakdown:
            incompatible = "--plain-dynamic" if attr == "prover_args" else "--with-dyn-live"
            if any(tok.startswith("--") and incompatible.startswith(tok.split("=")[0]) for tok in tokens):
                ap.error(f"{incompatible} conflicts with the four-stage breakdown")
        setattr(args, attr, tokens)

    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    run(args)


if __name__ == "__main__":
    main()
