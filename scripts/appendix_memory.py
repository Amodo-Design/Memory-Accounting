"""Measure Qwen 32B memory with verifier-owned registration and the 128 B codec.

The load phases are idle, 16 concurrent requests and the selected load concurrency.
The challenge chains use 128 static links and a configurable dynamic length
(128 in the 24 November launcher).
"""

from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys
import time

from bench_full import ROOT, VERIFIER_DEFAULTS, check_running, prepare_registered_weights


def wait_for(log, marker, *processes, timeout):
    end = time.monotonic() + timeout
    with log.open() as reader:
        while time.monotonic() < end:
            if any(line.startswith(marker) for line in reader):
                return
            check_running(*processes)
            time.sleep(1)
    raise TimeoutError(f"{log.name} never reported {marker}")


def stop(proc, remote=None):
    if proc is not None and proc.poll() is None:
        if remote is None:
            proc.terminate()
        else:
            remote('test ! -s verifier.pid || kill -TERM "$(cat verifier.pid)"')
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=10)


def run(args):
    sys.path.insert(0, str(ROOT))
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(json.dumps({
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "model_revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
        "codec": "p89_element_packed128",
        "load_concurrency": args.load_concurrency,
        "calibration_load_concurrency": args.load_concurrency,
        "vllm_max_num_seqs": args.load_concurrency,
        "gpu_memory_utilization": 0.75,
        "block_bytes": 1 << 20, "static_links": 128, "dynamic_links": args.dynamic_links,
        "static_check_fraction": .02, "dynamic_record_fraction": .08,
        "warmup_per_kind": args.warmup,
        "calibrate_per_kind": args.calibrate,
        "phase_seconds_minimum": {"idle": args.idle_seconds,
                                  "light_16": args.light_seconds,
                                  "selected_load": args.selected_seconds},
    }, indent=2) + "\n")
    ssh = ["ssh", "-o", "BatchMode=yes", "--", args.verifier_ssh]
    profile = ["POSM_VDE_BITS=89", "POSM_VDE_TWEAK=element", "POSM_VDE_LAYOUT=packed128"]

    def remote(command):
        return subprocess.check_output([*ssh, command], text=True).strip()

    repo = remote("cd " + shlex.quote(args.verifier_repo) + " && pwd -P")
    run_dir = remote("cd " + shlex.quote(repo) + ' && mktemp -d "$PWD/memory.XXXXXXXX"')
    cwd = "cd " + shlex.quote(run_dir) + " && "
    python = ["env", f"PYTHONPATH={repo}", *profile, repo + "/.venv/bin/python"]
    cli = [*python, "-m", "memory_accounting.setup.cli"]
    seed, key, registration = prepare_registered_weights(remote, cwd, cli, args.verifier_weight_reference, out)
    print(f"Verifier files: {args.verifier_ssh}:{run_dir}", flush=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "scripts"))),
               POSM_VDE_BITS="89", POSM_VDE_TWEAK="element", POSM_VDE_LAYOUT="packed128",
               FILLER_SEED=seed, WEIGHT_KEY=key,
               MEMORY_OUT=str(out), VERIFIER_ADDRESS=args.verifier_address,
               MEMORY_LOAD_CONCURRENCY=str(args.load_concurrency),
               MEMORY_IDLE_SECONDS=str(args.idle_seconds),
               MEMORY_LIGHT_SECONDS=str(args.light_seconds),
               MEMORY_SELECTED_SECONDS=str(args.selected_seconds))
    with (out / "prover.log").open("w") as log:
        prover = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/measure32.py")],
                                  cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for(out / "prover.log", "SETUP_DONE", prover, timeout=1800)
            subprocess.run(["rsync", "-a", "--protect-args", str(out / "bundle"),
                            f"{args.verifier_ssh}:{run_dir}/"], check=True)
            options = [
                *VERIFIER_DEFAULTS, "--filler-seed", seed, *registration, "--port", "5558",
                "--host", args.prover_address, "--manifest", "bundle/manifest.json",
                "--round-mix", "equal", "--interval", ".25", "--n", "128", "--dyn-n", str(args.dynamic_links),
                "--warmup", str(args.warmup), "--calibrate", str(args.calibrate),
                "--evidence-dir", "evidence",
                "--check-fraction", ".02", "--dynamic-check-fraction", ".08", "--verify-threads", "32",
                "--save-baseline", "long-baseline.json", "--out", "long.jsonl",
            ]
            cmd = cwd + "echo $$ > verifier.pid && exec " + shlex.join(
                [*python, "-u", "-m", "memory_accounting.verifier.net", "live", *options])
            with (out / "long.log").open("w") as v_log:
                verifier = subprocess.Popen([*ssh, cmd], stdout=v_log, stderr=subprocess.STDOUT)
                try:
                    wait_for(out / "long.log", "calibrated:", prover, verifier, timeout=3600)
                    (out / "go").touch()
                    end = time.monotonic() + 3600
                    while prover.poll() is None and time.monotonic() < end:
                        check_running(verifier)
                        time.sleep(1)
                    if prover.poll() != 0:
                        raise RuntimeError("memory load phases failed or timed out")
                finally:
                    stop(verifier, lambda command: remote(cwd + command))
            for filename in ("long.jsonl", "long-baseline.json"):
                with (out / filename).open("wb") as dest:
                    subprocess.run([*ssh, cwd + f"cat {filename}"], check=True, stdout=dest)
        finally:
            stop(prover)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-ssh", required=True)
    parser.add_argument("--verifier-repo", default="LVHEM_MACC_SW")
    parser.add_argument("--verifier-weight-reference", required=True)
    parser.add_argument("--prover-address", required=True)
    parser.add_argument("--verifier-address", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--load-concurrency", type=int, default=256)
    parser.add_argument("--dynamic-links", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--calibrate", type=int, default=3000)
    parser.add_argument("--idle-seconds", type=float, default=40)
    parser.add_argument("--light-seconds", type=float, default=40)
    parser.add_argument("--selected-seconds", type=float, default=70)
    args = parser.parse_args()
    if args.load_concurrency < 1:
        parser.error("--load-concurrency must be positive")
    if args.dynamic_links < 1:
        parser.error("--dynamic-links must be positive")
    if min(args.warmup, args.calibrate) < 1:
        parser.error("--warmup and --calibrate must be positive")
    if min(args.idle_seconds, args.light_seconds, args.selected_seconds) <= 0:
        parser.error("phase durations must be positive")
    if not Path(args.verifier_weight_reference).is_absolute():
        parser.error("--verifier-weight-reference is an absolute path on the verifier")
    run(args)


if __name__ == "__main__":
    main()
