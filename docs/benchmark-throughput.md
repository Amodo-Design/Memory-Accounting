# Throughput benchmark

To reproduce the five stages and workload used in the blog post, use the [appendix performance launcher](../experiments/2026-11-24-256/README.md). To change the workload or run one stage, use `scripts/bench_full.py` on the prover with the same checkout on the verifier. The driver uses SSH and rsync.

```sh
export POSM_VDE_BITS=89 POSM_VDE_TWEAK=element POSM_VDE_LAYOUT=packed128
PYTHONPATH=. .venv/bin/python scripts/bench_full.py --help
```

`--breakdown` starts a fresh model process for each stage. `--breakdown-stages` selects a subset that includes `full`. The default order is `vanilla`, `settings`, `encoded`, `dynamic`, `full`. `--prover-args` sets the inference workload. `--verifier-args` sets challenge and calibration options. Use a new `--out-dir` for each run.

Checks are enabled by default. For diagnostics, `--skip-response-checks` skips verifier digest checks and evidence fetches. The prover still captures evidence unless `--no-record-capture` is also set. The appendix throughput run enabled checks and capture.

## Outputs

- `breakdown.csv` gives generated tokens per second, elapsed time and challenge counts for each stage and batch size.
- The stage JSON files record measured calls. `full.json` contains the `reproduction` object with the hosts’ parsed options, model and package versions, codec, hardware, scheduler and calibration. `round_schedule` records the configured wait and pool selection probabilities.
- `challenges.jsonl` is the verifier event log. The output directory also contains process logs and a source snapshot. These generated artifacts can include hostnames, local paths, model material or run keys. Keep them outside the source release.

## Managing the processes separately

The underlying interfaces are `scripts/bench_throughput.py full` on the prover and `python -m memory_accounting.verifier.net live` on the verifier. For a result containing both configurations, set `--benchmark-ready-file` on the verifier. Once calibration completes, copy that JSON to a temporary file on the prover and rename it to the path supplied to `--go-file`. The benchmark checks the manifest digest and mode before measuring. The managed driver handles this exchange automatically.
