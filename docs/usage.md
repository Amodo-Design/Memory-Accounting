# Operator guide

Start with the [local CPU example](../README.md#try-it-on-one-computer). The commands below run from the checkout where you installed the corresponding environment.

## CLI entry points

| Task | Command |
| --- | --- |
| Register weights, issue a seed, prepare or check a bundle | `.venv/bin/python -m memory_accounting.setup.cli --help` |
| See the inputs for one setup step | `.venv/bin/python -m memory_accounting.setup.cli verify --help` |
| Start the synthetic prover | `.venv/bin/python -m memory_accounting.prover.udp --help` |
| Run a baseline, a fixed number of checks or the live dashboard | `.venv/bin/python -m memory_accounting.verifier.net --help` |
| Serve a model with vLLM | `.venv/bin/python -m memory_accounting.prover.vllm_serve --help` |
| Replay saved vLLM records | `.venv/bin/python -m memory_accounting.verifier.vllm_replay --help` |

## Live dashboard

Start the CPU prover as in the README. From another terminal, run the verifier against the same array and open <http://127.0.0.1:8080/>. Ctrl-C stops the verifier.

```sh
.venv/bin/python -m memory_accounting.verifier.net live --device cpu --host 127.0.0.1 --port 5555 \
  --size 4M --B 64K --r 0 --n 4 --drain --strategies honest --ui-port 8080
```

Use an SSH tunnel to view a remote verifier’s local dashboard. For a dashboard listening on a LAN interface, configure `POSM_UI_TOKEN` and `--ui-allow-host` as described in the verifier CLI help. For a GPU run, the prover’s `--verifier-addr` takes the verifier’s network address. `--host` on the verifier takes the prover’s address.

## Model runs and saved data

Use the [appendix launchers](../experiments/2026-11-24-256/README.md) for the Qwen runs. For a manually configured verifier, `--save-baseline` saves its calibrated settings and `--baseline` loads them on a subsequent run with the same shape, load and timing settings. `--out` sets the event log and `--evidence-dir` the directory for sampled records. Keep the setup manifest, reference data and verifier log with the matching run for replay.

The separate `scripts/demo.sh` driver supports the integer-model and vLLM demos. Its header lists the host, environment, model and network settings, and the `start`, `status`, `open`, `replay` and `stop` commands. `scripts/run_sweeps.sh` and `scripts/run_load_sweeps.sh` run the parameter sweeps. Their output directories feed the plotting tools in `memory_accounting/experiments/`.
