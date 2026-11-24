# Memory Accounting

Read the [Memory Accounting blog post](https://amododesign.com/notes/2026-11-24-memory-accounting/) for a description of the protocol.

## Installation

Use Python 3.12 or later, a C compiler and [uv](https://docs.astral.sh/uv/). Run these commands from the repository root. Use the same source revision and lockfile on the prover and verifier.

```sh
# CPU verifier
uv sync --frozen --no-default-groups

# Local CPU example (the prover imports PyTorch)
uv sync --frozen --no-default-groups --group gpu

# GPU prover or replay host (Linux x86_64)
uv sync --frozen
```

The CPU library builds on first use. The GPU build also needs `ninja`, a CUDA driver and `nvcc` compatible with the installed PyTorch CUDA build. The experiments used an NVIDIA H200 NVL. A host with the NVIDIA wheels can use [the CUDA toolkit helper](scripts/cuda_home_from_wheels.sh).

## Try it on one computer

This example checks that the prover and verifier exchange challenges and agree on the answers. Use the local CPU installation above. Both processes run on the CPU and use a synthetic 4 MiB array. Start the prover in one terminal.

```sh
.venv/bin/python -m memory_accounting.prover.udp --device cpu --bind 127.0.0.1 --port 5555 --size 4M
```

In a second terminal, run six checks and save the replies. Stop the prover with Ctrl-C when the verifier exits.

```sh
mkdir -p results
.venv/bin/python -m memory_accounting.verifier.net verifier --device cpu --host 127.0.0.1 --port 5555 \
  --size 4M --B 64K --r 0 --n 4 --warmup 1 --samples 6 --drain \
  --strategies honest --verify-all --out results/loopback.json
```

## Run the GPU experiments

Use the [experiment instructions](experiments/2026-11-24-256/README.md) for machine setup, model inputs and the performance, memory and timing launchers. Those scripts specify the settings used for the appendix, including the 128-byte codec, 256-request load and 128 links for both challenge kinds. The underlying Python programs also support other configurations and have different defaults.

For individual commands, see the [operator guide](docs/usage.md) and [throughput options](docs/benchmark-throughput.md).

## Repository map

| Location | Contents |
| --- | --- |
| `memory_accounting/setup/` | Weight registration, packing, manifests and setup CLI |
| `memory_accounting/core/`, `memory_accounting/kernels/` | Shared formats, CPU reference and CUDA kernels |
| `memory_accounting/models/`, `memory_accounting/prover/` | Model integration, inference and prover services |
| `memory_accounting/verifier/` | Verifier CLI, web dashboard, timing tests and replay |
| `memory_accounting/experiments/`, `scripts/` | Benchmark drivers, measurement and plotting tools |
| `experiments/` | Final performance, memory and timing launchers |
| `tests/` | Unit, integration and hardware-dependent tests |

## Development

Run the suite with `uv run --frozen --with pytest pytest -rs` or select a file, e.g. `tests/test_registered_setup.py`. Skips list missing hardware or optional dependencies. After changing a kernel or codec, run the matching CPU and CUDA checks on a GPU host. Ruff settings are in `pyproject.toml`.

To check the 128-byte codec against the independent reference vectors, run this from the repository root. Add `--cuda` to check GPU results as well.

```sh
POSM_VDE_BITS=89 POSM_VDE_TWEAK=element POSM_VDE_LAYOUT=packed128 .venv/bin/python -m scripts.verify128
```
