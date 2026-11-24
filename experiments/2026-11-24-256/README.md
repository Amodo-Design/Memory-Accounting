# Qwen2.5-32B appendix experiments

These launchers run the performance, memory and timing experiments. Use the same checkout on both machines. The original configuration used one H200 NVL on the prover, a CPU verifier on a direct network link, and a matching GPU for offline replay. Install the environments as described in the [repository README](../../README.md#installation). Run the commands below from the repository root.

## Prepare model inputs

The launchers run offline. Cache `Qwen/Qwen2.5-32B-Instruct` at revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` on the prover. For replay, `POSM_MODEL_SNAPSHOT` points to that revision’s local snapshot directory.

Prepare the trusted weight reference on a trusted model-loading host using the same GPU environment and checkpoint. Choose a new file with space for about 65 GiB. Copy it to a private location on the verifier before starting any experiment.

```sh
export POSM_VDE_BITS=89 POSM_VDE_TWEAK=element POSM_VDE_LAYOUT=packed128
PYTHONPATH=. .venv/bin/python scripts/bench_throughput.py export-weight-reference \
  --model Qwen/Qwen2.5-32B-Instruct --model-revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
  --max-num-seqs 256 --gpu-memory-utilization 0.75 --out reference.bin
```

## Configure the machines

From the prover, check that non-interactive SSH and rsync reach the verifier. Allow disk space on both machines for the model-sized image, and on the verifier for retained images and evidence. Export these variables in the prover shell.

| Variable | Value supplied by the operator |
| --- | --- |
| `POSM_RUNTIME_ENV` | Absolute path to a shell file that activates the prover environment and exports its CUDA settings |
| `POSM_TRANSPORT_BIN` | Directory on the prover’s `PATH` containing the SSH/rsync programs or local wrappers, e.g. `/usr/bin` |
| `POSM_VERIFIER_SSH` | SSH destination for the verifier as reached from the prover |
| `POSM_VERIFIER_REPO` | Absolute path to the matching checkout with `.venv` on the verifier |
| `POSM_VERIFIER_WEIGHT_REFERENCE` | Absolute path on the verifier to the trusted `reference.bin` |
| `POSM_PROVER_ADDRESS`, `POSM_VERIFIER_ADDRESS` | Respective network addresses on the challenge link |
| `POSM_MODEL_SNAPSHOT` | Absolute path to the pinned model snapshot on the replay host |

## Run and inspect outputs

```sh
bash experiments/2026-11-24-256/run-all.sh
bash experiments/2026-11-24-256/run-timing-replay.sh
```

The first command runs performance, memory and timing in order. Each stage writes its own log, results directory and exit status under `results/`. After the performance and memory stages, `run-all.sh` deletes that stage’s local `bundle/image_gpu0.npy` to make room for the next export. Copies remain on the verifier. The second command checks evidence coverage and replays the saved timing records on the GPU. To run stages individually, invoke `run-performance.sh`, `run-memory.sh` or `run-timing.sh` from this directory with Bash. Each needs a fresh output directory.

| Run | Workload | Measurement |
| --- | --- | --- |
| Performance | 256 requests, 256 prompt and 256 generated tokens | Stock vLLM and four cumulative stages, one warm-up and three timed calls per stage |
| Memory | Idle, then batches of 16 and 256 requests, 256 prompt and 48 generated tokens per active request | Minimum 40, 40 and 70 seconds respectively, finishing each active batch |
| Timing | Batches of 256 requests, 256 prompt and 48 generated tokens, 1.5 s between batches | 23 rounds of six 40-second slots, 92 minutes before setup and calibration time |

All three use the packed 128-byte layout and 128 links for both challenge kinds. Calibration runs under the 256-request load. Full performance uses 200 warm-ups and 1,000 calibration challenges per kind. Memory and timing use 200 and 3,000 respectively. Check each run’s `.rc` file, logs and replay output before using its results.
