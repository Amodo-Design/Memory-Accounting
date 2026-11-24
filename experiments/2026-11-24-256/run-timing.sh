#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
trap 'run_rc=$?; echo "$run_rc" > results/timing-256-128x128-23-rounds.rc' EXIT
.venv/bin/python -u scripts/qwen_timing.py \
    --verifier-ssh "$POSM_VERIFIER_SSH" \
    --verifier-repo "$POSM_VERIFIER_REPO" \
    --verifier-weight-reference "$POSM_VERIFIER_WEIGHT_REFERENCE" \
    --prover-address "$POSM_PROVER_ADDRESS" --verifier-address "$POSM_VERIFIER_ADDRESS" \
    --warmup 200 --calibrate 3000 --rounds 23 --slot-seconds 40 \
    --timing-concurrency 256 --dynamic-links 128 --gpu-memory-utilization 0.75 \
    --out-dir "$PWD/results/timing-256-128x128-23-rounds"
