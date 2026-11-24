#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
trap 'run_rc=$?; echo "$run_rc" > results/memory-256-128x128.rc' EXIT
.venv/bin/python -u scripts/appendix_memory.py \
    --verifier-ssh "$POSM_VERIFIER_SSH" \
    --verifier-repo "$POSM_VERIFIER_REPO" \
    --verifier-weight-reference "$POSM_VERIFIER_WEIGHT_REFERENCE" \
    --prover-address "$POSM_PROVER_ADDRESS" --verifier-address "$POSM_VERIFIER_ADDRESS" \
    --load-concurrency 256 --dynamic-links 128 \
    --warmup 200 --calibrate 3000 \
    --idle-seconds 40 --light-seconds 40 --selected-seconds 70 \
    --out-dir "$PWD/results/memory-256-128x128"
