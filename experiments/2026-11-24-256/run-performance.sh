#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
trap 'run_rc=$?; echo "$run_rc" > results/performance-256-128x128.rc' EXIT
.venv/bin/python -u scripts/bench_full.py --breakdown \
    --verifier-ssh "$POSM_VERIFIER_SSH" \
    --verifier-repo "$POSM_VERIFIER_REPO" \
    --verifier-weight-reference "$POSM_VERIFIER_WEIGHT_REFERENCE" \
    --prover-address "$POSM_PROVER_ADDRESS" --verifier-address "$POSM_VERIFIER_ADDRESS" \
    --prover-args '--batches 256 --max-num-seqs 256 --scratch-reserve-gb 4.25 --calls 3' \
    --verifier-args '--dyn-n 128 --warmup 200 --calibrate 1000 --check-fraction .02 --dynamic-check-fraction .08' \
    --out-dir "$PWD/results/performance-256-128x128"
