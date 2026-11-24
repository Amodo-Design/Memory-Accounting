#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results
trap 'run_rc=$?; echo "$run_rc" > results/three-run-256-128x128.rc' EXIT
for experiment in performance memory timing; do
    echo "START $experiment $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a results/three-run-256-128x128.progress
    bash "experiments/2026-11-24-256/run-$experiment.sh" \
        > "results/three-run-256-128x128-$experiment.log" 2>&1
    echo "DONE $experiment $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a results/three-run-256-128x128.progress
    if [[ "$experiment" != timing ]]; then
        image="results/$experiment-256-128x128/bundle/image_gpu0.npy"
        test -f "$image" || { echo "Missing expected image: $image" >&2; exit 1; }
        rm -- "$image"
        echo "REMOVED Dell image for $experiment; verifier copy and logs remain" | tee -a results/three-run-256-128x128.progress
    fi
done
