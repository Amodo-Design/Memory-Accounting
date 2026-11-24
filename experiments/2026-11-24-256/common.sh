#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${POSM_RUNTIME_ENV:?point this at the prover Python environment script}"
: "${POSM_TRANSPORT_BIN:?point this at the prover transport binaries}"
: "${POSM_VERIFIER_SSH:?SSH alias for the verifier from the prover}"
: "${POSM_VERIFIER_REPO:?path to the matching verifier checkout}"
: "${POSM_VERIFIER_WEIGHT_REFERENCE:?trusted model reference on the verifier}"
: "${POSM_PROVER_ADDRESS:?prover address on the direct cable}"
: "${POSM_VERIFIER_ADDRESS:?verifier address on the direct cable}"

source "$POSM_RUNTIME_ENV"
export PATH="$POSM_TRANSPORT_BIN:$PATH"
export PYTHONPATH="$PWD:$PWD/scripts"
export POSM_VDE_BITS=89 POSM_VDE_TWEAK=element POSM_VDE_LAYOUT=packed128 HF_HUB_OFFLINE=1
unset POSM_TIMING_TRACE_OUT
mkdir -p results
