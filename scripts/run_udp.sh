#!/usr/bin/env bash
# End-to-end UDP check at one configuration: B n r [samples]. Starts a prover process on
# localhost, drives it from a verifier process, writes results/udp_B{B}_n{n}_r{r}.json.
set -e
B=${1:-1M}; N=${2:-8}; R=${3:-128}; S=${4:-200}
PY=${PY:-uv run python}
mkdir -p results
$PY -m memory_accounting.prover.udp --port 5555 > results/udp_prover.log 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null" EXIT
for i in $(seq 1 120); do grep -q "prover ready" results/udp_prover.log && break; kill -0 $PID 2>/dev/null || { echo "prover exited:"; cat results/udp_prover.log; exit 1; }; sleep 1; done
grep -q "prover ready" results/udp_prover.log || { echo "prover did not start within 120 s"; exit 1; }
$PY -m memory_accounting.verifier.net verifier --host 127.0.0.1 --port 5555 --B $B --n $N --r $R --samples $S --out results/udp_B${B}_n${N}_r${R}.json
