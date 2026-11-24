#!/usr/bin/env bash
# Honest timing envelope vs inference-like GPU load.  Runs the same grid at several load duty
# cycles, for a coarse load (8192^2 matmul + 1 GiB streaming: long kernels, coarse preemption) and
# a fine load (2048^2 + 64 MiB: short kernels), plus one idle baseline.  Then draws the load curves.
# Long-running by design (hours at SAMPLES=200 on an H200).  Override via env: FRACS, SAMPLES, PY.
#
#   scripts/run_load_sweeps.sh            run the sweeps not yet in results/load/, then the figures
#   scripts/run_load_sweeps.sh --force    run every sweep again, overwriting results/load/*.json
#
# The grid (r = 0, 32, 128) is the one results/load/ holds and memory_accounting.experiments.loadcurve's key configurations
# (64 KB n16 r32, 1 MB n8 r128, 1 MB n16 r0, 256 KB n32 r128) are drawn from.  The output is kept in
# results/load/load_sweeps.log (appended to).
set -eo pipefail
FORCE=0
case "${1:-}" in
  --force) FORCE=1 ;;
  "") ;;
  *) echo "usage: scripts/run_load_sweeps.sh [--force]" >&2; exit 1 ;;
esac
PY=${PY:-uv run python}
FRACS=${FRACS:-"0.1 0.25 0.5 0.75 1.0"}
SAMPLES=${SAMPLES:-200}
OUT=results/load
mkdir -p $OUT
exec > >(tee -a $OUT/load_sweeps.log) 2>&1
echo "#### $(date -u "+%Y-%m-%d %H:%M:%S") UTC: scripts/run_load_sweeps.sh $*"
GRID="--size 1G --encoding vde --blocks 64K,256K,1M --links 1,4,8,16,32,64 --r 0,32,128 --samples $SAMPLES"
CHEATS=other_gpu,encode_otf,dram_copy,dram_mapped,cpu
EXTRA=""
if $PY -c "import torch, sys; sys.exit(0 if torch.cuda.device_count() > 1 else 1)"; then EXTRA="--canon-device cuda:1"; fi

# needed FILE: true when FILE is to be (re)measured
needed() { [ "$FORCE" = 1 ] || [ ! -f "$1" ]; }

echo "== idle baseline"
if needed $OUT/idle.json; then $PY -m memory_accounting.experiments.bench $GRID --provers $CHEATS $EXTRA --out $OUT/idle.json; else echo "(kept $OUT/idle.json; --force to measure again)"; fi

for GRAN in coarse fine; do
    if [ $GRAN = coarse ]; then LOAD="--load-dim 8192 --load-buffer-mb 1024"; else LOAD="--load-dim 2048 --load-buffer-mb 64"; fi
    for f in $FRACS; do
        echo "== $GRAN load, fraction $f"
        if needed $OUT/${GRAN}_f${f}.json; then
            $PY -m memory_accounting.experiments.bench $GRID --provers $CHEATS $EXTRA --background-load --load-fraction $f $LOAD --out $OUT/${GRAN}_f${f}.json
        else
            echo "(kept $OUT/${GRAN}_f${f}.json; --force to measure again)"
        fi
    done
done

echo "== figures"
mkdir -p figures/loadcurve
$PY -m memory_accounting.experiments.loadcurve $OUT/idle.json $OUT/coarse_f*.json $OUT/fine_f*.json --outdir figures/loadcurve --jitter-us 45
for f in $OUT/coarse_f*.json $OUT/fine_f*.json; do
    name=$(basename $f .json)
    $PY -m memory_accounting.experiments.plots $f --outdir figures/loadcurve/$name --jitter-us 45 > figures/loadcurve/$name.txt
done
echo "done: figures/loadcurve"
