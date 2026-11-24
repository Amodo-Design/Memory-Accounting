#!/usr/bin/env bash
# Full honest-vs-cheat sweep on a CUDA host. Run from the repo root.
# The cheats include the zero-copy PCIe one (dram_mapped), which reads pinned host memory from a CUDA kernel:
# this script needs a CUDA host. With two GPUs, other_gpu's copy lives on the second one (--canon-device cuda:1).
set -e
PY=${PY:-uv run python}
CHEATS=other_gpu,encode_otf,dram_copy,dram_mapped,cpu
MVP_CHEATS=other_gpu,dram_copy,dram_mapped,cpu
EXTRA=""
# with two GPUs the 'other GPU' cheat really lives on the second GPU and is read over NVLink
if $PY -c "import torch, sys; sys.exit(0 if torch.cuda.device_count() > 1 else 1)"; then EXTRA="--canon-device cuda:1"; fi
$PY -m memory_accounting.experiments.bench --size 1G --encoding vde --blocks 16K,64K,256K,1M --links 1,2,4,8,16,32,64 --r 0,256,2048 \
    --provers $CHEATS --samples 100 --out results/sweep_vde.json $EXTRA
# MVP-PoSM mode: weights trusted not to be compressible, no encoding; only the permutation distinguishes GPUs.
$PY -m memory_accounting.experiments.bench --size 1G --encoding none --blocks 64K,256K --links 1,4,16,64 --r 0,2048 \
    --provers $MVP_CHEATS --samples 100 --out results/sweep_mvp.json $EXTRA
# Same mechanisms with a background inference-like load on the GPU (reduced grid).
$PY -m memory_accounting.experiments.bench --size 1G --encoding vde --blocks 64K,256K,1M --links 1,4,16,64 --r 0,256,2048 \
    --provers $CHEATS --samples 100 --background-load --out results/sweep_vde_load.json $EXTRA
