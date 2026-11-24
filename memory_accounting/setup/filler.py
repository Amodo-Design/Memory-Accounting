"""Verifier-seeded filler for the accounted region beyond the weights.

Every word of HBM the scheme accounts for must be challengeable, so the space after the packed
weights (up to the declared accounted size) is filled with pseudorandom values derived
from a seed the VERIFIER chooses at setup, then encoded like the weights. The prover cannot
prepare it in advance, and the verifier can regenerate it without storing it.
"""

from __future__ import annotations

import numpy as np

GOLD = np.uint64(0x9E3779B97F4A7C15)
MASK60 = np.uint64((1 << 60) - 1)
M64 = (1 << 64) - 1
STATIC_FILLER_DOMAIN = 0x535446494C4C4B32  # "STFILLK2": the static filler's key (dynamic filler: "DYNFILL1")


def static_filler_key(filler_seed: int, gpu: int) -> int:
    """The 128-bit key the static filler of GPU `gpu` is encoded under in the in-process provers (stream-decode
    bundles): a PRF of the verifier's filler seed and the GPU index (SipHash-2-4-128, its own domain, like the
    dynamic filler's `memory_accounting.prover.dynfill.dyn_key`). Both sides derive it; the verifier never takes it from the
    prover's manifest, so every GPU's filler is distinct (the GPU index is the verifier's) and the prover has no
    say in it. It is independent of the weights' key, which with Θ' gives the weights and stays with the prover.
    (It used to be one-way from the weights' key and carried by the bundle: the prover's choice, so two GPUs
    could share one filler, and the verifier could not tell.) The setup phase's arrays encode the filler
    under the per-GPU key; there the verifier knows the weights and recomputes everything."""
    from ..core.reference import siphash24_128

    if not 0 <= int(gpu) < 1 << 32:
        raise ValueError(f"GPU index {gpu!r} out of range")
    lo, hi = siphash24_128(int(filler_seed) & M64, STATIC_FILLER_DOMAIN, int(gpu), 0, 0, 0)
    return (hi << 64) | lo


def splitmix64(z: np.ndarray) -> np.ndarray:
    z = z + GOLD
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def filler_words(seed: int, start: int, count: int) -> np.ndarray:
    """Payload words [start, start+count) of the filler stream for `seed` (uint64, < 2^60)."""
    idx = np.arange(start, start + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return splitmix64(np.uint64(seed) ^ (idx * GOLD)) & MASK60


def filler_raw89(seed: int, start: int, count: int) -> np.ndarray:
    """Full-width raw words for the grouped 89-bit filler's 55-word rows."""
    idx = np.arange(start, start + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return splitmix64(np.uint64(seed) ^ np.uint64(0x89535446494C4C31) ^ (idx * GOLD))
