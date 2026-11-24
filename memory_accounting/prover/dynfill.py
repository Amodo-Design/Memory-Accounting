"""Filler for free dynamic memory: what a KV slot, KV block or activation row holds when nothing uses it.

Free dynamic memory must hold data a cheater cannot cheaply fake. Zeros (v0.3) are compressible, and a
plain pseudorandom stream can be regenerated on the GPU in microseconds per block, so a prover could keep
other data in "free" memory and produce the filler only for the blocks a chain visits. Free dynamic
memory therefore holds the same kind of filler as the static scheme: verifier-seeded payload words,
VDE-encoded with the selected codec, a key and the word's position in
the dynamic region as the tweak. The key is derived from the verifier's seed and the GPU index, so the
verifier can recompute any word; the per-position tweak and per-GPU key make every word and every GPU's
copy distinct.

The payload stream has its own domain too: dynamic words are filler(stream_seed(seed)), not filler(seed).
Domain-separating the key alone is not enough, because the static filler key is the prover's choice: with
the static key set to dyn_key(seed, 0), a shared payload stream made every free dynamic word equal the
static filler word at the same position (masked), so the KV pool could be dropped and read from the static
filler. A live region's pad has a third stream, so it is not the main region's filler at the same positions.

Words that land in floating-point buffers have the top exponent bit of each element cleared, so free
memory never holds NaN or Inf (a static cache whose free slots are read under a mask, as HF's attention
does, must stay finite). The verifier applies the same mask.
"""

from __future__ import annotations

from functools import lru_cache

from ..core.codec_profile import FIELD_BITS, ROW_WORDS89

import numpy as np

try:
    import torch
except ImportError:  # a torch-free verifier needs only the key checks and filler_np
    torch = None

M64 = (1 << 64) - 1
DOMAIN = 0x44594E46494C4C31  # "DYNFILL1": the key
STREAM = 0x44594E5354524D31  # "DYNSTRM1": the payload stream (main region; m1 = 1 for a live region's pad)
MASKS = {
    "bfloat16": 0xBFFFBFFFBFFFBFFF,
    "float16": 0xBFFFBFFFBFFFBFFF,
    "float32": 0xBFFFFFFFBFFFFFFF,
    "float64": 0xBFFFFFFFFFFFFFFF,
}


def dyn_key(seed: int, gpu: int = 0) -> int:
    """128-bit encoding key of a GPU's dynamic filler, derived from the verifier's seed."""
    from ..core.reference import siphash24_128

    lo, hi = siphash24_128(seed & M64, DOMAIN, gpu, 0, 0, 0)
    return (hi << 64) | lo


@lru_cache(maxsize=64)
def stream_seed(seed: int, live: bool = False) -> int:
    """The seed of the dynamic payload stream: never the static filler's seed, whatever key encodes it."""
    from ..core.reference import siphash24_128

    return siphash24_128(seed & M64, STREAM, int(live), 0, 0, 0)[0]


def key_collisions(static_keys, pad_seed: int, gpus) -> list[str]:
    """Static encoding keys (per-GPU keys, the filler key) that equal a dynamic filler key for this pad seed
    on one of these GPUs. With separate payload streams an equal key no longer makes equal words, but no
    honest setup produces one, so the verifier rejects it."""
    dyn = {dyn_key(pad_seed, int(g)): int(g) for g in gpus}
    return [
        f"static key {int(k):032x} is the dynamic filler key of GPU {dyn[int(k)]}" for k in static_keys if int(k) in dyn
    ]


def mask_for(dtype) -> int:
    """The per-word mask for a buffer of this dtype (all ones for integer buffers)."""
    name = str(dtype).replace("torch.", "")
    return MASKS.get(name, M64)


def _s64(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def filler_np(count: int, pos0: int, seed: int, key: int, mask: int = M64, live: bool = False) -> np.ndarray:
    """Dynamic filler words [pos0, pos0 + count) on the CPU (uint64): the verifier's reference. seed is the
    verifier's pad seed; live: a live region's pad (its own stream)."""
    from ..core.backend import CpuLib

    if FIELD_BITS == 89 and (pos0 % ROW_WORDS89 or count % ROW_WORDS89):
        offset = pos0 % ROW_WORDS89
        whole = -(-(offset+count)//ROW_WORDS89)*ROW_WORDS89
        return filler_np(whole, pos0-offset, seed, key, mask, live)[offset:offset+count].copy()
    lib = CpuLib()
    payload = np.empty(count, dtype=np.uint64)
    lib.filler(payload, stream_seed(seed, live), pos0)
    out = np.empty(count, dtype=np.uint64)
    lib.setup_at(payload, out, key, pos0)
    if mask != M64:
        out &= np.uint64(mask)
    return out


def fill_(
    words: torch.Tensor, pos0: int, seed: int, key: int, mask: int = M64, chunk: int = 1 << 25, live: bool = False
) -> None:
    """Write dynamic filler words [pos0, pos0 + n) into an int64 tensor in place (GPU kernels or C)."""
    n = words.numel()
    if n == 0:
        return
    if FIELD_BITS == 89 and (pos0 % ROW_WORDS89 or n % ROW_WORDS89):
        offset = pos0 % ROW_WORDS89
        tmp = torch.empty(-(-(offset+n)//ROW_WORDS89)*ROW_WORDS89, dtype=words.dtype, device=words.device)
        fill_(tmp, pos0-offset, seed, key, mask, chunk, live)
        words.copy_(tmp[offset:offset+n])
        return
    if words.device.type == "cuda":
        be = _backend(words.device)
        tmp = torch.empty(min(chunk, n), dtype=torch.int64, device=words.device)
        ss = _s64(stream_seed(seed, live))
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            be.filler(tmp[: hi - lo], ss, pos0 + lo)
            be.setup_at(tmp[: hi - lo], words[lo:hi], key, pos0 + lo)
        if mask != M64:
            words.bitwise_and_(_s64(mask))
    else:
        words.copy_(torch.from_numpy(filler_np(n, pos0, seed, key, mask, live).view(np.int64)))


_BACKENDS: dict = {}


def _backend(device):
    """One DeviceBackend per device (building one re-checks the compiled extension)."""
    be = _BACKENDS.get(str(device))
    if be is None:
        from ..core.backend import DeviceBackend

        be = _BACKENDS[str(device)] = DeviceBackend(str(device))
    return be


_TEMPLATES: dict = {}
TEMPLATE_BYTES = 1 << 30  # cap on cached templates: a region with many segments (the integer model's
# 122) keeps all of them, so a reset is copies, not filler recomputed


def template(n: int, pos0: int, seed: int, key: int, mask: int, device) -> torch.Tensor:
    """Cached filler for a fixed range (restoring a buffer is then a copy)."""
    k = (str(device), n, pos0, seed, key, mask)
    t = _TEMPLATES.get(k)
    if t is None:
        t = torch.empty(n, dtype=torch.int64, device=device)
        fill_(t, pos0, seed, key, mask)
        if sum(v.numel() for v in _TEMPLATES.values()) * 8 + n * 8 > TEMPLATE_BYTES:
            _TEMPLATES.clear()
        _TEMPLATES[k] = t
    return t
