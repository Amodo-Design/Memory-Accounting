"""Build the accounted array (packed weights + filler) and encode it for each GPU."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..core.codec_profile import GROUP_ENC, FIELD_BITS, GROUP_RAW
from ..core.params import setup_params
from .filler import filler_raw89, filler_words
from .pack import pack60, pack89


def accounted_length(
    payload_words: int, accounted_bytes: int | None, hbm_bytes: int | None, scratch_fraction: float, block_bytes: int
) -> int:
    """Words in the challenged array: the declared accounted size (or HBM minus the scratch allowance),
    rounded down to whole blocks, and at least the payload rounded up to whole blocks (with neither size
    known: the payload in whole blocks)."""
    bw = block_bytes // 8
    target = (
        accounted_bytes // 8 if accounted_bytes else (int(hbm_bytes * (1 - scratch_fraction)) // 8 if hbm_bytes else 0)
    )
    words = max(target - target % bw, -(-payload_words // bw) * bw)
    return words


def build_payload(raw: np.ndarray, filler_seed: int, accounted_words: int) -> np.ndarray:
    """Packed weights followed by verifier-seeded filler, in the selected codec layout."""
    packed = pack60(raw) if FIELD_BITS == 61 else pack89(raw)
    if len(packed) > accounted_words:
        raise ValueError(f"weights need {len(packed)} words, accounted region is {accounted_words}")
    if FIELD_BITS == 89:
        if accounted_words % GROUP_ENC:
            raise ValueError("grouped 89-bit accounted memory needs whole codec groups")
        raw_groups = (accounted_words - len(packed)) // GROUP_ENC
        raw_start = len(packed) // GROUP_ENC * GROUP_RAW
        filler = filler_raw89(filler_seed, raw_start, raw_groups * GROUP_RAW).view(np.uint8)
        return np.concatenate((packed, pack89(filler)))
    out = np.empty(accounted_words, dtype=np.uint64)
    out[: len(packed)] = packed
    out[len(packed) :] = filler_words(filler_seed, len(packed), accounted_words - len(packed))
    return out


def encode_array(
    payload: np.ndarray, enc_seed: int, device: str | None = None, threads: int | None = None
) -> np.ndarray:
    """VDE-encode the payload with a per-GPU seed. On CUDA in chunks; otherwise threaded C."""
    n = len(payload)
    if device and device.startswith("cuda"):
        import torch

        from ..core.backend import DeviceBackend, to_u64_tensor

        be = DeviceBackend("cuda")
        out = np.empty(n, dtype=np.uint64)
        chunk = 1 << 27  # 1 GiB of words per launch
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            raw_t = to_u64_tensor(payload[lo:hi], device)
            out_t = torch.empty_like(raw_t)
            be.setup_at(raw_t, out_t, enc_seed, lo)  # tweak positions are global: encode the chunk at its offset
            be.sync()
            out[lo:hi] = out_t.cpu().numpy().view(np.uint64)
        return out
    from ..core.backend import CpuLib

    lib = CpuLib()
    out = np.empty(n, dtype=np.uint64)
    source = np.ascontiguousarray(payload)
    if FIELD_BITS == 89:
        lib.setup_at(source, out, enc_seed, 0)
    else:
        lib.setup(source, out, setup_params(n, 1, enc_seed, 0, 0, 0), threads=threads or (os.cpu_count() or 1))
    return out


def decode_array(theta: np.ndarray, enc_seed: int, threads: int | None = None) -> np.ndarray:
    """Inverse of encode_array on the CPU (one squaring per word)."""
    from ..core.backend import CpuLib

    n = len(theta)
    out = np.empty(n, dtype=np.uint64)
    CpuLib().decode(
        np.ascontiguousarray(theta),
        out,
        setup_params(n, 1, enc_seed, 0, 0, 0),
        threads=threads or (os.cpu_count() or 1),
    )
    return out


def image_array(theta: np.ndarray, device: str | None = None, threads: int | None = None) -> np.ndarray:
    """One-way image T = H(Theta') (32-byte groups, position-bound): the verifier's reference."""
    n = len(theta)
    out = np.empty(n, dtype=np.uint64)
    if device and device.startswith("cuda"):
        import torch

        from ..core.backend import DeviceBackend, to_u64_tensor

        be = DeviceBackend("cuda")
        chunk = 1 << 27  # 1 GiB of words per launch
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            t = to_u64_tensor(theta[lo:hi], device)
            o = torch.empty_like(t)
            be.image(t, o, lo // 4)
            be.sync()
            out[lo:hi] = o.cpu().numpy().view(np.uint64)
        return out
    from ..core.backend import CpuLib

    CpuLib().image(np.ascontiguousarray(theta), out, threads or os.cpu_count() or 1)
    return out


def decode_tensor(theta: np.ndarray, entry, enc_key: int) -> np.ndarray:
    """Decode and unpack one tensor's groups straight out of the encoded array (CPU; the CUDA kernel
    `decode_unpack` does the same on the device for stream-decoding during inference)."""
    from ..core.backend import CpuLib
    from .layout import tensor_from_raw

    raw = np.empty(GROUP_RAW * entry.groups, dtype=np.uint64)
    CpuLib().decode_unpack(np.ascontiguousarray(theta), raw, entry.payload_offset, entry.groups, enc_key)
    return tensor_from_raw(raw, entry)


def write_array(path: str | Path, arr: np.ndarray) -> None:
    np.save(path, arr.view(np.uint64), allow_pickle=False)


def load_array(path: str | Path, mmap: bool = True) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
