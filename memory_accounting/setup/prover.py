"""Prover side of setup: from weights and hardware facts to encoded arrays, commitments and a manifest."""

from __future__ import annotations

import secrets
from pathlib import Path

import numpy as np

from ..core.params import Keys
from .commit import block_leaves, merkle_root
from .encode import accounted_length, build_payload, encode_array, image_array, write_array
from .manifest import GpuEntry, Manifest
from .pack import packed_words_for
from .weights import flatten, load_tensors, raw_hash


def detect_hbm_bytes(gpu_index: int = 0) -> int:
    import torch

    return int(torch.cuda.get_device_properties(gpu_index).total_memory)


def run_setup(
    weights_path: str | Path,
    out_dir: str | Path,
    gpu_indices: list[int],
    filler_seed: int,
    block_bytes: int = 1 << 20,
    hbm_bytes: int | None = None,
    accounted_bytes: int | None = None,
    scratch_fraction: float = 0.10,
    device: str | None = None,
    threads: int | None = None,
    enc_seeds: list[int] | None = None,
    expected_raw_hash: str | None = None,
    r: int = 2048,
    k: int = 1024,
    images: bool = True,
) -> Manifest:
    if len(set(gpu_indices)) != len(gpu_indices):
        raise ValueError(f"GPU indices repeat: {gpu_indices}")
    if not gpu_indices:
        raise ValueError("no GPU indices")
    if enc_seeds is not None and (
        len(enc_seeds) != len(gpu_indices) or len(set(enc_seeds)) != len(enc_seeds) or not all(enc_seeds)
    ):
        raise ValueError("one distinct, nonzero encoding key per GPU: one copy must not answer for two GPUs")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = load_tensors(weights_path)
    raw, layout = flatten(tensors)
    if expected_raw_hash is not None and raw_hash(raw, layout) != expected_raw_hash:
        raise ValueError("weights differ from the pre-seed registration")
    payload_words = packed_words_for(raw.nbytes)
    if hbm_bytes is None and device and device.startswith("cuda"):
        hbm_bytes = detect_hbm_bytes(gpu_indices[0])
    accounted_words = accounted_length(payload_words, accounted_bytes, hbm_bytes, scratch_fraction, block_bytes)
    if hbm_bytes is None:  # no capacity known (CPU setup, none given): nothing declared past the array
        hbm_bytes = accounted_words * 8
    if accounted_words * 8 > hbm_bytes:  # (this used to be printed as a negative scratch)
        raise ValueError(
            f"the challenged array ({accounted_words * 8} bytes: the weights in whole blocks, or --accounted-bytes) "
            f"does not fit in the declared HBM ({hbm_bytes} bytes)"
        )
    payload = build_payload(raw, filler_seed, accounted_words)
    commit_key = secrets.token_bytes(16)
    enc_seeds = enc_seeds or [secrets.randbits(128) for _ in gpu_indices]
    hash_key = Keys.from_seed(filler_seed).as_list()  # from the verifier's seed, as the verifier derives them
    manifest = Manifest(
        model_source=Path(weights_path).name,
        raw_bytes=int(raw.nbytes),
        raw_hash=raw_hash(raw, layout),
        layout=layout,
        block_bytes=block_bytes,
        payload_words=payload_words,
        filler_seed=filler_seed,
        accounted_words=accounted_words,
        hbm_bytes=hbm_bytes,
        commit_key=commit_key.hex(),
        hash_key=hash_key,
        r=r,
        k=k,
    )
    for gi, seed in zip(gpu_indices, enc_seeds):
        theta = encode_array(payload, seed, device=device, threads=threads)
        leaves = block_leaves(theta, block_bytes, commit_key)
        tf, cf = out_dir / f"theta_gpu{gi}.npy", out_dir / f"commit_gpu{gi}.npy"
        write_array(tf, theta)
        np.save(cf, leaves, allow_pickle=False)
        entry = GpuEntry(gi, seed, merkle_root(leaves).hex(), str(cf), str(tf))
        manifest.gpus.append(entry)
        if images:
            im = out_dir / f"image_gpu{gi}.npy"
            write_array(im, image_array(theta, device=device, threads=threads))
            entry.image_file = str(im)
    manifest.save(out_dir / "manifest.json")
    return manifest
