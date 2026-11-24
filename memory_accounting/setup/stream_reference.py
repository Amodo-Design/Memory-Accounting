"""Verifier-owned plaintext reference for the stream-decoding layout."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from ..core.codec_profile import GROUP_ENC, GROUP_RAW, PROFILE

MAGIC = b"POSMRAW1"
HEADER_LIMIT = 1 << 20


def export_reference(model, path):
    """Export the loaded model on a trusted host, before issuing a seed."""
    from ..prover.stream_decode import memory_bytes, plan_units

    units, params, words = plan_units(model)
    params = dict(params)
    slots = [slot for unit in units.values() for slot in unit.slots]
    layout = [{
        "name": slot.name, "dtype": str(slot.dtype), "shape": slot.shape, "stride": slot.stride,
        "bytes": slot.nbytes, "groups": slot.groups,
    } for slot in slots]
    header = json.dumps({"codec": PROFILE, "slots": layout}, sort_keys=True).encode()
    if len(header) > HEADER_LIMIT:
        raise ValueError("stream weight layout exceeds header limit")
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(MAGIC + struct.pack("<Q", len(header)) + header + bytes(-len(header) % 8))
            for slot in slots:
                raw = memory_bytes(params[slot.name], slot.name).cpu().numpy()
                f.write(memoryview(raw))
                f.write(bytes(slot.groups * GROUP_RAW * 8 - slot.nbytes))
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {"path": str(path), "payload_words": words, "tensor_count": len(slots)}


def read_reference(f):
    prefix = f.read(16)
    if len(prefix) != 16 or prefix[:8] != MAGIC:
        raise ValueError("invalid stream weight reference magic")
    length = struct.unpack("<Q", prefix[8:])[0]
    if not 0 < length <= HEADER_LIMIT:
        raise ValueError("invalid stream weight reference header length")
    header = json.loads(f.read(length))
    f.read(-length % 8)
    if not isinstance(header, dict) or header.get("codec") != PROFILE:
        raise ValueError("stream weight reference codec differs from this verifier")
    slots = header.get("slots")
    if (not isinstance(slots, list) or not slots
            or any(not isinstance(s, dict) or type(s.get("groups")) is not int or s["groups"] < 0
                   or type(s.get("bytes")) is not int or not 0 <= s["bytes"] <= s["groups"] * GROUP_RAW * 8
                   for s in slots)):
        raise ValueError("invalid stream weight reference slots")
    raw_bytes = sum(slot["groups"] for slot in slots) * GROUP_RAW * 8
    if not raw_bytes or os.fstat(f.fileno()).st_size != f.tell() + raw_bytes:
        raise ValueError("stream weight reference size differs from its layout")
    return header, raw_bytes


def verify_stream_weights(manifest, gpu, img, reference, registration, receipt_file, threads):
    """Recompute the image and encoded commitment using the verifier's registered bytes and seed."""
    from ..core.backend import CpuLib
    from .commit import leaves_of, merkle_root
    from .registration import encoding_key, load_receipt

    receipt = load_receipt(receipt_file, registration)
    if receipt.get("weight_layout") != "stream" or receipt["gpu_indices"] != [gpu]:
        raise ValueError("stream registration must identify exactly this GPU")
    if manifest.filler_seed != int(receipt["filler_seed"], 16):
        raise ValueError("bundle did not use the registered setup seed")
    key = encoding_key(receipt, gpu)
    entry = manifest.gpu(gpu)
    commit_key = bytes.fromhex(manifest.commit_key)
    block_bytes = manifest.block_bytes
    if block_bytes % (8 * GROUP_ENC):
        raise ValueError("registered stream challenge blocks must be whole codec groups")
    groups_per_chunk = max(1, (8 << 20) // block_bytes) * (block_bytes // (8 * GROUP_ENC))
    lib = CpuLib()
    with Path(reference).open("rb") as f:
        _, raw_bytes = read_reference(f)
        offset = f.tell()
        f.seek(0)
        if hashlib.file_digest(f, "sha256").hexdigest() != receipt["raw_hash"]:
            raise ValueError("stream reference changed after registration")
        groups = raw_bytes // (8 * GROUP_RAW)
        if img.shape != (groups * GROUP_ENC,):
            raise ValueError("bundle image has a different size from the registered weights")
        raw = np.memmap(f, dtype=np.uint64, mode="r", offset=offset, shape=(groups * GROUP_RAW,))

        def recompute(start):
            count = min(groups - start, groups_per_chunk)
            q = start * GROUP_ENC
            payload = np.empty(count * GROUP_ENC, dtype=np.uint64)
            encoded = np.empty_like(payload)
            image = np.empty_like(payload)
            lib.pack(raw[start * GROUP_RAW:(start + count) * GROUP_RAW], payload, count)
            lib.setup_at(payload, encoded, key, q)
            lib.image_at(encoded, image, q // 4)
            if not np.array_equal(img[q:q + len(image)], image):
                raise ValueError(f"bundle image differs from the registered encoding at word range starting {q}")
            return leaves_of(encoded, block_bytes, commit_key)

        with ThreadPoolExecutor(max_workers=threads) as pool:
            leaves = list(pool.map(recompute, range(0, groups, groups_per_chunk)))
    if merkle_root(np.concatenate(leaves)).hex() != entry.commit_root:
        raise ValueError("bundle encoded commitment differs from the registered encoding")
