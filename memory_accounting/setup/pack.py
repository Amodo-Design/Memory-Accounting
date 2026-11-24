"""Bit packing between raw bytes and codec payloads.

The 61-bit profile packs 15 raw words into 16 60-bit payload words. The grouped
89-bit profile packs 55 raw words into 64 payload words: five 11-byte values
per 64-byte row, each placed in a 12-byte slot.
"""

from __future__ import annotations

import numpy as np

from ..core.codec_profile import FIELD_BITS, GROUP_ENC, GROUP_RAW, PACKED128

MASK60 = np.uint64((1 << 60) - 1)
RAW_PER_GROUP, PACKED_PER_GROUP = 15, 16


def packed_words_for(raw_bytes: int) -> int:
    """Payload words needed for raw bytes, in whole codec groups."""
    return -(-raw_bytes // (GROUP_RAW * 8)) * GROUP_ENC


def pack89(raw: np.ndarray) -> np.ndarray:
    """55 raw words to 64 payload words: eight rows of five 11-byte values and padding."""
    if FIELD_BITS != 89:
        raise ValueError("pack89 requires the grouped 89-bit profile")
    source = np.ascontiguousarray(raw, dtype=np.uint8).ravel()
    groups = -(-len(source) // (GROUP_RAW * 8))
    padded = np.zeros(groups * GROUP_RAW * 8, dtype=np.uint8)
    padded[:len(source)] = source
    if PACKED128:
        payload = np.zeros((8*groups,128), dtype=np.uint8)
        payload[:, :121] = padded.reshape(-1,121)
        return payload.reshape(-1).view(np.uint64)
    rows = padded.reshape(groups * 8, 55)
    slots = np.zeros((groups * 8, 64), dtype=np.uint8)
    for lane in range(5):
        slots[:, 12 * lane:12 * lane + 11] = rows[:, 11 * lane:11 * lane + 11]
    return slots.reshape(-1).view(np.uint64)


def pack60(raw: np.ndarray) -> np.ndarray:
    """uint8 byte stream -> uint64 array of 60-bit payload words (zero padded to a whole group)."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8).ravel()
    groups = -(-len(raw) // (RAW_PER_GROUP * 8))
    buf = np.zeros(groups * RAW_PER_GROUP * 8, dtype=np.uint8)
    buf[: len(raw)] = raw
    words = buf.view("<u8").reshape(groups, RAW_PER_GROUP)
    out = np.zeros((groups, PACKED_PER_GROUP), dtype=np.uint64)
    for k in range(PACKED_PER_GROUP):
        start = 60 * k
        a, off = divmod(start, 64)
        v = words[:, a] >> np.uint64(off)
        if off > 4 and a + 1 < RAW_PER_GROUP:  # the 60-bit window crosses into the next raw word
            v = v | (words[:, a + 1] << np.uint64(64 - off))
        out[:, k] = v & MASK60
    return out.ravel()


def unpack60(packed: np.ndarray, raw_bytes: int) -> np.ndarray:
    """Inverse of pack60: 60-bit payload words -> the first raw_bytes bytes."""
    packed = np.ascontiguousarray(packed, dtype=np.uint64).ravel()
    groups = len(packed) // PACKED_PER_GROUP
    rows = packed.reshape(groups, PACKED_PER_GROUP)
    words = np.zeros((groups, RAW_PER_GROUP), dtype=np.uint64)
    for a in range(RAW_PER_GROUP):
        # raw word a covers payload bits [64a, 64a+64): from packed word k = 64a // 60 at offset (64a mod 60)
        start = 64 * a
        k, off = divmod(start, 60)
        v = rows[:, k] >> np.uint64(off)
        if k + 1 < PACKED_PER_GROUP:
            v = v | (rows[:, k + 1] << np.uint64(60 - off))
        if k + 2 < PACKED_PER_GROUP and 60 - off + 60 < 64:  # a 64-bit window can span three 60-bit words
            v = v | (rows[:, k + 2] << np.uint64(120 - off))
        words[:, a] = v
    return words.ravel().view(np.uint8)[:raw_bytes].copy()
