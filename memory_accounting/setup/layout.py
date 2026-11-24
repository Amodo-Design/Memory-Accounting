"""Group-aligned layout of tensors inside the payload.

At 61 bits a group expands 15 raw words into 16 encoded words. At 89 bits it expands 55 into 64. Every tensor is
padded to whole groups, so a tensor occupies payload words [payload_offset, payload_offset + payload_words)
exactly and can be decoded on its own: the decode-and-unpack kernel takes (q0, groups) per tensor.
This is what lets inference stream-decode one layer at a time out of Θ'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.codec_profile import GROUP_RAW, GROUP_ENC as GROUP_PAYLOAD_WORDS

GROUP_RAW_BYTES = GROUP_RAW * 8


@dataclass
class TensorEntry:
    name: str
    dtype: str
    shape: list[int]
    nbytes: int
    byte_offset: int  # in the padded raw stream
    payload_offset: int  # first payload word
    payload_words: int  # whole groups * GROUP_PAYLOAD_WORDS

    @property
    def groups(self) -> int:
        return self.payload_words // GROUP_PAYLOAD_WORDS


def plan_layout(tensors: list[tuple[str, np.ndarray, str]]) -> list[TensorEntry]:
    """tensors: [(name, array, dtype_name)] in the order they should be laid out."""
    layout, byte_off, pay_off = [], 0, 0
    for name, arr, dt in tensors:
        nbytes = int(np.ascontiguousarray(arr).nbytes)
        groups = -(-nbytes // GROUP_RAW_BYTES)
        layout.append(TensorEntry(name, dt, list(arr.shape), nbytes, byte_off, pay_off, groups * GROUP_PAYLOAD_WORDS))
        byte_off += groups * GROUP_RAW_BYTES
        pay_off += groups * GROUP_PAYLOAD_WORDS
    return layout


def padded_stream(tensors: list[tuple[str, np.ndarray, str]], layout: list[TensorEntry]) -> np.ndarray:
    """The raw byte stream with each tensor zero-padded to whole groups."""
    total = layout[-1].byte_offset + layout[-1].groups * GROUP_RAW_BYTES if layout else 0
    out = np.zeros(total, dtype=np.uint8)
    for (name, arr, _), e in zip(tensors, layout):
        b = np.ascontiguousarray(arr).view(np.uint8).ravel()
        out[e.byte_offset : e.byte_offset + e.nbytes] = b
    return out


def payload_words_total(layout: list[TensorEntry]) -> int:
    return (layout[-1].payload_offset + layout[-1].payload_words) if layout else 0


def tensor_from_raw(raw_words: np.ndarray, entry: TensorEntry) -> np.ndarray:
    """Decoded raw words of one tensor -> its bytes (dtype restored where numpy knows it)."""
    b = raw_words.view(np.uint8)[: entry.nbytes]
    if entry.dtype == "bfloat16":
        return b.view(np.uint16).reshape(entry.shape)
    if entry.dtype.startswith("float8"):  # numpy has no fp8: the bit pattern
        return b.reshape(entry.shape)
    return b.view(np.dtype(entry.dtype)).reshape(entry.shape)
