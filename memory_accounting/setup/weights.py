"""Load model weights into one flat byte stream with a layout, and hash them.

Supports .safetensors (preferred: read directly, no torch and no safetensors package, memory-mapped),
.npy/.npz, and .pt/.pth/.bin (a torch state dict, as transformers writes `pytorch_model.bin`: needs torch). The layout records (name, dtype, shape,
byte offset) so the encoded array can be decoded back into tensors for inference. Only the bytes matter:
a dtype numpy lacks (bfloat16, the fp8 types) is kept as its bit pattern in an unsigned array of its width.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .layout import TensorEntry, padded_stream, plan_layout

# safetensors dtype -> (numpy dtype of the bit pattern, the layout's dtype name). The names are numpy's (as
# torch's .numpy() gave them before this reader), so the model hashes of existing manifests are unchanged.
SAFETENSORS_DTYPES = {
    "BOOL": ("bool", "bool"),
    "U8": ("<u1", "uint8"),
    "I8": ("<i1", "int8"),
    "U16": ("<u2", "uint16"),
    "I16": ("<i2", "int16"),
    "U32": ("<u4", "uint32"),
    "I32": ("<i4", "int32"),
    "U64": ("<u8", "uint64"),
    "I64": ("<i8", "int64"),
    "F16": ("<f2", "float16"),
    "F32": ("<f4", "float32"),
    "F64": ("<f8", "float64"),
    "BF16": ("<u2", "bfloat16"),
    "F8_E4M3": ("<u1", "float8_e4m3fn"),
    "F8_E5M2": ("<u1", "float8_e5m2"),
}


def _from_safetensors(path: Path) -> list[tuple[str, np.ndarray, str]]:
    """The safetensors format read directly: an 8-byte little-endian header length, a JSON header of
    {name: {dtype, shape, data_offsets}}, then the data. Arrays are views of a memory map (no copy)."""
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    if len(mm) < 8:
        raise ValueError(f"{path}: not a safetensors file")
    n = int.from_bytes(bytes(mm[:8]), "little")
    if 8 + n > len(mm):
        raise ValueError(f"{path}: header length {n} past the end of the file")
    header = json.loads(bytes(mm[8 : 8 + n]))
    base = 8 + n
    out = []
    for name in sorted(k for k in header if k != "__metadata__"):
        e = header[name]
        if e["dtype"] not in SAFETENSORS_DTYPES:
            raise ValueError(f"{path}: {name} has dtype {e['dtype']}, not supported")
        npdt, dname = SAFETENSORS_DTYPES[e["dtype"]]
        b, end = (int(x) for x in e["data_offsets"])
        shape = tuple(int(x) for x in e["shape"])
        dt = np.dtype(npdt)
        if end - b != int(np.prod(shape, dtype=np.int64)) * dt.itemsize or base + end > len(mm):
            raise ValueError(
                f"{path}: {name} spans {end - b} bytes, its shape {list(shape)} of {e['dtype']} needs {int(np.prod(shape)) * dt.itemsize}"
            )
        out.append((name, mm[base + b : base + end].view(dt).reshape(shape), dname))
    return out


def _from_torch(path: Path) -> list[tuple[str, np.ndarray, str]]:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        raise ValueError(
            f"{path}: a torch state dict (.pt/.pth/.bin) needs torch to read; convert it to .safetensors for a verifier without torch"
        )
    try:
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as e:  # not a torch pickle (a raw blob is no longer accepted as weights)
        raise ValueError(f"{path}: not a torch state dict ({type(e).__name__}: {e})") from e
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: expected a state dict")
    out = []
    for name in sorted(sd):
        t = sd[name].detach().contiguous()
        if t.dtype == torch.bfloat16:  # numpy has no bf16: keep the bit pattern
            out.append((name, t.view(torch.int16).numpy().view(np.uint16), "bfloat16"))
        else:
            a = t.numpy()
            out.append((name, a, str(a.dtype)))
    return out


def load_tensors(path: str | Path) -> list[tuple[str, np.ndarray, str]]:
    """Returns [(name, array, dtype_name)] sorted by name."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".safetensors":
        return _from_safetensors(path)
    if suf == ".npy":
        a = np.load(path)
        return [(path.stem, np.ascontiguousarray(a), str(a.dtype))]
    if suf == ".npz":
        z = np.load(path)
        return [(n, np.ascontiguousarray(z[n]), str(z[n].dtype)) for n in sorted(z.files)]
    # .bin: transformers' pytorch_model.bin, a torch pickle (was read as one raw blob)
    if suf in (
        ".pt",
        ".pth",
        ".bin",
    ):
        return _from_torch(path)
    raise ValueError(f"unsupported weight file {path}")


def flatten(tensors: list[tuple[str, np.ndarray, str]]) -> tuple[np.ndarray, list[TensorEntry]]:
    """Group-aligned layout: return (padded byte stream, layout)."""
    layout = plan_layout(tensors)
    return padded_stream(tensors, layout), layout


def raw_hash(raw: np.ndarray, layout: list[TensorEntry]) -> str:
    """BLAKE2b over the layout description and the raw bytes: the identity of the declared model."""
    h = hashlib.blake2b(digest_size=32)
    for e in layout:
        h.update(repr(asdict(e)).encode())
    h.update(memoryview(np.ascontiguousarray(raw)))
    return h.hexdigest()
