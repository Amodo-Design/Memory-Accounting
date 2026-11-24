"""Verifier-owned, pre-seed registration of plaintext weights.

The registration is written before a verifier issues a fresh seed. The receipt binds
the seed to the exact registration file; per-GPU encoding keys are derived from an
independent, fresh 128-bit seed in that receipt.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

from ..core.codec_profile import PROFILE
from .weights import flatten, load_tensors, raw_hash

REGISTRATION_VERSION = 1
RECEIPT_VERSION = 1
KEY_DOMAIN = b"PoSM weight encoding key v1\x00"


def _valid_indices(indices) -> bool:
    return (isinstance(indices, list) and bool(indices)
            and all(type(i) is int and 0 <= i < 1 << 32 for i in indices)
            and len(indices) == len(set(indices)))


def _write_new(path: str | Path, value: dict) -> None:
    """Never overwrite an issued registration or seed receipt."""
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def register_weights(weights_path: str | Path, out: str | Path, gpu_indices: list[int]) -> dict:
    if not _valid_indices(gpu_indices):
        raise ValueError("registration needs distinct GPU indices")
    raw, layout = flatten(load_tensors(weights_path))
    registration = {
        "version": REGISTRATION_VERSION,
        "codec": PROFILE,
        "raw_hash": raw_hash(raw, layout),  # includes the tensor layout and padded bytes
        "raw_bytes": int(raw.nbytes),
        "tensor_count": len(layout),
        "gpu_indices": gpu_indices,
    }
    _write_new(out, registration)
    return registration


def register_stream_weights(reference: str | Path, out: str | Path, gpu_index: int) -> dict:
    from .stream_reference import read_reference

    if not _valid_indices([gpu_index]):
        raise ValueError("invalid GPU index")
    with Path(reference).open("rb") as f:
        header, raw_bytes = read_reference(f)
        f.seek(0)
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    registration = {
        "version": REGISTRATION_VERSION,
        "codec": PROFILE,
        "weight_layout": "stream",
        "raw_hash": digest,
        "raw_bytes": raw_bytes,
        "tensor_count": len(header["slots"]),
        "gpu_indices": [gpu_index],
    }
    _write_new(out, registration)
    return registration


def load_registration(path: str | Path) -> dict:
    registration = json.loads(Path(path).read_text())
    if (not isinstance(registration, dict) or registration.get("version") != REGISTRATION_VERSION
            or registration.get("codec") != PROFILE
            or not isinstance(registration.get("raw_hash"), str)
            or len(registration["raw_hash"]) != 64
            or not all(c in "0123456789abcdef" for c in registration["raw_hash"])
            or not isinstance(registration.get("raw_bytes"), int) or registration["raw_bytes"] < 0
            or not isinstance(registration.get("tensor_count"), int) or registration["tensor_count"] < 0):
        raise ValueError("invalid pre-seed weight registration")
    indices = registration.get("gpu_indices")
    if not _valid_indices(indices):
        raise ValueError("invalid GPU indices in pre-seed weight registration")
    return registration


def issue_seed(registration_path: str | Path, out: str | Path) -> dict:
    registration_path = Path(registration_path)
    registration = load_registration(registration_path)
    receipt = {
        "version": RECEIPT_VERSION,
        "codec": PROFILE,
        "registration_sha256": hashlib.sha256(registration_path.read_bytes()).hexdigest(),
        "raw_hash": registration["raw_hash"],
        "weight_layout": registration.get("weight_layout", "setup"),
        "gpu_indices": registration["gpu_indices"],
        "filler_seed": f"{secrets.randbits(63):016x}",
        "weight_key_seed": secrets.token_hex(16),
    }
    _write_new(out, receipt)
    return receipt


def load_receipt(path: str | Path, registration_path: str | Path | None = None) -> dict:
    receipt = json.loads(Path(path).read_text())
    if (not isinstance(receipt, dict) or receipt.get("version") != RECEIPT_VERSION
            or receipt.get("codec") != PROFILE
            or not _valid_indices(receipt.get("gpu_indices"))):
        raise ValueError("invalid seed receipt")
    for name in ("raw_hash", "registration_sha256"):
        value = receipt.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {name} in seed receipt")
    for name, length in (("filler_seed", 16), ("weight_key_seed", 32)):
        value = receipt.get(name)
        if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {name} in seed receipt")
    if registration_path is not None:
        registration_path = Path(registration_path)
        registration = load_registration(registration_path)
        if (receipt["registration_sha256"] != hashlib.sha256(registration_path.read_bytes()).hexdigest()
                or receipt["raw_hash"] != registration["raw_hash"]
                or receipt.get("weight_layout", "setup") != registration.get("weight_layout", "setup")
                or receipt["gpu_indices"] != registration["gpu_indices"]):
            raise ValueError("seed receipt does not match the pre-seed weight registration")
    return receipt


def encoding_key(receipt: dict, gpu_index: int) -> int:
    if type(gpu_index) is not int or gpu_index not in receipt["gpu_indices"]:
        raise ValueError("GPU index absent from seed receipt")
    key = bytes.fromhex(receipt["weight_key_seed"])
    value = hmac.digest(key, KEY_DOMAIN + gpu_index.to_bytes(4, "big"), "sha256")[:16]
    return int.from_bytes(value, "big")
