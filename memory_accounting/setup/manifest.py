"""The setup manifest: everything both parties need to agree on before challenges start."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.codec_profile import PROFILE
from .layout import TensorEntry

VERSION = 3  # 3: Merkle roots with leaf/node/root domains and the leaf count (`memory_accounting.setup.commit`),
# files relative to the manifest, fields checked on load; version 2 roots no longer verify
BENCHMARK = "throughput-layout"  # sizes only, never accepted by a correctness verifier
SETUP, BUNDLE = "setup", "bundle"  # Manifest.kind
FILE_FIELDS = ("commit_file", "theta_file", "image_file")
M64, M128 = (1 << 64) - 1, (1 << 128) - 1
PM61 = (1 << 61) - 1


@dataclass
class GpuEntry:
    index: int  # GPU index on the prover
    # 128-bit per-GPU VDE key (distinct, never 0); None in a bundle, whose key the prover withholds
    enc_seed: int | None
    commit_root: str  # hex Merkle root of the per-block digests
    commit_file: str  # (nb, 32) uint8 .npy of keyed block digests
    theta_file: str  # encoded array, uint64 .npy (prover loads it)
    image_file: str = ""  # one-way image T = H(Theta'), uint64 .npy the size of the array (what the verifier keeps)
    image_root: str = (
        ""  # stream-decode bundles: Merkle root of keyed block digests of the image (the verifier checks its copy)
    )


@dataclass
class Manifest:
    model_source: str
    raw_bytes: int
    raw_hash: str
    layout: list[TensorEntry]
    block_bytes: int
    payload_words: int  # packed weight words; 0 in a bundle, whose Θ' is its image's length
    filler_seed: int  # verifier-chosen; filler covers [payload_words, accounted_words) (bundle: from the image's end)
    accounted_words: int  # length of the challenged array (multiple of the block)
    hbm_bytes: int  # declared HBM capacity per GPU
    commit_key: str  # hex s0 for the per-block digests
    hash_key: list[int]  # [k0, k1, polyk] challenge-hash keys: `Keys.from_seed(filler_seed)`, derived by both sides;
    # written for the record, must equal it (the verifier never takes its keys from here)
    gpus: list[GpuEntry] = field(default_factory=list)
    # bundles: words [len(image), accounted_words) are filler the verifier regenerates instead of holding its
    # image, Enc(static_filler_key(filler_seed, gpu), filler(filler_seed)) at their global positions. The key is
    # derived by both sides (`memory_accounting.setup.filler.static_filler_key`); this field, written for the record, must equal it
    filler_key: str = ""
    # "setup": the setup phase's manifest, every array recomputable from the weights and the per-GPU keys;
    # "bundle": a stream-decode bundle (`StreamDecoded.export_bundle`), keys withheld, the image committed
    kind: str = SETUP
    # default challenge shape
    r: int = 2048
    k: int = 1024
    g: int = 1
    version: int = VERSION
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    codec: str = PROFILE
    benchmark_dynamic_bytes: int | None = None  # allocated main dynamic challenge region, including padding

    # ---- derived
    @property
    def accounted_bytes(self) -> int:
        return self.accounted_words * 8

    @property
    def scratch_bytes(self) -> int:
        return self.hbm_bytes - self.accounted_bytes

    @property
    def num_blocks(self) -> int:
        return self.accounted_bytes // self.block_bytes

    def is_bundle(self) -> bool:
        """A stream-decode bundle (keys withheld; checked by `verify_bundle`), not a setup manifest (checked by
        recomputation, `verify_setup`). This is what the manifest says it is: the verifier still decides which
        kind it accepts, since only a setup manifest lets it recompute the arrays from the weights."""
        return self.kind == BUNDLE

    def memory_config(self, gpu_index: int, canon_gpu_index: int | None = None, image: bool = True):
        """Prover side: image=True (hashes Theta' groups on the fly). Verifier side: image=False over T
        (a bundle's keys are withheld: the verifier's chain over T does not use them)."""
        from ..core.params import MemoryConfig

        g = self.gpu(gpu_index)
        others = [o for o in self.gpus if o.index != gpu_index]
        other = self.gpu(canon_gpu_index) if canon_gpu_index is not None else (others[0] if others else g)
        return MemoryConfig(
            words=self.accounted_words,
            encoding="vde",
            perm=False,
            seed_t=g.enc_seed or 0,
            seed_c=other.enc_seed or 0,
            image=image,
        )

    def challenge(self, n: int, nonce: int, j0: int):
        from ..core.params import Challenge

        return Challenge(self.block_bytes, n, self.r, nonce, j0, self.k, self.g)

    def gpu(self, index: int) -> GpuEntry:
        for g in self.gpus:
            if g.index == index:
                return g
        raise KeyError(f"no GPU {index} in manifest")

    def save(self, path: str | Path) -> None:
        """Write the manifest. The GPU entries' files are written relative to the manifest's directory, where they
        must lie (`load` refuses a path outside it)."""
        path = Path(path)
        d = asdict(self)
        if self.benchmark_dynamic_bytes is None:
            del d["benchmark_dynamic_bytes"]
        for g in d["gpus"]:
            for f in FILE_FIELDS:
                if g[f]:
                    g[f] = _inside(path.parent, g[f], f, written=True)
        path.write_text(json.dumps(d, indent=1))

    @classmethod
    def load(cls, path: str | Path, *, allow_benchmark: bool = False) -> Manifest:
        """Read and check a manifest: every field has its type and range (keys 128-bit and canonical, sizes
        positive, the challenged array whole blocks, a GPU index an integer), and the GPU entries' files lie in
        the manifest's directory (they come back as absolute paths there). The manifest is the prover's, and the
        verifier opens the files it names: a path outside that directory is refused, not read. Version 2
        manifests (v0.11 and earlier, which could also lack `kind`) are refused: their Merkle roots are of the
        old tree."""
        path = Path(path)
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: not a manifest (a JSON object)")
        recorded = data.get("codec", "p61_element")
        if recorded != PROFILE:
            raise ValueError(f"bundle codec {recorded} differs from verifier codec {PROFILE}")
        if data.get("version") != VERSION:
            raise ValueError(
                f"manifest version {data.get('version')} != {VERSION}"
                + (
                    "; its commitments are Merkle roots of the old tree (no leaf/node domains): run the setup or "
                    "the bundle export again"
                    if data.get("version") == 2
                    else ""
                )
            )
        if data.get("kind") == BENCHMARK and not allow_benchmark:
            raise ValueError("performance-only layout has no reference image; use the full checked bundle for verification")
        if data.get("kind") not in (SETUP, BUNDLE, BENCHMARK):
            raise ValueError(f"manifest kind {data.get('kind')!r} is neither {SETUP!r} nor {BUNDLE!r}")
        try:
            data["layout"] = [TensorEntry(**e) for e in _typed(data, "layout", list)]
            data["gpus"] = [GpuEntry(**g) for g in _typed(data, "gpus", list)]
            manifest = cls(**data)
        except TypeError as e:
            raise ValueError(f"{path}: malformed manifest: {e}") from None
        problems = manifest.field_problems()
        if problems:
            raise ValueError(f"{path}: " + "; ".join(problems))
        for g in manifest.gpus:
            for f in FILE_FIELDS:
                if getattr(g, f):
                    setattr(g, f, _inside(path.parent, getattr(g, f), f))
        return manifest

    def field_problems(self) -> list[str]:
        """Fields of the wrong type or out of range (`load` refuses a manifest with any). What comes before the
        verifier's checks of what the fields say (`memory_accounting.setup.verify.manifest_problems`): a block of 0 bytes used
        to raise ZeroDivisionError there, and a GPU index could be a string that went into file names."""
        out = []
        if self.benchmark_dynamic_bytes is not None and (
            not _is_int(self.benchmark_dynamic_bytes) or not _is_int(self.hbm_bytes)
            or not 0 < self.benchmark_dynamic_bytes <= self.hbm_bytes
        ):
            out.append("benchmark_dynamic_bytes must be a positive byte count within the declared HBM")
        for name in ("raw_bytes", "payload_words", "accounted_words", "hbm_bytes"):
            if not _is_int(getattr(self, name)) or getattr(self, name) < 0:
                out.append(f"{name} {getattr(self, name)!r} is not a non-negative integer")
        if (
            not _is_int(self.block_bytes)
            or self.block_bytes <= 0
            or self.block_bytes % 32
            or self.block_bytes > 1 << 30
        ):
            out.append(f"block_bytes {self.block_bytes!r} is not a positive multiple of 32 (at most 1 GiB)")
        elif _is_int(self.accounted_words) and (
            self.accounted_words <= 0 or self.accounted_words % (self.block_bytes // 8)
        ):
            out.append(
                f"the challenged array ({self.accounted_words!r} words) is not a positive number of {self.block_bytes}-byte blocks"
            )
        if not _is_int(self.filler_seed) or not 0 <= self.filler_seed <= M64:
            out.append("filler_seed is not a 64-bit non-negative integer")
        for name, lo in (("r", 0), ("k", 1), ("g", 1)):
            if not _is_int(getattr(self, name)) or getattr(self, name) < lo:
                out.append(f"{name} {getattr(self, name)!r} is out of range")
        if not (
            isinstance(self.hash_key, list)
            and len(self.hash_key) == 3
            and all(_is_int(x) for x in self.hash_key)
            and 0 <= self.hash_key[0] <= M64
            and 0 <= self.hash_key[1] <= M64
            and 0 < self.hash_key[2] < PM61
        ):
            out.append("hash_key is not [k0, k1, polyk] with 64-bit k0, k1 and 0 < polyk < 2^61 - 1")
        for name in ("model_source", "raw_hash", "commit_key", "filler_key", "kind", "created"):
            if not isinstance(getattr(self, name), str):
                out.append(f"{name} is not a string")
        if isinstance(self.raw_hash, str) and self.raw_hash and not re.fullmatch(r"[0-9a-f]{64}", self.raw_hash):
            out.append("raw_hash is not 64 lowercase hex digits")
        if (
            isinstance(self.commit_key, str)
            and self.commit_key
            and not re.fullmatch(r"(?:[0-9a-f]{2}){1,64}", self.commit_key)
        ):
            out.append("commit_key is not 1 to 64 bytes of lowercase hex")
        if isinstance(self.filler_key, str) and self.filler_key and parse_key(self.filler_key) is None:
            out.append(
                f"filler_key {self.filler_key!r} is not a 128-bit key in plain lowercase hex (1 to 32 digits, not 0)"
            )
        for e in self.layout:
            if not (
                isinstance(e.name, str)
                and isinstance(e.dtype, str)
                and isinstance(e.shape, list)
                and all(_is_int(x) and x >= 0 for x in e.shape)
                and all(_is_int(x) and x >= 0 for x in (e.nbytes, e.byte_offset, e.payload_offset, e.payload_words))
            ):
                out.append(f"layout entry {e.name!r} has a field of the wrong type or sign")
        for g in self.gpus:
            if not _is_int(g.index) or not 0 <= g.index < 1 << 32:
                out.append(f"GPU index {g.index!r} is not a non-negative integer")
            # 0 passes here and is refused by manifest_problems ("no encoding key"), which says why
            if g.enc_seed is not None and (not _is_int(g.enc_seed) or not 0 <= g.enc_seed <= M128):
                out.append(f"GPU {g.index!r}: the encoding key is not a 128-bit non-negative integer")
            for f in ("commit_root", "image_root"):
                v = getattr(g, f)
                if not isinstance(v, str) or (v and not re.fullmatch(r"[0-9a-f]{64}", v)):
                    out.append(f"GPU {g.index!r}: {f} is not 64 lowercase hex digits")
            for f in FILE_FIELDS:
                if not isinstance(getattr(g, f), str):
                    out.append(f"GPU {g.index!r}: {f} is not a string")
        return out


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _typed(d: dict, name: str, typ):
    v = d.get(name)
    if not isinstance(v, typ):
        raise ValueError(f"manifest field {name!r} is not a {typ.__name__}")
    return v


def parse_key(s) -> int | None:
    """A 128-bit key written as plain lowercase hex (no sign, prefix, spaces or underscores; 1 to 32 digits) and
    not 0, as an int; None for anything else. `int(s, 16)` also takes '-…', '0x…', ' …' and '_', so one key had
    many spellings, and a negative one aliased a real key in the 128-bit arithmetic while passing the collision
    checks, which compared the value as written."""
    if not isinstance(s, str) or not re.fullmatch(r"[0-9a-f]{1,32}", s):
        return None
    v = int(s, 16)
    return v if v else None


def _inside(base: Path, value: str, what: str, written: bool = False) -> str:
    """A GPU entry's file, which must lie in the manifest's directory (after resolving links and '..'). Loading
    (written=False): a relative path is relative to that directory; returns the absolute path. Saving
    (written=True): a relative path is relative to the working directory, as the writer made it; returns the
    path relative to the manifest's directory."""
    root = base.resolve()
    p = Path(value)
    full = (p if p.is_absolute() else (Path.cwd() / p if written else root / p)).resolve()
    if not full.is_relative_to(root):
        raise ValueError(
            f"{what} {value!r} is outside the manifest's directory {root}: the verifier opens only files that lie there"
        )
    return full.relative_to(root).as_posix() if written else str(full)
