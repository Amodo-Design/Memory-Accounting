"""Verifier side of setup: recompute every encoded array from the weights and the manifest, check the
prover's commitment tables and roots, and make the one-way image T = H(Theta') the verifier answers
challenges against. The image is computed here from the recomputed array, never taken from the prover;
the prover's image file, if there is one, is only compared with it.

The manifest is the prover's: before recomputing anything the verifier checks the fields that are not the
prover's to choose. The filler seed must be the one the verifier minted (else the prover could have
prepared the filler, or chosen one it can regenerate cheaply); the per-GPU keys must be nonzero and differ
(else one copy answers for every GPU); and no static key may be a dynamic filler key (`memory_accounting.prover.dynfill`).
A stream-decode bundle (`Manifest.kind == "bundle"`) withholds its key and is checked by `verify_bundle`
against its commitment; `bundle_set_problems` checks several. It says nothing about the weights (no hash,
no layout, no size): the verifier does not hold the weights at setup (only the replay of the dynamic
regions uses them), and nothing at setup could tie Θ' to them without the key anyway. Θ''s size is the
length of the image T. A bundle's Θ' region is not credited in the memory claim (`credited_static_words`).
Numpy and the C library only: no torch on the verifier box."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.params import Keys
from .commit import block_leaves, leaves_of, merkle_root
from .encode import build_payload, encode_array, image_array, load_array, write_array
from .filler import static_filler_key
from .layout import GROUP_PAYLOAD_WORDS
from .manifest import Manifest, parse_key
from .weights import flatten, load_tensors, raw_hash


class SetupRejected(ValueError):
    """The manifest fails a check that comes before any recomputation (seed, keys, GPU entries)."""


@dataclass
class GpuCheck:
    index: int
    root_ok: bool
    table_ok: bool
    first_bad_block: int | None
    image_ok: bool | None = None  # the prover's image file equals the verifier's (None: the prover kept none)
    image_file: str = ""  # the verifier's own image, for `memory_accounting.verifier.net verifier --state`


def manifest_problems(manifest: Manifest, filler_seed: int, pad_seed: int | None = None) -> list[str]:
    """What is wrong with a manifest before any recomputation: the filler seed is not the verifier's, the
    challenge-hash keys are not the ones derived from that seed (`Keys.from_seed`), there are
    no GPU entries or they repeat, a setup manifest's per-GPU key is missing, 0 or repeated (a key of 0 used to
    be skipped by the distinct-key check, so two GPUs keyed 0 passed with byte-identical arrays), a bundle
    carries a key it should withhold, or a static key (a per-GPU key, the filler key) is the dynamic filler key
    of one of its GPUs for the pad seed (default: the filler seed, which the provers use as the pad seed
    unless told otherwise)."""
    from ..prover.dynfill import key_collisions

    out = manifest.field_problems()  # (what `Manifest.load` refuses; here for a manifest made in memory)
    if out:
        return out
    if int(manifest.filler_seed) != int(filler_seed):
        out.append(
            f"the manifest's filler seed {int(manifest.filler_seed):x} is not the one the verifier minted ({int(filler_seed):x})"
        )
    if list(manifest.hash_key) != Keys.from_seed(int(filler_seed)).as_list():
        out.append(
            "the manifest's challenge-hash keys are not the ones derived from the verifier's filler seed "
            "(made before v0.18, or by a prover that chose its own: set the prover up again)"
        )
    if not manifest.gpus:
        out.append("the manifest has no GPU entries")
    idx = [g.index for g in manifest.gpus]
    if len(set(idx)) != len(idx):
        out.append(f"GPU entries repeat: {idx}")
    if manifest.is_bundle():
        keyed = []
        if any(g.enc_seed is not None for g in manifest.gpus):
            out.append(
                "a stream-decode bundle carries a weights key: the prover withholds it (None), it is never 0 or a value"
            )
    else:
        keyed = [g.enc_seed for g in manifest.gpus]
        for g in manifest.gpus:
            if g.enc_seed is None or int(g.enc_seed) == 0:
                out.append(
                    f"GPU {g.index} has no encoding key (0 or missing): a setup manifest names a nonzero key per GPU"
                )
        given = [int(k) for k in keyed if k is not None]
        if len(set(given)) != len(given):
            out.append("two GPUs share an encoding key: one copy would answer for both")
    roots = [g.image_root for g in manifest.gpus if g.image_root]
    if len(set(roots)) != len(roots):
        out.append("two GPUs have the same image root")
    fk = None
    if manifest.filler_key:
        fk = parse_key(manifest.filler_key)
        if fk is None:
            out.append(f"the filler key {manifest.filler_key!r} is not a 128-bit key in plain lowercase hex")
    given = [int(k) for k in keyed if k]
    static = given + ([fk] if fk else [])
    if manifest.is_bundle():  # the keys the bundle's filler is encoded under, derived from the verifier's seed
        static += [static_filler_key(int(filler_seed), int(g.index)) for g in manifest.gpus]
    if fk and fk in given:
        out.append("the filler key is a GPU's weight key")
    seeds = {int(filler_seed)} | ({int(pad_seed)} if pad_seed is not None else set())
    for s in seeds:
        out += key_collisions(static, s, idx)
    return out


def verify_setup(
    manifest: Manifest,
    weights_path: str | Path,
    out_dir: str | Path,
    device: str | None = None,
    threads: int | None = None,
    keep_arrays: bool = True,
    *,
    filler_seed: int,
    pad_seed: int | None = None,
    expected_raw_hash: str | None = None,
    expected_enc_seeds: dict[int, int] | None = None,
) -> list[GpuCheck]:
    """filler_seed: the seed the verifier minted and handed the prover (required: the manifest's own is not
    trusted); pad_seed: the dynamic filler's seed if it is not the filler seed. Raises SetupRejected before
    any recomputation when the manifest fails those checks, ValueError on foreign weights."""
    if manifest.is_bundle():
        raise SetupRejected(
            "a stream-decode bundle, not a setup manifest: its keys are withheld, so nothing can be recomputed; "
            "check it with `memory_accounting.setup.cli check-bundle`"
        )
    problems = manifest_problems(manifest, filler_seed, pad_seed)
    if problems:
        raise SetupRejected("; ".join(problems))
    if expected_raw_hash is not None and manifest.raw_hash != expected_raw_hash:
        raise SetupRejected("manifest weights differ from the pre-seed registration")
    if expected_enc_seeds is not None:
        found = {g.index: g.enc_seed for g in manifest.gpus}
        if found != expected_enc_seeds:
            raise SetupRejected("GPU encoding keys differ from the verifier's post-registration seed receipt")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, layout = flatten(load_tensors(weights_path))
    if raw_hash(raw, layout) != manifest.raw_hash:
        raise ValueError("weights do not match the manifest's model hash")
    if raw.nbytes != manifest.raw_bytes:
        raise ValueError("weights size differs from the manifest")
    payload = build_payload(raw, int(filler_seed), manifest.accounted_words)
    key = bytes.fromhex(manifest.commit_key)
    checks = []
    for g in manifest.gpus:
        theta = encode_array(payload, g.enc_seed, device=device, threads=threads)
        leaves = block_leaves(theta, manifest.block_bytes, key)
        root_ok = merkle_root(leaves).hex() == g.commit_root
        table_ok, first_bad = True, None
        cf = Path(g.commit_file)
        if cf.exists():
            theirs = np.load(cf, allow_pickle=False)
            bad = np.nonzero(np.any(theirs != leaves, axis=1))[0] if theirs.shape == leaves.shape else np.array([0])
            table_ok, first_bad = len(bad) == 0, (int(bad[0]) if len(bad) else None)
        else:
            table_ok = False
        image = image_array(theta, device=device, threads=threads)
        image_file = out_dir / f"image_gpu{g.index}.npy"
        write_array(image_file, image)
        image_ok = None
        if g.image_file and Path(g.image_file).exists() and Path(g.image_file).resolve() != image_file.resolve():
            theirs = np.asarray(load_array(g.image_file)).view(np.uint64)
            image_ok = theirs.shape == image.shape and bool(np.array_equal(theirs, image))
        if keep_arrays:
            write_array(out_dir / f"theta_gpu{g.index}.npy", theta)
        np.save(out_dir / f"commit_gpu{g.index}.npy", leaves, allow_pickle=False)
        checks.append(GpuCheck(g.index, root_ok, table_ok, first_bad, image_ok, str(image_file)))
    return checks


def verify_bundle(
    manifest: Manifest,
    gpu_index: int,
    filler_seed: int,
    image_path: str | Path | None = None,
    pad_seed: int | None = None,
) -> list[str]:
    """The verifier's check of a stream-decode bundle (`StreamDecoded.export_bundle`), where it holds neither
    Θ' nor the weights' key and so cannot recompute the image: the manifest checks above, the image's length,
    the filler key, and the image against the bundle's commitment (`image_root`). An empty list: accepted.
    What this binds, and what only an audit with the key can check, is in `export_bundle`'s docstring.

    The bundle's model is not checked: the verifier needs no copy of the weights here, and the bundle's
    `raw_hash`, `layout`, `raw_bytes` and `payload_words` are not read (a bundle writes them empty). Θ''s size
    is the image's length (`bundle_theta_words`): the prover's choice, as everything in the bundle is, and safe
    because it only moves the line between what the claim credits and what it does not
    (`credited_static_words`): a longer image is a smaller credit, and a shorter one puts verifier-regenerated
    filler where the prover's Θ' would have been. The image's length is bound by its commitment (the last
    block's digest is over the words there are).

    The image is Θ''s and nothing more: the rest of the array is the verifier's filler, which it regenerates
    from its own seed and the key it derives for this GPU (`static_filler_key`).
    The bundle used to fix its own image length (anywhere from its declared payload size to the whole array),
    so an image of zeros covering the whole array was accepted and the verifier never regenerated anything:
    a prover holding nothing answered every chain. Such an image is now accepted but credited nothing: it is
    all Θ'."""
    out = manifest_problems(manifest, filler_seed, pad_seed)
    if manifest.field_problems():  # nothing below can be read from fields of the wrong type or range
        return out
    if not manifest.is_bundle():
        out.append(
            "not a stream-decode bundle (a setup manifest: check it with `memory_accounting.setup.cli verify`, which recomputes it)"
        )
    try:
        g = manifest.gpu(gpu_index)
    except KeyError as e:
        return out + [str(e)]
    if manifest.filler_key and parse_key(manifest.filler_key) != static_filler_key(int(filler_seed), int(gpu_index)):
        out.append(
            f"the bundle's filler key is not the one derived from the verifier's filler seed for GPU {gpu_index}"
        )
    path = Path(image_path) if image_path else Path(g.image_file)
    if not g.image_file and not image_path:
        return out + ["the bundle names no image"]
    if not path.exists():
        return out + [f"no image at {path}"]
    img = np.asarray(load_array(path)).view(np.uint64)
    bw = manifest.block_bytes // 8
    if manifest.accounted_words < bw or manifest.accounted_words % bw:
        out.append(
            f"the challenged array is {manifest.accounted_words} words, not a positive number of {manifest.block_bytes}-byte blocks"
        )
    want = int(img.size)
    if img.ndim != 1 or want <= 0 or want % GROUP_PAYLOAD_WORDS:
        out.append(
            f"the image has {want} words, not a positive number of {GROUP_PAYLOAD_WORDS}-word groups (its length is Θ''s)"
        )
    if manifest.accounted_words < want - want % bw:
        out.append(
            f"the challenged array ({manifest.accounted_words} words) does not cover Θ' ({want} words) in whole blocks"
        )
    if not g.image_root or not manifest.commit_key:
        out.append("the bundle carries no commitment to its image")
    elif merkle_root(leaves_of(img, manifest.block_bytes, bytes.fromhex(manifest.commit_key))).hex() != g.image_root:
        out.append("the image does not match the bundle's commitment (image_root)")
    return out


def bundle_theta_words(manifest: Manifest, gpu_index: int, image_path: str | Path | None = None) -> int:
    """Θ''s size in a bundle, in words: the length of its image T (`image_path`, default the bundle's own). The
    bundle does not declare it."""
    path = Path(image_path) if image_path else Path(manifest.gpu(gpu_index).image_file)
    return int(np.asarray(load_array(path)).size)


def credited_static_words(manifest: Manifest, theta_words: int | None = None) -> int:
    """The words of the static array the memory claim may credit: a setup manifest's whole array (the verifier
    recomputed it from the weights and the per-GPU keys, `verify_setup`); a bundle's filler only, words
    [theta_words, accounted_words), where theta_words is its image's length (`bundle_theta_words`). A bundle's Θ' region is challenged, but the verifier holds only the
    image the prover committed to and not the key: nothing it can check shows that the region encodes the
    weights rather than something cheap (an image of zeros regenerates on the fly), so it is not memory the
    chains force. Only an audit with the key (or, later, a proof) can make it creditable. The image's length
    is the prover's choice: a longer one only lowers the credit, and a shorter one moves words into the
    filler, which the verifier regenerates and the chains check. Call it after `verify_bundle`."""
    if manifest.is_bundle():
        if theta_words is None:
            raise ValueError("a bundle's credit needs Θ''s size, its image's length (bundle_theta_words)")
        return max(0, int(manifest.accounted_words) - int(theta_words))
    return int(manifest.accounted_words)


def image_digest(path: str | Path) -> str:
    """Unkeyed BLAKE2b of an image's words: equal for equal images (a bundle's `image_root` is keyed by its
    own random commit key, so two bundles of one image have different roots)."""
    h = hashlib.blake2b(digest_size=32)
    a = np.asarray(load_array(path)).view(np.uint64)
    for lo in range(0, len(a), 1 << 24):
        h.update(memoryview(np.ascontiguousarray(a[lo : lo + (1 << 24)])))
    return h.hexdigest()


def bundle_set_problems(bundles: list[tuple[Manifest, int, str | Path | None]]) -> list[str]:
    """Several bundles checked together (one per GPU of a prover): (manifest, GPU index, image path or None for
    the bundle's own). Each bundle withholds its weights key, so nothing in one bundle shows that another GPU
    holds the same copy of Θ'; across bundles, equal GPU indices, equal filler keys, equal image roots and equal
    images (byte-identical Θ') are refused. The filler keys are derived from the verifier's seed and the GPU
    index (`static_filler_key`; `verify_bundle` checks each), so distinct indices already mean distinct filler;
    keys are compared as integers (as strings, '0x00<hex>' and '<hex>' passed as two keys)."""
    out = []
    seen_idx, seen_fk, seen_root, seen_img = {}, {}, {}, {}
    for i, (m, gi, path) in enumerate(bundles):
        try:
            g = m.gpu(gi)
        except KeyError as e:
            out.append(f"bundle {i}: {e}")
            continue
        fk = parse_key(m.filler_key) if m.filler_key else None
        if m.filler_key and fk is None:
            out.append(f"bundle {i}: the filler key {m.filler_key!r} is not a 128-bit key in plain lowercase hex")
        for table, key, what in (
            (seen_idx, int(gi), "the same GPU index"),
            (seen_fk, fk, "the same filler key"),
            (seen_root, g.image_root.lower() or None, "the same image root"),
        ):
            if key is None:
                continue
            if key in table:
                out.append(f"bundles {table[key]} and {i} have {what}")
            else:
                table[key] = i
        p = Path(path) if path else Path(g.image_file)
        if p.exists():
            dig = image_digest(p)
            if dig in seen_img:
                out.append(
                    f"bundles {seen_img[dig]} and {i} have byte-identical images: one copy would answer for both GPUs"
                )
            else:
                seen_img[dig] = i
    return out
