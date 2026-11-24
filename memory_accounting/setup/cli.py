"""posm-setup: the offline phase for both parties.

  python -m memory_accounting.setup.cli prover  --weights model.safetensors --out setup/ --gpus 0,1 --filler-seed <hex from verifier> [--device cuda]
  python -m memory_accounting.setup.cli verify  --allow-unregistered --manifest setup/manifest.json --weights model.safetensors --out verifier_state/ --filler-seed <the seed it minted> [--device cuda]
  python -m memory_accounting.setup.cli roundtrip --manifest setup/manifest.json --weights model.safetensors --gpu 0
  python -m memory_accounting.setup.cli register-weights --weights model.safetensors --out registration.json
  python -m memory_accounting.setup.cli issue-seed --registration registration.json --out seed-receipt.json
  python -m memory_accounting.setup.cli prover --weights model.safetensors --out setup/ --gpus 0 --seed-receipt seed-receipt.json
  python -m memory_accounting.setup.cli verify-registered --manifest setup/manifest.json --weights model.safetensors --out verifier_state/ --registration registration.json --seed-receipt seed-receipt.json
  python -m memory_accounting.setup.cli filler-seed        # legacy filler-only seed; does not bind weights before setup
  python -m memory_accounting.setup.cli check-bundle --manifest bundle/manifest.json [more bundles...] --filler-seed <hex>
                                                     # verifier: stream-decode bundles (several: one per GPU, checked together)

The verifier's commands (filler-seed, verify, check-bundle) need numpy and the C library only, no torch, for
.safetensors, .npz and .npy weights (.pt/.pth/.bin state dicts are torch pickles and need torch). check-bundle
needs no weights at all: a bundle declares no model.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time

import numpy as np

from ..core.params import parse_bytes
from .encode import decode_tensor, load_array
from .manifest import Manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PoSM setup phase")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_prover_args(sub.add_parser("prover", help="encode weights per GPU, commit, write manifest"))
    _add_verify_args(sub.add_parser("verify", help="recompute the arrays and check the prover's commitments"))
    reg = sub.add_parser("register-weights", help="verifier: fix plaintext weights before issuing any setup seed")
    reg.add_argument("--weights", required=True)
    reg.add_argument("--out", required=True)
    reg.add_argument("--gpus", default="0,1", help="GPU indices that receive this replicated model")
    stream = sub.add_parser("register-stream", help="verifier: register a trusted stream weight reference")
    stream.add_argument("--reference", required=True)
    stream.add_argument("--out", required=True)
    stream.add_argument("--gpu", type=int, default=0)
    int_reference = sub.add_parser("export-int-reference", help="export the trusted integer model's stream layout on CPU")
    int_reference.add_argument("--weights", required=True)
    int_reference.add_argument("--out", required=True)
    int_reference.add_argument("--max-len", type=int, default=512)
    issue = sub.add_parser("issue-seed", help="verifier: issue fresh seeds bound to a weight registration")
    issue.add_argument("--registration", required=True)
    issue.add_argument("--out", required=True)
    key = sub.add_parser("seed-keys", help="print the filler seed and GPU weight key from a seed receipt")
    key.add_argument("--seed-receipt", required=True)
    key.add_argument("--registration")
    key.add_argument("--gpu", type=int, default=0)
    secure = sub.add_parser("verify-registered", help="verify setup against pre-seed weights and verifier-issued keys")
    _add_registered_verify_args(secure)
    _add_roundtrip_args(
        sub.add_parser("roundtrip", help="decode an encoded array back to the weights and compare byte for byte")
    )
    sub.add_parser("filler-seed", help="mint a filler seed (verifier)")
    _add_check_bundle_args(
        sub.add_parser(
            "check-bundle", help="check stream-decode bundles: image against its commitment, seed and keys (verifier)"
        )
    )
    return ap


def _add_prover_args(parser) -> None:
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", default="0,1", help="prover GPU indices that will hold a copy")
    parser.add_argument("--filler-seed", help="legacy hex filler seed; not a pre-seed weight binding")
    parser.add_argument("--seed-receipt", help="verifier-issued receipt after registering the weights")
    parser.add_argument("--block-bytes", default="1M")
    parser.add_argument("--hbm-bytes", default=None, help="override detected HBM capacity, e.g. 80G")
    parser.add_argument(
        "--accounted-bytes", default=None, help="size of the challenged array, e.g. 1G (default: HBM minus scratch)"
    )
    parser.add_argument("--scratch-fraction", type=float, default=0.10)
    parser.add_argument("--device", default=None, help="cuda to encode on the GPU, default CPU threads")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--r", type=int, default=2048)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument(
        "--no-images", action="store_true", help="skip the one-way image T = H(Theta') the verifier keeps"
    )


def _add_verify_args(parser) -> None:
    parser.add_argument("--allow-unregistered", action="store_true",
                        help="legacy verification without a pre-seed model registration; no Sealstack defence")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--filler-seed",
        required=True,
        help="hex: the filler seed this verifier minted (the manifest's own is not trusted)",
    )
    parser.add_argument(
        "--pad-seed",
        default=None,
        help="hex: the dynamic filler's seed, if it is not the filler seed (key collisions are checked for both)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--no-arrays", action="store_true", help="keep only the commitment tables and the images (not Theta')"
    )


def _add_registered_verify_args(parser) -> None:
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--seed-receipt", required=True)
    parser.add_argument("--pad-seed", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-arrays", action="store_true")


def _add_roundtrip_args(r) -> None:
    r.add_argument("--manifest", required=True)
    r.add_argument("--weights", required=True)
    r.add_argument("--gpu", type=int, default=0)


def _add_check_bundle_args(parser) -> None:
    parser.add_argument(
        "--manifest",
        required=True,
        nargs="+",
        help="one bundle's manifest, or one per GPU (checked together: no two may share a key or an image)",
    )
    parser.add_argument("--filler-seed", required=True, help="hex: the filler seed this verifier minted")
    parser.add_argument("--gpu", type=int, default=None, help="the GPU entry to check (default: the bundle's only one)")
    parser.add_argument("--image", default=None, help="the image to check (default: the bundle's; one bundle only)")
    parser.add_argument("--pad-seed", default=None)


def _hex(text: str | None) -> int | None:
    return int(text, 16) if text else None


def cmd_filler_seed(args) -> int:
    print(f"{secrets.randbits(63):016x}")
    return 0


def cmd_register_weights(args) -> int:
    from .registration import register_weights

    registration = register_weights(args.weights, args.out, [int(x) for x in args.gpus.split(",")])
    print(f"registered {registration['tensor_count']} tensors, raw hash {registration['raw_hash']} in {args.out}")
    return 0


def cmd_issue_seed(args) -> int:
    from .registration import issue_seed

    receipt = issue_seed(args.registration, args.out)
    print(f"issued setup seed after registration {receipt['registration_sha256']} in {args.out}")
    return 0


def cmd_seed_keys(args) -> int:
    from .registration import encoding_key, load_receipt

    receipt = load_receipt(args.seed_receipt, args.registration)
    print(receipt["filler_seed"], f"{encoding_key(receipt, args.gpu):032x}")
    return 0


def cmd_register_stream(args) -> int:
    from .registration import register_stream_weights

    registration = register_stream_weights(args.reference, args.out, args.gpu)
    print(f"registered {registration['tensor_count']} loaded tensors, SHA-256 {registration['raw_hash']}")
    return 0


def cmd_export_int_reference(args) -> int:
    from ..models.intllm import IntLLM
    from .stream_reference import export_reference

    info = export_reference(IntLLM.from_file(args.weights, args.max_len, "cpu"), args.out)
    print(f"wrote {info['tensor_count']} loaded tensors to {args.out}")
    return 0


def cmd_check_bundle(args) -> int:
    from .verify import bundle_set_problems

    if args.image and len(args.manifest) > 1:
        raise SystemExit("--image names one bundle's image: pass one --manifest with it")
    ok, checked = True, []
    for path in args.manifest:
        entry = _check_one_bundle(path, args)
        if entry is None:
            ok = False
        else:
            checked.append(entry)
    if len(checked) > 1:
        for p in bundle_set_problems(checked):
            print("  " + p)
            ok = False
    if len(args.manifest) > 1:
        print("bundles accepted together" if ok else "BUNDLES REJECTED")
    return 0 if ok else 1


def _check_one_bundle(path: str, args):
    """Check one bundle and print the verdict; (manifest, GPU index, image) when it is accepted, else None."""
    from .verify import bundle_theta_words, credited_static_words, verify_bundle

    try:
        manifest = Manifest.load(path)
    except ValueError as e:  # malformed, a path outside its directory, an old version
        print(f"  {path}: {e}\n{path}: BUNDLE REJECTED")
        return None
    gi = args.gpu if args.gpu is not None else (manifest.gpus[0].index if len(manifest.gpus) == 1 else None)
    if gi is None:
        print(f"{path}: {len(manifest.gpus)} GPU entries: pass --gpu")
        return None
    problems = verify_bundle(manifest, gi, int(args.filler_seed, 16), args.image, _hex(args.pad_seed))
    for p in problems:
        print(f"  {path}: {p}")
    if problems:
        print(f"{path}: BUNDLE REJECTED")
        return None
    g = manifest.gpu(gi)
    tw = bundle_theta_words(manifest, gi, args.image)
    print(
        f"{path}: bundle accepted: GPU {gi}, Θ' {tw * 8 / (1 << 20):.1f} MiB (its image's length), image matches its commitment "
        f"(image root {g.image_root[:16]}…, Θ' root {g.commit_root[:16]}…); the memory claim credits its filler, "
        f"{credited_static_words(manifest, tw) * 8 / (1 << 20):.1f} MiB, not Θ' (only an audit with the key can tie Θ' to the weights)"
    )
    return manifest, gi, args.image


def cmd_prover(args) -> int:
    from .prover import run_setup
    from .registration import encoding_key, load_receipt

    if not args.seed_receipt and not args.filler_seed:
        raise SystemExit("prover needs --seed-receipt (registered setup) or legacy --filler-seed")
    receipt = load_receipt(args.seed_receipt) if args.seed_receipt else None
    if receipt and args.filler_seed and int(args.filler_seed, 16) != int(receipt["filler_seed"], 16):
        raise SystemExit("--filler-seed disagrees with --seed-receipt")
    gpu_indices = [int(x) for x in args.gpus.split(",")]
    if receipt and gpu_indices != receipt["gpu_indices"]:
        raise SystemExit("--gpus differs from the pre-seed registration")

    t0 = time.perf_counter()
    manifest = run_setup(
        args.weights,
        args.out,
        gpu_indices,
        int(receipt["filler_seed"] if receipt else args.filler_seed, 16),
        block_bytes=parse_bytes(args.block_bytes),
        hbm_bytes=parse_bytes(args.hbm_bytes) if args.hbm_bytes else None,
        accounted_bytes=parse_bytes(args.accounted_bytes) if args.accounted_bytes else None,
        scratch_fraction=args.scratch_fraction,
        device=args.device,
        threads=args.threads,
        r=args.r,
        k=args.k,
        images=not args.no_images,
        enc_seeds=[encoding_key(receipt, i) for i in gpu_indices] if receipt else None,
        expected_raw_hash=receipt["raw_hash"] if receipt else None,
    )
    print(
        f"model {manifest.raw_bytes / 1e9:.3f} GB -> {manifest.payload_words * 8 / 1e9:.3f} GB packed; accounted {manifest.accounted_bytes / 1e9:.3f} GB "
        f"in {manifest.num_blocks} blocks of {manifest.block_bytes >> 10} KB; scratch {manifest.scratch_bytes / 1e9:.3f} GB of {manifest.hbm_bytes / 1e9:.1f} GB HBM"
    )
    print(
        f"default challenge shape: r={manifest.r} k={manifest.k}; the verifier keeps image_gpu<i>.npy (one-way image of Theta', same size)"
    )
    for g in manifest.gpus:
        print(
            f"  gpu {g.index}: key {g.enc_seed:032x} root {g.commit_root[:16]}…  -> {g.theta_file}"
            + (f", {g.image_file}" if g.image_file else "")
        )
    print(f"manifest {args.out}/manifest.json  ({time.perf_counter() - t0:.1f} s)")
    return 0


def cmd_verify(args) -> int:
    from .verify import SetupRejected, verify_setup
    from .registration import encoding_key, load_receipt

    registered = args.cmd == "verify-registered"
    if not registered and not args.allow_unregistered:
        print("UNREGISTERED setup refused: use verify-registered or explicitly pass --allow-unregistered")
        return 1
    try:
        receipt = load_receipt(args.seed_receipt, args.registration) if registered else None
    except ValueError as e:
        print(f"REGISTRATION REJECTED: {e}")
        return 1

    try:
        m = Manifest.load(args.manifest)
    except ValueError as e:
        print(f"MANIFEST REJECTED: {e}")
        return 1
    t0 = time.perf_counter()
    try:
        checks = verify_setup(
            m,
            args.weights,
            args.out,
            device=args.device,
            threads=args.threads,
            keep_arrays=not args.no_arrays,
            filler_seed=int(receipt["filler_seed"] if receipt else args.filler_seed, 16),
            pad_seed=_hex(args.pad_seed),
            expected_raw_hash=receipt["raw_hash"] if receipt else None,
            expected_enc_seeds={i: encoding_key(receipt, i) for i in receipt["gpu_indices"]} if receipt else None,
        )
    except SetupRejected as e:
        print(f"MANIFEST REJECTED: {e}")
        return 1
    ok = True
    for check in checks:
        good = check.root_ok and check.table_ok and check.image_ok is not False
        status = (
            "OK"
            if good
            else (
                f"FAIL (root_ok={check.root_ok}, table_ok={check.table_ok}, first bad block={check.first_bad_block}, "
                f"prover image matches={check.image_ok})"
            )
        )
        print(f"  gpu {check.index}: {status}; verifier image {check.image_file}")
        ok &= good
    print(f"{'commitments verified' if ok else 'COMMITMENT MISMATCH'} ({time.perf_counter() - t0:.1f} s)")
    return 0 if ok else 1


def cmd_roundtrip(args) -> int:
    from .weights import load_tensors

    manifest = Manifest.load(args.manifest)
    g = manifest.gpu(args.gpu)
    if manifest.is_bundle() or not g.enc_seed:
        raise SystemExit("roundtrip decodes with the GPU's key: a stream-decode bundle withholds it")
    theta = np.asarray(load_array(g.theta_file))
    tensors = {n: a for n, a, _ in load_tensors(args.weights)}
    bad = []
    for e in manifest.layout:
        got = decode_tensor(theta, e, g.enc_seed)
        src = tensors[e.name]
        if not np.array_equal(got.view(np.uint8).ravel(), np.ascontiguousarray(src).view(np.uint8).ravel()):
            bad.append(e.name)
    print(
        f"decoded {len(manifest.layout)} tensors from Θ' (gpu {args.gpu}): "
        + ("all match the source byte for byte" if not bad else f"MISMATCH in {bad[:5]}")
    )
    return 0 if not bad else 1


COMMANDS = {
    "filler-seed": cmd_filler_seed,
    "register-weights": cmd_register_weights,
    "register-stream": cmd_register_stream,
    "export-int-reference": cmd_export_int_reference,
    "issue-seed": cmd_issue_seed,
    "seed-keys": cmd_seed_keys,
    "check-bundle": cmd_check_bundle,
    "prover": cmd_prover,
    "verify": cmd_verify,
    "verify-registered": cmd_verify,
    "roundtrip": cmd_roundtrip,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    code = COMMANDS[args.cmd](args)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
