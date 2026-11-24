"""Exercise the pre-seed reference check with an independently generated stream bundle."""

import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from memory_accounting.core.backend import CpuLib
from memory_accounting.core.codec_profile import GROUP_ENC, GROUP_RAW, PROFILE
from memory_accounting.core.params import Keys
from memory_accounting.setup.commit import leaves_of, merkle_root
from memory_accounting.setup.filler import static_filler_key
from memory_accounting.setup.manifest import BUNDLE, GpuEntry, Manifest
from memory_accounting.setup.registration import encoding_key, issue_seed, register_stream_weights
from memory_accounting.setup.stream_reference import MAGIC
from memory_accounting.verifier.net import _make_verifier, _static_credit


class RegisteredStreamTest(unittest.TestCase):
    def test_registered_image_and_root_are_verified_at_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, registration, receipt = [root / name for name in ("ref.bin", "registration.json", "receipt.json")]
            groups = 8
            raw = np.arange(groups * GROUP_RAW, dtype=np.uint64)
            header = json.dumps({"codec": PROFILE, "slots": [
                {"name": "model.weight", "dtype": "torch.int64", "shape": [len(raw)], "stride": [],
                 "groups": groups, "bytes": raw.nbytes}
            ]}).encode()
            reference.write_bytes(MAGIC + struct.pack("<Q", len(header)) + header + bytes(-len(header) % 8)
                                  + raw.tobytes())
            register_stream_weights(reference, registration, 0)
            resolved = issue_seed(registration, receipt)
            filler_seed = int(resolved["filler_seed"], 16)
            keys = Keys.from_seed(filler_seed)
            words = groups * GROUP_ENC
            lib = CpuLib()
            payload, theta, image = [np.empty(words, dtype=np.uint64) for _ in range(3)]
            lib.pack(raw, payload, groups)
            lib.setup_at(payload, theta, encoding_key(resolved, 0), 0)
            lib.image(theta, image)
            image_path = root / "image.npy"
            commit_key, block = b"test weight key!", 8 * words

            def root_hash(arr):
                return merkle_root(leaves_of(arr, block, commit_key)).hex()

            gpu = GpuEntry(0, None, root_hash(theta), "", "", str(image_path), image_root=root_hash(image))
            manifest = Manifest(model_source="test", raw_bytes=0, raw_hash="", layout=[], block_bytes=block,
                                payload_words=0, filler_seed=filler_seed, accounted_words=2 * words,
                                hbm_bytes=2 * block, commit_key=commit_key.hex(), hash_key=keys.as_list(),
                                kind=BUNDLE, gpus=[gpu], filler_key=f"{static_filler_key(filler_seed, 0):032x}")
            args = SimpleNamespace(manifest="bundle.json", gpu=0, state=None, filler_seed=resolved["filler_seed"],
                                   weight_reference=reference, weight_registration=registration,
                                   weight_seed_receipt=receipt, verify_threads=2, skip_response_checks=False)
            np.save(image_path, image)
            _, cfg = _make_verifier(args, manifest, keys)
            self.assertTrue(args._registered_stream)
            self.assertGreater(_static_credit(args, cfg), block)

            # A self-consistent image commitment from the prover is insufficient.
            wrong_image = image.copy()
            wrong_image[11] ^= np.uint64(1)
            np.save(image_path, wrong_image)
            gpu.image_root = root_hash(wrong_image)
            with self.assertRaisesRegex(SystemExit, "image differs from the registered encoding"):
                _make_verifier(args, manifest, keys)
            np.save(image_path, image)
            gpu.image_root = root_hash(image)

            # An unchanged image cannot make a changed raw encoding commitment valid.
            gpu.commit_root = "a" * 64
            args.skip_response_checks = True
            with self.assertRaisesRegex(SystemExit, "encoded commitment differs"):
                _make_verifier(args, manifest, keys)
            gpu.commit_root = root_hash(theta)
            self.assertIsNone(_make_verifier(args, manifest, keys)[0])
            self.assertLess(_static_credit(args, cfg), block)

            with reference.open("r+b") as f:
                f.seek(-8, 2)
                f.write(bytes(8))
            with self.assertRaisesRegex(SystemExit, "reference changed after registration"):
                _make_verifier(args, manifest, keys)


if __name__ == "__main__":
    unittest.main()
