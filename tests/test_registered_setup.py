"""The verifier must bind weights and GPU keys before accepting setup."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from memory_accounting.setup.encode import decode_tensor, load_array
from memory_accounting.setup.prover import run_setup
from memory_accounting.setup.registration import encoding_key, issue_seed, load_receipt, register_weights
from memory_accounting.setup.verify import SetupRejected, verify_setup


class RegisteredSetupTest(unittest.TestCase):
    def test_two_gpu_registered_roundtrip_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.npz"
            np.savez(weights, layer=np.arange(77, dtype=np.float32), embedding=np.arange(131, dtype=np.int16))
            registration_path, receipt_path = root / "registration.json", root / "receipt.json"
            registration = register_weights(weights, registration_path, [0, 1])
            receipt = issue_seed(registration_path, receipt_path)
            self.assertEqual(receipt, load_receipt(receipt_path, registration_path))
            keys = {i: encoding_key(receipt, i) for i in registration["gpu_indices"]}
            self.assertNotEqual(keys[0], keys[1])
            seed = int(receipt["filler_seed"], 16)
            manifest = run_setup(
                weights, root / "prover", [0, 1], seed, block_bytes=4096, accounted_bytes=4096,
                threads=2, enc_seeds=list(keys.values()), expected_raw_hash=registration["raw_hash"],
            )
            checks = verify_setup(
                manifest, weights, root / "verifier", threads=2, filler_seed=seed,
                expected_raw_hash=registration["raw_hash"], expected_enc_seeds=keys,
            )
            self.assertEqual(len(checks), 2)
            self.assertTrue(all(c.root_ok and c.table_ok and c.image_ok is not False for c in checks))
            source = np.load(weights)
            for gpu in manifest.gpus:
                theta = load_array(gpu.theta_file)
                for entry in manifest.layout:
                    np.testing.assert_array_equal(decode_tensor(theta, entry, gpu.enc_seed), source[entry.name])

            altered = root / "altered.npz"
            np.savez(altered, layer=np.arange(77, dtype=np.float32) + 1, embedding=source["embedding"])
            with self.assertRaisesRegex(ValueError, "pre-seed registration"):
                run_setup(
                    altered, root / "bad-prover", [0, 1], seed, block_bytes=4096,
                    accounted_bytes=4096, enc_seeds=list(keys.values()),
                    expected_raw_hash=registration["raw_hash"],
                )
            wrong_key = deepcopy(manifest)
            wrong_key.gpus[0].enc_seed ^= 1
            with self.assertRaisesRegex(SetupRejected, "encoding keys differ"):
                verify_setup(
                    wrong_key, weights, root / "bad-verifier", filler_seed=seed,
                    expected_raw_hash=registration["raw_hash"], expected_enc_seeds=keys,
                )
            missing_gpu = deepcopy(manifest)
            missing_gpu.gpus.pop()
            with self.assertRaisesRegex(SetupRejected, "encoding keys differ"):
                verify_setup(
                    missing_gpu, weights, root / "missing-verifier", filler_seed=seed,
                    expected_raw_hash=registration["raw_hash"], expected_enc_seeds=keys,
                )
            changed_receipt = json.loads(receipt_path.read_text())
            changed_receipt["registration_sha256"] = "0" * 64
            (root / "changed-receipt.json").write_text(json.dumps(changed_receipt))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_receipt(root / "changed-receipt.json", registration_path)
            with self.assertRaises(FileExistsError):
                issue_seed(registration_path, receipt_path)


if __name__ == "__main__":
    unittest.main()
