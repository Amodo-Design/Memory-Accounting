"""CPU checks for the 89-bit paths reached by synthetic and dynamic sweeps."""

import unittest

import numpy as np

from memory_accounting.core.backend import CpuLib
from memory_accounting.core.codec_profile import FIELD_BITS
from memory_accounting.core.memory import cpu_synthetic_theta, random_words_np
from memory_accounting.core.params import (Challenge, Keys, MemoryConfig, MODE_DYN_ENCODE_OTF,
                              MODE_ENCODE_OTF, MODE_HONEST, MODE_OTHER_GPU,
                              pack_params, setup_params)
from memory_accounting.prover.kvenc import KV_POS


@unittest.skipUnless(FIELD_BITS == 89, "89-bit codec test")
class Grouped89ReachabilityTest(unittest.TestCase):
    def test_synthetic_setup_roundtrip_and_legacy_guard(self):
        cfg = MemoryConfig(words=8192, perm=False)
        raw = random_words_np(cfg.words, 17).view(np.uint64)
        theta = cpu_synthetic_theta(cfg, 17)
        decoded = np.empty_like(raw)
        lib = CpuLib()
        lib.decode(theta, decoded, setup_params(cfg.words, 1, cfg.seed_t, 0, 0, 0))
        np.testing.assert_array_equal(decoded, raw)
        with self.assertRaisesRegex(ValueError, "61-bit only"):
            lib.setup(raw, np.empty_like(raw), setup_params(cfg.words, 1, cfg.seed_t, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "unpermuted"):
            MemoryConfig(words=8192, perm=True)

    def test_dynamic_share_rebuilds_full_grouped_row(self):
        words, block_bytes = 8192, 4096
        key = 0x102030405060708090A0B0C0D0E0F000
        payload = random_words_np(words, 23).view(np.uint64)
        theta = np.empty_like(payload)
        lib = CpuLib()
        lib.setup_at(payload, theta, key, KV_POS)
        cfg = MemoryConfig(words=words, perm=False)
        ch = Challenge(block_bytes, 1, 0, 12345, words * 8 // block_bytes - 1)
        keys = Keys.deterministic(9)
        honest = np.zeros(2 + ch.n, dtype=np.uint64)
        rebuilt = np.zeros_like(honest)
        keep = words - ch.be
        tail = payload[keep:].copy()
        lib.chain(theta, pack_params(cfg, ch, keys, MODE_HONEST), honest)
        lib.chain(theta, pack_params(cfg, ch, keys, MODE_DYN_ENCODE_OTF,
                                     qcheat=keep, qcheat_hi=words,
                                     canonp=tail.ctypes.data - 8 * keep, ckey=key), rebuilt)
        np.testing.assert_array_equal(rebuilt, honest)

    def test_static_reconstruction_share_uses_grouped_codec(self):
        words, block_bytes = 8192, 16384
        key, canonical_key = 0x102030405060708090A0B0C0D0E0F000, 0xFEDCBA98765432100123456789ABCDEF
        payload = random_words_np(words, 31).view(np.uint64)
        theta, canonical = np.empty_like(payload), np.empty_like(payload)
        lib = CpuLib()
        lib.setup_at(payload, theta, key, 0)
        lib.setup_at(payload, canonical, canonical_key, 0)
        cfg = MemoryConfig(words=words, perm=False, image=True, seed_t=key, seed_c=canonical_key)
        ch = Challenge(block_bytes, 1, 16, 23456, words * 8 // block_bytes - 1)
        keys = Keys.deterministic(11)
        honest = np.zeros(2 + ch.n, dtype=np.uint64)
        lib.chain(theta, pack_params(cfg, ch, keys, MODE_HONEST), honest)
        for mode, source in ((MODE_ENCODE_OTF, payload), (MODE_OTHER_GPU, canonical)):
            rebuilt = np.zeros_like(honest)
            lib.chain(theta, pack_params(cfg, ch, keys, mode,
                                         qcheat=words - ch.be, canonp=source.ctypes.data), rebuilt)
            np.testing.assert_array_equal(rebuilt, honest)


if __name__ == "__main__":
    unittest.main()
