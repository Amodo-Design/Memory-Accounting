"""Run with POSM_VDE_BITS=61|89 and POSM_VDE_TWEAK=element|group64."""
import numpy as np

from memory_accounting.core.backend import CpuLib
from memory_accounting.core.codec_profile import FIELD_BITS, GROUP_ENC, GROUP_RAW, TWEAK
from memory_accounting.core.reference import siphash24_128
from memory_accounting.prover.kvenc import decode_np, empty_np, encode_np


def test_static_and_kv_encoding():
    lib = CpuLib()
    rng = np.random.default_rng(88819)
    key = (1 << 100) + 1701
    start = 504
    groups = 3
    raw = rng.integers(0, 2**64 - 1, GROUP_RAW * groups, dtype=np.uint64)
    payload = np.empty(GROUP_ENC * groups, dtype=np.uint64)
    encoded = np.empty_like(payload)
    decoded = np.empty_like(raw)
    lib.pack(raw, payload, groups)
    lib.setup_at(payload, encoded, key, start)
    memory = np.concatenate([np.zeros(start, dtype=np.uint64), encoded])
    lib.decode_unpack(memory, decoded, start, groups, key)
    np.testing.assert_array_equal(raw, decoded)

    for length in (1, len(raw) - 1, len(raw)):
        group_count = (length + GROUP_RAW - 1) // GROUP_RAW
        kv = encode_np(raw[:length], start, key, 0, group_count)
        np.testing.assert_array_equal(decode_np(kv, start, key, length), raw[:length])
    for offset in (0, 1, 3, 7):
        np.testing.assert_array_equal(empty_np(32, start, key)[offset : offset + 9], empty_np(9, start + offset, key))


def test_p89_encoding_against_python_integers():
    if FIELD_BITS != 89:
        import pytest

        pytest.skip("89-bit representation")
    lib = CpuLib()
    key = (1 << 100) + 1701
    start = 504
    prime = (1 << 89) - 1
    rng = np.random.default_rng(62119)
    raw = rng.integers(0, 2**64 - 1, GROUP_RAW, dtype=np.uint64)
    payload = np.empty(GROUP_ENC, dtype=np.uint64)
    encoded = np.empty_like(payload)
    lib.pack(raw, payload, 1)
    lib.setup_at(payload, encoded, key, start)
    plain = raw.tobytes()
    for row in range(len(encoded) // 8):
        wire = encoded[8 * row : 8 * row + 8].tobytes()
        assert wire[60:] == bytes(4)
        for lane in range(5):
            q = start + 8 * row + (lane if TWEAK == "element" else 0)
            low, high = siphash24_128(key & ((1 << 64) - 1), key >> 64, q, 0x5457454B00000001, 0, 0)
            tweak = ((high << 64) | low) & prime
            if TWEAK == "group64":
                tweak += lane
            offset = 55 * row + 11 * lane
            x = (int.from_bytes(plain[offset : offset + 11], "little") + tweak) % prime
            root = pow(x, 1 << 87, prime)
            expected = root | ((root * root % prime != x) << 89)
            assert int.from_bytes(wire[12 * lane : 12 * lane + 12], "little") == expected


def test_verifier_filler_image_has_both_halves_of_each_row():
    lib = CpuLib()
    seed = 0x83FF_CBB2_9133_10A8
    key = 0xB045_1BCD_8F10_2359_300D_0008_030B_37A3
    start = 10_000_000_000
    payload = np.empty(128, dtype=np.uint64)
    lib.filler(payload, seed, start)
    encoded = np.empty_like(payload)
    lib.setup_at(payload, encoded, key, start)
    expected = np.array(
        [lib.image_group((start + i) // 4, encoded[i : i + 4]) for i in range(0, len(encoded), 4)], dtype=np.uint64
    ).ravel()
    for half_row_offset in (0, 4):
        actual = np.empty(len(encoded) - half_row_offset, dtype=np.uint64)
        lib.filler_image(actual, seed, key, start + half_row_offset, threads=4)
        np.testing.assert_array_equal(actual, expected[half_row_offset:])
