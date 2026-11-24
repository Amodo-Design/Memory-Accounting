"""The six 128 B timing paths and their controls match honest digests and traces."""

import torch


def test_compact_weight_share():
    if not torch.cuda.is_available():
        return
    from memory_accounting.core.backend import DeviceBackend
    from memory_accounting.core.codec_profile import PROFILE
    from memory_accounting.core.memory import random_words_np
    from memory_accounting.core.params import Challenge, Keys
    from memory_accounting.prover.responder import ResidentArray
    from memory_accounting.prover.strategies import CpuProver, HonestProver, ShareCheat, host_span_table

    block_words = (1 << 20) // 8
    assert PROFILE == "p89_element_packed128"
    key = 0x123456789ABCDEF0123456789ABCDEF
    words = 16 * block_words
    raw = torch.from_numpy(random_words_np(words, 7).view("int64")).to("cuda")
    theta = torch.empty_like(raw)
    DeviceBackend("cuda").setup_at(raw, theta, key, 0)
    share = (3 * block_words, 4 * block_words)
    cut = 8 * block_words + 16  # an honest block crosses the two-segment boundary
    resident = ResidentArray(
        [(theta[:cut].data_ptr(), cut, theta[:cut]),
         (theta[cut:].data_ptr(), words - cut, theta[cut:])], key,
        strategies=("honest", "cpu", "encode_otf@0.01", "partial@0.01",
                    "other_gpu@0.01", "dram_copy@0.01", "cpu@0.01"),
        segment_keys=(key, key), share_span=share,
    )
    keys = Keys.deterministic(5)
    honest = HonestProver(resident, keys)
    cpu = CpuProver(resident, keys, threads=4)
    challenges = [Challenge(1 << 20, 4, 256, 100, 3, 1024),
                  Challenge(1 << 20, 4, 256, 101, 8, 1024),
                  Challenge(1 << 20, 4, 256, 102, 2, 1024)]
    full = Challenge(1 << 20, 128, 2048, 303, 3, 1024)
    expected = [honest.respond(ch)[:3] for ch in challenges]
    full_expected = honest.respond(full)[:3]
    assert all(int(expected[i][2][0]) == challenges[i].j0 for i in range(len(challenges)))
    assert any(ch.j0 == share[0] // block_words for ch in challenges)
    try:
        for cheat in ("encode_otf", "other_gpu", "dram_copy", "cpu"):
            honest.share = ShareCheat(cheat, resident, .01, cpu=cpu)
            assert [honest.respond(ch)[:3] for ch in challenges] == expected, cheat
            if cheat in ("encode_otf", "other_gpu"):
                assert honest.respond(full)[:3] == full_expected, (cheat, "full")
        honest.share = None
        honest.seg_override = (resident.partial_nseg, resident.partial_table)
        assert [honest.respond(ch)[:3] for ch in challenges] == expected, "partial"
        honest.seg_override = None
        nseg, tab = host_span_table(resident.backend, resident.segments, resident.words,
                                    resident.theta_host, *share, "cuda", use_host=False)
        assert nseg == resident.partial_nseg
        honest.seg_override = (nseg, tab)
        assert [honest.respond(ch)[:3] for ch in challenges] == expected, "partial control"
        for ch, expected_one in zip(challenges, expected):
            honest.arm(ch)
            assert honest.fire(ch)[:3] == expected_one, ("partial control", "armed")
        honest.seg_override = None
        for cheat in ("encode_otf", "other_gpu", "dram_copy", "cpu"):
            honest.share = ShareCheat(cheat, resident, .01, cpu=cpu, control=True)
            assert [honest.respond(ch)[:3] for ch in challenges] == expected, (cheat, "control")
            for ch, expected_one in zip(challenges, expected):
                honest.arm(ch)
                assert honest.fire(ch)[:3] == expected_one, (cheat, "control", "armed")
        honest.share = None
        for cheat in ("encode_otf", "other_gpu", "dram_copy", "cpu"):
            honest.share = ShareCheat(cheat, resident, .01, cpu=cpu)
            for ch, expected_one in zip(challenges, expected):
                honest.arm(ch)
                assert honest.fire(ch)[:3] == expected_one, (cheat, "armed")
    finally:
        honest.share = None
        honest.seg_override = None
        cpu.close()


def test_dynamic_timing_share_tracks_live_writes_and_resets():
    """All five hidden representations answer a changed live KV page exactly."""
    if not torch.cuda.is_available():
        return
    from memory_accounting.core.codec_profile import PROFILE
    from memory_accounting.core.params import Challenge, Keys
    from memory_accounting.prover.dynamic import DynamicRegion, DynamicResponder, PauseController
    from memory_accounting.prover.kvenc import EncStore, enc_words
    from memory_accounting.prover.strategies import HonestProver

    assert PROFILE == "p89_element_packed128"
    device = torch.device("cuda:0")
    plain_words, pages = 8192, 1700  # >100 blocks, so the 1% share is smaller than the region
    stride = enc_words(plain_words)
    held = torch.empty(pages * stride, dtype=torch.int64, device=device)
    scratch = torch.zeros((pages, plain_words), dtype=torch.int64, device=device)
    store = EncStore(held, plain_words, pages)
    prefix = torch.zeros((1 << 20) // 8, dtype=torch.int64, device=device)
    buffers = [("prefix", prefix), ("kv.0", store)]
    region = DynamicRegion(buffers, 1 << 20, 0x1234, device=device, gpu=0)
    honest_region = DynamicRegion(buffers, 1 << 20, 0x1234, device=device, gpu=0)
    region.clear("kv.0")
    keys = Keys.deterministic(11)
    honest = HonestProver(honest_region, keys)
    responder = DynamicResponder(region, lambda: {}, PauseController(), keys)
    names = ["partial@0.01", "encode_otf@0.01", "other_gpu@0.01", "dram_copy@0.01", "cpu@0.01"]
    responder.prepare_timing_shares(names)
    assert responder.timing_span[1] == 0  # low page IDs, including a live page
    challenge = Challenge(1 << 20, 4, 2048, 99, 1, 1024)
    stale_scratch = torch.zeros_like(scratch)
    try:
        for index, name in enumerate(names, 1):
            scratch[0].fill_(index * 17)
            region.codec.encode_range(scratch[0], held[:stride], (1 << 20) // 8, 0, plain_words)
            responder.update_timing_pages([0], scratch=scratch)
            responder.set_posture(name)
            assert responder.prover.respond(challenge)[:2] == honest.respond(challenge)[:2], (name, "switch")
            scratch[0].fill_(index * 31)
            region.codec.encode_range(scratch[0], held[:stride], (1 << 20) // 8, 0, plain_words)
            # The encoded store is authoritative even when a caller's scratch
            # page is stale or only partly refreshed after a live write.
            responder.update_timing_pages([0], scratch=stale_scratch)
            assert responder.prover.respond(challenge)[:2] == honest.respond(challenge)[:2], name
            region.fill_pages([("kv.0", 0)], stride)
            responder.update_timing_pages([0], zeroed=True)
            assert responder.prover.respond(challenge)[:2] == honest.respond(challenge)[:2], (name, "reset")
        responder.stage = {"requests": [{"block_ids": [0]}]}
        responder._count_timing_share_live()
        assert responder.timing_live_arms == 1 and responder.timing_live_pages == 1
        assert responder.timing_write_refreshes >= 10
    finally:
        responder.close()
        honest.close()
