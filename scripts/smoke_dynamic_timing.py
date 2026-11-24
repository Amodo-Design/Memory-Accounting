"""Small GPU correctness check for all dynamic timing paths and the armed response path."""

import json
import torch

from memory_accounting.core.params import Challenge, Keys
from memory_accounting.prover.dynamic import DynamicRegion, DynamicResponder, PauseController
from memory_accounting.prover.kvenc import EncStore, enc_words


def main():
    page_plain_words = 1920
    pages = 4096
    page_encoded_words = enc_words(page_plain_words)
    storage = torch.empty(2 * pages * page_encoded_words, dtype=torch.int64, device="cuda:0")
    stores = [
        (f"kv.{i}", EncStore(storage[i * pages * page_encoded_words:(i + 1) * pages * page_encoded_words],
                              page_plain_words, pages, "bfloat16", (2, 16, 480)))
        for i in range(2)
    ]
    region = DynamicRegion(stores, 1 << 20, 1234567, device="cuda:0", gpu=0)
    for name, _ in stores:
        region.fill(name)
    torch.cuda.synchronize()
    d = DynamicResponder(region, lambda: {"requests": [], "in_flight": False}, PauseController(),
                         Keys.deterministic(91), prearm=False)
    specs = [f"{n}{suffix}@0.01" for suffix in ("", "_control")
             for n in ("encode_otf", "partial", "other_gpu", "dram_copy", "cpu")]
    d.prepare_timing_shares(specs)
    assert d.timing_shares["partial@0.01"][0][2] == d.timing_shares["partial_control@0.01"][0][2]
    keep, end = region.share_span
    ch = Challenge(1 << 20, 4, 256, 412348734, keep // ((1 << 20) // 8))
    full = Challenge(1 << 20, 64, 2048, 412348735, keep // ((1 << 20) // 8))
    d.set_posture("honest")
    expected = d.prover.respond(ch)[:3]
    full_expected = d.prover.respond(full)[:3]
    assert expected[2][0] == keep // ((1 << 20) // 8)
    out = {"span": [keep, end], "honest": list(expected[:2]), "trace": expected[2]}
    for spec in specs:
        d.set_posture(spec)
        got = d.prover.respond(ch)[:3]
        out[spec] = {"digest": list(got[:2]), "correct": got == expected}
        d.prover.arm(ch)
        armed = d.prover.fire(ch)[:3]
        out[spec]["armed_correct"] = armed == expected
        if spec.startswith(("encode_otf", "other_gpu")):
            out[spec]["full_correct"] = d.prover.respond(full)[:3] == full_expected
    print(json.dumps(out), flush=True)
    if not all(out[spec]["correct"] and out[spec]["armed_correct"] and
               out[spec].get("full_correct", True) for spec in specs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
