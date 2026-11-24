"""The filler allocation: free HBM after the weights and the scratch allowance, filled with the
verifier-seeded encoded pseudorandom payload so that it is challengeable and regenerable."""

from __future__ import annotations

import torch


class FillerRegion:
    def __init__(self, tensor: torch.Tensor, words: int, key: int | None = None):
        # key: what the filler is encoded under (`static_filler_key`)
        self.tensor, self.words, self.key = (
            tensor,
            words,
            key,
        )

    def segments(self) -> list[tuple[int, int, torch.Tensor]]:
        return [(self.tensor.data_ptr(), self.words, self.tensor)] if self.words else []


def allocate_filler(
    filler_seed: int,
    enc_key: int,
    payload_offset: int,
    scratch_fraction: float,
    device,
    block_bytes: int = 1 << 20,
    reserve_bytes: int = 0,
    max_bytes: int | None = None,
    gpu: int = 0,
) -> FillerRegion:
    """Allocate (free HBM - scratch_fraction of the total - reserve_bytes), rounded down so that Θ' and the
    filler together end on a whole block (the bundle's array is whole blocks: nothing past it), and fill it
    with Enc(key, filler(seed)) at global positions starting at payload_offset (so it continues Θ').
    The key is `memory_accounting.setup.filler.static_filler_key(filler_seed, gpu)`: derived from the verifier's seed and
    this GPU's index as the verifier knows it (the bundle's GPU entry), which the verifier derives too, so the
    filler is distinct per GPU and not the prover's choice. enc_key is the weights' key, which must differ
    from it (the verifier knows the filler key). Grouped 89-bit reconstruction decodes each segment
    with its own key because its plaintext slots carry only 88 bits each.
    reserve_bytes is the prover's own headroom (the engine's activations, workspaces, the chain's
    buffers): whatever is not filled is memory no chain covers. max_bytes caps the region (experiments
    whose cheat simulators need HBM headroom for their copies)."""
    from ..core.backend import DeviceBackend

    dev = torch.device(device)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()  # blocks the caching allocator holds unused count as used in mem_get_info
    free, total = torch.cuda.mem_get_info(dev)
    budget = int(free - scratch_fraction * total - reserve_bytes)
    if max_bytes is not None:
        budget = min(budget, int(max_bytes))
    bw = block_bytes // 8
    words = max(0, (payload_offset + budget // 8) // bw * bw - payload_offset)  # Θ' ‖ filler ends on a block boundary
    from ..setup.filler import static_filler_key

    key = static_filler_key(int(filler_seed), int(gpu))
    if key == int(enc_key):
        raise ValueError("the weights' key is this GPU's filler key, which the verifier knows: pick another --enc-key")
    if words == 0:
        return FillerRegion(torch.empty(0, dtype=torch.int64, device=dev), 0, key)
    be = DeviceBackend("cuda")
    buf = torch.empty(words, dtype=torch.int64, device=dev)
    chunk = 1 << 26
    tmp = torch.empty(min(chunk, words), dtype=torch.int64, device=dev)
    for lo in range(0, words, chunk):
        hi = min(words, lo + chunk)
        be.filler(tmp[: hi - lo], filler_seed, payload_offset + lo)
        be.setup_at(tmp[: hi - lo], buf[lo:hi], key, payload_offset + lo)
    be.sync()
    return FillerRegion(buf, words, key)


def hbm_line(device, static_bytes: int, dynamic_bytes: int, free_min: int | None = None) -> str:
    """The prover's own view of its memory: what the chains cover and what they do not (its scratch)."""
    dev = torch.device(device)
    free, total = torch.cuda.mem_get_info(dev)
    cached = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)  # this process's allocator: reusable
    G = 1 << 30
    line = (
        f"HBM {total / G:.2f} GiB: static {static_bytes / G:.2f} + dynamic {dynamic_bytes / G:.2f} GiB challenged, "
        f"{(total - static_bytes - dynamic_bytes) / G:.2f} GiB not ({100 * (total - static_bytes - dynamic_bytes) / total:.1f}%); "
        f"free now {free / G:.2f} GiB + {cached / G:.2f} GiB unused in the allocator's cache"
    )
    if free_min is not None:
        line += f", least free at a freeze {free_min / G:.2f} GiB"
    return line
