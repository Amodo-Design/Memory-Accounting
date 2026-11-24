"""Verifier: recomputes the expected digest from its own copy (synthetic mode, no setup phase)
and applies a deadline."""

from __future__ import annotations

import dataclasses

import numpy as np

from ..core.backend import CpuLib, default_threads
from ..core.params import MAX_LINKS, MODE_HONEST, Challenge, Keys, MemoryConfig, pack_params


class ImageWithFiller:
    """The verifier's static array when it does not store the filler's image: the one-way image of Θ' as
    the setup gave it, then, from word `len(image)` on, the image of the encoded filler, regenerated for
    each block a chain reads from the verifier's own filler seed and the filler key. Filler fills all of
    the GPU's spare memory (over 100 GiB on an H200); its image would be as large, while regenerating a
    1 MiB block costs a few ms on a few cores."""

    def __init__(self, image: np.ndarray, words: int, filler_seed: int, filler_key: int, lib: CpuLib, threads: int = 8):
        if len(image) % 4 or len(image) > words:
            raise ValueError("the stored image must be whole 32-byte groups and no longer than the array")
        self.image, self.words, self.seed, self.key, self.lib, self.threads = (
            image,
            words,
            filler_seed,
            filler_key,
            lib,
            threads,
        )

    def block(self, j: int, bw: int) -> np.ndarray:
        lo, hi = j * bw, (j + 1) * bw
        if hi > self.words:
            raise IndexError(f"block {j} past the end of the array")
        out = np.empty(bw, dtype=np.uint64)
        n = len(self.image)
        if lo < n:
            take = min(hi, n) - lo
            out[:take] = self.image[lo : lo + take]
        if hi > n:
            s = max(lo, n)
            self.lib.filler_image(out[s - lo :], self.seed, self.key, s, self.threads)
        return out


class Verifier:
    def __init__(self, theta_host_np: np.ndarray, cfg: MemoryConfig, keys: Keys, threads: int | None = None):
        self.mem, self.cfg, self.keys = theta_host_np, cfg, keys
        self.lib = CpuLib()
        self.out = np.zeros(2 + MAX_LINKS, dtype=np.uint64)
        # threads caps the recomputation burst on a shared verifier box; default leaves two cores free
        self.lib.pool_start(threads or default_threads())  # idempotent; shared with the CPU cheater

    def expected(self, ch: Challenge):
        if hasattr(self.mem, "block"):
            return self._expected_lazy(ch)
        prm = pack_params(self.cfg, ch, self.keys, MODE_HONEST)
        self.lib.chain(self.mem, prm, self.out, mt=True)
        return int(self.out[0]), int(self.out[1]), self.out[2 : 2 + ch.n].tolist()

    def _expected_lazy(self, ch: Challenge):
        """The same chain one link at a time, asking the array for the block each link reads (the C chain
        derives the next block from the digest; here that step is done in Python)."""
        bw = ch.be
        nb = self.mem.words // bw
        cfg1 = dataclasses.replace(self.cfg, words=bw)
        digest, idx, visited = (0, 0), ch.j0, []
        for link in range(ch.n):
            visited.append(idx)
            blk = self.mem.block(idx, bw)
            self.lib.chain(
                blk,
                pack_params(cfg1, ch, self.keys, MODE_HONEST, nb=1, d0=digest, link0=link, n=1, j0=0, qoff=idx * bw),
                self.out,
                mt=True,
            )
            digest = (int(self.out[0]), int(self.out[1]))
            idx = digest[0] % nb
        return digest[0], digest[1], visited

    def check(self, ch: Challenge, digest: tuple[int, int], rtt_ns: int, deadline_ns: int) -> dict:
        lo, hi, _ = self.expected(ch)
        correct = (lo, hi) == tuple(digest)
        return {"correct": correct, "in_time": rtt_ns <= deadline_ns, "pass": correct and rtt_ns <= deadline_ns}
