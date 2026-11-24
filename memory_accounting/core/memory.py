"""Builds the prototype's 'HBM' contents: canonical weights, the challenged GPU's encoded copy,
the cheater's canonical copy, and a host mirror for the verifier / DRAM cheats."""

from __future__ import annotations

import time

import numpy as np

try:
    import torch
except ImportError:
    torch = None
from .backend import DeviceBackend, to_u64_tensor
from .codec_profile import GROUP_ENC, GROUP_RAW, FIELD_BITS
from .params import MemoryConfig, setup_params


def random_words_np(words: int, seed: int) -> np.ndarray:
    """Profile-correct synthetic payload, shared by GPU and CPU setup paths."""
    rng = np.random.default_rng(seed)
    if FIELD_BITS == 89:
        from ..setup.pack import pack89

        if words % GROUP_ENC:
            raise ValueError("grouped 89-bit synthetic memory needs whole codec groups")
        raw = rng.integers(0, 256, words // GROUP_ENC * GROUP_RAW * 8, dtype=np.uint8)
        return pack89(raw)
    return (rng.integers(0, 1 << 28, words, dtype=np.int64) << 32) | rng.integers(0, 1 << 32, words, dtype=np.int64)


def random_words(words: int, device: str, seed: int) -> torch.Tensor:
    """random_words_np as a tensor on the device."""
    return torch.from_numpy(random_words_np(words, seed)).to(device)


def cpu_synthetic_theta(cfg: MemoryConfig, data_seed: int = 1) -> np.ndarray:
    """The synthetic array encoded on the CPU (uint64 view): lets a prover and a verifier run on machines
    without a GPU, for protocol and UI tests. Same payload generator as Hbm; encoded by the C library."""
    import os

    from .backend import CpuLib
    from .params import setup_params

    raw = random_words_np(cfg.words, data_seed).view(np.uint64)
    out = np.empty(cfg.words, dtype=np.uint64)
    prm = np.asarray(setup_params(cfg.words, cfg.enc, cfg.seed_t, int(cfg.perm), cfg.perm_seed, cfg.bits)).view(
        np.uint64
    )
    if not cfg.enc:
        out[:] = raw
    elif FIELD_BITS == 89:
        CpuLib().setup_at(np.ascontiguousarray(raw), out, cfg.seed_t, 0)
    else:
        CpuLib().setup(np.ascontiguousarray(raw), out, np.ascontiguousarray(prm), threads=max(1, (os.cpu_count() or 2) - 1))
    return out


class Hbm:
    def __init__(
        self,
        backend: DeviceBackend,
        cfg: MemoryConfig,
        data_seed: int = 1,
        pattern: str = "random",
        canon_device: str | None = None,
    ):
        """Synthetic contents (sweep experiments, no setup phase). canon_device: where the cheater's canonical copy
        lives ("cuda:1" = the real 'same weights on another GPU' experiment; kernels on the challenged GPU
        then read it over NVLink via peer access)."""
        self.backend, self.cfg = backend, cfg
        dev = backend.device
        self.canon_device = canon_device or dev
        t0 = time.perf_counter()
        if pattern == "random":
            self.raw = random_words(cfg.words, dev, data_seed)
        elif pattern == "zeros":  # trivially compressible weights
            self.raw = torch.zeros(cfg.words, dtype=torch.int64, device=dev)
        else:
            raise ValueError(pattern)
        # Challenged GPU: Theta'_g = Enc(seed_t, perm_g(Theta))
        self.theta_prime = torch.empty_like(self.raw)
        if not cfg.enc:
            self.theta_prime.copy_(self.raw)
        elif FIELD_BITS == 89:
            backend.setup_at(self.raw, self.theta_prime, cfg.seed_t, 0)
        else:
            backend.setup(
                self.raw,
                self.theta_prime,
                to_u64_tensor(setup_params(cfg.words, cfg.enc, cfg.seed_t, int(cfg.perm), cfg.perm_seed, cfg.bits), dev),
            )
        # Cheater's canonical copy ("the same weights on another GPU"): unpermuted, own encoding seed
        if cfg.enc:
            self.canon = torch.empty_like(self.raw)
            if FIELD_BITS == 89:
                backend.setup_at(self.raw, self.canon, cfg.seed_c, 0)
            else:
                backend.setup(
                    self.raw, self.canon, to_u64_tensor(setup_params(cfg.words, 1, cfg.seed_c, 0, 0, cfg.bits), dev)
                )
        else:
            self.canon = self.raw
        backend.sync()
        self._place_canon(dev)
        self.setup_seconds = time.perf_counter() - t0
        self._host_mirror()

    @classmethod
    def from_manifest(
        cls,
        backend: DeviceBackend,
        manifest,
        gpu_index: int,
        canon_device: str | None = None,
        canon_gpu_index: int | None = None,
    ) -> Hbm:
        """Prover runtime: load this GPU's encoded array from the setup phase. The 'other GPU' cheat's
        canonical copy is the other GPU's array (its own seed) placed on canon_device."""
        from ..setup.encode import load_array

        self = cls.__new__(cls)
        self.backend = backend
        dev = backend.device
        self.canon_device = canon_device or dev
        g = manifest.gpu(gpu_index)
        others = [x for x in manifest.gpus if x.index != gpu_index]
        other = manifest.gpu(canon_gpu_index) if canon_gpu_index is not None else (others[0] if others else g)
        self.cfg = manifest.memory_config(gpu_index, canon_gpu_index, image=True)
        t0 = time.perf_counter()
        theta_np = np.ascontiguousarray(np.asarray(load_array(g.theta_file)))
        self.theta_prime = to_u64_tensor(theta_np, dev)
        self.raw = None  # the raw payload is not kept on the prover
        self.canon = (
            to_u64_tensor(np.ascontiguousarray(np.asarray(load_array(other.theta_file))), dev)
            if other is not g
            else self.theta_prime
        )
        backend.sync()
        self._place_canon(dev)
        self.setup_seconds = time.perf_counter() - t0
        self._host_mirror()
        return self

    def _place_canon(self, dev: str) -> None:
        if self.canon_device != dev:
            peer = torch.device(self.canon_device).index
            self.backend.enable_peer(torch.device(dev).index or 0, peer)
            self.canon = self.canon.to(self.canon_device)
            torch.cuda.synchronize(self.canon_device)

    def _host_mirror(self) -> None:
        # Host mirror of the challenged array (verifier reference; DRAM cheats)
        self.theta_host = self.theta_prime.cpu().pin_memory()
        self.theta_host_np = self.theta_host.numpy().view(np.uint64)

    def num_blocks(self, block_bytes: int) -> int:
        return self.cfg.nbytes // block_bytes

    def decode_benchmark(self, reps: int = 5) -> float:
        """Inference-side cost: decode throughput of Theta' in GB/s (bytes of Theta' decoded per second)."""
        out = torch.empty_like(self.theta_prime)
        prm = to_u64_tensor(setup_params(self.cfg.words, self.cfg.enc, self.cfg.seed_t, 0, 0, 0), self.backend.device)
        self.backend.decode(self.theta_prime, out, prm)
        self.backend.sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            self.backend.decode(self.theta_prime, out, prm)
        self.backend.sync()
        dt = (time.perf_counter() - t0) / reps
        if self.cfg.enc and not self.cfg.perm:  # with a permutation, out[q] == raw[perm(q)] (checked in tests)
            assert torch.equal(out, self.raw), "decode(encode(x)) != x"
        return self.cfg.nbytes / dt / 1e9
