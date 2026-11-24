"""Kernel backends: CUDA (torch cpp_extension) and the C reference library."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

from .codec_profile import FLAGS, PROFILE, FIELD_BITS, PACKED128, ROW_WORDS89
from .params import P_FIELDS, SETUP_FIELDS

try:
    import torch
except ImportError:  # CPU-only verifier boxes: the C library and numpy are enough
    torch = None

KDIR = Path(__file__).parents[1] / "kernels"
M64 = (1 << 64) - 1
BUILD = Path(__file__).parents[1] / "build" / PROFILE
THREADS = 1024


def _s64(v: int) -> int:
    """uint64 -> the int64 with the same bits (pybind int64 arguments)."""
    return v - (1 << 64) if v >= 1 << 63 else v


def _tensor_path(tensor: bool | None, r: int) -> bool:
    """Whether a link runs on the tensor-core kernel.  That kernel is the matmul: at r = 0 it has no columns, so
    it would hash nothing and give zero entries (all-zero tables, wrong digests) without an error.  None picks it
    exactly when there is a matmul; asking for it at r = 0 is refused."""
    if tensor is None:
        return r > 0
    if tensor and r == 0:
        raise ValueError("the tensor-core path needs r > 0 (r = 0 hashes the words on the scalar path: tensor=False)")
    return bool(tensor)


def pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


def to_u64_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    a = np.ascontiguousarray(arr.view(np.int64))
    if not a.flags.writeable:  # read-only memmaps (setup outputs): copy rather than alias
        a = a.copy()
    return torch.from_numpy(a).to(device)


def default_threads() -> int:
    """Pool threads when none are given: all cores but two, at most 32. The pool's workers spin while a chain
    is in flight; with nearly every hardware thread spinning (62 of 64 on the H200 host) a chain ran 50 to 130
    times slower than at 32 threads, which is also where scaling stopped (1 MiB, r = 2048: 0.9 ms a link)."""
    return max(1, min((os.cpu_count() or 1) - 2, 32))


class CpuLib:
    """ctypes wrapper around posm_cpu.c (built on demand with the system C compiler)."""

    def __init__(self):
        lib = BUILD / "libposm_cpu.so"
        src = KDIR / "posm_cpu.c"
        if not lib.exists() or lib.stat().st_mtime < max(p.stat().st_mtime for p in (src, *KDIR.glob("*.h"))):
            BUILD.mkdir(parents=True, exist_ok=True)
            cc = os.environ.get("CC", "cc")
            subprocess.check_call(
                [cc, *FLAGS, "-O3", "-march=native", "-shared", "-fPIC", f"-I{KDIR}", "-o", str(lib), str(src), "-lpthread"]
            )
        self.lib = ctypes.CDLL(str(lib))
        L = self.lib
        vp, u64, u32 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32
        L.posm_chain_cpu.argtypes = [vp, vp, vp]
        L.posm_chain_cpu_mt.argtypes = [vp, vp, vp]
        L.posm_pool_start.argtypes = [ctypes.c_int]
        L.posm_pool_start.restype = ctypes.c_int
        L.posm_pool_stop.argtypes = []
        L.posm_setup_cpu.argtypes = [vp, vp, vp]
        L.posm_decode_cpu.argtypes = [vp, vp, vp]
        L.posm_setup_cpu_mt.argtypes = [vp, vp, vp, ctypes.c_int]
        L.posm_filler_cpu.argtypes = [vp, u64, u64, u64]
        L.posm_filler_image_cpu_mt.argtypes = [vp, u64, u64, u64, u64, u64, ctypes.c_int]
        L.posm_filler_image_cpu_mt.restype = ctypes.c_int
        L.posm_decode_cpu_mt.argtypes = [vp, vp, vp, ctypes.c_int]
        for f in ("posm_vde_encode", "posm_vde_decode"):
            getattr(L, f).argtypes = [u64, u64]
            getattr(L, f).restype = u64
        L.posm_feistel.argtypes = [u64, u64, u32]
        L.posm_feistel.restype = u64
        L.posm_tweak.argtypes = [u64, u64, u64]
        L.posm_tweak.restype = u64
        L.posm_pack_cpu.argtypes = [vp, vp, u64]
        L.posm_entry_cpu.argtypes = [vp, vp, u64, u64, vp]
        L.posm_image_cpu.argtypes = [vp, vp, u64]
        L.posm_image_at_cpu.argtypes = [vp, vp, u64, u64]
        L.posm_setup_at_cpu.argtypes = [vp, vp, u64, u64, u64, u64]
        L.posm_image_cpu_mt.argtypes = [vp, vp, u64, ctypes.c_int]
        L.posm_decode_unpack_cpu.argtypes = [vp, vp, u64, u64, u64, u64]
        L.posm_siphash.argtypes = [u64] * 6 + [vp]
        L.posm_kv_code_cpu.argtypes = [u64, u64, u64, vp, u64, u64, u64, u64, ctypes.c_int, ctypes.c_int]
        L.posm_kv_code_cpu.restype = None
        L.posm_image_group.argtypes = [u64, vp, vp]
        L.posm_chain_scalar_cpu.argtypes = [vp, vp, vp, vp, u32]
        L.posm_flag_store_release.argtypes = [vp, u64]
        L.posm_flag_store_release.restype = None
        L.posm_fence_acquire.argtypes = []
        L.posm_fence_acquire.restype = None
        L.posm_wait_done.argtypes = [vp]
        L.posm_wait_done.restype = None
        L.posm_param_words.argtypes = []
        L.posm_param_words.restype = u64
        L.posm_setup_param_words.argtypes = []
        L.posm_setup_param_words.restype = u64
        if (L.posm_param_words(), L.posm_setup_param_words()) != (len(P_FIELDS), len(SETUP_FIELDS)):
            raise RuntimeError("posm_core.h's parameter structs and memory_accounting.core.params' field lists differ")

    @staticmethod
    def _ptr(a) -> int:
        if torch is not None and isinstance(a, torch.Tensor):
            assert a.device.type == "cpu" and a.is_contiguous()
            return a.data_ptr()
        return a.ctypes.data

    def chain(self, mem, prm, out, mt: bool = False):
        (self.lib.posm_chain_cpu_mt if mt else self.lib.posm_chain_cpu)(self._ptr(mem), self._ptr(prm), self._ptr(out))

    def flag_store_release(self, flag, value: int):
        """flag[0] = value after a system-scope barrier (the pre-armed chain's host-mapped flag: every store before
        it, the parameter block, is visible to the GPU first, also on a weakly ordered host)."""
        self.lib.posm_flag_store_release(self._ptr(flag), value)

    def fence_acquire(self):
        """System-scope barrier after observing the GPU's done flag, before reading what it published."""
        self.lib.posm_fence_acquire()

    def chain_scalar(self, mem, canon, prm, out, lanes: int = 1):
        """The CUDA scalar kernels' decomposition (posm_core.h, `lanes` strided threads) run serially: tests the
        device code shared through the header on a host without a GPU.  canon: the cheat's source (modes 1, 3)."""
        self.lib.posm_chain_scalar_cpu(self._ptr(mem), self._ptr(canon), self._ptr(prm), self._ptr(out), lanes)

    def setup(self, raw, out, prm, threads: int = 1):
        if FIELD_BITS == 89:
            raise ValueError("single-word setup is 61-bit only; use setup_at with grouped 89-bit payload")
        if threads > 1:
            self.lib.posm_setup_cpu_mt(self._ptr(raw), self._ptr(out), self._ptr(prm), threads)
        else:
            self.lib.posm_setup_cpu(self._ptr(raw), self._ptr(out), self._ptr(prm))

    def decode(self, mem, out, prm, threads: int = 1):
        if threads > 1:
            self.lib.posm_decode_cpu_mt(self._ptr(mem), self._ptr(out), self._ptr(prm), threads)
        else:
            self.lib.posm_decode_cpu(self._ptr(mem), self._ptr(out), self._ptr(prm))

    def filler(self, out, seed: int, start: int):
        self.lib.posm_filler_cpu(self._ptr(out), seed, start, len(out))

    def filler_image(self, out, seed: int, key: int, q0: int, threads: int = 1):
        """out[i] = image of the encoded filler word at global position q0 + i: T = H(Enc(key, filler(seed)))
        for filler the verifier regenerates instead of storing (q0 and len(out) multiples of 4)."""
        if (
            self.lib.posm_filler_image_cpu_mt(self._ptr(out), seed & M64, key & M64, key >> 64, q0, len(out), threads)
            != 0
        ):
            raise ValueError("filler image range must be whole 32-byte groups")

    def entry(self, mem, prm, seed: int, idx: int, out):
        """T_seed[idx] (G words) of block idx: the setup table pass / verifier reference on the CPU."""
        self.lib.posm_entry_cpu(self._ptr(mem), self._ptr(prm), seed, idx, self._ptr(out))

    def pack(self, raw, payload, groups: int):
        self.lib.posm_pack_cpu(self._ptr(raw), self._ptr(payload), groups)

    def setup_at(self, payload, out, key: int, q0: int):
        """Encode payload words whose global positions start at q0 (no permutation)."""
        if FIELD_BITS == 89 and (q0 % ROW_WORDS89 or len(payload) % ROW_WORDS89):
            raise ValueError("p89 setup_at requires complete codec rows")
        self.lib.posm_setup_at_cpu(self._ptr(payload), self._ptr(out), key & M64, key >> 64, q0, len(payload))

    def image_at(self, theta, out, g0: int):
        self.lib.posm_image_at_cpu(self._ptr(theta), self._ptr(out), len(theta) // 4, g0)

    def image(self, theta, out, threads: int = 1):
        """One-way image T = H(Theta'), 32-byte groups, position-bound."""
        if threads > 1:
            self.lib.posm_image_cpu_mt(self._ptr(theta), self._ptr(out), len(theta) // 4, threads)
        else:
            self.lib.posm_image_cpu(self._ptr(theta), self._ptr(out), len(theta) // 4)

    def decode_unpack(self, theta, raw, q0: int, groups: int, key: int):
        self.lib.posm_decode_unpack_cpu(self._ptr(theta), self._ptr(raw), q0, groups, key & M64, key >> 64)

    def kv_code(
        self,
        raw_base: int,
        enc_base: int,
        q_base: int,
        ent: np.ndarray,
        rw: int,
        key: int,
        decode: bool,
        threads: int = 1,
    ):
        """`DeviceBackend.kv_code` on the CPU: raw_base and enc_base are host addresses, ent a uint64 array of 4-word entries."""
        ent = np.ascontiguousarray(ent, dtype=np.uint64)
        self.lib.posm_kv_code_cpu(
            raw_base & M64,
            enc_base & M64,
            q_base & M64,
            ent.ctypes.data,
            len(ent) // 4,
            rw,
            key & M64,
            key >> 64,
            int(bool(decode)),
            max(1, int(threads)),
        )

    def pool_start(self, n: int) -> int:
        return self.lib.posm_pool_start(n)

    def pool_stop(self):
        self.lib.posm_pool_stop()

    def siphash(self, k0, k1, m0, m1, m2, m3):
        out = np.zeros(2, dtype=np.uint64)
        self.lib.posm_siphash(k0, k1, m0, m1, m2, m3, out.ctypes.data)
        return int(out[0]), int(out[1])

    def image_group(self, g: int, w) -> list[int]:
        """The image of one group (4 words) at global group index g (tests)."""
        wa, out = np.asarray([int(x) for x in w], dtype=np.uint64), np.zeros(4, dtype=np.uint64)
        self.lib.posm_image_group(g & M64, wa.ctypes.data, out.ctypes.data)
        return [int(x) for x in out]


class DeviceBackend:
    """The CUDA kernels, built on first use with torch's cpp_extension (needs nvcc and ninja)."""

    def __init__(self, device: str = "cuda"):
        if torch is None:
            raise RuntimeError("the CUDA backend needs torch (uv sync with the gpu group)")
        if not str(device).startswith("cuda"):  # "cuda" or "cuda:N"; kernels run where their tensors live
            raise ValueError(f"unsupported device {device}: this prototype targets CUDA GPUs")
        self.device = device
        self._index = torch.device(device).index  # None for plain "cuda": whatever device is current
        from torch.utils.cpp_extension import load_inline

        cpp = (
            "#include <torch/extension.h>\n"
            "void chain(torch::Tensor mem, torch::Tensor canon, torch::Tensor out, torch::Tensor prm, torch::Tensor state, torch::Tensor pw, torch::Tensor mbuf, int64_t k, int64_t r, int64_t threads);\n"
            "void chain_multi(torch::Tensor mem, torch::Tensor canon, torch::Tensor out, torch::Tensor prm, torch::Tensor state, torch::Tensor partials, torch::Tensor pw, torch::Tensor mbuf, torch::Tensor ablk, int64_t groups, int64_t threads, int64_t n, int64_t k, int64_t r, int64_t be, int64_t g, bool per_link_prep, bool tensor, bool graph);\n"
            "void entry_table(torch::Tensor mem, torch::Tensor canon, torch::Tensor prm, torch::Tensor state, torch::Tensor partials, torch::Tensor pw, torch::Tensor mbuf, torch::Tensor ablk, torch::Tensor table, int64_t seed, int64_t j0, int64_t nblocks, int64_t groups, int64_t threads, int64_t k, int64_t r, int64_t be, int64_t g, bool tensor);\n"
            "void enable_peer(int64_t dev, int64_t peer);\n"
            "void image(torch::Tensor theta, torch::Tensor out, int64_t g0);\n"
            "void filler(torch::Tensor out, int64_t seed, int64_t start);\n"
            "void fill_pages(torch::Tensor dst, torch::Tensor pos0, int64_t pw, int64_t seed, int64_t key_lo, int64_t key_hi, int64_t mask);\n"
            "void pack(torch::Tensor raw, torch::Tensor payload, int64_t groups);\n"
            "void kv_code(int64_t raw_base, int64_t enc_base, int64_t q_base, torch::Tensor ent, int64_t rw, int64_t key_lo, int64_t key_hi, bool decode);\n"
            "void decode_unpack(torch::Tensor theta, torch::Tensor raw, int64_t q0, int64_t groups, int64_t key_lo, int64_t key_hi);\n"
            "void setup_at(torch::Tensor raw, torch::Tensor out, int64_t key_lo, int64_t key_hi, int64_t q0);\n"
            "void setup(torch::Tensor raw, torch::Tensor out, torch::Tensor prm);\n"
            "void decode(torch::Tensor mem, torch::Tensor out, torch::Tensor prm);\n"
            "void decode_at_packed128(torch::Tensor mem, torch::Tensor out, int64_t key_lo, int64_t key_hi, int64_t q0);\n"
            "int64_t side_stream(int64_t dev);\n"
            "int64_t host_device_ptr(torch::Tensor t);\n"
            "int64_t vmm_granularity(int64_t dev);\n"
            "torch::Tensor vmm_reserve(int64_t words, int64_t bytes, int64_t chunk, int64_t dev);\n"
            "void vmm_map(torch::Tensor t, int64_t upto);\n"
            "void chain_multi_armed(torch::Tensor mem, torch::Tensor canon, torch::Tensor out, torch::Tensor prm, torch::Tensor state, torch::Tensor partials, torch::Tensor pw, torch::Tensor mbuf, torch::Tensor ablk, int64_t groups, int64_t threads, int64_t n, int64_t k, int64_t r, int64_t be, int64_t g, bool per_link_prep, bool tensor, int64_t flag_host, int64_t prm_host, int64_t out_host, int64_t done_host);\n"
        )
        cu = "#include <torch/extension.h>\n" + (KDIR / "posm.cu").read_text()
        self._lib = load_inline(
            f"posm_cuda_{PROFILE}",
            cpp_sources=[cpp],
            cuda_sources=[cu],
            functions=[
                "chain",
                "chain_multi",
                "entry_table",
                "enable_peer",
                "image",
                "filler",
                "fill_pages",
                "kv_code",
                "pack",
                "decode_unpack",
                "setup_at",
                "setup",
                "decode",
                "decode_at_packed128",
                "side_stream",
                "chain_multi_armed",
                "host_device_ptr",
                "vmm_granularity",
                "vmm_reserve",
                "vmm_map",
            ],
            extra_include_paths=[str(KDIR)],
            extra_cuda_cflags=["-O3", *FLAGS],
            verbose=False,
        )

    def chain_multi_armed(
        self,
        mem,
        canon,
        out,
        prm,
        state,
        partials,
        pw,
        mbuf,
        groups: int,
        threads: int,
        n: int,
        k: int,
        r: int,
        be: int,
        g: int,
        per_link_prep: bool,
        tensor: bool,
        ablk: torch.Tensor | None,
        flag: torch.Tensor,
        prm_host: torch.Tensor,
        out_host: torch.Tensor,
        done: torch.Tensor,
    ):
        """Pre-launch the chain graph: it waits on `flag` (pinned host memory) and publishes to `out_host`/`done`."""
        tensor = _tensor_path(tensor, r)
        if ablk is None:
            ablk = torch.empty(0, dtype=torch.int8, device=self.device)
        self._lib.chain_multi_armed(
            mem,
            canon,
            out,
            prm,
            state,
            partials,
            pw,
            mbuf,
            ablk,
            groups,
            threads,
            n,
            k,
            r,
            be,
            g,
            per_link_prep,
            tensor,
            flag.data_ptr(),
            prm_host.data_ptr(),
            out_host.data_ptr(),
            done.data_ptr(),
        )

    def host_device_ptr(self, t: torch.Tensor) -> int:
        """Device address of a pinned host tensor, for segment tables that point into host DRAM."""
        return int(self._lib.host_device_ptr(t))

    def reserve(self, words: int, chunk_bytes: int = 32 << 20) -> torch.Tensor:
        """An int64 tensor of `words` over a reserved virtual address range with only its first chunk mapped: `map_reserved`
        backs it in chunks of about chunk_bytes (rounded to the driver's granularity). The last chunk's tail past the words is
        mapped but unused (unchallenged), so chunks are small. Freeing it releases what was mapped."""
        gran = int(self._lib.vmm_granularity(self.device_index))
        chunk = max(gran, chunk_bytes // gran * gran)
        nbytes = -(-max(words, 1) * 8 // chunk) * chunk
        return self._lib.vmm_reserve(words, nbytes, chunk, self.device_index)

    def map_reserved(self, t: torch.Tensor, upto_bytes: int):
        """Back a tensor from `reserve` with device memory up to at least upto_bytes from its start."""
        self._lib.vmm_map(t, upto_bytes)

    @property
    def device_index(self) -> int:
        """The GPU this backend's buffers live on (the extension runs each call on its tensors' device)."""
        return self._index if self._index is not None else torch.cuda.current_device()

    def side_stream(self) -> torch.cuda.ExternalStream:
        """The private highest-priority stream the chains run on (one per device), as a torch stream (event timing)."""
        d = self.device_index
        return torch.cuda.ExternalStream(self._lib.side_stream(d), device=torch.device("cuda", d))

    def sync(self):
        """Device-wide barrier (setup, benchmarks). Not for timing under background load."""
        torch.cuda.synchronize(self.device_index)

    def sync_stream(self):
        """Wait only for the challenge's own stream: the graph side stream signals torch's current
        stream (of this backend's device) with an event, so this returns as soon as the chain is done
        even while a background load keeps other streams busy (torch streams are non-blocking)."""
        torch.cuda.current_stream(self.device_index).synchronize()

    def chain(self, mem, canon, out, prm, state, pw, mbuf, k: int, r: int, threads: int = THREADS):
        """Persistent single-block chain (comparison only)."""
        self._lib.chain(mem, canon, out, prm, state, pw, mbuf, k, r, threads)

    def chain_multi(
        self,
        mem,
        canon,
        out,
        prm,
        state,
        partials,
        pw,
        mbuf,
        groups: int,
        threads: int,
        n: int,
        k: int,
        r: int,
        be: int,
        g: int = 1,
        per_link_prep: bool = False,
        tensor: bool = False,
        graph: bool = False,
        ablk: torch.Tensor | None = None,
    ):
        """Whole-GPU chain on a highest-priority stream: seed tables (once per chain, or per link when the
        seed changes per link), then per link a prepare kernel (tensor path), a grid partial kernel and a
        finish kernel; all shape arguments from the host.  graph=True replays a captured CUDA graph.
        tensor=True (the tensor-core matmul) needs r > 0; None picks it when r > 0."""
        tensor = _tensor_path(tensor, r)
        if ablk is None:
            ablk = torch.empty(0, dtype=torch.int8, device=self.device)
        self._lib.chain_multi(
            mem,
            canon,
            out,
            prm,
            state,
            partials,
            pw,
            mbuf,
            ablk,
            groups,
            threads,
            n,
            k,
            r,
            be,
            g,
            per_link_prep,
            tensor,
            graph,
        )

    def image(self, theta: torch.Tensor, out: torch.Tensor, g0: int = 0):
        """One-way image T = H(Theta') on the device; g0 = global group index of theta[0] (chunking)."""
        self._lib.image(theta, out, g0)

    def entry_table(
        self,
        mem,
        canon,
        prm,
        state,
        partials,
        pw,
        mbuf,
        ablk,
        table,
        seed: int,
        j0: int,
        nblocks: int,
        groups: int,
        threads: int,
        k: int,
        r: int,
        be: int,
        g: int,
        tensor: bool | None = None,
    ):
        """Setup table pass: table[jb * g : (jb + 1) * g] = T_seed[j0 + jb] for jb < nblocks.  tensor: None (the
        default) uses the tensor-core kernel exactly when r > 0; True at r = 0 is refused (it would write zeros)."""
        tensor = _tensor_path(tensor, r)
        self._lib.entry_table(
            mem,
            canon,
            prm,
            state,
            partials,
            pw,
            mbuf,
            ablk,
            table,
            _s64(seed),
            j0,
            nblocks,
            groups,
            threads,
            k,
            r,
            be,
            g,
            tensor,
        )

    def block_buffer(self, block_bytes: int) -> torch.Tensor:
        """Per-link scratch holding the fetched block in the tensor kernel's tiled layout."""
        return torch.zeros(block_bytes, dtype=torch.int8, device=self.device)

    def power_buffer(self, count: int) -> torch.Tensor:
        """Scratch for the per-chain key powers (T1 | T2); filled by the chain's prep kernels."""
        return torch.zeros(count, dtype=torch.int64, device=self.device)

    def matrix_buffer(self, nbytes: int) -> torch.Tensor:
        """Per-chain matrix scratch (tiled int8 matrix + int32 column constants)."""
        return torch.zeros(max(nbytes, 256), dtype=torch.int8, device=self.device)

    def enable_peer(self, dev: int, peer: int):
        """Allow kernels on `dev` to read tensors resident on `peer` (NVLink / PCIe peer access)."""
        self._lib.enable_peer(dev, peer)

    def filler(self, out: torch.Tensor, seed: int, start: int):
        """out[i] = verifier-seeded filler payload word (start + i)."""
        self._lib.filler(out, seed, start)

    def fill_pages(self, dst: torch.Tensor, pos0: torch.Tensor, pw: int, seed: int, key: int, mask: int):
        """Dynamic filler for many pages in one launch: page j (pw words at device address dst[j]) gets
        Enc(key, filler(seed)) at positions pos0[j] .. pos0[j] + pw, masked (dst, pos0: int64 on the device)."""
        self._lib.fill_pages(dst, pos0, pw, _s64(seed & M64), _s64(key & M64), _s64(key >> 64), _s64(mask & M64))

    def kv_code(self, raw_base: int, enc_base: int, q_base: int, ent: torch.Tensor, rw: int, key: int, decode: bool):
        """Encoded dynamic memory (`memory_accounting.prover.kvenc`): encode (or decode) the groups each entry of `ent` names (int64 on
        the device, 4 per entry: raw word offset, encoded word offset, first group, groups); a unit's plaintext is
        rw words at raw_base, its encoded words at enc_base, region positions from q_base (device addresses)."""
        self._lib.kv_code(
            _s64(raw_base), _s64(enc_base), _s64(q_base), ent, rw, _s64(key & M64), _s64(key >> 64), decode
        )

    def pack(self, raw: torch.Tensor, payload: torch.Tensor, groups: int):
        """Pack raw words into selected codec groups."""
        self._lib.pack(raw, payload, groups)

    def decode_unpack(self, theta: torch.Tensor, raw: torch.Tensor, q0: int, groups: int, key: int):
        """Decode selected codec groups of the encoded array into raw words."""
        self._lib.decode_unpack(theta, raw, q0, groups, _s64(key & M64), _s64(key >> 64))

    def setup(self, raw: torch.Tensor, out: torch.Tensor, prm: torch.Tensor):
        if FIELD_BITS == 89:
            raise ValueError("single-word setup is 61-bit only; use setup_at with grouped 89-bit payload")
        self._lib.setup(raw, out, prm)

    def setup_at(self, payload: torch.Tensor, out: torch.Tensor, key: int, q0: int):
        """Encode payload words whose global positions start at q0 into out (no permutation)."""
        self._lib.setup_at(payload, out, _s64(key & M64), _s64(key >> 64), q0)

    def decode(self, mem: torch.Tensor, out: torch.Tensor, prm: torch.Tensor):
        self._lib.decode(mem, out, prm)

    def decode_at_packed128(self, mem: torch.Tensor, out: torch.Tensor, key: int, q0: int):
        """Decode selected packed rows into 128-byte padded payloads at global positions."""
        if not PACKED128:
            raise ValueError("decode_at_packed128 requires packed128")
        self._lib.decode_at_packed128(mem, out, _s64(key & M64), _s64(key >> 64), q0)
