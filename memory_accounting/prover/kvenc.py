"""Encoded dynamic memory (v0.17): the KV cache and the activation rows held VDE-encoded; plaintext only in scratch.

Until v0.16 a live request's KV and the activation rows sat in the dynamic region as plain values. They were
challenged (the live chain) and replayed, but not credited: the prover chooses the tokens, so fabricated or
repeated KV may compress, and a prover could hold less than the region's size. Now every word of the region is
a VDE word, as the free memory around it already was:

  * a *unit* of plaintext (a vLLM KV page of one layer, one cache buffer of the integer model, the activation
    rows) of `plain` 8-byte words is held as `enc_words(plain)` codec words: the plaintext
    packed in the selected profile's groups (plaintext past the unit zero), each payload
    word encoded with the dynamic key (`memory_accounting.prover.dynfill.dyn_key`, from the verifier's seed and the GPU
    index) at its region position, in a tweak domain of its own (`KV_POS`): no word of Θ', the filler or the
    dynamic filler is encoded at such a position, so none equals one of these under any key;
  * memory no request uses holds the encoding of zeros ("empty"). Regenerating its encoding incurs the
    selected codec's setup cost, so a prover holding the plaintext (or nothing, for
    empty memory) instead of the words fails the timing as the "encode on the fly" cheat does;
  * inference reads the plaintext from a scratch buffer of one layer: before a layer runs, the live units of
    that layer are decoded into it (one squaring per word); after it, the groups the layer wrote are encoded
    back. At any pause point the region is the encoding of exactly what the replay rebuilds, with zeros where
    nothing was written, so the verifier encodes the replayed plaintext at the prover's positions;
  * so the whole region, including live data, is credited according to the selected codec's
    variable-bit count, and its credit no longer falls as the prover gets busier.

The plaintext scratch (one layer's KV, the activation buffers) is not in the region: it is scratch, counted
against --max-scratch like Θ''s decode scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # a torch-free verifier encodes with the C library only
    torch = None

from ..core.codec_profile import GROUP_RAW, GROUP_ENC as STATIC_GROUP_ENC, FIELD_BITS, PACKED128, ROW_WORDS89
# Dynamic buffers are byte-addressed: independently writable 121-byte groups.
GROUP_ENC = 16 if PACKED128 else STATIC_GROUP_ENC
DYNAMIC_RAW_BYTES = 121 if PACKED128 else 8 * GROUP_RAW
KV_POS = 1 << 62
M64 = (1 << 64) - 1


def groups_of(plain: int) -> int:
    """Packing groups of a unit of `plain` plaintext words."""
    return -(-8 * int(plain) // DYNAMIC_RAW_BYTES)


def enc_words(plain: int) -> int:
    """Words a unit of `plain` plaintext words takes encoded."""
    return GROUP_ENC * groups_of(plain)


def word_groups(lo: int, hi: int) -> tuple[int, int]:
    """(first group, groups) holding plaintext words [lo, hi) of a unit."""
    if hi <= lo:
        return (8 * lo) // DYNAMIC_RAW_BYTES, 0
    g0 = (8 * lo) // DYNAMIC_RAW_BYTES
    return g0, groups_of(hi) - g0


@dataclass
class EncStore:
    """A segment of the region held encoded: `units` units of `plain` plaintext words of `dtype` (its element type,
    for the layout), as `units * enc_words(plain)` int64 words in `tensor` (None: geometry only, the verifier's
    layout). `shape` is the plaintext's shape (a unit's, or the buffer's), for the layout."""

    tensor: object
    plain: int
    units: int = 1
    dtype: str = "uint8"
    shape: tuple = ()

    @property
    def unit_words(self) -> int:
        return enc_words(self.plain)

    @property
    def words(self) -> int:
        return self.units * self.unit_words

    def layout(self, name: str) -> dict:
        return {
            "name": name,
            "dtype": f"enc.{self.dtype}",
            "shape": [self.units, self.plain] + list(self.shape),
            "words": self.words,
        }

    def prefix(self, groups: int) -> EncStore:
        """The first `groups` groups of a one-unit store (the live part of a static cache buffer)."""
        if self.units != 1:
            raise ValueError("a prefix of a store of one unit only")
        groups = min(int(groups), groups_of(self.plain))
        t = self.tensor[: GROUP_ENC * groups] if self.tensor is not None else None
        return EncStore(t, min(self.plain, (DYNAMIC_RAW_BYTES * groups) // 8), 1, self.dtype, self.shape)


def store_for(t: torch.Tensor, device=None, alloc: bool = True) -> EncStore:
    """The encoded store of a plaintext buffer (one unit of its words), allocated on `device` (default the buffer's)
    and holding the encoding of zeros once `Codec.empty_` or the region's reset fills it."""
    nbytes = t.numel() * t.element_size()
    if nbytes % 8:
        raise ValueError(f"buffer of {nbytes} bytes is not whole 8-byte words")
    plain = nbytes // 8
    dev = device if device is not None else t.device
    e = torch.empty(enc_words(plain), dtype=torch.int64, device=dev) if alloc else None
    return EncStore(e, plain, 1, str(t.dtype).replace("torch.", ""), tuple(t.shape))


ENTRY_GROUPS = 128  # the most groups one entry carries: the CUDA kernel gives each entry one block


def entries(rows, split: int | None = ENTRY_GROUPS) -> np.ndarray:
    """4-word entries (raw word offset, encoded word offset, first group, groups) as the kv kernels take them; an entry
    of more than `split` groups is cut into pieces (the kernels index a unit by absolute group, so the offsets stay)."""
    entries = np.asarray(rows, dtype=np.uint64).reshape(-1, 4) if len(rows) else np.zeros((0, 4), dtype=np.uint64)
    if split and len(entries) and int(entries[:, 3].max()) > split:
        n = entries[:, 3].astype(np.int64)
        k = np.maximum(1, -(-n // split))  # pieces per entry
        r = np.repeat(entries, k, axis=0)
        first = np.repeat(np.cumsum(k) - k, k)
        g = (np.arange(len(r)) - first) * split  # each piece's offset into its entry's groups
        r[:, 2] += g.astype(np.uint64)
        r[:, 3] = np.minimum(split, np.repeat(n, k) - g).astype(np.uint64)
        entries = r
    return np.ascontiguousarray(entries)


class Codec:
    """Encode and decode units of one key on one device (CUDA kernels, or the C library on the CPU)."""

    def __init__(self, key: int, device="cpu", threads: int = 1):
        self.key, self.device, self.threads = int(key), torch.device(device) if torch is not None else device, threads
        self.cuda = torch is not None and self.device.type == "cuda"
        if self.cuda:
            from .dynfill import _backend

            self._be = _backend(self.device)
        else:
            from ..core.backend import CpuLib

            self._lib = CpuLib()

    def ent(self, rows, slot: str | None = None) -> object:
        """Entries ready for `code` on this device (an int64 tensor on CUDA, a uint64 array on the CPU). slot: on CUDA,
        upload through a pinned buffer kept for this slot, without blocking the host (a step's plan, made at every step:
        a blocking copy would wait for the queued forward); the previous upload in the slot is waited for first."""
        e = entries(rows)
        if self.cuda:
            if not len(e):
                return None
            if slot is None:
                from ..core.backend import to_u64_tensor

                return to_u64_tensor(e.reshape(-1), str(self.device))
            return self._upload(slot, e.reshape(-1).view(np.int64))
        return e.reshape(-1)

    def _upload(self, slot: str, e: np.ndarray) -> torch.Tensor:
        bufs = getattr(self, "_slots", None)
        if bufs is None:
            bufs = self._slots = {}
        host, dev, ev = bufs.get(slot, (None, None, None))
        if host is None or host.numel() < len(e):
            n = max(len(e), 2 * (host.numel() if host is not None else 0), 1 << 12)
            host = torch.empty(n, dtype=torch.int64, pin_memory=True)
            dev = torch.empty(n, dtype=torch.int64, device=self.device)
            ev = None
        if ev is not None:
            ev.synchronize()  # the pinned buffer's last copy has been read
        host[: len(e)].copy_(torch.from_numpy(e))
        out = dev[: len(e)]
        out.copy_(host[: len(e)], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        bufs[slot] = (host, dev, ev)
        return out

    def code(self, raw_base: int, enc_base: int, q_base: int, ent, rw: int, decode: bool) -> None:
        """Encode (decode) the groups `ent` names: plaintext units of rw words at raw_base, encoded at enc_base, region
        positions from q_base (addresses on this device; ent from `ent`)."""
        if ent is None or len(ent) == 0:
            return
        if self.cuda:
            self._be.kv_code(raw_base, enc_base, q_base, ent, rw, self.key, decode)
        else:
            self._lib.kv_code(raw_base, enc_base, q_base, ent, rw, self.key, decode, self.threads)

    def encode_range(self, raw: torch.Tensor | None, enc: torch.Tensor, q0: int, lo: int, hi: int) -> None:
        """One unit: encode the groups holding plaintext words [lo, hi) of `raw` (None: zeros) into `enc` (int64,
        the unit's encoded words, region position q0)."""
        g0, ng = word_groups(lo, hi)
        rw = raw.numel() * raw.element_size() // 8 if raw is not None else 0
        self.code(raw.data_ptr() if raw is not None else 0, enc.data_ptr(), q0, self.ent([(0, 0, g0, ng)]), rw, False)

    def decode_range(self, enc: torch.Tensor, q0: int, raw: torch.Tensor, lo: int, hi: int) -> None:
        """One unit: decode the groups holding plaintext words [lo, hi) from `enc` into `raw`."""
        g0, ng = word_groups(lo, hi)
        self.code(
            raw.data_ptr(), enc.data_ptr(), q0, self.ent([(0, 0, g0, ng)]), raw.numel() * raw.element_size() // 8, True
        )

    def empty_(self, enc: torch.Tensor, q0: int) -> None:
        """Encoded zeros over a whole store's words (int64, a multiple of 16, region position q0)."""
        n = enc.numel()
        if n % GROUP_ENC:
            raise ValueError(f"{n} encoded words are not whole groups")
        self.code(0, enc.data_ptr(), q0, self.ent([(0, 0, 0, n // GROUP_ENC)]), 0, False)


# ---- the verifier's side (numpy, the C library)


def encode_np(
    plain: np.ndarray, q0: int, key: int, g0: int = 0, ng: int | None = None, threads: int = 1, lib=None
) -> np.ndarray:
    """Groups [g0, g0 + ng) of a unit's encoding (uint64), its plaintext `plain` (uint64 words, zero past its end),
    the unit's word 0 at region position q0."""
    from ..core.backend import CpuLib

    lib = lib or CpuLib()
    plain = np.ascontiguousarray(plain, dtype=np.uint64)
    ng = groups_of(len(plain)) - g0 if ng is None else ng
    out = np.empty(GROUP_ENC * max(0, ng), dtype=np.uint64)
    if ng > 0:
        # the kernel indexes a unit by absolute group: the output holds groups from g0, so its base is GROUP_ENC * g0 words
        # before it
        lib.kv_code(
            plain.ctypes.data if len(plain) else 0,
            out.ctypes.data - 8 * GROUP_ENC * g0,
            q0,
            entries([(0, 0, g0, ng)]).reshape(-1),
            len(plain),
            key,
            False,
            threads,
        )
    return out


def encode_words_np(plain: np.ndarray, q0: int, key: int, wa: int, wb: int, threads: int = 1, lib=None) -> np.ndarray:
    """Encoded words [wa, wb) of a unit (its word 0 at region position q0, plaintext `plain`)."""
    g0, g1 = wa // GROUP_ENC, -(-wb // GROUP_ENC)
    e = encode_np(plain, q0, key, g0, g1 - g0, threads, lib)
    return e[wa - GROUP_ENC * g0 : wb - GROUP_ENC * g0]


def empty_np(n: int, pos0: int, key: int, lib=None) -> np.ndarray:
    """Encoded zeros at region positions [pos0, pos0 + n): what free encoded memory holds (word by word: a zero
    payload packs to zeros, so a word does not depend on its group)."""
    from ..core.backend import CpuLib

    lib = lib or CpuLib()
    if FIELD_BITS == 89 and (pos0 % ROW_WORDS89 or n % ROW_WORDS89):
        offset = pos0 % ROW_WORDS89
        whole = -(-(offset + n) // ROW_WORDS89) * ROW_WORDS89
        return empty_np(whole, pos0 - offset, key, lib)[offset:offset+n].copy()
    out = np.empty(int(n), dtype=np.uint64)
    if n:
        lib.setup_at(np.zeros(int(n), dtype=np.uint64), out, key, KV_POS | int(pos0))
    return out


def decode_np(enc: np.ndarray, q0: int, key: int, plain_words: int, lib=None) -> np.ndarray:
    """A whole unit's plaintext (tests, debugging)."""
    from ..core.backend import CpuLib

    lib = lib or CpuLib()
    enc = np.ascontiguousarray(enc, dtype=np.uint64)
    out = np.zeros(plain_words, dtype=np.uint64)
    lib.kv_code(
        out.ctypes.data,
        enc.ctypes.data,
        q0,
        entries([(0, 0, 0, groups_of(plain_words))]).reshape(-1),
        plain_words,
        key,
        True,
        1,
    )
    return out[:plain_words]


_EMPTY: dict = {}
EMPTY_BYTES = 1 << 30


def empty_template(n: int, pos0: int, key: int, device) -> torch.Tensor:
    """Cached encoded zeros for a fixed range of a store (restoring a store is then a copy)."""
    k = (str(device), int(n), int(pos0), int(key))
    t = _EMPTY.get(k)
    if t is None:
        t = torch.empty(int(n), dtype=torch.int64, device=device)
        empty_(t, pos0, key)
        if sum(v.numel() for v in _EMPTY.values()) * 8 + n * 8 > EMPTY_BYTES:
            _EMPTY.clear()
        _EMPTY[k] = t
    return t


def empty_(words: torch.Tensor, pos0: int, key: int, chunk: int = 1 << 24) -> None:
    """Encoded zeros into an int64 tensor of any length (region positions from pos0). On CUDA whole groups go through the
    kv kernel (no plaintext: a zero payload is not read), a tail of fewer than 16 words through setup_at."""
    n = words.numel()
    if n == 0:
        return
    if FIELD_BITS == 89 and (pos0 % ROW_WORDS89 or n % ROW_WORDS89):
        offset = pos0 % ROW_WORDS89
        buf = torch.empty(-(-(offset+n)//ROW_WORDS89)*ROW_WORDS89, dtype=words.dtype, device=words.device)
        empty_(buf, pos0-offset, key)
        words.copy_(buf[offset:offset+n])
        return
    if words.device.type == "cuda":
        c = Codec(key, words.device)
        whole = n - n % GROUP_ENC
        if whole:
            c.code(0, words.data_ptr(), pos0, c.ent([(0, 0, 0, whole // GROUP_ENC)]), 0, False)
        if whole < n:
            from .dynfill import _backend

            _backend(words.device).setup_at(
                torch.zeros(n - whole, dtype=torch.int64, device=words.device),
                words[whole:],
                key,
                KV_POS | (int(pos0) + whole),
            )
    else:
        words.copy_(torch.from_numpy(empty_np(n, pos0, key).view(np.int64)))


# ---- static caches (the integer model): one plaintext cache shared by every layer, each layer's held encoded


class EncodedStatic:
    """A static-cache model's KV cache and activation buffer held encoded (`memory_accounting.models.toy.attach_dynamic(encoded=True)`).

    `stores` are the region's buffers: an encoded store per layer per cache buffer ("kv.<l>.<part>", the model's
    `kv_buffers()` names and shapes), then the activation buffer's ("act"). The model's own cache buffers become
    one plaintext set shared by every layer (`share`, the model's: `IntLLM.share_cache`), scratch outside the
    region. Per layer: a forward pre-hook decodes the slots the layer reads and writes, [0, pos0 + T), from the
    layer's stores into the shared buffers; a forward hook encodes back the groups holding the slots it wrote,
    [pos0, pos0 + T). The activation buffer is plaintext scratch too: `act_written(n)` (from the layer boundary
    hook, after the hidden state was written over zeros) encodes it into its store, every group that held or
    holds rows, so the store is always the encoding of the rows of the last boundary and zeros after.
    Slot ranges are in the cache's dim 0 (the integer model's buffers are slots outermost)."""

    def __init__(self, model, act: torch.Tensor, key: int, share):
        self.model, self.act = model, act
        named = model.kv_buffers()
        self.n_layers = len(model.layers)
        per = len(named) // self.n_layers
        if per * self.n_layers != len(named):
            raise ValueError(f"{len(named)} cache buffers for {self.n_layers} layers")
        self.slots = int(named[0][1].shape[0])
        self.stores = [(n, store_for(t)) for n, t in named]
        self.act_store = store_for(act)
        # bytes per slot of each part
        self.row_bytes = [t.numel() * t.element_size() // self.slots for _, t in named[:per]]
        self.per = per
        share()  # the model's layers now all use layer 0's buffers
        self.plain = [t for _, t in model.kv_buffers()[:per]]
        self.codec = None  # the region's (set by `attach`, once the region knows its key)
        self.q0: list[int] = []
        self.act_q0 = 0
        self._act_words = 0
        self._handles = []

    def buffers(self) -> list:
        """The region's buffers, in canonical order: the stores, then the activation buffer's."""
        return self.stores + [("act", self.act_store)]

    def attach(self, region, layers) -> None:
        """Hooks on the layers, once the region (its key, the stores' positions) exists. Register after the
        pause hooks, so a pause at a layer boundary comes before the layer's decode."""
        self.codec = region.codec
        pos = {s.name: s.offset for s in region.segments}
        self.q0 = [pos[n] for n, _ in self.stores]
        self.act_q0 = pos["act"]
        for i, mod in enumerate(layers):
            self._handles.append(mod.register_forward_pre_hook(self._pre(i)))
            self._handles.append(mod.register_forward_hook(self._post(i)))

    def words_of(self, part: int, slots: int) -> int:
        """Plaintext words of a part holding slots [0, slots)."""
        return -(-slots * self.row_bytes[part] // 8)

    def _span(self, args) -> tuple[int, int]:
        h, pos0 = args[0], int(args[1])
        return pos0, pos0 + int(h.shape[0])

    def _pre(self, layer: int):
        def hook(module, args):
            _, end = self._span(args)
            for p in range(self.per):
                k = layer * self.per + p
                self.codec.decode_range(self.stores[k][1].tensor, self.q0[k], self.plain[p], 0, self.words_of(p, end))

        return hook

    def _post(self, layer: int):
        def hook(module, args, out):
            start, end = self._span(args)
            for p in range(self.per):
                k = layer * self.per + p
                lo = start * self.row_bytes[p] // 8
                self.codec.encode_range(self.plain[p], self.stores[k][1].tensor, self.q0[k], lo, self.words_of(p, end))

        return hook

    def act_written(self, n_elems: int) -> None:
        """The activation buffer holds n_elems elements of hidden state, zeros after: encode every group that holds or
        held rows."""
        w = -(-n_elems * self.act.element_size() // 8)
        hi = max(w, self._act_words)
        self.codec.encode_range(self.act, self.act_store.tensor, self.act_q0, 0, hi)
        self._act_words = w

    def reset(self) -> None:
        """The region was put back to encoded zeros (a request ended): no rows held."""
        self._act_words = 0

    def live(self, n: int) -> list:
        """The live region's stores for n fed tokens: each store's groups holding slots [0, n), then the whole
        activation store."""
        out = []
        for k, (name, st) in enumerate(self.stores):
            out.append((name, st.prefix(groups_of(self.words_of(k % self.per, n)))))
        return out + [("act", self.act_store)]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
