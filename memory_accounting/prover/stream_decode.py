"""Stream-decode: keep a model's parameters resident as Θ' and decode one layer at a time.

After the model has loaded its weights, `convert(model)`:
  1. groups the parameters into decode units (one per layer module, one for everything else),
  2. packs and encodes them, unit by unit, into one contiguous Θ' on the device (group-aligned layout),
  3. as soon as a unit is encoded, re-points its parameters at views into a shared scratch buffer sized
     for the largest unit (the original storage is freed),
  4. registers a forward pre-hook per unit that decodes that unit's slice of Θ' into the scratch
     just before the unit runs. The parameters outside the layers (embeddings, norms, head) are decoded
     into the same scratch when they are needed: those used before the layers by a pre-hook on the model,
     those used after them by a hook on the last layer (`rest_phases`). Only parameters read inside the
     layers (the integer model's rope and exp tables) stay decoded, in a small resident buffer.

On CUDA, Θ' is a reserved address range backed with memory chunk by chunk as the encoding reaches it
(`DeviceBackend.reserve`), and the freed weights go back to the driver after each unit, so conversion
needs the weights plus about one unit, not the weights and all of Θ' at once (which capped the model
at under half the GPU). `convert_peak_bytes` is the most of the GPU in use while converting.

HBM then holds Θ' (16/15 = 1.067x the weights), one unit of scratch, and activations: the layout the PoSM
doc assumes (weights encoded, one layer plus activations unverified). The scratch is the larger of one
layer and the embedding or head (a model with a large vocabulary and small layers decodes its head into
more than a layer). Decode is memory-bound, so the cost is about one extra read of Θ' and one write of
the model per forward pass.

A unit is keyed by its layer module's path ("model.layers.3", "encoder.layers.0"), so the layers of
different stacks that share an index (encoder-decoder and vision-language models) stay distinct units,
each decoded by its own module's hook. A parameter keeps its memory layout: Θ' holds its elements in
memory order and the scratch view has its strides, so a parameter stored transposed (vLLM's fp8
`w.t()`) is still transposed after the conversion. A parameter that is not dense in memory is refused.

Works on CPU tensors too (C decode) so the mechanism is testable without a GPU; on CUDA the decode
is the `decode_unpack` kernel and the hooks capture into CUDA graphs because all addresses are fixed.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

import numpy as np
import torch

from ..setup.layout import GROUP_PAYLOAD_WORDS, GROUP_RAW_BYTES

SLOT_GROUPS = 32  # slots padded to 32 groups = 3840 bytes, so every scratch view is 256-byte aligned
# (cuBLAS picks kernels by pointer alignment; unaligned views change accumulation order)
REST = "rest"


@dataclass
class ParamSlot:
    name: str
    shape: tuple
    dtype: torch.dtype
    nbytes: int
    scratch_offset: int  # byte offset inside the unit's scratch
    payload_offset: int  # first payload word in Θ'
    groups: int
    stride: tuple = ()  # the parameter's strides (its elements are stored in memory order)
    view_offset: int | None = (
        None  # where its view starts in the scratch or the resident buffer, if not scratch_offset (a rest slot)
    )

    @property
    def memory_shape(self) -> list[int]:
        """The shape in memory order (outermost first): what the slot's bytes are, read as a dense array."""
        if not self.stride:
            return list(self.shape)
        dims = sorted(range(len(self.shape)), key=lambda d: (-self.stride[d], d))
        return [self.shape[d] for d in dims]


@dataclass
class DecodeUnit:
    key: str  # the layer module's path, e.g. "model.layers.7", or "rest"
    module: torch.nn.Module | None
    slots: list[ParamSlot] = field(default_factory=list)
    scratch_bytes: int = 0
    payload_offset: int = 0
    payload_words: int = 0


def _unit_key(name: str, layer_re: str) -> str:
    """A parameter's unit: the path of the layer module it is in (the name up to the layer index)."""
    m = re.search(layer_re, name)
    return name[: m.end(1)] if m else REST


def _unit_order(keys, first_seen: dict) -> list[str]:
    """rest first, then each stack in the order its parameters appear, its layers by index."""

    def k(key):
        if key == REST:
            return (0, 0, 0)
        m = re.match(r"(.*?)(\d+)$", key)
        stack, idx = (m.group(1), int(m.group(2))) if m else (key, 0)
        return (1, first_seen.get(stack, 0), idx)

    return sorted(keys, key=k)


def memory_bytes(tensor: torch.Tensor, name: str = "") -> torch.Tensor:
    """A tensor's elements in memory order, as a uint8 view (no copy). The tensor must be dense: its
    elements fill one contiguous range of its storage with positive strides (a contiguous tensor, or a
    permutation of one such as a transpose). Anything else (a broadcast, a slice with gaps) is refused:
    its layout could not be rebuilt over a slot."""
    tensor = tensor.detach()
    if tensor.is_contiguous():
        return tensor.reshape(-1).view(torch.uint8)
    big = [(n, s) for n, s in zip(tensor.shape, tensor.stride()) if n > 1]
    span = sum((n - 1) * s for n, s in big) + 1
    if any(s <= 0 for _, s in big) or span != tensor.numel():
        raise ValueError(
            f"{name or 'parameter'}: not dense in memory (shape {tuple(tensor.shape)}, stride {tuple(tensor.stride())}); "
            "stream-decode keeps a parameter's layout and can do so only for a dense one"
        )
    return tensor.as_strided((tensor.numel(),), (1,)).view(torch.uint8)


def plan_units(
    model: torch.nn.Module, layer_re: str = r"layers\.(\d+)\.", module_re: str = r"layers\.(\d+)$"
) -> tuple[dict, list, int]:
    """Decode units of a model, laid out in Θ' (no encoding): (units in payload order, [(name, param)],
    payload words). Every parameter matching layer_re belongs to the unit of the module whose path is its
    name up to the layer index; that module (matched by module_re) gets the decode hook."""
    params = [(n, p) for n, p in model.named_parameters()]
    if not params:
        raise ValueError("model has no parameters")
    units: dict[str, DecodeUnit] = {}
    first_seen: dict[str, int] = {}
    for name, param in params:
        key = _unit_key(name, layer_re)
        units.setdefault(key, DecodeUnit(key, None))
        if key != REST:
            m = re.match(r"(.*?)(\d+)$", key)
            first_seen.setdefault(m.group(1) if m else key, len(first_seen))
    mods = {}
    for mod_name, mod in model.named_modules():
        mm = re.search(module_re, mod_name)
        if mm:
            mods[mod_name[: mm.end(1)]] = mod
    missing = [k for k in units if k != REST and k not in mods]
    if missing:
        raise ValueError(
            f"parameters of {missing[:3]} match layer_re {layer_re!r} but no module matches module_re {module_re!r} "
            "at that path: their unit would never be decoded"
        )
    pay_off = 0
    ordered: dict[str, DecodeUnit] = {}
    for key in _unit_order(units, first_seen):
        unit = units[key]
        unit.module = mods.get(key) if key != REST else model
        unit.payload_offset = pay_off
        s_off = 0
        for name, param in params:
            if _unit_key(name, layer_re) != key:
                continue
            memory_bytes(param, name)  # refuses a parameter whose layout cannot be kept
            nbytes = param.numel() * param.element_size()
            groups = -(-nbytes // (GROUP_RAW_BYTES * SLOT_GROUPS)) * SLOT_GROUPS
            unit.slots.append(
                ParamSlot(
                    name,
                    tuple(param.shape),
                    param.dtype,
                    nbytes,
                    s_off,
                    pay_off,
                    groups,
                    () if param.is_contiguous() else tuple(param.stride()),
                )
            )
            s_off += groups * GROUP_RAW_BYTES
            pay_off += groups * GROUP_PAYLOAD_WORDS
        unit.scratch_bytes = s_off
        unit.payload_words = pay_off - unit.payload_offset
        ordered[key] = unit
    return ordered, params, pay_off


BEFORE, AFTER, RESIDENT = "before", "after", "resident"


def rest_phases(
    model: torch.nn.Module, names: list[str], layer_paths: list[str], resident: str | None = None
) -> dict[str, frozenset]:
    """When each parameter outside the layers (the rest unit) must be decoded: BEFORE the first layer runs, AFTER the
    last, both (a head tied to the embedding), or RESIDENT (decoded once and kept). Names matching `resident` stay
    resident. A model may say (`model.stream_phases(name)`: the integer model reads its embedding and head in its own
    forward and its tables inside every layer); otherwise it comes from where the modules holding the parameter sit
    among the model's modules, in registration order, which is the order transformers and vLLM models run them. A
    parameter held by the model itself, by a module that contains layers, or by one between two layer modules (a
    projector between two stacks) has no such place: it stays resident, and `StreamDecoded` bounds those."""
    declared = getattr(model, "stream_phases", None)
    paths = [p for p, _ in model.named_modules(remove_duplicate=False)]
    pos = {p: i for i, p in enumerate(paths)}
    owners: dict[int, list[str]] = {}
    for path, m in model.named_modules(remove_duplicate=False):  # every path, so a tied head is seen at both
        for q in m._parameters.values():
            if q is not None:
                owners.setdefault(id(q), []).append(path)
    lay = sorted(pos[p] for p in layer_paths)
    pdict = dict(model.named_parameters())
    out = {}
    for n in names:
        if resident and re.search(resident, n):
            out[n] = frozenset({RESIDENT})
        elif declared is not None:
            out[n] = frozenset(declared(n))
        elif not lay:
            out[n] = frozenset({BEFORE})  # no layers: nothing reuses the scratch
        else:
            ph = set()
            for path in owners.get(id(pdict[n]), []):
                if (
                    path == ""
                    or any(layer_path.startswith(path + ".") for layer_path in layer_paths)
                    or lay[0] < pos[path] < lay[-1]
                ):
                    ph = {RESIDENT}
                    break
                ph.add(BEFORE if pos[path] < lay[0] else AFTER)
            out[n] = frozenset(ph or {RESIDENT})
        if not out[n] or not out[n] <= {BEFORE, AFTER, RESIDENT} or (RESIDENT in out[n] and len(out[n]) > 1):
            raise ValueError(f"{n}: phase {sorted(out[n])} is not before, after, both or resident")
    return out


def unit_raw(u: DecodeUnit, pdict: dict, device) -> torch.Tensor:
    """The unit's raw bytes as they are packed: each slot's parameter bytes in memory order, zero padded."""
    raw = torch.zeros(u.scratch_bytes, dtype=torch.uint8, device=device)
    for sl in u.slots:
        raw[sl.scratch_offset : sl.scratch_offset + sl.nbytes] = memory_bytes(pdict[sl.name], sl.name).to(device)
    return raw


class StreamDecoded:
    """The converted model: Θ' on the device, one scratch buffer, per-unit decode hooks."""

    def __init__(
        self,
        model: torch.nn.Module,
        enc_key: int,
        layer_re: str = r"layers\.(\d+)\.",
        device: torch.device | None = None,
        module_re: str = r"layers\.(\d+)$",
        resident: str | None = None,
        max_resident_bytes: int | None = None,
    ):
        """layer_re: parameter names matching it belong to the unit of their layer module (the name up to
        the layer index); module_re: module names matching it are the layer modules that get the decode
        hook (their forward pre-hook fires before the layer runs).

        The parameters outside the layers (embeddings, norms, head) are decoded into the same scratch when they are
        needed, at the phase `rest_phases` gives them: before the first layer runs (a pre-hook on the model) or after
        the last (a hook on it). Those read inside the layers stay decoded in a resident buffer (the integer model's
        tables, or names matching `resident`); so do those whose phase cannot be told, at most max_resident_bytes of
        them (default: one layer's scratch), or the conversion is refused. `plain_bytes` is all the plaintext."""
        from ..core.backend import CpuLib

        self.model, self.enc_key, self.layer_re, self.module_re = model, enc_key, layer_re, module_re
        units, params, pay_off = plan_units(model, layer_re, module_re)
        self.device = device or params[0][1].device
        self.cuda = self.device.type == "cuda"
        self._lib = None if self.cuda else CpuLib()
        self.units = units
        self.payload_words = pay_off
        layers = [u for u in units.values() if u.key != REST]
        layer_max = max((u.scratch_bytes for u in layers), default=0)
        # ---- the rest unit keeps its place in Θ' (scratch_offset is its packing); its views go where its phases put
        # them:
        # parameters needed at both phases first, then those of either phase alone over the same bytes, so the scratch
        # holds the larger of the two (an untied embedding and head are never both decoded)
        self.rest_groups: dict[str, list[ParamSlot]] = {BEFORE: [], AFTER: []}
        self.resident_slots: list[ParamSlot] = []
        rest_view = 0
        if REST in units:
            rest = units[REST]
            phases = rest_phases(model, [sl.name for sl in rest.slots], [u.key for u in layers], resident)
            both = [sl for sl in rest.slots if phases[sl.name] == {BEFORE, AFTER}]
            off = 0
            for sl in both:
                sl.view_offset = off
                off += sl.groups * GROUP_RAW_BYTES
            rest_view = off
            for ph in (BEFORE, AFTER):
                off = sum(sl.groups * GROUP_RAW_BYTES for sl in both)
                for sl in rest.slots:
                    if phases[sl.name] == {ph}:
                        sl.view_offset = off
                        off += sl.groups * GROUP_RAW_BYTES
                rest_view = max(rest_view, off)
                self.rest_groups[ph] = [sl for sl in rest.slots if ph in phases[sl.name]]
            off = 0
            for sl in rest.slots:
                if RESIDENT in phases[sl.name]:
                    sl.view_offset = off
                    off += sl.groups * GROUP_RAW_BYTES
                    self.resident_slots.append(sl)

            def said(n):
                return getattr(model, "stream_phases", None) is not None or bool(resident and re.search(resident, n))

            unplaced = [sl for sl in self.resident_slots if not said(sl.name)]
            cap = layer_max if max_resident_bytes is None else max_resident_bytes
            if sum(sl.groups * GROUP_RAW_BYTES for sl in unplaced) > cap:
                raise ValueError(
                    f"parameters outside the layers that no module before or after the layers holds would stay decoded "
                    f"({[sl.name for sl in unplaced][:5]}), over the {cap}-byte bound; give the model a "
                    "stream_phases(name), or pass resident= and max_resident_bytes="
                )
        self.resident = (
            torch.zeros(
                sum(sl.groups * GROUP_RAW_BYTES for sl in self.resident_slots), dtype=torch.uint8, device=self.device
            )
            if self.resident_slots
            else None
        )
        self.scratch = torch.zeros(max(layer_max, rest_view, 1), dtype=torch.uint8, device=self.device)
        self._hooks, self._hooked = [], set()
        # ---- build Θ' unit by unit, freeing each unit's original parameters as soon as it is encoded. On CUDA, Θ' is
        # one
        # reserved address range backed with memory as the encoding reaches it, so the plain weights and Θ' are never
        # resident in full at the same time: the peak is the weights plus about one unit (and one chunk), not twice the
        # model
        if self.cuda:
            self.theta = self._be.reserve(pay_off)
            free0, total = torch.cuda.mem_get_info(self.device)
            self.convert_peak_bytes = total - free0
        else:
            self.theta = torch.empty(pay_off, dtype=torch.int64, device=self.device)
        pdict = dict(params)
        for unit in units.values():
            raw = unit_raw(unit, pdict, self.device)
            if self.cuda:
                self._be.map_reserved(self.theta, (unit.payload_offset + unit.payload_words) * 8)
            self._encode_into(raw, unit.payload_offset, unit.scratch_bytes // GROUP_RAW_BYTES)
            if self.cuda:
                free, total = torch.cuda.mem_get_info(self.device)
                self.convert_peak_bytes = max(self.convert_peak_bytes, total - free)
            del raw
            if unit.key == REST:
                self._decode_slots(self.resident_slots, self.resident)
                kept = {sl.name for sl in self.resident_slots}
                for sl in unit.slots:
                    pdict[sl.name].data = self._view(self.resident if sl.name in kept else self.scratch, sl)
            else:
                for sl in unit.slots:
                    pdict[sl.name].data = self._view(self.scratch, sl)  # frees the original storage
                if unit.module is not None:
                    self._hooks.append(unit.module.register_forward_pre_hook(self._make_hook(unit)))
                    self._hooked.add(unit.key)
            if self.cuda:
                torch.cuda.empty_cache()  # hand the freed weights back to the driver for the next chunks of Θ'
        self.current_unit: str | None = None
        if self.rest_groups[BEFORE]:
            self._hooks.append(model.register_forward_pre_hook(lambda m, args: self.decode_rest(BEFORE)))
        if self.rest_groups[AFTER]:
            order = {p: i for i, (p, _) in enumerate(model.named_modules(remove_duplicate=False))}
            last = max(layers, key=lambda u: order[u.key]).module
            self._hooks.append(last.register_forward_hook(lambda m, args, out: self.decode_rest(AFTER)))
            # vLLM: compute_logits, after the forward, hands lm_head to its
            lp = getattr(model, "logits_processor", None)
            if isinstance(lp, torch.nn.Module):  # LogitsProcessor and never calls it; decoded again only if overwritten
                self._hooks.append(lp.register_forward_pre_hook(lambda m, args: self.decode_rest(AFTER)))

    @property
    def plain_bytes(self) -> int:
        """Every byte of the weights in plaintext on the device: the scratch and the resident buffer."""
        return self.scratch.numel() + (self.resident.numel() if self.resident is not None else 0)

    @staticmethod
    def _view(base: torch.Tensor, sl: ParamSlot) -> torch.Tensor:
        """The parameter over its slot, with its own strides when it was not contiguous."""
        off = sl.scratch_offset if sl.view_offset is None else sl.view_offset
        flat = base[off : off + sl.nbytes].view(sl.dtype)
        return flat.as_strided(sl.shape, sl.stride) if sl.stride else flat.view(sl.shape)

    @property
    def _be(self):
        """The CUDA kernels for this model's device, built once: `DeviceBackend()` runs `load_inline`, which takes
        torch's build-directory file lock (once per layer per forward when the decode hook made one; a kill -9
        while it was held left a stale lock that hung the next start)."""
        from .dynfill import _backend

        return _backend(self.device)

    # ---- encode / decode primitives (device or CPU)
    def _encode_into(self, raw_u8: torch.Tensor, payload_offset: int, groups: int):
        raw64 = raw_u8.view(torch.int64)
        if self.cuda:
            be = self._be
            payload = torch.empty(groups * GROUP_PAYLOAD_WORDS, dtype=torch.int64, device=self.device)
            be.pack(raw64, payload, groups)
            out = self.theta[payload_offset : payload_offset + groups * GROUP_PAYLOAD_WORDS]
            be.setup_at(payload, out, self.enc_key, payload_offset)
            be.sync()
        else:
            raw_np = raw64.numpy().view(np.uint64)
            payload = np.zeros(groups * GROUP_PAYLOAD_WORDS, dtype=np.uint64)
            self._lib.pack(raw_np, payload, groups)
            enc = np.zeros_like(payload)
            self._lib.setup_at(payload, enc, self.enc_key, payload_offset)
            self.theta[payload_offset : payload_offset + len(enc)] = torch.from_numpy(enc.view(np.int64))

    def decode_unit(self, u: DecodeUnit, dst: torch.Tensor):
        """Decode unit u's slice of Θ' into dst (a uint8 buffer of at least u.scratch_bytes)."""
        self._decode_range(u.payload_offset, u.scratch_bytes // GROUP_RAW_BYTES, dst)

    def _decode_range(self, payload_offset: int, groups: int, dst: torch.Tensor):
        """Decode `groups` groups of Θ' from payload_offset into the start of dst (a uint8 buffer)."""
        if groups == 0:
            return
        raw64 = dst[: groups * GROUP_RAW_BYTES].view(torch.int64)
        if self.cuda:
            self._be.decode_unpack(self.theta, raw64, payload_offset, groups, self.enc_key)
        else:
            out = np.empty((GROUP_RAW_BYTES // 8) * groups, dtype=np.uint64)
            self._lib.decode_unpack(self.theta.numpy().view(np.uint64), out, payload_offset, groups, self.enc_key)
            raw64.copy_(torch.from_numpy(out.view(np.int64)))

    def _decode_slots(self, slots: list[ParamSlot], base: torch.Tensor):
        """Decode rest slots into base at their view offsets: one call per run that is contiguous in Θ' and in base."""
        runs: list[list[int]] = []
        for sl in sorted(slots, key=lambda s: s.view_offset):
            r = runs[-1] if runs else None
            if (
                r
                and r[0] + r[2] * GROUP_PAYLOAD_WORDS == sl.payload_offset
                and r[1] + r[2] * GROUP_RAW_BYTES == sl.view_offset
            ):
                r[2] += sl.groups
            else:
                runs.append([sl.payload_offset, sl.view_offset, sl.groups])
        for p, v, g in runs:
            self._decode_range(p, g, base[v:])

    def decode_rest(self, phase: str):
        """Decode the parameters outside the layers that this phase (BEFORE or AFTER the layers) needs into the scratch."""
        key = f"{REST}:{phase}"
        if self.current_unit != key:
            self._decode_slots(self.rest_groups[phase], self.scratch)
            self.current_unit = key

    def _make_hook(self, u: DecodeUnit):
        def hook(module, args):
            if self.current_unit != u.key:
                self.decode_unit(u, self.scratch)
                self.current_unit = u.key

        return hook

    def check_parameters(self, originals: dict[str, torch.Tensor]) -> list[str]:
        """Decode every unit and compare each parameter's values, byte for byte, with the originals. Also
        reports what decoding would never reach: a parameter (of the model, or among the originals) in
        no decode unit, a unit with no hook, a unit's parameter outside the module whose hook decodes it,
        and a parameter that no longer points at its slot. Returns the names (with the reason) that fail."""
        bad = []
        slots = {sl.name: (u, sl) for u in self.units.values() for sl in u.slots}
        for name in list(dict.fromkeys([n for n, _ in self.model.named_parameters()] + list(originals))):
            if name not in slots:
                bad.append(f"{name} (in no decode unit)")
        for unit in self.units.values():
            if unit.key != REST:
                if unit.key not in self._hooked:
                    bad += [f"{sl.name} (unit {unit.key} has no decode hook)" for sl in unit.slots]
                    continue
                path = next((n for n, m in self.model.named_modules() if m is unit.module), None)
                bad += [
                    f"{sl.name} (outside {path}, whose hook decodes unit {unit.key})"
                    for sl in unit.slots
                    if path is None or not sl.name.startswith(path + ".")
                ]
        pdict = dict(self.model.named_parameters())

        def compare(buf, slots):  # right after each decode: the scratch is shared
            for sl in slots:
                got = self._view(buf, sl)
                p = pdict.get(sl.name)
                if p is not None and (p.data_ptr() != got.data_ptr() or tuple(p.stride()) != tuple(got.stride())):
                    bad.append(f"{sl.name} (not a view of its slot)")
                    continue
                if sl.name not in originals:
                    continue
                exp = originals[sl.name].detach().to(got.device)
                if (
                    tuple(exp.shape) != tuple(got.shape)
                    or exp.dtype != got.dtype
                    or not torch.equal(
                        got.contiguous().reshape(-1).view(torch.uint8), exp.contiguous().reshape(-1).view(torch.uint8)
                    )
                ):
                    bad.append(sl.name)

        for unit in self.units.values():
            if unit.key != REST:
                self.decode_unit(unit, self.scratch)
                self.current_unit = unit.key
                compare(self.scratch, unit.slots)
                continue
            if self.resident_slots:
                compare(self.resident, self.resident_slots)
            for ph in (BEFORE, AFTER):
                if self.rest_groups[ph]:
                    self.decode_rest(ph)
                    compare(self.scratch, self.rest_groups[ph])
        return bad

    # ---- what the challenge phase needs
    def segments(self) -> list[tuple[int, int, torch.Tensor]]:
        """(device pointer, words, tensor) of the challenged array: Θ' is one contiguous allocation."""
        return [(self.theta.data_ptr(), self.theta.numel(), self.theta)]

    def image(self, filler=None, out: np.ndarray | None = None) -> np.ndarray:
        """T = H(Θ' ‖ filler) over the resident array, group offsets continuing across segments. out: where to write it
        (an array or a np.memmap of as many uint64 or int64 words); by default a new array. On CUDA it goes 1 GiB of words at
        a time from the device into out, so writing into a memmap never holds the whole image in host memory."""
        parts = [self.theta] + ([filler.tensor[: filler.words]] if filler is not None and filler.words else [])
        n_all = sum(t.numel() for t in parts)
        dest = np.empty(n_all, dtype=np.uint64) if out is None else out.view(np.uint64)
        if dest.shape != (n_all,):
            raise ValueError(f"out holds {dest.shape} words, the image {n_all}")
        base, g0 = 0, 0
        for part in parts:
            if self.cuda:
                be = self._be
                n = part.numel()
                chunk = 1 << 27  # 1 GiB of words per launch
                for lo in range(0, n, chunk):
                    hi = min(n, lo + chunk)
                    oc = torch.empty(hi - lo, dtype=part.dtype, device=part.device)
                    be.image(part[lo:hi], oc, g0 + lo // 4)
                    be.sync()
                    dest[base + lo : base + hi] = oc.cpu().numpy().view(np.uint64)
            else:
                # CPU path: position-bound groups need the global offset; image group by group
                from ..core.reference import image_group

                tn = part.numpy().view(np.uint64)
                for g in range(part.numel() // 4):
                    dest[base + 4 * g : base + 4 * g + 4] = image_group(g0 + g, tn[4 * g : 4 * g + 4].tolist())
            base += part.numel()
            g0 += part.numel() // 4
        return dest

    def theta_leaves(self, block_bytes: int, key: bytes) -> np.ndarray:
        """Keyed BLAKE2b digests of Θ' per block (the last one short), from the device in chunks."""
        from ..setup.commit import leaves_of

        bw = block_bytes // 8
        chunk = max(bw, (1 << 27) // bw * bw)
        parts = [
            leaves_of(self.theta[lo : lo + chunk].cpu().numpy().view(np.uint64), block_bytes, key)
            for lo in range(0, self.theta.numel(), chunk)
        ]
        return np.concatenate(parts) if parts else np.zeros((0, 32), dtype=np.uint8)

    def export_bundle(
        self,
        out_dir,
        filler=None,
        filler_seed: int = 0,
        block_bytes: int = 1 << 20,
        r: int = 2048,
        k: int = 1024,
        hbm_bytes: int | None = None,
        model_name: str = "resident",
        gpu: int = 0,
        *,
        benchmark_only: bool = False,
        benchmark_dynamic_bytes: int | None = None,
    ) -> str:
        """Write what the verifier needs: the one-way image of Θ' and a manifest of kind "bundle". No weight
        key: the GPU entry's key is None (withheld). The image is Θ''s and nothing more: the filler fills the
        GPU's spare memory, so its image would be as large, and the verifier regenerates any filler block from
        its own seed and the key it derives for this GPU (`memory_accounting.setup.filler.static_filler_key(filler_seed,
        gpu)`, which `allocate_filler` encodes under; the manifest's `filler_key` records it). The verifier
        refuses an image of any other length: one that covered the filler too (an export option before) let a
        prover put an image of zeros over the whole array and answer every chain holding nothing.
        The manifest's challenge-hash keys are the ones both sides derive from the filler seed
        (`memory_accounting.core.params.Keys.from_seed`), written for the record; the prover does not choose them.

        The commitment. The verifier holds neither the weights' key nor Θ', so it cannot recompute the
        image as `memory_accounting.setup.verify` does; what the bundle can do is hold the prover to one image and one Θ'
        for as long as the accounting runs:
          * the GPU entry's `commit_root` (with `commit_file`): a Merkle root over keyed digests of Θ''s
            blocks, and `image_root`: the same over the image's blocks (key `commit_key`).
        It says nothing about the weights: `raw_hash` is empty, `layout` is empty and `raw_bytes` and
        `payload_words` are 0. The verifier does not hold the weights at setup, a hash of them would not show
        that Θ' encodes them, and the layout would only tell it the model's architecture. Θ''s size is the
        image's length, which is where the verifier's filler starts.
        The verifier checks its image against `image_root` (`memory_accounting.setup.verify.verify_bundle`): an image
        changed after the bundle was made is caught, and the three roots are small enough to be logged
        when the bundle is accepted. They tie the image to the weights only for a party holding the
        weights' key: an audit with the key disclosed (or, later, a proof) recomputes Θ' from the weights,
        then its image, and must arrive at both roots. Until then the memory claim does not credit Θ''s
        region (`memory_accounting.setup.verify.credited_static_words`). gpu: this GPU's index as the verifier knows it.

        The challenged array is Θ' ‖ filler in whole blocks. `allocate_filler` ends the filler on a block
        boundary, so nothing is left over; without filler, the words of Θ' past its last whole block are not
        challenged (a warning says how many), and a Θ' smaller than one block is refused."""
        from pathlib import Path

        from ..core.params import Keys
        from ..setup.commit import leaves_of, merkle_root
        from ..setup.filler import static_filler_key
        from ..setup.manifest import BENCHMARK, BUNDLE, GpuEntry, Manifest

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fw = filler.words if filler is not None else 0
        if fw and self.theta.numel() != self.payload_words:
            raise RuntimeError("filler positions must start right after Θ' for the verifier to regenerate them")
        fkey = static_filler_key(int(filler_seed), int(gpu)) if fw else None
        if fw:
            if getattr(filler, "key", None) is None or int(filler.key) != fkey:
                raise ValueError(
                    f"the filler is not encoded under the key the verifier derives for GPU {gpu} from the filler seed "
                    "(memory_accounting.setup.filler.static_filler_key; allocate_filler with the same gpu and seed encodes under it)"
                )
            if fkey == int(self.enc_key):
                raise ValueError(
                    "the weights' key is the filler key, which the verifier knows: it could decode Θ' (pick another --enc-key)"
                )
        words = self.theta.numel() + fw
        bw = block_bytes // 8
        if words < bw:
            raise ValueError(
                f"Θ' and the filler are {words * 8} bytes, less than one {block_bytes}-byte block: no chain can cover them "
                "(add filler or use smaller blocks)"
            )
        if words % bw:
            import warnings

            warnings.warn(
                f"the last {words % bw} words of Θ'{' and filler' if fw else ''} are past the last whole {block_bytes}-byte block "
                "and are not challenged",
                stacklevel=2,
            )
        manifest = Manifest(
            model_source=model_name,
            raw_bytes=0,
            raw_hash="",
            layout=[],
            block_bytes=block_bytes,
            payload_words=0,
            filler_seed=filler_seed,
            accounted_words=words - words % bw,
            hbm_bytes=hbm_bytes
            or (torch.cuda.get_device_properties(self.device).total_memory if self.cuda else words * 8),
            commit_key="",
            hash_key=Keys.from_seed(int(filler_seed)).as_list(),
            r=r,
            k=k,
            kind=BUNDLE,
            benchmark_dynamic_bytes=benchmark_dynamic_bytes,
            filler_key=f"{fkey:032x}" if fw else "",
        )
        if benchmark_only:
            manifest.kind = BENCHMARK
            manifest.payload_words = self.theta.numel()
            manifest.gpus = [GpuEntry(int(gpu), None, "", "", "")]
            manifest.save(out_dir / "manifest.json")
            return str(out_dir / "manifest.json")

        image_file = out_dir / f"image_gpu{int(gpu)}.npy"  # written in place: never the whole image in host memory
        img = np.lib.format.open_memmap(image_file, mode="w+", dtype=np.int64, shape=(self.theta.numel(),))
        self.image(out=img)
        img.flush()
        commit_key = secrets.token_bytes(16)
        tleaves = self.theta_leaves(block_bytes, commit_key)
        commit_file = out_dir / f"commit_gpu{int(gpu)}.npy"
        np.save(commit_file, tleaves, allow_pickle=False)
        manifest.commit_key = commit_key.hex()
        manifest.gpus = [
            GpuEntry(
                int(gpu), None, merkle_root(tleaves).hex(), str(commit_file), "", str(image_file),
                image_root=merkle_root(leaves_of(img.view(np.uint64), block_bytes, commit_key)).hex(),
            )
        ]
        manifest.save(out_dir / "manifest.json")
        del img
        return str(out_dir / "manifest.json")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks, self._hooked = [], set()


def convert(model: torch.nn.Module, enc_key: int, **kw) -> StreamDecoded:
    return StreamDecoded(model, enc_key, **kw)
