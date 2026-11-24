"""An integer-exact transformer: the same bytes on a GPU prover and a CPU verifier.

Why: the dynamic check recomputes the prover's KV cache and activations and compares a hash chain over
them, so the recomputation must be bit-exact. A float model is bit-exact only on identical hardware and
kernels (vLLM's batch-invariant mode on the same GPU type). This model is bit-exact across a GPU and a
CPU, so a verifier without a GPU can check the dynamic region today. It is a W8A8 quantisation of a
Llama-architecture checkpoint (Llama, Mistral, Qwen2) where every operation is one of:

  E1  integer arithmetic (int64, no overflow by construction): exact and order-independent;
  E2  float64 matmuls whose operands are integers small enough that every partial sum is an exact
      integer below 2^53 (int8 x int8 over K < 2^39, attention weights < 2^20 x int8): exact in any
      summation order, so cuBLAS, CPU BLAS and any tiling give the same result;
  E3  float64 elementwise operations that the hardware rounds correctly in every torch backend:
      + - * /, round (half to even), floor, abs, max, comparisons, int<->float casts, bit views. One op
      per torch call (no fused or `alpha=` forms, which a GPU may contract into an FMA), eager mode.
      Not sqrt: torch's x86 CPU kernels compute float64 sqrt with a vector math library that is not
      correctly rounded (measured on an AVX-512 host: one ulp off where CUDA and numpy agree), so the
      norm uses an exact integer square root (`isqrt`);
  E4  division only tensor / tensor on the same device: CUDA turns `tensor / python_scalar` into a
      multiply by the reciprocal, which rounds differently; CPU does not (see `fdiv`);
  E5  no float reductions except max (sums of floats depend on the order); norms and softmax sums are
      taken over integers;
  E6  no transcendental functions at run time (exp, sigmoid, cos, sin are not correctly rounded and
      differ between libm and CUDA): lookup tables built once at conversion and shipped in the model
      file, indexed by integers.

Under these rules every row of every tensor is computed the same way whatever the batch, the chunking
of the prompt or the device, so the model is also batch- and chunk-invariant for free.

Quantisation: weights int8 per output channel (absmax); activations int8 per token, dynamic (absmax);
K and V cached as int8 per (token, head) with a float64 scale; softmax via an exp table on a 1/256 grid;
SiLU via a sigmoid table on a 1/256 grid. The residual stream is float64. Quality is that of a W8A8
model with coarse softmax: fine as a proof of concept, not a production quantisation.

  python -m memory_accounting.models.intllm convert --model HuggingFaceTB/SmolLM2-135M --out models/smollm2-135m.int.safetensors
  python -m memory_accounting.models.intllm generate --model models/smollm2-135m.int.safetensors --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

FORMAT = "posm-intllm-1"
EXP_RES, EXP_RANGE = 256, 32  # exp(-i / 256) for i in [0, 32 * 256]; the last entry is 0
EXP_BITS = 30  # exp table values are integers scaled by 2^30
SIG_RES, SIG_RANGE = 256, 16  # sigmoid(i / 256) for i in [-16 * 256, 16 * 256]
P_SHIFT = 10  # attention weights e * (v_scale / max v_scale) / 2^10 -> integers < 2^20
INV127 = 1.0 / 127.0
F64 = torch.float64


# ---- exact-arithmetic helpers (the rules above)


def fdiv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IEEE true division, both operands tensors on the same device (rule E4)."""
    if not isinstance(b, torch.Tensor) or b.device != a.device:
        raise TypeError(
            "fdiv: the divisor must be a tensor on the dividend's device (a python scalar divisor rounds differently on CUDA)"
        )
    return a / b


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row int8 quantisation over the last dim: (q as float64 integers in [-127, 127], scale) with
    x ~= q * scale. Rows of zeros quantise to zeros with scale 0."""
    a = x.abs().amax(dim=-1, keepdim=True)
    a_safe = torch.where(a == 0, torch.ones_like(a), a)
    q = torch.round(fdiv(x, a_safe) * 127.0)
    return q, (a * INV127).squeeze(-1)


def lin(
    q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    """y = (q @ w^T) * s * ws (+ b): an exact integer matmul (E2), then two correctly rounded multiplies."""
    acc = q @ w.to(F64).t()
    y = acc * s.unsqueeze(-1)
    y = y * ws
    if b is not None:
        y = y + b
    return y


def pow2(k: torch.Tensor) -> torch.Tensor:
    """2^k as float64 for integer k in [-1022, 1023], built from the exponent bits (exact everywhere)."""
    return ((k.to(torch.int64) + 1023) << 52).view(F64)


def exponent(x: torch.Tensor) -> torch.Tensor:
    """e with 2^(e-1) <= x < 2^e for positive normal float64 x (frexp's exponent), from the bits."""
    return ((x.view(torch.int64) >> 52) & 0x7FF) - 1022


def isqrt(n: torch.Tensor) -> torch.Tensor:
    """floor(sqrt(n)) for int64 0 <= n < 2^62, exact: a float estimate corrected by integer comparisons
    (any sqrt implementation is close enough for the estimate)."""
    r = torch.sqrt(n.to(F64)).to(torch.int64)
    for _ in range(3):
        r = torch.where(r * r > n, r - 1, r)
    for _ in range(3):
        r = torch.where((r + 1) * (r + 1) <= n, r + 1, r)
    return r


def rms_quant(h: torch.Tensor, g: torch.Tensor, eps: float, inv_h: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(h) * g, quantised per token: returns (q, scale). Exact on every device (E1, E3, E5):

    hi = round(h / amax * 2^24) (so the largest |hi| is 2^24), ss = sum hi^2 exactly in int64;
    rms = amax / 2^24 * sqrt(ss / H + eps * 2^48 / amax^2), and the square root is an integer one:
    N = floor((ss / H + eps term) * 2^E) with E even and chosen so that N < 2^61, R = isqrt(N),
    rms = amax * R * 2^-(24 + E/2)."""
    H = h.shape[-1]
    a = h.abs().amax(dim=-1, keepdim=True)
    a_safe = torch.where(a == 0, torch.ones_like(a), a)
    hi = torch.round(fdiv(h, a_safe) * 16777216.0).to(torch.int64)  # |hi| <= 2^24
    ss = (hi * hi).sum(dim=-1)  # in [2^48, 2^48 * H] for a nonzero row, exact
    a1 = a_safe.squeeze(-1)
    aa = a1 * a1
    eps_rel = fdiv(torch.full_like(a1, eps), aa) * 281474976710656.0  # eps * 2^48 / amax^2 (correctly rounded ops)
    m_est = ss.to(F64) * (1.0 / H)
    m_est = m_est + eps_rel  # ~ the real M, only used to pick E
    E = (60 - exponent(m_est)).clamp(0, 60)
    E = E - (E % 2)  # even: sqrt(2^E) is a power of two
    q_, r_ = ss // H, ss % H
    base = (q_ << E) + ((r_ << E) // H)  # floor(ss * 2^E / H); ss >= 2^48 keeps E <= 26
    # floor(eps term * 2^E), clamped for vanishing rows
    e_i = torch.clamp(eps_rel * pow2(E), max=2.0**61).to(torch.int64)
    R = isqrt(base + e_i)
    rms = R.to(F64) * a1
    rms = rms * pow2(-(24 + E // 2))
    q, s = quant_rows(h * g)
    return q, fdiv(s, rms)


class IntLayer(nn.Module):
    """One decoder layer. Called as layer(h, pos0): h [T, H] float64, the hidden state entering the layer
    (the first positional argument, which is what the dynamic-memory hooks record)."""

    def __init__(self, cfg: dict, max_len: int):
        super().__init__()
        H, nh, nkv, hd, inter = cfg["hidden"], cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["intermediate"]
        self.cfg, self.max_len = cfg, max_len

        def P(*shape, dtype=torch.int8):
            return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)

        self.in_norm = P(H, dtype=F64)
        self.post_norm = P(H, dtype=F64)
        self.wq, self.wk, self.wv, self.wo = P(nh * hd, H), P(nkv * hd, H), P(nkv * hd, H), P(H, nh * hd)
        self.sq, self.sk, self.sv, self.so = (
            P(nh * hd, dtype=F64),
            P(nkv * hd, dtype=F64),
            P(nkv * hd, dtype=F64),
            P(H, dtype=F64),
        )
        if cfg["qkv_bias"]:
            self.bq, self.bk, self.bv = P(nh * hd, dtype=F64), P(nkv * hd, dtype=F64), P(nkv * hd, dtype=F64)
        else:
            self.bq = self.bk = self.bv = None
        self.wg, self.wu, self.wd = P(inter, H), P(inter, H), P(H, inter)
        self.sg, self.su, self.sd = P(inter, dtype=F64), P(inter, dtype=F64), P(H, dtype=F64)
        # static KV cache: int8 values, float64 scale per (slot, kv head); reset() zeroes it
        self.register_buffer("k_cache", torch.zeros(max_len, nkv, hd, dtype=torch.int8))
        self.register_buffer("k_scale", torch.zeros(max_len, nkv, dtype=F64))
        self.register_buffer("v_cache", torch.zeros(max_len, nkv, hd, dtype=torch.int8))
        self.register_buffer("v_scale", torch.zeros(max_len, nkv, dtype=F64))
        self.tables = None  # set by IntLLM: shared rope / exp / sigmoid tables

    def forward(self, h: torch.Tensor, pos0: int) -> torch.Tensor:
        cfg, tables = self.cfg, self.tables
        T = h.shape[0]
        nh, nkv, hd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
        grp = nh // nkv
        S = pos0 + T
        # ---- attention
        xq, xs = rms_quant(h, self.in_norm, cfg["eps"], tables.inv_h)
        q = lin(xq, xs, self.wq, self.sq, self.bq).reshape(T, nh, hd)
        k = lin(xq, xs, self.wk, self.sk, self.bk).reshape(T, nkv, hd)
        v = lin(xq, xs, self.wv, self.sv, self.bv).reshape(T, nkv, hd)
        q, k = tables.rope(q, pos0), tables.rope(k, pos0)
        kq, ks = quant_rows(k)
        vq, vs = quant_rows(v)
        self.k_cache[pos0:S] = kq.to(torch.int8)
        self.k_scale[pos0:S] = ks
        self.v_cache[pos0:S] = vq.to(torch.int8)
        self.v_scale[pos0:S] = vs
        # everything below reads K and V back from the cache, so a token's own K/V is the cached one
        Kh = self.k_cache[:S].to(F64).permute(1, 0, 2).repeat_interleave(grp, dim=0)  # [nh, S, hd]
        Vh = self.v_cache[:S].to(F64).permute(1, 0, 2).repeat_interleave(grp, dim=0)
        Ks = self.k_scale[:S].t().repeat_interleave(grp, dim=0)  # [nh, S]
        Vs = self.v_scale[:S].t().repeat_interleave(grp, dim=0)
        qq, qs = quant_rows(q)  # [T, nh, hd], [T, nh]
        sc = torch.matmul(qq.permute(1, 0, 2), Kh.transpose(1, 2))  # [nh, T, S] exact integers
        sc = sc * qs.t().unsqueeze(-1)
        sc = sc * Ks.unsqueeze(1)
        sc = sc * tables.att_scale  # a multiply by a constant (E3)
        valid = tables.causal(T, S, pos0, h.device)  # [T, S]: key j <= query position
        m = torch.where(valid, sc, torch.full_like(sc, -torch.inf)).amax(dim=-1, keepdim=True)
        idx = torch.round((m - sc) * float(EXP_RES))  # >= 0 for valid keys
        idx = (
            torch.where(valid, idx, torch.full_like(idx, float(tables.exp_n - 1)))
            .clamp(0, tables.exp_n - 1)
            .to(torch.int64)
        )
        e = tables.exp_table[idx]  # int64, 0 for masked keys
        esum = e.sum(dim=-1)  # exact
        # per row: max V scale over its keys
        vmax = torch.where(valid, Vs.unsqueeze(1).expand(nh, T, S), torch.zeros_like(sc)).amax(dim=-1)
        vmax_safe = torch.where(vmax == 0, torch.ones_like(vmax), vmax)
        r = fdiv(Vs.unsqueeze(1).expand(nh, T, S), vmax_safe.unsqueeze(-1))  # <= 1
        pw = e.to(F64) * r
        pw = torch.round(pw * tables.p_scale)  # integers < 2^20
        o = torch.matmul(pw, Vh)  # [nh, T, hd] exact integers
        o = o * vmax.unsqueeze(-1)
        o = o * tables.p_unscale
        o = fdiv(o, esum.to(F64).unsqueeze(-1))
        o = o.permute(1, 0, 2).reshape(T, nh * hd)
        oq, os_ = quant_rows(o)
        h = h + lin(oq, os_, self.wo, self.so)
        # ---- MLP (SiLU-gated)
        xq, xs = rms_quant(h, self.post_norm, cfg["eps"], tables.inv_h)
        gate = lin(xq, xs, self.wg, self.sg)
        up = lin(xq, xs, self.wu, self.su)
        sidx = (torch.round(gate * float(SIG_RES)) + float(tables.sig_off)).clamp(0, tables.sig_n - 1).to(torch.int64)
        a = gate * tables.sig_table[sidx]
        a = a * up
        aq, as_ = quant_rows(a)
        return h + lin(aq, as_, self.wd, self.sd)


class Tables(nn.Module):
    """Lookup tables and constants shared by all layers (built at conversion, shipped in the file)."""

    def __init__(self, cfg: dict, max_pos: int):
        super().__init__()
        hd = cfg["head_dim"]

        def P(*shape, dtype=F64):
            return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)

        self.cos, self.sin = P(max_pos, hd // 2), P(max_pos, hd // 2)
        self.exp_table = P(EXP_RES * EXP_RANGE + 1, dtype=torch.int64)
        self.sig_table = P(2 * SIG_RES * SIG_RANGE + 1)
        self.exp_n = EXP_RES * EXP_RANGE + 1
        self.sig_n, self.sig_off = 2 * SIG_RES * SIG_RANGE + 1, SIG_RES * SIG_RANGE
        self.inv_h = 2.0**-48 / cfg["hidden"]
        self.att_scale = float(cfg["att_scale"])
        self.p_scale, self.p_unscale = 2.0**-P_SHIFT, 2.0**P_SHIFT
        self._causal: dict = {}

    def rope(self, x: torch.Tensor, pos0: int) -> torch.Tensor:
        """HF rotate_half convention: (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin)."""
        T, half = x.shape[0], x.shape[-1] // 2
        c = self.cos[pos0 : pos0 + T].unsqueeze(1)
        s = self.sin[pos0 : pos0 + T].unsqueeze(1)
        x1, x2 = x[..., :half], x[..., half:]
        a = x1 * c
        b = x2 * s
        y1 = a - b
        a = x2 * c
        b = x1 * s
        y2 = a + b
        return torch.cat([y1, y2], dim=-1)

    def causal(self, T: int, S: int, pos0: int, device) -> torch.Tensor:
        qpos = torch.arange(pos0, pos0 + T, device=device).unsqueeze(1)
        return torch.arange(S, device=device).unsqueeze(0) <= qpos


class IntLLM(nn.Module):
    """The model: embedding (int8, per-row scale), layers, final norm, head (tied or separate int8).
    Feed token chunks in order after reset(); forward(ids [1, T]) returns the logits of the last token."""

    def __init__(self, cfg: dict, max_len: int = 512):
        super().__init__()
        self.cfg, self.max_len = cfg, max_len
        H, V = cfg["hidden"], cfg["vocab"]
        if H * 127 * 127 >= 1 << 53 or cfg["intermediate"] * 127 * 127 >= 1 << 53 or H > 1 << 14:
            raise ValueError("shapes outside the exactness bounds")
        if max_len > cfg["max_pos"]:
            raise ValueError(f"max_len {max_len} beyond the rope table ({cfg['max_pos']} positions)")

        def P(*shape, dtype=torch.int8):
            return nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)

        self.embed, self.embed_s = P(V, H), P(V, dtype=F64)
        self.final_norm = P(H, dtype=F64)
        if not cfg["tied"]:
            self.head, self.head_s = P(V, H), P(V, dtype=F64)
        self.tables = Tables(cfg, cfg["max_pos"])
        self.layers = nn.ModuleList([IntLayer(cfg, max_len) for _ in range(cfg["layers"])])
        for layer in self.layers:
            object.__setattr__(layer, "tables", self.tables)  # shared, not a submodule of each layer
        self.pos = 0
        self.info: dict = {"kind": "int"}

    @property
    def d(self) -> int:
        return self.cfg["hidden"]

    def stream_phases(self, name: str) -> tuple[str, ...]:
        """When stream-decode (`memory_accounting.prover.stream_decode.rest_phases`) must have a parameter outside the layers decoded:
        forward reads the embedding before the layers, and the final norm and the head after them (a tied head is the
        embedding, so it is needed at both); the tables are read inside every layer and stay decoded."""
        if name.startswith("tables."):
            return ("resident",)
        if name in ("embed", "embed_s"):
            return ("before", "after") if self.cfg["tied"] else ("before",)
        if name in ("final_norm", "head", "head_s"):
            return ("after",)
        raise ValueError(f"{name}: no stream-decode phase for this parameter")

    @classmethod
    def from_file(cls, path, max_len: int = 512, device="cpu") -> IntLLM:
        from safetensors import safe_open
        from safetensors.numpy import load_file

        path = Path(path)
        with safe_open(str(path), "np") as f:
            meta = f.metadata()
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        cfg = json.loads(meta["config"])
        model = cls(cfg, max_len)
        tensors = load_file(str(path))
        sd = model.state_dict()
        missing = [k for k in sd if k not in tensors and "cache" not in k and "_scale" not in k.split(".")[-1]]
        if missing:
            raise ValueError(f"{path}: missing tensors {missing[:5]}")
        with torch.no_grad():
            for k, v in tensors.items():
                sd[k].copy_(torch.from_numpy(v))
        model.info = {
            "kind": "int",
            "file": path.name,
            "sha256": file_sha256(path),
            "max_len": max_len,
            "tokenizer": cfg.get("source", ""),
        }
        return model.to(device).eval()

    def reset(self) -> None:
        for layer in self.layers:
            layer.k_cache.zero_()
            layer.k_scale.zero_()
            layer.v_cache.zero_()
            layer.v_scale.zero_()
        self.pos = 0

    def share_cache(self) -> None:
        """Every layer uses layer 0's cache buffers (the others are freed): the plaintext scratch of a prover that holds
        each layer's cache encoded and decodes it there before the layer runs (`memory_accounting.prover.kvenc.EncodedStatic`)."""
        l0 = self.layers[0]
        for layer in self.layers[1:]:
            layer.k_cache, layer.k_scale, layer.v_cache, layer.v_scale = l0.k_cache, l0.k_scale, l0.v_cache, l0.v_scale

    def resize(self, max_len: int) -> None:
        """New, empty KV cache buffers of max_len slots; the weights stay. A verifier replays records of any cache
        length the file allows with one model instead of one per length."""
        if not 1 <= int(max_len) <= self.cfg["max_pos"]:
            raise ValueError(f"max_len {max_len} outside 1 to {self.cfg['max_pos']} (the rope table)")
        max_len = int(max_len)
        nkv, hd = self.cfg["kv_heads"], self.cfg["head_dim"]
        for layer in self.layers:
            dev = layer.k_cache.device
            layer.k_cache = layer.v_cache = None  # drop the old buffers before making the new ones
            layer.k_scale = layer.v_scale = None
            layer.k_cache = torch.zeros(max_len, nkv, hd, dtype=torch.int8, device=dev)
            layer.v_cache = torch.zeros(max_len, nkv, hd, dtype=torch.int8, device=dev)
            layer.k_scale = torch.zeros(max_len, nkv, dtype=F64, device=dev)
            layer.v_scale = torch.zeros(max_len, nkv, dtype=F64, device=dev)
            layer.max_len = max_len
        self.max_len, self.pos = max_len, 0
        if "max_len" in self.info:
            self.info["max_len"] = max_len

    @torch.no_grad()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.reshape(-1).to(self.embed.device)
        T = ids.shape[0]
        if self.pos + T > self.max_len:
            raise ValueError(f"KV cache full ({self.max_len} slots)")
        h = self.embed[ids].to(F64) * self.embed_s[ids].unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, self.pos)
        self.pos += T
        xq, xs = rms_quant(h[-1:], self.final_norm, self.cfg["eps"], self.tables.inv_h)
        w, ws = (self.embed, self.embed_s) if self.cfg["tied"] else (self.head, self.head_s)
        return lin(xq, xs, w, ws)[0]

    def kv_buffers(self) -> list[tuple[str, torch.Tensor]]:
        out = []
        for i, layer in enumerate(self.layers):
            out += [
                (f"kv.{i}.k", layer.k_cache),
                (f"kv.{i}.ks", layer.k_scale),
                (f"kv.{i}.v", layer.v_cache),
                (f"kv.{i}.vs", layer.v_scale),
            ]
        return out

    def live_kv_buffers(self, n: int) -> list[tuple[str, torch.Tensor]]:
        """The first n slots of every cache buffer (contiguous views: the live part of the cache)."""
        out = []
        for i, layer in enumerate(self.layers):
            out += [
                (f"kv.{i}.k", layer.k_cache[:n]),
                (f"kv.{i}.ks", layer.k_scale[:n]),
                (f"kv.{i}.v", layer.v_cache[:n]),
                (f"kv.{i}.vs", layer.v_scale[:n]),
            ]
        return out

    @torch.no_grad()
    def generate(self, prompt: list[int], new_tokens: int, reset: bool = True, on_step=None, stop_ids=()) -> list[int]:
        """Greedy decoding (argmax ties go to the lowest id on both CPU and CUDA)."""
        if reset:
            self.reset()
        out = list(prompt)
        logits = self(torch.tensor([prompt]))
        for step in range(new_tokens):
            nxt = int(torch.argmax(logits))
            out.append(nxt)
            if on_step is not None:
                on_step(step)
            if nxt in stop_ids or self.pos >= self.max_len:
                break
            logits = self(torch.tensor([[nxt]]))
        return out


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def int_feed(model: IntLLM):
    """The verifier's replay driver: chunk s of the stage, in order, after the replay has reset the cache
    and put the filler back."""

    def feed(step: int, chunk: list) -> None:
        model(torch.tensor(chunk))

    return feed


def exactness_selftest(device: str) -> list[str]:
    """Every float64 operation the model uses, on `device` and on this host's CPU, against numpy's IEEE
    results: the operations that differ (an empty list means the rule set holds here)."""
    rng = np.random.default_rng(12345)
    n = 1 << 20
    x = rng.standard_normal(n) * 10.0 ** rng.integers(-8, 9, n)
    y = rng.standard_normal(n) * 10.0 ** rng.integers(-8, 9, n)
    y[y == 0] = 1.0
    ints = rng.integers(-(1 << 50), 1 << 50, n)
    ref = {
        "mul": x * y,
        "add": x + y,
        "sub": x - y,
        "div": x / y,
        "round": np.rint(x * 1000.0),
        "abs": np.abs(x),
        "int->f64": ints.astype(np.float64),
        "f64->int": np.trunc(x * 1e6).astype(np.int64).astype(np.float64),
    }
    bad = []
    for dev in (device, "cpu"):
        X, Y, ints_dev = (torch.from_numpy(v).to(dev) for v in (x, y, ints))
        got = {
            "mul": X * Y,
            "add": X + Y,
            "sub": X - Y,
            "div": fdiv(X, Y),
            "round": torch.round(X * 1000.0),
            "abs": X.abs(),
            "int->f64": ints_dev.to(F64),
            "f64->int": (X * 1e6).to(torch.int64).to(F64),
        }
        for k, v in got.items():
            if not np.array_equal(v.cpu().numpy().view(np.uint64), ref[k].view(np.uint64)):
                bad.append(f"{k} on {dev}")
        N = torch.from_numpy(rng.integers(0, 1 << 62, 1 << 16)).to(dev)
        r = isqrt(N).cpu().numpy()
        import math

        if any(int(a) != math.isqrt(int(b)) for a, b in zip(r[:4096], N.cpu().numpy()[:4096])):
            bad.append(f"isqrt on {dev}")
    return bad


def crosscheck(path, device="cuda", tokens: int = 24, max_len: int = 128) -> tuple[bool, list[str]]:
    """The property the CPU verifier relies on, measured: generate on `device`, then feed the same tokens
    on the CPU one at a time (other chunking, other hardware) and compare every KV byte, the logits and
    a mid-forward activation. Returns (all equal, report lines)."""
    dev_model = IntLLM.from_file(path, max_len, device)
    cpu_model = IntLLM.from_file(path, max_len, "cpu")
    prompt = [int(x) for x in (torch.arange(7) * 37 + 11) % dev_model.cfg["vocab"]]
    out = dev_model.generate(prompt, tokens)
    fed = out[: dev_model.pos]
    acts = {}
    L = len(dev_model.layers) // 2

    def grab(tag):
        def h(m, args):
            acts[tag] = args[0][-1].detach().to("cpu").clone()

        return h

    ha = dev_model.layers[L].register_forward_pre_hook(grab("a"))
    hb = cpu_model.layers[L].register_forward_pre_hook(grab("b"))
    la = dev_model(torch.tensor([[fed[-1]]])).to("cpu")  # one more step on each side, at the same position
    cpu_model.reset()
    for t in fed:
        cpu_model(torch.tensor([[t]]))
    lb = cpu_model(torch.tensor([[fed[-1]]]))
    ha.remove()
    hb.remove()
    lines, ok = [], True
    ops = exactness_selftest(device)
    ok &= not ops
    lines.append(
        "float64 ops against IEEE (numpy): "
        + ("all exact on " + device + " and cpu" if not ops else "NOT EXACT: " + ", ".join(ops))
    )
    bad = []
    for (name, ta), (_, tb) in zip(dev_model.kv_buffers(), cpu_model.kv_buffers()):
        if not torch.equal(ta.to("cpu").contiguous().view(torch.uint8), tb.contiguous().view(torch.uint8)):
            bad.append(name)
    ok &= not bad
    lines.append(
        f"KV cache after {len(fed) + 1} tokens: {'identical bytes' if not bad else 'DIFFERS in ' + ', '.join(bad[:6])} ({device} generate vs cpu token by token)"
    )
    same_logits = torch.equal(la.view(torch.int64), lb.view(torch.int64))
    same_act = torch.equal(acts["a"].view(torch.int64), acts["b"].view(torch.int64))
    ok &= same_logits and same_act
    lines.append(
        f"logits {'identical' if same_logits else 'DIFFER'}; layer {L} input {'identical' if same_act else 'DIFFERS'}"
    )
    return ok, lines


# ---- conversion (prover side, once; the file is the declared model)


def _quant_w(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.abs(w).max(axis=1) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s.astype(np.float64)


def convert_hf(src: str, out: str, max_pos: int = 4096) -> dict:
    """Quantise a Llama-architecture HF checkpoint into the integer-exact format."""
    from safetensors.numpy import save_file
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).eval()
    hf_cfg = model.config
    if hf_cfg.model_type not in ("llama", "mistral", "qwen2"):
        raise ValueError(f"model_type {hf_cfg.model_type} not supported (llama, mistral, qwen2)")
    hd = getattr(hf_cfg, "head_dim", None) or hf_cfg.hidden_size // hf_cfg.num_attention_heads
    rot = model.model.rotary_emb
    inv_freq = rot.inv_freq.detach().double().numpy()
    if inv_freq.shape[0] != hd // 2:
        raise ValueError("partial rotary embeddings are not supported")
    att_scaling = float(getattr(rot, "attention_scaling", 1.0))
    sd = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    qkv_bias = "model.layers.0.self_attn.q_proj.bias" in sd
    tied = bool(getattr(hf_cfg, "tie_word_embeddings", False)) or "lm_head.weight" not in sd
    cfg = {
        "source": src,
        "model_type": hf_cfg.model_type,
        "vocab": hf_cfg.vocab_size,
        "hidden": hf_cfg.hidden_size,
        "layers": hf_cfg.num_hidden_layers,
        "heads": hf_cfg.num_attention_heads,
        "kv_heads": hf_cfg.num_key_value_heads,
        "head_dim": hd,
        "intermediate": hf_cfg.intermediate_size,
        "eps": float(hf_cfg.rms_norm_eps),
        "max_pos": max_pos,
        "qkv_bias": qkv_bias,
        "tied": tied,
        "att_scale": 1.0 / float(np.sqrt(hd)),
        "exp": [EXP_RES, EXP_RANGE, EXP_BITS],
        "sig": [SIG_RES, SIG_RANGE],
        "p_shift": P_SHIFT,
    }
    tensors: dict[str, np.ndarray] = {}
    tensors["embed"], tensors["embed_s"] = _quant_w(sd["model.embed_tokens.weight"])
    tensors["final_norm"] = sd["model.norm.weight"]
    if not tied:
        tensors["head"], tensors["head_s"] = _quant_w(sd["lm_head.weight"])
    ang = np.outer(np.arange(max_pos, dtype=np.float64), inv_freq)
    tensors["tables.cos"], tensors["tables.sin"] = np.cos(ang) * att_scaling, np.sin(ang) * att_scaling
    ex = np.rint(np.exp(-np.arange(EXP_RES * EXP_RANGE + 1, dtype=np.float64) / EXP_RES) * (1 << EXP_BITS)).astype(
        np.int64
    )
    ex[-1] = 0  # masked keys and the far tail contribute nothing
    tensors["tables.exp_table"] = ex
    tensors["tables.sig_table"] = 1.0 / (
        1.0 + np.exp(-np.arange(-SIG_RES * SIG_RANGE, SIG_RES * SIG_RANGE + 1, dtype=np.float64) / SIG_RES)
    )
    for i in range(hf_cfg.num_hidden_layers):
        prefix = f"model.layers.{i}."
        L = f"layers.{i}."
        tensors[L + "in_norm"] = sd[prefix + "input_layernorm.weight"]
        tensors[L + "post_norm"] = sd[prefix + "post_attention_layernorm.weight"]
        for ours, theirs in (
            ("q", "self_attn.q_proj"),
            ("k", "self_attn.k_proj"),
            ("v", "self_attn.v_proj"),
            ("o", "self_attn.o_proj"),
            ("g", "mlp.gate_proj"),
            ("u", "mlp.up_proj"),
            ("d", "mlp.down_proj"),
        ):
            tensors[L + "w" + ours], tensors[L + "s" + ours] = _quant_w(sd[prefix + theirs + ".weight"])
            if ours in "qkv" and qkv_bias:
                tensors[L + "b" + ours] = sd[prefix + theirs + ".bias"]
    tensors = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out, metadata={"format": FORMAT, "config": json.dumps(cfg)})
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="integer-exact transformer: convert a HF checkpoint, or generate with one")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-pos", type=int, default=4096)
    x = sub.add_parser("crosscheck", help="same tokens on a device and on the CPU: are the KV bytes identical?")
    x.add_argument("--model", required=True)
    x.add_argument("--device", default="cuda")
    x.add_argument("--tokens", type=int, default=32)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt", default="The capital of France is")
    g.add_argument("--tokens", type=int, default=24)
    g.add_argument("--device", default="cpu")
    g.add_argument("--max-len", type=int, default=512)
    g.add_argument(
        "--compare-hf", action="store_true", help="also generate with the float HF model the file was converted from"
    )
    args = ap.parse_args(argv)
    if args.cmd == "convert":
        cfg = convert_hf(args.model, args.out, args.max_pos)
        print(
            f"wrote {args.out}: {cfg['layers']} layers, hidden {cfg['hidden']}, {cfg['heads']}/{cfg['kv_heads']} heads, vocab {cfg['vocab']}, "
            f"sha256 {file_sha256(args.out)[:16]}…"
        )
        return
    if args.cmd == "crosscheck":
        ok, lines = crosscheck(args.model, args.device, args.tokens)
        print("\n".join(lines))
        print("crosscheck", "passed" if ok else "FAILED")
        raise SystemExit(0 if ok else 1)
    from transformers import AutoTokenizer

    m = IntLLM.from_file(args.model, args.max_len, args.device)
    tok = AutoTokenizer.from_pretrained(m.cfg["source"])
    ids = tok(args.prompt)["input_ids"]
    out = m.generate(ids, args.tokens, stop_ids=(tok.eos_token_id,) if tok.eos_token_id is not None else ())
    print("integer model:", repr(tok.decode(out)))
    if args.compare_hf:
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(m.cfg["source"], dtype=torch.float32).eval()
        ref = hf.generate(torch.tensor([ids]), max_new_tokens=args.tokens, do_sample=False)
        print("float HF model:", repr(tok.decode(ref[0])))


if __name__ == "__main__":
    main()
