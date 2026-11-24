"""Print the digest every implementation produces for a few shapes, to localise a mismatch.
Run on the CUDA host:  uv run python tests/diag_paths.py"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from memory_accounting.core import reference as R
from memory_accounting.core.backend import CpuLib, DeviceBackend, pick_device, to_u64_tensor
from memory_accounting.core.params import MODE_HONEST, Challenge, Keys, MemoryConfig, pack_params

lib = CpuLib()
rng = np.random.default_rng(3)
words = 1 << 14
cfg = MemoryConfig(words=words, encoding="none", perm=False)  # plain data: isolates the hash from the encoding
raw_np = rng.integers(0, 1 << 60, words, dtype=np.int64).view(np.uint64)
keys = Keys.deterministic(5)
dev = pick_device()
be = DeviceBackend(dev) if dev == "cuda" else None
raw = to_u64_tensor(raw_np, dev) if be else None


def hx(o):
    return f"{int(o[0]):016x}{int(o[1]):016x}"


for k, r, B in ((1024, 0, 32768), (128, 0, 2048), (1024, 16, 32768), (128, 16, 2048)):
    ch = Challenge(block_bytes=B, n=3, r=r, nonce=987654321, j0=1, k=k)
    prm = pack_params(cfg, ch, keys, MODE_HONEST)
    rows = {}
    if r == 0 or B <= 2048:
        lo, hi, _ = R.chain(raw_np.tolist(), prm.view(np.uint64).tolist())
        rows["python"] = f"{lo:016x}{hi:016x}"
    out = np.zeros(2 + ch.n, dtype=np.uint64)
    lib.chain(raw_np, prm, out)
    rows["c serial"] = hx(out)
    lib.pool_start(4)
    out = np.zeros(2 + ch.n, dtype=np.uint64)
    lib.chain(raw_np, prm, out, mt=True)
    lib.pool_stop()
    rows["c pool4"] = hx(out)
    if be:
        prm_t = to_u64_tensor(prm, dev)
        state = torch.zeros(8, dtype=torch.int64, device=dev)
        pw = be.power_buffer(ch.pw_count)
        mbuf = be.matrix_buffer(ch.mbuf_bytes)
        ablk = be.block_buffer(ch.block_bytes)
        for T in (256, 1024):
            result = torch.zeros(2 + ch.n, dtype=torch.int64, device=dev)
            be.chain(raw, raw, result, prm_t, state, pw, mbuf, k, r, T)
            be.sync()
            rows[f"gpu persistent T{T}"] = hx(result.cpu().numpy().view(np.uint64))
        for G in (1, 8):
            result = torch.zeros(2 + ch.n, dtype=torch.int64, device=dev)
            parts = torch.zeros(G, dtype=torch.int64, device=dev)
            be.chain_multi(
                raw, raw, result, prm_t, state, parts, pw, mbuf, G, 256, ch.n, k, r, ch.be, tensor=False, graph=False
            )
            be.sync()
            rows[f"gpu multi G{G}"] = hx(result.cpu().numpy().view(np.uint64))
            if r:
                result.zero_()
                be.chain_multi(
                    raw,
                    raw,
                    result,
                    prm_t,
                    state,
                    parts,
                    pw,
                    mbuf,
                    G,
                    256,
                    ch.n,
                    k,
                    r,
                    ch.be,
                    tensor=True,
                    graph=False,
                    ablk=ablk,
                )
                be.sync()
                rows[f"gpu tensor G{G}"] = hx(result.cpu().numpy().view(np.uint64))
        # power table as the GPU filled it vs python
        pwh = pw.cpu().numpy().view(np.uint64)
        p = dict(zip(__import__("memory_accounting.core.params", fromlist=["P_FIELDS"]).P_FIELDS, prm.view(np.uint64).tolist()))
        kn = R.chain_polyk(p["k0"], p["k1"], p["nonce"], p["polyk"])
        t1 = r if r else 64  # x = 0: link seed = nonce, so K_s = K_n
        t2 = len(pwh) - t1 - ch.g  # [T1 | T2 | nonce fold K_n^(g+1)]
        exp_pw = [
            pow(kn, i + 1, R.PM61)
            if i < t1
            else pow(kn, (i - t1) * (r if r else 64), R.PM61)
            if i < t1 + t2
            else pow(kn, i - t1 - t2 + 1, R.PM61)
            for i in range(len(pwh))
        ]
        rows["power table"] = (
            "ok"
            if pwh.tolist() == exp_pw
            else f"MISMATCH at {next(i for i, (a, b) in enumerate(zip(pwh.tolist(), exp_pw)) if a != b)}"
        )
    print(f"\n== k={k} r={r} B={B}")
    ref = rows.get("python") or rows["c serial"]
    for name, v in rows.items():
        print(f"  {name:20s} {v}  {'' if v in (ref, 'ok') else '<-- differs'}")
