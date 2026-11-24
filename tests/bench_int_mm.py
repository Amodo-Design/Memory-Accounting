"""Feasibility check for using cuBLASLt's int8 GEMM (torch._int_mm) as the challenge matmul.
Run on the CUDA host:  uv run python tests/bench_int_mm.py"""

import time

import torch

assert torch.cuda.is_available()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False


def bench(rows, K, r, iters=200):
    A = torch.randint(-128, 127, (rows, K), dtype=torch.int8, device=dev)
    M = torch.randint(-128, 127, (K, r), dtype=torch.int8, device=dev)
    C = torch._int_mm(A, M)
    torch.cuda.synchronize()
    # exactness vs int32 reference on a slice
    ref = A[:64].cpu().int() @ M.cpu().int()  # int32 matmul exists on CPU only
    assert torch.equal(C[:64].cpu(), ref), "int_mm mismatch"
    stream = torch.cuda.Stream(priority=torch.cuda.Stream.priority_range()[1])
    with torch.cuda.stream(stream):
        for _ in range(5):
            C = torch._int_mm(A, M)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            C = torch._int_mm(A, M)
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    # graph capture of the same op
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        _out = torch._int_mm(A, M)  # kept: the graph replays into it
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    dg = (time.perf_counter() - t0) / iters
    macs = rows * K * r
    print(
        f"rows={rows:5d} K={K} r={r:5d}: eager {dt * 1e6:7.1f} us ({macs / dt / 1e12:6.1f} TMAC/s)   graph {dg * 1e6:7.1f} us ({macs / dg / 1e12:6.1f} TMAC/s)"
    )


for rows, r in ((64, 256), (64, 2048), (256, 2048), (1024, 256), (1024, 2048), (1024, 4096)):
    bench(rows, 1024, r)
