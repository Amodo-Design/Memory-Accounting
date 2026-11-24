"""Background GPU load standing in for a running inference workload.

A thread keeps a low-priority stream busy with a mix of large bf16 matmuls (compute-bound, like
prefill) and large elementwise passes over a resident buffer (bandwidth-bound, like decode).  The
challenge kernels run on the highest-priority stream, which is what an honest inference server
would legitimately do.  Throughput achieved by the load is reported so the run is auditable.
"""

from __future__ import annotations

import threading
import time

import torch


def is_cuda(device) -> bool:
    """'cuda', 'cuda:1' or a torch.device of type cuda."""
    try:
        return torch.device(device).type == "cuda"
    except (RuntimeError, TypeError, ValueError):
        return False


class BackgroundLoad:
    def __init__(
        self,
        device: str,
        matmul_dim: int = 8192,
        bandwidth_bytes: int = 1 << 30,
        inflight: int = 4,
        fraction: float = 1.0,
    ):
        """fraction: target duty cycle in (0, 1]; below 1 the thread sleeps between iterations so the
        GPU is busy roughly that fraction of the time (coarse: one iteration is a few ms)."""
        self.device, self.dim, self.inflight = device, matmul_dim, inflight
        self.fraction = min(max(fraction, 0.01), 1.0)
        self.cuda = is_cuda(device)
        if not self.cuda:
            raise RuntimeError("--background-load needs a CUDA device")
        self.a = torch.randn(matmul_dim, matmul_dim, dtype=torch.bfloat16, device=device)
        self.b = torch.randn(matmul_dim, matmul_dim, dtype=torch.bfloat16, device=device)
        self.c = torch.empty_like(self.a)
        self.buf = torch.zeros(bandwidth_bytes // 2, dtype=torch.bfloat16, device=device)
        self.iters = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.elapsed = 0.0
        self.error: str | None = None  # why the load thread stopped before stop(), if it did
        least, _greatest = torch.cuda.Stream.priority_range()
        self.stream = torch.cuda.Stream(device=self.a.device, priority=least)  # lowest priority, on the load's GPU

    def _loop(self):
        t0 = time.perf_counter()
        try:
            torch.cuda.set_device(self.a.device)  # give this thread a context before cuBLAS runs
            events = []
            busy_since = time.perf_counter()
            with torch.cuda.stream(self.stream):
                while not self._stop.is_set():
                    torch.matmul(self.a, self.b, out=self.c)  # ~2*dim^3 FLOP
                    self.buf.mul_(1.0001).add_(0.5)  # 2 passes over the buffer
                    self.iters += 1
                    ev = torch.cuda.Event()
                    ev.record(self.stream)
                    events.append(ev)
                    if len(events) > self.inflight:
                        events.pop(0).synchronize()  # bound the queue depth
                    if self.fraction < 1.0 and len(events) >= self.inflight:
                        # duty cycle: drain, then idle so busy/(busy+idle) ~= fraction
                        self.stream.synchronize()
                        events.clear()
                        busy = time.perf_counter() - busy_since
                        time.sleep(busy * (1.0 - self.fraction) / self.fraction)
                        busy_since = time.perf_counter()
            self.stream.synchronize()
        except BaseException as e:  # the sweep goes on without load: say so in the report
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.elapsed = time.perf_counter() - t0

    @property
    def alive(self) -> bool:
        """Whether the load is running (started, not stopped, and its thread has not died)."""
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="posm-background-load", daemon=True)
        self._thread.start()
        time.sleep(0.5)  # let the queue fill

    def stop(self) -> dict:
        """Stop the load and report what it did. A load thread that died before (an error in it: the sweep ran
        without load) or never ran reports "error" and no throughput, not a rate divided by a zero time."""
        died = self._thread is not None and not self._thread.is_alive()
        self._stop.set()
        if self._thread:
            self._thread.join()
        error = self.error or (
            "the load thread was never started"
            if self._thread is None
            else "the load thread stopped before the sweep ended"
            if died
            else None
        )
        ok = error is None and self.elapsed > 0
        flops = 2.0 * self.dim**3 * self.iters
        gbytes = 4.0 * self.buf.numel() * 2 * self.iters / 1e9  # 2 passes x read+write
        return {
            "iterations": self.iters,
            "seconds": self.elapsed,
            "tflops": flops / self.elapsed / 1e12 if ok else None,
            "bandwidth_gbps": gbytes / self.elapsed if ok else None,
            "matmul_dim": self.dim,
            "buffer_bytes": self.buf.numel() * 2,
            "fraction": self.fraction,
            **({"error": error or "no time measured"} if not ok else {}),
        }
