"""What the verifier knows about the prover's hardware, independent of anything the prover reports.

The memory bound needs the GPU's capacity from the verifier's side: the prover's own figure (its
`cudaMemGetInfo`, the `hbm_bytes` of a bundle it wrote) would let it under-report and keep the rest. The
capacity here is what CUDA exposes on the device (the driver's own reservation is outside it), measured
on a device of that type. In a deployment the GPU type itself would come from attestation; here the
verifier is told it (`--gpu-type`)."""

from __future__ import annotations

# name as torch.cuda.get_device_name reports it -> bytes CUDA exposes (cudaMemGetInfo total)
CAPACITY: dict[str, int] = {
    "NVIDIA H200 NVL": 150_111_977_472,  # measured on an H200 NVL, driver 595, Oct 2026 (139.8 GiB)
}


def capacity_bytes(gpu_type: str | None = None, hbm_bytes: int | None = None) -> int | None:
    """The verifier's capacity figure: an explicit byte count wins, else the table entry for the type."""
    if hbm_bytes:
        return int(hbm_bytes)
    if gpu_type is None:
        return None

    # "H200-NVL" = "NVIDIA H200 NVL"
    def norm(t):
        return "".join(c for c in t.upper().removeprefix("NVIDIA") if c.isalnum())

    hits = [v for k, v in CAPACITY.items() if norm(k) == norm(gpu_type)]
    if not hits:
        raise SystemExit(f"unknown GPU type {gpu_type!r}; known: {sorted(CAPACITY)} (or give --hbm-bytes)")
    return hits[0]
