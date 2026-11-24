"""vLLM under PoSM, the parts prover and verifier share: the engine settings both sides run with and dense byte views of
KV pages. The prover side is memory_accounting.prover.vllm_serve, the verifier's replay memory_accounting.verifier.vllm_replay.

Required settings (both sides; `settings()` returns them): VLLM_BATCH_INVARIANT=1, the V1 model runner,
enforce_eager, TRITON_ATTN, no prefix caching (a cached prefix would be in blocks the stage does not
account for), no chunked prefill, no async scheduling, no speculative decoding, tensor parallel size 1, a
uniform bf16/fp16/fp32 KV cache (vLLM 0.30 zeroes new blocks only for Mamba layers and mixed-precision
caches; attention reads only the slots of a sequence it has written, so filler in unwritten slots is
never read, and it is finite in any case).

Attention backend, measured on an H200 NVL with vLLM 0.30.0 (bi_probe in the design log): FLASH_ATTN in
batch-invariant mode is batch- and decode/prefill-invariant only while a sequence fits in one KV tile
of 128 tokens (beyond it, 11 of 12 decode logprobs differ from prefill and from batched runs);
TRITON_ATTN and FLEX_ATTENTION are invariant at 152 tokens and in the self-test. TRITON_ATTN is the
default; POSM_VLLM_ATTN overrides it (both sides must agree).
"""

from __future__ import annotations

import os

import torch

ENV = {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "0"}


def settings(
    model: str,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.5,
    block_size: int = 16,
    verifier: bool = False,
    backend: str | None = None,
) -> dict:
    """LLM(...) keyword arguments shared by prover and verifier; set ENV in os.environ before importing vllm."""
    kw = {
        "model": model,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "async_scheduling": False,
        "max_model_len": max_model_len,
        "block_size": block_size,
        "max_num_batched_tokens": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "attention_config": {"backend": backend or os.environ.get("POSM_VLLM_ATTN", "TRITON_ATTN")},
        "worker_extension_cls": "memory_accounting.prover.vllm_ext.PosmWorkerExtension",
    }
    if verifier:
        kw["max_num_seqs"] = 1  # one request per forward: the replay is per request
    return kw


def apply_env() -> None:
    for k, v in ENV.items():
        os.environ[k] = v


# ---- pages: dense views of KV memory as bytes in physical order


def page_bytes(page: torch.Tensor) -> torch.Tensor:
    """A uint8 view of a tensor's bytes in memory order (no copy). The tensor must be dense (its elements
    fill one contiguous byte range): a page, or a whole layer's pool in a layer-major layout."""
    if page.is_contiguous():
        return page.reshape(-1).view(torch.uint8)
    if any(s <= 0 for s in page.stride()):
        raise ValueError("page with non-positive strides")
    span = sum((n - 1) * s for n, s in zip(page.shape, page.stride())) + 1
    if span != page.numel():
        raise ValueError("KV memory is not dense in this layout; set VLLM_KV_CACHE_LAYOUT=LBNHC")
    nbytes = page.numel() * page.element_size()
    out = torch.empty(0, dtype=torch.uint8, device=page.device)
    out.set_(page.untyped_storage(), page.storage_offset() * page.element_size(), (nbytes,), (1,))
    return out


def physical_order(page: torch.Tensor) -> list[int]:
    """The logical dims of a page sorted from outermost to innermost in memory."""
    return sorted(range(page.dim()), key=lambda d: -page.stride(d))
