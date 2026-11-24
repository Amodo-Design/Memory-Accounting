"""The integer-exact model: conversion (Llama, Qwen2), exact integer matmuls, invariance to the chunking
of the input (which a float model does not have), the division rule, and stream-decode of its weights
from Θ' with bit-identical outputs. CPU only; `python -m memory_accounting.models.intllm crosscheck --device cuda` is the GPU check."""

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from memory_accounting.models.intllm import IntLLM, convert_hf, crosscheck, fdiv, lin, quant_rows


def _tiny(kind: str, tmp: pathlib.Path, seed: int = 0) -> pathlib.Path:
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    Cfg, M = (LlamaConfig, LlamaForCausalLM) if kind == "llama" else (Qwen2Config, Qwen2ForCausalLM)
    cfg = Cfg(
        vocab_size=101,
        hidden_size=64,
        intermediate_size=160,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=(kind == "llama"),
    )
    M(cfg).eval().save_pretrained(tmp / kind)
    out = tmp / f"{kind}.int.safetensors"
    convert_hf(str(tmp / kind), str(out), max_pos=256)
    return out


def _kv_bytes(m):
    return [t.contiguous().view(torch.uint8).clone() for _, t in m.kv_buffers()]


def test_exact_integer_matmul_and_division_rule():
    rng = np.random.default_rng(1)
    q = torch.from_numpy(rng.integers(-127, 128, (37, 4096)).astype(np.float64))
    w = torch.from_numpy(rng.integers(-127, 128, (53, 4096)).astype(np.int8))
    exact = q.numpy().astype(np.int64) @ w.numpy().astype(np.int64).T
    one = torch.ones(37, dtype=torch.float64)
    assert np.array_equal(lin(q, one, w, torch.ones(53, dtype=torch.float64)).numpy().astype(np.int64), exact)
    x = torch.randn(5, 9, dtype=torch.float64)
    qq, s = quant_rows(x)
    assert (
        qq.abs().max() <= 127
        and torch.equal(qq, torch.round(qq))
        and torch.allclose(qq * s.unsqueeze(-1), x, atol=float(s.max()) / 2 + 1e-12)
    )
    try:
        fdiv(x, 3.0)
        raise AssertionError("fdiv accepted a python scalar divisor")
    except TypeError:
        pass


def test_conversion_and_chunk_invariance():
    d = pathlib.Path(tempfile.mkdtemp())
    for kind in ("llama", "qwen2"):
        model = IntLLM.from_file(_tiny(kind, d), max_len=64)
        assert model.cfg["qkv_bias"] == (kind == "qwen2") and model.cfg["tied"] == (kind == "llama")
        out = model.generate([5, 17, 3, 99, 42, 7, 7, 1], 12)
        fed = out[: model.pos]
        kv = _kv_bytes(model)
        variants = {
            "token by token": [[t] for t in fed],
            "one prefill": [fed],
            "uneven chunks": [fed[:5], fed[5:13], fed[13:]],
        }
        for name, chunks in variants.items():
            model.reset()
            for c in chunks:
                model(torch.tensor([c]))
            assert all(torch.equal(x, y) for x, y in zip(kv, _kv_bytes(model))), (kind, name)
        assert model.generate([5, 17, 3, 99, 42, 7, 7, 1], 12) == out  # deterministic


def test_stream_decoded_weights_give_identical_outputs():
    """The integer model's parameters (int8 weights, float64 scales and tables) can live as Θ' and be
    decoded per layer, as intserve does on the GPU: outputs and KV bytes do not change. Outside the layers only
    the tables stay decoded (every layer reads them); the embedding is decoded before the layers, the final norm
    and the head after them, and a tied head (llama here) is the embedding at both."""
    from memory_accounting.prover.stream_decode import convert

    d = pathlib.Path(tempfile.mkdtemp())
    for kind, seed in (("qwen2", 3), ("llama", 5)):
        path = _tiny(kind, d, seed=seed)
        model = IntLLM.from_file(path, max_len=48)
        ref = model.generate([1, 2, 3, 4], 10)
        ref_kv = _kv_bytes(model)
        originals = {n: p.detach().clone() for n, p in model.named_parameters()}
        sd = convert(model, enc_key=(0x1234 << 64) | 0x5678)
        assert set(sd.units) == {"rest", "layers.0", "layers.1", "layers.2"}
        assert {sl.name for sl in sd.resident_slots} == {n for n in originals if n.startswith("tables.")}
        tied = kind == "llama"
        assert {sl.name for sl in sd.rest_groups["before"]} == {"embed", "embed_s"}
        assert {sl.name for sl in sd.rest_groups["after"]} == {"final_norm"} | (
            {"embed", "embed_s"} if tied else {"head", "head_s"}
        )
        assert model.generate([1, 2, 3, 4], 10) == ref, kind
        assert all(torch.equal(x, y) for x, y in zip(ref_kv, _kv_bytes(model)))
        assert sd.check_parameters(originals) == [], kind


def test_resize_keeps_the_weights():
    """A verifier replays records of any cache length with one model: `resize` makes new, empty cache buffers
    and keeps the weights, and the model computes what one loaded at that length does (KV bytes included)."""
    d = pathlib.Path(tempfile.mkdtemp())
    path = _tiny("llama", d, seed=4)
    fresh = IntLLM.from_file(path, max_len=40)
    ref = fresh.generate([3, 1, 4, 1, 5], 9)
    m = IntLLM.from_file(path, max_len=1)
    m.resize(40)
    assert m.max_len == 40 and m.info["max_len"] == 40 and all(t.shape[0] == 40 for _, t in m.kv_buffers())
    assert m.generate([3, 1, 4, 1, 5], 9) == ref
    assert all(torch.equal(x, y) for x, y in zip(_kv_bytes(fresh), _kv_bytes(m)))
    for bad in (0, m.cfg["max_pos"] + 1):
        try:
            m.resize(bad)
            raise AssertionError(f"resize({bad}) accepted")
        except ValueError:
            pass


def test_crosscheck_cpu():
    d = pathlib.Path(tempfile.mkdtemp())
    ok, lines = crosscheck(_tiny("llama", d), "cpu", tokens=8, max_len=32)
    assert ok, lines


if __name__ == "__main__":
    from _runner import run

    run(globals())
