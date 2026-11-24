"""Inference throughput under PoSM: plain vLLM against vLLM with the scheme's parts turned on, one variant per process.

  python scripts/bench_throughput.py <variant> --model Qwen/Qwen2.5-0.5B-Instruct --out results/throughput/<variant>.json

Variants:
  vanilla   stock vLLM (CUDA graphs, its default attention backend), no PoSM
  settings  vLLM with the settings PoSM needs for replay (memory_accounting.models.vllm: batch-invariant kernels, eager, TRITON_ATTN,
            no prefix caching or chunked prefill), no PoSM
  encoded   settings + the weights held only as Θ' and stream-decoded layer by layer, and the free memory filled with
            filler (posm_convert); no responder, no challenges
  dynamic   encoded + the dynamic region (posm_dynamic_setup: the KV pool and activation rows held encoded, v0.17, or plain
            with --plain-dynamic); no responder, no challenges
  full      encoded + the dynamic region (posm_dynamic_setup) + the responder (posm_serve), measured while a live verifier
            challenges it. Exports the bundle, then generates (unmeasured) until --go-file exists, so the verifier can be
            started and calibrate under load; then measures.
  stock+<s>[+<s>...]  stock vLLM with some of the PoSM settings (names in STOCK_SETTINGS), to split their cost; `encode` among
            them also runs posm_convert (stream-decode needs eager: under CUDA graphs and torch.compile, set up before the
            conversion, its decode hooks would not run)

Every variant that encodes checks that greedy output after the conversion matches the output before it.

Workload: for each batch size, that many requests of --prompt-len random token ids each generating exactly --gen-len
tokens (greedy, ignore_eos), one warm-up call, then calls repeated for --seconds. Throughput = generated tokens / wall
time of the call (prefill included). Prompts differ in every request and call, so no cache helps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

# each PoSM setting (memory_accounting.models.vllm) on its own: (environment, LLM keyword arguments)
STOCK_SETTINGS = {
    "batch_invariant": ({"VLLM_BATCH_INVARIANT": "1"}, {}),
    "v1_runner": ({"VLLM_USE_V2_MODEL_RUNNER": "0"}, {}),
    "eager": ({}, {"enforce_eager": True}),
    "triton_attn": ({}, {"attention_config": {"backend": "TRITON_ATTN"}}),
    "no_prefix_cache": ({}, {"enable_prefix_caching": False}),
    "no_chunked_prefill": ({}, {"enable_chunked_prefill": False}),
    "no_async": ({}, {"async_scheduling": False}),
    "token_budget": ({}, {"max_num_batched_tokens": None}),  # = max_model_len, as settings() sets it
}


def stock_parts(variant: str) -> list[str]:
    parts = variant.split("+")[1:]
    bad = [x for x in parts if x != "encode" and x not in STOCK_SETTINGS]
    if variant.split("+")[0] != "stock" or bad:
        raise SystemExit(f"unknown variant {variant!r} (stock+<{'|'.join(STOCK_SETTINGS)}|encode>...)")
    return parts


def encodes(variant: str) -> bool:
    return variant in ("encoded", "dynamic", "full") or (
        variant.startswith("stock+") and "encode" in stock_parts(variant)
    )


def build_llm(variant: str, args):
    if variant == "vanilla" or variant.startswith("stock+"):
        kw = {
            "model": args.model,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        for part in stock_parts(variant) if variant != "vanilla" else []:
            if part == "encode":
                kw["worker_extension_cls"] = "memory_accounting.prover.vllm_ext.PosmWorkerExtension"
                continue
            env, extra = STOCK_SETTINGS[part]
            os.environ.update(env)
            kw.update({k: (args.max_model_len if v is None else v) for k, v in extra.items()})
        from vllm import LLM

        if args.model_revision:
            kw["revision"] = args.model_revision
        if args.max_num_seqs is not None:
            kw["max_num_seqs"] = args.max_num_seqs
        return LLM(**kw)
    from memory_accounting.models.vllm import apply_env, settings

    apply_env()
    from vllm import LLM

    kw = settings(args.model, args.max_model_len, args.gpu_memory_utilization)
    if args.model_revision:
        kw["revision"] = args.model_revision
    if args.max_num_seqs is not None:
        kw["max_num_seqs"] = args.max_num_seqs
    return LLM(**kw)


def setup_posm(llm, variant: str, args) -> dict:
    """The PoSM parts of the variant; returns what they report."""
    info = {}
    if encodes(variant):
        from vllm import SamplingParams

        prompts, sp = ["The quick brown fox", "In 1969, the first"], SamplingParams(temperature=0.0, max_tokens=32)
        before = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
        if variant in ("dynamic", "full"):  # before the conversion: the encoded pool's memory must not go to the filler
            info["dynamic"] = llm.collective_rpc(
                "posm_dynamic_setup",
                args=(args.dyn_block, args.filler_seed, False),
                kwargs={"encoded": not args.plain_dynamic},
            )[0]
        reserve_bytes = -1 if args.scratch_reserve_gb is None else int(args.scratch_reserve_gb * (1 << 30))
        info["converted"] = llm.collective_rpc(
            "posm_convert", args=(args.enc_key, args.filler_seed, 0.0), kwargs={"reserve_bytes": reserve_bytes}
        )[0]
        info["output_matches"] = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)] == before
        if not info["output_matches"]:
            raise SystemExit(f"{variant}: greedy output changed after the conversion")
    if variant == "full":
        capture_evidence = not args.no_record_capture
        info["bundle"] = llm.collective_rpc(
            "posm_export", args=(args.bundle, args.filler_seed),
            kwargs={"benchmark_only": args.skip_response_checks and not getattr(args, "registered_stream", False),
                    "include_dynamic_size": True}
        )[0]
        info["serve"] = llm.collective_rpc(
            "posm_serve",
            args=(args.port, None, "honest", None, True, 100.0),
            kwargs={"verifier_addrs": args.verifier_addr, "capture_evidence": capture_evidence},
        )[0]
    if encodes(variant):
        info["hbm"] = llm.collective_rpc("posm_hbm")[0]
    return info


def served(llm, variant: str) -> dict:
    if variant != "full":
        return {}
    i = llm.collective_rpc("posm_info")[0]
    return {"static": i["static_served"], "dynamic": (i.get("dynamic") or {}).get("served", 0)}


def diagnostic_snapshot(llm, variant: str) -> dict | None:
    """Worker counters around a measured call, queried outside its wall-time interval."""
    if variant != "full":
        return None
    return llm.collective_rpc("posm_diagnostics")[0]


def call_diagnostics(before: dict | None, after: dict | None) -> dict | None:
    if before is None or after is None:
        return None
    gc0, gc1 = before.get("gc") or {}, after.get("gc") or {}
    keys = ("collections", "gen2_collections", "objects_collected", "uncollectable_objects",
            "slow_collections", "total_ms", "slow_ms")
    return {
        "worker_gc_during_call": {k: gc1.get(k, 0) - gc0.get(k, 0) for k in keys},
        "worker_gc_max_seen_ms": gc1.get("max_ms"),
        "evidence_store_before": before.get("evidence_store"),
        "evidence_store_after": after.get("evidence_store"),
    }


def requests(rng, batch: int, prompt_len: int):
    return [{"prompt_token_ids": rng.integers(1000, 30000, prompt_len).tolist()} for _ in range(batch)]


def prover_runtime(llm):
    import torch

    from memory_accounting.core.benchmark import provenance

    runtime = provenance()
    engine = llm.llm_engine.vllm_config
    model = engine.model_config

    gpu = torch.cuda.get_device_properties(0)
    runtime["gpu"] = {"name": gpu.name, "bytes": gpu.total_memory, "compute_capability": [gpu.major, gpu.minor],
                      "torch_cuda": torch.version.cuda}
    try:
        runtime["gpu"]["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=5
        ).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        runtime["gpu"]["driver"] = None
    runtime["engine"] = {
        "model_revision": getattr(model.hf_config, "_commit_hash", None) or getattr(model, "revision", None),
        "tokenizer": model.tokenizer, "tokenizer_revision": getattr(model, "tokenizer_revision", None),
        "dtype": str(model.dtype), "seed": model.seed, "enforce_eager": model.enforce_eager,
        "tensor_parallel_size": engine.parallel_config.tensor_parallel_size,
        "attention_backend": str(getattr(engine.attention_config, "backend", None)),
        "max_num_seqs": engine.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": engine.scheduler_config.max_num_batched_tokens,
        "enable_chunked_prefill": engine.scheduler_config.enable_chunked_prefill,
        "async_scheduling": getattr(engine.scheduler_config, "async_scheduling", False),
        "block_size": engine.cache_config.block_size, "enable_prefix_caching": engine.cache_config.enable_prefix_caching,
    }
    return runtime


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "variant", help="vanilla, settings, encoded, dynamic, full or stock+<setting>[+...] (see the module docstring)"
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model-revision", help="Hugging Face revision to load (a commit pins the weights)")
    ap.add_argument("--prompt-seed", type=int, default=1)
    ap.add_argument("--skip-response-checks", action="store_true",
                    help="full throughput benchmark: disable timed reply checks")
    ap.add_argument("--no-record-capture", action="store_true",
                    help="diagnostic full run only: disable prover evidence capture separately from response checks")
    ap.add_argument("--registered-stream", action="store_true",
                    help="export the complete setup proof even when timed response checks are disabled")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="1,16,64,256")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="pin the vLLM scheduling cap; otherwise use its default")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    ap.add_argument("--scratch-reserve-gb", type=float, default=None,
                    help="optional explicit GPU scratch reserve for encoded stages")
    ap.add_argument("--enc-key", default="0123456789abcdef0123456789abcdef")
    ap.add_argument("--filler-seed", default="5eed5eed5eed5eed")
    ap.add_argument("--dyn-block", default="1M")
    ap.add_argument("--bundle", default="verifier_bundle_bench")
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--verifier-addr", action="append", default=None)
    ap.add_argument("--go-file", default="bench_go", help="full: start measuring once this file exists")
    ap.add_argument(
        "--plain-dynamic", action="store_true", help="dynamic, full: the dynamic region plain, as before v0.17"
    )
    ap.add_argument("--calls", type=int, help="exact measured calls after warm-up (overrides --seconds)")
    args = ap.parse_args()
    if args.skip_response_checks and args.variant != "full":
        ap.error("--skip-response-checks applies to full runs with a live verifier")
    if args.no_record_capture and (args.variant != "full" or not args.skip_response_checks):
        ap.error("--no-record-capture requires full and --skip-response-checks")
    if args.calls is not None and args.calls < 1:
        ap.error("--calls must be positive")
    from memory_accounting.core.codec_profile import PROFILE
    if args.variant == "export-weight-reference":
        llm = build_llm("settings", args)
        print(llm.collective_rpc("posm_export_weight_reference", args=(str(Path(args.out).resolve()),))[0])
        return
    if args.variant not in ("vanilla", "settings", "encoded", "dynamic", "full"):
        stock_parts(args.variant)

    from vllm import SamplingParams

    t0 = time.perf_counter()
    llm = build_llm(args.variant, args)
    posm = setup_posm(llm, args.variant, args)
    print(f"{args.variant}: ready in {time.perf_counter() - t0:.0f} s; {posm}", flush=True)
    rng = np.random.default_rng(args.prompt_seed)
    sp = SamplingParams(temperature=0.0, max_tokens=args.gen_len, ignore_eos=True)

    if args.variant == "full":  # the verifier is started now; generate until it says go
        calibration_batch = max(int(batch) for batch in args.batches.split(","))
        go = Path(args.go_file)
        go.unlink(missing_ok=True)
        print(f"full: waiting for {go} (generating {calibration_batch}-request batches meanwhile)", flush=True)
        last = 0.0
        while not go.exists():
            llm.generate(
                requests(rng, calibration_batch, args.prompt_len),
                SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
                use_tqdm=False,
            )
            if time.perf_counter() - last > 30:
                print(f"  waiting; served {served(llm, 'full')}", flush=True)
                last = time.perf_counter()

    from memory_accounting.core.benchmark import read_verifier_handoff

    reproduction = {
        "prover_options": vars(args), "runtime": prover_runtime(llm),
        "generation": {"temperature": sp.temperature, "ignore_eos": sp.ignore_eos, "max_tokens": sp.max_tokens,
                       "prompt_token_range": [1000, 30000], "warmup_calls_per_batch": 1,
                       "rng_before_first_batch": rng.bit_generator.state},
    }
    if args.variant == "full":
        reproduction["responder"] = {"posture": "honest", "drain": True, "hold_ms": 100.0,
                                      "capture_evidence": not args.no_record_capture}
        reproduction["layout"] = json.loads(Path(posm["bundle"]).read_text())
        reproduction["verifier"] = read_verifier_handoff(args.go_file, posm["bundle"], args.skip_response_checks)
        if reproduction["verifier"].get("settings_available") is False:
            print("Legacy go-file: external verifier settings are unavailable here. Use scripts/bench_full.py for a complete run bundle.", flush=True)
        else:
            # Setup duration differs substantially between checked and timing-only runs.
            # Give both the same warm-up and measured prompts regardless of that duration.
            rng = np.random.default_rng(args.prompt_seed)
            reproduction["generation"]["rng_before_first_batch"] = rng.bit_generator.state
        reproduction["calibration_load"] = {"requests": calibration_batch, "prompt_len": args.prompt_len, "max_tokens": 64,
                                            "temperature": 0, "ignore_eos": True}
    result = {
        "reproduction": reproduction,
        "variant": args.variant,
        "vde_profile": PROFILE,
        "model": args.model,
        "prompt_len": args.prompt_len,
        "gen_len": args.gen_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "posm": posm,
        "batches": [],
    }
    for batch in [int(b) for b in args.batches.split(",")]:
        llm.generate(requests(rng, batch, args.prompt_len), sp, use_tqdm=False)  # warm-up
        calls, s0, end = [], served(llm, args.variant), time.perf_counter() + args.seconds
        while (len(calls) < args.calls if args.calls is not None else (time.perf_counter() < end or len(calls) < 3)):
            reqs = requests(rng, batch, args.prompt_len)
            diag0 = diagnostic_snapshot(llm, args.variant)
            started_unix_ns = time.time_ns()
            c0 = time.perf_counter()
            outs = llm.generate(reqs, sp, use_tqdm=False)
            dt = time.perf_counter() - c0
            diag1 = diagnostic_snapshot(llm, args.variant)
            calls.append({"started_unix_ns": started_unix_ns, "seconds": dt,
                          "tokens": sum(len(o.outputs[0].token_ids) for o in outs),
                          "diagnostics": call_diagnostics(diag0, diag1)})
        s1 = served(llm, args.variant)
        toks, secs = sum(c["tokens"] for c in calls), sum(c["seconds"] for c in calls)
        per_call = [c["tokens"] / c["seconds"] for c in calls]
        row = {
            "batch": batch,
            "calls": calls,
            "tok_s": toks / secs,
            "tok_s_std": float(np.std(per_call)),
            "served": {k: s1[k] - s0[k] for k in s1} if s1 else None,
            "seconds": secs,
        }
        if args.variant == "full" and any(row["served"][kind] <= 0 for kind in ("static", "dynamic")):
            raise RuntimeError(f"full benchmark did not serve both challenge kinds: {row}")
        result["batches"].append(row)
        print(
            f"{args.variant} batch {batch}: {row['tok_s']:.0f} tok/s (± {row['tok_s_std']:.0f} over {len(calls)} calls)"
            + (f"; challenges served {row['served']}" if s1 else ""),
            flush=True,
        )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
