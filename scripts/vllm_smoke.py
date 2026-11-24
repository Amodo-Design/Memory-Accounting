"""vLLM smoke test for the PoSM worker extension.

  uv run python scripts/vllm_smoke.py --model models/qwen0.5b --port 5557 [--strategies honest,dram_mapped,...]

Starts vLLM with the extension, generates, converts the resident weights to Theta' + per-layer
decode, generates again and compares, exports the verifier bundle, starts the in-process responder,
then keeps generating so challenges arrive under a real vLLM decode load.
"""

from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--enc-key", default="0123456789abcdef0123456789abcdef")
    ap.add_argument("--filler-seed", default="5eed5eed5eed5eed")
    ap.add_argument(
        "--scratch-fraction", type=float, default=0.10, help="of total HBM; vLLM's KV cache counts as scratch here"
    )
    ap.add_argument(
        "--filler-gb",
        type=float,
        default=1.0,
        help="cap on the filler (GiB); 0 = fill free HBM. The cheat simulators need "
        "headroom: a raw copy on this GPU (encode_otf, other_gpu) and pinned host copies (dram_*)",
    )
    ap.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="vLLM's share (weights + KV cache); the filler takes the rest minus scratch",
    )
    ap.add_argument("--strategies", default="honest")
    ap.add_argument("--canon-device", default="cuda:1")
    ap.add_argument(
        "--no-drain", action="store_true", help="answer challenges concurrently with vLLM steps instead of between them"
    )
    ap.add_argument(
        "--partial-fraction",
        type=float,
        default=0.10,
        help="partial cheat: fraction of the array served from host DRAM",
    )
    ap.add_argument(
        "--lora-fraction",
        type=float,
        default=0.05,
        help="lora cheat: fraction of the blocks altered in the prover's copy",
    )
    ap.add_argument("--bundle", default="verifier_bundle_vllm")
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        worker_extension_cls="memory_accounting.prover.vllm_ext.PosmWorkerExtension",
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    prompts = ["The quick brown fox", "In 1969, the first"]
    ref = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    print("vanilla:", ref, flush=True)
    cap = int(args.filler_gb * (1 << 30)) if args.filler_gb > 0 else None
    info = llm.collective_rpc(
        "posm_convert", args=(args.enc_key, args.filler_seed, args.scratch_fraction), kwargs={"filler_max_bytes": cap}
    )
    print("converted:", info, flush=True)
    out = [o.outputs[0].text for o in llm.generate(prompts, sp)]
    print("stream-decode output matches vanilla:", out == ref, flush=True)
    if out != ref:
        print("  got:", out)
    bundle = llm.collective_rpc("posm_export", args=(args.bundle, args.filler_seed))
    print("verifier bundle:", bundle, flush=True)
    print(
        llm.collective_rpc(
            "posm_serve",
            args=(
                args.port,
                None,
                args.strategies,
                args.canon_device,
                not args.no_drain,
                100.0,
                args.partial_fraction,
                args.lora_fraction,
            ),
        ),
        flush=True,
    )
    t0 = time.perf_counter()
    toks = 0
    try:
        while True:
            outs = llm.generate(prompts * 8, SamplingParams(temperature=0.0, max_tokens=64), use_tqdm=False)
            toks += sum(len(o.outputs[0].token_ids) for o in outs)
            if time.perf_counter() - t0 > 10:
                served = llm.collective_rpc("posm_info")[0].get("served")
                print(
                    f"{toks / (time.perf_counter() - t0):.0f} tok/s under vLLM + stream-decode; {served} challenges served",
                    flush=True,
                )
                t0 = time.perf_counter()
                toks = 0
    except KeyboardInterrupt:
        print("\nstopping", flush=True)


if __name__ == "__main__":
    main()
