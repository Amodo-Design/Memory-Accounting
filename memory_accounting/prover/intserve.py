"""The integer-exact model as a prover (GPU), whose dynamic memory a CPU verifier can check exactly:

  python -m memory_accounting.prover.intserve --model models/smollm2-135m.int.safetensors --enc-key <hex> --filler-seed <hex> --port 5557

1. loads the integer model (`memory_accounting.models.intllm`) on the GPU and checks that `crosscheck` passes (the same
   tokens give the same KV bytes on this GPU and on this host's CPU);
2. stream-decodes its weights from Θ' (the static scheme), fills free HBM with the verifier-seeded
   filler, and exports the verifier bundle (one-way image + manifest);
3. builds the dynamic region (int8 KV cache + float64 scales per layer, the float64 activation buffer,
   the pad), held encoded (v0.17, `memory_accounting.prover.kvenc`: one plaintext cache shared by the layers as scratch), with
   pause points at every layer and step boundary;
4. starts the responder (static challenges over Θ' + filler, dynamic challenges over the region,
   stage records over TCP) and keeps generating from a cycle of prompts.

The verifier, on any machine with the model file and torch (CPU wheels; `uv sync`, not
`--no-default-groups`, because the replay runs the model):

  python -m memory_accounting.verifier.net verifier --manifest bundle/manifest.json --host <gpu host> --port 5557 --drain --dynamic \\
      --replay int:models/smollm2-135m.int.safetensors --dyn-B 64K --n 8
"""

from __future__ import annotations

import argparse
import time

import torch

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Memory accounting checks what sits in a GPU's memory by",
    "Once upon a time",
    "The three laws of thermodynamics are",
]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="integer-exact model prover: static + dynamic challenges, CPU-verifiable replay"
    )
    ap.add_argument("--model", required=True, help="integer model file (python -m memory_accounting.models.intllm convert ...)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-crosscheck", action="store_true")
    _add_static_args(ap.add_argument_group("static: Θ', filler and the verifier bundle"))
    _add_dynamic_args(ap.add_argument_group("dynamic: the KV cache and activation region"))
    _add_responder_args(ap.add_argument_group("responder: challenges and cheats"))
    _add_inference_args(ap.add_argument_group("inference load"))
    return ap


def _add_static_args(g) -> None:
    g.add_argument("--enc-key", required=True)
    g.add_argument(
        "--filler-seed", required=True, help="hex seed supplied by the verifier (memory_accounting.setup.cli filler-seed)"
    )
    g.add_argument("--scratch-fraction", type=float, default=0.0)
    g.add_argument(
        "--scratch-gb",
        type=float,
        default=1.0,
        help="memory left unfilled for the model's own use: the prover's scratch",
    )
    g.add_argument(
        "--filler-gb", type=float, default=0.0, help="cap the filler (GiB); 0 = free HBM minus the scratch allowance"
    )
    g.add_argument("--bundle", default="verifier_bundle_int", help="where to write the verifier's image + manifest")
    g.add_argument(
        "--chain-B",
        default="1M",
        help="the static challenge's block (the bundle's; Θ' and the filler end on a whole one)",
    )
    g.add_argument(
        "--chain-r",
        type=int,
        default=2048,
        help="the challenge's matmul width r (the bundle's; dynamic chains use it too)",
    )
    g.add_argument("--chain-k", type=int, default=1024, help="the challenge's matmul inner dimension k (the bundle's)")
    g.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="this GPU's index as the verifier knows it: keys the static and dynamic filler, names the bundle's GPU entry "
        "(default: the device's index)",
    )


def _add_dynamic_args(g) -> None:
    g.add_argument("--max-len", type=int, default=512, help="KV cache slots (the whole cache is in the dynamic region)")
    g.add_argument(
        "--dyn-block", default="64K", help="block size the dynamic region is padded to (the verifier's --dyn-B)"
    )
    g.add_argument("--pad-seed", default=None, help="hex seed of the region's pad (default: the filler seed)")
    g.add_argument(
        "--evidence-bytes", action="store_true", help="attach the region's bytes to each record (debugging a mismatch)"
    )
    g.add_argument(
        "--plain-dynamic",
        action="store_true",
        help="hold the KV cache and activation buffer plain, as before v0.17 "
        "(default: encoded, memory_accounting.prover.kvenc; the verifier's --dyn-encoded credits the whole region)",
    )


def _add_responder_args(g) -> None:
    g.add_argument("--port", type=int, default=5555)
    g.add_argument(
        "--verifier-addr", action="append", default=None, help="answer only datagrams from this address (repeatable)"
    )
    g.add_argument("--no-drain", action="store_true", help="answer static challenges concurrently with inference")
    g.add_argument(
        "--strategies",
        default="honest",
        help="cheats to offer (see memory_accounting.verifier.net); evict, evict_dram, tamper and dyn_dram work with the GPU full",
    )
    g.add_argument(
        "--evict-fraction", type=float, default=0.10, help="evict / evict_dram: share of the array the cheater reclaims"
    )
    g.add_argument("--tamper-fraction", type=float, default=0.10, help="tamper: share of the weight blocks it modifies")
    g.add_argument(
        "--dyn-evict-fraction",
        type=float,
        default=0.5,
        help="dyn_dram: share of every KV buffer's slots (the top ones) whose "
        "filler is kept in pinned host memory and read from there while free",
    )
    g.add_argument("--canon-device", default="cuda:1")


def _add_inference_args(g) -> None:
    g.add_argument("--new-tokens", type=int, default=96)
    g.add_argument(
        "--step-sleep-ms",
        type=float,
        default=0.0,
        help="pause between decode steps, spread over the layers (demo pacing)",
    )
    g.add_argument(
        "--request-gap-ms", type=float, default=0.0, help="idle time between requests (exercises the idle freeze)"
    )


def load_model(args):
    """The integer model on the GPU, checked against this host's CPU, with its prompts and a token printer."""
    from ..models.intllm import IntLLM, crosscheck

    if not args.skip_crosscheck:
        ok, lines = crosscheck(args.model, args.device, tokens=24, max_len=min(args.max_len, 128))
        for line in lines:
            print("crosscheck:", line, flush=True)
        if not ok:
            raise SystemExit(
                "crosscheck failed: this GPU and this CPU do not compute the same bytes; a CPU verifier would reject every dynamic challenge"
            )
    model = IntLLM.from_file(args.model, args.max_len, args.device)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model.cfg["source"])
        prompts = [tok(t)["input_ids"] for t in DEFAULT_PROMPTS]
        stop = (tok.eos_token_id,) if tok.eos_token_id is not None else ()

        def show(ids):
            return repr(tok.decode(ids))
    except Exception as e:
        print(f"no tokenizer for {model.cfg.get('source')} ({e}); using fixed token prompts", flush=True)
        prompts, stop, show = [[1, 2, 3, 4, 5], [7, 11, 13]], (), str
    return model, prompts, stop, show


def to_stream_decode(model, args, prompt, show):
    """Keep the weights resident only as Θ' and check that the model's output does not change."""
    from .stream_decode import convert

    ref = model.generate(prompt, 24)
    print("integer model:", show(ref), flush=True)
    sd = convert(model, int(args.enc_key, 16))
    out = model.generate(prompt, 24)
    print("stream-decode output identical:", out == ref, flush=True)
    print(
        f"weights in plaintext: scratch {sd.scratch.numel() >> 10} KiB, resident {(sd.plain_bytes - sd.scratch.numel()) >> 10} KiB",
        flush=True,
    )
    if out != ref:
        raise SystemExit("stream-decode changed the integer model's output")
    return sd


def build_dynamic(model, args, dev, gpu: int, keys):
    """The dynamic region (int8 K/V + scales per layer, the activation buffer, the pad), its pause points and its responder."""
    from ..core.params import parse_bytes
    from ..models.toy import attach_dynamic, static_dram_fn, static_live_fn
    from .dynamic import DynamicResponder

    region, tracker, pause, _hooks = attach_dynamic(
        model,
        dev,
        parse_bytes(args.dyn_block),
        int(args.pad_seed or args.filler_seed, 16),
        args.max_len,
        torch.float64,
        gpu=gpu,
        encoded=not args.plain_dynamic,
    )
    dyn = DynamicResponder(
        region,
        tracker,
        pause,
        keys,
        model_info=model.info,
        include_bytes=args.evidence_bytes,
        live_fn=static_live_fn(model, region),
        dram_fn=static_dram_fn(model, region),
    )
    return region, tracker, pause, dyn


def build_static(model, sd, args, dev, gpu: int, keys):
    """The filler over free HBM, the challenged array (Θ' + filler) and the verifier bundle."""
    from ..core.params import parse_bytes
    from .filler import allocate_filler
    from .responder import ResidentArray

    # the filler's key: from the verifier's seed and this GPU's index, as the verifier derives it
    fill = allocate_filler(
        int(args.filler_seed, 16),
        sd.enc_key,
        sd.payload_words,
        args.scratch_fraction,
        dev,
        block_bytes=parse_bytes(args.chain_B),
        max_bytes=int(args.filler_gb * (1 << 30)) if args.filler_gb > 0 else None,
        reserve_bytes=int(args.scratch_gb * (1 << 30)),
        gpu=gpu,
    )
    resident = ResidentArray(
        sd.segments() + fill.segments(),
        sd.enc_key,
        str(dev),
        strategies=_strategies(args),
        canon_device=args.canon_device,
        segment_keys=(sd.enc_key, fill.key) if fill.words else (sd.enc_key,),
        evict_fraction=args.evict_fraction,
        tamper_fraction=args.tamper_fraction,
    )
    bundle = sd.export_bundle(
        args.bundle,
        filler=fill,
        filler_seed=int(args.filler_seed, 16),
        model_name=model.info["file"],
        gpu=gpu,
        block_bytes=parse_bytes(args.chain_B),
        r=args.chain_r,
        k=args.chain_k,
    )
    print("verifier bundle:", bundle, flush=True)
    return fill, resident


def _strategies(args) -> tuple:
    return tuple(s for s in args.strategies.split(",") if s)


def start_responder(model, resident, dyn, args, keys):
    """The responder for static and dynamic challenges; with draining, every forward waits for the challenge in flight."""
    from .responder import Responder, watch_gc

    resp = Responder(
        resident,
        keys,
        port=args.port,
        strategies=_strategies(args),
        drain=not args.no_drain,
        dynamic=dyn,
        verifier_addrs=args.verifier_addr,
        dyn_evict_fraction=args.dyn_evict_fraction,
    )
    if not args.no_drain:
        from .serve import gate_forward

        gate_forward(model, resp.gate)  # released by the forward that took it, also when it raises
    resp.start()
    watch_gc()
    return resp


def report_forever(loop, resp, dyn, pause, dev, static_bytes: int, region, show) -> None:
    """Print throughput and challenge counts every 10 s until interrupted."""
    from .filler import hbm_line

    t0 = time.perf_counter()
    tok0 = 0
    while True:
        time.sleep(10)
        rate = (loop.tokens - tok0) / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        tok0 = loop.tokens
        print(
            f"{rate:.1f} tok/s; {resp.served} challenges served; dynamic {dyn.served} ({dyn.paused_in_flight} mid-step, "
            f"{dyn.paused_between} between steps, {dyn.idle} idle; {dyn.static_freezes} static freezes); inference held off by a frozen region {pause.held_off} times; "
            f"{hbm_line(dev, static_bytes, region.nbytes, dyn.free_min)}; "
            f"last: {show(loop.last_output)[:60]}",
            flush=True,
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .responder import require_free_ports

    require_free_ports(args.port)  # before the model loads: the challenge port and the records' TCP port
    from ..core.params import Keys
    from ..models.toy import InferenceLoop
    from .filler import hbm_line

    model, prompts, stop, show = load_model(args)
    sd = to_stream_decode(model, args, prompts[0], show)
    keys = Keys.from_seed(int(args.filler_seed, 16))  # the challenge-hash keys: from the verifier's seed
    dev = torch.device(args.device)
    gpu = args.gpu if args.gpu is not None else (dev.index or 0)
    region, tracker, pause, dyn = build_dynamic(model, args, dev, gpu, keys)
    fill, resident = build_static(model, sd, args, dev, gpu, keys)
    resp = start_responder(model, resident, dyn, args, keys)

    print(
        f"Θ' {sd.theta.numel() * 8 >> 20} MiB + filler {fill.words * 8 >> 20} MiB challenged; dynamic region {region.nbytes >> 20} MiB "
        f"({len(model.layers)} layers of int8 K/V + scales over {args.max_len} slots, float64 activation buffer, pad"
        f"{', encoded' if not args.plain_dynamic else ''}) in {region.nseg} segments; "
        f"responder and stage records on :{args.port}",
        flush=True,
    )
    static_bytes = (sd.theta.numel() + fill.words) * 8
    print(hbm_line(dev, static_bytes, region.nbytes), flush=True)
    loop = InferenceLoop(
        model,
        tracker,
        pause,
        step_sleep=args.step_sleep_ms / 1000,
        prompts=prompts,
        new_tokens=args.new_tokens,
        stop_ids=stop,
        region=region,
        request_gap=args.request_gap_ms / 1000,
    )
    loop.start()
    try:
        report_forever(loop, resp, dyn, pause, dev, static_bytes, region, show)
    except KeyboardInterrupt:
        print(f"\nstopping: {resp.served} challenges served", flush=True)
    finally:
        loop.stop()
        resp.stop()
        resp.join(timeout=2)


if __name__ == "__main__":
    main()
