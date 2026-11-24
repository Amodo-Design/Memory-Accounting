"""Stream-decode + responder on a plain torch model, without vLLM (for testing on a GPU box):

  python -m memory_accounting.prover.serve --model <hf model dir or name> --enc-key <hex> --filler-seed <hex> --port 5555

Loads the model with transformers (bf16, single GPU), converts it to Θ' + scratch, fills free HBM
with the verifier-seeded filler, starts the responder, then keeps generating tokens in a loop so the
GPU is under a decode load while challenges arrive.

With dynamic memory (the default): generation runs on a static KV cache whose buffers, together
with an activation buffer written at every layer boundary, form the dynamic region. A dynamic
challenge freezes generation at the next layer or step boundary, chains over the frozen region, and
keeps the stage record for the verifier, which replays to it and recomputes the chain
(`memory_accounting.verifier.consistency --model`). A float model replays bit-exactly only on the same GPU type and
kernels; for a CPU verifier use the integer-exact model (`memory_accounting.prover.intserve`). `--no-dynamic`
gives the static-only prover.
"""

from __future__ import annotations

import argparse
import threading
import time

import torch


def gate_forward(module: torch.nn.Module, gate) -> None:
    """Run every forward of `module` under `gate` (the responder's drain lock): taken before the forward,
    released after it, also when it raises (always_call). Released only by the forward that took it: a
    failure in an earlier pre-hook must not release a gate the responder holds."""
    held = threading.local()

    def pre(m, a):
        gate.acquire()
        held.v = True
        # a pre-hook's return value replaces the args: return None

    def post(m, a, o):
        if getattr(held, "v", False):
            held.v = False
            gate.release()

    module.register_forward_pre_hook(pre)
    module.register_forward_hook(post, always_call=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--enc-key", required=True)
    ap.add_argument(
        "--filler-seed", required=True, help="hex seed supplied by the verifier (memory_accounting.setup.cli filler-seed)"
    )
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--scratch-fraction", type=float, default=0.0)
    ap.add_argument(
        "--scratch-gb",
        type=float,
        default=2.0,
        help="memory left unfilled for the model's own use: the prover's scratch",
    )
    ap.add_argument(
        "--no-drain",
        action="store_true",
        help="answer challenges concurrently with inference instead of between forward calls",
    )
    ap.add_argument("--no-load", action="store_true", help="do not run inference in the background")
    ap.add_argument("--prompt", default="The quick brown fox")
    ap.add_argument("--bundle", default="verifier_bundle", help="where to write the verifier's image + manifest")
    ap.add_argument(
        "--strategies",
        default="honest",
        help="comma list: honest,dram_mapped,dram_copy,cpu,encode_otf,other_gpu (cheats need host RAM / GPU 1)",
    )
    ap.add_argument("--canon-device", default="cuda:1", help="where the other_gpu cheat's canonical copy lives")
    ap.add_argument(
        "--no-dynamic", action="store_true", help="static-only prover: no KV cache / activation region, no pause point"
    )
    ap.add_argument(
        "--max-cache-len",
        type=int,
        default=512,
        help="static KV cache length (slots per layer); the whole cache is challenged",
    )
    ap.add_argument("--dyn-block", default="64K", help="block size the dynamic region is padded to and challenged with")
    ap.add_argument("--pad-seed", default=None, help="hex seed of the region's filler pad (default: the filler seed)")
    ap.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="this GPU's index as the verifier knows it: keys the dynamic filler, names the bundle's GPU entry "
        "(default: the device's index)",
    )
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument(
        "--evidence-bytes",
        action="store_true",
        help="attach the region's bytes to each record (debugging a replay mismatch)",
    )
    args = ap.parse_args(argv)
    from .responder import require_free_ports

    require_free_ports(args.port, tcp=not args.no_dynamic)  # before the model loads
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..core.params import Keys, parse_bytes
    from .filler import allocate_filler
    from .responder import ResidentArray, Responder
    from .stream_decode import convert

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = tok(args.prompt, return_tensors="pt").to("cuda")
    gpu = args.gpu if args.gpu is not None else torch.cuda.current_device()

    def bench(label, steps=3, new=64):
        with torch.no_grad():
            model.generate(**ids, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(steps):
            with torch.no_grad():
                model.generate(**ids, max_new_tokens=new, do_sample=False)
        torch.cuda.synchronize()
        rate = steps * new / (time.perf_counter() - t)
        print(f"{label}: {rate:.1f} tok/s", flush=True)
        return rate

    with torch.no_grad():
        ref = model.generate(**ids, max_new_tokens=32, do_sample=False)
    vanilla_rate = bench("vanilla")
    originals = {n: p.detach().clone() for n, p in model.named_parameters()}
    sd = convert(model, int(args.enc_key, 16))
    torch.cuda.synchronize()
    bad = sd.check_parameters(originals)
    del originals
    print(
        "decoded parameters match originals byte for byte:",
        not bad,
        (f"mismatch in {bad[:5]}" if bad else ""),
        flush=True,
    )
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=32, do_sample=False)
    print(
        "stream-decode output matches vanilla:",
        torch.equal(ref, out),
        "|",
        tok.decode(out[0], skip_special_tokens=True),
        flush=True,
    )
    sd_rate = bench("stream-decode")
    print(f"overhead: {vanilla_rate / sd_rate:.2f}x", flush=True)

    # ---- dynamic region: static KV cache + activation buffer, hooks at every layer boundary
    dyn = None
    cache = None
    tracker = None
    keys = Keys.from_seed(int(args.filler_seed, 16))  # the challenge-hash keys: from the verifier's seed
    if not args.no_dynamic:
        from transformers import StaticCache

        from .dynamic import (
            DeviceSync,
            DynamicResponder,
            Hooks,
            PauseController,
            StageTracker,
            act_buffer,
            free_all,
            hf_cache_buffers,
            layer_modules,
            make_region,
        )

        cache = StaticCache(model.config, max_cache_len=args.max_cache_len)
        with torch.no_grad():  # one step initialises the (lazy) cache buffers
            model.generate(**ids, max_new_tokens=2, do_sample=False, past_key_values=cache)
        cache.reset()
        torch.use_deterministic_algorithms(True, warn_only=True)  # same-device replay needs run-to-run determinism
        layers = layer_modules(model)
        act = act_buffer(args.max_cache_len, model.config.hidden_size, model.dtype, "cuda")
        pad_seed = int(args.pad_seed, 16) if args.pad_seed else int(args.filler_seed, 16)
        region = make_region(hf_cache_buffers(cache), act, parse_bytes(args.dyn_block), pad_seed, gpu=gpu)
        tracker = StageTracker(len(layers), act)
        pause = PauseController()
        Hooks(model, layers, act, tracker, pause, on_pause=DeviceSync("cuda"), region=region)
        dyn = DynamicResponder(
            region,
            tracker,
            pause,
            keys,
            include_bytes=args.evidence_bytes,
            model_info={
                "kind": "hf",
                "name": args.model,
                "dtype": str(model.dtype).replace("torch.", ""),
                "max_cache_len": args.max_cache_len,
                "gpu": gpu,
            },
        )
        with torch.no_grad():
            # free slots hold filler (finite: HF attends over them, masked)
            free_all(model, region, tracker, pause, 0, reset=cache.reset)
            out2 = model.generate(**ids, max_new_tokens=32, do_sample=False, past_key_values=cache)
            free_all(model, region, tracker, pause, 1, reset=cache.reset)
        print(
            f"static-cache output matches vanilla: {torch.equal(ref, out2)}; dynamic region {region.nbytes >> 20} MiB in {region.nseg} segments "
            f"({len(layers)} layers x K,V of {args.max_cache_len} slots + activation buffer + pad), {region.num_blocks(region.block_bytes)} blocks of {region.block_bytes >> 10} KB",
            flush=True,
        )

    fill = allocate_filler(
        int(args.filler_seed, 16),
        sd.enc_key,
        sd.payload_words,
        args.scratch_fraction,
        "cuda",
        reserve_bytes=int(args.scratch_gb * (1 << 30)),
    )
    strategies = tuple(s for s in args.strategies.split(",") if s)
    resident = ResidentArray(
        sd.segments() + fill.segments(), sd.enc_key, "cuda", strategies=strategies,
        canon_device=args.canon_device,
        segment_keys=(sd.enc_key, fill.key) if fill.words else (sd.enc_key,),
    )
    bundle = sd.export_bundle(
        args.bundle, filler=fill, filler_seed=int(args.filler_seed, 16), model_name=args.model, gpu=gpu
    )
    print("verifier bundle:", bundle, flush=True)
    resp = Responder(resident, keys, port=args.port, strategies=strategies, drain=not args.no_drain, dynamic=dyn)
    if not args.no_drain:  # each forward runs under the gate: an armed challenge waits for it
        gate_forward(model, resp.gate)
    resp.start()
    from .responder import watch_gc

    watch_gc()
    print(
        f"Θ' {sd.theta.numel() * 8 >> 20} MiB + filler {fill.words * 8 >> 20} MiB challenged; weights in plaintext {sd.plain_bytes >> 20} MiB (scratch and resident); responder on :{args.port}"
        + (f"; dynamic region {dyn.region.nbytes >> 20} MiB, evidence over TCP :{args.port}" if dyn else ""),
        flush=True,
    )
    t0 = time.perf_counter()
    toks = 0
    rid = 0
    try:
        while True:
            if args.no_load:
                time.sleep(1)
                continue
            rid += 1
            with torch.no_grad():
                if cache is not None:
                    model.generate(**ids, max_new_tokens=args.new_tokens, do_sample=False, past_key_values=cache)
                    # the request ended: its memory is filler again
                    free_all(model, dyn.region, tracker, dyn.pause, rid + 1, reset=cache.reset)
                else:
                    model.generate(**ids, max_new_tokens=args.new_tokens, do_sample=False)
            toks += args.new_tokens
            if time.perf_counter() - t0 > 10:
                print(
                    f"{toks / (time.perf_counter() - t0):.1f} tok/s under stream-decode; {resp.served} challenges served"
                    + (
                        f"; dynamic {dyn.served} ({dyn.paused_in_flight} mid-step, {dyn.paused_between} between steps, {dyn.idle} idle)"
                        if dyn
                        else ""
                    ),
                    flush=True,
                )
                t0 = time.perf_counter()
                toks = 0
    except KeyboardInterrupt:
        print(f"\nstopping: {resp.served} challenges served", flush=True)
    finally:
        resp.stop()
        resp.join(timeout=2)


if __name__ == "__main__":
    main()
