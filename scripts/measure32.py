"""Qwen2.5-32B under vLLM + PoSM: GPU memory by item at each stage, then load phases while a verifier on another host challenges it."""
import json
import os
import time

from memory_accounting.models.vllm import apply_env, settings

apply_env()
from vllm import LLM, SamplingParams  # noqa: E402

MODEL, UTIL, DYN_B, PORT = "Qwen/Qwen2.5-32B-Instruct", 0.75, "1M", 5558
MODEL_REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
ENC_KEY = os.environ["WEIGHT_KEY"]
SEED = os.environ["FILLER_SEED"]
VERIFIER = os.environ["VERIFIER_ADDRESS"]
OUT = os.environ["MEMORY_OUT"]
LOAD_CONCURRENCY = int(os.environ.get("MEMORY_LOAD_CONCURRENCY", "256"))
IDLE_SECONDS = float(os.environ.get("MEMORY_IDLE_SECONDS", "40"))
LIGHT_SECONDS = float(os.environ.get("MEMORY_LIGHT_SECONDS", "40"))
SELECTED_SECONDS = float(os.environ.get("MEMORY_SELECTED_SECONDS", "70"))
GO, DONE = OUT + "/go", OUT + "/done"
os.makedirs(OUT, exist_ok=True)
for f in (GO, DONE):
    if os.path.exists(f):
        os.remove(f)


def info(llm):
    i = llm.collective_rpc("posm_info")[0]
    d = i.get("dynamic") or {}
    return {"static_served": i.get("served"), "armed": i.get("armed"), "dyn": {k: d.get(k) for k in ("served", "mid_step", "between_steps", "idle", "held_off", "static_freezes", "slow_freezes")}}


def probe(llm, label, with_info=False, **extra):
    p = llm.collective_rpc("posm_mem_probe", args=(label,))[0]
    p.update(extra)
    p["t"] = time.time()
    if with_info:
        p["info"] = info(llm)
    print("PROBE " + json.dumps(p, default=str), flush=True)
    return p


def ids(n, salt):
    return [1000 + (salt * 7919 + j * 31) % 50000 for j in range(n)]


t0 = time.perf_counter()
kw = settings(MODEL, 2048, UTIL)
kw["revision"] = MODEL_REVISION
kw["max_num_seqs"] = LOAD_CONCURRENCY
kw["worker_extension_cls"] = "measure_ext.MeasureExt"
llm = LLM(**kw)
probe(llm, "1 vLLM started")
sp = SamplingParams(temperature=0.0, max_tokens=32)
vanilla = [o.outputs[0].text for o in llm.generate(["The quick brown fox", "In 1969, the first"], sp, use_tqdm=False)]
probe(llm, "2 after a first generate")
print("dynamic:", llm.collective_rpc("posm_dynamic_setup", args=(DYN_B, SEED, False), kwargs={"gpu": 0, "encoded": True, "filler_seed_hex": SEED})[0], flush=True)
probe(llm, "3 dynamic region set up")
# Keep enough headroom for the 1 GiB export chunk and the measured load phase;
# the previous successful run reserved about 4.16 GiB.
print("converted:", llm.collective_rpc("posm_convert", args=(ENC_KEY, SEED, 0.0), kwargs={"filler_max_bytes": None, "reserve_bytes": 17 << 28, "gpu": 0})[0], flush=True)
probe(llm, "4 weights encoded, filler filled")
after = [o.outputs[0].text for o in llm.generate(["The quick brown fox", "In 1969, the first"], sp, use_tqdm=False)]
if after != vanilla:
    raise RuntimeError("weight conversion changed the reference output")
print("output unchanged:", True, flush=True)
if after != vanilla:
    raise RuntimeError("Encoded model output differed before memory measurement")
probe(llm, "4b after a generate on the encoded model")
print("bundle:", llm.collective_rpc("posm_export", args=(OUT + "/bundle", SEED), kwargs={"gpu": 0})[0], flush=True)
probe(llm, "4c bundle exported")
print(llm.collective_rpc("posm_serve", args=(PORT, None, "honest", None, True, 100.0), kwargs={"verifier_addrs": [VERIFIER]})[0], flush=True)
probe(llm, "5 responder started, no challenges yet", with_info=True)
llm.collective_rpc("posm_mem_reset")
llm.generate([{"prompt_token_ids": list(range(1000, 3000))}], SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)
probe(llm, "6 after a 2000-token prefill (peak since 5)")
llm.collective_rpc("posm_mem_reset")
llm.generate([{"prompt_token_ids": ids(8, i)} for i in range(16)], SamplingParams(temperature=0.8, seed=1, max_tokens=64), use_tqdm=False)
probe(llm, "7 after 16 requests of 64 tokens, sampled (peak since 6)")
print(f"SETUP_DONE in {time.perf_counter() - t0:.0f} s", flush=True)


def phase(name, seconds, work, until=None):
    """Repeat `work` (requests, prompt tokens, new tokens, sampling) for `seconds`, or until the file `until` exists."""
    llm.collective_rpc("posm_mem_reset")
    before, w0, p0, toks, rounds = info(llm), time.time(), time.perf_counter(), 0, 0
    print(f"PHASE_START {json.dumps({'name': name, 't': w0})}", flush=True)
    while (not os.path.exists(until)) if until else (time.perf_counter() - p0 < seconds):
        if work is None:
            time.sleep(1.0)
        else:
            n, plen, new, spk = work
            outs = llm.generate([{"prompt_token_ids": ids(plen, i + rounds * 4096)} for i in range(n)], SamplingParams(max_tokens=new, ignore_eos=True, **spk), use_tqdm=False)
            toks += sum(len(o.outputs[0].token_ids) for o in outs)
            rounds += 1
    dt = time.perf_counter() - p0
    print(f"WORK {name}: {rounds} batches, {toks} tokens out in {dt:.1f} s", flush=True)
    probe(llm, name, with_info=True, seconds=dt, tokens_out=toks, batches=rounds, before=before, t_start=w0, t_end=time.time())


S = dict(temperature=0.8, seed=1)
LIGHT = (16, 256, 48, S)
SELECTED = (LOAD_CONCURRENCY, 256, 48, S)
# Run the selected load phase last so an unexpected allocation failure retains the earlier probes.
phase(f"8 selected load while both verifiers calibrate ({LOAD_CONCURRENCY} requests, 256-token prompts, 48 new tokens)",
      0, SELECTED, until=GO)
phase("9 idle, challenges only", IDLE_SECONDS, None)
phase("10 light load (16 requests, 256-token prompts, 48 new tokens)", LIGHT_SECONDS, LIGHT)
phase(f"11 A: {LOAD_CONCURRENCY} requests, 256-token prompts, 48 new tokens, sampled", SELECTED_SECONDS,
      SELECTED)
print("PHASES_DONE", flush=True)
