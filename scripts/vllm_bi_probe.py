"""Does vLLM's batch-invariant mode give the properties the dynamic replay needs, on this GPU and backend?

  VLLM_BATCH_INVARIANT=1 VLLM_USE_V2_MODEL_RUNNER=0 python scripts/vllm_bi_probe.py --model Qwen/Qwen2.5-0.5B-Instruct --backend TRITON_ATTN

For a short prompt and one longer than a 128-token KV tile, greedy-decode 12 tokens with logprobs, then
  A  score each generated token by a prefill of the prompt plus the tokens before it (decode vs prefill),
  B  decode the same prompt inside a batch of 15 others of similar length (alone vs batched),
and count logprobs that are not bitwise equal. The replay in memory_accounting.verifier.vllm_replay needs both counts at 0.
Measured on an H200 NVL, vLLM 0.30.0: FLASH_ATTN 11/12 and 11/12 mismatches at 152 tokens (0 at 43);
TRITON_ATTN and FLEX_ATTENTION 0 everywhere.
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--backend", default="TRITON_ATTN")
    ap.add_argument("--words", default="40,140", help="prompt lengths in words")
    args = ap.parse_args()
    os.environ["POSM_VLLM_ATTN"] = args.backend
    from memory_accounting.models.vllm import apply_env, settings

    apply_env()
    from vllm import LLM, SamplingParams

    llm = LLM(**settings(args.model, 2048, 0.5))
    tok = llm.get_tokenizer()
    rng = random.Random(1)
    words = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "memory",
        "hash",
        "chain",
        "matrix",
        "verifier",
        "prover",
        "tile",
        "kernel",
    ]

    def prompt(n):
        return tok(" ".join(rng.choice(words) for _ in range(n)))["input_ids"]

    for n in (int(x) for x in args.words.split(",")):
        ids = prompt(n)
        sp = SamplingParams(temperature=0.0, max_tokens=12, logprobs=1)
        o = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
        gen = list(o.outputs[0].token_ids)
        dec = [o.outputs[0].logprobs[i][gen[i]].logprob for i in range(len(gen))]
        a_bad = 0
        for i in range(len(gen)):
            q = llm.generate(
                [{"prompt_token_ids": ids + gen[:i]}],
                SamplingParams(temperature=0.0, max_tokens=1, logprobs=1),
                use_tqdm=False,
            )[0]
            a_bad += int(q.outputs[0].token_ids[0] != gen[i] or q.outputs[0].logprobs[0][gen[i]].logprob != dec[i])
        others = [{"prompt_token_ids": prompt(n + rng.randint(-10, 10))} for _ in range(15)]
        ob = llm.generate(others[:7] + [{"prompt_token_ids": ids}] + others[7:], sp, use_tqdm=False)[7]
        same = list(ob.outputs[0].token_ids) == gen
        b_bad = sum(int(ob.outputs[0].logprobs[i][gen[i]].logprob != dec[i]) for i in range(len(gen))) if same else -1
        print(
            f"{args.backend}: prompt {len(ids)} tokens: decode-vs-prefill mismatches {a_bad}/{len(gen)}; "
            f"alone-vs-batch mismatches {b_bad}/{len(gen)}{'' if same else ' (tokens differ)'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
