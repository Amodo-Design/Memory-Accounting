#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
: "${POSM_MODEL_SNAPSHOT:?path to the pinned Qwen model snapshot on the Dell}"
out="$PWD/results/timing-256-128x128-23-rounds"
test "$(cat results/timing-256-128x128-23-rounds.rc)" = 0
test -s "$out/demo_live.jsonl"
test -s "$out/verifier-run-path.txt"
test -f "$out/bundle/manifest.json"
verifier_run=$(cat "$out/verifier-run-path.txt")
mkdir -p "$out/evidence"
rsync -a --protect-args "$POSM_VERIFIER_SSH:$verifier_run/evidence/" "$out/evidence/"
.venv/bin/python - "$out" <<'PY'
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
checks = []
for line in (out / "demo_live.jsonl").open():
    event = json.loads(line)
    if event.get("type") == "check" and isinstance(event.get("dyn_check"), dict):
        check = event["dyn_check"]
        nonce = check.get("nonce")
        if not nonce or event.get("record_saved") is False or check.get("record_saved") is False:
            raise SystemExit("a sampled dynamic response has no retained record")
        checks.append(nonce)
if not checks:
    raise SystemExit("no sampled dynamic records to replay")
missing = sorted(set(checks) - {p.stem for p in (out / "evidence").glob("*.json")})
if missing:
    raise SystemExit(f"{len(missing)} sampled dynamic records are missing, first {missing[0]}")
print(f"Saved records cover all {len(checks)} sampled dynamic checks", flush=True)
PY
seed=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["filler_seed"])' "$out/seed-receipt.json")
.venv/bin/python -u -m memory_accounting.verifier.vllm_replay replay \
    --model "$POSM_MODEL_SNAPSHOT" --evidence "$out/evidence" \
    --log "$out/demo_live.jsonl" --manifest "$out/bundle/manifest.json" \
    --dyn-B 1M --dyn-n 128 --partial --max-model-len 2048 \
    --gpu-memory-utilization 0.75 \
    --filler-seed "$seed" --pad-seed "$seed" --gpu 0 \
    | tee "$out/dynamic-replay.log"
if grep -E -q 'MISMATCH|FAULT' "$out/dynamic-replay.log"; then
    echo "At least one sampled dynamic record did not replay exactly" >&2
    exit 1
fi
grep -q 'records reproduced exactly and match the verifier' "$out/dynamic-replay.log"
