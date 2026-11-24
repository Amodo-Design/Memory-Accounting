#!/usr/bin/env bash
# November 17 timing comparison, using the registered grouped 89-bit setup.
set -euo pipefail
cd "$(dirname "$0")/.."

tag=""
plan_only=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --tag) tag="${2:?missing tag}"; shift 2 ;;
        --plan) plan_only=1; shift ;;
        *) echo "Usage: scripts/timing.sh --tag TAG [--plan]" >&2; exit 2 ;;
    esac
done
[[ "$tag" =~ ^[a-z0-9_]+$ ]] || { echo "Provide --tag with a new lowercase run name" >&2; exit 2; }
postures=honest,encode_otf@0.01,partial@0.01,other_gpu@0.01,dram_copy@0.01,cpu@0.01
export POSM_SIM_CHEATS="cpu,$postures"
export POSM_REGISTER_WEIGHTS=1 POSM_NO_OPEN=1
export POSM_N=128 POSM_DYN_N=64 POSM_B=1M POSM_DYN_B=1M POSM_R=2048 POSM_K=1024
export POSM_INTERVAL=.125 POSM_CHECK_FRACTION=.02 POSM_WARMUP=200 POSM_CALIBRATE=3000
export POSM_CLAIM_F=.01 POSM_CLAIM_C=.999 POSM_CLAIM_MODE=window POSM_CLAIM_DRIFT=0
export POSM_CLAIM_MAX_FILL=4200
export POSM_LIVE_EXTRA="--claim-delta-c 15.9 --claim-delta-c-dyn 16.4 --dynamic-check-fraction .02"

out="results/timing_$tag"
plan="p89_group64, registered weights, 1 MiB blocks, 128 static / 64 dynamic links
Matrix: 1024 x 2048, 0.125 s configured interval per kind, driver RTT requested
200 warm-up and 3000 calibration attempts per kind, 2% correctness sample per kind
23 rounds x 40 s per condition, 2 s excluded after each switch
Postures: $postures
Results: $out"
printf '%s\n' "$plan"
[ "$plan_only" -eq 0 ] || exit 0
mkdir -p results
mkdir "$out"
printf '%s\n' "$plan" > "$out/plan.txt"

verifier=${POSM_VERIFIER:-verifier-host}
verifier_dir=${POSM_VERIFIER_DIR:-LVHEM_MACC_SW}
[[ "$verifier_dir" =~ ^[a-zA-Z0-9_./-]+$ ]] || { echo "Invalid verifier checkout path" >&2; exit 2; }
port=${POSM_UI_PORT:-8765}
[[ "$port" =~ ^[0-9]+$ ]] || { echo "Invalid verifier UI port" >&2; exit 2; }
trap 'bash scripts/demo.sh stop >> "$out/stop.log" 2>&1' EXIT
bash scripts/demo.sh start sim | tee "$out/start.log"
ssh -o BatchMode=yes "$verifier" "cd $verifier_dir &&
  python3 -u scripts/posture_cycle.py --dir . --name $tag --rounds 23 --slot 40 \
  --postures $postures --url http://127.0.0.1:$port/posture" | tee "$out/cycle.log"
bash scripts/demo.sh stop > "$out/stop.log" 2>&1
trap - EXIT
for name in demo_live.jsonl demo_live.log baseline_demo.json; do
    ssh -o BatchMode=yes "$verifier" "cd $verifier_dir && cat $name" > "$out/$name"
done
weight_dir=$(sed -n 's/^Verifier weight registration and receipt: //p' "$out/start.log")
[[ "$weight_dir" =~ ^weights\.[[:alnum:]]+$ ]] || exit 1
for name in registration.json receipt.json; do
    ssh -o BatchMode=yes "$verifier" "cd $verifier_dir && cat $weight_dir/$name" > "$out/$name"
done

# Use the verifier's acknowledged switch markers, not the wall-clock target schedule.
python3 - "$out" "$postures" <<'PY'
import bisect
from collections import Counter
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
expected = set(sys.argv[2].split(","))
with (out / "demo_live.jsonl").open() as source:
    markers = [event for line in source if (event := json.loads(line))["type"] == "posture"]
if (len(markers) != 139 or markers[-1]["posture"] != "honest"
        or any(m["posture"] != m.get("requested") for m in markers)
        or any({m["posture"] for m in markers[i:i+6]} != expected for i in range(0, 138, 6))):
    raise SystemExit("Incomplete or incorrect six-condition switch schedule. See the raw verifier log.")
times = [m["t"] for m in markers]
counts = Counter()
with (out / "demo_live.jsonl").open() as source, (out / "retained-static.jsonl").open("x") as retained:
    for line in source:
        event = json.loads(line)
        if (event["type"] != "verification" or event.get("round") != "static"
                or event.get("phase") == "calibration" or event.get("rtt_ns") is None):
            continue
        slot = bisect.bisect_right(times, event["t"]) - 1
        if 0 <= slot < 138 and event["t"] - times[slot] > 2:
            event.update(posture=markers[slot]["posture"], slot=slot + 1)
            retained.write(json.dumps(event) + "\n")
            counts[event["posture"]] += 1
print(json.dumps(counts, sort_keys=True))
PY
