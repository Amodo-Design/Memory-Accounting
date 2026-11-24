#!/usr/bin/env bash
# The timing analysis, end to end, from the laptop: the static and dynamic parameter sweeps on the GPU host, the choice
# of challenge shapes (memory_accounting.experiments.select), the overnight labelled run with those shapes (scripts/demo.sh start sim
# on the GPU host and the verifier host, scripts/posture_cycle.py on the verifier host), and the figures from it.
# First run for v0.17 (as timing_v017.sh).
#
#   scripts/timing.sh --tag appendix [--plan]     # the fixed 23-round November 17 comparison on grouped p89
#   scripts/timing.sh --sweep --tag v019 [--until HH:MM]   # legacy sweep and deadline-based run
#       [--interval S]                             # mean seconds between rounds of each kind (default 0.5)
#       [--postures honest,encode_otf@0.01,...]     # the postures cycled (default: honest, every cheat at 1%, dyn_dram)
#
# Everything it decides and runs goes to results/timing_<tag>/: commands.log (every remote command, as run), the sweeps'
# selection (selection.json, .md), run_config.json (the shapes, every POSM_* setting, the verifier's and the prover's
# command lines as they started, versions, git state), a snapshot of the code that ran (code_snapshot.tar.gz), the
# posture cycle's log, and summary.json (the claims the verifier set); the sweeps to results/sweep_<tag>/. The night's
# log goes to ../posm_runs/<tag>_live (beside the repository)/run.jsonl and the figures to docs/figures/<tag>/ (the tag
# must be one of scripts/figures.py's LIVE_SETS).
set -uo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" != --sweep ]; then
    exec bash scripts/timing_appendix.sh "$@"
fi
shift
TAG=
UNTIL=09:00
INTERVAL=0.5
POSTURES=honest,encode_otf@0.01,partial@0.01,other_gpu@0.01,dram_copy@0.01,cpu@0.01,dyn_dram
SKIP_SWEEPS=0
FINISH_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG=$2; shift 2 ;;
        --until) UNTIL=$2; shift 2 ;;
        --interval) INTERVAL=$2; shift 2 ;;
        --postures) POSTURES=$2; shift 2 ;;
        --skip-sweeps) SKIP_SWEEPS=1; shift ;;          # the sweeps' results are already in results/sweep_<tag>/
        --finish-only) FINISH_ONLY=1; shift ;;          # the run is going: wait for its end, then the log, figures, claims
        *) echo "usage: scripts/timing.sh --tag TAG [--until HH:MM] [--interval S] [--postures LIST] [--skip-sweeps] [--finish-only]" >&2; exit 2 ;;
    esac
done
[[ "$TAG" =~ ^[a-z0-9]+$ ]] || { echo "--tag: a name such as v019 (results/timing_<tag>/, docs/figures/<tag>/)" >&2; exit 2; }
OUT=results/timing_$TAG
SW=results/sweep_$TAG
RUNS=${POSM_RUNS:-../posm_runs}                                   # the saved runs, beside the repository
PROVER=${POSM_PROVER:-gpu-host}                                 # ssh names of the two hosts, as for scripts/demo.sh
VERIFIER=${POSM_VERIFIER:-verifier-host}
PROVER_ADDR=${POSM_PROVER_ADDR:-$PROVER}                          # the GPU host as the verifier host reaches it
VERIFIER_ADDR=${POSM_VERIFIER_ADDR:-$VERIFIER}                    # the verifier host as the GPU host reaches it
mkdir -p $OUT $SW
CMDLOG=$OUT/commands.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/progress.log; }
fail() { log "FAILED: $*"; exit 1; }
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
export RSYNC_RSH="ssh $SSH_OPTS"
on() {   # host command: logged, then run (a dropped connection fails it, never hangs it)
    echo "$(date -u +%FT%TZ) [$1] $2" >> $CMDLOG
    ssh $SSH_OPTS "$1" "$2"
}
quiet() {   # a short status check, retried through a connection drop (not logged)
    for t in 1 2 3 4 5 6; do out=$(ssh $SSH_OPTS "$1" "$2" 2>/dev/null) && { echo "$out"; return 0; }; sleep 20; done; return 1
}
on_long() {   # a long job on a host: started detached (it survives a dropped connection), then waited for; its exit code
    local host=$1 tag=$2 cmd=$3
    echo "$(date -u +%FT%TZ) [$host, detached] $cmd" >> $CMDLOG
    ssh $SSH_OPTS "$host" "cd LVHEM_MACC_SW && rm -f .job_$tag.rc && (setsid nohup bash -c 'source ~/posm_env.sh; $cmd; echo \$? > .job_$tag.rc' < /dev/null > /dev/null 2>&1 &)" || return 255
    while true; do
        rc=$(quiet "$host" "cat LVHEM_MACC_SW/.job_$tag.rc 2>/dev/null") && [ -n "$rc" ] && return "$rc"
        sleep 30
    done
}
local_cmd() { echo "$(date -u +%FT%TZ) [local] $*" >> $CMDLOG; "$@"; }
PY=.venv/bin/python

if [ $FINISH_ONLY = 0 ]; then
# ---- 0. the code that runs, on both hosts
log "code snapshot and sync"
git rev-parse HEAD > $OUT/git_head.txt; git status --short > $OUT/git_status.txt; git diff > $OUT/git_diff.patch
tar czf $OUT/code_snapshot.tar.gz --exclude __pycache__ --exclude build memory_accounting scripts tests pyproject.toml
local_cmd rsync -a --exclude build --exclude __pycache__ memory_accounting tests scripts pyproject.toml $PROVER:LVHEM_MACC_SW/
local_cmd rsync -a --exclude build --exclude __pycache__ memory_accounting tests scripts pyproject.toml $VERIFIER:LVHEM_MACC_SW/
for h in $PROVER $VERIFIER; do on $h "rm -f LVHEM_MACC_SW/memory_accounting/build/libposm_cpu.so"; done      # rebuilt from the synced sources on first use
PROFILE_CHECK='from memory_accounting.core.codec_profile import PROFILE; print(PROFILE); assert PROFILE == "p89_group64"'
on $PROVER "cd LVHEM_MACC_SW && source ~/posm_env.sh && python -c '$PROFILE_CHECK'" || fail "prover is not running the grouped 89-bit profile"
on $VERIFIER "cd LVHEM_MACC_SW && .venv/bin/python -c '$PROFILE_CHECK'" || fail "verifier is not running the grouped 89-bit profile"

# ---- 0b. the dynamic sweep's harness, briefly, before anything long
[ $SKIP_SWEEPS = 1 ] || on_long $PROVER smoke "mkdir -p $SW && python -m memory_accounting.experiments.dynbench --geometry int --blocks 64K --links 8 --r 2048 --samples 5 --warm-seconds 0.5 --out $SW/smoke_int.json > $SW/smoke_int.log 2>&1 && python -m memory_accounting.experiments.dynbench --geometry vllm --pool-gib 2 --blocks 1M --links 8 --r 2048 --samples 5 --warm-seconds 0.5 --out $SW/smoke_vllm.json > $SW/smoke_vllm.log 2>&1"
rc=$?; [ $SKIP_SWEEPS = 1 ] || { log "dynamic sweep smoke runs: exit $rc"; [ $rc = 0 ]; } || fail "dynamic sweep smoke run (see $SW/smoke_*.log on $PROVER)"

# ---- 1. static sweep (the verifier's shape policy: 256K to 1M blocks, r >= 2048, k = 1024)
STATIC="--size 4G --encoding vde --blocks 256K,512K,1M --links 16,32,64,128,256 --r 2048,4096 --k 1024 --provers encode_otf,dram_mapped,dram_copy,cpu --samples 100"
log "static sweep: $STATIC"
[ $SKIP_SWEEPS = 1 ] && log "(skipped: --skip-sweeps)" || on_long $PROVER static "mkdir -p $SW && python -m memory_accounting.experiments.bench $STATIC --out $SW/static.json > $SW/static.log 2>&1"
rc=$?; log "static sweep: exit $rc"; [ $rc = 0 ] || fail "static sweep (see $SW/static.log on $PROVER)"
local_cmd rsync -a $PROVER:LVHEM_MACC_SW/$SW/ $SW/
R=$($PY -c "
import json, sys; sys.argv = ['x']
from memory_accounting.experiments import select as S
import argparse
a = argparse.Namespace(F=0.01, C=0.999, jitter_us=15.0, budget=0.05, max_rate=2.0, static_cost_us=300.0, dyn_cost_us=1500.0, min_blocks=4, n_cal=3000)
print(S.rank_static('$SW/static.json', a)[0]['r'])")
[ -n "$R" ] || fail "no static choice"
log "static choice r = $R"

# ---- 2. dynamic sweeps at that r: the integer model's region (the night's prover) and vLLM's (for the record)
DYN_INT="--geometry int --int-model models/smollm2-135m.int.safetensors --max-len 512 --live-tokens 256 --blocks 16K,64K,256K,1M --links 4,8,16,32,64 --r $R --samples 200"
DYN_VLLM="--geometry vllm --pool-gib 16 --requests 16 --live-tokens 384 --blocks 64K,256K,1M,4M --links 4,8,16,32,64 --r $R --samples 200"
log "dynamic sweep (int): $DYN_INT"
[ $SKIP_SWEEPS = 1 ] && log "(skipped: --skip-sweeps)" || on_long $PROVER dynint "python -m memory_accounting.experiments.dynbench $DYN_INT --out $SW/dyn_int.json > $SW/dyn_int.log 2>&1"
rc=$?; log "dynamic sweep (int): exit $rc"; [ $rc = 0 ] || fail "dynamic sweep (int)"
log "dynamic sweep (vllm): $DYN_VLLM"
[ $SKIP_SWEEPS = 1 ] && log "(skipped: --skip-sweeps)" || on_long $PROVER dynvllm "python -m memory_accounting.experiments.dynbench $DYN_VLLM --out $SW/dyn_vllm.json > $SW/dyn_vllm.log 2>&1"
rc=$?; log "dynamic sweep (vllm): exit $rc"; [ $rc = 0 ] || fail "dynamic sweep (vllm)"
local_cmd rsync -a $PROVER:LVHEM_MACC_SW/$SW/ $SW/

# ---- 3. the choice
SEL="--static $SW/static.json --dynamic $SW/dyn_int.json --dynamic $SW/dyn_vllm.json --F 0.01 --C 0.999 --jitter-us 15 --budget 0.05 --max-rate 2 --n-cal 3000"
local_cmd $PY -m memory_accounting.experiments.select $SEL --out $OUT/selection.json --md $OUT/selection.md > /dev/null
cat > $OUT/.choice.py <<'EOF'
import json, math, os
c = json.load(open(os.environ["SELF"]))["choice"]
s, d = c["static"], c["dynamic_int"]
blk = lambda b: f"{b >> 20}M" if b >= 1 << 20 and b % (1 << 20) == 0 else f"{b >> 10}K"
print(f"B={blk(s['B'])} N_LINKS={s['n']} R={s['r']} DC={math.floor(s['delta_c_us'] * 10) / 10}")
print(f"DYN_B={blk(d['B'])} DYN_N={d['n']} DCD={max(0.1, math.floor(d['delta_c_us'] * 10) / 10)}")
print(f"CHECK={min(1.0, round(2.56 / s['n'], 4))}")
EOF
SELF=$OUT/selection.json $PY $OUT/.choice.py > $OUT/choice.env || fail "reading the choice"
. $OUT/choice.env
[ -n "${DYN_N:-}" ] || fail "no dynamic choice"
log "choice: static B=$B n=$N_LINKS r=$R Δc=$DC µs; dynamic B=$DYN_B n=$DYN_N Δc=$DCD µs; digest-check fraction $CHECK"

# ---- 4. the night: the sim profile (integer model, every cheat at 1%) with those shapes
CHEATS=$(echo "honest,$POSTURES" | tr , '\n' | awk '!seen[$0]++' | paste -sd , -)
[[ ",$POSTURES," == *,cpu@* ]] && CHEATS=$(echo "$CHEATS" | sed 's/^honest,/honest,cpu,/')   # cpu@f needs cpu itself, for its thread pool
ENVS=(POSM_PROVER=$PROVER POSM_PROVER_ADDR=$PROVER_ADDR POSM_VERIFIER=$VERIFIER POSM_VERIFIER_ADDR=$VERIFIER_ADDR
      POSM_SIM_CHEATS=$CHEATS POSM_INTERVAL=$INTERVAL
      POSM_N=$N_LINKS POSM_B=$B POSM_R=$R POSM_K=1024 POSM_DYN_B=$DYN_B POSM_DYN_N=$DYN_N
      POSM_CLAIM_F=0.01 POSM_CLAIM_C=0.999 POSM_CALIBRATE=3000 POSM_CHECK_FRACTION=$CHECK
      "POSM_LIVE_EXTRA=--claim-delta-c $DC --claim-delta-c-dyn $DCD" POSM_NO_OPEN=1)
log "demo start: ${ENVS[*]} scripts/demo.sh start sim"
echo "$(date -u +%FT%TZ) [local] env ${ENVS[*]} scripts/demo.sh start sim" >> $CMDLOG
env "${ENVS[@]}" scripts/demo.sh start sim > $OUT/demo_start.log 2>&1
rc=$?; log "demo start: exit $rc"; [ $rc = 0 ] || fail "demo start (see $OUT/demo_start.log)"
on $VERIFIER "cd LVHEM_MACC_SW && touch .${TAG}_cycle_start && (setsid nohup python3 scripts/posture_cycle.py --until $UNTIL --name ${TAG}_live --postures $POSTURES > ${TAG}_cycle.log 2>&1 < /dev/null &)"
log "posture cycle started (until $UNTIL on $VERIFIER)"

# ---- 5. what started, exactly
sleep 60
VCMD=$(quiet $VERIFIER "ps -eo args | grep -E '^.venv/bin/python -m memory_accounting.verifier.net live' | head -1")
PCMD=$(quiet $PROVER "ps -eo args | grep -E '^python -m memory_accounting.prover.intserve' | head -1")
$PY - <<EOF
import json, subprocess, platform
cfg = {"started_utc": "$(date -u +%FT%TZ)", "until_on_verifier_host": "$UNTIL", "postures": "$POSTURES".split(","), "slot_s": 40, "interval_s": $INTERVAL,
       "static": {"B": "$B", "n": $N_LINKS, "r": $R, "k": 1024, "claim_delta_c_us": $DC},
       "dynamic": {"B": "$DYN_B", "n": $DYN_N, "claim_delta_c_us": $DCD},
       "claim": {"F": 0.01, "C": 0.999, "calibrate": 3000, "mode": "window"}, "check_fraction": $CHECK,
       "env": """${ENVS[*]}""".split(" POSM_"), "verifier_command": """$VCMD""", "prover_command": """$PCMD""",
       "selection": json.load(open("$OUT/selection.json"))["choice"], "selection_assumptions": json.load(open("$OUT/selection.json"))["assumptions"],
       "sweeps": {"static": "$STATIC", "dynamic_int": "$DYN_INT", "dynamic_vllm": "$DYN_VLLM", "select": "$SEL"},
       "git_head": open("$OUT/git_head.txt").read().strip(), "code": "code_snapshot.tar.gz (memory_accounting, scripts, tests as synced to both hosts)",
       "hosts": {"prover": "$PROVER ($PROVER_ADDR)", "verifier": "$VERIFIER ($VERIFIER_ADDR)"}}
json.dump(cfg, open("$OUT/run_config.json", "w"), indent=1)
EOF
log "run config written"

fi

# ---- 6. the morning: the night's log, its figures, the claims (posture_cycle.py copies the log to runs/ when it ends)
RUNLOG=
for i in $(seq 9000); do   # this run's log: newer than its posture cycle's start (an earlier run's may be there too)
    RUNLOG=$(quiet $VERIFIER "find LVHEM_MACC_SW/runs -name '${TAG}_live_*.jsonl' -newer LVHEM_MACC_SW/.${TAG}_cycle_start 2>/dev/null | sort | tail -1" | grep '\.jsonl$')
    [ -n "$RUNLOG" ] && break
    sleep 20
done
[ -n "$RUNLOG" ] || fail "no run log after waiting"
log "run log $RUNLOG"
mkdir -p $RUNS/${TAG}_live
for t in 1 2 3 4 5 6; do local_cmd rsync -a --partial $VERIFIER:"$RUNLOG" $RUNS/${TAG}_live/run.jsonl && break; sleep 30; done   # through a dropped connection
local_cmd rsync -a $VERIFIER:LVHEM_MACC_SW/${TAG}_cycle.log $OUT/posture_cycle.log
local_cmd env POSM_PROVER=$PROVER POSM_PROVER_ADDR=$PROVER_ADDR POSM_VERIFIER=$VERIFIER POSM_VERIFIER_ADDR=$VERIFIER_ADDR scripts/demo.sh stop > $OUT/demo_stop.log 2>&1
local_cmd uv run --no-sync python scripts/figures.py extract $RUNS/${TAG}_live/run.jsonl $RUNS/${TAG}_live/run.npz
local_cmd uv run --no-sync python scripts/figures.py all --live $TAG --only main_m6_1pct,main_m8_1pct,main_m9,main_m12 > $OUT/figures.log 2>&1
$PY - <<EOF
import json, numpy as np
d = np.load("$RUNS/${TAG}_live/run.npz", allow_pickle=True)
json.dump(json.loads(str(d["meta"])), open("$OUT/summary.json", "w"), indent=1)
EOF
log "done: figures in docs/figures/$TAG/, claims in $OUT/summary.json"
