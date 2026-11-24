#!/usr/bin/env bash
# One-command demo of the live verifier.
#
# Runs from your laptop. Starts the prover on the GPU host, the live verifier (and its page) on the
# verifier host over the LAN, and tunnels the page to http://localhost:8765 on this machine.
#
#   scripts/demo.sh start [int|vllm|sim]   start a demo (default: int)
#   scripts/demo.sh posture <name>          switch the prover: honest, evict, evict_dram, tamper, dyn_dram (sim: the old simulators)
#   scripts/demo.sh stop                    stop everything this script started
#   scripts/demo.sh status                  what is running, last prover and verifier lines
#   scripts/demo.sh open                    reopen the tunnel and the page
#   scripts/demo.sh replay                  vllm: stop the prover, replay its records on the GPU host
#
# Profiles (int and vllm fill the whole GPU; the prover starts honest and can be switched to a cheat
# from the page's "Prover simulation" panel or with `posture`):
#   int     integer-exact model; every dynamic record is replayed exactly on the verifier's CPU as you
#           watch; a verification fails if it leaves more than ${INT_MAX_SCRATCH} of the GPU unchallenged
#           or uncredited (--max-scratch; INT_MAX_SCRATCH in this script). (Aliases: honest, cheats.)
#   vllm    a real vLLM engine (Qwen2.5-0.5B, batch-invariant kernels); at most ${VLLM_MAX_SCRATCH} unchallenged
#           or uncredited (--max-scratch; VLLM_MAX_SCRATCH in this script); records are stored, not replayed
#           online (that needs a GPU of the same type): `replay` afterwards, with the verifier's log.
#   sim     the old whole-array simulators (dram_copy, cpu, encode_otf, partial, lora), which need room
#           for a second copy of the array: the filler is capped at 8 GiB, so most of the GPU is unaccounted.
# Cheats with the GPU full: evict (overwrite a fifth of the filler with the prover's own data, answer from
# what is there: wrong digests), evict_dram (keep that filler in host memory, read it over PCIe: drift),
# tamper (change a tenth of the weight blocks in place: the dynamic replay no longer matches), dyn_dram (keep
# the filler of half the dynamic memory, the top of the KV pool or cache, in host memory and read it over PCIe
# while it is free: slower dynamic links, POSM_DYN_EVICT_FRACTION).
# The dynamic chains read POSM_DYN_B blocks (default 1M for vllm, 64K for int: its region is about 10 MiB, too
# few 1 MiB blocks): the prover's region block and the verifier's --dyn-B, both set from it. At 64K dyn_dram is not
# seen (a link from host memory costs 0.3 to 1.9 µs); at 1M it costs 17 µs and the dynamic chain's claim catches it.
# Each start has the verifier mint a fresh filler seed (memory_accounting.setup.cli filler-seed), which the prover
# fills with and the verifier checks the bundle against.
#
# Hosts are ssh names: set POSM_PROVER, POSM_PROVER_ADDR, POSM_VERIFIER and POSM_VERIFIER_ADDR, or define gpu-host and
# verifier-host in ~/.ssh/config (the defaults).
set -uo pipefail

PROVER=${POSM_PROVER:-gpu-host}                                  # ssh name of the GPU host
PROVER_DIR=${POSM_PROVER_DIR:-LVHEM_MACC_SW}                     # repo copy on it
PROVER_ENV=${POSM_PROVER_ENV:-'source ~/posm_env.sh'}            # puts its venv python and CUDA toolkit on PATH
PROVER_ADDR=${POSM_PROVER_ADDR:-gpu-host}                        # the GPU host as the verifier host reaches it
VERIFIER=${POSM_VERIFIER:-verifier-host}                         # ssh name of the verifier host
VERIFIER_DIR=${POSM_VERIFIER_DIR:-LVHEM_MACC_SW}
VERIFIER_PY=${POSM_VERIFIER_PY:-.venv/bin/python}
VERIFIER_ADDR=${POSM_VERIFIER_ADDR:-verifier-host}               # the verifier host as the GPU host reaches it (bundle copy)
UI_PORT=${POSM_UI_PORT:-8765}                                    # page port, here and on the verifier host
GPU_TYPE=${POSM_GPU_TYPE:-H200-NVL}
LIVE_EXTRA=${POSM_LIVE_EXTRA:-}                                  # extra memory_accounting.verifier.net live options, e.g. "--sigmas 12" for fewer honest misses
INT_MODEL=models/smollm2-135m.int.safetensors
ENC_KEY=0123456789abcdef0123456789abcdef
WEIGHT_SETUP=${POSM_REGISTER_WEIGHTS:-0}
CHEATS=honest,evict,evict_dram,tamper,dyn_dram
EVICT_FRACTION=${POSM_EVICT_FRACTION:-0.2}                       # share of the GPU's filler the evict cheats reclaim
DYN_EVICT_FRACTION=${POSM_DYN_EVICT_FRACTION:-0.5}               # dyn_dram: share of the KV pool (int: of the cache's slots) kept in host memory
DYN_B=${POSM_DYN_B:-}                                             # dynamic block: the prover's region block and the verifier's --dyn-B (default per profile)
SIM_CHEATS=${POSM_SIM_CHEATS:-honest,dram_mapped,dram_copy,encode_otf,partial,lora,cpu}   # the sim profile's postures (<cheat>@f: the cheat on a share f of the array)
LIVE_TIMING="--timing kernel"                                     # kernel timestamps (no retry of misses: it gives a cheater slack)
INTERVAL=${POSM_INTERVAL:-0.5}                                    # mean seconds between rounds of each kind (the verifier's --interval)
CHECK_FRACTION=${POSM_CHECK_FRACTION:-1}                         # share of static rounds whose digests the verifier recomputes (1: all)
N_LINKS=${POSM_N:-8}                                              # reads per static challenge chain (the verifier's --n; the claim's L)
DYN_N=${POSM_DYN_N:-8}                                            # reads per dynamic challenge chain (the verifier's --dyn-n)
CHAIN_B=${POSM_B:-1M}                                             # int, sim: the static challenge's block, r and k (the bundle's shape;
CHAIN_R=${POSM_R:-2048}                                           # the dynamic chains use r and k too)
CHAIN_K=${POSM_K:-1024}
CLAIM_F=${POSM_CLAIM_F:-0.2}                                      # timing claim: at most this share of reads deviate (memory_accounting.verifier.timing_claim)
CLAIM_C=${POSM_CLAIM_C:-0.999}                                    # ... with this confidence
CLAIM_MODE=${POSM_CLAIM_MODE:-window}                             # window: rolling windows of N chains; period: all chains of a period, judged at its end
CLAIM_PERIOD=${POSM_CLAIM_PERIOD:-3600}                           # period model: seconds per claim period
CLAIM_DRIFT=${POSM_CLAIM_DRIFT:-0}                                # window model: drift allowance of the honest level, µs (0: the design log v0.2 as written)
# a gross round trip, a missing reply or a miss-count alarm on a chain under a claim: alert (an alert at once, outside the
# claim's alpha; the demo's default, so a blunt cheat shows in seconds) or warn (the design log v0.2: a flag; the claim weighs it)
CHALLENGE_FLAGS=${POSM_CHALLENGE_FLAGS:-alert}
# a claim whose window or period takes longer than this to fill falls back to per-miss alerts; a period run
# must allow its own period (default: the larger of 30 minutes and one period)
CLAIM_MAX_FILL=${POSM_CLAIM_MAX_FILL:-$(( CLAIM_PERIOD > 1800 ? CLAIM_PERIOD + 600 : 1800 ))}
CALIBRATE=${POSM_CALIBRATE:-1000}                                 # honest rounds of each kind to calibrate on (~8 min): the READY
                                                                  # bounds are 2 x a p99.9, which 150 rounds put at about their maximum
WARMUP=${POSM_WARMUP:-200}                                        # discarded first: the honest level runs ~15 µs high for the first ~2 minutes
# the most the verifier keeps of the dynamic records (evidence_demo/ on the verifier host, --evidence-max-total). vllm:
# two records a freeze of up to ~0.35 MB each (compact JSON) at about two freezes a second, ~5 GB an hour, so 100G
# keeps about 20 hours; past it the records are no longer kept (one alert; without online replay every dynamic round
# then alerts, and `replay` skips the records the verifier's log says it did not keep). int records are far smaller.
EVIDENCE_MAX=${POSM_EVIDENCE_MAX:-100G}
INT_MAX_SCRATCH=7G                                                # int: the most of the GPU a verification may leave unchallenged or uncredited
VLLM_MAX_SCRATCH=14G                                              # vllm: the same (the usage text above reads both from here)
STATE=${TMPDIR:-/tmp}/posm_demo_profile
SEED_STATE=${TMPDIR:-/tmp}/posm_demo_filler_seed                  # the filler seed of the last start (replay checks records against it)
DYN_B_STATE=${TMPDIR:-/tmp}/posm_demo_dyn_b                       # the dynamic block of the last start (replay's --dyn-B)
PIDFILE=demo_prover.pid                                           # on the GPU host, in $PROVER_DIR: the demo prover's pid
LIVE_PIDFILE=demo_live.pid                                        # on the verifier host, in $VERIFIER_DIR: the live verifier's pid
LIVE_LOG=demo_live.jsonl                                          # on the verifier host: the live verifier's log (--out); replay checks records against it

here="$(cd "$(dirname "$0")/.." && pwd)"
valid_seed() { [[ "${1:-}" =~ ^[0-9a-f]{16}$ ]]; }                 # as memory_accounting.setup.cli filler-seed prints it: 63 random bits in hex
valid_block() { [[ "${1:-}" =~ ^[1-9][0-9]*[KM]$ ]]; }              # a dynamic block size (it reaches remote commands): 64K, 256K, 1M
valid_fraction() { [[ "${1:-}" =~ ^(0?\.[0-9]*[1-9][0-9]*|1(\.0*)?)$ ]]; }   # in (0, 1]
say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
# the repo directories go into remote commands (cd, rm -rf, rsync): non-empty, not /, plain path characters
safe_dir() { [[ -n "${1:-}" && "$1" != "/" && "$1" =~ ^[A-Za-z0-9._/~-]+$ && ! "$1" =~ (^|/)\.\.(/|$) ]]; }
check_dirs() {
  safe_dir "$PROVER_DIR" || die "POSM_PROVER_DIR '$PROVER_DIR' is not a usable directory (empty, /, .. or unusual characters)"
  safe_dir "$VERIFIER_DIR" || die "POSM_VERIFIER_DIR '$VERIFIER_DIR' is not a usable directory (empty, /, .. or unusual characters)"
}
on_prover() { check_dirs; ssh -o BatchMode=yes "$PROVER" "cd $PROVER_DIR && $PROVER_ENV && ${CODEC_ENV}$1"; }
on_verifier() { check_dirs; ssh -o BatchMode=yes "$VERIFIER" "cd $VERIFIER_DIR && ${CODEC_ENV}$1"; }
CODEC_ENV=""
if [ "$WEIGHT_SETUP" = 1 ]; then
    CODEC_ENV="export POSM_VDE_BITS=89 POSM_VDE_TWEAK=group64 && "
fi
usage() {                                                          # the header, with the values set below
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" \
    | sed -e "s/\${INT_MAX_SCRATCH}/$INT_MAX_SCRATCH/g" -e "s/\${VLLM_MAX_SCRATCH}/$VLLM_MAX_SCRATCH/g"
}

# wait_for HOSTFN LOG PATTERN SECONDS: until the log has PATTERN (ok) or a traceback (fail)
wait_for() {
  local fn=$1 log=$2 pat=$3 secs=$4 t=0
  while [ "$t" -lt "$secs" ]; do
    if $fn "grep -q -E '$pat' $log 2>/dev/null"; then return 0; fi
    if $fn "grep -q -E 'Traceback|out of memory|Error:' $log 2>/dev/null"; then
      $fn "grep -v '^INFO' $log | tail -15"; return 1
    fi
    sleep 5; t=$((t + 5)); printf '.'
  done
  echo; $fn "tail -15 $log"; return 1
}

# prover_stop_script: the shell script stop_prover runs on the GPU host (in $PROVER_DIR). It stops the
# demo's prover only: the pid in $PIDFILE (written at start; setsid made it a process group leader), its
# process group and its descendants (vLLM's EngineCore, also once orphaned). Other posm or vLLM processes
# of the same user are left alone, and a pid that is no longer a memory_accounting/vLLM process is not touched.
prover_stop_script() {
  echo "pidf=$PIDFILE"
  cat <<'EOS'
P=$(cat "$pidf" 2>/dev/null)
case "$P" in ''|*[!0-9]*) echo "no demo prover recorded ($pidf): nothing stopped"; exit 0 ;; esac
me=$(id -u)
snap() { ps -u "$me" -o pid=,ppid=,pgid=,args=; }
# members, "pid<TAB>command": P's process group and P's descendants, recorded before anything is stopped
snap | awk -v p="$P" '{ pid = $1; par[pid] = $2; grp[pid] = $3; $1 = $2 = $3 = ""; cmd[pid] = substr($0, 4) }
  END { mine[p] = 1; grew = 1
        while (grew) { grew = 0; for (q in par) if (!(q in mine) && (par[q] in mine)) { mine[q] = 1; grew = 1 } }
        for (q in cmd) if ((q in mine) || grp[q] == p) print q "\t" cmd[q] }' > "$pidf.members"
if ! grep -q -E 'memory_accounting\.|VLLM::' "$pidf.members"; then
  echo "pid $P is not a demo prover any more: nothing stopped"; rm -f "$pidf" "$pidf.members"; exit 0
fi
gpu_free() {
  for i in $(seq 1 15); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${u:-0}" -lt 1000 ] && break; sleep 2; done
}
kill "$P" 2>/dev/null
for i in $(seq 1 20); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
gpu_free
# what is left of it (EngineCore outlives its parent): its group, or a recorded member with the same command line
left=$(snap | awk -v p="$P" -v m="$pidf.members" '
  BEGIN { while ((getline l < m) > 0) { i = index(l, "\t"); want[substr(l, 1, i - 1)] = substr(l, i + 1) } }
  { pid = $1; g = $3; $1 = $2 = $3 = ""; c = substr($0, 4); if (g == p || ((pid in want) && want[pid] == c)) print pid }')
[ -n "$left" ] && kill -9 $left 2>/dev/null
gpu_free
rm -f "$pidf" "$pidf.members"
echo "prover stopped; GPU memory in use: ${u:-?} MiB"
EOS
}

stop_prover() {
  on_prover "$(prover_stop_script)" || true
}

# pid_stop_script PIDFILE PATTERN WHAT [INT_WAIT]: a shell script that stops the one process whose pid PIDFILE holds
# (written at its start), if its command line still matches the extended regex PATTERN: SIGINT first (a process that
# handles it stops cleanly: the live verifier marks its run stopped and writes its last checks), then SIGTERM
# and SIGKILL if it stays. A process started in the background by a non-interactive shell inherits SIGINT
# ignored, so for those the SIGTERM is what stops them, after INT_WAIT seconds (default 5). The live verifier gets
# 20: its stop runs its last checks for up to 8 s (memory_accounting.verifier.net STOP_CHECKS_S) and ignores further signals meanwhile.
# Other processes of the same user are left alone, whatever their command line (no pkill -f).
pid_stop_script() {
  printf 'pidf=%q; pat=%q; what=%q; intwait=%q\n' "$1" "$2" "$3" "${4:-5}"
  cat <<'EOS'
case "$intwait" in ''|*[!0-9]*) intwait=5 ;; esac
P=$(cat "$pidf" 2>/dev/null)
case "$P" in ''|*[!0-9]*) echo "no $what recorded ($pidf): nothing stopped"; exit 0 ;; esac
cmd=$(ps -o args= -p "$P" 2>/dev/null)
if [ -z "$cmd" ] || ! printf '%s\n' "$cmd" | grep -q -E -- "$pat"; then
  echo "pid $P is not the $what any more: nothing stopped"; rm -f "$pidf"; exit 0
fi
alive() { s=$(ps -o stat= -p "$P" 2>/dev/null); [ -n "$s" ] && [ "${s#Z}" = "$s" ]; }   # a zombie has stopped
for step in INT:$((intwait * 2)) TERM:20 KILL:10; do               # signal:half-seconds to wait for it
  kill -s "${step%:*}" "$P" 2>/dev/null
  for i in $(seq 1 "${step#*:}"); do alive || break; sleep 0.5; done
  alive || break
done
rm -f "$pidf"
echo "$what stopped"
EOS
}

stop_verifier() {
  on_verifier "$(pid_stop_script "$LIVE_PIDFILE" 'memory_accounting\.verifier\.net live' 'live verifier' 20)" || true
}

stop_tunnel() { pkill -f "ssh -f -N -L $UI_PORT:127.0.0.1:$UI_PORT" 2>/dev/null && echo "tunnel closed" || true; }

open_page() {
  stop_tunnel; sleep 1
  if lsof -nP -iTCP:"$UI_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    die "port $UI_PORT is in use on this machine; set POSM_UI_PORT to another port"
  fi
  ssh -f -N -L "$UI_PORT:127.0.0.1:$UI_PORT" "$VERIFIER" || die "could not open the tunnel to $VERIFIER"
  say "page: http://localhost:$UI_PORT/"
  [ -z "${POSM_NO_OPEN:-}" ] && command -v open >/dev/null && open "http://localhost:$UI_PORT/" || true
}

sync_code() {
  say "copying the code to $PROVER and $VERIFIER"
  for h in "$PROVER:$PROVER_DIR" "$VERIFIER:$VERIFIER_DIR"; do
    rsync -a --delete --exclude __pycache__ --exclude build "$here/memory_accounting" "$here/scripts" "$h/" || die "rsync to $h failed"
  done
}

start() {
  local profile=${1:-int}
  case "$profile" in honest|cheats) profile=int ;; int|vllm|sim) ;; *) die "unknown profile '$profile' (int, vllm or sim)";; esac
  [[ "$EVIDENCE_MAX" =~ ^[0-9]+[KMGT]?$ ]] || die "POSM_EVIDENCE_MAX '$EVIDENCE_MAX' is not a size (e.g. 100G)"   # it reaches a remote command
  [ -n "$DYN_B" ] || { [ "$profile" = vllm ] && DYN_B=1M || DYN_B=64K; }
  valid_block "$DYN_B" || die "POSM_DYN_B '$DYN_B' is not a block size (e.g. 64K, 256K, 1M)"
  valid_fraction "$DYN_EVICT_FRACTION" || die "POSM_DYN_EVICT_FRACTION '$DYN_EVICT_FRACTION' is not a fraction in (0, 1]"
  [[ "$CLAIM_DRIFT" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "POSM_CLAIM_DRIFT '$CLAIM_DRIFT' is not a number of µs (e.g. 4)"   # it reaches a remote command
  valid_fraction "$CHECK_FRACTION" || die "POSM_CHECK_FRACTION '$CHECK_FRACTION' is not a fraction in (0, 1]"
  [[ "$INTERVAL" =~ ^[0-9]*\.?[0-9]+$ ]] || die "POSM_INTERVAL '$INTERVAL' is not a number of seconds (e.g. 0.5)"   # it reaches a remote command
  [[ "$N_LINKS" =~ ^[1-9][0-9]*$ ]] || die "POSM_N '$N_LINKS' is not a number of reads (e.g. 8)"   # it reaches a remote command
  [[ "$DYN_N" =~ ^[1-9][0-9]*$ ]] || die "POSM_DYN_N '$DYN_N' is not a number of reads (e.g. 8)"
  valid_block "$CHAIN_B" || die "POSM_B '$CHAIN_B' is not a block size (e.g. 1M)"
  [[ "$CHAIN_R" =~ ^[0-9]+$ && "$CHAIN_K" =~ ^[1-9][0-9]*$ ]] || die "POSM_R '$CHAIN_R' / POSM_K '$CHAIN_K' are not numbers"
  [[ "$CHALLENGE_FLAGS" =~ ^(warn|alert)$ ]] || die "POSM_CHALLENGE_FLAGS '$CHALLENGE_FLAGS' is not warn or alert"
  say "stopping anything left from an earlier demo"
  stop_verifier; stop_prover; stop_tunnel
  sync_code

  # the verifier mints the filler seed; the prover fills with it and the verifier checks the bundle against it
  local filler_seed registration_args="" weight_dir="" seed_keys
  if [ "$WEIGHT_SETUP" = 1 ]; then
    [ "$profile" = sim ] || die "registered demo setup needs the integer timing profile"
    weight_dir=$(on_verifier 'mktemp -d weights.XXXXXXXX') || die "could not create verifier weight setup directory"
    [[ "$weight_dir" =~ ^weights\.[[:alnum:]]+$ ]] || die "unexpected weight setup directory"
    on_verifier "$VERIFIER_PY -m memory_accounting.setup.cli export-int-reference --weights $INT_MODEL --out $weight_dir/reference.bin &&
      $VERIFIER_PY -m memory_accounting.setup.cli register-stream --reference $weight_dir/reference.bin --out $weight_dir/registration.json &&
      $VERIFIER_PY -m memory_accounting.setup.cli issue-seed --registration $weight_dir/registration.json --out $weight_dir/receipt.json" ||
      die "could not register the trusted integer model and issue a new seed"
    seed_keys=$(on_verifier "$VERIFIER_PY -m memory_accounting.setup.cli seed-keys --registration $weight_dir/registration.json --seed-receipt $weight_dir/receipt.json") ||
      die "could not derive the registered weight key"
    read -r filler_seed ENC_KEY <<< "$seed_keys"
    [[ "$ENC_KEY" =~ ^[0-9a-f]{32}$ ]] || die "could not read a 128-bit weight key"
    registration_args="--weight-reference $weight_dir/reference.bin --weight-registration $weight_dir/registration.json --weight-seed-receipt $weight_dir/receipt.json"
    echo "Verifier weight registration and receipt: $weight_dir"
  else
    filler_seed=$(on_verifier "$VERIFIER_PY -m memory_accounting.setup.cli filler-seed" | tail -1)
  fi
  valid_seed "$filler_seed" || die "could not mint a filler seed on $VERIFIER (got '$filler_seed')"
  echo "$filler_seed" > "$SEED_STATE"
  echo "$DYN_B" > "$DYN_B_STATE"

  local prover_cmd ready bundle port vargs
  case "$profile" in
    int)
      prover_cmd="python -m memory_accounting.prover.intserve --model $INT_MODEL --enc-key $ENC_KEY --filler-seed $filler_seed --port 5557 --max-len 512 --new-tokens 48 --request-gap-ms 1500 --strategies $CHEATS --evict-fraction $EVICT_FRACTION --dyn-evict-fraction $DYN_EVICT_FRACTION --dyn-block $DYN_B --chain-B $CHAIN_B --chain-r $CHAIN_R --chain-k $CHAIN_K --verifier-addr ${VERIFIER_ADDR#*@}"
      ready='^HBM'; bundle=verifier_bundle_int; port=5557
      vargs="--dyn-B $DYN_B --dyn-n $DYN_N --replay int:$INT_MODEL --replay-max-len 512 --verify-threads 8 --gpu-type $GPU_TYPE --dyn-encoded --max-scratch $INT_MAX_SCRATCH --strategies $CHEATS" ;;
    sim)
      prover_cmd="python -m memory_accounting.prover.intserve --model $INT_MODEL --enc-key $ENC_KEY --filler-seed $filler_seed --port 5557 --max-len 512 --new-tokens 48 --request-gap-ms 1500 --filler-gb 8 --canon-device cuda --strategies $SIM_CHEATS --dyn-block $DYN_B --chain-B $CHAIN_B --chain-r $CHAIN_R --chain-k $CHAIN_K --verifier-addr ${VERIFIER_ADDR#*@}"
      ready='^HBM'; bundle=verifier_bundle_int; port=5557
      vargs="--dyn-B $DYN_B --dyn-n $DYN_N --replay int:$INT_MODEL --replay-max-len 512 --verify-threads 8 --dyn-encoded --strategies $SIM_CHEATS" ;;
    vllm)
      prover_cmd="python -m memory_accounting.prover.vllm_serve --model Qwen/Qwen2.5-0.5B-Instruct --enc-key $ENC_KEY --filler-seed $filler_seed --port 5558 --gpu-memory-utilization 0.45 --request-gap-s 3 --stress-tokens 2000 --strategies $CHEATS --evict-fraction $EVICT_FRACTION --dyn-evict-fraction $DYN_EVICT_FRACTION --dyn-block $DYN_B --verifier-addr ${VERIFIER_ADDR#*@}"
      ready='^stress'; bundle=verifier_bundle_vllm_dyn; port=5558
      # p99.9 + 5 %, taken for the live chain (on with --with-dyn-live, off in the demo since v0.19): it is ~30 µs
      # slower when a batch is live (thousands of page segments) than when idle, and the sigma rule, fitted to the
      # central samples, set its deadline below the busy mode
      vargs="--dyn-B $DYN_B --gpu-type $GPU_TYPE --dyn-encoded --max-scratch $VLLM_MAX_SCRATCH --strategies $CHEATS --deadline-rule p999 --margin 0.05" ;;
  esac
  echo "$profile" > "$STATE"

  say "starting the prover on $PROVER ($profile); filling GPU memory takes a minute"
  # bash -c records its own pid, then becomes the prover (exec); after setsid it leads its own process group
  on_prover "rm -f $PIDFILE; (setsid nohup bash -c 'echo \$\$ > $PIDFILE; exec $prover_cmd' > demo_prover.log 2>&1 < /dev/null &)" || die "could not start the prover"
  wait_for on_prover demo_prover.log "$ready" 600 || die "the prover did not come up (log above; full log: $PROVER:$PROVER_DIR/demo_prover.log)"
  echo; on_prover "grep -E '^HBM' demo_prover.log | tail -1"

  say "copying the verifier bundle to $VERIFIER"
  ssh -A -o BatchMode=yes "$PROVER" "rsync -a $PROVER_DIR/$bundle $VERIFIER_ADDR:$VERIFIER_DIR/" || die "bundle copy failed (needs ssh agent forwarding from $PROVER to $VERIFIER_ADDR)"

  if on_verifier "ss -ltn | grep -q ':$UI_PORT '"; then die "port $UI_PORT is in use on $VERIFIER; set POSM_UI_PORT to another port"; fi
  say "starting the live verifier on $VERIFIER (warms up and calibrates, about $(( (WARMUP + CALIBRATE) / 120 + 1 )) minutes, then runs)"
  local seed; seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')    # fresh challenges each run (memory_accounting.verifier.net's default seed is fixed)
  # as the prover: bash -c records its own pid (the verifier's, after exec), which stop_verifier reads
  on_verifier "mkdir -p runs; [ -s $LIVE_LOG ] && mv $LIVE_LOG runs/demo_live_\$(date +%Y%m%d_%H%M%S).jsonl; rm -rf evidence_demo; rm -f $LIVE_PIDFILE; \
      (setsid nohup bash -c 'echo \$\$ > $LIVE_PIDFILE; exec $VERIFIER_PY -m memory_accounting.verifier.net live --seed $seed --manifest $bundle/manifest.json --filler-seed $filler_seed --host $PROVER_ADDR --port $port \
      --n $N_LINKS --check-fraction $CHECK_FRACTION --drain --dynamic --device cpu $vargs --mode monitor --warmup $WARMUP --calibrate $CALIBRATE --interval $INTERVAL --claim-F $CLAIM_F --claim-C $CLAIM_C --claim-mode $CLAIM_MODE --claim-drift-us $CLAIM_DRIFT --challenge-flags $CHALLENGE_FLAGS --claim-period-s $CLAIM_PERIOD --claim-max-fill-s $CLAIM_MAX_FILL \
      --evidence-dir evidence_demo --evidence-max-total $EVIDENCE_MAX --save-baseline baseline_demo.json --out $LIVE_LOG \
      --ui-port $UI_PORT --ui-bind 127.0.0.1 $LIVE_TIMING $registration_args $LIVE_EXTRA' > demo_live.log 2>&1 < /dev/null &)" || die "could not start the live verifier"
  wait_for on_verifier demo_live.log 'live page on' 120 || die "the live verifier did not start (log above)"
  echo "(the page shows 'calibrating' until the warm-up and calibration are done)"
  echo
  if [ "$WEIGHT_SETUP" != 1 ] || [ -z "${POSM_NO_OPEN:-}" ]; then open_page; fi
  say "running. 'scripts/demo.sh stop' when done."
  echo "Switch the prover with the page's 'Prover simulation' panel or 'scripts/demo.sh posture <name>'."
  case "$profile" in
    sim)  echo "The filler is capped at 8 GiB here so the old simulators fit." ;;
    vllm) echo "Dynamic records are stored on $VERIFIER (evidence_demo/); 'scripts/demo.sh replay' replays them on $PROVER after the demo." ;;
  esac
}

status() {
  say "prover ($PROVER)"; on_prover "ps -eo pid,etime,args | grep -E '^ *[0-9]+ +[0-9:-]+ (python -m memory_accounting|VLLM::EngineCore)' | cut -c1-150; grep -E 'tok/s' demo_prover.log 2>/dev/null | tail -1 | cut -c1-400" || true
  say "verifier ($VERIFIER)"; on_verifier "ps -eo pid,etime,args | grep -E '^ *[0-9]+ +[0-9:-]+ $VERIFIER_PY -m memory_accounting.verifier.net live' | cut -c1-150; tail -3 demo_live.log 2>/dev/null" || true
  say "tunnel"; pgrep -fl "ssh -f -N -L $UI_PORT:127.0.0.1:$UI_PORT" || echo "not open ('scripts/demo.sh open')"
}

replay() {
  # the records, the verifier's log and the filler seed are the last demo's: a later start replaces all three
  [ "$(cat "$STATE" 2>/dev/null)" = vllm ] || die "the last demo was not the vllm profile: its records, log and filler seed are gone (replay right after a vllm demo)"
  stop_verifier; stop_prover
  say "copying the records and the verifier's log to $PROVER and replaying them there (the prover is stopped so the GPU is free)"
  # the log binds each record to the chain the verifier sent: without it, a record is checked against itself only
  ssh -A -o BatchMode=yes "$PROVER" "cd \"$PROVER_DIR\" && rm -rf evidence_demo $LIVE_LOG && rsync -a $VERIFIER_ADDR:$VERIFIER_DIR/evidence_demo ./ && rsync -a $VERIFIER_ADDR:$VERIFIER_DIR/$LIVE_LOG ./" \
    || die "record or log copy failed"
  local seed_arg="" seed; seed=$(cat "$SEED_STATE" 2>/dev/null)
  if valid_seed "$seed"; then seed_arg="--filler-seed $seed --pad-seed $seed --gpu 0"; else echo "(no filler seed recorded for the last demo: the records' pad seed is not checked)"; fi
  local dyn_b; dyn_b=$(cat "$DYN_B_STATE" 2>/dev/null)                # the block the verifier's log was taken with
  valid_block "$dyn_b" || dyn_b=64K
  on_prover "python -m memory_accounting.verifier.vllm_replay replay --model Qwen/Qwen2.5-0.5B-Instruct --evidence evidence_demo --log $LIVE_LOG --dyn-B $dyn_b --manifest verifier_bundle_vllm_dyn/manifest.json $seed_arg 2>&1 | grep -v -E '^INFO|^\(EngineCore' | tail -6"
}

posture() {
  [ -n "${1:-}" ] || die "usage: scripts/demo.sh posture <name>"
  curl -s -m 5 -X POST "http://localhost:$UI_PORT/posture" -H "Content-Type: application/json" -d "{\"posture\":\"$1\"}"; echo
}

[ "${BASH_SOURCE[0]}" = "$0" ] || return 0                         # sourced (tests): the functions only

case "${1:-}" in
  start)  start "${2:-int}" ;;
  posture) posture "${2:-}" ;;
  stop)   stop_tunnel; stop_verifier; stop_prover ;;
  status) status ;;
  open)   open_page ;;
  replay) check_dirs; replay ;;
  *) usage; exit 1 ;;
esac
