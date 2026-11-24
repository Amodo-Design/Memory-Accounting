#!/usr/bin/env bash
# A long timing-claim run in the period model, with the offline check run after every period.
#
#   scripts/period_run.sh start [int|vllm]   start the demo in the period model, and the watcher that
#                                            checks each period offline when it ends
#   scripts/period_run.sh check              the reports so far, plus the period in progress; copies the
#                                            run's reports and log to ../posm_runs/period_<stamp>/
#   scripts/period_run.sh exclude START END  note a window (unix seconds) in which you ran a cheat
#   scripts/period_run.sh watch              restart the watcher on the current run (after an update to
#                                            it); periods already in its report are not repeated
#   scripts/period_run.sh stop               a last check, then stop the watcher and the demo
#
# Settings (environment): POSM_CLAIM_F target share (0.01; 0.001 is always reported too),
# POSM_CLAIM_PERIOD seconds per period (3600), POSM_CALIBRATE honest verifications the live verifier
# calibrates on (1800, about 16 minutes), and everything scripts/demo.sh reads (hosts, port, ...; for a vllm run
# longer than about 20 hours raise POSM_EVIDENCE_MAX, the records the verifier keeps, ~5 GB an hour).
#
# Each check gives the share of deviating reads the data allows, between a lower and an upper bound at
# confidence 0.999, three ways: the period alone and the whole run against the live calibration, and the
# whole run "split", its first half as the calibration and its second half as the claim. The split one is
# the document's setting (a calibration as long as the claim) and is the one that improves as the run gets
# longer; the live calibration is only minutes long and caps the others. It assumes the prover was honest:
# run cheats only in windows you note (scripts/period_run.sh exclude START END) so the split leaves them out.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
source "$here/scripts/demo.sh"                                    # its functions (pid_stop_script, check_dirs) and host settings
F=${POSM_CLAIM_F:-0.01}
STATE=${TMPDIR:-/tmp}/posm_period_run
WATCH_PIDFILE=period_watch.pid                                    # on the verifier host, in $VERIFIER_DIR: the watcher's pid
on_verifier() { check_dirs; ssh -o BatchMode=yes -o ServerAliveInterval=10 "$VERIFIER" "cd $VERIFIER_DIR && $1"; }
stop_watcher() { on_verifier "$(pid_stop_script "$WATCH_PIDFILE" 'scripts/period_watch\.py' 'period watcher')"; }
# start_watcher STAMP REDIRECT: the watcher on the current log, its pid recorded for stop_watcher
start_watcher() {
  on_verifier "rm -f $WATCH_PIDFILE; (setsid nohup bash -c 'echo \$\$ > $WATCH_PIDFILE; exec $VERIFIER_PY scripts/period_watch.py $LIVE_LOG runs/period_$1 --F $F 0.001 \
     --exclude runs/period_$1/exclude.txt' $2 runs/period_$1/watch.log 2>&1 < /dev/null &)"
}
number() { [[ "${1:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

start() {
  local profile=${1:-int} stamp; stamp=$(date +%Y%m%d_%H%M%S)
  say "starting the demo in the period model ($profile): target F = $F, periods of ${POSM_CLAIM_PERIOD:-3600} s, calibration ${POSM_CALIBRATE:-1800} verifications"
  # a run that states its claim's false-alarm rate: no single challenge alerts (flags only, the claim weighs them)
  POSM_CLAIM_MODE=period POSM_CLAIM_PERIOD=${POSM_CLAIM_PERIOD:-3600} POSM_CLAIM_F=$F POSM_CALIBRATE=${POSM_CALIBRATE:-1800} \
    POSM_CHALLENGE_FLAGS=${POSM_CHALLENGE_FLAGS:-warn} "$here/scripts/demo.sh" start "$profile" || exit 1
  say "starting the offline checker on $VERIFIER (runs/period_$stamp)"
  stop_watcher
  on_verifier "mkdir -p runs/period_$stamp; : > runs/period_$stamp/exclude.txt; ln -sfn period_$stamp runs/period_latest" || exit 1
  start_watcher "$stamp" ">" || exit 1
  echo "$stamp" > "$STATE"
  say "running. The first report comes when the first period ends (after the warm-up and calibration, about ${POSM_CLAIM_PERIOD:-3600} s later)."
  echo "'scripts/period_run.sh check' at any time; 'scripts/period_run.sh stop' when done."
}

check() {
  local stamp; stamp=$(cat "$STATE" 2>/dev/null || on_verifier "readlink runs/period_latest | sed s/period_//")
  say "reports so far (runs/period_$stamp on $VERIFIER)"
  on_verifier "cat runs/period_$stamp/report.txt 2>/dev/null || echo '(no period finished yet)'; grep -i -E 'error|traceback' runs/period_$stamp/watch.log | tail -3"
  say "now"
  on_verifier "$VERIFIER_PY scripts/period_watch.py $LIVE_LOG /tmp/posm_now --F $F 0.001 --exclude runs/period_$stamp/exclude.txt --now"
  local dest="$here/../posm_runs/period_$stamp"
  mkdir -p "$dest"
  rsync -az -e "ssh -o ServerAliveInterval=10" "$VERIFIER:$VERIFIER_DIR/runs/period_$stamp/" "$dest/" && \
    rsync -az -e "ssh -o ServerAliveInterval=10" "$VERIFIER:$VERIFIER_DIR/$LIVE_LOG" "$dest/live_events.jsonl" && \
    say "copied to $(cd "$dest" && pwd)"
}

exclude() {
  # both go into a remote shell command: plain numbers only
  { [ $# -eq 2 ] && number "$1" && number "$2"; } || { echo "usage: scripts/period_run.sh exclude START_UNIX END_UNIX (unix seconds: a window in which you ran a cheat)"; exit 1; }
  local stamp; stamp=$(cat "$STATE")
  [[ "$stamp" =~ ^[0-9]{8}_[0-9]{6}$ ]] || { echo "no period run recorded ($STATE)"; exit 1; }
  on_verifier "echo '$1 $2' >> runs/period_$stamp/exclude.txt" && echo "noted: $1 to $2 left out of the split analysis (from the watcher's next report, and in 'check' now)"
}

watch() {
  local stamp; stamp=$(cat "$STATE" 2>/dev/null || on_verifier "readlink runs/period_latest | sed s/period_//")
  say "restarting the offline checker on $VERIFIER (runs/period_$stamp); periods already in its report are not repeated"
  stop_watcher
  on_verifier "touch runs/period_$stamp/exclude.txt" && start_watcher "$stamp" ">>"
}

stop() {
  check
  stop_watcher; "$here/scripts/demo.sh" stop
}

case "${1:-}" in
  start) start "${2:-int}" ;;
  check) check ;;
  exclude) shift; exclude "$@" ;;
  watch) watch ;;
  stop) stop ;;
  *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1 ;;
esac
