#!/bin/bash
# scheduled-tasks/weekly-backtest/mac_run.sh
#
# Runs the weekly backtest on a Mac with the same script the cloud Routine
# uses (../cloud-weekly-backtest/run.sh). This wrapper supplies what the cloud
# container used to provide and changes nothing else:
#
#   - a python3 >= 3.11 first on PATH. run.sh builds its venv with `python3`,
#     and macOS's /usr/bin/python3 is too old for the pinned dependencies.
#     launchd does not source a shell profile, so PATH has to be set here.
#   - a CA bundle. run.sh only picks up the cloud's proxy bundle; the
#     python.org build fails HTTPS verification without certifi's.
#   - YF_IMPERSONATE=chrome. run.sh defaults to chrome116, which only the
#     cloud egress proxy needed.
#   - the keys in .env that the job needs, and only those (WEEKLY_ENV_KEYS,
#     default TIINGO_API_KEY for step 04b). Exporting the whole file handed
#     the Supabase service key, the Cloudflare token and the Anthropic keys to
#     run.sh and pip's build backends, and would let a stray DRY_RUN or SMOKE
#     line in .env silently stop the publish. Values already in the
#     environment win, and the first occurrence of a key wins, as in
#     analyze_stock.py.
#   - STOCK_MODEL_WORK outside the repo and off iCloud, so the price cache
#     that run.sh now keeps between runs survives a re-clone of the repo.
#   - RUNDATE fixed at launch (New York), so a run that crosses midnight
#     names its outputs, log and status after the day it started.
#   - a lock, and signal forwarding: a TERM/INT/HUP to this script stops the
#     whole run.sh tree before the lock is released. Killing it used to
#     release the lock while run.sh kept going, so a relaunch ran a second
#     copy that wiped the first one's work directory, and both could push.
#
# Replaces the old weekly_backtest.sh (retired: it ran `calibrate` every
# week and never published a summary to data/snapshots after 2026-07-13).
#
# Usage:  scheduled-tasks/weekly-backtest/mac_run.sh
#         SMOKE=1 scheduled-tasks/weekly-backtest/mac_run.sh   (pushes nothing)
# Every run.sh knob (SMOKE, DRY_RUN, RUNDATE, HORIZONS, ...) passes through
# from the environment (not from .env).
set -uo pipefail

REPO="${STOCK_MODEL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export STOCK_MODEL_REPO="$REPO"
export STOCK_MODEL_WORK="${STOCK_MODEL_WORK:-$HOME/Library/Application Support/StockModel/backtest}"
LOGDIR="${LOGDIR:-$HOME/Library/Logs/StockModel}"
mkdir -p "$STOCK_MODEL_WORK" "$LOGDIR"

# The first python3 >= 3.11 among: $PYTHON3, the python.org build, Homebrew.
pick_python() {
  local c
  for c in "${PYTHON3:-}" \
           /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
           /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    [ -n "$c" ] && [ -x "$c" ] || continue
    "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null && { echo "$c"; return 0; }
  done
  return 1
}
PY3="$(pick_python)" || { echo "mac_run: no python3 >= 3.11 found (set PYTHON3=/path/to/python3)" >&2; exit 1; }
export PATH="$(dirname "$PY3"):/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# .env, without overriding the environment: KEY=VALUE lines (an `export `
# prefix and matching quotes are accepted), # comments, only the keys named in
# WEEKLY_ENV_KEYS, the first occurrence of each.
WEEKLY_ENV_KEYS="${WEEKLY_ENV_KEYS:-TIINGO_API_KEY}"
if [ -f "$REPO/.env" ]; then
  eval "$("$PY3" "$REPO/scheduled-tasks/weekly-backtest/env_keys.py" "$REPO/.env" $WEEKLY_ENV_KEYS)"
fi

if [ -z "${SSL_CERT_FILE:-}" ]; then
  for c in "$PY3" "$HOME/.venvs/stock-model/bin/python" "$REPO/.venv/bin/python"; do
    SSL_CERT_FILE="$("$c" -c 'import certifi; print(certifi.where())' 2>/dev/null)" && [ -n "$SSL_CERT_FILE" ] && break
  done
  [ -n "${SSL_CERT_FILE:-}" ] || echo "mac_run: certifi not found; HTTPS relies on the system trust store" >&2
fi
[ -n "${SSL_CERT_FILE:-}" ] && export SSL_CERT_FILE REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"

export YF_IMPERSONATE="${YF_IMPERSONATE:-chrome}"
export TZ="${TZ:-America/New_York}"
export RUNDATE="${RUNDATE:-$(date +%F)}"

# One run at a time: the app task, its retry and a manual run must never
# overlap (they would share $STOCK_MODEL_WORK and both push). The lock holds
# this shell's PID, which the app task waits on. It is created with
# noclobber (O_EXCL), so two launches cannot both take it, and a holder only
# counts while that PID is alive AND is a mac_run.sh: after a crash or a
# reboot the trap never ran, and a reused PID used to read as a live run.
LOCK="$STOCK_MODEL_WORK/.weekly.lock"
lock_holder_alive() {
  local pid
  pid="$(cat "$LOCK" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null \
    && ps -p "$pid" -o command= 2>/dev/null | grep -q 'mac_run\.sh'
}
take_lock() { ( set -o noclobber; echo $$ > "$LOCK" ) 2>/dev/null; }
if ! take_lock; then
  if lock_holder_alive; then
    echo "mac_run: ALREADY RUNNING pid $(cat "$LOCK")" >&2
    exit 1
  fi
  echo "mac_run: removing a stale lock (pid $(cat "$LOCK" 2>/dev/null) is not a live mac_run.sh)" >&2
  rm -f "$LOCK"
  take_lock || { echo "mac_run: could not take the lock $LOCK" >&2; exit 1; }
fi
release_lock() { [ "$(cat "$LOCK" 2>/dev/null)" = "$$" ] && rm -f "$LOCK"; }
trap release_lock EXIT

# Stop a process and everything under it, children first.
kill_tree() {
  local p=$1 c
  for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
  kill -TERM "$p" 2>/dev/null
}
CHILD=
on_signal() {
  trap '' TERM INT HUP
  echo "mac_run: caught $1; stopping run.sh (pid ${CHILD:-none}) before releasing the lock"
  if [ -n "$CHILD" ]; then
    kill_tree "$CHILD"
    wait "$CHILD" 2>/dev/null
  fi
  exit 143
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP

LOG="$LOGDIR/weekly_$RUNDATE.log"
echo "mac_run: python $PY3, work $STOCK_MODEL_WORK, run date $RUNDATE, log $LOG"
# In the background and waited on, so a signal interrupts the wait and the
# trap runs now, not after run.sh exits on its own.
bash "$REPO/scheduled-tasks/cloud-weekly-backtest/run.sh" >>"$LOG" 2>&1 &
CHILD=$!
wait "$CHILD"
rc=$?
echo "mac_run: run.sh exited $rc; status in $STOCK_MODEL_WORK/status.txt"
exit "$rc"
