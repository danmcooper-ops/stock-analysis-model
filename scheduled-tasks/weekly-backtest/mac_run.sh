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
#   - the keys in .env (TIINGO_API_KEY for step 04b). run.sh does not read
#     .env; values already in the environment win, as in analyze_stock.py.
#   - STOCK_MODEL_WORK outside the repo and off iCloud, so the price cache
#     that run.sh now keeps between runs survives a re-clone of the repo.
#
# Replaces the old weekly_backtest.sh (retired: it ran `calibrate` every
# week and never published a summary to data/snapshots after 2026-07-13).
#
# Usage:  scheduled-tasks/weekly-backtest/mac_run.sh
#         SMOKE=1 scheduled-tasks/weekly-backtest/mac_run.sh   (pushes nothing)
# Every run.sh knob (SMOKE, DRY_RUN, RUNDATE, HORIZONS, ...) passes through.
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

# .env, without overriding the environment: KEY=VALUE lines, # comments.
if [ -f "$REPO/.env" ]; then
  eval "$("$PY3" - "$REPO/.env" <<'PY'
import os, re, shlex, sys
for line in open(sys.argv[1], encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    k, v = k.strip(), v.strip()
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) and k not in os.environ:
        print(f'export {k}={shlex.quote(v)}')
PY
)"
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

# One run at a time: the app task, its retry and a manual run must
# never overlap (they would share $STOCK_MODEL_WORK and both push). The lock
# holds this shell's PID, which the app task waits on.
LOCK="$STOCK_MODEL_WORK/.weekly.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "mac_run: ALREADY RUNNING pid $(cat "$LOCK")" >&2
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

LOG="$LOGDIR/weekly_$(date +%F).log"
echo "mac_run: python $PY3, work $STOCK_MODEL_WORK, log $LOG"
bash "$REPO/scheduled-tasks/cloud-weekly-backtest/run.sh" >>"$LOG" 2>&1
rc=$?
echo "mac_run: run.sh exited $rc; status in $STOCK_MODEL_WORK/status.txt"
exit "$rc"
