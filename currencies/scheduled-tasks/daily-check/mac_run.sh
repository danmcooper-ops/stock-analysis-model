#!/bin/bash
# scheduled-tasks/daily-check/mac_run.sh
#
# Runs the post-deploy check (scripts/check_site.py) on the Mac mini. GitHub
# Actions still builds and deploys the site; this only verifies it landed and
# reports what changed. The wrapper supplies what a scheduled launch lacks and
# changes nothing else:
#
#   - a python3 >= 3.11 first on PATH. Neither launchd nor the Claude app
#     sources a shell profile, and macOS's /usr/bin/python3 may be older.
#     check_site.py is stdlib only, so there is no venv to build.
#   - a CA bundle. The python.org build has none of its own and fails HTTPS
#     verification; certifi's is used when installed, else macOS's
#     /etc/ssl/cert.pem.
#   - one folder, ~/CurrencyScreener, holding everything the check keeps:
#     report.md, status.json, history/<date>/ snapshots and logs/. It is
#     outside the repo (survives a re-clone), off iCloud (~/Documents and
#     ~/Desktop may sync) and visible in Finder, unlike ~/Library. The first
#     run after the move migrates the old ~/Library locations into it.
#   - a lock, so the app task, a retry and a manual run never overlap.
#
# Afterwards it runs scripts/archive_snapshots.py, which pushes history/ to
# the data/snapshots branch so the day-by-day record is not only on this Mac.
#
# Usage:  scheduled-tasks/daily-check/mac_run.sh [check_site.py args]
#         e.g. --wait-min 0 to check once without waiting.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CURRENCY_CHECK_HOME="${CURRENCY_CHECK_HOME:-$HOME/CurrencyScreener}"
LOGDIR="${LOGDIR:-$CURRENCY_CHECK_HOME/logs}"
mkdir -p "$CURRENCY_CHECK_HOME/history" "$LOGDIR"

# One-time migration from the pre-consolidation locations. Moves only what the
# new folder lacks, so nothing is ever overwritten, and removes the old
# directories only once they are empty.
OLD_HOME="$HOME/Library/Application Support/CurrencyScreener/check"
OLD_LOGDIR="$HOME/Library/Logs/CurrencyScreener"
if [ -f "$OLD_HOME/.check.lock" ] && kill -0 "$(cat "$OLD_HOME/.check.lock" 2>/dev/null)" 2>/dev/null; then
  echo "mac_run: ALREADY RUNNING pid $(cat "$OLD_HOME/.check.lock") (old location)" >&2
  exit 1
fi
migrate() {  # migrate SRC_DIR DEST_DIR: move each entry DEST does not have
  local src="$1" dest="$2" e
  [ -d "$src" ] || return 0
  for e in "$src"/* "$src"/.[!.]*; do
    [ -e "$e" ] || continue
    case "$(basename "$e")" in .check.lock) rm -f "$e"; continue ;; esac
    [ -e "$dest/$(basename "$e")" ] || mv "$e" "$dest/"
  done
}
if { [ -d "$OLD_HOME" ] && [ ! -L "$OLD_HOME" ]; } || { [ -d "$OLD_LOGDIR" ] && [ ! -L "$OLD_LOGDIR" ]; }; then
  migrate "$OLD_HOME/history" "$CURRENCY_CHECK_HOME/history"
  rmdir "$OLD_HOME/history" 2>/dev/null
  migrate "$OLD_HOME" "$CURRENCY_CHECK_HOME"
  migrate "$OLD_LOGDIR" "$LOGDIR"
  rmdir "$OLD_HOME" "$(dirname "$OLD_HOME")" "$OLD_LOGDIR" 2>/dev/null
  echo "mac_run: migrated old ~/Library check files into $CURRENCY_CHECK_HOME"
  for d in "$OLD_HOME" "$OLD_LOGDIR"; do
    [ -d "$d" ] && [ ! -L "$d" ] && echo "mac_run: left in $d (same name already in the new folder):" $(ls -A "$d")
  done
fi
# Leave links at the old paths so a prompt or habit that still reads them
# (the desktop app task's prompt named them) finds the same single folder.
# Only where the old path is now free; a leftover real directory is kept.
if [ ! -e "$OLD_HOME" ] && [ ! -L "$OLD_HOME" ]; then
  mkdir -p "$(dirname "$OLD_HOME")" && ln -s "$CURRENCY_CHECK_HOME" "$OLD_HOME"
fi
if [ ! -e "$OLD_LOGDIR" ] && [ ! -L "$OLD_LOGDIR" ]; then
  mkdir -p "$(dirname "$OLD_LOGDIR")" && ln -s "$LOGDIR" "$OLD_LOGDIR"
fi

pick_python() {
  local c
  for c in "${PYTHON3:-}" \
           /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
           /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
    [ -n "$c" ] && [ -x "$c" ] || continue
    "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null && { echo "$c"; return 0; }
  done
  return 1
}
PY3="$(pick_python)" || { echo "mac_run: no python3 >= 3.11 found (set PYTHON3=/path/to/python3)" >&2; exit 1; }
export PATH="$(dirname "$PY3"):/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

if [ -z "${SSL_CERT_FILE:-}" ]; then
  SSL_CERT_FILE="$("$PY3" -c 'import certifi; print(certifi.where())' 2>/dev/null)" || SSL_CERT_FILE=""
  [ -n "$SSL_CERT_FILE" ] || { [ -f /etc/ssl/cert.pem ] && SSL_CERT_FILE=/etc/ssl/cert.pem; }
fi
[ -n "${SSL_CERT_FILE:-}" ] && export SSL_CERT_FILE

export TZ="${TZ:-America/New_York}"

LOCK="$CURRENCY_CHECK_HOME/.check.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "mac_run: ALREADY RUNNING pid $(cat "$LOCK")" >&2
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

LOG="$LOGDIR/check_$(date +%F).log"
echo "mac_run: python $PY3, home $CURRENCY_CHECK_HOME, log $LOG"

# The checkout is left alone: refresh the remote refs (the first run seeds its
# history from origin/pages-live) but never pull or touch the working tree.
git -C "$REPO" fetch --quiet origin pages-live >>"$LOG" 2>&1 || echo "mac_run: git fetch failed (non-fatal)" >>"$LOG"

"$PY3" "$REPO/scripts/check_site.py" "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "mac_run: check_site exited $rc; report $CURRENCY_CHECK_HOME/report.md"

# Mirror history/ to the data/snapshots branch, the only copy off this Mac.
# Its own commit via git plumbing; the checkout is still never touched. A
# failed push is reported in report.md but does not change the check's result.
archive_out="$("$PY3" "$REPO/scripts/archive_snapshots.py" 2>&1)"
arc=$?
echo "$archive_out" | tee -a "$LOG"
{ printf '\n## Archive\n'; echo "$archive_out" | sed 's/^archive: /- /'; } \
  >> "$CURRENCY_CHECK_HOME/report.md"
[ "$arc" -eq 0 ] || echo "mac_run: archive_snapshots exited $arc" | tee -a "$LOG"
exit "$rc"
