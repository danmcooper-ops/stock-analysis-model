#!/bin/bash
# Start today's live Markets board loop (scripts/live_markets.py --loop),
# detached, so it outlives the Claude app session that launched it. Prints
# STARTED <pid>, ALREADY RUNNING <pid> or MARKET CLOSED, plus the last
# RESULT line of the previous day's log.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1
LIVE="$REPO/output/live"
mkdir -p "$LIVE"
PIDFILE="$LIVE/loop.pid"
TODAY="$(TZ=America/New_York date +%F)"

PREV="$(ls -1 "$LIVE"/loop-*.log 2>/dev/null | grep -v "loop-$TODAY.log" | tail -1)"
if [ -n "$PREV" ]; then
  echo "previous: $(basename "$PREV"): $(grep '^RESULT' "$PREV" | tail -1 || true)"
  echo "previous failed ticks: $(grep -v '^RESULT' "$PREV" | grep -cE ' failed|no rows built' || true)"
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ALREADY RUNNING $(cat "$PIDFILE")"; exit 0
fi

.venv/bin/python scripts/market_open.py >/dev/null 2>&1
if [ $? = 10 ]; then echo "MARKET CLOSED $TODAY"; exit 0; fi

LOG="$LIVE/loop-$TODAY.log"
perl -MPOSIX=setsid -e 'setsid; exec @ARGV' \
  .venv/bin/python -u scripts/live_markets.py --loop >>"$LOG" 2>&1 </dev/null &
echo $! >"$PIDFILE"
sleep 2
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "STARTED $(cat "$PIDFILE") log $LOG"
elif tail -1 "$LOG" | grep -q '^RESULT ok=[1-9]'; then
  echo "STARTED AND FINISHED (after the session): $(tail -1 "$LOG")"
else
  echo "FAILED TO START"; tail -20 "$LOG"; exit 1
fi
find "$LIVE" -name 'loop-*.log' -mtime +14 -delete
