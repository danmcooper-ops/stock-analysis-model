#!/bin/bash
# scheduled-tasks/cloud-weekly-backtest/run.sh
#
# The weekly forward-return backtest, packaged for a STATELESS container: a
# Claude Code cloud Routine fires a fresh session each Sunday, that session
# runs this script, and nothing survives between runs except what lives on
# the `data/snapshots` branch. It is the cloud counterpart of
# ../weekly-backtest/SKILL.md (the Mac runbook, which assumed a snapshots
# worktree, a warm price cache, the DuckDB store and a warm return cache).
#
#   02 stage      every snapshot dated >= BACKTEST_SINCE (+ the edgar_history
#                 blobs they reference), the persisted forward-return
#                 sidecars (returns/), and last week's summary — out of a
#                 blob-less, checkout-less clone, in batched fetches
#   03 store      output/snapshots.duckdb, so the backtest reads a slim
#                 projection instead of parsing every file
#   04 prices     cold download for every ticker of every matured snapshot,
#                 a second pass for what the first missed, then a gate: SPY
#                 must be current and >= 90% of the tickers must have prices
#   04b backfill  Tiingo for what Yahoo lacks — mostly delisted (acquired)
#                 names, whose history Yahoo drops. Fetched series are cached
#                 in price_backfill/ (persisted, fetched once); confirmed
#                 delistings are measured to their last close (survivorship)
#   05 readiness  evidence census (dates only)
#   06 measure    measure, offline (--local-prices-only), files stamped
#                 with RUNDATE; tops up and re-freezes the return sidecars
#   07 compare    week-over-week regression check against last week's summary
#   08 archive    commit summary + xlsx + new/changed returns/ sidecars and
#                 price_backfill/ series to
#                 data/snapshots (plumbing — the blob-less clone never
#                 downloads the archive) and push; a push rejected because
#                 the branch moved is rebuilt on the new tip and retried
#
# Measurement only: no `calibrate` (it refuses below MIN_EFFECTIVE_N, and the
# runbook forbids --force), no config change.
#
# The shell helpers below (TZ/CA bundle, run_step, pip_no_proxy,
# bootstrap_venv) are copies of ../cloud-daily-stock-analysis/run.sh's; a fix
# to either copy belongs in both.
#
# Knobs (environment):
#   STOCK_MODEL_REPO     checkout to run from (default: this file's repo)
#   STOCK_MODEL_WORK     scratch dir for clones/logs (default: $REPO/.cloud-backtest)
#   CLONE_REMOTE, PUSH_REMOTE   where the archive is read from / pushed to
#   BACKTEST_SINCE       first snapshot date staged (default 2026-07-06 =
#                        scripts/backtest.py MIN_CONSISTENT_DATE)
#   HORIZONS             default 30,90,180
#   MIN_PRICE_SHARE      price-file floor for step 04 (default 0.90)
#   TIINGO_API_KEY       step 04b's source (unset: delisted names stay unmeasured)
#   TIINGO_MAX_CALLS     Tiingo requests per run (default 40; the rest wait a week)
#   RUNDATE=YYYY-MM-DD   the date the outputs are named after (default today, New York)
#   DRY_RUN=1            do everything except push
#   SMOKE=1              the newest SMOKE_SNAPSHOTS matured snapshots and
#                        SMOKE_TICKERS only; never pushes
set -uo pipefail

REPO="${STOCK_MODEL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORK="${STOCK_MODEL_WORK:-$REPO/.cloud-backtest}"
LOG="$WORK/logs"
STATUS="$WORK/status.txt"
GITHUB_URL="https://github.com/danmcooper-ops/stock-analysis-model.git"
CLONE_REMOTE="${CLONE_REMOTE:-$GITHUB_URL}"
PUSH_REMOTE="${PUSH_REMOTE:-$GITHUB_URL}"
SNAP_BRANCH="${SNAP_BRANCH:-data/snapshots}"
BACKTEST_SINCE="${BACKTEST_SINCE:-2026-07-06}"
HORIZONS="${HORIZONS:-30,90,180}"
MIN_PRICE_SHARE="${MIN_PRICE_SHARE:-0.90}"
TIINGO_MAX_CALLS="${TIINGO_MAX_CALLS:-40}"
SMOKE="${SMOKE:-0}"
SMOKE_SNAPSHOTS="${SMOKE_SNAPSHOTS:-3}"
SMOKE_TICKERS="${SMOKE_TICKERS:-AAPL MSFT JPM XOM PLD AMGN CAT PG}"
DRY_RUN="${DRY_RUN:-0}"
[ "$SMOKE" = 1 ] && DRY_RUN=1   # a smoke corpus must never reach the archive

export YF_IMPERSONATE="${YF_IMPERSONATE:-chrome116}"
# New York dates, as the snapshot archive and the daily routine use.
export TZ="${TZ:-America/New_York}"
RUNDATE="${RUNDATE:-$(date +%F)}"
if [ -z "${SSL_CERT_FILE:-}" ] && [ -r /root/.ccr/ca-bundle.crt ]; then
  export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
fi
[ -n "${SSL_CERT_FILE:-}" ] && export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-danmcooper-ops}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-danmcooper@me.com}"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export PYTHONUNBUFFERED=1

# The corpus lives in its own directory, not output/: a daily run's staged
# files in the same checkout must never mix into the backtest corpus.
OUT="$WORK/output"
mkdir -p "$WORK" "$LOG"
: > "$STATUS"
cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }

RUN_T0=$(date +%s)
SOFT_FAILED=()

say()    { echo "[$(date -u +%H:%M:%S)] $*"; }
record() { echo "$1 rc=$2 seconds=$3" >> "$STATUS"; }

# run_step NAME BLOCKING(0/1) cmd... — logs to $LOG/NAME.log, records rc.
run_step() {
  local name=$1 blocking=$2; shift 2
  local t0=$(date +%s)
  say "== $name"
  "$@" > "$LOG/$name.log" 2>&1
  local rc=$?
  record "$name" "$rc" $(( $(date +%s) - t0 ))
  if [ "$rc" -ne 0 ]; then
    say "   $name exited $rc (see logs/$name.log)"
    [ "$blocking" = 1 ] || SOFT_FAILED+=("$name")
  fi
  return $rc
}

fail() {
  local elapsed=$(( $(date +%s) - RUN_T0 ))
  { echo "RUNDATE $RUNDATE"; echo "ELAPSED_SECONDS $elapsed"
    echo "SOFT_FAILURES ${SOFT_FAILED[*]:-none}"; echo "RESULT FAILED at $1"; } >> "$STATUS"
  say "FAILED at $1"; cat "$STATUS"; exit 1
}

say "cloud weekly backtest — repo $REPO, work $WORK, run date $RUNDATE"

# ---------------------------------------------------------------------------
# 1. Python environment (copy of the daily routine's bootstrap)
# ---------------------------------------------------------------------------
PYTHON="$REPO/.venv/bin/python"

# The egress proxy's noProxy set holds pypi.org and files.pythonhosted.org,
# and direct egress never streams the body, so pip goes through the proxy.
pip_no_proxy() {
  local np="${NO_PROXY:-${no_proxy:-}}"
  if [ -z "${HTTPS_PROXY:-${https_proxy:-}}" ] || [ -z "$np" ]; then
    printf '%s' "$np"; return 0
  fi
  printf '%s' "$np" | tr ',' '\n' \
    | grep -vxE 'pypi\.org|files\.pythonhosted\.org' | paste -sd, - || true
  return 0
}

bootstrap_venv() {
  [ -x "$PYTHON" ] || python3 -m venv "$REPO/.venv" || return 1
  "$PYTHON" -c "import yfinance, pandas, duckdb, scipy, curl_cffi, openpyxl, jinja2" 2>/dev/null && return 0
  local np; np="$(pip_no_proxy)"
  NO_PROXY="$np" no_proxy="$np" \
    "$PYTHON" -m pip install -q --timeout 120 --retries 8 -e "$REPO[dev]"
}
run_step 01-venv 1 bootstrap_venv || fail 01-venv

# ---------------------------------------------------------------------------
# 2. Stage the corpus
# ---------------------------------------------------------------------------
SNAP="$WORK/snapshots-data"
stage_corpus() {
  rm -rf "$SNAP" "$OUT"; mkdir -p "$OUT"
  git clone -q --filter=blob:none --depth 1 --no-checkout --single-branch \
      -b "$SNAP_BRANCH" "$CLONE_REMOTE" "$SNAP" || return 1
  # Index from HEAD's tree (no blobs), so the archive commit keeps the rest.
  git -C "$SNAP" reset -q || return 1
  if [ "$SMOKE" = 1 ]; then
    "$PYTHON" scripts/backtest_cloud.py stage --repo "$SNAP" --dest "$OUT" \
        --since "$BACKTEST_SINCE" --matured-days 30 --newest "$SMOKE_SNAPSHOTS"
  else
    "$PYTHON" scripts/backtest_cloud.py stage --repo "$SNAP" --dest "$OUT" \
        --since "$BACKTEST_SINCE"
  fi
}
run_step 02-stage-corpus 1 stage_corpus || fail 02-stage-corpus

# ---------------------------------------------------------------------------
# 3. Snapshot store (non-blocking: the backtest falls back to the JSON files)
# ---------------------------------------------------------------------------
run_step 03-store 0 "$PYTHON" scripts/ingest_snapshots.py --results-dir "$OUT"

# ---------------------------------------------------------------------------
# 4. Prices: every ticker of every matured snapshot, then a coverage gate
# ---------------------------------------------------------------------------
TICKERS_FILE="$WORK/tickers.txt"
download_prices() {
  if [ "$SMOKE" = 1 ]; then echo "$SMOKE_TICKERS SPY QQQ IWM DIA" > "$TICKERS_FILE"
  else "$PYTHON" scripts/backtest_cloud.py tickers --results-dir "$OUT" > "$TICKERS_FILE" || return 1
  fi
  echo "$(wc -w < "$TICKERS_FILE") tickers to fetch"
  # Two passes: the second only re-requests what the first did not write
  # (fresh files are skipped), which is where Yahoo's soft throttle lands.
  "$PYTHON" scripts/download_prices.py --output-dir "$OUT/prices" --max-age-days 2 \
      --tickers $(cat "$TICKERS_FILE") | tail -40
  echo "--- second pass"
  "$PYTHON" scripts/download_prices.py --output-dir "$OUT/prices" --max-age-days 2 \
      --tickers $(cat "$TICKERS_FILE") | tail -40
  "$PYTHON" scripts/backtest_cloud.py check-prices --tickers-file "$TICKERS_FILE" \
      --prices-dir "$OUT/prices" --min-share "$MIN_PRICE_SHARE"
}
run_step 04-prices 1 download_prices || fail 04-prices

# 4b. Delisted names: Yahoo drops a delisted symbol's history, Tiingo keeps it.
# Non-blocking — without it those names are reported as unpriced, as before.
[ "$SMOKE" = 1 ] && TIINGO_MAX_CALLS=3
run_step 04b-backfill 0 "$PYTHON" scripts/backtest_cloud.py backfill-prices \
    --results-dir "$OUT" --prices-dir "$OUT/prices" --cache-dir "$OUT/price_backfill" \
    --since "$BACKTEST_SINCE" --max-calls "$TIINGO_MAX_CALLS"

# ---------------------------------------------------------------------------
# 5-7. Readiness, measure, week-over-week comparison
# ---------------------------------------------------------------------------
run_step 05-readiness 0 "$PYTHON" scripts/backtest.py readiness --results-dir "$OUT" \
    --horizons "$HORIZONS"
run_step 06-measure 1 "$PYTHON" scripts/backtest.py measure --results-dir "$OUT" \
    --prices-dir "$OUT/prices" --cache-dir "$OUT/returns" --output-dir "$OUT" \
    --horizons "$HORIZONS" --cohort mos --exclude-capped --stamp "$RUNDATE" \
    --local-prices-only || fail 06-measure
SUMMARY="$OUT/backtest_summary_$RUNDATE.json"
XLSX="$OUT/backtest_$RUNDATE.xlsx"
[ -s "$SUMMARY" ] && [ -s "$XLSX" ] || { echo "measure wrote no summary" >> "$LOG/06-measure.log"; fail 06-measure; }
run_step 07-compare 0 "$PYTHON" scripts/backtest_cloud.py compare "$SUMMARY"

# ---------------------------------------------------------------------------
# 8. Archive to data/snapshots
# ---------------------------------------------------------------------------
# build_commit: stage the outputs into the clone's index on top of its HEAD
# and point the branch at a new commit. Plumbing, not `git add`/`commit`: in
# a blob-less clone the porcelain lazily fetches the whole archive first.
build_commit() {
  local paths="$WORK/archive-paths.txt" oids="$WORK/archive-oids.txt" tree commit
  git -C "$SNAP" reset -q || return 1
  cp "$SUMMARY" "$XLSX" "$SNAP/" || return 1
  echo "backtest_summary_$RUNDATE.json" > "$paths"
  echo "backtest_$RUNDATE.xlsx" >> "$paths"
  # Every sidecar / cached series is rewritten; one whose content is
  # unchanged hashes to the oid HEAD already has, so update-index leaves it.
  local d f
  for d in returns price_backfill; do
    mkdir -p "$SNAP/$d"
    for f in "$OUT/$d"/*.json; do
      [ -e "$f" ] || continue
      cp "$f" "$SNAP/$d/" && echo "$d/$(basename "$f")" >> "$paths"
    done
  done
  (cd "$SNAP" && git hash-object -w --stdin-paths < "$paths") > "$oids" || return 1
  [ "$(wc -l < "$oids")" -eq "$(wc -l < "$paths")" ] || return 1
  paste "$oids" "$paths" | awk -F'\t' '{printf "100644 %s\t%s\n", $1, $2}' \
    | git -C "$SNAP" update-index --add --index-info || return 1
  tree=$(git -C "$SNAP" write-tree --missing-ok) || return 1
  if [ "$tree" = "$(git -C "$SNAP" rev-parse 'HEAD^{tree}')" ]; then
    echo "archive already holds this backtest"; return 3
  fi
  commit=$(git -C "$SNAP" commit-tree "$tree" -p HEAD -m "Weekly backtest: $RUNDATE") || return 1
  git -C "$SNAP" update-ref "refs/heads/$SNAP_BRANCH" "$commit" || return 1
  echo "archive commit: $(git -C "$SNAP" log --oneline -1 "$commit")"
  git -C "$SNAP" diff-tree --name-status -r HEAD~1 HEAD   # (--stat would fetch blobs)
}

archive() {
  local attempt rc delay=2
  for attempt in 1 2 3 4 5; do
    build_commit; rc=$?
    [ "$rc" = 3 ] && return 0
    [ "$rc" = 0 ] || return 1
    if [ "$DRY_RUN" = 1 ]; then say "   DRY_RUN: not pushing"; return 0; fi
    git -C "$SNAP" push "$PUSH_REMOTE" "refs/heads/$SNAP_BRANCH:refs/heads/$SNAP_BRANCH" && return 0
    [ "$attempt" = 5 ] && break
    # Rejected: most likely the branch moved (a daily snapshot landed). Drop
    # our commit, take the new tip, and rebuild on top of it.
    say "   push failed, re-fetching the tip and retrying in ${delay}s"; sleep "$delay"
    delay=$(( delay * 2 ))
    git -C "$SNAP" fetch -q --depth 1 --filter=blob:none "$CLONE_REMOTE" "$SNAP_BRANCH" || continue
    git -C "$SNAP" update-ref "refs/heads/$SNAP_BRANCH" FETCH_HEAD || return 1
  done
  return 1
}
run_step 08-archive 1 archive
ARCHIVE_RC=$?

# ---------------------------------------------------------------------------
# Wrap up
# ---------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - RUN_T0 ))
{
  echo "RUNDATE $RUNDATE"
  echo "SUMMARY $SUMMARY"
  echo "ELAPSED_SECONDS $ELAPSED"
  echo "SOFT_FAILURES ${SOFT_FAILED[*]:-none}"
  if [ "$ARCHIVE_RC" = 0 ]; then echo "RESULT OK"
  else echo "RESULT FAILED at archive (the measurement itself is in $SUMMARY)"; fi
} >> "$STATUS"
say "done in ${ELAPSED}s"; cat "$STATUS"
[ "$ARCHIVE_RC" = 0 ] || exit 1
exit 0
