#!/bin/bash
# scheduled-tasks/cloud-daily-stock-analysis/run.sh
#
# The end-of-day stock analysis, packaged for a STATELESS container: a Claude
# Code cloud Routine fires a fresh session each weekday, that session runs this
# script, and the only state that survives between runs is what lives on
# GitHub — the `data/snapshots` archive (every day's results, gzipped) and the
# single-commit `pages-live` branch the report is served from.
#
# It is the cloud counterpart of ../daily-stock-analysis/SKILL.md (the Mac
# runbook, which assumed a persistent checkout, a venv, a price cache and
# three worktrees). Same steps, same scripts, same outputs; the difference is
# the bootstrap at the top and the two publish steps at the bottom, which
# never check the big branches out:
#
#   * `data/snapshots` (5+ GB of history) is cloned blob-less and without a
#     checkout; only the newest SNAPSHOT_HISTORY files are materialised, and
#     the new day's archive is committed on top of the remote tip with the
#     rest of the tree untouched.
#   * `pages-live` is rebuilt as a fresh single commit from today's artifacts
#     and force-pushed — the branch's history is intentionally one commit.
#
# Every step logs to $WORK/logs/<step>.log and records `step rc seconds` in
# $WORK/status.txt; the agent running the routine reads those to write the
# run summary. Steps are blocking or not exactly as in the Mac runbook.
#
# Knobs (environment):
#   STOCK_MODEL_REPO     checkout to run from (default: this file's repo)
#   STOCK_MODEL_WORK     scratch dir for clones/logs (default: $REPO/.cloud-run)
#   CLONE_REMOTE         where the snapshot archive is read from
#   PUSH_REMOTE          where the archive and the pages branch are pushed to
#   SNAPSHOT_HISTORY     prior snapshots to stage (default 10: the report's
#                        rate-change look-back is 7; gate deltas need 1)
#   YF_IMPERSONATE       curl_cffi browser profile (default chrome116 here —
#                        the newer Chrome profiles are reset by the egress
#                        proxy; see data/yf_session.py)
#   SEC_EMAIL, FMP_API_KEY, TIINGO_API_KEY, FINNHUB_API_KEY, ANTHROPIC_API_KEY,
#   FRED_API_KEY         as in the Mac runbook; all optional but SEC_EMAIL
#   FORCE=1              run even when scripts/market_open.py says closed
#   DRY_RUN=1            do everything except push
#   SMOKE=1              tiny universe (SMOKE_TICKERS), for testing this script
set -uo pipefail

REPO="${STOCK_MODEL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORK="${STOCK_MODEL_WORK:-$REPO/.cloud-run}"
LOG="$WORK/logs"
STATUS="$WORK/status.txt"
GITHUB_URL="https://github.com/danmcooper-ops/stock-analysis-model.git"
CLONE_REMOTE="${CLONE_REMOTE:-$GITHUB_URL}"
PUSH_REMOTE="${PUSH_REMOTE:-$GITHUB_URL}"
SNAP_BRANCH="${SNAP_BRANCH:-data/snapshots}"
PAGES_BRANCH="${PAGES_BRANCH:-pages-live}"
PAGES_URL="${PAGES_URL:-https://danmcooper-ops.github.io/stock-analysis-model/}"
SNAPSHOT_HISTORY="${SNAPSHOT_HISTORY:-10}"
SMOKE="${SMOKE:-0}"
SMOKE_TICKERS="${SMOKE_TICKERS:-AAPL MSFT JPM XOM PLD AMGN CAT PG}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
BENCHMARKS="SPY QQQ IWM DIA XLK XLV XLF XLY XLP XLE XLI XLB XLU XLRE XLC"

export YF_IMPERSONATE="${YF_IMPERSONATE:-chrome116}"
export SEC_EMAIL="${SEC_EMAIL:-stockanalysis@example.com}"
# The cloud egress proxy re-terminates TLS; every client must trust its CA.
if [ -z "${SSL_CERT_FILE:-}" ] && [ -r /root/.ccr/ca-bundle.crt ]; then
  export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
fi
[ -n "${SSL_CERT_FILE:-}" ] && export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}"
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-danmcooper-ops}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-danmcooper@me.com}"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
export PYTHONUNBUFFERED=1

mkdir -p "$WORK" "$LOG" "$REPO/output"
: > "$STATUS"
cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }

RUN_T0=$(date +%s)
FAILED=0          # a blocking step failed
SOFT_FAILED=()    # non-blocking steps that failed (reported, not fatal)

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
    if [ "$blocking" = 1 ]; then FAILED=1; else SOFT_FAILED+=("$name"); fi
  fi
  return $rc
}

# push_with_retry DIR REFSPEC [extra git-push args]
push_with_retry() {
  local dir=$1 refspec=$2; shift 2
  if [ "$DRY_RUN" = 1 ]; then say "   DRY_RUN: not pushing $refspec"; return 0; fi
  local delay=2 rc=1
  for attempt in 1 2 3 4 5; do
    git -C "$dir" push "$@" "$PUSH_REMOTE" "$refspec" && return 0
    rc=$?
    [ "$attempt" = 5 ] && break
    say "   push failed (rc=$rc), retry in ${delay}s"; sleep "$delay"; delay=$(( delay * 2 ))
  done
  return $rc
}

# ---------------------------------------------------------------------------
# Preflight: was the market open? (exit 10 = closed -> skip the whole run)
# ---------------------------------------------------------------------------
say "cloud daily stock analysis — repo $REPO, work $WORK"
python3 scripts/market_open.py > "$LOG/00-preflight.log" 2>&1; gate_rc=$?
record preflight "$gate_rc" 0
cat "$LOG/00-preflight.log"
if [ "$gate_rc" = 10 ] && [ "$FORCE" != 1 ]; then
  echo "SKIPPED market closed: $(cat "$LOG/00-preflight.log")" >> "$STATUS"
  say "market closed — run skipped"; exit 0
fi
[ "$gate_rc" != 0 ] && [ "$gate_rc" != 10 ] && say "market gate itself failed (rc=$gate_rc) — failing open, continuing"

# ---------------------------------------------------------------------------
# 1. Python environment
# ---------------------------------------------------------------------------
PYTHON="$REPO/.venv/bin/python"
bootstrap_venv() {
  [ -x "$PYTHON" ] || python3 -m venv "$REPO/.venv" || return 1
  "$PYTHON" -c "import yfinance, pandas, duckdb, scipy, curl_cffi, openpyxl, jinja2" 2>/dev/null && return 0
  "$PYTHON" -m pip install -q -e "$REPO[dev]"
}
run_step 01-venv 1 bootstrap_venv || exit 1

# ---------------------------------------------------------------------------
# 2. Stage the snapshot archive: newest N days + the rating-history cache
# ---------------------------------------------------------------------------
SNAP="$WORK/snapshots-data"
stage_snapshots() {
  rm -rf "$SNAP"
  git clone -q --filter=blob:none --depth 1 --no-checkout --single-branch \
      -b "$SNAP_BRANCH" "$CLONE_REMOTE" "$SNAP" || return 1
  # Populate the index from HEAD's tree (no blobs needed) so a later
  # `git add <new file>` commits on top of the WHOLE tree, not an empty one.
  git -C "$SNAP" reset -q || return 1
  # Never `git ls-tree -l` here: sizes force every blob to download.
  "$PYTHON" - "$SNAP" "$REPO/output" "$SNAPSHOT_HISTORY" <<'PY'
import re, subprocess, sys
snap, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
names = subprocess.check_output(['git', '-C', snap, 'ls-tree', '--name-only', 'HEAD'], text=True).split()
names = [nm for nm in names if re.fullmatch(r'results_\d{4}-\d{2}-\d{2}\.json(\.gz)?', nm)]
if not names:
    sys.exit('no results_<date>.json[.gz] files on the archive branch')
by_date = {}
for nm in sorted(names):
    d = nm[8:18]
    if d not in by_date or not nm.endswith('.gz'):   # plain form wins, as in list_snapshot_files
        by_date[d] = nm
picked = [by_date[d] for d in sorted(by_date)[-n:]] if n > 0 else []
print(f"archive holds {len(by_date)} snapshot dates ({min(by_date)} .. {max(by_date)}); staging {len(picked)}")
for nm in picked:
    with open(f"{out}/{nm}", 'wb') as fh:
        subprocess.check_call(['git', '-C', snap, 'show', f'HEAD:{nm}'], stdout=fh)
    print('  staged', nm)
PY
  [ $? -eq 0 ] || return 1
  if git -C "$SNAP" cat-file -e HEAD:rating_history.json 2>/dev/null; then
    git -C "$SNAP" show HEAD:rating_history.json > "$REPO/output/rating_history.json" || return 1
    echo "staged rating_history.json ($(wc -c < "$REPO/output/rating_history.json") bytes)"
  else
    # First cloud run (or the cache was dropped): build the report's
    # rating-history cache over the WHOLE archive, one snapshot at a time,
    # so the rating history is not truncated to the staged days. ~5 s per
    # snapshot; the result is committed back with today's snapshot.
    echo "no rating_history.json in the archive — building it from every archived snapshot"
    "$PYTHON" - "$SNAP" "$REPO/output" <<'PY'
import os, re, subprocess, sys, time
sys.path.insert(0, '.')
from scripts.report_html import _load_rating_history
snap, out = sys.argv[1], sys.argv[2]
staged = {nm for nm in os.listdir(out) if nm.startswith('results_')}
names = subprocess.check_output(['git', '-C', snap, 'ls-tree', '--name-only', 'HEAD'], text=True).split()
by_date = {}
for nm in sorted(names):
    if re.fullmatch(r'results_\d{4}-\d{2}-\d{2}\.json(\.gz)?', nm):
        d = nm[8:18]
        if d not in by_date or not nm.endswith('.gz'):
            by_date[d] = nm
older = [by_date[d] for d in sorted(by_date) if by_date[d] not in staged]
t0 = time.time()
for nm in older:
    dest = os.path.join(out, nm)
    with open(dest, 'wb') as fh:
        subprocess.check_call(['git', '-C', snap, 'show', f'HEAD:{nm}'], stdout=fh)
    try:
        _load_rating_history(out, None)      # appends this day's change-points to the cache
    finally:
        os.remove(dest)
hist = _load_rating_history(out, None)       # fold the staged days in as well
print(f"rating history rebuilt over {len(older) + len(staged)} snapshots, "
      f"{len(hist)} tickers, {time.time() - t0:.0f}s")
PY
    [ $? -eq 0 ] || return 1
  fi
}
run_step 02-stage-snapshots 1 stage_snapshots || exit 1

PRIOR_DATE=$(ls "$REPO/output" | grep -E '^results_[0-9-]+\.json(\.gz)?$' | sed 's/results_//;s/\.json.*//' | sort | tail -1)
say "   newest prior snapshot: ${PRIOR_DATE:-none}"

# ---------------------------------------------------------------------------
# 3. Price cache: full history for every ticker in the newest snapshot
#    (they all re-enter Phase 2 via carry-forward) plus the benchmarks.
# ---------------------------------------------------------------------------
price_tickers() {
  if [ "$SMOKE" = 1 ]; then echo $SMOKE_TICKERS; else
  "$PYTHON" - "$REPO/output" <<'PY'
import sys
sys.path.insert(0, '.')
from data.snapshot_store import list_snapshot_files, read_snapshot, split_snapshot
files = list_snapshot_files(sys.argv[1])
tk = set()
if files:
    _, rows = split_snapshot(read_snapshot(files[-1][1]))
    tk = {r.get('ticker') for r in rows if isinstance(r, dict) and r.get('ticker')}
print(' '.join(sorted(tk)))
PY
  fi
  echo $BENCHMARKS
}
download_prices() {
  local tickers; tickers=$(price_tickers) || return 1
  echo "$(echo $tickers | wc -w) tickers to fetch"
  "$PYTHON" scripts/download_prices.py --output-dir output/prices --max-age-days 2 --tickers $tickers
}
run_step 03-prices 0 download_prices

# ---------------------------------------------------------------------------
# 4. The analysis (blocking)
# ---------------------------------------------------------------------------
RUNDATE=$(date +%F)     # analyze_stock names its outputs after date.today() at start
ANALYZE_ARGS=(--macro --prices-dir output/prices --universe us --min-spread 0 --mcap-min 300e6)
[ "$SMOKE" = 1 ] && ANALYZE_ARGS=(--macro --prices-dir output/prices --tickers $SMOKE_TICKERS)
run_step 04-analyze 1 "$PYTHON" scripts/analyze_stock.py "${ANALYZE_ARGS[@]}"
RESULTS="output/results_$RUNDATE.json"
HTML="output/stock_analysis_results_$RUNDATE.html"
if [ "$FAILED" = 1 ] || [ ! -s "$RESULTS" ] || [ ! -s "$HTML" ]; then
  say "analysis did not produce $RESULTS and $HTML — stopping"
  echo "RESULT FAILED at analyze" >> "$STATUS"; exit 1
fi

# ---------------------------------------------------------------------------
# 5. Enrichment (each non-blocking), price top-up for new entrants, re-render
# ---------------------------------------------------------------------------
run_step 05a-enrich-fdic     0 "$PYTHON" scripts/enrich_fdic.py "$RESULTS"
run_step 05b-enrich-reit     0 "$PYTHON" scripts/enrich_reit.py "$RESULTS"
run_step 05c-enrich-xbrl     0 "$PYTHON" scripts/enrich_xbrl.py "$RESULTS"
run_step 05d-enrich-pipeline 0 "$PYTHON" scripts/enrich_pipeline.py "$RESULTS"
topup_prices() {
  # Tickers that entered Phase 2 today without a cached parquet got a
  # Close-only stub from YFinanceClient; --max-age-days upgrades stubs and
  # skips everything already current, so this is cheap.
  local tickers; tickers=$("$PYTHON" - "$RESULTS" <<'PY'
import sys
sys.path.insert(0, '.')
from data.snapshot_store import read_snapshot, split_snapshot
_, rows = split_snapshot(read_snapshot(sys.argv[1]))
print(' '.join(sorted({r['ticker'] for r in rows if isinstance(r, dict) and r.get('ticker')})))
PY
  ) || return 1
  "$PYTHON" scripts/download_prices.py --output-dir output/prices --max-age-days 2 --tickers $tickers
}
run_step 05e-prices-topup 0 topup_prices
run_step 05f-rerender 1 "$PYTHON" scripts/rescore_and_render.py "$RESULTS"
if [ "$FAILED" = 1 ]; then echo "RESULT FAILED at rerender" >> "$STATUS"; exit 1; fi

# ---------------------------------------------------------------------------
# 6. Archive today's snapshot (+ the rating-history cache) to data/snapshots
# ---------------------------------------------------------------------------
archive_snapshot() {
  "$PYTHON" scripts/archive_snapshot.py "$RESULTS" --dest "$SNAP"; local rc=$?
  case $rc in
    0) ;;
    2) echo "ARCHIVE OVER THE 80 MiB HARD GUARD — not pushed; the snapshot needs a size fix"; return 2 ;;
    *) echo "archive failed (rc=$rc) — not pushed"; return 1 ;;
  esac
  cp "$REPO/output/rating_history.json" "$SNAP/rating_history.json" 2>/dev/null
  # Re-render already refreshed hist.json/rating_history.json for today; the
  # cache's last_scanned is the newest PRIOR day, so tomorrow's run only
  # parses today's file on top of it.
  #
  # Plumbing, not `git add` + `git commit`: in a blob-less clone the porcelain
  # commit lazily fetches the archive's blobs (gigabytes) before it writes
  # anything. hash-object/update-index/write-tree/commit-tree touch only the
  # objects being added, and the push sends only those.
  local blob tree commit
  for f in "results_$RUNDATE.json.gz" rating_history.json; do
    [ -s "$SNAP/$f" ] || continue
    blob=$(git -C "$SNAP" hash-object -w "$f") || return 1
    git -C "$SNAP" update-index --add --cacheinfo "100644,$blob,$f" || return 1
  done
  # --missing-ok: without it write-tree verifies every index entry's blob
  # exists locally, which lazily downloads the whole archive.
  tree=$(git -C "$SNAP" write-tree --missing-ok) || return 1
  if [ "$tree" = "$(git -C "$SNAP" rev-parse 'HEAD^{tree}')" ]; then
    echo "archive already holds today's snapshot"; return 0
  fi
  commit=$(git -C "$SNAP" commit-tree "$tree" -p HEAD -m "Snapshot: $RUNDATE") || return 1
  git -C "$SNAP" update-ref "refs/heads/$SNAP_BRANCH" "$commit" || return 1
  echo "archive commit: $(git -C "$SNAP" log --oneline -1 "$commit")"
  git -C "$SNAP" diff-tree --name-status "HEAD~1" HEAD   # (--stat would fetch blobs)
  push_with_retry "$SNAP" "refs/heads/$SNAP_BRANCH:refs/heads/$SNAP_BRANCH"
}
run_step 06-archive 1 archive_snapshot
ARCHIVE_RC=$?

# ---------------------------------------------------------------------------
# 7. Reports for the run summary (non-blocking)
# ---------------------------------------------------------------------------
run_step 07a-portfolio-report 0 "$PYTHON" scripts/portfolio_report.py --results-dir output/ --prices-dir output/prices
run_step 07b-gate-na-report   0 "$PYTHON" scripts/gate_na_report.py "$RESULTS"
run_step 07c-validate-ratings 0 "$PYTHON" scripts/validate_ratings.py --snapshot "$RESULTS" --prices-dir output/prices

# ---------------------------------------------------------------------------
# 8. Publish: rebuild pages-live as one fresh commit and force-push it
# ---------------------------------------------------------------------------
PAGES="$WORK/pages"
publish_pages() {
  rm -rf "$PAGES"; mkdir -p "$PAGES/docs" "$PAGES/.github/workflows"
  cp "$HTML" "$PAGES/docs/index.html" || return 1
  for f in prices_meta.json hist.json details.json; do
    cp "$REPO/output/$f" "$PAGES/docs/$f" || { echo "missing sidecar $f"; return 1; }
  done
  if [ -s "$REPO/output/macro.json" ]; then cp "$REPO/output/macro.json" "$PAGES/docs/macro.json"
  else echo "no macro.json this run — the Macro Outlook tab is absent, by design"; fi
  STOCK_MODEL_REPO="$REPO" PAGES_DOCS="$PAGES/docs" "$PYTHON" scripts/publish_vol_shards.py || return 1
  # The deploy workflow must live on the branch itself for the push trigger.
  cp "$REPO/.github/workflows/deploy-pages.yml" "$PAGES/.github/workflows/deploy-pages.yml" || return 1
  printf 'docs/vol/* *.json\ndocs/px/* *.json\n' > "$PAGES/.gitignore"
  git -C "$PAGES" init -q -b "$PAGES_BRANCH" || return 1
  git -C "$PAGES" add -A && git -C "$PAGES" commit -q -m "Pages: $RUNDATE" || return 1
  echo "pages commit: $(git -C "$PAGES" rev-parse --short HEAD), $(git -C "$PAGES" ls-files | wc -l) files"
  push_with_retry "$PAGES" "HEAD:refs/heads/$PAGES_BRANCH" --force || return 1
  [ "$DRY_RUN" = 1 ] && return 0
  # GitHub Pages redeploys from the push; give it a few minutes.
  # Fetch to a file, then grep: with pipefail, `curl | grep -q` fails on the
  # SIGPIPE grep -q sends curl after the first match.
  for i in $(seq 1 20); do
    sleep 30
    curl -sSL --max-time 30 -o "$WORK/live.html" "$PAGES_URL" 2>/dev/null || true
    if grep -q "$RUNDATE" "$WORK/live.html" 2>/dev/null; then
      echo "live: $PAGES_URL serves the $RUNDATE report"; return 0
    fi
  done
  echo "WARNING: $PAGES_URL did not show $RUNDATE within 10 minutes — check the deploy-pages workflow"
  return 1
}
run_step 08-publish 0 publish_pages
PUBLISH_RC=$?

# ---------------------------------------------------------------------------
# Wrap up
# ---------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - RUN_T0 ))
{
  echo "RUNDATE $RUNDATE"
  echo "PRIOR_SNAPSHOT ${PRIOR_DATE:-none}"
  echo "ELAPSED_SECONDS $ELAPSED"
  echo "SOFT_FAILURES ${SOFT_FAILED[*]:-none}"
  if [ "$ARCHIVE_RC" = 0 ] && [ "$PUBLISH_RC" = 0 ]; then echo "RESULT OK"
  elif [ "$ARCHIVE_RC" = 0 ]; then echo "RESULT OK-BUT-PUBLISH-FAILED"
  else echo "RESULT FAILED at archive (publish rc=$PUBLISH_RC)"; fi
} >> "$STATUS"
say "done in ${ELAPSED}s"; cat "$STATUS"
[ "$ARCHIVE_RC" = 0 ] || exit 1
exit 0
