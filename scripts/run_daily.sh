#!/bin/bash
# Nightly stock-analysis pipeline, end to end, in one deterministic script.
#
# Replaces the step-by-step commands that used to live only in
# scheduled-tasks/daily-stock-analysis/SKILL.md. When Claude drove each step
# by hand, the chain broke easily: on 2026-09-09 the analysis finished after
# midnight, the $(date)-named enrichment paths pointed at a file that did not
# exist, and the snapshot was never archived or published. Here RUNDATE is
# fixed once, every step's exit code is recorded, and the scheduled task only
# launches this script and summarizes run_summary_<RUNDATE>.json.
#
# Usage:
#   scripts/run_daily.sh                      full run (preflight → publish)
#   scripts/run_daily.sh --from enrich --date 2026-09-09
#                                             resume an existing snapshot from a step
#   scripts/run_daily.sh --from publish --date 2026-09-09   republish only
#   scripts/run_daily.sh --from analyze --date 2026-09-11   re-run a failed session the
#                                             next day (an interrupted analysis resumes
#                                             from output/.checkpoint)
#   scripts/run_daily.sh --dry-run            print the plan, run nothing
#   scripts/run_daily.sh --force              skip the market-open gate
#
# Steps, in order (names are valid --from values):
#   preflight  market_open.py gate (exit 10 = closed → whole run skipped)
#   prices     refresh output/prices parquets              (non-blocking)
#   pull       fast-forward main so the render uses merged templates (non-blocking)
#   analyze    analyze_stock.py                            (BLOCKING), then the
#              companyfacts cache backup to Supabase Storage (non-blocking)
#   enrich     FDIC, REIT, XBRL, FDA enrichment, price top-up for new entrants,
#              price cache backup, re-render, portfolio alerts (non-blocking each)
#   archive    Supabase publish (non-blocking unless DB_PRIMARY=1), then gzip
#              snapshot + state files to data/snapshots, commit, push
#              (BLOCKING for publish)
#   reports    portfolio, gate N/A, momentum, store sync check, database
#              night check (non-blocking)
#   publish    copy artifacts to pages-live, amend, force-push, verify, then
#              Cloudflare Pages when configured (non-blocking)
#   compact    gzip aged output/ artifacts, keeping the newest 5 snapshots plain
#              (non-blocking; scripts/compact_output.py)
#
# Kept in step with scheduled-tasks/cloud-daily-stock-analysis/run.sh, which
# ran alone from 2026-09-10 until the Mac mini move (MAC-MINI-SETUP.md). The
# Supabase and Cloudflare steps only run when their secrets are set (.env or
# the environment), so a dev box behaves as before.
#
# Exit codes: 0 success or market-closed skip; 1 a blocking step failed;
# 3 finished, but at least one non-blocking step failed.
set -uo pipefail

# The checkout this script lives in; STOCK_MODEL_REPO overrides.
REPO="${STOCK_MODEL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# The pipeline's interpreter lives OUTSIDE the repo so deleting or recreating
# a worktree cannot take it with it (the phase-1-api worktree's .venv, which
# every routine used, was removed on 2026-09-13 mid-run). Rebuild it with
# scheduled-tasks/RECOVERY.md step 3; override with PYTHON=... .
VPY="${PYTHON:-$HOME/.venvs/stock-model/bin/python}"
PAGES_WT="$REPO/.claude/worktrees/pages-live"
SNAP_WT="$REPO/.claude/worktrees/snapshots-data"
PAGES_URL="https://danmcooper-ops.github.io/stock-analysis-model/"
LOGDIR="$HOME/Library/Logs/StockModel"
STEPS=(preflight prices pull analyze enrich archive reports publish compact)
# Cloudflare Pages response headers and the deploy workflow, copied onto
# pages-live as the cloud publish does.
PAGES_HEADERS="$REPO/scheduled-tasks/cloud-daily-stock-analysis/pages_headers"
WRANGLER_VERSION="${WRANGLER_VERSION:-4.141.0}"
DB_PRIMARY="${DB_PRIMARY:-0}"
# The snapshot is rewritten several times tonight (enrich steps, re-render);
# with a database backend selected, sync_snapshot_file would republish each
# time. The archive step publishes once instead. Readers stay on the local
# DuckDB store unless SNAPSHOT_STORE_BACKEND says otherwise: unlike the cloud
# container, this machine keeps it warm.
export DB_DEFER_PUBLISH=1
# Snapshots left as plain JSON by the compact step. A --from resume of a date
# older than these finds only results_<date>.json.gz; it is restored below.
KEEP_PLAIN=5

# ---------------------------------------------------------------- arguments
FROM="preflight"; RUNDATE=""; DRY_RUN=0; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from)    FROM="$2"; shift 2 ;;
    --date)    RUNDATE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force)   FORCE=1; shift ;;
    -h|--help) sed -n '2,47p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case " ${STEPS[*]} " in *" $FROM "*) ;; *) echo "--from must be one of: ${STEPS[*]}" >&2; exit 2 ;; esac
if [ -n "$RUNDATE" ] && ! [[ "$RUNDATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "--date must be YYYY-MM-DD" >&2; exit 2
fi
step_index() { local i; for i in "${!STEPS[@]}"; do [ "${STEPS[$i]}" = "$1" ] && echo "$i"; done; }
FROM_I=$(step_index "$FROM")
if [ "$FROM_I" -gt "$(step_index analyze)" ] && [ -z "$RUNDATE" ]; then
  echo "--from $FROM needs --date YYYY-MM-DD (the snapshot to resume)" >&2; exit 2
fi
# New York dates, as the archive carries them, whatever the machine's zone.
export TZ="${TZ:-America/New_York}"
# Fixed once, at start: a 12-20 h run crosses midnight.
RUNDATE="${RUNDATE:-$(date +%F)}"

# Keep the Mac awake for the whole run (re-exec under caffeinate once). The
# 2026-09-09 run lost 85 minutes to a sleep mid-analysis. -s only holds on AC
# power; a battery run is flagged in the summary.
if [ "$DRY_RUN" = 0 ] && [ -z "${RUN_DAILY_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export RUN_DAILY_CAFFEINATED=1
  exec caffeinate -ims "$0" --from "$FROM" --date "$RUNDATE" \
       $([ "$FORCE" = 1 ] && echo --force)
fi

cd "$REPO" || { echo "repo not found: $REPO" >&2; exit 1; }
mkdir -p "$LOGDIR"
SNAPSHOT="output/results_$RUNDATE.json"
HTML="output/stock_analysis_results_$RUNDATE.html"
SUMMARY="output/run_summary_$RUNDATE.json"
STEPLOG="$(mktemp -t run_daily_steps)"
NOTES="$(mktemp -t run_daily_notes)"

note() { echo "[run_daily] $*"; printf '%s\n' "$*" >>"$NOTES"; }
trap 'rm -f "$STEPLOG" "$NOTES"' EXIT

if [ "$DRY_RUN" = 0 ]; then
  exec > >(tee -a "$LOGDIR/daily_$RUNDATE.log") 2>&1

  # One run at a time: an overrunning night must not collide with the next.
  LOCK="output/.run_daily.lock"
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "[run_daily] another run is in progress (pid $(cat "$LOCK")); exiting"
    exit 1
  fi
  echo $$ >"$LOCK"
  trap 'rm -f "$LOCK" "$STEPLOG" "$NOTES"' EXIT
fi

SSL_CERT_FILE="$("$VPY" -c 'import certifi; print(certifi.where())')"
export SSL_CERT_FILE REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
# enrich_xbrl and the other step scripts do not parse .env themselves, and
# the Supabase/Cloudflare/cache steps read their secrets from the
# environment. Load it with analyze_stock.py's rules: KEY=VALUE lines,
# # comments, a value already in the environment wins.
if [ -f .env ]; then
  eval "$("$VPY" - .env <<'PYEOF'
import os, re, shlex, sys
for line in open(sys.argv[1], encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    k, v = k.strip(), v.strip()
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) and k not in os.environ:
        print(f'export {k}={shlex.quote(v)}')
PYEOF
)"
fi
export SEC_EMAIL="${SEC_EMAIL:-}"

echo "=== run_daily start $(date) RUNDATE=$RUNDATE from=$FROM dry_run=$DRY_RUN ==="
[ -z "$SEC_EMAIL" ] && note "SEC_EMAIL is not set (.env) — SEC requests use a placeholder User-Agent"
if pmset -g batt 2>/dev/null | grep -q "Battery Power"; then
  note "running on battery — the Mac can still sleep; plug in for unattended runs"
fi

# The compact step gzips all but the newest $KEEP_PLAIN snapshots, but the
# enrich/render scripts rewrite results_<date>.json in place. Resuming an
# older date therefore restores the plain file first (gunzip checks the CRC)
# and drops the .gz, so a later compact does not find two differing copies.
if [ "$FROM_I" -ge "$(step_index analyze)" ] && [ ! -f "$SNAPSHOT" ] && [ -f "$SNAPSHOT.gz" ]; then
  if [ "$DRY_RUN" = 1 ]; then
    note "would restore $SNAPSHOT from $SNAPSHOT.gz"
  elif gunzip -c "$SNAPSHOT.gz" >"$SNAPSHOT.tmp" && mv "$SNAPSHOT.tmp" "$SNAPSHOT"; then
    rm -f "$SNAPSHOT.gz"
    note "restored $SNAPSHOT from its compacted .gz"
  else
    rm -f "$SNAPSHOT.tmp"
    note "could not restore $SNAPSHOT from $SNAPSHOT.gz"
  fi
fi

# ---------------------------------------------------------------- helpers
BLOCKED=0      # set when a blocking step failed: later steps are skipped
SOFT_FAIL=0

# run NAME BLOCKING CMD... — run one command, record rc and duration.
run() {
  local name="$1" blocking="$2"; shift 2
  if [ "$BLOCKED" = 1 ]; then
    printf '%s\t%s\t%s\t%s\n' "$name" "skipped" 0 "$blocking" >>"$STEPLOG"; return 0
  fi
  echo; echo "--- [$name] $(date +%T): $*"
  if [ "$DRY_RUN" = 1 ]; then
    printf '%s\t%s\t%s\t%s\n' "$name" "dry-run" 0 "$blocking" >>"$STEPLOG"; return 0
  fi
  local t0 rc; t0=$(date +%s)
  "$@"; rc=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$blocking" >>"$STEPLOG"
  echo "--- [$name] rc=$rc"
  if [ "$rc" != 0 ]; then
    if [ "$blocking" = blocking ]; then BLOCKED=1; else SOFT_FAIL=1; fi
  fi
  return "$rc"
}
wants() { [ "$FROM_I" -le "$(step_index "$1")" ]; }
has_supabase() { [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; }

write_summary() {
  local status="$1"
  [ "$DRY_RUN" = 1 ] && { echo "=== dry run: $status ==="; return; }
  "$VPY" - "$STEPLOG" "$NOTES" "$SUMMARY" "$RUNDATE" "$status" <<'PYEOF'
import json, sys, datetime
steplog, notes, out, rundate, status = sys.argv[1:6]
steps = []
for line in open(steplog, encoding='utf-8'):
    name, rc, secs, blocking = line.rstrip('\n').split('\t')
    steps.append({'step': name, 'rc': int(rc) if rc.lstrip('-').isdigit() else rc,
                  'seconds': int(secs), 'blocking': blocking == 'blocking'})
doc = {'rundate': rundate, 'status': status,
       'finished_at': datetime.datetime.now().isoformat(timespec='seconds'),
       'total_seconds': sum(s['seconds'] for s in steps),
       'steps': steps,
       'notes': [n for n in open(notes, encoding='utf-8').read().splitlines() if n]}
with open(out, 'w', encoding='utf-8') as f:
    json.dump(doc, f, indent=2)
print(f'[run_daily] summary -> {out}')
PYEOF
}

finish() {
  local status="$1" code="$2"
  write_summary "$status"
  echo "=== run_daily end $(date) status=$status rc=$code ==="
  exit "$code"
}

# ---------------------------------------------------------------- preflight
if wants preflight; then
  if [ "$FORCE" = 1 ]; then
    note "market-open gate bypassed (--force)"
  else
    run preflight gate "$VPY" scripts/market_open.py --date "$RUNDATE"
    rc=$(tail -1 "$STEPLOG" | cut -f2)
    if [ "$rc" = 10 ]; then
      note "market closed — run skipped"
      finish skipped 0
    elif [ "$rc" != 0 ] && [ "$rc" != dry-run ]; then
      # Fail open: a gate bug costs one wasted run; failing closed would
      # silently stop the corpus growing.
      note "market_open.py failed (rc=$rc) — continuing anyway"
      SOFT_FAIL=1
    fi
  fi
fi

# ---------------------------------------------------------------- prices
if wants prices; then
  # shellcheck disable=SC2046
  run prices soft "$VPY" scripts/download_prices.py --output-dir output/prices --max-age-days 2 \
      --tickers $(ls output/prices/ | sed -n 's/\.parquet$//p') SPY QQQ IWM DIA
fi

# ---------------------------------------------------------------- pull
if wants pull; then
  branch="$(git rev-parse --abbrev-ref HEAD)"
  if [ "$branch" = main ]; then
    run pull soft git pull --ff-only origin main \
      || note "git pull --ff-only failed — rendering from the current checkout (stale templates possible)"
  else
    note "checkout is on '$branch', not main — skipped the fast-forward; the render uses this branch's templates"
  fi
fi

# ---------------------------------------------------------------- analyze
if wants analyze; then
  run analyze blocking bash -c "set -o pipefail; \"$VPY\" scripts/analyze_stock.py --run-date $RUNDATE --macro --prices-dir output/prices \
      --universe us --min-spread 0 --mcap-min 300e6 2>&1 | tee output/run_$RUNDATE.log"
  if [ "$BLOCKED" = 1 ]; then
    note "analyze_stock.py failed — nothing downstream was run"
    finish failed 1
  fi
  if [ "$DRY_RUN" = 0 ] && [ ! -f "$SNAPSHOT" ]; then
    note "analyze_stock.py exited 0 but $SNAPSHOT does not exist"
    finish failed 1
  fi
  # Off-site copy of the companyfacts cache (cloud step 04b): straight after
  # the analysis, the only step that fetches or evicts facts. Evictions are
  # deleted from the bucket too, and the store refuses a local set far
  # smaller than what it holds. Only with the Supabase secrets.
  if has_supabase; then
    run sec_cache_save soft "$VPY" scripts/sec_cache.py save
  fi
fi

if [ "$DRY_RUN" = 0 ] && [ ! -f "$SNAPSHOT" ]; then
  note "snapshot $SNAPSHOT not found"
  finish failed 1
fi

# ---------------------------------------------------------------- enrich
prices_topup() {
  local tickers
  tickers=$("$VPY" - "$SNAPSHOT" <<'PYEOF'
import sys
sys.path.insert(0, '.')
from data.snapshot_store import read_snapshot, split_snapshot
_, rows = split_snapshot(read_snapshot(sys.argv[1]))
print(' '.join(sorted({r['ticker'] for r in rows if isinstance(r, dict) and r.get('ticker')})))
PYEOF
  ) || return 1
  [ -n "$tickers" ] || { echo "prices_topup: no tickers in $SNAPSHOT"; return 1; }
  # shellcheck disable=SC2086
  "$VPY" scripts/download_prices.py --output-dir output/prices --max-age-days 2 --tickers $tickers
}

if wants enrich; then
  run enrich_fdic     soft "$VPY" scripts/enrich_fdic.py     "$SNAPSHOT"
  run enrich_reit     soft "$VPY" scripts/enrich_reit.py     "$SNAPSHOT"
  run enrich_xbrl     soft "$VPY" scripts/enrich_xbrl.py     "$SNAPSHOT"
  run enrich_pipeline soft "$VPY" scripts/enrich_pipeline.py "$SNAPSHOT"
  # Tickers that entered Phase 2 tonight without a parquet got a Close-only
  # stub; --max-age-days upgrades stubs and skips everything already current.
  run prices_topup    soft prices_topup
  if has_supabase; then
    # Off-site copy of the price parquets (cloud step 05e2); only files
    # whose size changed are uploaded.
    run price_cache_save soft "$VPY" scripts/price_cache.py save --prices-dir output/prices
  fi
  # Must follow every enrichment so the banners pick the fields up.
  run render          soft "$VPY" scripts/rescore_and_render.py "$SNAPSHOT" --prices-dir output/prices
  # Your portfolio groupings (portfolio/portfolios.json): stats + change
  # alerts. Before the archive, so portfolio_alerts.json rides into
  # data/snapshots, where .github/workflows/portfolio-alerts.yml reads it.
  run portfolio_alerts soft "$VPY" scripts/portfolios.py alerts --results-dir output --date "$RUNDATE" \
    --out "output/portfolio_alerts_$RUNDATE.txt" --json output/portfolio_alerts.json \
    --markdown "output/portfolio_alerts_$RUNDATE.md" --pages-url "$PAGES_URL"
fi

# ---------------------------------------------------------------- archive
# Supabase publish (cloud step 06a). Skipped without the secrets. With
# DB_PRIMARY=1 (the P6 cutover) a failure fails the run, but never stops the
# git archive below, so the day is never lost.
db_publish() {
  if ! has_supabase; then
    if [ "$DB_PRIMARY" = 1 ]; then
      echo "DB_PRIMARY=1 but SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY are not set"; return 1
    fi
    echo "SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY not set; skipping"; return 0
  fi
  "$VPY" scripts/db_publish.py "$SNAPSHOT"
}

# The weekly backtest commits to data/snapshots too, from its own clone, so
# the worktree can be behind the remote. Fast-forward before writing into it.
archive_sync() {
  git -C "$SNAP_WT" pull -q --ff-only origin data/snapshots
}

# The state files that ride with the snapshot. The Mac keeps them locally, but
# the branch is their off-machine record: the weekly backtest, RECOVERY.md and
# a move back to the cloud all read them there, and the portfolio-alerts
# workflow reads portfolio_alerts.json from it.
archive_state_files() {
  local f
  cp output/rating_history.json "$SNAP_WT/" 2>/dev/null || true
  for f in portfolio_nav.json portfolio_alerts.json; do
    [ -s "output/$f" ] && cp "output/$f" "$SNAP_WT/$f"
  done
  # Guarded on size: a near-empty skip list must not replace the learned one.
  if [ "$(wc -c < data/cache/screen_skip.json 2>/dev/null || echo 0)" -gt 10000 ]; then
    cp data/cache/screen_skip.json "$SNAP_WT/screen_skip.json"
  fi
  git -C "$SNAP_WT" add "results_$RUNDATE.json.gz" blobs || return 1
  for f in rating_history.json portfolio_nav.json portfolio_alerts.json screen_skip.json; do
    [ -f "$SNAP_WT/$f" ] && { git -C "$SNAP_WT" add "$f" || return 1; }
  done
  return 0
}

# A rejected push is most likely the weekly backtest's commit landing first:
# rebase on the new tip (the two never touch the same files) and retry.
archive_push() {
  local i
  for i in 1 2 3; do
    git -C "$SNAP_WT" push -q origin data/snapshots && return 0
    [ "$i" = 3 ] && break
    echo "archive: push failed (attempt $i); rebasing on the remote tip"
    if ! git -C "$SNAP_WT" pull -q --rebase origin data/snapshots; then
      git -C "$SNAP_WT" rebase --abort 2>/dev/null
      return 1
    fi
    sleep $(( i * 10 ))
  done
  return 1
}

DB_RC=""
ARCHIVED=1
if wants archive; then
  run db_publish soft db_publish; DB_RC=$?
  [ "$DRY_RUN" = 1 ] && DB_RC=0
  if [ "$DB_RC" != 0 ]; then
    if [ "$DB_PRIMARY" = 1 ]; then note "db_publish failed with DB_PRIMARY=1 — the run is failed (the git archive still runs)"
    else note "db_publish failed (non-blocking until DB_PRIMARY=1)"; fi
  fi
  ARCHIVED=0
  run archive_sync soft archive_sync \
    || note "archive: could not fast-forward the data/snapshots worktree; the push will rebase"
  run archive_snapshot blocking "$VPY" scripts/archive_snapshot.py "$SNAPSHOT" --dest "$SNAP_WT"
  rc=$(tail -1 "$STEPLOG" | cut -f2)
  case "$rc" in
    0|dry-run)
      # blobs/: the snapshot's edgar_history is stored there by hash; only
      # the histories that changed since the last run are new files.
      run archive_add blocking archive_state_files
      if [ "$DRY_RUN" = 1 ] || ! git -C "$SNAP_WT" diff --cached --quiet; then
        run archive_commit blocking git -C "$SNAP_WT" commit -q -m "Snapshot: $RUNDATE"
      else
        note "archive: results_$RUNDATE.json.gz unchanged on data/snapshots — nothing to commit"
      fi
      run archive_push blocking archive_push
      [ "$BLOCKED" = 0 ] && ARCHIVED=1 ;;
    2) note "archive: snapshot breached the 80 MiB guard — NOT pushed (size fix needed)" ;;
    *) note "archive: archive_snapshot.py failed (rc=$rc) — NOT pushed" ;;
  esac
  # Reports and publish still run: the local snapshot is valid; only the
  # corpus copy is missing, and that is reported as the run's failure.
  BLOCKED=0
fi

# ---------------------------------------------------------------- reports
if wants reports; then
  run portfolio_report soft "$VPY" scripts/portfolio_report.py --results-dir output/ --prices-dir output/prices
  # portfolio_alerts moved to the enrich step, before the archive.
  run gate_na_report   soft "$VPY" scripts/gate_na_report.py "$SNAPSHOT"
  run validate_ratings soft "$VPY" scripts/validate_ratings.py --snapshot "$SNAPSHOT" --prices-dir output/prices
  # The store's syncs never fail a step; this is where a store that stopped
  # keeping up with the snapshot shows as a failure.
  run store_check      soft "$VPY" scripts/check_snapshot_store.py --results-dir output --date "$RUNDATE" \
    || note "snapshot store is out of step with results_$RUNDATE.json (see the store_check output)"
  # The night's database verdict (cloud step 07e): published, row count and
  # SHA, rating-history parity, recorded in core.night_checks for the
  # DB_CUTOVER_STREAK. Needs the archive step's publish rc, so only when this
  # invocation ran it.
  if has_supabase && [ -n "$DB_RC" ]; then
    run db_check soft "$VPY" scripts/db_night_check.py record --date "$RUNDATE" --publish-rc "$DB_RC" \
      --results-dir output --status-file "$LOGDIR/db_check_$RUNDATE.txt"
  fi
fi

# ---------------------------------------------------------------- publish
publish() {
  local docs="$PAGES_WT/docs" f
  [ -f "$HTML" ] || { echo "publish: $HTML missing"; return 1; }
  [ -d "$docs" ] || { echo "publish: $docs missing (git worktree add .claude/worktrees/pages-live pages-live)"; return 1; }
  # Written every time, as the cloud publish does: the shard families are
  # tracked by manifest and conflict copies (a space in the name) never are;
  # _worker.js carries the Access config and is deployed to Cloudflare only.
  printf 'docs/vol/* *.json\ndocs/px/* *.json\ndocs/hist/* *.json\ndocs/details/* *.json\ndocs/_worker.js\n' \
    > "$PAGES_WT/.gitignore" || return 1
  cp "$HTML" "$docs/index.html" || return 1
  cp "$PAGES_HEADERS" "$docs/_headers" || return 1   # Cloudflare headers (inert on GitHub Pages)
  # The deploy workflow must live on the branch itself for the push trigger.
  mkdir -p "$PAGES_WT/.github/workflows" \
    && cp .github/workflows/deploy-pages.yml "$PAGES_WT/.github/workflows/deploy-pages.yml" || return 1
  for f in prices_meta.json hist_index.json details_index.json; do
    cp "output/$f" "$docs/$f" || { echo "publish: required sidecar output/$f missing"; return 1; }
  done
  # macro.json is optional; never publish yesterday's under today's HTML.
  if [ -f output/macro.json ]; then cp output/macro.json "$docs/macro.json" || return 1
  else rm -f "$docs/macro.json"; fi
  rm -f "$docs/prices.json"   # retired 2026-08-11
  rm -f "$docs/hist.json" "$docs/details.json"   # split into hist/ and details/ (P4c)
  "$VPY" scripts/publish_vol_shards.py || { echo "publish: shard sync failed"; return 1; }
  git -C "$PAGES_WT" add -A || return 1
  if git -C "$PAGES_WT" rev-parse -q --verify HEAD >/dev/null; then
    git -C "$PAGES_WT" commit -q --amend -m "Pages: $RUNDATE" || return 1
  else
    git -C "$PAGES_WT" commit -q -m "Pages: $RUNDATE" || return 1
  fi
  git -C "$PAGES_WT" push -q --force origin pages-live || return 1
  local code
  for _ in 1 2 3 4 5 6; do
    sleep 60
    code="$(curl -sS -o /dev/null -w '%{http_code}' -L "$PAGES_URL")"
    [ "$code" = 200 ] && { echo "publish: $PAGES_URL -> 200"; return 0; }
  done
  echo "publish: $PAGES_URL returned $code after push"; return 1
}

# The same docs/ to Cloudflare Pages (cloud step 08b), behind the Access
# login (cloudflare/README.md). Refuses to deploy without the Access config,
# and fails if an anonymous request gets the report.
publish_cloudflare() {
  local docs="$PAGES_WT/docs" url code i live
  [ -s "$docs/index.html" ] || { echo "no $docs/index.html — the GitHub publish did not build the site"; return 1; }
  "$VPY" scripts/stage_pages_worker.py "$docs" \
    || { echo "refusing to deploy: set CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD (cloudflare/README.md)"; return 1; }
  "$VPY" scripts/check_pages_limits.py "$docs" || { rm -f "$docs/_worker.js"; return 1; }
  command -v npx >/dev/null || { rm -f "$docs/_worker.js"; echo "npx not found — install Node.js for wrangler"; return 1; }
  npx -y "wrangler@$WRANGLER_VERSION" pages deploy "$docs" --project-name "$CF_PAGES_PROJECT" \
    --branch main --commit-dirty=true --commit-message "Pages: $RUNDATE"
  code=$?; rm -f "$docs/_worker.js"
  [ "$code" = 0 ] || return 1
  url="${CF_PAGES_URL:-https://$CF_PAGES_PROJECT.pages.dev/}"
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 "$url" 2>/dev/null || true)"
  case "$code" in
    200) echo "ERROR: $url serves the report WITHOUT a login — check the Access application (cloudflare/README.md)"; return 1 ;;
    302|303|401|403) echo "anonymous request refused as expected ($code)" ;;
    *) echo "WARNING: anonymous request to $url returned $code" ;;
  esac
  if [ -z "${CF_ACCESS_CLIENT_ID:-}" ] || [ -z "${CF_ACCESS_CLIENT_SECRET:-}" ]; then
    echo "WARNING: deployed, but CF_ACCESS_CLIENT_ID/CF_ACCESS_CLIENT_SECRET are unset, so the live check cannot sign in"
    return 1
  fi
  live="$(mktemp -t run_daily_cf)"
  for i in $(seq 1 10); do
    # The service token goes in a curl config on stdin, never argv.
    printf 'header = "CF-Access-Client-Id: %s"\nheader = "CF-Access-Client-Secret: %s"\n' \
        "$CF_ACCESS_CLIENT_ID" "$CF_ACCESS_CLIENT_SECRET" \
      | curl -sS --max-time 30 -K - -o "$live" "$url" 2>/dev/null || true
    if grep -q "$RUNDATE" "$live" 2>/dev/null; then
      rm -f "$live"; echo "live: $url serves the $RUNDATE report"; return 0
    fi
    sleep 30
  done
  rm -f "$live"
  echo "WARNING: $url did not show $RUNDATE within 5 minutes of the deploy"
  return 1
}

if wants publish; then
  run publish soft publish || note "publish failed — retry with: scripts/run_daily.sh --from publish --date $RUNDATE"
  if [ -n "${CLOUDFLARE_API_TOKEN:-}" ] && [ -n "${CLOUDFLARE_ACCOUNT_ID:-}" ] && [ -n "${CF_PAGES_PROJECT:-}" ]; then
    run publish_cloudflare soft publish_cloudflare \
      || note "Cloudflare publish failed — the GitHub Pages site is unaffected"
  fi
fi

# ---------------------------------------------------------------- compact
# Last, so nothing later in the run reads a file this gzips. Every snapshot
# reader accepts the .gz form; see scripts/compact_output.py.
if wants compact; then
  run compact_output soft "$VPY" scripts/compact_output.py --results-dir output \
      --keep-plain "$KEEP_PLAIN" --apply
fi

# ---------------------------------------------------------------- done
if [ "$ARCHIVED" = 0 ]; then
  finish failed 1
elif [ "$DB_PRIMARY" = 1 ] && [ -n "$DB_RC" ] && [ "$DB_RC" != 0 ]; then
  finish failed 1
elif [ "$SOFT_FAIL" = 1 ]; then
  finish degraded 3
else
  finish ok 0
fi
