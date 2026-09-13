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
#   scripts/run_daily.sh --dry-run            print the plan, run nothing
#   scripts/run_daily.sh --force              skip the market-open gate
#
# Steps, in order (names are valid --from values):
#   preflight  market_open.py gate (exit 10 = closed → whole run skipped)
#   prices     refresh output/prices parquets              (non-blocking)
#   pull       fast-forward main so the render uses merged templates (non-blocking)
#   analyze    analyze_stock.py                            (BLOCKING)
#   enrich     FDIC, REIT, XBRL, FDA enrichment + re-render (non-blocking each)
#   archive    gzip snapshot to data/snapshots, commit, push (BLOCKING for publish)
#   reports    portfolio, gate N/A, momentum check          (non-blocking)
#   publish    copy artifacts to pages-live, amend, force-push, verify (non-blocking)
#
# Exit codes: 0 success or market-closed skip; 1 a blocking step failed;
# 3 finished, but at least one non-blocking step failed.
set -uo pipefail

REPO="/Users/danmcooper/Projects/Workspace Folder"
# The repo-root .venv lacks duckdb/scipy; this is the interpreter every
# routine has used. Override with PYTHON=... if the venv moves.
VPY="${PYTHON:-$REPO/.claude/worktrees/phase-1-api/.venv/bin/python}"
PAGES_WT="$REPO/.claude/worktrees/pages-live"
SNAP_WT="$REPO/.claude/worktrees/snapshots-data"
PAGES_URL="https://danmcooper-ops.github.io/stock-analysis-model/"
LOGDIR="$HOME/Library/Logs/StockModel"
STEPS=(preflight prices pull analyze enrich archive reports publish)

# ---------------------------------------------------------------- arguments
FROM="preflight"; RUNDATE=""; DRY_RUN=0; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from)    FROM="$2"; shift 2 ;;
    --date)    RUNDATE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force)   FORCE=1; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
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
# Fixed once, at start: a 3-6 h run crosses midnight.
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
# enrich_xbrl and the other step scripts do not parse .env themselves.
if [ -z "${SEC_EMAIL:-}" ] && [ -f .env ]; then
  SEC_EMAIL="$(grep -E '^SEC_EMAIL=' .env | tail -1 | cut -d= -f2-)"
fi
export SEC_EMAIL="${SEC_EMAIL:-}"

echo "=== run_daily start $(date) RUNDATE=$RUNDATE from=$FROM dry_run=$DRY_RUN ==="
[ -z "$SEC_EMAIL" ] && note "SEC_EMAIL is not set (.env) — SEC requests use a placeholder User-Agent"
if pmset -g batt 2>/dev/null | grep -q "Battery Power"; then
  note "running on battery — the Mac can still sleep; plug in for unattended runs"
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
    run preflight gate "$VPY" scripts/market_open.py
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
  run analyze blocking bash -c "set -o pipefail; \"$VPY\" scripts/analyze_stock.py --macro --prices-dir output/prices \
      --universe us --min-spread 0 --mcap-min 300e6 2>&1 | tee output/run_$RUNDATE.log"
  if [ "$BLOCKED" = 1 ]; then
    note "analyze_stock.py failed — nothing downstream was run"
    finish failed 1
  fi
  if [ "$DRY_RUN" = 0 ] && [ ! -f "$SNAPSHOT" ]; then
    note "analyze_stock.py exited 0 but $SNAPSHOT does not exist"
    finish failed 1
  fi
fi

if [ "$DRY_RUN" = 0 ] && [ ! -f "$SNAPSHOT" ]; then
  note "snapshot $SNAPSHOT not found"
  finish failed 1
fi

# ---------------------------------------------------------------- enrich
if wants enrich; then
  run enrich_fdic     soft "$VPY" scripts/enrich_fdic.py     "$SNAPSHOT"
  run enrich_reit     soft "$VPY" scripts/enrich_reit.py     "$SNAPSHOT"
  run enrich_xbrl     soft "$VPY" scripts/enrich_xbrl.py     "$SNAPSHOT"
  run enrich_pipeline soft "$VPY" scripts/enrich_pipeline.py "$SNAPSHOT"
  # Must follow every enrichment so the banners pick the fields up.
  run render          soft "$VPY" scripts/rescore_and_render.py "$SNAPSHOT" --prices-dir output/prices
fi

# ---------------------------------------------------------------- archive
ARCHIVED=1
if wants archive; then
  ARCHIVED=0
  run archive_snapshot blocking "$VPY" scripts/archive_snapshot.py "$SNAPSHOT" --dest "$SNAP_WT"
  rc=$(tail -1 "$STEPLOG" | cut -f2)
  case "$rc" in
    0|dry-run)
      run archive_add blocking git -C "$SNAP_WT" add "results_$RUNDATE.json.gz"
      if [ "$DRY_RUN" = 1 ] || ! git -C "$SNAP_WT" diff --cached --quiet; then
        run archive_commit blocking git -C "$SNAP_WT" commit -q -m "Snapshot: $RUNDATE"
      else
        note "archive: results_$RUNDATE.json.gz unchanged on data/snapshots — nothing to commit"
      fi
      run archive_push blocking git -C "$SNAP_WT" push -q origin data/snapshots
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
  run gate_na_report   soft "$VPY" scripts/gate_na_report.py "$SNAPSHOT"
  run validate_ratings soft "$VPY" scripts/validate_ratings.py --snapshot "$SNAPSHOT" --prices-dir output/prices
fi

# ---------------------------------------------------------------- publish
publish() {
  local docs="$PAGES_WT/docs" f
  [ -f "$HTML" ] || { echo "publish: $HTML missing"; return 1; }
  [ -d "$docs" ] || { echo "publish: $docs missing (git worktree add .claude/worktrees/pages-live pages-live)"; return 1; }
  [ -f "$PAGES_WT/.gitignore" ] || { echo "publish: pages-live .gitignore missing — refusing to stage iCloud conflict copies"; return 1; }
  cp "$HTML" "$docs/index.html" || return 1
  for f in prices_meta.json hist.json details.json; do
    cp "output/$f" "$docs/$f" || { echo "publish: required sidecar output/$f missing"; return 1; }
  done
  # macro.json is optional; never publish yesterday's under today's HTML.
  if [ -f output/macro.json ]; then cp output/macro.json "$docs/macro.json" || return 1
  else rm -f "$docs/macro.json"; fi
  rm -f "$docs/prices.json"   # retired 2026-08-11
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

if wants publish; then
  run publish soft publish || note "publish failed — retry with: scripts/run_daily.sh --from publish --date $RUNDATE"
fi

# ---------------------------------------------------------------- done
if [ "$ARCHIVED" = 0 ]; then
  finish failed 1
elif [ "$SOFT_FAIL" = 1 ]; then
  finish degraded 3
else
  finish ok 0
fi
