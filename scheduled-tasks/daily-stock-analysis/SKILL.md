---
name: daily-stock-analysis
description: Run full stock analysis and publish updated report to GitHub Pages
---

You are running the end-of-day stock analysis routine. The whole pipeline is
one script, `scripts/run_daily.sh`; your job is to launch it, wait for it, and
write the run summary from what it produced.

> **DORMANT — do not run.** Since 2026-09-10 the live daily run is the cloud
> Routine in `../cloud-daily-stock-analysis/` (`run.sh`). Both pipelines
> archive to `data/snapshots` and force-push `pages-live`, so they must never
> both run. **If this task fires while the cloud routine is live, run nothing:
> report "Mac daily routine is dormant (cloud routine is live) — skipped" as
> the entire summary and stop.** Only follow the steps below when the user has
> explicitly switched the daily run back to this Mac (and paused the cloud
> Routine). The steps stay here as the Mac reference; `scripts/run_daily.sh`
> is also the manual backfill tool (`--from enrich --date YYYY-MM-DD`).

## Execution mode
- **Always run fully autonomously (auto mode).** Do not pause for confirmation or ask clarifying questions — this is an unattended scheduled run. The only write actions permitted are the ones the script performs itself (snapshot commit/push to `data/snapshots`, force-push of `pages-live`) and the retry commands listed below. Do not take other outward-facing or destructive actions.
- **Always run on the latest available model.**

## Why a script
Until 2026-09-13 this file listed ~15 commands for Claude to run one by one.
That chain broke often. On 2026-09-09 the 15-hour analysis finished after midnight. The
`$(date)`-named enrichment paths then pointed at a file that did not exist, and
the snapshot was never archived or published. `run_daily.sh` fixes RUNDATE
once at start, runs every step in order, keeps the Mac awake (`caffeinate`),
holds a lock so two runs never overlap, and records each step's exit code and
duration in `output/run_summary_<RUNDATE>.json`.

## Step 1 — Launch the pipeline
Run as a **single Bash call, in the background** (it takes hours; the Bash tool
would time out in the foreground):
```
cd "$HOME/Projects/Workspace Folder"; scripts/run_daily.sh
```
You will be notified when it exits. Do not start a second copy while waiting;
the lock file makes a second copy exit immediately anyway.

Exit codes: `0` success **or** market-closed skip · `1` a blocking step
failed (analysis, or the snapshot archive) · `3` finished, but a non-blocking
step failed.

## Step 2 — Read the results
RUNDATE is the date in the script's first log line (`RUNDATE=YYYY-MM-DD`).
1. `output/run_summary_RUNDATE.json` holds `status` (`ok` / `degraded` /
   `failed` / `skipped`), each step's `rc` and `seconds`, and `notes`.
2. The full log is `~/Library/Logs/StockModel/daily_RUNDATE.log`. The
   analysis output is also in `output/run_RUNDATE.log`.

If `status` is `skipped`, the entire run summary is: "market closed — run
skipped" plus the reason `market_open.py` printed. Stop there.

## Step 3 — Write the run summary
Include, from the log:
- **Status and timing:** the overall status, total duration, and each step's rc/duration. Call out anything over the 6 h analysis budget.
- **Notes:** every entry in `notes` (battery power, SEC_EMAIL missing, skipped fast-forward, archive not pushed, publish failure…).
- **Gate N/A coverage** (`gate_na_report` section): print the full table.
  - Call out every **⚠ JUMP**, a gate's N/A share up ≥10 points vs the prior snapshot. It means a data source degraded today.
  - Mention **⚠ HIGH** only if the set of HIGH gates changed vs recent runs. Several gates are structurally high.
- **Momentum check** (`validate_ratings` section): print the rating buckets and Spearman r.
  - Flag it if BUY/LEAN BUY had clearly higher trailing returns than HOLD/PASS.
  - Flag it if r is above +0.15 with p < 0.05. A value model is expected to show a negative r.
- **Portfolio report** (`portfolio_report` section): flag any sector above 35% of the BUY/LEAN BUY bucket, any highly correlated pair (r > 0.85) that isn't an obvious duplicate (GOOG/GOOGL), and any BUY with a 2020 drawdown worse than -50%.
- **Snapshot store** (`store_check` section): if it failed, quote its `PROBLEM:` lines. Store syncs never fail a step, so this is the only place a store that stopped updating shows up.
- **Run quality:** the analysis's closing `RUN QUALITY:` lines, and the `Screen skip cache:` line from Phase 1.
- **Publish:** the result, with the live URL's HTTP code.

## Step 4 — Recover a failed step (only when the summary shows one)
Each step can be resumed without re-running the analysis. Run the command in the background, with RUNDATE substituted literally:

| Symptom | Command |
|---|---|
| enrichment / render failed transiently | `cd "$HOME/Projects/Workspace Folder"; scripts/run_daily.sh --from enrich --date RUNDATE` |
| `archive_push` failed (network) | `cd "$HOME/Projects/Workspace Folder"; scripts/run_daily.sh --from archive --date RUNDATE` |
| publish failed | `cd "$HOME/Projects/Workspace Folder"; scripts/run_daily.sh --from publish --date RUNDATE` |

Retry once at most. Do **not** retry these:
- **`archive_snapshot` rc=2:** the snapshot breached the 80 MiB guard. It needs a size fix, likely splitting out the `edgar_history` series, which are ~25% of a snapshot.
- **`archive_snapshot` rc=1:** the gzip round-trip failed verification.
- **`analyze` failures.**

Report them as failed success criteria.

## Success criteria
A market-closed skip is a success, and the criteria below don't apply to it.
- `output/stock_analysis_results_RUNDATE.html` was created this run.
- `results_RUNDATE.json.gz` was archived and pushed to `data/snapshots`. That means the `archive_*` steps are all rc=0, or the note says it was unchanged.
- The run summary includes the gate N/A table (with JUMP flags called out) and the momentum check.
- Publish succeeded, or its failure was reported clearly with the retry command.

## Reference
- **Paths:**
  - Main repo: `$HOME/Projects/Workspace Folder`
  - Snapshots worktree: `.claude/worktrees/snapshots-data` (branch `data/snapshots`)
  - Pages worktree: `.claude/worktrees/pages-live` (branch `pages-live`)
  - Python: `~/.venvs/stock-model/bin/python`, outside the repo so worktree changes cannot delete it (rebuild: `scheduled-tasks/RECOVERY.md` step 3).
  - If a worktree is missing, recreate it: `git worktree add .claude/worktrees/pages-live pages-live` or `git worktree add .claude/worktrees/snapshots-data data/snapshots`.
- **Branch:** the script fast-forwards `main` only when the checkout is on `main`. A feature branch left checked out renders with that branch's templates, and the summary notes it. These task files are symlinked from `~/.claude/scheduled-tasks`, so the checked-out branch also decides which version of this file runs.
- **Market-open gate:** `scripts/market_open.py` computes the NYSE calendar offline and fails open. Unscheduled closures go in its `AD_HOC_CLOSURES`.
- **Screen skip cache:** `data/cache/screen_skip.json` makes Phase 1 skip tickers that were recently far below the $300M floor, or had no data from any source. It is disposable. `analyze_stock.py --no-screen-cache` ignores it for one run.
- **Phase 2 prefetch:** Phase 2 fetches network data on 4 threads (`--workers`). The analysis itself stays single-threaded.
- **Power:** the Mac must be on AC power. `caffeinate` can't hold off sleep on a nearly empty battery (the 2026-09-09 run hibernated at 1%).
- **API keys:** read from `.env` in the repo root, which must never be committed. `SEC_EMAIL` sets the SEC User-Agent. `ANTHROPIC_API_KEY` powers the macro narrative, which is cached per run date. `TIINGO_API_KEY`, `FMP_API_KEY` and `FINNHUB_API_KEY` are optional.
- **Snapshot store:** `output/snapshots.duckdb` is a derived index. To rebuild it: `"$PYTHON" scripts/ingest_snapshots.py --results-dir output`.
- **Find days missing from the archive:** `"$PYTHON" scripts/archive_snapshot.py --dest .claude/worktrees/snapshots-data --audit`.
