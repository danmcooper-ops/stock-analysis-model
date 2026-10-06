---
name: daily-stock-analysis
description: Run full stock analysis and publish updated report to GitHub Pages
---

You are running the end-of-day stock analysis routine. The whole pipeline is
one script, `scripts/run_daily.sh`; your job is to launch it, wait for it, and
write the run summary from what it produced.

> **LIVE since 2026-10-05** on the Mac mini. The cloud Routine
> ("Daily Stock Analysis", `../cloud-daily-stock-analysis/`) was paused on
> 2026-10-03 and is now dormant. Both pipelines archive to `data/snapshots`
> and force-push `pages-live`, so they must never both run: to go back to the
> cloud, pause this task first (`../MAC-MINI-SETUP.md`, "Going back to the
> cloud"). `scripts/run_daily.sh` is also the manual backfill tool
> (`--from enrich --date YYYY-MM-DD`).

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

## Step 1 — Launch the pipeline, detached
The run takes 4-6 h, but the Bash tool stops any background command after
**2 h**, killing whatever it launched. On 2026-10-05 that killed the
analysis at ticker 4,528 of 9,182, and nothing was archived or published. So
the pipeline must **not** be a child of a Bash tool call. Launch it detached
in its own session, then wait for its PID in a separate, disposable loop.

**1a. Launch:** run this as one ordinary **foreground** Bash call. It returns at once:
```
cd "$HOME/Projects/Workspace Folder"; L="output/.run_daily.lock"; if [ -f "$L" ] && kill -0 "$(cat "$L")" 2>/dev/null; then echo "ALREADY RUNNING pid $(cat "$L")"; else perl -MPOSIX=setsid -e 'setsid() or die "setsid: $!"; exec @ARGV or die "exec: $!"' nohup scripts/run_daily.sh </dev/null >>"$HOME/Library/Logs/StockModel/run_daily_launch.out" 2>&1 & echo "LAUNCHED pid $! at $(date '+%F %T')"; fi
```
- `setsid` gives the run its own session and process group, and the launching shell exits right away, so the run is reparented to launchd. Nothing the Bash tool stops can reach it.
- `$!` stays the pipeline's PID for the whole run: perl, `nohup` and the script's `caffeinate` re-exec all `exec` in place.
- The script tees its own log to `~/Library/Logs/StockModel/daily_<RUNDATE>.log`. `run_daily_launch.out` only catches anything printed before that tee starts.
- If it prints `ALREADY RUNNING`, do not launch. Wait on that PID instead.

**1b. Wait:** run this as a Bash call **in the background**, with the longest timeout allowed, substituting the PID:
```
pid=PID; while kill -0 "$pid" 2>/dev/null; do sleep 60; done; echo "run_daily pid $pid exited at $(date '+%F %T')"
```
- You are notified when it exits.
- If it is instead stopped at the 2 h limit, the pipeline is unaffected. Check with `kill -0 PID` and start the same wait loop again. Repeat until it reports the exit.
- Never relaunch the pipeline to "resume" it while its PID is alive.

A detached run has no exit code to hand back; `status` in
`output/run_summary_<RUNDATE>.json` carries it:

| `status` | Meaning | Old exit code |
|---|---|---|
| `ok` | success | `0` |
| `skipped` | market closed | `0` |
| `failed` | a blocking step failed (analysis, or the snapshot archive) | `1` |
| `degraded` | finished, but a non-blocking step failed | `3` |

If the summary is missing, or its `finished_at` is older than the launch, the script died before it could write one. That covers a crash, a kill, or the lock refusing a second copy. Report the tail of the daily log and `run_daily_launch.out`.

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
- **Your portfolios** (`portfolio_alerts` section): lead with any `!!` banner line verbatim (data problem / model-wide shift). Then one line per portfolio (size, rating mix, median MoS) and every ACTION and WATCH line verbatim; give the FYI count as a number only. Say "no portfolio alerts" when there are no ACTION/WATCH lines; skip it when it reports "No portfolios defined."
- **Snapshot store** (`store_check` section): if it failed, quote its `PROBLEM:` lines. Store syncs never fail a step, so this is the only place a store that stopped updating shows up.
- **Run quality:** the analysis's closing `RUN QUALITY:` lines, and the `Screen skip cache:` line from Phase 1.
- **Publish:** the result, with the live URL's HTTP code.

## Step 4 — Recover a failed step (only when the summary shows one)
Each step can be resumed without re-running the analysis.
- Launch the retry **detached**, exactly as in Step 1a: put the arguments after `scripts/run_daily.sh` in the launch command and substitute RUNDATE literally. Then wait on its PID as in 1b.
- A publish retry is short. A retry `--from enrich` can still outlast the 2 h limit, so never run any of these as a plain background Bash call.

The retries:

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
- **API keys:** read from `.env` in the repo root, which must never be committed. `SEC_EMAIL` sets the SEC User-Agent. `MACRO_ANTHROPIC_API_KEY` powers the macro narrative, which is cached per run date (`ANTHROPIC_API_KEY` is read as a fallback). `TIINGO_API_KEY`, `FMP_API_KEY` and `FINNHUB_API_KEY` are optional.
- **Snapshot store:** `output/snapshots.duckdb` is a derived index. To rebuild it: `"$PYTHON" scripts/ingest_snapshots.py --results-dir output`.
- **Find days missing from the archive:** `"$PYTHON" scripts/archive_snapshot.py --dest .claude/worktrees/snapshots-data --audit`.
