---
name: weekly-backtest
description: Weekly forward-return backtest over the snapshot corpus, readiness census, and a versioned summary on the data/snapshots branch
---

You are running the weekly backtest routine. It MEASURES whether the model's
ratings and composite score predict forward returns; it does NOT recalibrate
anything. Execute the steps in order. Stop and report an error if any step
fails.

## Execution mode
- **Always run fully autonomously (auto mode).** Do not pause for confirmation. Only the "write" actions described below (writing under `output/`, committing/pushing the summary to `data/snapshots`) are permitted.
- **Run on Sunday** (or the first quiet day after), never concurrently with `daily-stock-analysis` — both read and refresh `output/prices`.

## This job is already automated — check before running it by hand
A launchd agent runs the unattended half of this routine every **Sunday at
20:00 America/New_York**, with no agent involved:

- `com.stockmodel.weekly.plist` → `~/Library/LaunchAgents/`
- `weekly_backtest.sh` → `~/Library/Application Support/StockModel/`

Both are versioned next to this file; the copies in those two locations are
what actually run, so edit here and copy out (there is no symlink). The script
covers price refresh, `annotate`, `measure`, and per-horizon `calibrate`. It
does **not** do the readiness census or publish the summary — those steps below
are the agent's job.

Before running this runbook on a Sunday, check the log for that day at
`~/Library/Logs/StockModel/weekly_<date>.log`. Starting a manual run while the
launchd job is working violates the concurrency rule above just as surely as
overlapping with `daily-stock-analysis` — both refresh `output/prices`.

## Paths
- **Main repo:** `$HOME/Projects/Workspace Folder`
- **Snapshots worktree:** `$HOME/Projects/Workspace Folder/.claude/worktrees/snapshots-data` (branch `data/snapshots`; holds every `results_YYYY-MM-DD.json` and is the backtest's input)
- **Python:** `$HOME/Projects/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python`
- **SSL fix:** set `SSL_CERT_FILE` to the output of `.venv/bin/python -m certifi` before running any Python script

## IMPORTANT: Command format
All Python script invocations **must be sent as a single-line semicolon-separated Bash command** (not multi-line). Use the exact format shown in each step.

## Steps

### 1. Refresh the price cache
Forward returns are read from `output/prices/*.parquet`; a (snapshot, horizon) pair only counts once a bar exists at snapshot date + horizon, so stale files silently shrink the sample. Run as a **single Bash call**:
```
PYTHON="$HOME/Projects/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/download_prices.py --output-dir output/prices --max-age-days 2 --tickers $(ls output/prices/*.parquet | sed 's|.*/||;s|\.parquet||') SPY
```
Idempotent; the day after the nightly refresh this is nearly a no-op. Individual delisted-ticker errors are routine noise.

### 2. Readiness census (dates only, instant)
```
PYTHON="$HOME/Projects/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/backtest.py readiness --results-dir .claude/worktrees/snapshots-data --horizons 30,90,180
```
Print the table in the run summary. It reports, per horizon, how many snapshots have matured, the **effective independent sample** (`span // horizon + 1` — daily snapshots over one horizon are ONE observation, whatever the window count), and the calendar dates at which (a) de-overlapped walk-forward becomes possible, (b) `calibrate` stops refusing (effective n ≥ 8), and (c) a rank-IC significance test has power (effective n ≥ 16). Snapshots before 2026-07-06 (the first with the full post-rebalance gate set) are excluded by default; do not pass `--since none`.

### 3. Warm the forward-return cache
```
PYTHON="$HOME/Projects/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/backtest.py annotate --results-dir .claude/worktrees/snapshots-data --prices-dir output/prices --cache-dir output/returns --horizons 30,90,180
```
Writes one sidecar per matured (date, horizon) under `output/returns/`; matured returns are immutable, so reruns only fetch new pairs. Horizons with 0 matured snapshots are skipped with a message — that is expected for 180d until early 2027 (first 180d maturity 2027-01-02).

### 4. Measure
```
PYTHON="$HOME/Projects/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/backtest.py measure --results-dir .claude/worktrees/snapshots-data --prices-dir output/prices --horizons 30,90,180 --cohort mos --exclude-capped
```
Writes `output/backtest_YYYY-MM-DD.xlsx` and `output/backtest_summary_YYYY-MM-DD.json`. **Print in the run summary:** the "Composite-score rank IC" block (mean IC, share of snapshots positive, effective n, t at effective n), the aggregated rating-bucket table, the Signal Quartile spreads, and the readiness table. Interpretation rules:
- The headline is **t(eff)**, not the snapshot count or the pooled n. Below |t| = 2 the signal is not distinguishable from zero; say so plainly rather than reading the sign.
- Rating buckets should stay ordered BUY > LEAN BUY > HOLD > PASS on mean excess return. A reversal that persists three weeks running is worth a note; a single week is noise.
- Snapshots listed as "skipped — missing gate fields" are pre-rebalance files; that list should be stable week to week. A **new** entry means a nightly run wrote a snapshot without current gate fields — flag it.
- "FV accuracy" is not measured below a 365d horizon by design (it would only re-measure the margin of safety).

### 5. Commit the summary to data/snapshots
Run each as a **separate** Bash call:
```
cp "output/backtest_summary_$(date +%Y-%m-%d).json" "output/backtest_$(date +%Y-%m-%d).xlsx" "$HOME/Projects/Workspace Folder/.claude/worktrees/snapshots-data/"
```
```
git -C "$HOME/Projects/Workspace Folder/.claude/worktrees/snapshots-data" add "backtest_summary_$(date +%Y-%m-%d).json" "backtest_$(date +%Y-%m-%d).xlsx"
```
```
git -C "$HOME/Projects/Workspace Folder/.claude/worktrees/snapshots-data" commit -m "Weekly backtest: $(date +%Y-%m-%d)"
```
```
git -C "$HOME/Projects/Workspace Folder/.claude/worktrees/snapshots-data" push origin data/snapshots
```
A non-zero exit code should be reported but does **not** invalidate the measurement.

## What this routine deliberately does NOT do
- **No `calibrate`.** `scripts/backtest.py calibrate` refuses to run below 8 effective independent periods (`MIN_EFFECTIVE_N`), and the readiness table says when that clears (30d horizon: 2027-03-03; 90d: 2028-06-25, counting 8 horizons from 2026-07-06). Do not pass `--force` from this routine; a forced run is an exploratory manual step whose output must never be copied into `scripts/config.py`.
- **No config changes.** Weights and thresholds in `scripts/config.py` / `scripts/scoring.py` change only through a reviewed PR, after a non-forced calibration whose recommendation won a majority of de-overlapped windows.

## Success criteria
- Readiness table and the composite-IC block are in the run summary with the t(eff) interpretation stated
- `backtest_summary_YYYY-MM-DD.json` was committed to `data/snapshots` and pushed (or the failure reported)
