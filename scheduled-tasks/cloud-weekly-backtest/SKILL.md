---
name: cloud-weekly-backtest
description: Weekly forward-return backtest as a Claude Code cloud Routine — a fresh container each Sunday runs run.sh, measures the snapshot corpus, and commits the versioned summary to data/snapshots
---

> **DORMANT — do not run.** Since 2026-10-11 the live weekly backtest is the
> Mac mini (`../weekly-backtest/`: launchd runs `mac_run.sh`, a desktop-app
> task writes the summary); this Routine was paused before that Sunday. On
> 2026-09-27 the cloud egress proxy blocked every Yahoo request and the run
> measured nothing. Both commit a `Weekly backtest: <date>` summary and
> rewrite the `returns/` sidecars on `data/snapshots`, so they must never both
> run. **If this Routine fires, run nothing: report "Cloud weekly backtest is
> dormant (Mac routine is live) — skipped" as the entire summary and stop.**
> Only follow the steps below when the user has unloaded the Mac job and
> switched the weekly back to the cloud (`../MAC-MINI-SETUP.md`, "Going back
> to the cloud"). §3 (Read the outcome) and §4 (Write the summary) stay live:
> the Mac runbook follows them.

You are running the weekly backtest routine **in a Claude Code cloud session**.
It MEASURES whether the model's ratings and composite score predict forward
returns. It does NOT recalibrate anything. Nothing persists between runs
except what is on GitHub, so the whole job is packaged in
`scheduled-tasks/cloud-weekly-backtest/run.sh` next to this file. Your job is
to start it, wait for it, and write the summary from its logs. Do not
re-implement the steps by hand.

This replaced the Mac runbook (`../weekly-backtest/SKILL.md` + its launchd job)
when the local checkout was deleted on 2026-09-09. That runbook's summaries
had not reached `data/snapshots` since July: the branch held one backtest
artifact, `backtest_2026-07-13.xlsx`.

## Execution mode
- **Fully autonomous.** Nobody is watching. Never pause for confirmation.
- The only outward action permitted is the one `run.sh` performs: a
  `Weekly backtest: <date>` commit on `data/snapshots` with
  `backtest_summary_<date>.json`, `backtest_<date>.xlsx` and the
  forward-return sidecars under `returns/`. Never push anything else, never
  open a PR, never edit scoring/config files, never run `calibrate`.
- **Sundays only.** Friday's daily run (fired 21:00 UTC, up to ~20 h) is done
  by Saturday afternoon, so this job never races a daily push. The script
  still rebuilds its commit on a moved tip if a push is rejected.

## What the script does (so you can read its logs)
Everything lands under `$REPO/.cloud-backtest/`: `status.txt` (one
`step rc=N seconds=S` line per step, then `RUNDATE`, `SUMMARY`,
`SOFT_FAILURES`, `RESULT ...`), `logs/<step>.log`, and the corpus and outputs
in `output/`.

| step | blocking? | notes |
|---|---|---|
| 01-venv | yes | `.venv` + `pip install -e ".[dev]"` |
| 02-stage-corpus | yes | `scripts/backtest_cloud.py stage`: every snapshot dated ≥ 2026-07-06 (`MIN_CONSISTENT_DATE`) with its edgar_history blobs, the persisted `returns/` sidecars and last week's summary, fetched in batches out of a blob-less clone (~1.5 GB) |
| 03-store | no | `ingest_snapshots.py` → `output/snapshots.duckdb`; on failure the backtest parses the JSON instead (slower, more RAM) |
| 04-prices | yes | cold download for every ticker of every matured snapshot (two passes), then a **gate**: SPY must end within 5 days of today and ≥ 90% of the tickers must have a price file. Failing it stops the run, so that a throttled download night never shrinks the sample. |
| 04b-backfill | no | `backtest_cloud.py backfill-prices`: Tiingo for tickers Yahoo lacks, mostly delisted (acquired) names whose history Yahoo drops. At most `TIINGO_MAX_CALLS` (40) fetches; each series is cached in `price_backfill/` on the branch. Writes the list of confirmed delistings that `measure` measures to their last close. rc 1 = key unset or rate limited |
| 05-readiness | no | evidence census, dates only |
| 06-measure | yes | `backtest.py measure --local-prices-only --stamp <RUNDATE>`: reuses sidecars at ≥ 90% coverage whose window is settled, tops up the rest, never measures against a missing SPY. **Expect one `DEFERRED:` line most weeks** — the newest matured snapshot's eval date is the run day itself, so SPY has no bar there yet and the pair is measured next week rather than frozen on a short window |
| 07-compare | no | `backtest_cloud.py compare`: `REGRESSION:` lines vs last week's summary; rc 1 = at least one |
| 08-archive | yes | plumbing commit + push to `data/snapshots` |

## Steps for you

### 1. Get the checkout
```
cd /home/user/stock-analysis-model && git fetch origin main && git checkout -q main && git reset -q --hard origin/main && git log --oneline -1
```
(Clone `https://github.com/danmcooper-ops/stock-analysis-model.git` there
first if the session has no checkout.)

### 2. Start the script in the background and wait for it
Run it with the Bash tool's `run_in_background`. It takes roughly 1–2 hours,
most of it the price download:
```
cd /home/user/stock-analysis-model && bash scheduled-tasks/cloud-weekly-backtest/run.sh > .cloud-backtest.log 2>&1
```
Do not poll with `sleep` loops, and never start a second copy.

### 3. Read the outcome
Read `.cloud-backtest/status.txt` first.
- `RESULT OK` → full summary (below).
- `RESULT FAILED at 04-prices` → quote the `PROBLEM:` lines and the download
  tally from `logs/04-prices.log`. Nothing was measured or pushed. One retry
  of the whole script is allowed if the log shows Yahoo throttling (many
  `empty` results). A second failure is reported, not retried.
- `RESULT FAILED at <other step>` → name the step, quote the last ~30 lines
  of its log, and say plainly what did not happen.
- `RESULT FAILED at archive` → the measurement exists (the `SUMMARY` path);
  report it anyway. If the push was rejected for credentials, call
  `mcp__Claude_Code_Remote__add_repo` (`owner: danmcooper-ops`,
  `repo: stock-analysis-model`, `access: push`) and push once by hand:
  `git -C .cloud-backtest/snapshots-data push https://github.com/danmcooper-ops/stock-analysis-model.git refs/heads/data/snapshots:refs/heads/data/snapshots`.

### 4. Write the summary
Lead with the result line and the run date, then, from `logs/06-measure.log`
and the summary JSON:
1. **Readiness** table (`logs/05-readiness.log`): matured snapshots,
   effective independent n, and the dates calibration and a significance
   test become possible, per horizon.
2. **Composite-score rank IC** block: mean IC, share of snapshots positive,
   effective n, t at effective n. The headline is **t(eff)**, not the
   snapshot count or the pooled n. Below |t| = 2 the signal is not
   distinguishable from zero; say so plainly rather than reading the sign.
2a. **Scoring models**: the "SCORING MODELS IN THE CORPUS" and "CURRENT MODEL
    ... RE-SCORED" blocks (`regimes` and `rescored_current` in the summary).
    Give the re-scored IC and buckets next to the as-recorded ones. The
    as-recorded headline measures whichever model rated each day, so once the
    weights change it pools models. The re-scored view holds today's model
    fixed. It is in-sample for weights that were calibrated on this corpus;
    say so if a weight PR landed since the corpus began.
3. **Rating buckets**: the aggregated table. They should stay ordered
   BUY > LEAN BUY > HOLD > PASS on mean excess return. A reversal that
   persists three weeks running is worth a note; a single week is noise.
4. **Signal quartile spreads**, and the mos-cohort spreads.
5. **Coverage and attrition** from `coverage` in the summary JSON and
   `logs/04b-backfill.log`:
   - the lowest per-(date, horizon) coverage
   - delisted names measured to their last close (`delisted_by_rating`, by
     kind), with any `performance` delisting named
   - `gone_before_snapshot`: rows for tickers already delisted when the
     snapshot was taken, so stale rows are excluded
   - `unpriced_by_rating`: names no source could price
   - how many backfill fetches were `deferred` to next week (call cap or
     Tiingo rate limit)

   If 04b failed or TIINGO_API_KEY is absent, say that delisted names were
   dropped this week.
6. **Deferred (date, horizon) pairs** from `deferred` in the summary JSON —
   not to be confused with the backfill's deferred *fetches* above. One or two
   per week is normal and expected: the newest matured snapshot's eval date is
   the run day itself, so SPY has no bar there yet and the pair is measured
   next week instead of being frozen on a short window. The same pair deferred
   two weeks running is a `REGRESSION:` — the price download has stopped
   advancing, so check `logs/04-prices.log` and SPY's last bar.
7. **Week over week** (`logs/07-compare.log`): list every `REGRESSION:` line
   verbatim, or say "no regressions". Quote `NOTICE:` lines too: a scoring
   model change is expected after a weight PR, not a failure. A newly skipped snapshot ("missing gate
   fields") means a nightly run wrote a snapshot without current gate
   fields — flag it prominently.
8. **Archive**: the `Weekly backtest: <date>` commit from `logs/08-archive.log`
   and how many `returns/` sidecars it added or changed.

"FV accuracy" is not measured below a 365d horizon by design.

## What this routine deliberately does NOT do
- **No `calibrate`.** It refuses below 8 effective independent periods
  (`MIN_EFFECTIVE_N`). The readiness table says when that clears: 30d
  horizon 2027-03-03, 90d 2028-06-25. Never pass `--force` from this
  routine; a forced run is an exploratory manual step whose output must never
  be copied into `scripts/config.py`.
- **No config changes.** Weights and thresholds change only through a
  reviewed PR, after a non-forced calibration whose recommendation won a
  majority of de-overlapped windows. After such a PR merges, the new model's
  own as-recorded evidence starts from zero (its regime in `regimes`). Until
  it matures, `rescored_current` is the only measure of the new weights, and
  it is in-sample.

## Running it by hand
- `SMOKE=1 bash scheduled-tasks/cloud-weekly-backtest/run.sh` measures the
  newest 3 matured snapshots on 8 tickers in under a minute and never pushes
  (SMOKE implies DRY_RUN). Its compare step always reports low coverage.
- `DRY_RUN=1` runs the full job without pushing.
- `RUNDATE=YYYY-MM-DD` names the outputs after another day (e.g. a Sunday
  that failed, re-run on Monday).

## Success criteria
- The readiness table and the composite-IC block are in the summary, with the
  t(eff) interpretation stated
- `backtest_summary_<date>.json` was committed to `data/snapshots` and
  pushed, or the failure was reported
- every `REGRESSION:` line is in the summary
