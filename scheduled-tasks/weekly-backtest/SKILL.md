---
name: weekly-backtest
description: Weekly forward-return backtest on the Mac — runs the cloud routine's run.sh through mac_run.sh and summarizes it
---

> **DORMANT — do not run while the cloud weekly Routine is live.** The live
> weekly backtest is `../cloud-weekly-backtest/`. Both commit a
> `Weekly backtest: <date>` summary and rewrite the `returns/` sidecars on
> `data/snapshots`, so they must never both run. **If this task fires while
> the cloud Routine is live, run nothing: report "Mac weekly backtest is
> dormant (cloud routine is live) — skipped" and stop.** Switch over with
> `../MAC-MINI-SETUP.md` section 7.

You are running the weekly backtest routine on the Mac. It MEASURES whether
the model's ratings and composite score predict forward returns. It does NOT
recalibrate anything.

The Mac runs the **same script as the cloud Routine**,
`../cloud-weekly-backtest/run.sh`, through `mac_run.sh` next to this file.
The wrapper supplies what the container used to: a python3 ≥ 3.11 on `PATH`,
certifi's CA bundle, `YF_IMPERSONATE=chrome`, the keys in `.env`
(`TIINGO_API_KEY`), and a work directory outside the repo. The old
`weekly_backtest.sh` is retired: it ran `calibrate` every week and published
nothing to `data/snapshots` after 2026-07-13.

## Execution mode
- **Fully autonomous.** Nobody is watching. Never pause for confirmation.
- The only outward action permitted is the one `run.sh` performs: the
  `Weekly backtest: <date>` commit on `data/snapshots`. Never push anything
  else, never edit scoring/config files, never run `calibrate`.
- **Sundays only.** Friday's daily run can last into Saturday afternoon. The
  daily's push rebases over this job's commit if the two meet (`run_daily.sh`
  `archive_push`), and this job rebuilds its commit on a moved tip.

## Where things are
- Script: `"$HOME/Projects/Workspace Folder/scheduled-tasks/weekly-backtest/mac_run.sh"`
- Work directory (`STOCK_MODEL_WORK`): `~/Library/Application Support/StockModel/backtest`,
  holding `status.txt`, `logs/<step>.log`, the staged corpus in `output/`, and
  the price cache in `output/prices/`, which is **kept between Sundays**.
  Only stale files are re-fetched, and the step-04 gate counts only files
  written this run or already current.
- Full log: `~/Library/Logs/StockModel/weekly_<date>.log`
- launchd alternative (no written summary): `com.stockmodel.weekly.plist`,
  which runs `mac_run.sh` in place. Install instructions are in its header.

## Steps

### 1. Start it in the background and wait
```
cd "$HOME/Projects/Workspace Folder"; git fetch -q origin main; git checkout -q main; git merge -q --ff-only origin/main; scheduled-tasks/weekly-backtest/mac_run.sh
```
Run it with the Bash tool's `run_in_background`. It takes minutes when the
price cache is warm and 1–2 hours on the first Sunday. Never start a second
copy.

### 2. Read the outcome and write the summary
Read `~/Library/Application Support/StockModel/backtest/status.txt`, then
follow `../cloud-weekly-backtest/SKILL.md` **§3 (Read the outcome) and §4
(Write the summary)** exactly. The step names, logs and summary JSON are the
same; only the directory differs (the work directory above, in place of
`.cloud-backtest/`). The retry rules are the same too. If a push is refused
for credentials, fix the Mac's git credential. There is no `add_repo` here.

## What this routine deliberately does NOT do
The same as the cloud routine: no `calibrate` (it refuses below
`MIN_EFFECTIVE_N`, and `--force` is never passed from here), and no config
changes. See the end of `../cloud-weekly-backtest/SKILL.md`.
