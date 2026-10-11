---
name: weekly-backtest
description: Weekly forward-return backtest on the Mac mini — the Claude app's Sunday task launches mac_run.sh detached, waits for it, and writes the summary
---

> **LIVE since 2026-10-11** on the Mac mini. The cloud Routine ("Weekly
> Backtest", `../cloud-weekly-backtest/`) was paused before that Sunday and is
> now dormant. Both commit a `Weekly backtest: <date>` summary and rewrite the
> `returns/` sidecars on `data/snapshots`, so they must never both run: to go
> back to the cloud, disable the app task first (`../MAC-MINI-SETUP.md`,
> "Going back to the cloud").

You are running the weekly backtest routine on the Mac. It MEASURES whether
the model's ratings and composite score predict forward returns. It does NOT
recalibrate anything.

The Claude desktop app's scheduled task `the-weekly-backtest` (Sundays 09:45
New York) sends you here. The Mac runs **the same script as the cloud
Routine**, `../cloud-weekly-backtest/run.sh`, through `mac_run.sh` next to
this file. The wrapper supplies what the container used to: a python3 ≥ 3.11
on `PATH`, certifi's CA bundle, `YF_IMPERSONATE=chrome`, the keys in `.env`
(`TIINGO_API_KEY`), a work directory outside the repo, and a lock so two runs
never overlap.

## Execution mode
- **Fully autonomous.** Nobody is watching. Never pause for confirmation.
- The only outward action permitted is the one `run.sh` performs: the
  `Weekly backtest: <date>` commit on `data/snapshots`. Never push anything
  else, never edit scoring/config files, never run `calibrate`.
- **Sundays only.** Friday's daily run can last into Saturday. The daily's
  push rebases over this job's commit if the two meet (`run_daily.sh`
  `archive_push`), and this job rebuilds its commit on a moved tip.

## Where things are
- Work directory (`W`): `~/Library/Application Support/StockModel/backtest`,
  holding `status.txt`, `.weekly.lock` (the PID, while a run is live),
  `logs/<step>.log`, the staged corpus in `output/`, and the price cache in
  `output/prices/`, which is **kept between Sundays**. Only stale files are
  re-fetched, and the step-04 gate counts only files written this run or
  already current.
- Full log: `~/Library/Logs/StockModel/weekly_<date>.log`
- Launch output (anything printed before the log starts):
  `~/Library/Logs/StockModel/weekly_launch.out`

## Step 1 — Launch the backtest, detached
Warm, it takes minutes. Cold (a lost price cache) it takes 1–2 h or more,
and the Bash tool stops any background command after **2 h**, killing
whatever it launched. So the job must **not** be a child of a Bash tool call:
launch it in its own session, then wait for its PID in a separate,
disposable loop — the same pattern as the daily runbook.

**1a. Launch:** run this as one ordinary **foreground** Bash call. It returns at once:
```
cd "$HOME/Projects/Workspace Folder"; W="$HOME/Library/Application Support/StockModel/backtest"; L="$W/.weekly.lock"; if [ -f "$L" ] && kill -0 "$(cat "$L")" 2>/dev/null; then echo "ALREADY RUNNING pid $(cat "$L")"; else mkdir -p "$HOME/Library/Logs/StockModel"; perl -MPOSIX=setsid -e 'setsid() or die "setsid: $!"; exec @ARGV or die "exec: $!"' nohup scheduled-tasks/weekly-backtest/mac_run.sh </dev/null >>"$HOME/Library/Logs/StockModel/weekly_launch.out" 2>&1 & echo "LAUNCHED pid $! at $(date '+%F %T')"; fi
```
- There is no `git pull` here: the daily run fast-forwards `main` every
  weeknight, and the job reads its scripts from this checkout.
- `$!` is `mac_run.sh`'s PID for the whole run (perl and `nohup` `exec` in
  place), and it is the PID `mac_run.sh` writes to `.weekly.lock`.
- If it prints `ALREADY RUNNING`, do not launch. Wait on that PID instead.

**1b. Wait:** run this as a Bash call **in the background**, with the longest
timeout allowed, substituting the PID:
```
pid=PID; while kill -0 "$pid" 2>/dev/null; do sleep 60; done; echo "weekly pid $pid exited at $(date '+%F %T')"
```
- You are notified when it exits.
- If the loop is stopped at the 2 h limit, the job is unaffected. Check with
  `kill -0 PID` and start the same loop again. Repeat until it reports the exit.
- Never relaunch the job to "resume" it while its PID is alive.

## Step 2 — Did it finish?
TODAY is `TZ=America/New_York date +%F`. Read `"$W/status.txt"`.
- Its `RUNDATE` line is TODAY and it ends with a `RESULT` line → Step 3.
- Otherwise the job died before it could finish `status.txt` (a crash, a
  kill, or the lock refusing a second copy). Report it with the tail of
  `~/Library/Logs/StockModel/weekly_TODAY.log` and of `weekly_launch.out`,
  then see Step 4.

## Step 3 — Read the outcome and write the summary
Follow `../cloud-weekly-backtest/SKILL.md` **§3 (Read the outcome) and §4
(Write the summary)** exactly. Ignore its DORMANT banner; it refers to the
cloud Routine, and those two sections are shared. The step names, logs and
summary JSON are the same; only the directory differs (`W` above, in place
of `.cloud-backtest/`). If a push was refused for credentials, report it
(the fix is the Mac's git credential — there is no `add_repo` here) and do
not push by hand.

## Step 4 — Retry (at most once)
Retry only where `../cloud-weekly-backtest/SKILL.md` §3 allows one (Yahoo
throttling at `04-prices`), or when Step 2 found that the job died without a
`RESULT`. Launch it exactly as in Step 1a and wait as in 1b, then return to
Step 3. A second failure is reported, not retried.

## What this routine deliberately does NOT do
The same as the cloud routine: no `calibrate` (it refuses below
`MIN_EFFECTIVE_N`, and `--force` is never passed from here), and no config
changes. See the end of `../cloud-weekly-backtest/SKILL.md`.

## Missing-week alarm
`scripts/check_eod_delivery.py --kind weekly` (run each Monday by
`.github/workflows/weekly-backtest-check.yml`) goes red when the newest
`backtest_summary_<date>.json` on `data/snapshots` is more than 8 days old,
so a Sunday that produced nothing — the app was closed, or the task never
fired — is noticed.

## Alternative: launchd
`com.stockmodel.weekly.plist` runs `mac_run.sh` in place on the same
schedule without the app, but writes no summary. It is **not loaded**; never
load it while the app task is enabled, or both fire on Sunday (the lock
would refuse the second, which would then report a failure).
