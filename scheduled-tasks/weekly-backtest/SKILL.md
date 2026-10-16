---
name: weekly-backtest
description: Weekly forward-return backtest on the Mac mini — the Claude app's Sunday task launches mac_run.sh detached, waits for it, and writes the summary
---

> **LIVE on the Mac mini** since the cloud Routine ("Weekly Backtest",
> `../cloud-weekly-backtest/`) was paused before Sunday 2026-10-11; it is now
> dormant. That first Sunday never fired (the app task's first run was a
> Friday catch-up on 10-16), so the newest published summary was still the
> cloud's 2026-10-04 when this was written. Both commit a `Weekly backtest: <date>` summary and rewrite the
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
- **Sundays, plus catch-ups.** The app task fires Sundays at 09:45; when the
  app was closed or the Mac asleep it fires late, on whatever day the app
  wakes. Step 1a decides whether such a catch-up runs: a missed week is worth
  recovering, but not while the nightly analysis is running (both would hit
  Yahoo, and the nightly is the one that must not be throttled). The two
  pushes never conflict: the daily rebases over this job's commit
  (`run_daily.sh` `archive_push`), and this job rebuilds its commit on a
  moved tip.

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

## Step 1 — Decide, launch detached, wait

**1a. Should this firing run?** Run this as one foreground Bash call:
```
cd "$HOME/Projects/Workspace Folder"; echo "now: $(TZ=America/New_York date '+%F %A %H:%M')"; pgrep -fl 'run_daily\.sh|analyze_stock\.py' || echo "nightly: not running"; .venv/bin/python scripts/check_eod_delivery.py --repo danmcooper-ops/stock-analysis-model --kind weekly --max-age-days 5; echo "weekly-check rc=$?"
```
- **Sunday** (New York): always run. Go to 1b.
- **Any other day** (a catch-up), run only when all three hold:
  - `weekly-check rc=1`: no summary landed in the last 5 days, so this
    week has none yet;
  - `nightly: not running`;
  - it is a weekend, or a weekday before 14:00 New York (a weekday run
    started later could still be downloading prices when the nightly
    starts at 17:05).

  Otherwise run nothing and report "weekly backtest catch-up skipped:" with
  the reason. `rc=2` means GitHub could not be read: skip, and say so.
- A weekday catch-up during market hours is safe for the prices:
  `download_prices` drops a bar dated today until 16:30 New York, so a live
  intraday price is never frozen as an eval-date close.

**Timing.** Warm, the run takes minutes. Cold (a lost price cache) it takes 1–2 h or more,
and the Bash tool stops any background command after **2 h**, killing
whatever it launched. So the job must **not** be a child of a Bash tool call:
launch it in its own session, then wait for its PID in a separate,
disposable loop — the same pattern as the daily runbook.

**1b. Launch:** run this as one ordinary **foreground** Bash call. It returns at once:
```
cd "$HOME/Projects/Workspace Folder"; W="$HOME/Library/Application Support/StockModel/backtest"; L="$W/.weekly.lock"; P="$(cat "$L" 2>/dev/null)"; if [ -n "$P" ] && ps -p "$P" -o command= 2>/dev/null | grep -q 'mac_run\.sh'; then echo "ALREADY RUNNING pid $P"; else R="$(TZ=America/New_York date +%F)"; mkdir -p "$HOME/Library/Logs/StockModel"; RUNDATE="$R" perl -MPOSIX=setsid -e 'setsid() or die "setsid: $!"; exec @ARGV or die "exec: $!"' nohup scheduled-tasks/weekly-backtest/mac_run.sh </dev/null >>"$HOME/Library/Logs/StockModel/weekly_launch.out" 2>&1 & echo "LAUNCHED pid $! RUNDATE $R at $(date '+%F %T')"; fi
```
- There is no `git pull` here: the daily run fast-forwards `main` every
  weeknight, and the job reads its scripts from this checkout.
- `$!` is `mac_run.sh`'s PID for the whole run (perl and `nohup` `exec` in
  place), and it is the PID `mac_run.sh` writes to `.weekly.lock`.
- Note the `RUNDATE` it prints: the run's outputs, log and status are named
  after it even if the run crosses midnight, and Step 2 checks against it.
- If it prints `ALREADY RUNNING`, do not launch. Wait on that PID instead
  (its RUNDATE is the `RUNDATE` line of `status.txt` once the run ends). A
  lock whose PID is dead, or is not a `mac_run.sh`, is stale: `mac_run.sh`
  removes it itself.

**1c. Wait:** run this as a Bash call **in the background**, with the longest
timeout allowed, substituting the PID:
```
pid=PID; while kill -0 "$pid" 2>/dev/null; do sleep 60; done; echo "weekly pid $pid exited at $(date '+%F %T')"
```
- You are notified when it exits.
- If the loop is stopped at the 2 h limit, the job is unaffected. Check with
  `kill -0 PID` and start the same loop again. Repeat until it reports the exit.
- Never relaunch the job to "resume" it while its PID is alive.

## Step 2 — Did it finish?
Read `"$W/status.txt"`.
- Its `RUNDATE` line is the RUNDATE from 1b and it ends with a `RESULT` line
  → Step 3.
- Otherwise the job died before it could finish `status.txt` (a crash, a
  kill, or the lock refusing a second copy). Report it with the tail of
  `~/Library/Logs/StockModel/weekly_<RUNDATE>.log` and of
  `weekly_launch.out`, then see Step 4.
- Killing `mac_run.sh` (TERM/INT/HUP) stops the whole run before the lock is
  released, so a dead PID really does mean nothing of the run is left.

## Step 3 — Read the outcome and write the summary
Follow `../cloud-weekly-backtest/SKILL.md` **§3 (Read the outcome) and §4
(Write the summary)** exactly, and always quote the `PUBLISHED` line: it is
what step 08 actually did. Only `PUBLISHED pushed <sha>` means the summary
reached `data/snapshots`; `RESULT OK-DRYRUN` / `PUBLISHED no (DRY_RUN)` is a
run that published nothing, and must be reported as such. Ignore its DORMANT banner; it refers to the
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
`backtest_summary_<date>.json` on `data/snapshots` is more than 7 days old,
so a Sunday that produced nothing — the app was closed, or the task never
fired — is red the next day. (Until 2026-10-16 this section described the
workflow before it existed, which is how 10-11 went unnoticed.)

## Alternative: launchd
`com.stockmodel.weekly.plist` runs `mac_run.sh` in place on the same
schedule without the app, but writes no summary. It is **not loaded**; never
load it while the app task is enabled, or both fire on Sunday (the lock
would refuse the second, which would then report a failure).
