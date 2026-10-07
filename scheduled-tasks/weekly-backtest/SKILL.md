---
name: weekly-backtest
description: Weekly forward-return backtest on the Mac mini — launchd runs mac_run.sh on Sunday morning; this task reads the outcome and writes the summary
---

> **LIVE since 2026-10-11** on the Mac mini. The cloud Routine ("Weekly
> Backtest", `../cloud-weekly-backtest/`) was paused before that Sunday and is
> now dormant. Both commit a `Weekly backtest: <date>` summary and rewrite the
> `returns/` sidecars on `data/snapshots`, so they must never both run: to go
> back to the cloud, unload the launchd job first (`../MAC-MINI-SETUP.md`,
> "Going back to the cloud").

You are writing the summary of the weekly backtest on the Mac. It MEASURES
whether the model's ratings and composite score predict forward returns. It
does NOT recalibrate anything.

**You do not start the job.** launchd does, every Sunday at 09:45 New York
(`com.stockmodel.weekly.plist`, which runs `mac_run.sh` in place), so the
backtest runs whether or not the Claude app is open. `mac_run.sh` runs the
same script the cloud Routine used, `../cloud-weekly-backtest/run.sh`, and
supplies what the container used to: a python3 ≥ 3.11 on `PATH`, certifi's CA
bundle, `YF_IMPERSONATE=chrome`, the keys in `.env` (`TIINGO_API_KEY`), and a
work directory outside the repo. This task fires a few hours later, reads
what the job left behind, and writes the summary.

## Execution mode
- **Fully autonomous.** Nobody is watching. Never pause for confirmation.
- Read-only, apart from the one retry below. Never push anything yourself,
  never edit scoring/config files, never run `calibrate`.

## Where things are
- Work directory (`W`): `~/Library/Application Support/StockModel/backtest`,
  holding `status.txt`, `.weekly.lock` (while a run is live), `logs/<step>.log`,
  the staged corpus in `output/`, and the price cache in `output/prices/`,
  which is **kept between Sundays**.
- Full log: `~/Library/Logs/StockModel/weekly_<date>.log`
- launchd's own output: `~/Library/Logs/StockModel/launchd_weekly.out` / `.err`
- Script: `"$HOME/Projects/Workspace Folder/scheduled-tasks/weekly-backtest/mac_run.sh"`

## Steps

### 1. Is the job still running?
```
W="$HOME/Library/Application Support/StockModel/backtest"; L="$W/.weekly.lock"; if [ -f "$L" ] && kill -0 "$(cat "$L")" 2>/dev/null; then echo "RUNNING pid $(cat "$L")"; else echo "NOT RUNNING"; fi
```
If it prints `RUNNING`, wait for it. Run this as a Bash call **in the
background** with the longest timeout allowed, substituting the PID:
```
pid=PID; while kill -0 "$pid" 2>/dev/null; do sleep 60; done; echo "weekly pid $pid exited at $(date '+%F %T')"
```
If the loop is stopped at the Bash tool's 2 h limit, the job is unaffected
(launchd owns it): check `kill -0 PID` and start the same loop again. Never
start a second copy.

### 2. Did today's run happen?
TODAY is `TZ=America/New_York date +%F`. Read `"$W/status.txt"`.
- Its `RUNDATE` line is TODAY and it ends with a `RESULT` line → go to step 3.
- Otherwise the job did not run today (or died before it could finish
  `status.txt`). Report **"weekly backtest did not run today"** and quote:
  - the tail of `~/Library/Logs/StockModel/weekly_TODAY.log` if it exists;
  - `~/Library/Logs/StockModel/launchd_weekly.err`;
  - the `state`, `runs` and `last exit code` lines of
    `launchctl print gui/$(id -u)/com.stockmodel.weekly` (a "Could not find
    service" means the plist is not loaded).

  Then do the retry in step 4. Do not start it any other way.

### 3. Read the outcome and write the summary
Follow `../cloud-weekly-backtest/SKILL.md` **§3 (Read the outcome) and §4
(Write the summary)** exactly. Ignore its DORMANT banner; it refers to the
cloud Routine, and those two sections are shared. The step names, logs and
summary JSON are the same; only the directory differs (`W` above, in place
of `.cloud-backtest/`). If a push was refused for credentials, report it
(the fix is the Mac's git credential — there is no `add_repo` here) and do
not push by hand.

### 4. Retry (at most once)
Retry only where `../cloud-weekly-backtest/SKILL.md` §3 allows one (Yahoo
throttling at `04-prices`), or when step 2 found that the job never ran.
Launch it **detached**, as one ordinary foreground Bash call that returns at
once, then wait on the PID as in step 1:
```
cd "$HOME/Projects/Workspace Folder"; W="$HOME/Library/Application Support/StockModel/backtest"; L="$W/.weekly.lock"; if [ -f "$L" ] && kill -0 "$(cat "$L")" 2>/dev/null; then echo "ALREADY RUNNING pid $(cat "$L")"; else perl -MPOSIX=setsid -e 'setsid() or die "setsid: $!"; exec @ARGV or die "exec: $!"' nohup scheduled-tasks/weekly-backtest/mac_run.sh </dev/null >>"$HOME/Library/Logs/StockModel/weekly_launch.out" 2>&1 & echo "LAUNCHED pid $! at $(date '+%F %T')"; fi
```
A cold price cache can take more than the Bash tool's 2 h limit, which is
why the retry must not be a plain background Bash call. When it finishes, go
back to step 3. A second failure is reported, not retried.

## What this routine deliberately does NOT do
The same as the cloud routine: no `calibrate` (it refuses below
`MIN_EFFECTIVE_N`, and `--force` is never passed from here), and no config
changes. See the end of `../cloud-weekly-backtest/SKILL.md`.

## Missing-week alarm
`.github/workflows/weekly-backtest-check.yml` runs every Monday and goes red
when the newest `backtest_summary_<date>.json` on `data/snapshots` is more
than 8 days old, so a Sunday that produced nothing is noticed even if this
task never fired.
