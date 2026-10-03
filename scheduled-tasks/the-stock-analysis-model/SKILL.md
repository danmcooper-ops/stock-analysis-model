---
name: the-stock-analysis-model
description: Run the daily stock analysis model.
---

You are running the end-of-day stock analysis on the Mac mini. This is an
unattended scheduled run: work fully autonomously and never pause for
confirmation.

The runbook lives in the repo, version-controlled, so that it changes with the
code it drives. Read it now and follow it exactly, from Step 1 to the end:

    /Users/dmcooper/Projects/Workspace Folder/scheduled-tasks/daily-stock-analysis/SKILL.md

In short, it has you:
1. Launch `cd "$HOME/Projects/Workspace Folder"; scripts/run_daily.sh` as a
   single **background** Bash call (it takes hours), and wait to be notified
   when it exits. Never start a second copy.
2. Read `output/run_summary_<RUNDATE>.json` and
   `~/Library/Logs/StockModel/daily_<RUNDATE>.log`.
3. Write the run summary the runbook describes.

If the runbook cannot be read (the repo moved or was deleted), run nothing:
report "daily-stock-analysis runbook not found at the path above — skipped;
see scheduled-tasks/RECOVERY.md" and stop.

If the runbook carries a **DORMANT** banner, obey it: run nothing and report
it as skipped.

The only outward actions allowed are the ones `run_daily.sh` performs itself
(the snapshot commit and push to `data/snapshots`, the force-push of
`pages-live`) and the retry commands the runbook lists. Do not push to
`main`, open PRs, or edit code or config.
