---
name: the-weekly-backtest
description: Summarize the weekly backtest that launchd ran on the Mac mini this morning.
---

You are writing the summary of the weekly forward-return backtest on the Mac
mini. This is an unattended scheduled run: work fully autonomously and never
pause for confirmation.

launchd has already started the backtest itself (Sundays 09:45 New York,
`com.stockmodel.weekly`). Your job is to read what it produced. The runbook
lives in the repo, version-controlled, so that it changes with the code it
drives. Read it now and follow it exactly, from Step 1 to the end:

    /Users/dmcooper/Projects/Workspace Folder/scheduled-tasks/weekly-backtest/SKILL.md

In short, it has you:
1. Check whether the job is still running (`.weekly.lock` in the work
   directory) and, if so, wait on its PID with a background loop, restarting
   the loop if it hits the 2 h limit.
2. Confirm `status.txt` is from today; if the job never ran, report why.
3. Write the summary that `cloud-weekly-backtest/SKILL.md` §3–4 describes.
4. Retry once, detached, only where the runbook allows it.

If the runbook cannot be read (the repo moved or was deleted), run nothing:
report "weekly-backtest runbook not found at the path above — skipped; see
scheduled-tasks/RECOVERY.md" and stop.

If the runbook carries a **DORMANT** banner, obey it: run nothing and report
it as skipped.

The only outward action allowed is the `Weekly backtest: <date>` commit and
push to `data/snapshots` that `mac_run.sh` performs itself, including in the
one retry the runbook lists. Do not push to `main`, open PRs, run
`calibrate`, or edit code or config.
