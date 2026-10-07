---
name: the-weekly-backtest
description: Run the weekly forward-return backtest on the Mac mini.
---

You are running the weekly backtest on the Mac mini. This is an unattended
scheduled run: work fully autonomously and never pause for confirmation.

The runbook lives in the repo, version-controlled, so that it changes with the
code it drives. Read it now and follow it exactly, from Step 1 to the end:

    /Users/dmcooper/Projects/Workspace Folder/scheduled-tasks/weekly-backtest/SKILL.md

In short, it has you:
1. Launch `scheduled-tasks/weekly-backtest/mac_run.sh` **detached** (its own
   session via `setsid`, so the Bash tool's 2 h background limit cannot kill
   a cold run), then wait on its PID with a background loop, restarting the
   loop if it hits the limit. Use the exact commands in the runbook's Step 1.
   Never start a second copy.
2. Read `~/Library/Application Support/StockModel/backtest/status.txt` and
   the step logs.
3. Write the summary the runbook describes.

If the runbook cannot be read (the repo moved or was deleted), run nothing:
report "weekly-backtest runbook not found at the path above — skipped; see
scheduled-tasks/RECOVERY.md" and stop.

If the runbook carries a **DORMANT** banner, obey it: run nothing and report
it as skipped.

The only outward action allowed is the `Weekly backtest: <date>` commit and
push to `data/snapshots` that `mac_run.sh` performs itself, including in the
one retry the runbook lists. Do not push to `main`, open PRs, run
`calibrate`, or edit code or config.
