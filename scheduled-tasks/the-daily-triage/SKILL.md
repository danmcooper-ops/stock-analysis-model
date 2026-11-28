---
name: the-daily-triage
description: Check last night's stock-model run, diagnose any failure, and email the verdict.
---

You are running the morning triage of the nightly stock-analysis run on the
Mac mini. This is an unattended scheduled run: work fully autonomously and
never pause for confirmation.

The runbook lives in the repo, version-controlled, so that it changes with the
code it checks. Read it now and follow it exactly, from Step 1 to the end:

    /Users/dmcooper/Projects/Workspace Folder/scheduled-tasks/daily-triage/SKILL.md

In short, it has you:
1. Work out last night's RUNDATE and classify the night (OK, still running,
   never started, died, failed, degraded) from the lock, the daily log and
   `output/run_summary_<RUNDATE>.json`.
2. Match any failure against the runbook's known causes, launch at most one
   allowed retry, and run the health checks for tonight.
3. Write `output/triage_<RUNDATE>.md` and email it with
   `scripts/email_triage.py`. The email goes out at the close of every run.

If the runbook cannot be read (the repo moved or was deleted), run nothing:
report "daily-triage runbook not found at the path above —
skipped; see scheduled-tasks/RECOVERY.md" and stop.

If the runbook carries a **DORMANT** banner, obey it: run nothing and report
it as skipped.

The only outward actions allowed are the triage email and the one detached
`scripts/run_daily.sh --from …` retry the runbook permits. Do not push to
`main`, open PRs, or edit code or config.
