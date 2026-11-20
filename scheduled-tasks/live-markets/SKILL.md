---
name: live-markets-board
description: Start the day's 15-minute live Markets board loop for the iOS app.
---

Unattended weekday run (09:25 New York). Work autonomously; never pause.

1. Run `scheduled-tasks/live-markets/mac_run.sh` from the repo root
   (`/Users/dmcooper/Projects/Workspace Folder`) in the foreground. It returns
   within seconds: it starts `scripts/live_markets.py --loop` detached, which
   rebuilds the Macro Markets board from Yahoo every 15 minutes from 09:30 to
   16:15 ET, uploads it to the Cloudflare `LIVE` KV and exits by itself.
2. Report in two or three lines: what it printed (STARTED <pid>, ALREADY
   RUNNING, MARKET CLOSED or FAILED TO START) and the previous day's RESULT
   line. Lead with FAILED or any previous day with `ok=0` prominently.

Do not wait for the loop, edit code or config, commit, push or publish
anything else. If the script is missing, report "live-markets runbook not
found" and stop. If this file carries a **DORMANT** banner, run nothing and
report it as skipped.
