---
name: daily-currency-check
description: Weekday check on the Mac mini — confirm today's GitHub Actions build of the currency screener reached the live site, and report what changed
---

You are running the daily currency screener check on the Mac mini. The
Claude desktop app's scheduled task `the-daily-currency-check` (weekdays
18:55 New York) sends you here. It replaced the cloud routine "Daily
Currency Screener".

GitHub Actions (`.github/workflows/build.yml`) builds and deploys the site;
GitHub often starts that run 2–3 hours after its 16:15 target. Your job is
the part Actions cannot do: confirm the build landed and report what changed
in the model. All of the checking is done by `scripts/check_site.py`; you
launch it, wait, and write the summary from its report.

## Execution mode
- **Fully autonomous.** Nobody is watching. Never pause for confirmation.
- **Read-only, with one exception.** Never rebuild, publish, commit or
  push, and never change gates, weights or thresholds. The only outward
  action is the `Snapshot: <date>` commit to the `data/snapshots` branch that
  `mac_run.sh` makes itself (via `scripts/archive_snapshots.py`); do not
  make or retry it by hand. The site can only be deployed by Actions on
  the default branch, so a manual rebuild here would not reach it anyway.

## Where things are
- Checkout: `~/Projects/Currency-Screener-` (adjust if it lives elsewhere).
- Work directory (`W`): `~/CurrencyScreener` — the single folder for
  everything the check keeps: `report.md`, `status.json`, `.check.lock` (the
  PID while a run is live), `history/<run_date>/{results,manifest}.json` (one
  snapshot per day) and `logs/`.
- Log: `~/CurrencyScreener/logs/check_<date>.log`
- Older runs used `~/Library/Application Support/CurrencyScreener/check` and
  `~/Library/Logs/CurrencyScreener`; `mac_run.sh` moves anything left there
  into `~/CurrencyScreener` on its next run and leaves symlinks at the old
  paths, so they still resolve to the same single folder.

## Step 1 — Run the check
Run this as a Bash call **in the background** (run_in_background), and wait
for its completion notification. It polls the site every 10 minutes for up
to 110 minutes until today's build lands, then exits; do not poll it
yourself with sleep loops, and never start a second copy:
```
cd "$HOME/Projects/Currency-Screener-" && scheduled-tasks/daily-check/mac_run.sh
```
If it prints `ALREADY RUNNING`, wait for that PID to exit instead.

## Step 2 — Read the outcome
TODAY is `TZ=America/New_York date +%F`. Read `"$W/status.json"` and
`"$W/report.md"`. If `status.json`'s `date` is not TODAY, the script died
before writing them: report that with the tail of today's log and stop.

`result` is one of:
- **OK** — today's build is live. Summarise the report (below).
- **SKIPPED** — the ECB published no rates today (TARGET holiday), so the
  build gate correctly skipped. One line; not an error.
- **LATE** — the site still shows an earlier date after 110 minutes and the
  build has not failed. Say what the report says about today's workflow
  runs (none started yet / queued / running).
- **FAILED** — today's scheduled build finished without success and nothing
  is still running. Lead with it and give the run URL from the report.
- **SITE-WRONG** — the page is missing (404) or README-sized (~16 KB):
  Pages is serving the wrong thing (Settings → Pages → Source must be
  "GitHub Actions"). Lead with it, prominently.
- **JPY-INVERTED** — JPY's Valuation pillar fell below 50. JPY is the
  model's canonical cheap currency (BIS REER 164 in 1994 → ~65), so this
  means the REER sign convention has inverted and the heaviest pillar runs
  backwards. Flag it loudly; it is the most consequential silent failure the
  model has.

## Step 3 — Write the summary
Concise, leading with what changed:
- **Rating changes** vs the previous archived day: currency, old → new, and
  the composite move. If none, one line, and keep the rest brief. If the
  report says the comparison is unavailable (first run on this machine), say
  so rather than inventing one.
- Top 5 by composite.
- JPY's Valuation pillar score.
- Any data-quality lines (stale cache slots, provenance events, an
  unusually sized page).
- One line confirming the live site: its size and run date.
- One line from the report's **Archive** section: the `data/snapshots`
  commit, "already holds every day", or the push failure verbatim. A failed
  push does not change the result, but say it plainly: until the next
  successful run, the new day exists only on this Mac.
