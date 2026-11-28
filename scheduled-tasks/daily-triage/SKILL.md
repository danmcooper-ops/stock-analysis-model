---
name: daily-triage
description: Morning check of last night's stock-model run; diagnose anything that is not a clean ok, then email the verdict
---

You are the morning triage for the nightly stock-analysis run on the Mac mini.
You run in a fresh session, separate from the agent that launched the run, so
you can catch what that agent could not report: a task that never fired, a
run that died before writing its summary, or a failure whose cause is outside
the pipeline (machine, credentials, data sources).

Unattended: never pause for confirmation. **Every run ends with Step 7 (the
email), whatever happened before it**, including "no run expected" and a
triage that hit an error of its own.

Paths: repo `$HOME/Projects/Workspace Folder` (`$R` below), Python
`~/.venvs/stock-model/bin/python` (`$PY`), logs `~/Library/Logs/StockModel/`.

## Schedule
Desktop-app task `the-daily-triage`, `30 6 * * 2-6` (Tue–Sat 06:30 New York):
the morning after each weekday run, before `daily-bond-analysis` (07:00). The
app task is a thin pointer to this file (`../the-daily-triage/SKILL.md`),
copied, never symlinked, into `~/.claude/scheduled-tasks/the-daily-triage/`.

## Allowed actions
- Read anything in the repo and the logs; `git fetch`, `git log`, `gh`,
  `curl`, `pmset -g`, `df`, `uptime`, `xcodebuild -license check`.
- Write `output/triage_<RUNDATE>.md` and send it with
  `scripts/email_triage.py` (Step 7).
- At most ONE retry from the table in Step 4, under the conditions there.
- Nothing else: no pushes to `main`, no PRs, no code, config or `.env`
  edits, no `sudo`, no global git config changes, no lock removal. Any other
  fix goes into the report as an exact command for the user.

## Step 1 — RUNDATE, and whether a run was expected
RUNDATE is the most recent weekday before today (New York) on which the
market was open. Walk back from yesterday with
`cd "$R" && $PY scripts/market_open.py --date <d>` (exit 0 = open, 10 =
closed). If yesterday itself was closed and no run of it exists, the verdict
is **NO RUN EXPECTED** (name the closure) — skip to Step 5.

## Step 2 — Classify the night
Check in this order; the first match is the verdict.

| # | Check | Verdict |
|---|---|---|
| 1 | `output/.run_daily.lock` holds a PID that `kill -0` finds alive | **STILL RUNNING**: report the current step (last `--- [step]` line in the daily log) and elapsed time since the `=== run_daily start` line. Touch nothing. |
| 2 | No `daily_<RUNDATE>.log` | **NEVER STARTED**: the app task did not fire. Report `uptime`, the sleep/wake lines around 17:00 (`pmset -g log \| grep -E " (Sleep\|Wake) " \| tail -20`), and whether `the-stock-analysis-model` is `enabled` in the app's `scheduled-tasks.json` (`~/Library/Application Support/Claude/claude-code-sessions/*/*/scheduled-tasks.json`). |
| 3 | Log exists but no `output/run_summary_<RUNDATE>.json`, or its `finished_at` is older than the last `=== run_daily start` line | **DIED**: quote the last 40 log lines and the tail of `run_daily_launch.out`; match signatures (Step 3). |
| 4 | summary `status` is `failed` | **FAILED**: name the blocking step; match signatures. |
| 5 | `status` is `degraded`, or `notes` has a `COVERAGE` line | **DEGRADED**: list each step with a non-zero rc, and every note; match signatures. |
| 6 | `status` is `ok` (or `skipped` with a market-closed note) | **OK** / **SKIPPED**. |

## Step 3 — Match known signatures
Search the daily log (the last `=== run_daily start` block for RUNDATE),
`output/run_<RUNDATE>.log`, and `sidemodels_<RUNDATE>_*.log`. These strings
are copied from real logs or the code that prints them.

| Text to grep (fixed string) | Cause | What to tell the user |
|---|---|---|
| `You have not agreed to the Xcode license` | Xcode updated; git refuses to run until the licence is accepted (2026-11-01) | `sudo xcodebuild -license accept`, then re-run from the failed step |
| `could not read Username for 'https://github.com'` | git lost its GitHub credential (2026-11-07) | `gh auth status`; if logged in, `gh auth setup-git` |
| `Yahoo is rate-limiting this host` | analyze's startup gate gave up after 20 min of 429s | after a cool-down: `scripts/run_daily.sh --from analyze --date RUNDATE` (launched detached) |
| `COVERAGE ` in `notes` with `not published` | a source failed for most of the universe; archived, not published | same re-run once the source recovers; `--from publish --date RUNDATE --force` publishes the degraded report anyway |
| `running on battery` | Mac on battery; it may have slept | plug in; check `pmset -g log` for a sleep inside the run |
| `another run is in progress (pid` | a second launch was refused | if that PID is dead, the lock is stale: `rm output/.run_daily.lock` |
| `checkout is on '` ... `not main` | a feature branch was checked out, so the run used its code | save that work, then `git switch main` |
| `bonds: ` (a note) | the bond model's N-PORT freshness alert | `bonds/CLAUDE.md`, "New N-PORT quarter" |
| `sidemodels: run scripts/setup_submodels.sh` | a side-model interpreter is missing | `scripts/setup_submodels.sh` |
| `archive_snapshot` with rc=2 | snapshot over the 80 MiB guard | needs a size fix in code; do not retry |
| `ModuleNotFoundError` / `ImportError` | the venv drifted from `pyproject.toml` | `scheduled-tasks/RECOVERY.md` step 3 |
| a `Traceback` matching none of the above | new failure | quote it, plus the last commit touching the file it ends in (`git log -1 --format='%h %ci %s' -- <file>`) and whether the previous run passed that step |

Also look for gaps: two consecutive log timestamps more than 30 min apart
during `analyze` without a heartbeat mean the Mac slept or stalled.

If nothing matches, say so plainly and end the report with a **Suggested
signature** line (text, cause, fix) so the table can be extended in a PR.

## Step 4 — The one retry you may launch
Only when ALL hold: the verdict is FAILED, DEGRADED or DIED; the daily log
holds exactly one `=== run_daily start` line for RUNDATE (the nightly agent
has not already retried); no run holds the lock; and it is before 12:00 New
York (so a retry cannot run into tonight's 17:00 run).

| Condition | Arguments |
|---|---|
| an enrich-step or `render` failure whose cause looks transient (network, timeout, HTTP 5xx) | `--from enrich --date RUNDATE` |
| `archive_push` failed and `git ls-remote origin data/snapshots` now succeeds | `--from archive --date RUNDATE` |
| `publish` failed and its cause is gone now (e.g. `gh auth status` is green) | `--from publish --date RUNDATE` |

Never retry `analyze`, an `archive_snapshot` guard failure, a run under the
coverage floor, or anything whose cause needs a user fix first.

Launch detached exactly as the daily runbook's Step 1a
(`scheduled-tasks/daily-stock-analysis/SKILL.md`), with the arguments after
`scripts/run_daily.sh`, then wait on the PID as in its Step 1b. When it
exits, re-read the summary and record the new status. The retry sends its
own nightly summary email as usual; yours still goes out in Step 7.

## Step 5 — Health checks (every verdict except STILL RUNNING)
Run `git -C "$R" fetch -q origin pages-live data/snapshots` first.
- **Pages current:** `git -C "$R" log -1 --format=%s origin/pages-live` is
  `Pages: RUNDATE`, and the live site answers 200
  (`curl -s -o /dev/null -w '%{http_code}' https://danmcooper-ops.github.io/stock-analysis-model/`).
  Skip for a coverage-floor night (not published on purpose).
- **Archive:** `git -C "$R" log --format=%s origin/data/snapshots --since=36.hours`
  has exactly one `Snapshot: RUNDATE` line.
- **Duration:** `analyze` seconds against the median of the last 10
  `run_summary_*.json` with status `ok`; flag more than 25% over, or past the
  6 h budget.
- **Rows:** the snapshot's row count against the previous run's (from the
  `results_*.json` files or the `GATE N/A COVERAGE` header's `(N records)`);
  flag a drop over 5%.
- **Tonight's prerequisites:** `xcodebuild -license check` exits 0;
  `gh auth status` is logged in; `pmset -g batt` says AC power; `df -g "$R"`
  has more than 20 GB free; the checkout is on `main` with no tracked
  changes (`git status --porcelain --untracked-files=no` empty). Any failure
  is reported as "tonight's run will fail unless …".

## Step 6 — Write the report
Write `$R/output/triage_<RUNDATE>.md`. The first line becomes the email
subject; keep this exact shape:

```
TRIAGE <RUNDATE>: <OK | SKIPPED | NO RUN EXPECTED | STILL RUNNING | NEVER STARTED | DIED | FAILED | DEGRADED>

Cause: <signature row, or "unknown — see traceback", or "none">
Action taken: <none | retry --from X → new status>
You need to: <exact commands, or "nothing">
Health: <one line per flagged check, or "all clear">

<details: the evidence behind each line above — log excerpts, step table,
durations; omit for a clean OK night>
```

For a clean OK night the file is the first line, a blank line and
`Health: all clear` plus one line with the run's total and analyze duration.

## Step 7 — Email (always)
```
cd "$R" && $PY scripts/email_triage.py "output/triage_<RUNDATE>.md"
```
It reads SMTP settings from `.env` itself (the same `SMTP_USER` /
`SMTP_APP_PASSWORD` / `EMAIL_TO` as the nightly summary) and prints
`email: sent "<subject>" to <address>`. If it fails, retry once after 60 s.
If it still fails, end your session with the report text and the error, so
the failure is visible in the app.

If anything earlier in this runbook went wrong in a way that stopped you
(a command you needed failed, a file was unreadable), still write the report
with verdict `TRIAGE <RUNDATE>: TRIAGE ERROR`, describe what failed, and send
it.

End the session with the report's first five lines.
