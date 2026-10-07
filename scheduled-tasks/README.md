# Scheduled task definitions

Version-controlled copies of the Claude Code scheduled tasks that drive the
daily pipeline.

| Task | What it does |
|---|---|
| `cloud-daily-stock-analysis/` | **Dormant since 2026-10-03** (paused at the Mac mini cut-over; live 2026-09-10 to 2026-10-02). A Claude Code cloud Routine ("Daily stock analysis (cloud)", `0 21 * * 1-5` UTC) starts a fresh session each weekday that runs `run.sh`: stage the newest days of the `data/snapshots` archive → price cache → analysis → enrichment → re-render → snapshot commit to `data/snapshots` → portfolio/gate/momentum reports → rebuild + force-push `pages-live`. `SKILL.md` is what the Routine's session follows |
| `daily-stock-analysis/SKILL.md` | **The live daily routine since 2026-10-05** (Mac mini, weekdays 17:00 New York). Launches `scripts/run_daily.sh` (preflight → prices → analysis → enrichment → re-render → snapshot archive → reports → publish), then summarizes `output/run_summary_<date>.json`. The cloud routine must stay paused while this is on |
| `the-stock-analysis-model/SKILL.md` | **The Claude desktop app's scheduled task** for the daily run (weekdays 17:00 New York, Mac mini). A thin entry point: it tells the session to follow `daily-stock-analysis/SKILL.md`, so the runbook keeps a single copy. The app refuses symlinks, so `~/.claude/scheduled-tasks/` holds a **copy** of this file (see below) |
| `publish-stock-report/SKILL.md` | Copies the five report artifacts into the `pages-live` worktree, amends its single commit, force-pushes to GitHub Pages |
| `cloud-weekly-backtest/` | **Dormant since 2026-10-11** (paused at the Mac mini cut-over; on 2026-09-27 the cloud egress proxy blocked Yahoo and the run measured nothing). `run.sh` is still the weekly pipeline: the Mac runs it unchanged through `weekly-backtest/mac_run.sh`, and `SKILL.md` §3–4 stay the summary instructions for both. As a cloud Routine each Sunday it runs `run.sh`: stage every snapshot since 2026-07-06 (+ the persisted `returns/` sidecars) out of `data/snapshots` → cold price download with a coverage gate → `measure` offline → week-over-week regression check → commit the summary, xlsx and sidecars to `data/snapshots`. Measurement only; calibration stays off until `readiness` clears it |
| `weekly-backtest/` | **The live weekly backtest since 2026-10-11** (Mac mini). launchd (`com.stockmodel.weekly.plist`, Sundays 09:45 New York) runs `mac_run.sh`, which runs `../cloud-weekly-backtest/run.sh` with the Mac's environment and a lock. `SKILL.md` is the summary runbook: it waits for the job, checks it ran, and writes the summary; it never starts the job except for its one retry. `.github/workflows/weekly-backtest-check.yml` goes red on Monday if no summary landed. The cloud routine must stay paused while this is on |
| `the-weekly-backtest/SKILL.md` | **The Claude desktop app's scheduled task** for the weekly summary (Sundays 12:30 New York, Mac mini). A thin entry point to `weekly-backtest/SKILL.md`, kept as a **copy** in `~/.claude/scheduled-tasks/` like the daily one (see below) |
| `mac-mini/` | Moving to a Mac mini (`MAC-MINI-SETUP.md`): `pack_old_mac.sh` inventories the old Mac and packs `.env` (plus caches on request); `bootstrap_mini.sh` sets up the new Mac, re-runnable, `--check` to report only. Neither schedules anything |

The Mac runbooks (and the weekly launchd job) assumed a persistent local checkout; that
checkout was deleted on 2026-09-09 and the daily run moved to the cloud
routine the next day. They stay here as the reference for the steps and for a
future Mac rebuild — but the cloud routine and a rebuilt Mac routine must not
both run: each appends to `data/snapshots` and force-pushes `pages-live`.

To move the daily run and the weekly backtest onto a dedicated Mac (a Mac
mini), follow `MAC-MINI-SETUP.md`. It covers machine setup, seeding the
local state, and the cut-over order. The scripts are ready: `run_daily.sh`
carries the cloud steps, and `weekly-backtest/mac_run.sh` runs the weekly
`run.sh` outside the container.

If the repo directory itself is gone — deleted, moved or restored from a
backup — see `RECOVERY.md` in this directory for the full rebuild.

To bring the hosted Supabase project up for the first time — pushing the
migrations, exposing the `pipeline` schema, backfilling the archive and
wiring 06a — see `HOSTED-SETUP.md`. Once it is live, `RECOVERY.md` covers
`DB_PRIMARY`, the cutover streak and restoring.

## The Mac routines (symlinked since 2026-08-10; dormant 2026-09-09 to 2026-10-04; live again on the Mac mini)

This section describes the arrangement the Mac routines used, for when they
are rebuilt. Nothing in it applies to the cloud routine, which reads
`cloud-daily-stock-analysis/run.sh` straight from a fresh clone of `main`.

**The desktop app's tasks are copies, not symlinks.** The Claude desktop app
runs `~/.claude/scheduled-tasks/the-stock-analysis-model/SKILL.md` (since
2026-10-03) and `~/.claude/scheduled-tasks/the-weekly-backtest/SKILL.md`
(since 2026-10-11), and refuses to open a task file reached through a symlink
("symlink detected before open; refusing to open"). That folder must be a real
folder holding a copy of `the-stock-analysis-model/SKILL.md` from here. After
changing the tracked copy, copy it over:

```bash
mkdir -p ~/.claude/scheduled-tasks/the-stock-analysis-model
cp "$HOME/Projects/Workspace Folder/scheduled-tasks/the-stock-analysis-model/SKILL.md" \
   ~/.claude/scheduled-tasks/the-stock-analysis-model/SKILL.md
mkdir -p ~/.claude/scheduled-tasks/the-weekly-backtest
cp "$HOME/Projects/Workspace Folder/scheduled-tasks/the-weekly-backtest/SKILL.md" \
   ~/.claude/scheduled-tasks/the-weekly-backtest/SKILL.md
```

The copy is a thin entry point: it sends the session to
`daily-stock-analysis/SKILL.md` by absolute path, so the runbook itself is
still read straight from this checkout. Editing the runbook here changes the
routine at runtime; committing gives the change history.

`~/.claude/scheduled-tasks/daily-stock-analysis` and
`~/.claude/scheduled-tasks/publish-stock-report` are older symlinks into this
directory, from before the desktop app task. The app does not run them. (The
old `weekly-backtest` symlink was removed at the 2026-10-11 cut-over.)

Consequences of the arrangement:

- **Moving or renaming this repo breaks the routine** — the entry point and
  the symlinks name the absolute path `~/Projects/Workspace Folder/scheduled-tasks/`.
  If the repo moves (as the bond-analysis repo did), edit the runbook path in
  `~/.claude/scheduled-tasks/the-stock-analysis-model/SKILL.md` (and the
  tracked copy) and recreate the symlinks:

  ```bash
  ln -sfn "<new-repo-path>/scheduled-tasks/daily-stock-analysis" \
     ~/.claude/scheduled-tasks/daily-stock-analysis
  ln -sfn "<new-repo-path>/scheduled-tasks/publish-stock-report" \
     ~/.claude/scheduled-tasks/publish-stock-report
  ```

- **A checkout changes the live routines.** The scheduler reads whatever the
  working tree holds — switching branches or checking out an old commit swaps
  the task definitions with it.

To verify the task copy is current (no output means identical) and the links
are intact:

```bash
diff "$HOME/Projects/Workspace Folder/scheduled-tasks/the-stock-analysis-model/SKILL.md" \
     ~/.claude/scheduled-tasks/the-stock-analysis-model/SKILL.md
diff "$HOME/Projects/Workspace Folder/scheduled-tasks/the-weekly-backtest/SKILL.md" \
     ~/.claude/scheduled-tasks/the-weekly-backtest/SKILL.md
readlink ~/.claude/scheduled-tasks/daily-stock-analysis
readlink ~/.claude/scheduled-tasks/publish-stock-report
```
