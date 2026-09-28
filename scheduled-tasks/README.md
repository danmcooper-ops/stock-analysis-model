# Scheduled task definitions

Version-controlled copies of the Claude Code scheduled tasks that drive the
daily pipeline.

| Task | What it does |
|---|---|
| `cloud-daily-stock-analysis/` | **The live daily routine since 2026-09-10.** A Claude Code cloud Routine ("Daily stock analysis (cloud)", `0 21 * * 1-5` UTC) starts a fresh session each weekday that runs `run.sh`: stage the newest days of the `data/snapshots` archive → price cache → analysis → enrichment → re-render → snapshot commit to `data/snapshots` → portfolio/gate/momentum reports → rebuild + force-push `pages-live`. `SKILL.md` is what the Routine's session follows |
| `daily-stock-analysis/SKILL.md` | **Dormant** (Mac). Launches `scripts/run_daily.sh` (preflight → prices → analysis → enrichment → re-render → snapshot archive → reports → publish), then summarizes `output/run_summary_<date>.json`. Must stay off while the cloud routine is live |
| `publish-stock-report/SKILL.md` | Copies the five report artifacts into the `pages-live` worktree, amends its single commit, force-pushes to GitHub Pages |
| `cloud-weekly-backtest/` | **The live weekly backtest.** A cloud Routine each Sunday runs `run.sh`: stage every snapshot since 2026-07-06 (+ the persisted `returns/` sidecars) out of `data/snapshots` → cold price download with a coverage gate → `measure` offline → week-over-week regression check → commit the summary, xlsx and sidecars to `data/snapshots`. Measurement only; calibration stays off until `readiness` clears it |
| `weekly-backtest/` | **Dormant** (Mac launchd job + runbook). The steps the cloud routine replaced; must stay off while it is live |

The Mac runbooks (and the weekly launchd job) assumed a persistent local checkout; that
checkout was deleted on 2026-09-09 and the daily run moved to the cloud
routine the next day. They stay here as the reference for the steps and for a
future Mac rebuild — but the cloud routine and a rebuilt Mac routine must not
both run: each appends to `data/snapshots` and force-pushes `pages-live`.

To move the daily run onto a dedicated Mac (a Mac mini), follow
`MAC-MINI-SETUP.md`: it lists what `run_daily.sh` still lacks compared with
the cloud script, and the cut-over order.

If the repo directory itself is gone — deleted, moved or restored from a
backup — see `RECOVERY.md` in this directory for the full rebuild.

## The Mac routines (symlinked since 2026-08-10; dormant since 2026-09-09)

This section describes the arrangement the Mac routines used, for when they
are rebuilt. Nothing in it applies to the cloud routine, which reads
`cloud-daily-stock-analysis/run.sh` straight from a fresh clone of `main`.

`~/.claude/scheduled-tasks/daily-stock-analysis` and
`~/.claude/scheduled-tasks/publish-stock-report` are symlinks into this
directory, so Claude Code executes these tracked files directly. Editing here
changes the routines at runtime; committing gives the change history.

Consequences of the symlink arrangement:

- **Moving or renaming this repo breaks both routines** — the symlinks point at
  the absolute path `~/Projects/Workspace Folder/scheduled-tasks/`.
  If the repo moves (as the bond-analysis repo did), recreate them:

  ```bash
  ln -sfn "<new-repo-path>/scheduled-tasks/daily-stock-analysis" \
     ~/.claude/scheduled-tasks/daily-stock-analysis
  ln -sfn "<new-repo-path>/scheduled-tasks/publish-stock-report" \
     ~/.claude/scheduled-tasks/publish-stock-report
  ```

- **A checkout changes the live routines.** The scheduler reads whatever the
  working tree holds — switching branches or checking out an old commit swaps
  the task definitions with it.

To verify the links are intact:

```bash
readlink ~/.claude/scheduled-tasks/daily-stock-analysis
readlink ~/.claude/scheduled-tasks/publish-stock-report
```
