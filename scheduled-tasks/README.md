# Scheduled task definitions

Version-controlled copies of the Claude Code scheduled tasks that drive the
daily pipeline.

| Task | What it does |
|---|---|
| `daily-stock-analysis/SKILL.md` | End-of-day run: analysis → enrichment (FDIC, REIT, XBRL, FDA) → re-render → snapshot commit → portfolio/gate/validation reports → publish |
| `publish-stock-report/SKILL.md` | Copies the five report artifacts into the `pages-live` worktree, amends its single commit, force-pushes to GitHub Pages |

## These ARE the live files (symlinked since 2026-08-10)

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
