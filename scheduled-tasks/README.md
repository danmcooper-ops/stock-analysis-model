# Scheduled task definitions

Version-controlled copies of the Claude Code scheduled tasks that drive the
daily pipeline.

| Task | What it does |
|---|---|
| `daily-stock-analysis/SKILL.md` | End-of-day run: analysis → enrichment (FDIC, REIT, XBRL, FDA) → re-render → snapshot commit → portfolio/gate/validation reports → publish |
| `publish-stock-report/SKILL.md` | Copies the five report artifacts into the `pages-live` worktree, amends its single commit, force-pushes to GitHub Pages |

## These are copies, not the live files

Claude Code executes the copies under `~/.claude/scheduled-tasks/<name>/SKILL.md`.
**Editing the files here changes nothing at runtime.** They are tracked so the
routines have history, reviewable diffs, and a recovery point — the live ones sit
outside any repo and are otherwise one `rm` from being unrecoverable.

After changing a routine, mirror it in both places:

```bash
cp scheduled-tasks/publish-stock-report/SKILL.md \
   ~/.claude/scheduled-tasks/publish-stock-report/SKILL.md
```

To check they have not drifted:

```bash
diff -r scheduled-tasks/daily-stock-analysis ~/.claude/scheduled-tasks/daily-stock-analysis
diff -r scheduled-tasks/publish-stock-report ~/.claude/scheduled-tasks/publish-stock-report
```

Symlinking `~/.claude/scheduled-tasks/<name>` at these files would remove the
drift risk, at the cost of breaking the routines if this repo is ever moved or
renamed.
