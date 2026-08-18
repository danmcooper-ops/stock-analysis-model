---
name: publish-stock-report
description: Publish today's stock analysis HTML to GitHub Pages via the single-commit pages-live branch
---

You are running the publish step for the daily stock analysis. This task copies the report artifacts into the `pages-live` worktree, amends that branch's single commit, and force-pushes so GitHub Pages serves the fresh report — **without growing git history** (the branch always holds exactly one commit).

This routine assumes `output/stock_analysis_results_YYYY-MM-DD.html` already exists for the run date — produced by the `daily-stock-analysis` routine. If the HTML is missing, stop and report; there is nothing to publish.

## Run date (IMPORTANT)
Use the **run-START date** of the analysis, not `$(date)` — if the 3–6 h analysis crossed midnight, `$(date)` is wrong. Determine RUNDATE from the newest `output/stock_analysis_results_*.html` and substitute it literally in the commands below.

## Paths
- **Main repo:** `/Users/danmcooper/Desktop/Workspace Folder`
- **Pages worktree:** `/Users/danmcooper/Desktop/Workspace Folder/.claude/worktrees/pages-live` (branch `pages-live`)
- **GitHub Pages URL:** https://danmcooper-ops.github.io/stock-analysis-model/

If the worktree is missing, recreate it: `git worktree add .claude/worktrees/pages-live pages-live`

## IMPORTANT: Command format
Each command is a **separate** Bash call (single line each). This is required for permission matching to work.

## Steps

### 1. Verify the run's HTML exists
```
ls -la "/Users/danmcooper/Desktop/Workspace Folder/output/stock_analysis_results_RUNDATE.html"
```
If the file is missing, stop and report — there's nothing to publish. Do not proceed.

### 2. Copy the six artifacts into the Pages worktree
The HTML lazy-loads `prices_meta.json`, `hist.json`, `details.json`, and the per-ticker shards in `vol/` and `px/` from its own directory at runtime, so **all six artifacts must be published together**. (The dense `prices.json` is retired — per-ticker `px/` shards + the small `prices_meta.json` replaced it on 2026-08-11; if a `docs/prices.json` is still present, delete it as part of the publish.) Run each as a **separate** Bash call:
```
cp "output/stock_analysis_results_RUNDATE.html" "/Users/danmcooper/Desktop/Workspace Folder/.claude/worktrees/pages-live/docs/index.html"
```
```
cp "/Users/danmcooper/Desktop/Workspace Folder/output/prices_meta.json" "/Users/danmcooper/Desktop/Workspace Folder/.claude/worktrees/pages-live/docs/prices_meta.json"
```
```
cp "/Users/danmcooper/Desktop/Workspace Folder/output/hist.json" "/Users/danmcooper/Desktop/Workspace Folder/.claude/worktrees/pages-live/docs/hist.json"
```
```
cp "/Users/danmcooper/Desktop/Workspace Folder/output/details.json" "/Users/danmcooper/Desktop/Workspace Folder/.claude/worktrees/pages-live/docs/details.json"
```
```
PYTHON="/Users/danmcooper/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; cd "/Users/danmcooper/Desktop/Workspace Folder"; "$PYTHON" scripts/publish_vol_shards.py
```
(The HTML cp source is relative — run from the main repo root. The sidecar paths are absolute so they work from any cwd. `publish_vol_shards.py` syncs **both** shard directories — `vol/` from the `vol` manifest and `px/` from the `manifest` key of `prices_meta.json`.)

**The shard sync is not optional, and must NOT be a `cp -R` or a bare `rsync`.** `docs/vol/` holds ~2,300 per-ticker volume shards (~59 MB) that feed the popup price chart's volume strip, and `docs/px/` holds ~2,300 per-ticker close-price shards (~53 MB) that feed every price chart.

Two independent hazards make a wholesale directory copy wrong:
1. **Staleness.** The shard set changes run to run (2,284 on 2026-08-05 → 2,288 on 2026-08-07), so "the directory is already there" is never evidence it is current. Skipping the sync publishes today's `index.html` against yesterday's volume data — no error, the popups just silently show stale or missing volume bars. Missed on the 2026-08-07 run, caught only by diffing shard counts.
2. **iCloud conflict copies.** The repo is under `~/Desktop`, which iCloud Drive syncs, so `output/vol/` continuously re-accumulates space-suffixed duplicates (`AAPL 2.json`, mode 600 vs 644 for real shards). They regenerate *after* any sweep — 2,137 of 4,425 files on 2026-08-08 — so no copy that walks the directory can be trusted.

`scripts/publish_vol_shards.py` therefore copies **by manifest**: it reads the `vol` and `manifest` keys of `output/prices_meta.json` (the only lists the client ever requests), copies exactly those `<ticker>.json` files into `docs/vol/` and `docs/px/` respectively, deletes anything in either destination not on its manifest, and asserts each destination set equals its manifest before returning non-zero on any mismatch. If it exits non-zero, stop and report — do not publish.

If any sidecar is missing, the `cp` will fail; if a manifested shard is missing, the shard script exits non-zero. Either way, stop and report rather than publishing partial state — the page silently fails to load charts/details instead of erroring.

### 3. Amend the single commit and force-push
```
git -C "/Users/danmcooper/Desktop/Workspace Folder/.claude/worktrees/pages-live" add -A
```
```
git -C "/Users/danmcooper/Desktop/Workspace Folder/.claude/worktrees/pages-live" commit --amend -m "Pages: RUNDATE"
```
```
git -C "/Users/danmcooper/Desktop/Workspace Folder/.claude/worktrees/pages-live" push --force origin pages-live
```
`git add -A` is safe against iCloud junk because the `pages-live` worktree carries a `.gitignore` with `docs/vol/* *.json` — no real ticker symbol contains a space, so this ignores every conflict copy while leaving genuine shards tracked. This is deliberate belt-and-braces: the shards regenerate continuously, including in the window between the manifest copy and the `add`. If that `.gitignore` ever goes missing, recreate it before staging, and sanity-check with `git status --porcelain | wc -l` — a publish should stage roughly the shard delta plus four files, never ~2,000 extra.

The push triggers `.github/workflows/deploy-pages.yml` (which lives on the `pages-live` branch itself). `--force` is expected and required — the branch history is intentionally always exactly one commit, so old artifact blobs become unreachable instead of accumulating.

### 4. Verify the deploy
Wait for the workflow run on branch `pages-live` to complete:
```
gh run list --repo danmcooper-ops/stock-analysis-model --branch pages-live --limit 1
```
Then confirm the site serves:
```
curl -sS -o /dev/null -w "%{http_code}" -L "https://danmcooper-ops.github.io/stock-analysis-model/"
```
Expect `200`. A `404` previously meant Pages had been disabled repo-side; the workflow's `enablement: true` re-enables it automatically, so retry once after a minute before reporting failure.

## Success criteria
- All six artifacts copied and committed to `pages-live` (single amended commit) — `index.html`, the three sidecars (`prices_meta.json`, `hist.json`, `details.json`), **and both shard directories (`vol/`, `px/`)**
- `scripts/publish_vol_shards.py` exited 0 (`docs/vol` and `docs/px` each match their `prices_meta.json` manifest exactly). Do **not** substitute a raw `ls output/vol | wc -l` comparison — those directories contain iCloud conflict copies and will not match by design
- Force-push succeeded and the deploy workflow completed green
- Live URL returns HTTP 200

## Notes
- **main is not touched by this routine.** `docs/` is gitignored on main; the report is served exclusively from `pages-live`. The old sweet-gauss copy→merge→fast-forward flow is retired (as is its merge-order snag).
- If `git commit --amend` fails because the branch has no commit yet (fresh worktree after a re-clone), use a plain `git commit -m "Pages: RUNDATE"` for that first publish.
- The `github-pages` environment allows deploys from `pages-live` (branch policy added 2026-07-18). If a deploy fails with "not allowed to deploy due to environment protection rules", re-add it: `gh api -X POST repos/danmcooper-ops/stock-analysis-model/environments/github-pages/deployment-branch-policies -f name=pages-live`
- This task can be invoked manually to retry publishing without re-running the 3–6 hour analysis.
- All required allow rules are in `.claude/settings.json` so the routine runs unassisted.
