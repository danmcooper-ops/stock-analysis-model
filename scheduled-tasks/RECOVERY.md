# Recovering the local working copy

What to do when `~/Projects/Workspace Folder` has been deleted, moved, renamed
or restored from a backup — the routines stop working because every path in
`daily-stock-analysis`, `publish-stock-report` and `weekly-backtest` is
absolute, and `~/.claude/scheduled-tasks/*` are symlinks into that directory
(see this directory's `README.md`).

Deleted **2026-09-09**; this file is the record of what that costs and how to
get back.

## Before anything else: check the Trash

macOS `rm` from Finder moves rather than deletes. If the folder is still in
`~/.Trash`, "Put Back" restores everything below — including the two things
git cannot give you back (`.env` and the price cache).

```bash
ls -la ~/.Trash | grep -i workspace
```

Restoring is strictly better than rebuilding: stop here if it works, then
verify with the "Check it worked" section at the bottom.

## What was in there, and what it costs

| Thing | Path (relative to the repo root) | Recoverable? |
|---|---|---|
| Source tree | — | **Yes** — `git clone` (branch `main`) |
| Snapshot corpus | `.claude/worktrees/snapshots-data` | **Yes** — branch `data/snapshots`, every archived day |
| Published site | `.claude/worktrees/pages-live` | **Yes** — branch `pages-live` (the live site was never affected) |
| Python venv | `.claude/worktrees/phase-1-api/.venv` | Rebuild — `pip install -e ".[dev]"` |
| **API keys** | `.env` | **No** — gitignored, never committed. Re-issue or re-enter by hand |
| **Price cache** | `output/prices/*.parquet` | **No** — not in git. Re-download, hours |
| Today's results | `output/results_*.json`, `*.html` | Only as far back as the last archived snapshot |
| DuckDB index | `output/snapshots.duckdb` | Rebuild — `ingest_snapshots.py` |
| SEC facts cache | `data/cache/sec_facts/` | Rebuild — disposable, costs one slow run (~0.4 GB) |
| Uncommitted work | any branch | **No** |

The two "No"s are the whole story: **the API keys and the price parquets.**
Everything else is either on GitHub or regenerates.

## Rebuild

### 1. Clone back to the exact same path

The path is hardcoded throughout the runbooks. Restore it verbatim — a
different path means editing all three `SKILL.md` files.

```bash
mkdir -p "$HOME/Projects"
git clone https://github.com/danmcooper-ops/stock-analysis-model.git \
  "$HOME/Projects/Workspace Folder"
cd "$HOME/Projects/Workspace Folder"
```

### 2. Recreate the three worktrees

```bash
git worktree add .claude/worktrees/pages-live      pages-live
git worktree add .claude/worktrees/snapshots-data  data/snapshots
git worktree add .claude/worktrees/phase-1-api     feature/phase-1-api
```

`snapshots-data` is the large one — it carries the whole
`results_YYYY-MM-DD.json.gz` corpus the weekly backtest calibrates on.

### 3. Rebuild the venv

The runbooks invoke Python as
`.claude/worktrees/phase-1-api/.venv/bin/python`, so the venv goes there:

```bash
VENV="$HOME/Projects/Workspace Folder/.claude/worktrees/phase-1-api/.venv"
python3 -m venv "$VENV"
cd "$HOME/Projects/Workspace Folder"
"$VENV/bin/pip" install -e ".[dev]"
"$VENV/bin/pip" install certifi
```

`certifi` is required — every runbook step sets `SSL_CERT_FILE` from
`python -m certifi` to work around macOS certificate verification.

### 4. Restore the scheduled-task symlinks

**This is why a manual run fails after a delete**: the symlinks dangle, so
Claude Code has no task definition to read.

```bash
mkdir -p ~/.claude/scheduled-tasks
ln -sfn "$HOME/Projects/Workspace Folder/scheduled-tasks/daily-stock-analysis" \
   ~/.claude/scheduled-tasks/daily-stock-analysis
ln -sfn "$HOME/Projects/Workspace Folder/scheduled-tasks/publish-stock-report" \
   ~/.claude/scheduled-tasks/publish-stock-report

readlink ~/.claude/scheduled-tasks/daily-stock-analysis   # verify: no error
readlink ~/.claude/scheduled-tasks/publish-stock-report
```

### 5. Recreate `.env`

Not recoverable — write it from your password manager, or re-issue each key.
`SEC_EMAIL` is mandatory (SEC EDGAR rejects requests without a User-Agent
contact); the rest degrade gracefully.

```bash
cat > "$HOME/Projects/Workspace Folder/.env" <<'ENV'
SEC_EMAIL=
FMP_API_KEY=
TIINGO_API_KEY=
FINNHUB_API_KEY=
ANTHROPIC_API_KEY=
ENV
chmod 600 "$HOME/Projects/Workspace Folder/.env"
```

Without `ANTHROPIC_API_KEY` the run still succeeds — the Macro Outlook tab and
the per-sector outlook bullets just render empty, and the log says "macro
narrative skipped".

### 6. Reseed the price cache

**Do this before the first analysis run.** Step 0 of the daily routine only
refreshes tickers it already has (`ls output/prices/*.parquet`), so against an
empty directory it fetches the four benchmarks and nothing else — and the
report silently loses its price charts and Yesterday's Rating.

```bash
PYTHON="$HOME/Projects/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/download_prices.py --output-dir output/prices --universe us
```

Several hours for ~7–10k tickers at the 0.35 s inter-request delay. Without
`--refresh`/`--max-age-days` it runs in resume mode (existing files skipped),
so it is safe to interrupt and re-run until it completes.

### 7. Rebuild the DuckDB index from the archive

```bash
PYTHON="$HOME/Projects/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/ingest_snapshots.py --results-dir .claude/worktrees/snapshots-data
```

Idempotent. Until this runs, cross-run readers (rating history, yesterday's
rating, gate N/A deltas) fall back to parsing the JSON snapshots — correct,
just slower.

## Check it worked

```bash
cd "$HOME/Projects/Workspace Folder"
readlink ~/.claude/scheduled-tasks/daily-stock-analysis    # symlink resolves
git worktree list                                          # three worktrees
.claude/worktrees/phase-1-api/.venv/bin/python -c "import yfinance, pandas, duckdb, certifi; print('deps ok')"
test -s .env && echo ".env present"
ls output/prices/*.parquet | wc -l                         # thousands, not 4
ls .claude/worktrees/snapshots-data/results_*.json* | wc -l # the corpus
```

Then a cheap end-to-end proof before trusting the nightly run — the S&P 500
universe instead of `us`, ~20 minutes rather than 3–6 hours:

```bash
PYTHON="$HOME/Projects/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/analyze_stock.py --prices-dir output/prices
```

If that produces `output/results_<today>.json` and the matching HTML, the
routine is back.

## Making the next delete cheaper

- `.env` belongs in a password manager, not only on disk. It is the one file
  in the tree that no amount of git can restore.
- `output/prices/` is the expensive rebuild. It is pure cache, but it is
  *hours* of cache — worth including in whatever backs up the Mac, or worth
  keeping outside the repo directory so a repo delete does not take it.
- The absolute paths and the `~/.claude` symlinks are what turn "deleted a
  folder" into "the routine is gone". Moving the repo has the same effect as
  deleting it; see this directory's `README.md`.
