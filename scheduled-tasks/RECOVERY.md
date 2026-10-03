# Recovering the local working copy

What to do when `~/Projects/Workspace Folder` has been deleted, moved, renamed
or restored from a backup — the routines stop working because every path in
`daily-stock-analysis`, `publish-stock-report` and `weekly-backtest` is
absolute, and the desktop app's task points at that directory by absolute
path (see this directory's `README.md`).

Deleted **2026-09-09**; this file is the record of what that costs and how to
get back.

**Since 2026-10-05 the daily run is the Mac mini again** (it ran as the
cloud Routine in `cloud-daily-stock-analysis/` from 2026-09-10; see this
directory's `README.md`). Rebuilding the local copy is what restores the
nightly run. If the Mac cannot be rebuilt quickly, re-enable the cloud
Routine instead, but only after pausing the Mac task: two daily runs would
both append to `data/snapshots` and both force-push `pages-live`.

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
| Python venv | `~/.venvs/stock-model` (outside the repo) | Rebuild — see step 3 |
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

### 2. Recreate the two worktrees

```bash
git worktree add .claude/worktrees/pages-live      pages-live
git worktree add .claude/worktrees/snapshots-data  data/snapshots
```

`snapshots-data` is the large one — it carries the whole
`results_YYYY-MM-DD.json.gz` corpus the weekly backtest calibrates on.

### 3. Rebuild the venv

The runbooks and `scripts/run_daily.sh` invoke Python as
`~/.venvs/stock-model/bin/python`. It lives outside the repo on purpose: the
old location, the `phase-1-api` worktree's `.venv`, was deleted along with that
worktree on 2026-09-13, in the middle of a run. Install the pinned
dependencies (not an editable install, which would tie the venv to one
checkout):

```bash
VENV="$HOME/.venvs/stock-model"
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -m venv "$VENV"
cd "$HOME/Projects/Workspace Folder"
"$VENV/bin/pip" install -r requirements.txt 'pytest~=9.1' 'pytest-cov~=7.0' 'hypothesis~=6.165' 'ruff~=0.16'
```

`certifi` is required — every runbook step sets `SSL_CERT_FILE` from
`python -m certifi` to work around macOS certificate verification.

### 4. Restore the scheduled task

The desktop app's task, `~/.claude/scheduled-tasks/the-stock-analysis-model/`,
is a thin entry point that reads `daily-stock-analysis/SKILL.md` by absolute
path, so after a delete it finds no runbook and reports the run as skipped.
It must be a real folder holding a **copy**: the app refuses a symlinked task
file ("symlink detected before open; refusing to open").

```bash
mkdir -p ~/.claude/scheduled-tasks/the-stock-analysis-model
cp "$HOME/Projects/Workspace Folder/scheduled-tasks/the-stock-analysis-model/SKILL.md" \
   ~/.claude/scheduled-tasks/the-stock-analysis-model/SKILL.md
```

Then check in the app that the task is enabled for weekdays 17:00.

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
MACRO_ANTHROPIC_API_KEY=
ENV
chmod 600 "$HOME/Projects/Workspace Folder/.env"
```

Without `MACRO_ANTHROPIC_API_KEY` (or the older `ANTHROPIC_API_KEY`, still
read as a fallback) the run still succeeds — the Macro Outlook tab and
the per-sector outlook bullets just render empty, and the log says "macro
narrative skipped".

### 6. Reseed the price cache

**Do this before the first analysis run.** Step 0 of the daily routine only
refreshes tickers it already has (`ls output/prices/*.parquet`), so against an
empty directory it fetches the four benchmarks and nothing else — and the
report silently loses its price charts and Yesterday's Rating.

```bash
PYTHON="$HOME/.venvs/stock-model/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/download_prices.py --output-dir output/prices --universe us
```

Several hours for ~7–10k tickers at the 0.35 s inter-request delay. Without
`--refresh`/`--max-age-days` it runs in resume mode (existing files skipped),
so it is safe to interrupt and re-run until it completes.

### 7. Rebuild the DuckDB index from the archive

```bash
PYTHON="$HOME/.venvs/stock-model/bin/python"; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/ingest_snapshots.py --results-dir .claude/worktrees/snapshots-data --db output/snapshots.duckdb
```

`--db` matters: without it the store is written beside the snapshots, inside
the worktree, and the nightly readers (which open `output/snapshots.duckdb`)
never see it.

Idempotent. Until this runs, cross-run readers (rating history, yesterday's
rating, gate N/A deltas) fall back to parsing the JSON snapshots — correct,
just slower.

## Check it worked

```bash
cd "$HOME/Projects/Workspace Folder"
diff -q ~/.claude/scheduled-tasks/the-stock-analysis-model/SKILL.md \
  scheduled-tasks/the-stock-analysis-model/SKILL.md          # task copy current
git worktree list                                          # three worktrees
"$HOME/.venvs/stock-model/bin/python" -c "import yfinance, pandas, duckdb, certifi; print('deps ok')"
test -s .env && echo ".env present"
ls output/prices/*.parquet | wc -l                         # thousands, not 4
ls .claude/worktrees/snapshots-data/results_*.json* | wc -l # the corpus
```

Then a cheap end-to-end proof before trusting the nightly run — the default
S&P 500 + Dow universe instead of `us`, a few hundred tickers rather than
~9,100 (`daily-stock-analysis/SKILL.md` carries the measured runtime of the
full nightly run; don't rely on a second copy of that figure here):

```bash
PYTHON="$HOME/.venvs/stock-model/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Projects/Workspace Folder"; "$PYTHON" scripts/analyze_stock.py --prices-dir output/prices
```

If that produces `output/results_<today>.json` and the matching HTML, the
routine is back.

## Making the next delete cheaper

- `.env` belongs in a password manager, not only on disk. It is the one file
  in the tree that no amount of git can restore.
- `output/prices/` is the expensive rebuild. It is pure cache, but it is
  *hours* of cache — worth including in whatever backs up the Mac, or worth
  keeping outside the repo directory so a repo delete does not take it.
- The absolute paths in the task and the runbooks are what turn "deleted a
  folder" into "the routine is gone". Moving the repo has the same effect as
  deleting it; see this directory's `README.md`.

---

# Recovering the Supabase database

The database (design/supabase-migration.md) is a derived store. Every day is
also kept in two other copies, and the git archive is independent of Supabase, so losing
the database never loses data:

- **the git archive**: `results_<date>.json.gz` plus `blobs/` on the
  `data/snapshots` branch (run.sh step 06), which is canonical;
- **Storage**: the same snapshot as `json/results_<date>.json.gz` in the
  private `snapshots` bucket (step 06a), recorded with its SHA-256 in
  `core.snapshot_objects`. It is in the same Supabase project, so it guards
  against a bad database, not a lost project.

Every reader falls back to the JSON files when the database is unreachable
or stale, so the report and the analysis keep working during a restore.

## Is the database the primary store?

`DB_PRIMARY` in the cloud environment decides it.

| `DB_PRIMARY` | step 06a | a failed publish |
|---|---|---|
| unset / `0` | non-blocking; skipped without the Supabase secrets | listed in `SOFT_FAILURES`, run OK |
| `1` | blocking; missing secrets are a failure | run ends `RESULT FAILED at db-publish (DB_PRIMARY=1; archived)` and exits 1 |

In both cases the git archive (06) still runs after 06a, so the day is
archived either way.

**When to set it.** Step 07e (`scripts/db_night_check.py record`) records
each night in `core.night_checks`: whether 06a published, whether the run is
complete with the right row count and source SHA, and whether its rating
history matches `output/rating_history.json`. It appends the streak to the
status file:

```
DB_CUTOVER_STREAK 20/20 (ready: set DB_PRIMARY=1 to make 06a blocking)
```

The streak counts consecutive NYSE trading days with a green record. A night
with no record breaks it. Set `DB_PRIMARY=1` only once it reads 20/20;
nothing flips it automatically. To check at any time:

```bash
SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... python scripts/db_night_check.py status
```

**Stepping back.** Unset `DB_PRIMARY`. The nightly run carries on with the
database as a non-blocking copy, and the readers fall back to the files
wherever the database is behind. To take the readers off the database
entirely, set `SNAPSHOT_STORE_BACKEND=duckdb`; the run otherwise selects
`postgres` whenever the Supabase secrets are set.

## A night where 06a failed

1. **Read `logs/06a-db-publish.log`.**
   - `publish refused` is a soft check: a row drop over 30%, or too many cast failures. Fix the cause, or re-run with `--force --reason "..."`, which is audited in `core.runs.meta`.
   - `outcome still unknown` / `failed after a lost response` means the gateway lost the response. `publish_outcome` has already decided it; re-running is safe either way.
   - Any other message is a real error.
2. **Re-publish the day** from any machine with the secrets. It is idempotent and replaces the date in one transaction:
   ```bash
   python scripts/db_publish.py --run-date YYYY-MM-DD      # output/ or a staged archive file
   ```
3. **Record the night again**, so the streak sees it:
   ```bash
   python scripts/db_night_check.py record --date YYYY-MM-DD --publish-rc 0 --results-dir output
   ```
   (It needs `output/rating_history.json`; the cloud run stages it from the archive.)

## Restoring the database

In order of preference:

1. **Point-in-time recovery** (Pro plan, PITR enabled).
   - In the dashboard: Database → Backups → Point in Time. Pick a time before the incident. It restores the whole project in place, and the Data API and keys are unchanged.
   - Afterwards, re-publish any days after the restore point (step 2 of the previous section, per date).
   - Then run `db_night_check.py status`.
2. **Rebuild from the git archive**, when there is no PITR or the project is gone:
   ```bash
   supabase link --project-ref <ref> && supabase db push          # schema: every migration
   git clone --filter=blob:none --depth 1 --no-checkout --single-branch \
       -b data/snapshots https://github.com/danmcooper-ops/stock-analysis-model.git /tmp/snaparch
   SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... \
       python scripts/db_backfill.py --archive-git /tmp/snaparch --keep-files --work output/archive
   python scripts/db_parity_check.py --results-dir output/archive --dsn "$SUPABASE_DB_URL"
   ```
   - It takes about 25 s per archived day, mostly git fetching blobs, and re-running it is safe.
   - Storage lives in the same project: Supabase keeps object metadata in the project's Postgres, so a lost project loses the bucket too. That is why the git archive is the canonical copy. Re-running `db_publish.py` for a date re-uploads its objects and re-records the manifest.
3. **Rehearse before you need it.** `scripts/db_restore_drill.py` rebuilds into a scratch database beside the live one and compares every run, row, change point and latest pointer. It drops the scratch database afterwards and refuses the live database's name.
   ```bash
   python scripts/db_restore_drill.py --source storage --server-dsn "$SUPABASE_DB_URL"
   python scripts/db_restore_drill.py --source archive --archive-git /tmp/snaparch \
       --since 2026-09-14 --server-dsn "$SUPABASE_DB_URL"
   ```
   On the local stack (10 days, 25k rows, 2026-09-27):

   | source | fetch | publish | total | result |
   |---|---|---|---|---|
   | Storage `.json.gz` | 16 s | 75 s | 112 s | identical |
   | git archive | 87 s | 70 s | 178 s | identical |

   That is about 11 s and 18 s per day, so the whole 92-day archive takes 17 to 27 minutes.
