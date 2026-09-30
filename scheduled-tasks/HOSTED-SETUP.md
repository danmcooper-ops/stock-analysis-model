# Setting up the hosted Supabase project

Everything the Supabase migration needs is built. P0–P6 landed between
**2026-09-26** and **2026-09-27** (`design/supabase-migration.md`): the
migrations, the column registry and codec, the write path, the backfill, the
readers, the Parquet exports and Storage uploads, the nightly check, the
readiness gate and the restore drill. All of it was verified against the local
Supabase stack.

What does not exist yet is the **hosted project**. No migrations have been
applied to it, no night has been published, and the cutover gate has not
started counting. This file is the one-time setup that closes that gap.

Once the database is live, `RECOVERY.md` takes over: it owns `DB_PRIMARY`, the
cutover streak, a night where 06a failed, and restoring. Nothing here repeats
those.

## State of play (verified 2026-09-27)

| Thing | State |
|---|---|
| Migrations (13, `supabase/migrations/`) | Written and applied locally; **never pushed to the hosted project** |
| Supabase CLI | Not installed; the project has never been linked (no `supabase/.temp`) |
| Data API | Authenticates, but `pipeline` is not exposed — every RPC returns `406 PGRST106` |
| Storage | Authenticates; bucket list empty (buckets auto-create, so this is fine) |
| Local `.env` | Has `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`; `SNAPSHOT_STORE_BACKEND` commented out |
| Cloud Routine env | No `SUPABASE_*` at all — step 06a has been skipping silently |
| Cutover streak | Not started (0/20) |

Two naming notes, both deliberate for now:

- The code reads the legacy **`SUPABASE_SERVICE_ROLE_KEY`** (`data/db/storage.py`,
  `data/db/reader.py`, `data/db/publish.py`, `data/price_cache_store.py`,
  `data/sec_facts_cache_store.py`, plus the tests). The project issues
  new-format `sb_secret_…` keys, which authenticate the same way. Renaming to
  `SUPABASE_SECRET_KEY` is worth doing as its own change, not mixed into setup.
- Only `scripts/analyze_stock.py` parses `.env` (a hand-rolled key=value loop).
  `db_publish.py`, `db_backfill.py`, `db_parity_check.py` and `db_night_check.py`
  read `os.environ` directly, so export first:

```bash
set -a; . ./.env; set +a
```

## Stage A — Provision the project

### A1. Choose the plan before backfilling

P3 measured the 92-date archive at **885 MB** in Postgres. The free tier caps
the database at **500 MB**. So either move to Pro before Stage B, or backfill a
shorter window and let it grow. Timing a PITR restore (stability check 8, and
item 4 of the plan's "Still open, and yours") also needs Pro.

### A2. Push the migrations

Interactive login, so run it by hand:

```bash
brew install supabase/tap/supabase
supabase login
supabase link --project-ref <project-ref>     # the subdomain in SUPABASE_URL
supabase db push
```

The migrations are hosted-safe: role creation is guarded with `IF NOT EXISTS`
against `pg_roles`, and there is no `CREATE EXTENSION` or `ALTER SYSTEM`.

One may need a hand. `*_service_role_timeout.sql` runs:

```sql
ALTER ROLE service_role SET statement_timeout = '10min';
```

If hosted privileges refuse it, run that statement from the dashboard SQL
editor as `postgres`. **Do not skip it.** P5 found Supabase's 8 s
`authenticator` timeout — which `service_role` inherits — cancelling an
8k-ticker publish at about 13 s (SQLSTATE 57014). At today's 2.5k rows it fits
inside 8 s, so the failure only appears as the universe grows.

### A3. Expose the `pipeline` schema

Dashboard → Settings → API → Exposed schemas → add `pipeline`.

`supabase/config.toml` sets `schemas = ["pipeline"]` for the local stack, and
its comment claims the hosted project is set to match. **It is not** — that
comment was aspirational. Until this is done every RPC returns:

```
406 PGRST106  Invalid schema: pipeline
```

`core` and `internal` stay unexposed by design (R1).

### A4. Verify

```bash
set -a; . ./.env; set +a
python3 -c "from data.db.publish import transport_from_env; \
t, _, where = transport_from_env(); print(where, t.call('list_runs', {}))"
```

An empty list is the pass. A `406` means A3 is not done; a `404` on the
function means A2 is not done.

## Stage B — Backfill the archive

The runbook in `design/supabase-migration.md` under P3, with the hosted
specifics filled in.

### B1. Blob-less clone of the archive

As `run.sh` step 02 does:

```bash
git clone --filter=blob:none --depth 1 --no-checkout --single-branch \
  -b data/snapshots <repo> /tmp/snaparch
```

### B2. Publish every archived day

```bash
set -a; . ./.env; set +a
python scripts/db_backfill.py --archive-git /tmp/snaparch --keep-files --work output/archive
```

HTTPS only, so this runs from anywhere, including the cloud container. About
25 s a date — roughly 40 minutes for 92 dates — most of it git fetching each
day's blobs. Re-running is safe: `pipeline.list_runs()` skips dates already
published and unchanged. It stops at a refusal unless `--keep-going`.

### B3. Parity check — needs a login role that does not exist yet

`db_parity_check.py` needs a **direct** Postgres connection, so it needs a
login role and the session pooler DSN. `supabase/migrations/*_roles_rls.sql`
leaves that commented out on purpose (`pipeline_writer` and `pipeline_reader`
are NOLOGIN group roles). Create one in the SQL editor:

```sql
CREATE ROLE archive_reader LOGIN PASSWORD '<password>' IN ROLE pipeline_reader;
```

Then, with the pooler DSN for that role:

```bash
python scripts/db_parity_check.py --results-dir output/archive --dsn "$DSN"
```

**It must report 0 mismatches before Stage D.** This is stability checks 1–2:
every row rebuilds exactly, and re-scoring the rebuilt rows gives the same
rating, cap and composite score as the files.

## Stage C — Wire the nightly

Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to the **cloud Routine's**
environment. The local `.env` does not reach it — that is why step 06a has been
skipping (`cloud-daily-stock-analysis/run.sh`, `db_publish()`).

The same pair also switches on the price-parquet and companyfacts caches in
Storage (steps 02b/05e2, 02c/04b), so the run stops paying a cold cache. All
three buckets (`snapshots`, `price-cache`, `sec-facts-cache`) are created on
first use by `ensure_bucket()` — nothing to create by hand.

**Check the names took.** `ANTHROPIC_API_KEY` turned out to be reserved in that
container — it is the Claude Code session's own — which is why the macro key is
`MACRO_ANTHROPIC_API_KEY`. Confirm `SUPABASE_*` is not claimed the same way
before trusting that 06a picked it up. Read `logs/06a-db-publish.log` and the
step's line in `status.txt` on the first night.

Leave `DB_PRIMARY` unset. 06a stays non-blocking, and the git archive (step 06)
runs after it either way.

## Stage D — Switch the readers

Set `SNAPSHOT_STORE_BACKEND=postgres` — uncomment it in `.env`, add it to the
Routine env. It is commented out until Stage B passes for a reason: selection
is explicit (R12) and every failure degrades to the JSON files (R10), so
switching early looks like success while reading nothing.

## Stage E — The cutover gate

From here `RECOVERY.md` ("Is the database the primary store?") owns it: twenty
consecutive green trading nights on step 07e's `DB_CUTOVER_STREAK n/20`, then
`DB_PRIMARY=1` by hand. A night with no record breaks the streak, so this is at
least four working weeks from the first green night, and nothing compresses it.

## Independent: Cloudflare Pages (P4c)

Needs `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` and `CF_PAGES_PROJECT`,
plus the four `CF_ACCESS_*` variables for the login (`cloudflare/README.md`);
the one-time runbook is in `design/supabase-migration.md` under P4c. Step 08b
is non-blocking and GitHub Pages stays live throughout, so this can run in
parallel with everything above and blocks none of it.

**Nothing in Cloudflare exists yet** as of 2026-09-28: no Pages project, no
Zero Trust team, and none of those variables are set anywhere, so step 08b has
only ever logged "Cloudflare publish skipped". Because it is non-blocking, a
skip and a success look the same in `status.txt` — read
`logs/08b-publish-cloudflare.log` to tell them apart. Until this is done the
report is public on GitHub Pages, with no login at all.

## Limits to watch

- **`index.html` is no longer near Cloudflare Pages' 25 MiB per-file cap.** It
  peaked at 25.0 MiB; `report_html.pack_rows` shipping the inline `DATA` rows
  column-wise (2026-09-28) took it to 9.5 MiB, about 38% of the cap, and
  `scripts/check_pages_limits.py` has stopped warning on it. The per-ticker
  shard folders still grow with the universe, so the 20,000-file limit is the
  one left to watch.
- **A3 (≤ 4 GiB/yr) holds only at today's universe.** At 3.79 KB a row that is
  about 2.2 GiB/yr at 2.5k tickers but **7.1 GiB/yr at 8k**, which needs the
  retention step or rows about 45% narrower. Decide before P6 if the universe
  grows.
- **The newest migrations are dated `20260928`**, ahead of when they were
  written. A migration added with today's date would sort before them and apply
  out of order — number past them instead.
