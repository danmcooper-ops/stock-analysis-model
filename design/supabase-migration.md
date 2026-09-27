<!-- Build guide: work phase by phase (P0 first). Each phase has an exit criterion; do not start the next until it passes. -->

# Plan: Supabase as the primary, scalable database

## Context

Today the only SQL database is DuckDB. `data/snapshot_store.py` keeps `output/snapshots.duckdb`, a derived index of ~270 columns that are added on the fly as new keys appear. The canonical record is gzipped JSON on the `data/snapshots` git branch, and the website is static on GitHub Pages.

The goal is to make Supabase the primary database: Postgres for the data, Storage for the files. It has to hold up as data grows (the full US universe of ~8k tickers and 10+ years of history) and under heavy public read traffic. The git archive stays as the disaster-recovery copy.

Three workloads shape the design:
- **One nightly writer.** It loads each day in one validated, atomic step.
- **Many public readers.** They are served from CDN-cached static files and never touch Postgres.
- **Full-history analytics (backtests).** These run DuckDB over Parquet exports.

An independent adversarial review checked this plan against the code for stability and scalability. The 13 issues it raised are fixed below and marked **[R#]**.

P0 (2026-09-26, `design/p0/FINDINGS.md`) led to amendments **A1–A4**, which are folded in below and were accepted, with **Cloudflare** chosen to host the public files:
- **A1:** the cloud cannot open raw TCP to Postgres, so every cloud connection goes over HTTPS through Data API RPCs.
- **A2:** every stable scalar key gets a typed column (~370), not ~120.
- **A3:** the size check is ≤ 4 GiB/yr at today's universe, with a 3-year window in Postgres at 8k tickers.
- **A4:** the public payloads are split small and served from Cloudflare, not GitHub Pages.

## Target architecture

```
nightly run ─HTTPS─► pipeline.stage_chunk() ×N ─► pipeline.publish_run(load_id) ── one transaction (A1)
                                        ├─ core.results         (yearly partitions, typed + jsonb)
                                        ├─ core.rating_changes  (recomputed per affected ticker)
                                        ├─ core.latest_results  (only if D ≥ latest complete)
                                        └─ core.runs.status='complete', source_sha256
          └─► Storage: full .json.gz (private) + Parquet (analytics)
          └─► public payloads: latest.json, changes.json, history shards → Cloudflare (Pages + R2) (A4)

public traffic ─► CDN (static JSON)                 ✗ no anon access to Postgres
pipeline readers ─► core.* (pipeline_reader, timeouts, JSON fallback)
backtest ─► DuckDB over Parquet (union_by_name), JSON fallback per date
```

## Schema: fixed, versioned, changed only by migrations

There are two private schemas: `core` (the data) and `internal` (helper functions). A third schema, `pipeline`, is the only one the Data API exposes. It holds only the RPCs, and only `service_role` can execute them (A1). Nothing is reachable by `anon` or `authenticated` **[R1]**.

As built in P1 (`supabase/migrations/`), there are a few simplifications:
- `core.tickers` holds only `ticker_id`, `ticker`, `first_seen` and `last_seen`. Sector and the other attributes are per-row values in `core.results`.
- `core.latest_results` is a pointer `(ticker_id, run_date)` that gets joined to `core.results`.
- `core.screen_skip` stores `observed_on` and computes expiry in code, as it does today.

**Tables**
- `core.tickers`: `ticker_id identity pk`, `ticker unique`, `cik`, `name`, `sector`, `industry`, `exchange`, `first_seen`, `last_seen`.
- `core.runs`: `run_date pk`, `status` (`loading`|`complete`|`failed`), `risk_free_rate`, `risk_free_rate_source`, `macro_regime`, `n_rows`, `source_sha256` (SHA-256 of the **uncompressed canonical JSON** **[R2]**), `pipeline_version`, `meta jsonb`, `started_at`, `completed_at`.
- `core.results`:
  - Primary key `(run_date, ticker_id)`, `PARTITION BY RANGE (run_date)` with **yearly** partitions, pre-created 5 years ahead by a migration. There is no DEFAULT partition, and publishing fails fast if a date has no partition. No DDL runs at runtime **[R7]**.
  - About 370 typed columns, one per stable scalar key, taken from the registry (A2).
  - `extra jsonb` holds every other key.
  - `edgar_history_sha` points into `core.edgar_blobs`.
  - Indexes: the primary key, `(ticker_id, run_date DESC)`, and `(run_date, rating)`.
  - `extra` uses `lz4` compression **[R13]**.
- `core.edgar_blobs`: `sha pk`, `value jsonb` (lz4). Each distinct history is stored once, the same way the existing `BLOB_KEYS` files are, since about 99% don't change from day to day.
- `core.rating_changes`: primary key `(ticker_id, run_date)`, plus `rating` and `prev_rating`.
- `core.latest_results`: primary key `(ticker_id)`, the serving columns, and `run_date`.
- `core.screen_skip`: `ticker pk`, `kind`, `mcap`, `date`, `expires_at`.
- `core.snapshot_objects`: `run_date pk`, `json_path`, `sha256`, `bytes`, `n_rows`, `parquet_path`.

**Column registry (`data/db/columns.py`)**
- It is generated from the archived snapshots by `scripts/db_gen_registry.py` (A2):
  - every key whose values are always scalar and of one compatible type gets a typed column;
  - dicts, lists, mixed types, non-identifier keys, and keys not seen in the newest 10 snapshots (retired) stay in `extra`.
- The generator also writes the typed column block into the core migration.
- `scripts/db_gen_registry.py --check <snapshot>` flags new keys and keys that no longer fit their column. A key that stays in `extra` for 30 days becomes a candidate for promotion.
- Promoting a key is a new migration (`ALTER TABLE … ADD COLUMN`) plus a backfill that moves it out of `extra`.
- `tests/test_db_registry.py` checks three things:
  - every key a scorer or reader touches is registered;
  - the scalar reader keys are typed columns;
  - the migration matches the registry byte for byte.

**Codec (`data/db/codec.py`) [R8]**
- **Non-finite numbers.** Postgres `double precision` columns keep NaN and ±Infinity natively. Inside jsonb, which can't hold them, they are encoded as `{"$nf": "NaN"|"Inf"|"-Inf"}`.
- **Reserved keys.** A real dict whose keys start with `$` is escaped by wrapping it as `{"$lit": …}`.
- **Encoding.** The codec walks every nested value and serializes with `json.dumps(allow_nan=False)`, so a bare `NaN` can never reach psycopg.
- **NUL characters and lone surrogates**, which jsonb rejects, are stored as `{"$s": "<JSON string literal>"}`. A text column never receives them; such a value goes to `extra` instead.
- **Fidelity contract:** `.get()`-equivalence, with NaN equal to NaN and int equal to float. A NULL typed column is left out of the rebuilt row.
  - P1 found no `'k' in row` checks in the scorers or readers, so no presence bitmap is needed.
  - The only place that iterates a row's keys, `_purge_stale_gate_fields`, copes with missing keys.
- **Pinned quirks, each with its own test:** `"Infinity"` strings become floats, and jsonb normalizes `-0.0` to `0`.
- **Tests.** Hypothesis property tests round-trip adversarial values: `$nf`-shaped dicts, NUL characters, nested NaN, and very large ints.

- **Float precision (found in P1).** The Supabase image runs with `extra_float_digits = 0`, so Postgres prints doubles with only 15 significant digits. Values are stored exactly, but reading them rounds them: `0.03932028370017462` came back as `0.0393202837001746`.
  - Every direct connection goes through `data/db/connect.connect()`, which sets `extra_float_digits=3` and `connect_timeout=5`.
  - **P2 RPCs must do the same:** set `SET extra_float_digits = 3` as a function attribute, and never convert a `double precision` value through `numeric` or `to_jsonb()`. Both truncate to 15 digits. Instead, return the values as text, or ship the stored `extra` and the typed values as text in `jsonb`.
- **P1 check:** `scripts/db_fidelity_check.py` round-tripped the 10 newest snapshots (25,052 rows) through the migrated schema on the Supabase Postgres 17 image. There were 0 mismatches and 0 cast failures.

**What stays out of the DB.** The report-only narrative keys (`DEFAULT_EXCLUDE_KEYS`, `data/snapshot_store.py:92`) live only in the full `.json.gz` in private Storage. That file is the canonical copy of the complete row.

**Roles and timeouts [R10, R11]**

| Role | Access | Timeouts and connection settings |
|---|---|---|
| `service_role` | Executes the `pipeline` RPCs over HTTPS; has no direct grants on `core` (A1) | Each RPC sets its own `statement_timeout`/`lock_timeout` |
| `pipeline_writer` | Direct connections from dev, CI and admin work; writes to `core` | `lock_timeout=5s`, `statement_timeout=10min` |
| `pipeline_reader` | Read-only on `core` | `statement_timeout=30s`, `idle_in_transaction_session_timeout=60s` |

`pipeline_writer` and `pipeline_reader` are NOLOGIN groups. Postgres does not inherit role settings from a group, so the timeouts go on each login user, and those users are created by hand per project:

```sql
CREATE ROLE nightly_writer LOGIN PASSWORD '...' IN ROLE pipeline_writer;
ALTER ROLE nightly_writer SET lock_timeout = '5s';
ALTER ROLE nightly_writer SET statement_timeout = '10min';
```

- `anon` and `authenticated` have no grants on `core`, and RLS is on for every table.
- Every Python connection sets `connect_timeout=5`.
- Anything that uses port 6543 sets `prepare_threshold=None`.

## Write path: one writer, validated, atomic, re-runnable

1. **Where it runs.** `scripts/db_publish.py --run-date D` is the blocking step `06a-db-publish` in `scheduled-tasks/cloud-daily-stock-analysis/run.sh`. It runs after `05f` rescore and before the git archive.
2. **Staged over HTTPS, applied in one transaction (A1, R6).** The cloud can't open raw TCP to Postgres, so:
   - `pipeline.stage_chunk(load_id, chunk_no, rows jsonb)` appends chunks of about 1 MB (rows already split with the registry and codec) to `internal.load_rows`, keyed by `load_id`. It is idempotent per `(load_id, chunk_no)`, so a retried request can't double-load. Uploads go through `data/throttle.py`.
   - `pipeline.publish_run(load_id, D, sha256, expectations)` takes `pg_advisory_xact_lock` and does everything below in one transaction.
   - Staged rows are keyed by `load_id`, so two publishers can never see or truncate each other's rows. Stale loads are purged after a day.
   - From a dev machine or CI, `db_publish.py --direct` does the same thing over `psycopg`, using a `TEMP` table and `COPY`.
3. **`publish_run` validates first [R9].** It raises and changes nothing if any check fails:
   - the row count matches the snapshot;
   - there are no duplicate tickers;
   - required columns have fewer nulls than their threshold;
   - at most 1% of values in registry columns fail their type cast. A key that isn't in the registry and lands in `extra` never counts against this.

   `--force --reason "…"` overrides a failed check. It is audited in `runs.meta`, and it is how a methodology change or the first backfill date gets through. A drift in the rating distribution only raises a warning.
4. **Then it applies the day:**
   - Upsert `core.tickers` and `core.edgar_blobs` (`ON CONFLICT DO NOTHING`).
   - `DELETE` + `INSERT` for date D. MVCC means readers never see a half-written day, and there is no partition swap or TRUNCATE **[R5]**.
   - `ANALYZE` the affected partition **[R7]**.
5. **Rating changes, correct for out-of-order dates [R4].** For each ticker whose rating at D changed or appeared, delete its `rating_changes` from D onward and recompute from `core.results`.
   - The recompute starts at the last change-point before D, over complete runs only.
   - Rows whose rating is null or empty are skipped. This matches today's lag-based `rating_history()`, where a gap with the same rating is not a change.
   - Backfill instead bulk-loads every date first, then builds `rating_changes` in one set-based window pass.
6. **Latest results [R5].** `latest_results` is replaced with `DELETE` + `INSERT` only if D is on or after the latest complete date. Re-publishing an older date never overwrites "latest". Finally, `runs.status` is set to `complete`.
7. **After the commit:**
   - Upload the `.json.gz` and a Parquet export to Storage and check their SHA-256.
   - Fill `core.snapshot_objects`.
   - Write the **public payloads** (step below).

   If this fails, the step fails but the DB stays consistent, and re-running `06a` repairs it.
8. **Rewrites outside the nightly run stay in sync [R2].** `sync_snapshot_file()` (called by `analyze_stock`, the `enrich_*` scripts and `rescore_and_render`) works like this on the Postgres backend:
   - **During the nightly run** (`DB_DEFER_PUBLISH=1`, set by `run.sh`): it defers, and `06a` publishes once.
   - **Otherwise** (a manual rescore or a repair): it calls `db_publish` for that date.

   As a backstop, a reader that has the local file uses a date only if `runs.source_sha256` matches it, and falls back to the JSON otherwise.
9. **Post-publish check.** `scripts/check_snapshot_store.py` confirms the status, `n_rows`, the SHA and the manifest, and it blocks the run.
10. **Screen-skip cache.** `data/screen_skip_cache.py` upserts `core.screen_skip` in batches. The JSON file and its git write-back stay as the fallback.

## Read paths

**Pipeline readers (`data/db/reader.py`)** implement the same interface and return shapes as the `SnapshotStore` readers:
- `rows`
- `last_known_rows`: complete dates only, one row per ticker with a non-empty rating, and unknown columns returned as `NULL`
- `rating_history`: served from `core.rating_changes`
- `run_meta`
- `dates` / `has_date`: complete runs only **[R12]**

Backend selection **[R12]**:
- Postgres is used only when `SNAPSHOT_STORE_BACKEND=postgres` is set explicitly. A DSN alone is not enough.
- Under pytest, a prod DSN is refused.
- The first connection failure is cached for the rest of the process, so an outage costs about 5 s once, not once per call site **[R10]**.
- On failure the reader returns None, and each caller's existing JSON fallback runs unchanged. Those call sites are `analyze_stock.py:2849-2893`, `report_html.py:247-297`, `track_portfolio.py:50`, `gate_na_report.py:66` and `query_results.py:252`.

The rating-history check in `report_html.py:297` changes for Postgres **[R3]**. Today it requires the store's dates to equal the file dates exactly. For Postgres, it passes when the store holds every file date with no gap after the oldest one, and Postgres is treated as the authority. `rating_history.json` keeps being committed until parity has held for 20 runs in a row.

**Analytics.**
- `scripts/backtest.py:load_corpus` reads Parquet through DuckDB with `read_parquet(…, union_by_name=true)`, the same pattern as `data/price_store.py`.
- A date without a Parquet file falls back to its JSON **[R13]**.
- Postgres is never scanned for a backtest.

**Public traffic [R1].**
- Right after publishing, `scripts/export_public.py` writes these static files:
  - `latest.json`: all ~8k rows, so the PostgREST 1,000-row cap never comes into play;
  - `changes_<window>.json`;
  - per-ticker history shards, the same shard pattern as the existing `vol/` and `px/` folders.
- The files go to **Cloudflare Pages** (A4), which has no soft bandwidth cap like GitHub Pages' 100 GB/month. P4c chose Pages alone, with no R2: once the two oversized sidecars were split, every file fits Pages' 25 MiB limit. R2 stays the option if a file ever outgrows it. The P5 load test targets that host.
- The site fetches only those files.
- There is no anon access to PostgREST and no Edge Function in the hot path. If a live query is ever needed, it will be a fixed-parameter RPC behind a CDN that I've confirmed caches it. P0 checks that with the `cf-cache-status` and `Age` headers.

## Operations

- **Environments:** local (`supabase start`), `staging` (a Supabase project that is also the load-test target), and `prod` (Pro plan with point-in-time recovery, PITR).
- **Migrations:**
  - live in `supabase/migrations/`;
  - CI's `db` job applies them with `supabase db start` (the Supabase Postgres 17 image, which includes its standard roles) and runs the `pg` tests;
  - `squawk` lints them for table locks;
  - The repo's Supabase GitHub integration builds a preview branch per PR and deploys to the linked project `llvjffhwuivwsusovjfg`, which is the **sandbox/staging** project, on merge. Prod will be a separate project, deployed by manual dispatch.
  - The hosted projects must expose only the `pipeline` schema to the Data API. This is set in the dashboard and matches `[api] schemas` in `supabase/config.toml`.
- **Retention.** Old yearly partitions are already exported to Parquet and can be removed with `DETACH … CONCURRENTLY`, outside a transaction. Backfill builds a standalone table with a CHECK constraint and then `ATTACH`es it **[R7]**.
- **Observability:**
  - `pg_stat_statements`;
  - a slow-query log for anything over 200 ms;
  - `core.runs` timings added to `provenance.timings`;
  - `daily-eod.yml` alerts if `runs.status` isn't `complete` by 07:00 ET.
- **Secrets** go in the cloud environment:
  - `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`: the pipeline RPCs and Storage over HTTPS (A1);
  - `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` and `CF_PAGES_PROJECT` (plus an optional `CF_PAGES_URL`), for the Cloudflare Pages deploy (A4, P4c).

  `SUPABASE_DB_URL`, the session-pooler DSN for a `pipeline_writer` login, is used only on dev machines and in admin work.
- **Dependencies.** `psycopg[binary]~=3.3` is in `pyproject.toml` and `requirements.txt`. `duckdb` stays.

## Phases and the check that closes each

- **P0: Spike.**
  - Connect from the cloud Routine to `*.pooler.supabase.com:5432`. Its egress goes through an HTTPS proxy, so the host may need allowing in the network settings. If TCP can't get through at all, `06a` falls back to an HTTPS ingest function.
  - Measure bytes per row on 10 archived snapshots.
  - Confirm the CDN caching behaviour.
  - *Passes when:* the connection works, and the projected size at **8k tickers** is ≤ 8 GB/yr **[R13]**.
  - **Ran 2026-09-26; see `design/p0/FINDINGS.md`.**
    - **Connection:** raw TCP from the cloud is blocked by the proxy. Publishing and cloud reads move to HTTPS (Data API RPCs).
    - **Size:** 2.8 GiB/yr at today's ~2.5k rows/day and 8.9 GiB/yr at 8k, once every scalar key is typed.
    - **CDN:** Pages caches, but its bandwidth cap is the real traffic limit.
    - Amendments A1–A4 in that file override this plan's direct-connection write path, its ~120-column registry, the pass line above, and hosting public payloads on Pages.
  - **Passed with amendments A1–A4** (accepted 2026-09-26; Cloudflare chosen for A4). The size check is now the A3 line: ≤ 4 GiB/yr at today's universe (measured 2.8).
- **P1: Migrations, registry, codec and roles.**
  - *Passes when:* CI applies the migrations cleanly, `squawk` reports nothing, and the codec property tests and registry coverage tests pass.
- **P2: Write path.**
  - *Passes when:* stability checks 3–6 pass.
  - **Built 2026-09-26:**
    - `supabase/migrations/*_publish_rpc.sql` adds `pipeline.stage_chunk` and `pipeline.publish_run`.
    - `data/db/publish.py` provides the REST and direct transports.
    - `scripts/db_publish.py` is the CLI.
    - `run.sh` step `06a-db-publish` is non-blocking. It is skipped when `SUPABASE_URL` is unset and runs as a dry run for SMOKE/DRY_RUN.
  - **Results:**
    - A real day (2.5k rows, about 45 chunks) publishes in about 5 s direct and about 11 s over the Data API. Most of that is the 0.2 s request throttle.
    - The first version took 190 s. Its `(jsonb_populate_record(...)).*` called the function once per column, and `CROSS JOIN LATERAL` fixed it.
    - `db_fidelity_check.py --published` finds 0 mismatches on REST-published days.
    - `core.rating_changes` equals a full recompute after publishing in order, publishing out of order, and repairs.
    - `anon` is refused at the gateway ("permission denied for schema pipeline"), and `core` is not exposed.
  - **Stability checks:**
    - **3:** a publisher killed mid-transaction, concurrent publishers, and idempotent republish all pass (`tests/test_db_publish_pg.py`).
    - **4:** out-of-order publishing and repair pass.
    - **5:** row-drop refusal, `--force` with its audit, structural errors, and cast-failure gating all pass.
    - **6:** a connect timeout on a silent server passes, and so does REST retrying only idempotent calls. The reader-side SHA fallback arrives with the readers in P4.
  - **Moved to P4,** alongside the readers and export they serve:
    - Storage and Parquet uploads, and `core.snapshot_objects`;
    - `sync_snapshot_file` deferring or publishing (R2);
    - `check_snapshot_store.py` becoming the post-publish check;
    - the `core.screen_skip` upsert.
- **P3: Backfill the whole archive.**
  - *Passes when:* fidelity and decision parity (stability checks 1–2) hold for every date.
  - **Built 2026-09-26:**
    - **`scripts/db_backfill.py`** publishes the archive in date order through the nightly path. It is resumable: `pipeline.list_runs()` (a new RPC) lets it skip dates that are already published and unchanged, which also works over HTTPS. It stops at a refusal unless `--keep-going` is given, and `--force-reason` is audited for each date.
    - **`scripts/db_parity_check.py`** runs stability checks 1–2 for each date, against a direct connection.
  - **Local run** against the Supabase Postgres 17 image:
    - All 92 archived dates (2026-04-20 → 2026-09-25), 195,132 rows. None were refused and none needed `--force`.
    - **Parity: passed.**
      - 0 fidelity mismatches: every row rebuilds exactly, and each run's status, `source_sha256`, risk-free rate and meta match.
      - 0 decision mismatches: re-scoring the rebuilt rows and the file rows with today's `score_and_rate` gives the same rating, raw rating, cap and composite score for every ticker.
      - The rating change points match exactly: 13,943 from the files and 13,943 in the database.
      - A deliberately corrupted input (one ticker's `mos`) was caught as a composite-score mismatch, so the check is not vacuous.
    - **Size:** 885 MB in the database, of which `core.results` is 730 MB (3.9 KB per row with indexes) and edgar blobs are 58 MB (19,931 distinct). That projects to about 2.6 GiB/yr at ~2.5k rows/day, inside the A3 limit.
  - **Performance fix** (migration `*_publish_run_perf.sql`): during the backfill, one day's publish grew from 4.5 s to 15.8 s.
    - ANALYZE of all ~370 columns took 8.5 s. It now analyzes only `run_date`, `ticker_id` and `rating`, and autoanalyze covers the rest.
    - The rating recompute anchored on each ticker's last change point, so a stable ticker rescanned every day since then. It now anchors on the previous rated day, with one index probe.
    - A republish of the newest date now takes 4.6 s in total, with the same change points. The P5 volume test re-checks this at 20M rows.
  - **Hosted backfill runbook** (sandbox first, then prod once it exists):
    1. Apply the migrations. Merging applies them to the sandbox through the GitHub integration; `supabase db push` applies them elsewhere.
    2. Make a blob-less clone of the archive, like `run.sh` step 02:
       `git clone --filter=blob:none --depth 1 --no-checkout --single-branch -b data/snapshots <repo> /tmp/snaparch`
    3. From any machine with `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` set (HTTPS only, so the cloud container works):
       `python scripts/db_backfill.py --archive-git /tmp/snaparch --keep-files --work output/archive`
       It takes about 25 s a date, mostly git fetching each day's blobs. Re-running it is safe.
    4. From a machine that can reach Postgres directly (the session pooler DSN of a `pipeline_reader` login):
       `python scripts/db_parity_check.py --results-dir output/archive --dsn "$DSN"`
       It must report 0 mismatches before P4 switches any reader to the database.
- **P4: Switch readers, Parquet backtest, public export.**
  - *Passes when:* the parity checks pass and the site renders from the exported files.
  - P4 is split into three PRs so each can be reviewed and verified on its own:
    - **P4a:** the reader switch (below).
    - **P4b:** Parquet exports and Storage uploads, `core.snapshot_objects`, backtests on Parquet, and the `core.screen_skip` upsert.
    - **P4c:** the public export to Cloudflare, which needs Cloudflare credentials.
  - **P4a built 2026-09-27:**
    - **Migration `*_read_rpcs.sql`:** read RPCs `pipeline.read_rows`, `last_known_rows`, `rating_history` and `run_meta`, callable by service_role and the pipeline roles.
      - They build jsonb only from the requested columns. The first version rendered all ~370 columns and filtered afterwards; that took 1.5 s for a 3-column read and 15 s for `last_known_rows`, now about 0.05 s and 0.8 s on a real day.
      - They set `extra_float_digits = 3`, because jsonb renders float8 through its output function and the Supabase default of 0 truncates the last two digits.
    - **`data/db/reader.DbStore`** provides the reader half of `SnapshotStore`. `SnapshotStore.for_results_dir()` returns it when `SNAPSHOT_STORE_BACKEND=postgres` is set and the database is reachable, so every existing call site switches unchanged:
      - carry-forward and lost-SEC-tickers in `analyze_stock`;
      - previous ratings and rating history in `report_html`;
      - `track_portfolio`, `gate_na_report` and `portfolios`.
    - **Still on DuckDB:** `query_results` (raw SQL) and `backtest` (moving to Parquet in P4b) pass `allow_db=False`.
    - **Transport:** the Data API when `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set, otherwise `SUPABASE_READER_URL`/`SUPABASE_DB_URL` direct.
    - **Failure handling:** the first failure disables the backend for the process (R10). Under pytest only a local host is accepted, and that is checked before connecting (R12).
    - **Staleness (R2):** a date is served only if its run is complete and any local plain `results_<date>.json` hashes to the published `source_sha256`. `sync_snapshot_file()` republishes a rewritten snapshot when the backend is selected; the nightly run sets `DB_DEFER_PUBLISH=1` and publishes once, at 06a.
    - **R3:** `report_html`'s rating-history check accepts a database whose dates cover the staged files, rather than requiring an exact match.
    - **Post-publish check:** `check_snapshot_store.py --database` confirms the run is complete, the row count, and `source_sha256`. It runs as `run.sh` step `07e-db-check`, non-blocking.
  - **P4a results:**
    - The 10 newest real snapshots (25k rows) were read through DuckDB and the database, both direct and over the local Data API, and every reader matched: dates, `rows` for five column sets (including unknown and nested keys), `last_known_rows` over three windows including the fallback look-back, `rating_history` (4,538 points), and `run_meta`.
    - `tests/test_db_reader_pg.py` (16 tests) repeats this on synthetic data with NaN, ±Inf, NUL, gaps and a ticker that drops out, and runs the real call sites with the backend selected.
    - One deliberate difference: the database keeps a NaN nested inside a dict or list as the file has it, where the DuckDB store's JSON columns turn it into null.
  - **P4b built 2026-09-27:**
    - **Parquet exports** (`data/db/parquet.py`): one `results_<date>.parquet` per run, with the registry's typed columns, `extra` as codec JSON text, the projected `edgar_history`, and the snapshot metadata in the file's key-value metadata. A day is about 4–5 MB, against about 90 MB of JSON.
      - `backtest.load_corpus` prefers a date's Parquet export, then the DuckDB store, then the JSON, deciding per date. The database is never read for a backtest.
      - `db_publish.py` writes the export after every publish, and `export_dir()` backfills it from an archive.
    - **Storage** (`data/db/storage.py`): after a publish, `db_publish.py` uploads `json/results_<date>.json.gz` and `parquet/results_<date>.parquet` to the private `snapshots` bucket.
      - The `.json.gz` is the canonical snapshot, gzipped deterministically, so it is the full row including the report-only keys.
      - Each upload is verified by downloading it back and comparing SHA-256, then recorded through `pipeline.record_snapshot_objects` (migration `*_objects_screen_skip_rpcs.sql`).
      - The bucket is created on first use. `--no-upload` skips the upload.
    - **Screen-skip cache:** with the database backend selected, `data/screen_skip_cache.py` merges `core.screen_skip` into what it loads (the newer observation of a ticker wins) and replaces the database copy on every save, through `pipeline.screen_skip_load/replace`. The file and its git write-back stay as the fallback.
    - **A pre-existing bug found and fixed:** the `edgar_history` projection shared by the DuckDB store and the Parquet exports (`DEFAULT_PROJECTIONS`) kept only `years_available` and `operating_income_history`.
      - Since 2026-09-16, the debt-free Int Coverage rule (`scoring._is_debt_free`) also reads seven more series: `total_debt_history`, `debt_current_history`, `debt_noncurrent_history`, `total_assets_history`, `revenue_history`, `earnings_history` and `operating_cf_history`.
      - So re-scoring through the store differed from the JSON for snapshots that need that rule. On 2026-09-14 to 16 there were 78–82 differing decision values per day.
      - The projection now keeps those series. The store's `SCHEMA_VERSION` goes to 5, so stale stores are ignored and rebuilt. A new test records every `edgar_history` key scoring reads and fails if the projection drops one.
  - **P4b results:**
    - The 10 newest real snapshots exported and read back from Parquet: 0 rows differ from the JSON, and 0 re-scored decisions differ (rating, raw rating, cap, composite) on every date.
    - Publishing 2026-09-24 and 09-25 through the local Data API and Storage uploaded 29 MB and 22 MB `.json.gz` objects. Each decompresses to exactly the published `source_sha256`, and `core.snapshot_objects` holds both.
    - The anon key cannot download from the private bucket.
  - **P4c built 2026-09-27: the site on Cloudflare Pages.**
    - **Why files had to change.** Cloudflare Pages (free plan) refuses any file of 25 MiB or more, and more than 20,000 files per deploy. Two sidecars were over the per-file limit on the 2026-09-25 report:
      - `hist.json`, 28.8 MB. All five consumers loaded the whole file to read one ticker.
      - `details.json`, 30.4 MiB. This one was not in the plan; the new size check caught it on the first real render.
    - **`hist/` shards.** `report_html` now writes `hist/<TICKER>.json`, one per ticker (~2.2k files, ~13 KB each), and `hist_index.json` (`{"tickers": [...]}`). The ticker list is also inlined in the page, so it knows which shards exist without an extra request.
      - The template's `_ensureHist(tickers, onDone)` is modelled on `_ensurePx`. It fetches only the shards a view needs, shares in-flight loads, and records a failed load as `null` so nothing retries in a loop.
      - The five consumers use it: the price-history chart (for its charted tickers), the popup chart, Track Record, the statements tabs and the PDF export.
      - The page no longer downloads 29 MB to show one company's fundamentals.
    - **`details/` parts.** The popup's heavy text fields are merged into every row after first paint, and some views may read them across rows. So the behaviour is kept and only the file is split: numbered parts of about 8 MiB each (4 on 2026-09-25), listed in `details_index.json`. All parts load in parallel and are merged exactly as the single file was.
    - **Publishing.** `publish_vol_shards.py` copies `hist/` and `details/` by manifest, like `vol/` and `px/`, and prunes and verifies each destination. `run.sh` step 08 and `run_daily.sh` copy `hist_index.json` and `details_index.json` in place of the two monoliths.
    - **`scripts/check_pages_limits.py`** fails a deploy directory with a file of 25 MiB or more, or more than 20,000 files. It warns at 80% of either limit. `index.html` is at 97% (24.2 MiB on 2026-09-25), so the warning is already firing: splitting the inline `DATA` blob is the next file to plan.
    - **Step `08b-publish-cloudflare`** (non-blocking) runs after the GitHub Pages push, on the same `docs/`. GitHub Pages stays live during the switch.
      - It skips cleanly unless `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` and `CF_PAGES_PROJECT` are set, and under SMOKE or DRY_RUN.
      - Otherwise it runs the size check, then `npx wrangler@4.141.0 pages deploy` (pinned, overridable with `WRANGLER_VERSION`), then polls the live URL (`CF_PAGES_URL`, default `https://$CF_PAGES_PROJECT.pages.dev/`) until it shows the run date.
    - **Headers.** `pages_headers` becomes `docs/_headers` and adds `nosniff` and a referrer policy. Cache rules are deliberately left at Pages' default (`max-age=0, must-revalidate` with an ETag). A `px/` shard is an offset into `prices_meta.json`'s dates axis, so yesterday's shard cached next to today's axis would draw a series shifted by a day. Revalidating is a 304 from the edge.
  - **P4c results:**
    - The 2026-09-25 snapshot (2,531 rows) rendered and built into `docs/` the way step 08 does: 2,280 files, 83.7 MiB, largest file `index.html` at 24.2 MiB. The size check passes, with the `index.html` warning.
    - That `docs/` was served by `wrangler pages dev` (the local Cloudflare Pages emulator) and driven in headless Chromium. The test opened a popup's Track Record, statements tab and fundamentals chart, a company with no shard, and the price-history chart with two tickers on Revenue.
      - Only `hist/A.json`, `hist/AAMI.json` and `hist/AAON.json` were fetched, never `hist.json`.
      - Each loaded value equals the old `hist.json` payload.
      - All four `details/` parts loaded and merged into every row.
      - There were no console errors.
      - The emulator served the `_headers` rules.
    - **Not yet done:** a real deploy. Nothing in Cloudflare exists yet; the setup runbook below lists what's needed.
  - **Cloudflare Pages setup runbook** (one time):
    1. In the Cloudflare dashboard, go to Workers & Pages → Create → Pages → **Direct Upload**. Name the project (for example `stock-analysis`); the name becomes `CF_PAGES_PROJECT`. Don't connect Git: the routine uploads the built `docs/`.
    2. Create an API token (My Profile → API Tokens → Create Token → Custom) with the single permission **Account → Cloudflare Pages → Edit**, scoped to that account. Note the account ID from the dashboard sidebar.
    3. Add `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` and `CF_PAGES_PROJECT` to the cloud environment that runs the nightly routine. Add `CF_PAGES_URL` if a custom domain is attached.
    4. Put the site behind the login before the first deploy: `cloudflare/README.md` (Cloudflare Access, plus `CF_ACCESS_TEAM_DOMAIN`/`CF_ACCESS_AUD` and a service token). Step 08b refuses to deploy without them.
    5. The next nightly run deploys. Check `logs/08b-publish-cloudflare.log` and the `status.txt` line for step 08b, then open the `*.pages.dev` URL.
    6. Optional: attach a custom domain under the project's Custom domains tab, then set `CF_PAGES_URL` to it.
    7. **Retiring GitHub Pages later:** after some green nights on both, make 08b blocking and drop the `pages-live` push and live check from step 08. Keep the `docs/` build. Then disable Pages in the repository settings and delete `.github/workflows/deploy-pages.yml` and the `pages-live` branch.
- **P5: Scale tests.**
  - *Passes when:* every scalability target is met.
  - **Run 2026-09-27, scaled down locally** (decision: build the tooling, measure what fits, extrapolate). The plan's 20M rows need about 75 GB; the container had 21 GB. So the run used the full ticker count, 8,000, over fewer days. Per-night costs depend on the tickers; history-dependent costs were measured at two sizes to get their slope:
    - half: 188 days, 1.5M rows;
    - full: 375 days, 3.06M rows, crossing two yearly partitions.
    - Everything ran against the local Supabase stack (Postgres 17, PostgREST, Kong), with 4 cores. For a cold cache, Postgres was restarted and the OS page cache dropped before each benchmark.
  - **Tooling** (`tests/load/`, see its README):
    - `scale.py` clones real snapshot rows into synthetic tickers and days (templates from 2026-09-25). Each ticker's rating walks through runs of 20–80 days, which gives about 2% changes a night, as the real data does. It then runs the checks: publish, EXPLAIN pruning, size, cold-cache latency, publish under load, and anon refusal. `compare` prints the growth table.
    - `k6_public.js` is the CDN load test.
    - `tests/test_db_scale.py` pins the generator's rating walk and change points against publish_run's rule, and runs the whole harness at 12 tickers × 30 days in CI's `db` job.
  - **Found and fixed:**
    1. **The Data API cancelled the nightly publish.** Supabase gives `authenticator` an 8 s `statement_timeout`, and `service_role` inherits it. publish_run is one transaction by design; at 2.5k rows it fit inside 8 s, and at 8k it took about 13 s and was cancelled (SQLSTATE 57014). Migration `*_service_role_timeout.sql` gives `service_role` 10 minutes. anon and authenticated keep 3 s and 8 s.
    2. **publish_run's cost grew with history.** The rating recompute's "one index probe" anchor joined `core.runs` inside its `ORDER BY … LIMIT 1`. The planner turned that into a scan of each ticker's whole history (about 3M heap blocks), which was 23.5 s of a 33 s publish, linear in days: about 160 s at 10 years.
       - Migration `*_publish_run_anchor_probe.sql` excludes the few non-complete runs by date, so the ordered Append stops at the first row: 85 ms for 8k tickers. The fallback for a ticker that left a republished day gets the same fix.
       - A new pg test covers a failed day in the history.
       - At 3M rows, publish_run now takes about 6 s and the whole publish 16–18 s. The old function took 38 s at 3M rows.
    3. **A committed publish could be reported as failed.** Under load at 3M rows, Kong's 60 s proxy timeout returned 504 while Postgres went on and committed the day, so run.sh 06a would have failed a good night. Fix 2 removes the long publish, and the lost response is also handled:
       - The new RPC `pipeline.publish_outcome(load_id, run_date)` (migration `*_publish_outcome_rpc.sql`) answers `published`, `running` (publisher lock held), `failed` (chunks still staged, lock free) or `unknown`.
       - `RestTransport` marks a 502/503/504 or a dropped connection as `PublishOutcomeUnknown`, and `publish()` polls `publish_outcome` for up to 15 minutes before deciding.
    4. **"Rating changes since a date" scanned every change point.** It was 10 ms at 46k change points, linear, and heading past 50 ms at 20M rows. Migration `*_rating_changes_date_idx.sql` adds an index on `(run_date)` covering `(ticker_id, rating, prev_rating)`, so the read is an index-only scan.
    5. **Ticker history was one cold heap page per day.** p95 was 12.7 ms at 189 rows and 25.5 ms at 380, which failed the 20 ms target at full scale and projected about 85 ms at the 1,260 rows of 5 real years.
       - Migration `*_ticker_history.sql` replaces the `(ticker_id, run_date DESC)` index with one that also carries rating, mos, price, dcf_fv and `_composite_score`, about 32 bytes more per row. It adds `pipeline.ticker_history(ticker, from, to)`, which requires both bounds.
       - Now 6.0 ms p95 at 382 rows. Measured cold across window sizes, it costs 3.4 ms plus 0.0065 ms per row, about 11 ms at 1,260 rows.
       - The same index makes publish_run's anchor probe index-only.
    6. **An open-ended history range touched the empty future partitions.** `run_date >= x` cannot prune the partitions created ahead of time. That is harmless while they are empty, but every history read is now bounded on both sides, which `ticker_history` enforces.
  - **Results at 3.06M rows, after the fixes:**

    | check | target | measured | at 20M rows (projected) |
    |---|---|---|---|
    | publish 8k rows, Data API, with change points | < 5 min | 16.4 s (20.4 s under load) | about the same: the anchor probe reads the newest partition only |
    | partition pruning, every reader query | yes | yes, with history index-only | same |
    | ticker history 5y, p95 cold | < 20 ms | 6.0 ms (382 rows) | about 11 ms (1,260 rows) |
    | `last_known_rows`, 8k tickers, 20 columns | < 2 s | 431 ms | same: reads 7 days |
    | rating changes since a date | < 50 ms | 6.9 ms | same for a fixed window |
    | export (read the day back and write Parquet) | < 60 s | 15.7 s | same: one day |
    | publish during load: no partial day | none | none; counts seen were only 0 and 8,000, raw and through `read_rows`, over a publish and a republish | same |
    | readers' p95 during a publish | within target | history 10.3 ms, changes 9.5 ms, `last_known_rows` 567 ms | same |
    | unauthenticated Data API calls | refused | 401/404 for `core` tables and the pipeline RPCs | same |

  - **Growth** (half to full, same reader schema; `scale.py compare`): rows ×2.02, bytes per row ×1.00, `last_known_rows` ×1.02, export ×0.96, rating changes since ×1.39 (5.9 to 8.1 ms, both with its index). Ticker history grew with the rows returned (×2.0), which is what fix 5 addresses. `rating_history()` over all change points is the one read that grows with history by design: 208 ms at 82k change points, about 1.3 s at 20M rows. The nightly render calls it once.
  - **Size:** 3.79 KB per row with indexes. That is 7.1 GiB/yr at 8,000 tickers and about 71 GiB at 20M rows. **A3 (≤ 4 GiB/yr) holds only at today's universe** (2.5k tickers, about 2.2 GiB/yr). At 8k tickers it needs either the retention step (move partitions older than N years to Parquet in Storage, which the plan already describes) or rows about 45% narrower. This is a decision for before P6, if the universe grows.
  - **Not measured here:**
    - **The 20M-row run on a hosted project.** It needs a staging project with about 80 GB of disk; the free tier caps the database at 500 MB. The command is in `tests/load/README.md`.
    - **k6 through Cloudflare** (check 3). It needs the P4c deploy. The script ran against `wrangler pages dev`: 2,502 requests, 0 errors, and its latency and cache thresholds fire correctly (the emulator has no CDN cache). Nothing in that path reaches Postgres, because the site is static files; the plan's `pg_stat_statements` confirmation belongs to the hosted run.
    - **Hosted gateway timeouts.** They may differ from the local stack's 60 s, and fix 3 covers either way.
- **P6: Cutover.**
  - `06a` becomes blocking, and `RECOVERY.md` and `CLAUDE.md` are updated.
  - *Passes when:* 20 nightly runs in a row publish green with rating-history parity, and the restore drill passes.
  - **Built 2026-09-27 as a gated cutover** (decision: the hosted project had not yet received a nightly publish, so the 20 nights had not started). Everything the cutover needs is in place; the flip is one environment variable, set by hand once the gate reads 20/20.
    - **The nightly check** (`scripts/db_night_check.py record`) replaces step 07e's store check. It checks three things:
      - the run is complete, with the right row count and source SHA;
      - rating-history parity, against the JSON cache the report keeps (`output/rating_history.json`), as the "BUY since …" line consumes it. Each ticker's rating and since-date must match as of the cache's last day. The two sources are compared from the later of their first days, and a ticker only one source knows, and only from before that day, is allowed: that is what the cache deliberately ignores.
      - 06a's exit code.
    - **The night log.** Each night's verdict goes to `core.night_checks` (migration `*_night_checks.sql`, RLS on, reachable only through `pipeline.record_night_check` and `pipeline.night_checks`).
    - **The readiness gate** (`db_night_check.py status`) counts consecutive NYSE trading days with a green record. A night with no record breaks the streak, so an unreachable database or a dead run cannot pass silently. 07e appends `DB_CUTOVER_STREAK n/20` to the run's status file.
    - **`DB_PRIMARY=1`** (`run.sh`) makes 06a blocking, and missing secrets become a failure rather than a skip. The git archive (06) still runs after it, so a failed publish never loses the day. The run ends `RESULT FAILED at db-publish (DB_PRIMARY=1; archived)` and exits 1.
    - **The restore drill** (`scripts/db_restore_drill.py`) rebuilds into a scratch database from Storage (SHA-checked against `core.snapshot_objects`) or from the git archive. It compares every run, row digest, change point and latest pointer with the live database, and refuses the live database's name. A pg test checks that it catches a tampered row.
    - **Runbooks.** `scheduled-tasks/RECOVERY.md` gains the database section: when to set `DB_PRIMARY`, stepping back, a failed 06a, and restore by PITR, from the archive, or by rehearsal.
  - **Results:**
    - Nightly check, on the 10 newest real runs published through the local Data API: parity over 2,538 tickers from 2026-09-14 to 2026-09-25 with 0 mismatches. The night recorded green, and the gate read 1/20, broken by the day before, which had no record.
    - Restore drill on the same 10 runs (25,052 rows): identical from both sources.

      | source | fetch | publish | total |
      |---|---|---|---|
      | Storage `.json.gz` | 16 s | 75 s | 112 s |
      | git archive | 87 s | 70 s | 178 s |

      That projects to 17–27 minutes for the 92-day archive. Every run, row and change point after the first restored day matched, and so did the latest pointers.
  - **Readers on the database (2026-09-27).**
    - `run.sh` exports `SNAPSHOT_STORE_BACKEND=postgres` whenever the Supabase secrets are set, so these read the database and fall back to the files:
      - carry-forward and the lost-SEC check;
      - the render's previous ratings and rating history;
      - portfolio alerts;
      - the gate N/A report;
      - the screen-skip cache.
    - Three fixes came with it:
      - `rating_history.json` keeps advancing from the files even when the database answers the render. Otherwise the cache freezes: 07e's parity check would compare the same day every night and count a vacuous green toward the cutover, and a fallback night would lose every change point older than the 10 staged days.
      - 07e treats a cache more than 5 trading days behind as "not checked".
      - A failed screen-skip save stops further database saves for the run, instead of paying the timeout at every 500-ticker flush.
  - **Still open, and yours:**
    1. Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to the cloud environment, and backfill the hosted project (the P3 runbook).
    2. Let 20 trading nights accumulate.
    3. Set `DB_PRIMARY=1`.
    4. Time a PITR restore once the project is on Pro, which is the third part of stability check 8.

## Verification: stability

1. **Fidelity.** For every backfilled date, the row rebuilt from the database equals the JSON row minus the excluded keys, under the codec's contract. `run_meta` matches too.
2. **Decision parity.**
   - Re-scoring the DB rows reproduces the stored `rating` for 100% of rows.
   - A backtest over Parquet gives the same metrics as over the JSON files (extends `tests/test_backtest_store.py`).
   - `rating_history` from Postgres matches the JSON cache, allowing only for dates the JSON cache deliberately ignores.
3. **Crash safety.**
   - SIGKILL during `COPY` and again inside `publish_run`: `core` is unchanged, `latest` still serves yesterday, and a re-run succeeds.
   - Two publishers at once: the second waits on the lock, and neither can see the other's temp rows.
   - Publishing the same date twice gives identical checksums.
4. **Out-of-order repair.** Publish D+1, then D, then D again with a changed rating. `rating_changes` matches a from-scratch window recompute, and `latest_results` still shows D+1.
5. **Bad input is rejected.**
   - A snapshot with 30% of rows missing fails.
   - More than 1% cast failures fails.
   - A new unregistered key passes.
   - `--force` passes and is audited.
6. **Staleness and outages.**
   - Rewrite a snapshot by hand without republishing: readers detect the SHA mismatch and fall back to the JSON.
   - With a DSN that drops traffic: the pipeline loses about 5 s once, the readers fall back, only `06a` fails, and the site is unaffected.
7. **Migration safety.** `squawk` passes, and every migration applies to a staging copy restored from prod.
8. **Restore drill.** Time a PITR restore into staging, then time a full rebuild from the Storage `.json.gz` files and another from the git archive alone.
9. **Two backends in CI.** The store tests run against both DuckDB and Postgres. The Postgres runs need `TEST_DATABASE_URL`, which CI provides from its `postgres:17` service. The offline pre-commit suite is unchanged.

## Verification: scalability (staging project)

1. **Volume.** Generate 10 years × 8,000 tickers (~20M rows) using the value distributions of real snapshots.
   - One night's publish of 8k rows, including the `rating_changes` recompute, takes under 5 minutes.
   - `EXPLAIN` shows partition pruning on every reader query.
   - Size and growth per year are recorded.
2. **Reader latency at 20M rows**, p95 with a cold cache:

   | Query | p95 target |
   |---|---|
   | ticker history (5y) | < 20 ms |
   | `last_known_rows` (8k tickers) | < 2 s |
   | rating changes since a date | < 50 ms |
   | the export step | < 60 s |

3. **Public load (k6).** Run 500 req/s for 30 minutes plus a 2,000 req/s burst against the exported files through the CDN.
   - p95 under 100 ms, errors under 0.1%, and a cache hit rate over 95% (from the `Age` / `cf-cache-status` headers).
   - Postgres receives zero queries, confirmed in `pg_stat_statements`.
   - Unauthenticated PostgREST calls to `core` are rejected.
4. **Publish during load.** Run a `publish_run` during a soak test with the readers active. No reader sees a partial day, and the readers' p95 stays within target.
5. **Growth.** Repeat checks 1–2 at 2× volume. Latency should grow sub-linearly; if it doesn't, add an index or a replica.

## Critical files
- **New:**
  - `supabase/config.toml` and the migrations: `0001_core.sql`, `0002_roles.sql`, `0003_publish_run.sql`
  - `data/db/columns.py`, `data/db/codec.py`, `data/db/reader.py`
  - `scripts/db_publish.py`, `scripts/export_public.py`
  - `tests/load/` (`scale.py`, the data generator and checks; `k6_public.js`)
- **Changed:**
  - `data/snapshot_store.py` (backend selection, `sync_snapshot_file` defer and publish)
  - `data/screen_skip_cache.py`
  - `scripts/report_html.py`, around line 297 (the rating-history coverage check)
  - `scripts/backtest.py` (the Parquet corpus)
  - `scripts/check_snapshot_store.py`, `scripts/ingest_snapshots.py`, `scripts/query_results.py`
  - `scheduled-tasks/cloud-daily-stock-analysis/run.sh` and `scheduled-tasks/RECOVERY.md`
  - `.github/workflows/ci.yml` and `daily-eod.yml`
  - `pyproject.toml`, `requirements.txt`, `CLAUDE.md`
- **Reused as they are:**
  - `read_snapshot`, `write_snapshot_file`, `BLOB_KEYS`
  - `archive_snapshot.py --list-blobs`
  - the `price_store.py` pattern of DuckDB over Parquet
  - the existing `vol/` and `px/` shard publishing
  - `data/throttle.py`, for the Storage HTTP calls
