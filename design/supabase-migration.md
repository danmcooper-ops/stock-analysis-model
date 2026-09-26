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

## Target architecture

```
nightly run ─COPY─► TEMP load table ─► core.publish_run() ── one transaction, session pooler
                                        ├─ core.results         (yearly partitions, typed + jsonb)
                                        ├─ core.rating_changes  (recomputed per affected ticker)
                                        ├─ core.latest_results  (only if D ≥ latest complete)
                                        └─ core.runs.status='complete', source_sha256
          └─► Storage: full .json.gz (private) + Parquet (analytics)
          └─► public payloads: latest.json, changes.json, history shards → Pages / public bucket (CDN)

public traffic ─► CDN (static JSON)                 ✗ no anon access to Postgres
pipeline readers ─► core.* (pipeline_reader, timeouts, JSON fallback)
backtest ─► DuckDB over Parquet (union_by_name), JSON fallback per date
```

## Schema: fixed, versioned, changed only by migrations

There are two private schemas: `core` (the data) and `internal` (functions). Nothing is exposed to PostgREST for the `anon` role **[R1]**.

**Tables**
- `core.tickers`: `ticker_id identity pk`, `ticker unique`, `cik`, `name`, `sector`, `industry`, `exchange`, `first_seen`, `last_seen`.
- `core.runs`: `run_date pk`, `status` (`loading`|`complete`|`failed`), `risk_free_rate`, `risk_free_rate_source`, `macro_regime`, `n_rows`, `source_sha256` (SHA-256 of the **uncompressed canonical JSON** **[R2]**), `pipeline_version`, `meta jsonb`, `started_at`, `completed_at`.
- `core.results`:
  - Primary key `(run_date, ticker_id)`, `PARTITION BY RANGE (run_date)` with **yearly** partitions, pre-created 5 years ahead by a migration. There is no DEFAULT partition, and publishing fails fast if a date has no partition. No DDL runs at runtime **[R7]**.
  - About 120 typed columns, taken from the registry.
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
- Maps each promoted key to `(pg_type, nullable)`. It is seeded with the ~120 keys the code actually reads:
  - `_PREV_DRIVER_KEYS` (`scripts/report_html.py:166`);
  - the 26 gates' `_gate_`/`_score_`/`_gp_` keys and their input fields;
  - `APPLICABILITY_FIELDS` (`scripts/scoring.py:103`);
  - the inputs to the rating caps, trap score and `prepare_scoring_fields`;
  - carry-forward keys (`shares_out`, `mcap`, `data_source`);
  - backtest keys (`pp_multiple`, `trap_score`, `_gates_passed_num`, …);
  - identity and price keys.
- Promoting another key is a reviewed migration plus a backfill that moves it out of `extra`.
- A test extends `tests/test_snapshot_store.py:421`: every key a scorer or reader touches must be in the registry.

**Codec (`data/db/codec.py`) [R8]**
- **Non-finite numbers.** Postgres `double precision` columns keep NaN and ±Infinity natively. Inside jsonb, which can't hold them, they are encoded as `{"$nf": "NaN"|"Inf"|"-Inf"}`.
- **Reserved keys.** A real dict whose keys start with `$` is escaped by wrapping it as `{"$lit": …}`.
- **Encoding.** The codec walks every nested value and serializes with `json.dumps(allow_nan=False)`, so a bare `NaN` can never reach psycopg.
- **NUL characters** (`\u0000`) in strings are escaped both in text columns and in jsonb.
- **Fidelity contract.** Before writing it, grep the scorers and readers for `'k' in row` checks.
  - **None found:** fidelity means `.get()`-equivalence, with NaN equal to NaN and int equal to float.
  - **Any found:** each row also stores a bitmap recording which typed keys were present.
- **Pinned quirks, each with its own test:** `"Infinity"` strings become floats, and jsonb normalizes `-0.0` to `0`.
- **Tests.** Hypothesis property tests round-trip adversarial values: `$nf`-shaped dicts, NUL characters, nested NaN, and very large ints.

**What stays out of the DB.** The report-only narrative keys (`DEFAULT_EXCLUDE_KEYS`, `data/snapshot_store.py:92`) live only in the full `.json.gz` in private Storage. That file is the canonical copy of the complete row.

**Roles and timeouts [R10, R11]**

| Role | Access | Timeouts and connection settings |
|---|---|---|
| `pipeline_writer` | Writes only through `internal.publish_run` | `lock_timeout=5s`, `statement_timeout=10min` |
| `pipeline_reader` | Read-only on `core` | `statement_timeout=30s`, `idle_in_transaction_session_timeout=60s` |
| `public_export` | Read-only; used by the export step | — |

- `anon` and `authenticated` have no grants on `core`, and RLS is on for every table.
- Every Python connection sets `connect_timeout=5`.
- Anything that uses port 6543 sets `prepare_threshold=None`.

## Write path: one writer, validated, atomic, re-runnable

1. **Where it runs.** `scripts/db_publish.py --run-date D` is the blocking step `06a-db-publish` in `scheduled-tasks/cloud-daily-stock-analysis/run.sh`. It runs after `05f` rescore and before the git archive.
2. **One transaction on the session pooler (port 5432) [R6].** Inside it:
   - Take `pg_advisory_xact_lock`.
   - Create a `TEMP TABLE … ON COMMIT DROP`.
   - Split each row with the registry and codec, and `COPY` the rows in with psycopg 3.
   - Call `internal.publish_run(D, sha256, expectations)`.

   No other writer can see the staged rows or truncate them. Nothing is unlogged, so replicas and crashes are safe.
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
- The files go to Pages, the existing CDN, and optionally to a public Storage bucket, where Smart CDN caches them.
- The site fetches only those files.
- There is no anon access to PostgREST and no Edge Function in the hot path. If a live query is ever needed, it will be a fixed-parameter RPC behind a CDN that I've confirmed caches it. P0 checks that with the `cf-cache-status` and `Age` headers.

## Operations

- **Environments:** local (`supabase start`), `staging` (a Supabase project that is also the load-test target), and `prod` (Pro plan with point-in-time recovery, PITR).
- **Migrations:**
  - live in `supabase/migrations/`;
  - CI applies them to a fresh `postgres:17` service;
  - `squawk` lints them for table locks;
  - `supabase db push` runs to staging on merge and to prod by manual dispatch.
- **Retention.** Old yearly partitions are already exported to Parquet and can be removed with `DETACH … CONCURRENTLY`, outside a transaction. Backfill builds a standalone table with a CHECK constraint and then `ATTACH`es it **[R7]**.
- **Observability:**
  - `pg_stat_statements`;
  - a slow-query log for anything over 200 ms;
  - `core.runs` timings added to `provenance.timings`;
  - `daily-eod.yml` alerts if `runs.status` isn't `complete` by 07:00 ET.
- **Secrets** go in the cloud environment:
  - `SUPABASE_DB_URL`: the session-pooler DSN for `pipeline_writer`;
  - `SUPABASE_READER_URL`;
  - `SUPABASE_URL`;
  - `SUPABASE_SERVICE_ROLE_KEY`: used for Storage only.
- **Dependencies.** Add `psycopg[binary]~=3.2` to `pyproject.toml` and `requirements.txt`. Keep `duckdb`.

## Phases and the check that closes each

- **P0: Spike.**
  - Connect from the cloud Routine to `*.pooler.supabase.com:5432`. Its egress goes through an HTTPS proxy, so the host may need allowing in the network settings. If TCP can't get through at all, `06a` falls back to an HTTPS ingest function.
  - Measure bytes per row on 10 archived snapshots.
  - Confirm the CDN caching behaviour.
  - *Passes when:* the connection works, and the projected size at **8k tickers** is ≤ 8 GB/yr **[R13]**.
- **P1: Migrations, registry, codec and roles.**
  - *Passes when:* CI applies the migrations cleanly, `squawk` reports nothing, and the codec property tests and registry coverage tests pass.
- **P2: Write path.**
  - *Passes when:* stability checks 3–6 pass.
- **P3: Backfill the whole archive.**
  - *Passes when:* fidelity and decision parity (stability checks 1–2) hold for every date.
- **P4: Switch readers, Parquet backtest, public export.**
  - *Passes when:* the parity checks pass and the site renders from the exported files.
- **P5: Scale tests.**
  - *Passes when:* every scalability target is met.
- **P6: Cutover.**
  - `06a` becomes blocking, and `RECOVERY.md` and `CLAUDE.md` are updated.
  - *Passes when:* 20 nightly runs in a row publish green with rating-history parity, and the restore drill passes.

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
  - `tests/load/` (the k6 scripts and the data generator)
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
