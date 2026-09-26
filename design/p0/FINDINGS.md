# P0 spike findings (2026-09-26)

These are the results of P0 from `design/supabase-migration.md`. P0 has three checks, and all ran in a Claude Code cloud container, which uses the same kind of egress proxy as the nightly cloud Routine.

| Check | Result | Verdict |
|---|---|---|
| Postgres connection from the cloud | Raw TCP to the Supabase pooler is blocked. HTTPS to the project works. | **Fails as designed.** The write path moves to HTTPS (A1). |
| Storage size (the check: ≤ 8 GB/yr at 8k tickers) | 2.8 GiB/yr at today's ~2.5k rows/day. 8.9 GiB/yr at 8k rows/day. | **Just misses** at 8k. The schema is revised (A2) and the check is amended (A3). |
| CDN caching | Pages (Fastly) caches, but the payloads are 25–29 MB. | **Caching works.** Bandwidth is the real limit (A4). |

## 1. Connection

- `aws-0-us-east-1.pooler.supabase.com` resolves, but TCP to ports 5432 and 6543 times out. That happens both directly and through the proxy's CONNECT.
- The proxy's own README lists "raw-TCP databases" under *not supported through the proxy*, so adding the host to the network allowlist would not help.
- HTTPS to the project works: `https://llvjffhwuivwsusovjfg.supabase.co/rest/v1/` answers `401` without an API key. The Data API (PostgREST), Storage and Edge Functions are therefore all reachable.
- The session pooler should work from a dev machine or from CI, since those have ordinary egress. I could not test that here.

## 2. Storage size

**Method.** `design/p0/pg_sizing.py` loads real snapshots into the planned schema on a local Postgres 16:
- yearly partitions;
- lz4-compressed `jsonb`;
- `edgar_history` stored once per distinct value;
- the report-only keys (`DEFAULT_EXCLUDE_KEYS`) excluded.

The sample was the 10 newest archived snapshots (2026-09-14 to 2026-09-25): 25,052 rows, 2,489–2,531 per day, 385 keys per row and 409 distinct keys overall. CLAUDE.md's "~270 keys" figure is out of date.

| Layout | Typed cols | Bytes/row (table + index) | edgar blobs | GiB/yr, ~2.5k/day | GiB/yr, 8k/day |
|---|---|---|---|---|---|
| Plan as written: ~120 hot keys typed, rest in `extra jsonb` | 126 | 7,082 | 24 MiB | 4.44 | 14.17 |
| **Every scalar key typed** (dict/list and new keys in `extra`) | 371 | 4,277 | 24 MiB | **2.79** | **8.90** |
| Also deduplicate `roic_by_year`, `_nopat_by_year`, `_ic_by_year` | 368 | 3,757 | 28 MiB | 2.68 | 8.57 |
| *Reference: today's DuckDB store (slim edgar only)* | ~370 | — | — | 91 MiB for the 10 days | — |

**Observations**
- **`jsonb` is expensive for many small scalars.** With ~260 numeric keys in `extra`, `pg_column_size(extra)` came to 5.4 KB per row, more than the 4.3 KB of raw JSON for all the kept keys. The same values as typed `double precision` columns cost about 8 bytes each. Promoting every scalar key saves about 40%.
- **Type conversion was clean.** Across 2.67M typed values from the 126 hot keys, and 7.0M from the 371 scalar keys, no value failed to convert to its column's type.
- **`edgar_history` deduplication works as expected.** New blobs per day ran at about 0.3–1.4%, with one exception: 2026-09-17 re-seeded 2,004 blobs, so a day of widespread history changes does happen.
- **The year-keyed dicts change about 9% of the time per day.** Deduplicating them saves only about 4%, which is not worth the extra complexity.
- **Loading is fast.** A day loads in 2.4–3.8 s locally with `COPY` (2.5k rows), so about 10 s for 8k rows. The 5-minute target has plenty of headroom, though a load over HTTPS will be slower.
- **The 8k figure is a ceiling, not the expected case.** The nightly run stores the ~2.5k tickers that pass the screen (`--mcap-min 300e6`), not the whole ~8k US universe.

## 3. CDN

- The live site `danmcooper-ops.github.io/stock-analysis-model/` is served by Fastly (`via: varnish`) with a fixed `cache-control: max-age=600`. The first request was a `MISS`, and a repeat within a second was a `HIT`.
- **The payloads are large:** `index.html` is 25.7 MB and `hist.json` 28.8 MB. GitHub Pages has a soft bandwidth limit of 100 GB/month, which works out to roughly 2,000 full page loads a month. Under heavy traffic, the static site's bandwidth runs out long before Postgres is the limit.
- I could not check Supabase Storage's Smart CDN, because there are no project credentials or public bucket in this session.

## Amendments to the plan

- **A1: every connection from the cloud goes over HTTPS.**
  - Publishing uses Data API RPCs with the service-role key:
    1. `internal.stage_chunk(load_id, rows jsonb)` appends chunks of about 1 MB to a staging table keyed by `load_id`. It is idempotent per `(load_id, chunk_no)`.
    2. `internal.publish_run(load_id, …)` validates and applies the day in one transaction. This is the fallback that plan item R6 already describes.
  - The functions go in a schema that only `service_role` can execute. Chunk uploads go through `data/throttle.py`.
  - Nightly readers use small RPCs (`last_known_rows(cols, before)`, `rating_changes(since)`), or DuckDB over Parquet staged from Storage. The existing JSON fallbacks still apply.
  - `psycopg` over the session pooler is only for dev machines, CI and admin work.
- **A2: the column registry types every stable scalar key** (about 371 today, well under Postgres's 1,600-column limit). It is generated from the observed snapshots and changed only through migrations. `extra jsonb` holds dicts, lists and keys not yet promoted. A key that appears in `extra` for 30 days is flagged for promotion. `edgar_history` stays the only content-addressed blob.
- **A3: the size check is amended.** Size at today's universe must be ≤ 4 GiB/yr (measured: 2.8). At 8k tickers the database runs on the Pro plan, keeping an N-year window of history (default 3 years, about 27 GiB at 8k) with older yearly partitions detached, since they already exist as Parquet. Extra disk costs little next to the plan fee.
- **A4: the public payloads are split small and moved off GitHub Pages.**
  - `latest.json` is kept slim, with per-ticker history shards.
  - For heavy traffic they are hosted on a CDN with no soft bandwidth cap: Cloudflare Pages/R2 (no egress fees), or Supabase Storage with Smart CDN (metered egress).
  - P5's load test targets that host.

## Reproducing

```bash
# a local Postgres 16 (or `supabase start`, which exposes port 54322)
python design/p0/pg_sizing.py --promote scalar \
    --dsn postgresql://postgres@127.0.0.1:55432/postgres output/results_2026-09-*.json.gz
```

Staging snapshots and blobs from the archive branch works the same way as step 02 in `scheduled-tasks/cloud-daily-stock-analysis/run.sh`, using `scripts/stage_snapshot_blobs.stage_blobs`.
