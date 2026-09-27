# Scale and load tests (P5)

The tools behind the scalability checks in `design/supabase-migration.md`.
Nothing here runs in the normal test suite, except the small `pg`
end-to-end pass in `tests/test_db_scale.py`.

## `scale.py`: the database at volume

This script builds a synthetic history from real snapshot rows, then measures
the plan's targets. Each step prints JSON; `all` runs them in order and writes
one report.

| step | what it does |
|---|---|
| `seed` | Publishes a real snapshot, `--template`, through the nightly path. Its rows are the templates. |
| `generate` | Clones the templates into `--tickers` synthetic tickers × `--days` business days from 2027-01-04. It uses bulk SQL, only appends, and rebuilds the change points and latest pointers afterwards. |
| `publish` | Publishes the next day as an 8k-row snapshot over the Data API (`--direct` for a DSN) and times it. The target is under 5 min. |
| `explain` | Checks that each reader query scans only the partitions its dates need. |
| `size` | Reports bytes per row and projects per year at 8k tickers and at 20M rows. |
| `bench` | Restarts Postgres and drops the OS page cache, then measures p95 latency for each reader query against its target. |
| `load` | Publishes a day, then republishes it, while reader processes run. It checks that no reader ever sees a partial day. |
| `anon` | Sends unauthenticated Data API calls, which must all be refused. |

Run it against the local stack (`supabase start`). Two passes give the
growth check, because the second one appends:

```bash
python tests/load/scale.py all --tickers 8000 --days 188 --out output/p5_half.json
python tests/load/scale.py all --tickers 8000 --days 375 --out output/p5_full.json
```

- **Disk:** a row takes about 3.8 KB with its indexes. 375 days × 8k tickers is 3M rows, about 11 GB.
- **Cold cache:** the cold-cache restart uses `docker restart supabase_db_stock-analysis-model`. Pass `--no-restart` to measure with a warm cache.
- **Hosted projects:** to run against one, pass `--dsn` (session pooler) and `--api`, and set `SUPABASE_SERVICE_ROLE_KEY`. The harness writes synthetic 2027+ days into the database, so use a scratch project.

## `k6_public.js`: the public site through the CDN

It runs 500 req/s for 30 minutes, then a 2,000 req/s burst, against the
Cloudflare Pages site, over the page, its sidecars and the `px/` and `hist/`
shards. The thresholds are:

- p95 under 100 ms;
- errors under 0.1%;
- a cache hit rate over 95%, read from `cf-cache-status`.

```bash
k6 run -e BASE_URL=https://<project>.pages.dev tests/load/k6_public.js
```

It needs the Cloudflare deploy from P4c. Until that exists, it can only be
smoke-run against `wrangler pages dev`. That checks the script, not the CDN.
