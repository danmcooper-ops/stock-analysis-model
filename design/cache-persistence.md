<!-- Build guide: work phase by phase (A first). Each phase has an exit criterion; do not start the next until it passes. Phase A stands alone and is worth shipping by itself. -->

# Plan: recover the cloud run's statelessness tax

## Context

The cloud Routine runs in a fresh container each weekday, so every cache starts
empty. `run.sh` rescues exactly two pieces of state out of the `data/snapshots`
branch — `rating_history.json` and `screen_skip.json` (`run.sh:258-266`, written
back in step 6 behind size and `SMOKE` guards). Two expensive caches are not
rescued:

- **`output/prices/`** — step 03 re-downloads full history for every ticker in
  the newest snapshot (~2,300) plus benchmarks, every night.
- **`data/cache/sec_facts/`** — the gzipped companyfacts blobs. The whole
  filing-index watermark design exists to avoid re-fetching them, and in the
  cloud it starts empty regardless.

On a persistent checkout both survived between runs, so this cost is specific
to the cloud move and is paid every weekday.

**Sizing the price leg.** `download_ticker()` (`scripts/download_prices.py:88`)
sleeps `--delay` (0.35 s) and then fetches `period="max"` — one request per
ticker, in a plain sequential `for` loop (`download_prices.py:170`). With an existing
fresh file it returns `"fresh"` *before* the sleep and before the network, so on
a warm cache the step is nearly free; on a cold one nothing short-circuits. At
2,300 tickers × (0.35 s + ~1.2 s per request — the measured yfinance mean is
1.03 s for lighter calls, and `period="max"` returns more) the step costs
**roughly 50-60 minutes of wall clock a night**. Treat that as an estimate to be
confirmed by measurement in Phase A, not a measured figure.

## The finding that reorders the work

The obvious fix is to transport the cache into the container. The cheaper fix is
to stop the download being sequential.

`data/throttle.py` is already lock-guarded and thread-safe, with
`penalize()`/`relax()` for adaptive backoff, and `ThreadPoolExecutor` is an
established pattern in this pipeline (`analyze_stock.py:3100` for Phase 1,
`:3644` for Phase 2). Four workers against a 0.4 s interval admit ~2.5 calls/s,
which turns ~55 minutes into **~15 minutes with no new storage, no new
credentials and no restored state to get wrong**.

So Phase A is parallelisation, and cache transport is Phase B — worth doing,
but it should be built on top of a download that is already fast, not instead
of one.

## Phase A — parallelise the price download

**Change (`scripts/download_prices.py`):**
- Replace the inline `time.sleep(delay)` with a shared `Throttle`, and the
  sequential loop with a `ThreadPoolExecutor(max_workers=...)`.
- New knob `--price-workers` / `PRICE_IO_WORKERS` (default 4), mirroring
  `--phase1-workers` / `PHASE1_IO_WORKERS`.
- The pool does network only. Progress printing, the failure list and the final
  census stay in the main thread, consumed in submission order, so stdout is
  unchanged apart from a pool banner — the same contract Phase 1's pool holds.
- Make the write atomic: `download_ticker()` currently calls
  `df.to_parquet(dest)` directly. Under a pool, and given the container can be
  restarted mid-step, write `dest.tmp.<pid>.<tid>` and `os.replace()`, matching
  `sec_facts_cache.put()` (`data/sec_facts_cache.py:122-125`).
- Feed Yahoo's soft-throttle signal into `penalize()`/`relax()` so a wrong
  worker count self-corrects rather than burning three attempts per ticker.

**Leave alone:** `period="max"`, the `"fresh"`/`"skipped"` short-circuits,
`_parquet_is_stub()` re-download, and `--max-age-days`. Backtests depend on full
history; this phase changes scheduling only.

*Passes when:* on a fixed ticker sample, the parallel run produces parquets with
identical bar coverage to a sequential run; wall clock improves ≥3×; the run
records no soft throttles; and `scripts/validate_ratings.py` over the resulting
prices is unchanged.

**Status: shipped.** `tests/test_download_prices_pool.py` pins the parity
(identical stdout rows and parquet set at 1 and 4 workers), that only real
fetches tick the throttle, the atomic write, and the streak governor.
Synthetically — 60 tickers, 0.2 s per request, 0.05 s interval — 12.2 s -> 3.2 s
at 4 workers; 8 workers gives nothing further because the throttle ceiling
(3.0 s) is then binding, which is the intended shape. The production saving is
still unmeasured: it needs a real nightly run, and step 03's wall clock in
`$WORK/status.txt` is where to read it. `run.sh` needs no change — the default
worker count applies on its own.

## Phase B — persist the price cache between runs

**Transport: Supabase Storage.** This plan originally argued the transport was
constrained to git, because `push_with_retry()` (`run.sh:121`) is a plain
`git push` through the container's credential helper and there was no API
token to reach anything else. That is no longer true. The Supabase migration's
P1-P3 landed while this plan was being written, and `SUPABASE_URL` /
`SUPABASE_SERVICE_ROLE_KEY` are now in the cloud environment and exercised
nightly by step `06a-db-publish` (`run.sh:366-376`). Storage is the same HTTPS
API with the same key, so the cache rides credentials and an egress path that
are already proven in this container.

That also settles the shape: P0 found the container **cannot reach Postgres
over TCP**, which is why `db_publish` speaks the Data API over HTTPS. Storage
is HTTPS too, so it is unaffected by the block that forced that fallback.

A private bucket holds one object per ticker, keyed `prices/<TICKER>.parquet`.
Objects are replaced in place; there is no history to accumulate and no
100 MiB blob cap to shard around.

**Why not a git cache repo** (the earlier recommendation, kept as the fallback
if Storage is ever unavailable): every parquet gains a bar each trading day, so
the whole set churns nightly. Even force-pushed as a single commit, superseded
objects linger until GitHub GCs them, and `data/snapshots` has already taken
that repo past 5 GB. It would work, but it trades a bucket for a repo that
needs periodic deletion and recreation.

**Restore** in a new non-blocking step `02b-restore-prices`, before step 03:
list the bucket prefix, download in parallel into `output/prices/` (the same
`Throttle` + pool shape Phase A established, against Storage rather than
Yahoo), and report the file count. **Save** after step 05e (the top-up), so
the stored cache includes new entrants. Guards, matching both the `screen_skip`
write-back and `db_publish`: skip the save when `SMOKE=1` or `DRY_RUN=1`, skip
silently when the Supabase secrets are unset, and refuse to push a set smaller
than a floor fraction of what was restored.

**Why this is safe.** Price freshness is read from parquet *content*
(`_parquet_max_date()` reads the index, `scripts/download_prices.py:52`), not
from file mtime. A restored file that is stale simply fails the
`last >= cutoff` test and gets re-downloaded. So a cache that is old, partial or
corrupt degrades to today's cold behaviour — it can never serve stale prices
silently, which matters because the docstring notes stale prices *"silently
truncate forward-return calculations"*. Step 02b is therefore non-blocking: a
failed restore costs the hour, not the run.

*Passes when:* a warm restore drops step 03 under 5 minutes; a deliberately
7-day-stale bundle yields parquets identical to a cold run; a corrupt object is
re-downloaded rather than used; a run without the Supabase secrets behaves
exactly as today; and a `SMOKE=1` run leaves the stored cache untouched.

**Status: shipped, unproven against the real bucket.**
`data/price_cache_store.py` and `scripts/price_cache.py` implement the round
trip; `run.sh` restores in `02b` and saves in `05e2`, both non-blocking.
`tests/test_price_cache_store.py` (17 tests) pins the request shape, list
pagination, the `x-upsert` replace, and the harm-avoidance properties: a local
file is never overwritten by a stored one, one bad object does not end the
sweep, a failed download leaves no truncated parquet, an under-floor local set
is refused, and absent credentials are a clean no-op.

What the tests **cannot** cover is the live API, since they run against a fake
session. Before relying on this, confirm on a real run that the bucket exists
and the service-role key can write it, that Storage's list pagination behaves
as assumed past 1,000 objects, and that the first night's `05e2` uploads the
full universe rather than erroring. The first `02b` finds an empty bucket and
correctly restores nothing, so night one is a normal cold run that seeds the
cache; the saving appears on night two.

## Phase C — the SEC facts cache

**Status: shipped, against this plan's own advice.** What follows is the
recommendation as written; it was overridden deliberately, and the value
judgement still stands — this buys fewer SEC requests and a smaller failure
surface, not wall clock. `facts_stats` in `provenance.timings` is where to
check whether that judgement was right.

The original case for deferring: the Phase-1 prefetch pool already moved the
`xbrl` leg from 37% of Phase 1 to 0.4% by hiding it behind other work, so
persisting these blobs buys politeness to SEC rather than time. (Note the
measurement behind that figure was 12 tickers, not 2,300, so it is weaker
evidence than it looks.) It also carried a hazard the price cache did not:

- **mtime-based freshness.** `get()`, `age_days()` and `prune_expired()` all
  read `os.path.getmtime()`. A restore writes every file now, so restored
  blobs look freshly downloaded and the `max_age_days` backstop never fires
  again — transport alone would have silently disabled the one guard against
  indefinitely-stale fundamentals.
- **The watermark travels with the blobs.** `_state.json` holds
  `last_index_sweep` and must be restored atomically with what it describes,
  or eviction is reasoning about a filing window the cache does not match.

**How it was resolved**, taking this plan's own preference for an explicit
stamp over engineering the transport to preserve mtime — though not the form
it expected. Rather than stamping each blob, the backstop now also reads the
**watermark**, which survives the trip and measures what the backstop
actually cares about: whether filing-driven eviction kept up.
`refresh_stale_facts()` never advances the watermark over a day it could not
read, so a current watermark means every filing day since has been walked and
its filers evicted — precisely the claim that lets an untouched blob stay
valid. `SECFactsCache.sweep_is_lagging()` makes the whole cache read as
missing when that lag exceeds `max_age_days`. A cache that never swept has no
watermark and falls back to mtime, which on a dev box is honest.

A third hazard surfaced during the build, in neither the plan nor the
original review: **evictions have to reach the bucket.** `invalidate()`
removes a blob locally because its filer filed; if the object stayed stored
it would come back on the next restore, and the sweep would *not* evict it
again, because the watermark has already moved past that day. That blob would
then be served indefinitely. So the save deletes stored objects with no local
counterpart, gated on the same 80% floor as the price cache.

`run.sh` restores in `02c` and saves in `04b`, immediately after the analysis
— the only step that fetches facts or evicts them, and before enrichment that
could still fail. Both non-blocking and skipped without the Supabase secrets.

*Passes when:* a restored cache with a current watermark serves its blobs; one
with a lagging watermark serves none despite fresh mtimes; an eviction
propagates to the bucket; and a run without the secrets behaves as before.
All four are pinned in `tests/test_sec_facts_cache_store.py` (16 tests) and
exercised together end to end. As with Phase B, the live API is unverified —
the tests run against a fake session.

## Relationship to the Supabase plan

Complementary, not independent. `design/supabase-migration.md` addresses the
*data* side of statelessness — the snapshots — and says nothing about these
caches; this plan covers the caches. But since that migration is now live
through P3, it supplies Phase B's transport rather than merely coexisting with
it, and Phase B should reuse `data/db/connect.py`'s credential handling rather
than reading the environment a second way.

Phase A is genuinely independent: it changes scheduling inside one script and
needs no credentials, no network path and no `run.sh` change.

## Verification

1. **Parity.** Same ticker sample, cold sequential vs. cold parallel vs. warm
   restore: identical parquet bar coverage in all three.
2. **Timing.** Record step 03 wall clock in `$WORK/status.txt` across the three
   modes and in `provenance.timings`, so the saving is measured rather than
   assumed.
3. **Degradation.** With the cache repo unreachable, step 02b fails
   non-blocking and the night proceeds at today's cost.
4. **Guards.** `SMOKE=1` and `DRY_RUN=1` never write the cache; a shrunken set
   is refused.
5. **Throttle health.** No increase in `empty_attempts` (`YFinanceClient.stats`)
   against a sequential baseline.

## Critical files

- **Phase A (done):** `scripts/download_prices.py` (pool, throttle, atomic
  write, interrupt handling), `scripts/config.py` (`PRICE_IO_WORKERS`),
  `tests/test_download_prices_pool.py`, `CLAUDE.md`. No `run.sh` change.
- **Phase B (to do):** a `data/price_cache_store.py` for the Storage
  round-trip; `scheduled-tasks/cloud-daily-stock-analysis/run.sh` (step 02b and
  the save after 05e); `scheduled-tasks/cloud-daily-stock-analysis/SKILL.md`;
  a private Storage bucket.
- **Reused as-is:** `data/throttle.py` and the Phase-A pool shape,
  `data/db/connect.py`'s credential handling, `db_publish`'s
  secrets/`SMOKE`/`DRY_RUN` skip idiom, the `screen_skip` size guard.

## What this does not do

It does not make the model more accurate, change any valuation output, or touch
the Phase-1/Phase-2 screen. It reclaims roughly an hour of nightly wall clock
and removes a standing dependency on re-fetching data that has not changed.
