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

## Phase B — persist the price cache between runs

**Transport is constrained to git.** `push_with_retry()` (`run.sh:121`) uses a
plain `git push` to an HTTPS remote via the container's credential helper. There
is no `gh` CLI and no API token, so GitHub Releases are not available without
adding a secret. Git it is.

**Where.** A **dedicated cache repository**, written as a single force-pushed
commit — the `pages-live` pattern (`run.sh:432-435`: `git init` a scratch dir,
one commit, `push --force`). Files are stored individually (~100 KB each, far
under the 100 MiB blob cap); a single tarball would breach it.

Not the main repo: every parquet gains a bar each trading day, so every file
changes nightly. Even force-pushed, the superseded objects linger until GitHub
GCs them, and `data/snapshots` has already taken that repo past 5 GB. Isolating
the churn means the cache repo can be deleted and recreated if it ever bloats,
with no risk to the archive. This needs the new repo added to the session's
GitHub scope.

**Restore** in a new non-blocking step `02b-restore-prices`, before step 03:
blob-less clone, materialise into `output/prices/`, report the file count.
**Save** after step 05e (the top-up), so the pushed cache includes new entrants.
Guards, matching the `screen_skip` write-back: skip the save when `SMOKE=1`,
skip on `DRY_RUN=1`, and refuse to push a set smaller than a floor fraction of
what was restored.

**Why this is safe.** Price freshness is read from parquet *content*
(`_parquet_max_date()` reads the index, `scripts/download_prices.py:52`), not
from file mtime. A restored file that is stale simply fails the
`last >= cutoff` test and gets re-downloaded. So a cache that is old, partial or
corrupt degrades to today's cold behaviour — it can never serve stale prices
silently, which matters because the docstring notes stale prices *"silently
truncate forward-return calculations"*. Step 02b is therefore non-blocking: a
failed restore costs the hour, not the run.

*Passes when:* a warm restore drops step 03 under 5 minutes; a deliberately
7-day-stale bundle yields parquets identical to a cold run; a corrupt file is
re-downloaded rather than used; and a `SMOKE=1` run leaves the stored cache
untouched.

## Phase C — the SEC facts cache (conditional; do not start with this)

Lower value than it looks. The Phase-1 prefetch pool already moved the `xbrl`
leg from 37% of Phase 1 to 0.4% by hiding it behind other work, so persisting
these blobs buys politeness to SEC and a smaller failure surface, **not** wall
clock. It also carries a hazard Phase B does not:

- **mtime-based freshness.** `get()`, `age_days()` and `prune_expired()` all
  read `os.path.getmtime()` (`data/sec_facts_cache.py:84, 94, 169`). A git
  checkout stamps every restored file with the checkout time, so restored blobs
  look freshly downloaded and the `max_age_days` backstop never fires. Transport
  must preserve mtime (a tar does; a git checkout does not), or the cache needs
  an explicit stamp inside the blob instead of mtime.
- **The watermark travels with the blobs.** `_state.json` (`_STATE_FILE`,
  line 40) lives inside the cache directory and holds `last_index_sweep`. It
  must be restored atomically with the blobs it describes, or eviction is
  reasoning about a filing window the cache does not match.

Recommendation: ship A and B, measure, and only then decide whether C is worth
the mtime work. If it is, prefer changing freshness to an explicit stamp over
engineering the transport to preserve mtime.

## Relationship to the Supabase plan

Independent, and deliberately so — this is a caching change, not a data-model
change. `design/supabase-migration.md` addresses the *data* side of
statelessness and says nothing about these caches. Keep Phase B's restore and
save behind one pair of functions so the backend is swappable: if the migration
lands, Supabase Storage replaces the cache repo without touching `run.sh`'s step
structure.

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

- **Changed:** `scripts/download_prices.py` (pool, throttle, atomic write),
  `scripts/config.py` (`PRICE_IO_WORKERS`),
  `scheduled-tasks/cloud-daily-stock-analysis/run.sh` (steps 02b and the save),
  `scheduled-tasks/cloud-daily-stock-analysis/SKILL.md`, `CLAUDE.md`.
- **New:** the cache repository; a `tests/test_download_prices.py` case for the
  pool's parity with the sequential path and for the atomic write.
- **Reused as-is:** `data/throttle.py`, `push_with_retry()`, the `pages-live`
  single-commit publish pattern, the `screen_skip` size/`SMOKE` guard idiom.

## What this does not do

It does not make the model more accurate, change any valuation output, or touch
the Phase-1/Phase-2 screen. It reclaims roughly an hour of nightly wall clock
and removes a standing dependency on re-fetching data that has not changed.
