-- "Rating changes since a date" by date, not by scanning every change point
-- (design/supabase-migration.md, P5).
--
-- core.rating_changes is keyed (ticker_id, run_date), so a since-date read
-- (the portfolio alerts, the report's recent-changes views, the public
-- export) scanned the whole table: 10 ms over 46k change points in the P5
-- test, growing linearly to ~460k change points at 20M result rows, past
-- the plan's 50 ms p95. INCLUDE makes it an index-only scan.
--
-- Not CONCURRENTLY: the CLI applies each migration in a transaction, and the
-- table is small (one row per rating change, ~14k on the 92-day archive) and
-- written only by the nightly publish, which holds the advisory lock and can
-- wait out the short build.
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

-- squawk-ignore require-concurrent-index-creation
CREATE INDEX rating_changes_date_idx ON core.rating_changes (run_date) INCLUDE (ticker_id, rating, prev_rating);
