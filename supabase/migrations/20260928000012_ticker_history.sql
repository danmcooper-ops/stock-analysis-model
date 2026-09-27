-- One ticker's headline history, read from the index alone
-- (design/supabase-migration.md, P5).
--
-- The plan's reader target is a 5-year ticker history in under 20 ms p95 on
-- a cold cache. Over (ticker_id, run_date DESC) each day of history is a
-- heap fetch from a different page (rows are clustered by date), so the
-- cost grew with the rows returned: 12.7 ms for 189 rows, 25.5 ms for 380 in
-- the P5 test, heading for ~85 ms at the 1,260 rows of 5 real years.
-- Carrying the headline columns in the index makes it an index-only scan.
-- The same index serves publish_run's anchor probe, which filters on
-- rating, so that probe goes index-only too. It replaces the narrower
-- index: same key, about 40 bytes more per row (~1% of a row's 3.75 KB).
--
-- Not CONCURRENTLY: impossible on a partitioned parent, and the CLI applies
-- each migration in a transaction. Built before the production backfill,
-- so the tables are small when this runs.

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '10min';

-- squawk-ignore require-concurrent-index-creation
CREATE INDEX results_ticker_hist_idx ON core.results (ticker_id, run_date DESC)
  INCLUDE (rating, mos, price, dcf_fv, _composite_score);
-- squawk-ignore require-concurrent-index-deletion
DROP INDEX core.results_ticker_date_idx;

-- [[date, rating, mos, price, dcf_fv, _composite_score], ...] for one ticker
-- over complete runs in [p_from, p_to], oldest first. Both bounds are
-- required: an open-ended range cannot prune the partitions created ahead
-- for future years.
CREATE FUNCTION pipeline.ticker_history(p_ticker text, p_from date, p_to date)
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
SET extra_float_digits = 3
AS $$
  SELECT coalesce(jsonb_agg(jsonb_build_array(r.run_date, r.rating, r.mos, r.price, r.dcf_fv, r._composite_score)
                            ORDER BY r.run_date), '[]'::jsonb)
    FROM core.tickers t
    JOIN core.results r ON r.ticker_id = t.ticker_id AND r.run_date BETWEEN p_from AND p_to
    JOIN core.runs ru ON ru.run_date = r.run_date AND ru.status = 'complete'
   WHERE t.ticker = p_ticker
$$;

REVOKE ALL ON FUNCTION pipeline.ticker_history(text, date, date) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION pipeline.ticker_history(text, date, date) TO service_role, pipeline_writer, pipeline_reader;
