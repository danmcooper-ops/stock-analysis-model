-- Read path (design/supabase-migration.md, P4): the pipeline's snapshot-store
-- readers over the Data API.
--
-- The cloud container reaches Supabase only over HTTPS and `core` is never
-- exposed (A1), so data/db/reader.DbStore calls these RPCs; they mirror the
-- DuckDB SnapshotStore readers (rows, last_known_rows, rating_history,
-- run_meta) so callers switch without change.
--
-- A row travels as {"ticker", "cols": {<typed column>: value}, "extra",
-- "blob"}. jsonb renders float8 through its output function, so
-- extra_float_digits = 3 is set on every function that builds rows: at the Supabase default of 0 doubles lose their last two digits
-- (0.03932028370017462 -> 0.0393202837001746). NaN and +-Infinity arrive as
-- the strings "NaN"/"Infinity"/"-Infinity"; the client casts by registry type.
-- NULL typed values are left out, as join_row does.

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

-- SQL for the typed columns of results row `r` as jsonb, restricted to
-- p_columns (every typed column when NULL). Built from the requested columns
-- only: to_jsonb(r) over all ~370 columns and filtering afterwards made a
-- 3-column read of one day take 1.5 s. NULLs are stripped (the values are
-- scalars, so jsonb_strip_nulls' recursion never reaches into anything), and
-- jsonb_build_object's 100-argument limit is met by chunks of 50 columns.
CREATE FUNCTION internal.typed_cols_sql(p_columns text[])
RETURNS text
LANGUAGE sql
STABLE
SET search_path = ''
AS $$
  SELECT coalesce('pg_catalog.jsonb_strip_nulls(' || string_agg(chunk, ' || ' ORDER BY grp) || ')',
                  $e$'{}'::jsonb$e$)
    FROM (SELECT grp, 'pg_catalog.jsonb_build_object('
                      || string_agg(format('%L, r.%I', attname, attname), ', ' ORDER BY attnum) || ')' AS chunk
            FROM (SELECT a.attname, a.attnum, (row_number() OVER (ORDER BY a.attnum) - 1) / 50 AS grp
                    FROM pg_catalog.pg_attribute a
                   WHERE a.attrelid = 'core.results'::regclass AND a.attnum > 0 AND NOT a.attisdropped
                     AND a.attname NOT IN ('run_date', 'ticker_id', 'extra', 'edgar_history_sha')
                     AND (p_columns IS NULL OR a.attname = ANY (p_columns))) c
           GROUP BY grp) g
$$;


-- Rows of one complete run, ordered by ticker. p_columns NULL means every
-- typed column; extra and the edgar blob only when asked for.
CREATE FUNCTION pipeline.read_rows(p_run_date date, p_columns text[] DEFAULT NULL,
                                   p_with_extra boolean DEFAULT false, p_with_blob boolean DEFAULT false)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
SET extra_float_digits = 3
AS $$
DECLARE
  v_rows jsonb;
BEGIN
  EXECUTE format($q$
    SELECT coalesce(jsonb_agg(jsonb_build_object(
             'ticker', t.ticker, 'cols', %s,
             'extra', CASE WHEN $2 THEN r.extra END,
             'blob', CASE WHEN $3 THEN b.value END)
           ORDER BY t.ticker), '[]'::jsonb)
      FROM core.results r
      JOIN core.runs ru ON ru.run_date = r.run_date AND ru.status = 'complete'
      JOIN core.tickers t ON t.ticker_id = r.ticker_id
      LEFT JOIN core.edgar_blobs b ON $3 AND b.sha = r.edgar_history_sha
     WHERE r.run_date = $1$q$, internal.typed_cols_sql(p_columns))
    INTO v_rows USING p_run_date, p_with_extra, p_with_blob;
  RETURN v_rows;
END
$$;


-- Each ticker's newest row among the p_max_lookback complete runs before
-- p_before, among rows whose p_require column is non-NULL (and non-empty for
-- text). A p_require that is not a column filters nothing, as in DuckDB.
-- Returns {"primary_date", "rows": [{"date", "ticker", "cols", "extra"}]}.
-- The winning (ticker, date) pairs are picked first on the narrow columns;
-- jsonb is built only for them.
CREATE FUNCTION pipeline.last_known_rows(p_before date, p_columns text[], p_max_lookback integer DEFAULT 7,
                                         p_require text DEFAULT 'rating', p_with_extra boolean DEFAULT false)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
SET extra_float_digits = 3
AS $$
DECLARE
  v_dates date[];
  v_type text;
  v_where text := '';
  v_rows jsonb;
BEGIN
  SELECT array_agg(run_date ORDER BY run_date) INTO v_dates
    FROM (SELECT run_date FROM core.runs
           WHERE status = 'complete' AND run_date < p_before
           ORDER BY run_date DESC LIMIT greatest(p_max_lookback, 0)) s;
  IF v_dates IS NULL THEN
    RETURN jsonb_build_object('primary_date', NULL, 'rows', '[]'::jsonb);
  END IF;
  IF p_require IS NOT NULL THEN
    SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) INTO v_type
      FROM pg_catalog.pg_attribute a
     WHERE a.attrelid = 'core.results'::regclass AND a.attname = p_require
       AND a.attnum > 0 AND NOT a.attisdropped;
    IF v_type IS NOT NULL THEN
      v_where := format(' AND k.%I IS NOT NULL', p_require);
      IF v_type = 'text' THEN
        v_where := v_where || format(' AND k.%I <> %L', p_require, '');
      END IF;
    END IF;
  END IF;
  EXECUTE format($q$
    WITH pick AS (
      SELECT DISTINCT ON (k.ticker_id) k.ticker_id, k.run_date
        FROM core.results k
       WHERE k.run_date = ANY ($1) %s
       ORDER BY k.ticker_id, k.run_date DESC)
    SELECT coalesce(jsonb_agg(jsonb_build_object(
             'date', r.run_date, 'ticker', t.ticker, 'cols', %s,
             'extra', CASE WHEN $2 THEN r.extra END) ORDER BY t.ticker), '[]'::jsonb)
      FROM pick p
      JOIN core.results r ON r.run_date = p.run_date AND r.ticker_id = p.ticker_id
      JOIN core.tickers t ON t.ticker_id = r.ticker_id$q$, v_where, internal.typed_cols_sql(p_columns))
    INTO v_rows USING v_dates, p_with_extra;
  RETURN jsonb_build_object('primary_date', v_dates[array_upper(v_dates, 1)], 'rows', v_rows);
END
$$;


-- Rating change points strictly before p_before (all when NULL):
-- [[ticker, date, rating], ...] ordered by ticker, date. Maintained by
-- publish_run over complete runs with the DuckDB store's rule.
CREATE FUNCTION pipeline.rating_history(p_before date DEFAULT NULL)
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT coalesce(jsonb_agg(jsonb_build_array(t.ticker, rc.run_date, rc.rating)
                            ORDER BY t.ticker, rc.run_date), '[]'::jsonb)
    FROM core.rating_changes rc JOIN core.tickers t ON t.ticker_id = rc.ticker_id
   WHERE p_before IS NULL OR rc.run_date < p_before
$$;


-- The run's top-level metadata, or NULL for an unknown date.
CREATE FUNCTION pipeline.run_meta(p_run_date date)
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
SET extra_float_digits = 3
AS $$
  SELECT jsonb_build_object('status', r.status, 'risk_free_rate', r.risk_free_rate,
                            'risk_free_rate_source', r.risk_free_rate_source, 'n_rows', r.n_rows,
                            'source_sha256', r.source_sha256, 'meta', r.meta - '_publish')
    FROM core.runs r WHERE r.run_date = p_run_date
$$;


REVOKE ALL ON FUNCTION internal.typed_cols_sql(text[]) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION pipeline.read_rows(date, text[], boolean, boolean) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION pipeline.last_known_rows(date, text[], integer, text, boolean) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION pipeline.rating_history(date) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION pipeline.run_meta(date) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION pipeline.read_rows(date, text[], boolean, boolean)
  TO service_role, pipeline_reader, pipeline_writer;
GRANT EXECUTE ON FUNCTION pipeline.last_known_rows(date, text[], integer, text, boolean)
  TO service_role, pipeline_reader, pipeline_writer;
GRANT EXECUTE ON FUNCTION pipeline.rating_history(date) TO service_role, pipeline_reader, pipeline_writer;
GRANT EXECUTE ON FUNCTION pipeline.run_meta(date) TO service_role, pipeline_reader, pipeline_writer;
