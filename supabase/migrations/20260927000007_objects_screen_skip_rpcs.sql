-- Storage manifest and screen-skip cache over the Data API
-- (design/supabase-migration.md, P4b).
--
-- After publishing a run, db_publish uploads the canonical snapshot
-- (.json.gz) and its Parquet export to Supabase Storage, checks both by
-- SHA-256, and records them here. The Phase-1 screen-skip cache
-- (data/screen_skip_cache.py) keeps its ~4.5k entries in core.screen_skip, so
-- the stateless cloud container no longer needs the copy on the git archive
-- (which stays as a fallback).

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

-- Upsert the Storage manifest row for a published run.
CREATE FUNCTION pipeline.record_snapshot_objects(p_run_date date, p_json_path text, p_json_sha256 text,
                                                 p_json_bytes bigint, p_parquet_path text,
                                                 p_parquet_sha256 text, p_n_rows bigint)
RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = ''
AS $$
  INSERT INTO core.snapshot_objects AS o (run_date, json_path, json_sha256, json_bytes, parquet_path,
                                          parquet_sha256, n_rows, uploaded_at)
  VALUES (p_run_date, p_json_path, p_json_sha256, p_json_bytes, p_parquet_path, p_parquet_sha256, p_n_rows, now())
  ON CONFLICT (run_date) DO UPDATE SET
    json_path = EXCLUDED.json_path, json_sha256 = EXCLUDED.json_sha256, json_bytes = EXCLUDED.json_bytes,
    parquet_path = EXCLUDED.parquet_path, parquet_sha256 = EXCLUDED.parquet_sha256,
    n_rows = EXCLUDED.n_rows, uploaded_at = now()
  RETURNING jsonb_build_object('run_date', o.run_date, 'json_path', o.json_path, 'parquet_path', o.parquet_path)
$$;


-- {ticker: {"kind", "mcap", "date"}}: the shape of screen_skip.json.
CREATE FUNCTION pipeline.screen_skip_load()
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
SET extra_float_digits = 3
AS $$
  SELECT coalesce(jsonb_object_agg(s.ticker, jsonb_strip_nulls(jsonb_build_object(
           'kind', s.kind, 'mcap', s.mcap, 'date', s.observed_on))), '{}'::jsonb)
    FROM core.screen_skip s
$$;


-- Replace the whole cache with p_entries (same shape as screen_skip_load).
-- The cache is small and written by one run at a time, so a full replace
-- keeps forget() (a ticker that passed the screen) exact.
CREATE FUNCTION pipeline.screen_skip_replace(p_entries jsonb)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_n integer;
BEGIN
  IF jsonb_typeof(p_entries) <> 'object' THEN
    RAISE EXCEPTION 'screen_skip_replace: expected an object of entries';
  END IF;
  DELETE FROM core.screen_skip;
  INSERT INTO core.screen_skip (ticker, kind, mcap, observed_on)
  SELECT e.key, e.value->>'kind', (e.value->>'mcap')::double precision, (e.value->>'date')::date
    FROM jsonb_each(p_entries) AS e;
  GET DIAGNOSTICS v_n = ROW_COUNT;
  RETURN v_n;
END
$$;


REVOKE ALL ON FUNCTION pipeline.record_snapshot_objects(date, text, text, bigint, text, text, bigint)
  FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION pipeline.screen_skip_load() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION pipeline.screen_skip_replace(jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION pipeline.record_snapshot_objects(date, text, text, bigint, text, text, bigint)
  TO service_role, pipeline_writer;
GRANT EXECUTE ON FUNCTION pipeline.screen_skip_load() TO service_role, pipeline_writer, pipeline_reader;
GRANT EXECUTE ON FUNCTION pipeline.screen_skip_replace(jsonb) TO service_role, pipeline_writer;
