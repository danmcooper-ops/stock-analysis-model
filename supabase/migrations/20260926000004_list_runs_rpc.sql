-- Run inventory over the Data API (design/supabase-migration.md, P3).
--
-- The cloud container reaches Supabase only over HTTPS and `core` is never
-- exposed, so a backfill (and later the post-publish check) asks this RPC
-- which dates are already published and from which file content, and skips
-- a date whose source_sha256 still matches.

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE FUNCTION pipeline.list_runs()
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT coalesce(jsonb_agg(jsonb_build_object(
           'run_date', r.run_date, 'status', r.status, 'n_rows', r.n_rows,
           'source_sha256', r.source_sha256, 'completed_at', r.completed_at)
         ORDER BY r.run_date), '[]'::jsonb)
    FROM core.runs r
$$;

REVOKE ALL ON FUNCTION pipeline.list_runs() FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION pipeline.list_runs() TO service_role, pipeline_writer, pipeline_reader;
GRANT USAGE ON SCHEMA pipeline TO pipeline_reader;
