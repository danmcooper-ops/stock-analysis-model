-- What became of a publish whose HTTP response was lost
-- (design/supabase-migration.md, P5).
--
-- In the P5 load test the API gateway gave up on a long publish_run (HTTP
-- 504 after 60 s) while Postgres carried on and committed the day. The
-- client saw a failure for a run that had succeeded. publish_run is one
-- transaction under an advisory lock, so after a lost response the client
-- asks this RPC, by load id, until it has a definite answer:
--   published  the run is complete and was written by this load
--   running    a publish_run holds the publisher lock (this one or a queued one)
--   failed     nothing holds the lock, and this load's chunks are still
--              staged: publish_run rolled back
--   unknown    none of the above (the load was purged, or another load
--              replaced the date since)

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE FUNCTION pipeline.publish_outcome(p_load_id uuid, p_run_date date)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_run record;
  -- pg_advisory_xact_lock(bigint) shows in pg_locks as classid = high 32
  -- bits, objid = low 32 bits, objsubid = 1.
  v_key bigint := pg_catalog.hashtext('core.publish_run')::bigint;
BEGIN
  SELECT r.status, r.n_rows, r.meta -> '_publish' AS pub INTO v_run FROM core.runs r WHERE r.run_date = p_run_date;
  IF FOUND AND v_run.status = 'complete' AND v_run.pub ->> 'load_id' = p_load_id::text THEN
    RETURN jsonb_build_object('state', 'published', 'rows', v_run.n_rows, 'publish', v_run.pub);
  END IF;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_locks l
              WHERE l.locktype = 'advisory' AND l.granted
                AND l.database = (SELECT oid FROM pg_catalog.pg_database WHERE datname = pg_catalog.current_database())
                AND l.classid = ((v_key >> 32) & 4294967295)::oid
                AND l.objid = (v_key & 4294967295)::oid AND l.objsubid = 1) THEN
    RETURN jsonb_build_object('state', 'running');
  END IF;
  IF EXISTS (SELECT 1 FROM internal.load_chunks c WHERE c.load_id = p_load_id) THEN
    RETURN jsonb_build_object('state', 'failed');
  END IF;
  RETURN jsonb_build_object('state', 'unknown');
END
$$;

REVOKE ALL ON FUNCTION pipeline.publish_outcome(uuid, date) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION pipeline.publish_outcome(uuid, date) TO service_role, pipeline_writer;
