-- The cutover readiness log (design/supabase-migration.md, P6).
--
-- P6 makes the database publish (run.sh 06a) blocking once 20 nightly runs
-- in a row have published green with rating-history parity. Each night,
-- step 07e records its verdict here through pipeline.record_night_check;
-- scripts/db_night_check.py reads them back and counts the streak over
-- trading days, so a night that never recorded (the database unreachable,
-- the run dead) breaks it. The flip itself stays manual (DB_PRIMARY=1).

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TABLE core.night_checks (
  run_date date PRIMARY KEY,
  publish_rc bigint NOT NULL,           -- step 06a's exit code (0 = published)
  db_check_ok boolean NOT NULL,         -- the run is complete, row count and source SHA match
  parity_ok boolean,                    -- rating history equals the JSON cache (NULL: not checked)
  details jsonb COMPRESSION lz4 NOT NULL DEFAULT '{}'::jsonb,
  recorded_at timestamptz NOT NULL DEFAULT now()
);

-- Same posture as every core table (migration 2): RLS on, readable and
-- writable only by the pipeline roles; the Data API reaches it only
-- through the RPCs below.
ALTER TABLE core.night_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY pipeline_read ON core.night_checks FOR SELECT TO pipeline_writer, pipeline_reader USING (true);
CREATE POLICY pipeline_write ON core.night_checks FOR ALL TO pipeline_writer USING (true) WITH CHECK (true);
REVOKE ALL ON core.night_checks FROM PUBLIC, anon, authenticated;

-- Record (or re-record, on a re-run of the night) one night's verdict.
CREATE FUNCTION pipeline.record_night_check(p_run_date date, p_publish_rc integer, p_db_check_ok boolean,
                                            p_parity_ok boolean, p_details jsonb)
RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = ''
AS $$
  INSERT INTO core.night_checks AS n (run_date, publish_rc, db_check_ok, parity_ok, details, recorded_at)
  VALUES (p_run_date, p_publish_rc, p_db_check_ok, p_parity_ok, coalesce(p_details, '{}'::jsonb), now())
  ON CONFLICT (run_date) DO UPDATE SET
    publish_rc = EXCLUDED.publish_rc, db_check_ok = EXCLUDED.db_check_ok, parity_ok = EXCLUDED.parity_ok,
    details = EXCLUDED.details, recorded_at = now()
  RETURNING jsonb_build_object('run_date', n.run_date, 'green',
                               n.publish_rc = 0 AND n.db_check_ok AND coalesce(n.parity_ok, false))
$$;

-- Every recorded night from p_since on, oldest first.
CREATE FUNCTION pipeline.night_checks(p_since date DEFAULT NULL)
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT coalesce(jsonb_agg(jsonb_build_object(
           'run_date', n.run_date, 'publish_rc', n.publish_rc, 'db_check_ok', n.db_check_ok,
           'parity_ok', n.parity_ok, 'details', n.details, 'recorded_at', n.recorded_at)
         ORDER BY n.run_date), '[]'::jsonb)
    FROM core.night_checks n
   WHERE p_since IS NULL OR n.run_date >= p_since
$$;

REVOKE ALL ON FUNCTION pipeline.record_night_check(date, integer, boolean, boolean, jsonb)
  FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION pipeline.night_checks(date) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION pipeline.record_night_check(date, integer, boolean, boolean, jsonb)
  TO service_role, pipeline_writer;
GRANT EXECUTE ON FUNCTION pipeline.night_checks(date) TO service_role, pipeline_writer, pipeline_reader;
