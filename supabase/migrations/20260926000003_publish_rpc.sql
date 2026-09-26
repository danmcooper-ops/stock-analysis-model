-- Write path (design/supabase-migration.md, P2): stage a run in chunks, then
-- publish it in one validated, atomic transaction.
--
-- The cloud container cannot open raw TCP to Postgres (P0, amendment A1), so
-- both functions are Data API RPCs in the `pipeline` schema, callable only by
-- service_role (and pipeline_writer for direct connections). Each chunk is a
-- separate HTTPS request, so staged rows live in ordinary tables keyed by
-- load_id: two publishers can never see or truncate each other's rows (R6).
--
-- Payload row (built by data/db/publish.py):
--   {"ticker": "AAPL",
--    "cols":   {<registry column>: <value>, ...},   -- non-NULL typed values;
--              NaN/Infinity/-Infinity sent as JSON strings, which float8in reads
--    "extra":  {...} | null,                        -- codec-encoded jsonb
--    "edgar_history_sha": "<sha256>" | null}
-- Numbers stay exact: JSON -> jsonb numeric -> float8in. Nothing here converts
-- float8 -> numeric/jsonb, which would truncate to 15 digits (see P1 notes).

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TABLE internal.load_chunks (
  load_id uuid NOT NULL,
  chunk_no bigint NOT NULL,
  n_rows bigint NOT NULL,
  n_blobs bigint NOT NULL,
  staged_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (load_id, chunk_no)
);

CREATE TABLE internal.load_rows (
  load_id uuid NOT NULL,
  ticker text NOT NULL,
  chunk_no bigint NOT NULL,
  cols jsonb NOT NULL,
  extra jsonb,
  edgar_history_sha text,
  -- a ticker staged twice in one load is a duplicate and fails the chunk
  PRIMARY KEY (load_id, ticker)
);

CREATE TABLE internal.load_blobs (
  load_id uuid NOT NULL,
  sha text NOT NULL CHECK (sha ~ '^[0-9a-f]{64}$'),
  value jsonb NOT NULL,
  PRIMARY KEY (load_id, sha)
);

-- No policies: only the SECURITY DEFINER functions below (owner) touch these.
ALTER TABLE internal.load_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE internal.load_rows ENABLE ROW LEVEL SECURITY;
ALTER TABLE internal.load_blobs ENABLE ROW LEVEL SECURITY;


CREATE FUNCTION pipeline.stage_chunk(p_load_id uuid, p_chunk_no integer, p_rows jsonb,
                                     p_blobs jsonb DEFAULT '[]'::jsonb)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
SET lock_timeout = '5s'
AS $$
DECLARE
  v_rows integer := jsonb_array_length(p_rows);
  v_existing integer;
BEGIN
  -- Idempotent per (load_id, chunk_no): a retried request is a no-op, but a
  -- different payload under the same chunk number is an error.
  SELECT n_rows INTO v_existing FROM internal.load_chunks
   WHERE load_id = p_load_id AND chunk_no = p_chunk_no;
  IF FOUND THEN
    IF v_existing <> v_rows THEN
      RAISE EXCEPTION 'stage_chunk: chunk % of load % re-sent with % rows, staged with %',
        p_chunk_no, p_load_id, v_rows, v_existing;
    END IF;
    RETURN jsonb_build_object('staged', false, 'rows', v_existing);
  END IF;

  IF p_chunk_no = 0 THEN
    -- Loads abandoned for a day (a crashed publisher) are purged.
    DELETE FROM internal.load_rows WHERE load_id IN (
      SELECT load_id FROM internal.load_chunks GROUP BY load_id
      HAVING max(staged_at) < now() - interval '1 day');
    DELETE FROM internal.load_blobs WHERE load_id IN (
      SELECT load_id FROM internal.load_chunks GROUP BY load_id
      HAVING max(staged_at) < now() - interval '1 day');
    DELETE FROM internal.load_chunks WHERE load_id IN (
      SELECT load_id FROM internal.load_chunks GROUP BY load_id
      HAVING max(staged_at) < now() - interval '1 day');
  END IF;

  INSERT INTO internal.load_chunks (load_id, chunk_no, n_rows, n_blobs)
  VALUES (p_load_id, p_chunk_no, v_rows, jsonb_array_length(p_blobs));

  INSERT INTO internal.load_rows (load_id, ticker, chunk_no, cols, extra, edgar_history_sha)
  SELECT p_load_id, r->>'ticker', p_chunk_no, coalesce(r->'cols', '{}'::jsonb),
         nullif(r->'extra', 'null'::jsonb), r->>'edgar_history_sha'
    FROM jsonb_array_elements(p_rows) AS r;

  INSERT INTO internal.load_blobs (load_id, sha, value)
  SELECT p_load_id, b->>'sha', b->'value' FROM jsonb_array_elements(p_blobs) AS b
  ON CONFLICT DO NOTHING;

  RETURN jsonb_build_object('staged', true, 'rows', v_rows);
END
$$;


-- p_run:    {"risk_free_rate", "risk_free_rate_source", "source_sha256",
--            "pipeline_version", "meta": {...}}
-- p_expect: {"n_rows", "n_chunks", "force": bool, "reason": text,
--            "min_row_ratio": 0.7, "client_stats": {...}}
CREATE FUNCTION pipeline.publish_run(p_load_id uuid, p_run_date date, p_run jsonb, p_expect jsonb)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
SET lock_timeout = '5s'
AS $$
DECLARE
  v_force boolean := coalesce((p_expect->>'force')::boolean, false);
  v_n_rows integer;
  v_n_chunks integer;
  v_prev_rows bigint;
  v_min_ratio double precision := coalesce((p_expect->>'min_row_ratio')::double precision, 0.7);
  v_bad text;
  v_old bigint[];
  v_new bigint[];
  v_new_tickers integer;
  v_new_blobs integer;
  v_changes integer;
  v_warnings jsonb := '[]'::jsonb;
BEGIN
  -- One publisher at a time. Wait for a running one rather than fail on the
  -- 5s lock_timeout, then restore it for the table locks below.
  PERFORM set_config('lock_timeout', '15min', true);
  PERFORM pg_advisory_xact_lock(hashtext('core.publish_run'));
  PERFORM set_config('lock_timeout', '5s', true);

  -- Structural checks: never overridable.
  SELECT count(*) INTO v_n_chunks FROM internal.load_chunks WHERE load_id = p_load_id;
  IF v_n_chunks <> (p_expect->>'n_chunks')::integer THEN
    RAISE EXCEPTION 'publish_run: % chunks staged for load %, expected %',
      v_n_chunks, p_load_id, p_expect->>'n_chunks';
  END IF;
  SELECT count(*) INTO v_n_rows FROM internal.load_rows WHERE load_id = p_load_id;
  IF v_n_rows <> (p_expect->>'n_rows')::integer THEN
    RAISE EXCEPTION 'publish_run: % rows staged for load %, expected %', v_n_rows, p_load_id, p_expect->>'n_rows';
  END IF;
  IF v_n_rows = 0 THEN
    RAISE EXCEPTION 'publish_run: load % is empty', p_load_id;
  END IF;
  -- jsonb_populate_record silently ignores unknown keys, so a column the
  -- database does not have (client registry newer than the migrations)
  -- must fail here rather than drop the values.
  SELECT string_agg(DISTINCT k, ', ') INTO v_bad
    FROM internal.load_rows lr, jsonb_object_keys(lr.cols) AS k
   WHERE lr.load_id = p_load_id
     AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_attribute a
                      WHERE a.attrelid = 'core.results'::regclass AND a.attname = k
                        AND a.attnum > 0 AND NOT a.attisdropped);
  IF v_bad IS NOT NULL THEN
    RAISE EXCEPTION 'publish_run: columns not in core.results (apply the migrations first): %', v_bad;
  END IF;
  SELECT string_agg(DISTINCT lr.edgar_history_sha, ', ') INTO v_bad
    FROM internal.load_rows lr
   WHERE lr.load_id = p_load_id AND lr.edgar_history_sha IS NOT NULL
     AND NOT EXISTS (SELECT 1 FROM internal.load_blobs b WHERE b.load_id = p_load_id AND b.sha = lr.edgar_history_sha)
     AND NOT EXISTS (SELECT 1 FROM core.edgar_blobs e WHERE e.sha = lr.edgar_history_sha);
  IF v_bad IS NOT NULL THEN
    RAISE EXCEPTION 'publish_run: edgar blobs referenced but never staged: %', v_bad;
  END IF;

  -- Soft check (R9): a run far smaller than the previous complete one is
  -- more likely a broken night than a real change. --force overrides it,
  -- audited in runs.meta.
  SELECT n_rows INTO v_prev_rows FROM core.runs
   WHERE status = 'complete' AND run_date < p_run_date ORDER BY run_date DESC LIMIT 1;
  IF v_prev_rows IS NOT NULL AND v_n_rows < v_min_ratio * v_prev_rows THEN
    IF NOT v_force THEN
      RAISE EXCEPTION 'publish_run: % rows is under % x the previous complete run (% rows); pass force with a reason to override',
        v_n_rows, v_min_ratio, v_prev_rows;
    END IF;
    v_warnings := v_warnings || jsonb_build_array(format('row count %s < %s of previous %s (forced)',
                                                         v_n_rows, v_min_ratio, v_prev_rows));
  END IF;

  INSERT INTO core.runs AS r (run_date, status, risk_free_rate, risk_free_rate_source, n_rows,
                              source_sha256, pipeline_version, meta, started_at, completed_at)
  VALUES (p_run_date, 'complete', (p_run->>'risk_free_rate')::double precision, p_run->>'risk_free_rate_source',
          v_n_rows, p_run->>'source_sha256', p_run->>'pipeline_version',
          coalesce(p_run->'meta', '{}'::jsonb) || jsonb_build_object('_publish', jsonb_build_object(
            'load_id', p_load_id, 'force', v_force, 'reason', p_expect->'reason',
            'client_stats', p_expect->'client_stats', 'warnings', v_warnings, 'published_at', now())),
          now(), now())
  ON CONFLICT (run_date) DO UPDATE SET
    status = 'complete', risk_free_rate = EXCLUDED.risk_free_rate,
    risk_free_rate_source = EXCLUDED.risk_free_rate_source, n_rows = EXCLUDED.n_rows,
    source_sha256 = EXCLUDED.source_sha256, pipeline_version = EXCLUDED.pipeline_version,
    meta = EXCLUDED.meta, completed_at = now();

  WITH ins AS (
    INSERT INTO core.tickers AS t (ticker, first_seen, last_seen)
    SELECT ticker, p_run_date, p_run_date FROM internal.load_rows WHERE load_id = p_load_id
    ON CONFLICT (ticker) DO UPDATE SET
      first_seen = least(t.first_seen, EXCLUDED.first_seen),
      last_seen = greatest(t.last_seen, EXCLUDED.last_seen)
    RETURNING (xmax = 0) AS inserted)
  SELECT count(*) FILTER (WHERE inserted) INTO v_new_tickers FROM ins;

  WITH ins AS (
    INSERT INTO core.edgar_blobs (sha, value)
    SELECT sha, value FROM internal.load_blobs WHERE load_id = p_load_id
    ON CONFLICT DO NOTHING RETURNING 1)
  SELECT count(*) INTO v_new_blobs FROM ins;

  -- Replace the day. One transaction, so readers see the old day or the new
  -- one, never a mix (R5); no TRUNCATE, no partition swap.
  SELECT coalesce(array_agg(ticker_id), '{}') INTO v_old FROM core.results WHERE run_date = p_run_date;
  DELETE FROM core.results WHERE run_date = p_run_date;
  -- LATERAL, not (jsonb_populate_record(...)).*: the latter calls the
  -- function once per output column (~370x per row; 190 s instead of ~2 s).
  INSERT INTO core.results
  SELECT rec.*
    FROM internal.load_rows lr
    JOIN core.tickers t ON t.ticker = lr.ticker
    CROSS JOIN LATERAL pg_catalog.jsonb_populate_record(NULL::core.results,
            lr.cols || jsonb_build_object('run_date', p_run_date, 'ticker_id', t.ticker_id,
                                          'extra', lr.extra, 'edgar_history_sha', lr.edgar_history_sha)) AS rec
   WHERE lr.load_id = p_load_id;
  SELECT array_agg(t.ticker_id) INTO v_new
    FROM internal.load_rows lr JOIN core.tickers t ON t.ticker = lr.ticker WHERE lr.load_id = p_load_id;

  -- Fresh statistics for the partition just written (R7). A missing
  -- partition already failed the INSERT above.
  EXECUTE format('ANALYZE core.%I', 'results_' || to_char(p_run_date, 'YYYY'));

  -- Rating change points (R4), recomputed for every ticker at this date
  -- (old or new) from its last change point before the date, over complete
  -- runs, so publishing out of order or re-publishing an older date leaves
  -- exactly what a full recompute would. Same rule as the DuckDB store's
  -- rating_history(): first observation plus every transition, NULL/empty
  -- ratings skipped (a gap with the same rating is not a change).
  DELETE FROM core.rating_changes
   WHERE ticker_id = ANY (v_old || v_new) AND run_date >= p_run_date;
  WITH affected AS (
    SELECT a.ticker_id,
           (SELECT max(rc.run_date) FROM core.rating_changes rc
             WHERE rc.ticker_id = a.ticker_id AND rc.run_date < p_run_date) AS anchor
      FROM unnest(v_old || v_new) AS a(ticker_id)
     GROUP BY a.ticker_id),
  seq AS (
    SELECT r.ticker_id, r.run_date, r.rating,
           lag(r.rating) OVER (PARTITION BY r.ticker_id ORDER BY r.run_date) AS prev
      FROM core.results r
      JOIN affected a ON a.ticker_id = r.ticker_id
      JOIN core.runs ru ON ru.run_date = r.run_date AND ru.status = 'complete'
     WHERE r.rating IS NOT NULL AND r.rating <> ''
       AND r.run_date >= coalesce(a.anchor, '-infinity'::date)),
  ins AS (
    INSERT INTO core.rating_changes (ticker_id, run_date, rating, prev_rating)
    SELECT ticker_id, run_date, rating, prev FROM seq
     WHERE run_date >= p_run_date AND (prev IS NULL OR prev <> rating)
    RETURNING 1)
  SELECT count(*) INTO v_changes FROM ins;

  -- Latest pointer (R5): only ever moves forward. A ticker that left this
  -- date on a re-publish falls back to its newest remaining complete run.
  INSERT INTO core.latest_results AS l (ticker_id, run_date)
  SELECT unnest(v_new), p_run_date
  ON CONFLICT (ticker_id) DO UPDATE SET run_date = EXCLUDED.run_date
   WHERE l.run_date <= EXCLUDED.run_date;
  UPDATE core.latest_results l SET run_date = sub.run_date
    FROM (SELECT gone.ticker_id,
                 (SELECT max(r.run_date) FROM core.results r
                    JOIN core.runs ru ON ru.run_date = r.run_date AND ru.status = 'complete'
                   WHERE r.ticker_id = gone.ticker_id) AS run_date
            FROM unnest(v_old) AS gone(ticker_id)
           WHERE gone.ticker_id <> ALL (v_new)) sub
   WHERE l.ticker_id = sub.ticker_id AND l.run_date = p_run_date AND sub.run_date IS NOT NULL;
  DELETE FROM core.latest_results l
   WHERE l.run_date = p_run_date AND l.ticker_id <> ALL (v_new);

  DELETE FROM internal.load_rows WHERE load_id = p_load_id;
  DELETE FROM internal.load_blobs WHERE load_id = p_load_id;
  DELETE FROM internal.load_chunks WHERE load_id = p_load_id;

  RETURN jsonb_build_object('run_date', p_run_date, 'rows', v_n_rows, 'replaced_rows', cardinality(v_old),
                            'new_tickers', v_new_tickers, 'new_blobs', v_new_blobs,
                            'rating_changes', v_changes, 'warnings', v_warnings);
END
$$;

-- New functions are executable by PUBLIC unless revoked (see the roles
-- migration); tests/test_db_schema.py fails on any function that skips this.
REVOKE ALL ON FUNCTION pipeline.stage_chunk(uuid, integer, jsonb, jsonb) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION pipeline.publish_run(uuid, date, jsonb, jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION pipeline.stage_chunk(uuid, integer, jsonb, jsonb) TO service_role, pipeline_writer;
GRANT EXECUTE ON FUNCTION pipeline.publish_run(uuid, date, jsonb, jsonb) TO service_role, pipeline_writer;
GRANT USAGE ON SCHEMA pipeline TO pipeline_writer;
