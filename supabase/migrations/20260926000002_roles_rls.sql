-- Roles, grants and row-level security (design/supabase-migration.md, P1).
--
-- Nothing public reaches Postgres (amendment A4): anon and authenticated get
-- no access to core, internal or pipeline. Cloud access goes through
-- service_role-only RPCs in `pipeline` (A1, added in P2).
--
-- pipeline_writer / pipeline_reader are NOLOGIN group roles for direct
-- connections from dev machines, CI and admin work. Login users and their
-- passwords are created by hand per project, never in a migration. Role
-- settings are NOT inherited from a group role, so set the timeouts on each
-- login user (see design/supabase-migration.md, "Roles and timeouts"):
--
--   CREATE ROLE nightly_writer LOGIN PASSWORD '...' IN ROLE pipeline_writer;
--   ALTER ROLE nightly_writer SET lock_timeout = '5s';
--   ALTER ROLE nightly_writer SET statement_timeout = '10min';

-- Fail fast rather than queue behind a lock held by readers.
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pipeline_writer') THEN
    CREATE ROLE pipeline_writer NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pipeline_reader') THEN
    CREATE ROLE pipeline_reader NOLOGIN;
  END IF;
END
$$;

-- Nobody but the owner and the grants below.
REVOKE ALL ON SCHEMA core, internal, pipeline FROM PUBLIC, anon, authenticated;
REVOKE ALL ON ALL TABLES IN SCHEMA core FROM PUBLIC, anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA core REVOKE ALL ON TABLES FROM PUBLIC, anon, authenticated;
-- Postgres grants EXECUTE on every new function to PUBLIC, and a schema-scoped
-- ALTER DEFAULT PRIVILEGES cannot take that away. So every function created in
-- `internal` or `pipeline` must be followed by
--   REVOKE ALL ON FUNCTION ... FROM PUBLIC, anon, authenticated;
--   GRANT EXECUTE ON FUNCTION ... TO service_role;   -- pipeline RPCs only
-- tests/test_db_schema.py fails on any function that skips it. The schema
-- USAGE revoked above is the second lock: without it no role can reach them.

GRANT USAGE ON SCHEMA core TO pipeline_writer, pipeline_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA core TO pipeline_writer, pipeline_reader;
GRANT INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA core TO pipeline_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA core GRANT SELECT ON TABLES TO pipeline_writer, pipeline_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA core GRANT INSERT, UPDATE, DELETE ON TABLES TO pipeline_writer;

GRANT USAGE ON SCHEMA pipeline TO service_role;

-- RLS on every core table, partitions included (a partition queried directly
-- does not apply its parent's policies). Only the pipeline roles have
-- policies; anon/authenticated have none. The table owner bypasses RLS.
DO $$
DECLARE
  t record;
BEGIN
  FOR t IN
    SELECT c.relname
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'core' AND c.relkind IN ('r', 'p')
  LOOP
    EXECUTE format('ALTER TABLE core.%I ENABLE ROW LEVEL SECURITY', t.relname);
    EXECUTE format('CREATE POLICY pipeline_read ON core.%I FOR SELECT TO pipeline_writer, pipeline_reader USING (true)',
                   t.relname);
    EXECUTE format('CREATE POLICY pipeline_write ON core.%I FOR ALL TO pipeline_writer USING (true) WITH CHECK (true)',
                   t.relname);
  END LOOP;
END
$$;
