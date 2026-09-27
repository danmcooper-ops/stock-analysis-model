-- Let the nightly publish outlive the Data API's 8-second statement timeout
-- (design/supabase-migration.md, P5).
--
-- PostgREST connects as `authenticator`, which Supabase configures with
-- statement_timeout = 8s, and an impersonated role without its own setting
-- inherits it. pipeline.publish_run is one transaction by design (all of a
-- day or none of it), so it cannot be split to fit: at 8,000 tickers it
-- takes ~13 s and was cancelled (SQLSTATE 57014) in the P5 scale test. At
-- the 2.5k-row universe of P3 it fit, which is why this only showed at scale.
--
-- PostgREST applies the impersonated role's own statement_timeout, so give
-- service_role (the pipeline's only API role, A1) a ceiling well above the
-- plan's 5-minute publish target. anon and authenticated keep 3 s and 8 s.
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

ALTER ROLE service_role SET statement_timeout = '10min';

-- Make PostgREST re-read role settings now rather than at its next restart.
NOTIFY pgrst, 'reload config';
