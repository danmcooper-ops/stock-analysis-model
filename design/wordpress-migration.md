# Plan: move the private report from Cloudflare to WordPress (citytosummit.com)

Status: proposal, 2026-10-11. Nothing here is built yet.

## What Cloudflare does today

Step `08b-publish-cloudflare` (`scripts/run_daily.sh` `publish_cloudflare`,
and the dormant cloud copy in `scheduled-tasks/cloud-daily-stock-analysis/run.sh`)
does four jobs. Each needs a WordPress replacement:

| Job | Cloudflare piece | WordPress replacement |
|---|---|---|
| Host ~2,300 static files / ~84 MiB a night (`docs/`: `index.html`, `vol/`, `px/`, `hist/`, `details/`, manifests) | Pages, `wrangler pages deploy` | Files on the WordPress host, **outside the web root**, uploaded by rsync over SSH |
| Login (allowlisted emails, one-time code) | Zero Trust Access | WordPress user accounts with a custom capability |
| Fail closed: no valid login → no file | `cloudflare/pages/_worker.js` re-checks the Access JWT | A small plugin that is the *only* path to the files |
| Prove it after each deploy | anonymous request must not be 200; service-token request must show today's date | Same two checks, against the plugin, with a deploy token |

Atomicity also matters: a `px/` shard is an offset into `prices_meta.json`'s
dates axis, so a visitor must never get files from two different nights.
Pages gave that for free (each deploy is immutable). Here it comes from
uploading each night into its own release directory and flipping a pointer.

## Target design

```
browser ─► https://www.citytosummit.com/report/...
             └─► WordPress (plugin cts-report-gate)
                   ├─ not logged in            → 302 to wp-login.php?redirect_to=/report/...
                   ├─ logged in, no capability → 403
                   └─ allowed → stream file from
                        ~/report-releases/<current>/<path>   (outside public_html)
nightly 08b: rsync docs/ → releases/<date>.partial → rename → write `current`
             → prune to 3 releases → anonymous check (must not be 200)
             → token check (must show <date>)
```

### 1. The plugin: `wordpress/cts-report-gate/` (new, in this repo)

One PHP file plus a readme, installed as a normal plugin (or `mu-plugins/` so
it cannot be deactivated from the dashboard — recommended).

- **Route:** a rewrite rule for `^report(/.*)?$` handled on `template_redirect`
  (or `init` early), so no theme renders. `/report` redirects to `/report/`
  so the report's relative sidecar paths resolve.
- **Auth:** `is_user_logged_in()` and `current_user_can('view_stock_report')`.
  The plugin adds the capability to `administrator` on activation; viewers get
  a dedicated `report_viewer` role (read + `view_stock_report`), so ordinary
  subscribers/commenters on the site do *not* see the report.
- **Path safety:** resolve `releases/<current>/<path>` with `realpath()` and
  refuse anything that does not stay under the release directory (`..`,
  symlinks, NUL). Only a whitelist of extensions (`html json js css svg png
  woff2`), with the matching `Content-Type`.
- **Fail closed:** missing `REPORT_ROOT` constant, missing/garbled `current`
  pointer, or a release directory that does not exist → 503, never a
  directory listing or a fallback.
- **Caching:** every response sends `Cache-Control: private, no-cache` and
  `Vary: Cookie`, plus `X-LiteSpeed-Cache-Control: no-cache` / `nocache_headers()`
  so host page caches (Varnish, LiteSpeed, WP Rocket, Jetpack/WordPress.com
  edge) never store a gated response and replay it to someone else. The
  verification step below checks this explicitly. `ETag` = release id + path,
  so a revisit within one night gets 304s.
- **Compression:** the 9 MiB `index.html` and the ~8 MiB `details/` parts went
  out compressed from Pages; PHP `readfile` will not be. Stage time writes a
  `.gz` beside each `.html`/`.json`; the plugin serves it with
  `Content-Encoding: gzip` when the browser accepts gzip. (`zlib.output_compression`
  is the fallback if the host allows it.)
- **Deploy check token:** requests carrying `X-Report-Token` are let through
  when `hash_equals(REPORT_CHECK_TOKEN_SHA256, hash('sha256', $token))`. Only
  the hash lives in `wp-config.php`; the token lives in the nightly `.env`.
  This replaces the Access service token.
- **Config in `wp-config.php`:** `REPORT_ROOT` (absolute path to
  `report-releases/`), `REPORT_CHECK_TOKEN_SHA256`.

Optional, later: a "Report" menu item / login page styling, and a
"passwordless" login plugin (magic link by email) if you want the
one-time-code feel of Access instead of passwords.

### 2. Deploy: `scripts/publish_wordpress.sh` (or a function in `run_daily.sh`)

Replaces `publish_cloudflare` as step `08b-publish-wordpress`, same
non-blocking (`soft`) slot, same refusal rules.

1. Refuse unless `WP_SSH_HOST`, `WP_SSH_USER`, `WP_REPORT_ROOT`, `WP_REPORT_URL`
   and `WP_REPORT_CHECK_TOKEN` are set (mirrors "refusing to deploy" today).
2. `scripts/stage_wordpress.py docs/ <stage>`: copy, write `.gz` siblings,
   write `release.json` (`{run_date, files, bytes, sha}`).
3. `rsync -a --delete --link-dest=../<current> <stage>/ host:$ROOT/releases/<date>.partial/`
   — `--link-dest` hard-links files unchanged since last night (most of
   `hist/`, many `vol/` shards), so the nightly transfer is the delta, not 84 MiB.
4. Over SSH, one command: `mv <date>.partial <date> && printf <date> > current.tmp && mv current.tmp current`
   (atomic rename), then keep the newest 3 releases. A re-run of the same date
   uploads to `<date>.partial` again and replaces `<date>`.
5. Anonymous `curl $WP_REPORT_URL`: 200 fails the step ("serves the report
   WITHOUT a login"); 302/401/403 pass.
6. Token check: `curl` with `X-Report-Token` (passed via `-K -` on stdin, never
   argv, as today) must contain `$RUNDATE`; retried briefly, no 5-minute wait
   needed since there is no CDN propagation.

SSH auth: a dedicated ed25519 key on the Mac mini, used only for this, with
the server-side `authorized_keys` entry restricted to rsync into
`report-releases/` (`command=`/`rrsync`) if the host permits it.

The coverage-floor gate (`DEGRADED=1` skips publish) applies unchanged.

### 3. Keep `check_pages_limits.py`?

Its limits (25 MiB / 20,000 files) are Cloudflare's. Keep it running during the
parallel period, then either delete it or retune to the host's real limits
(inode quota, disk quota — shared hosts often cap inodes at ~250k, and 3
releases × ~2,300 files, hard-linked, is fine). Recommendation: retune to a
disk/inode budget for the host and keep the 80% warning.

## Decisions needed before building

1. **What kind of WordPress hosting is citytosummit.com on?** This decides
   whether the plan works as written.
   - *Self-hosted / managed host with SSH* (SiteGround, Bluehost, WP Engine,
     Kinsta, a VPS…): plan works as written.
   - *WordPress.com*: needs the **Business/Creator** plan for custom plugins
     and SFTP/SSH; their edge cache must be confirmed to honour
     `Cache-Control: private`. On Personal/Premium this design is not possible
     — the fallback would be hosting the files elsewhere (e.g. the Supabase
     Storage bucket, private, with signed URLs minted by the plugin) and using
     WordPress only for the login.
   - *Host without SSH, SFTP only*: rsync becomes `lftp mirror`; the pointer
     flip still works via an SFTP rename.
2. **URL:** `www.citytosummit.com/report/` (one site, one login) vs a subdomain
   such as `stocks.citytosummit.com` (separate WordPress or a static vhost).
   Recommendation: `/report/` on the existing site.
3. **Who logs in:** WordPress accounts with passwords (+ optionally a 2FA
   plugin), or magic-link email login to match Access's one-time code.
4. **GitHub Pages:** step `08` still pushes the same `docs/` to
   `pages-live`, served publicly at `danmcooper-ops.github.io/stock-analysis-model/`.
   If the point of the login is that the report is private, that copy defeats
   it today and would still defeat it after this move. Decide whether to stop
   publishing it (make 08 build `docs/` only) as part of this work.
5. **DNS:** if citytosummit.com's DNS currently sits on Cloudflare, it can stay
   there (DNS only, proxy off for `/report` or entirely) — this plan does not
   require moving it. Only the Pages project and the Access app are retired.

## Phases

**W0 — confirm the host (½ day).** Answer decision 1. SSH in, note PHP
version, disk and inode quota, whether a page cache sits in front of
WordPress, and whether a path outside `public_html` is writable.

**W1 — plugin (1–2 days).** `wordpress/cts-report-gate/` with tests: PHPUnit
via `wp-env` (Docker) covering anonymous → 302, logged-in-no-cap → 403,
viewer → 200 with correct type, `..`/symlink escape → 404, missing pointer →
503, token header good/bad, gzip negotiation, cache headers. Python test in
`tests/` that the stage script's `.gz` siblings round-trip.

**W2 — deploy step (1 day).** `stage_wordpress.py`, `publish_wordpress`
function in `run_daily.sh` as a *new* step `08c` alongside `08b`; tests in the
style of `tests/test_pages_publish.py`. Add the env vars to `.env` docs.

**W3 — parallel run (≈1 week).** Both 08b (Cloudflare) and 08c (WordPress)
publish nightly. Check each morning: step log OK, the report loads and is
today's, a ticker page opens (`hist/`, `px/` fetched), Summary PDF prints, an
anonymous private-window request gets the login page, and a request made right
after a logged-in one in another browser is still refused (cache check). Time
a cold load vs Cloudflare.

**W4 — cutover and removal (½ day).**
- `run_daily.sh`: drop `publish_cloudflare` and its env checks; rename `08c` →
  `08b-publish-wordpress`. Same in the dormant cloud `run.sh` (or mark that
  step removed in its header).
- Delete `cloudflare/` (README, `_worker.js`, worker tests),
  `scripts/stage_pages_worker.py`, `tests/test_stage_pages_worker.py`, the CF
  parts of `tests/test_pages_publish.py`; retune or delete
  `check_pages_limits.py` (section 3).
- Remove `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CF_PAGES_PROJECT`,
  `CF_PAGES_URL`, `WRANGLER_VERSION`, `CF_ACCESS_*` from `.env` and docs.
- Update `CLAUDE.md` (Project Structure `cloudflare/` line, "Report sidecars and
  the Pages size limits", the CF env section), `design/supabase-migration.md`
  (A4/P4c note: superseded), `scheduled-tasks/` runbooks, and
  `tests/load/` k6 target URL.
- In the Cloudflare dashboard: delete the Access application and service
  token, then the Pages project (so `*.pages.dev` stops serving old deploys).
  Revoke the API token.

**Rollback:** until W4's dashboard deletions, re-enabling 08b is a revert of
the run_daily change; the Pages project still holds the last deploy.

## Risks

- **Host page cache serving a gated page anonymously.** The worst failure;
  mitigated by headers, an explicit cache-exclusion rule for `/report/*` in
  the host panel, and the W3 cross-browser check. Worth automating as part of
  the nightly check (anonymous request after the token request).
- **PHP streaming cost.** A first visit pulls ~9 MiB HTML (≈1.5 MiB gzipped)
  plus a few sidecars per view, through PHP. Fine for a handful of viewers on
  any host; not a public-traffic design. If it ever needs to scale, put the
  files on object storage with short-lived signed URLs minted by the plugin.
- **No CDN.** Load time depends on the host's location and bandwidth; W3
  measures it.
- **Login strength drops** from Access's email OTP to WordPress passwords
  unless a 2FA/magic-link plugin is added (decision 3). WordPress is also a
  larger attack surface than Access: keep core/plugins updated, limit login
  attempts.
- **Disk/inode quotas** on shared hosting; hard-linked releases keep it small.
