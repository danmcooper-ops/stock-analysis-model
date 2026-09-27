# The report behind a login (Cloudflare Access + Worker + R2)

The nightly report is served from Cloudflare, not GitHub Pages, so it can sit
behind a login:

```
browser ──► Cloudflare Access: email one-time code, allowlisted emails only
              └─► Worker "stock-report"  (stock-report.<sub>.workers.dev)
                     └─► R2 bucket "stock-report": index.html, *.json, px/, vol/
nightly step 08 ── scripts/publish_report.py ──► R2, then verifies with a service token
```

* **Access** does the login. It is free for up to 50 users, and nobody needs an
  account: people type their email, get a code and stay signed in for the
  session length.
* **The Worker** (`worker/src/index.js`) serves the bucket. It also re-checks the
  Access JWT itself, so if the Access application is ever removed the site
  returns 403 (or 500 if unconfigured) instead of going public.
* **R2, not Cloudflare Pages:** Pages and Workers static assets cap each file at
  25 MiB, and `details.json` (~32 MB) and `index.html` (~25 MB) are both past
  it. R2 has no practical per-file cap.
* **`scripts/publish_report.py`** uploads only changed files (it compares MD5
  against the R2 ETag), uploads `index.html` last, and deletes stale keys
  (dropped tickers' shards) after the new page is live.

## One-time setup

Everything here happens in the Cloudflare dashboard (dash.cloudflare.com)
except step 3.

1. **R2 bucket.** Go to R2 Object Storage → Create bucket → name it
   `stock-report` (keep the default location, with no public access and no
   custom domain on the bucket).
2. **R2 API token.** Go to R2 → Manage API tokens → Create → *Object Read &
   Write*, restricted to the `stock-report` bucket. Note the **Access Key ID**,
   the **Secret Access Key** and your **Account ID** (shown on the R2
   overview).
3. **Deploy the Worker** from this directory:
   ```bash
   cd cloudflare/worker
   npx wrangler login          # opens a browser once
   npx wrangler deploy
   ```
   It prints `https://stock-report.<sub>.workers.dev`. Until step 4 is done
   every request answers 500 ("Access is not configured"), which is expected
   and fails closed.
4. **Turn on Access.** Go to Workers & Pages → `stock-report` → Settings →
   Domains & Routes → the `workers.dev` row → **Enable Cloudflare Access**.
   (The first time, Zero Trust asks you to pick a team name, which gives
   `<team>.cloudflareaccess.com`, and the Free plan.) Then go to Zero Trust →
   Access → Applications → the new app → edit:
   * **Policies:** replace the default with an *Allow* policy whose *Include*
     rule is *Emails* = you and each person you invite. Add or remove people
     here at any time; no redeploy is needed.
   * **Login methods:** *One-time PIN* only.
   * **Session duration:** 30 days (or whatever you prefer).
   * Copy the **Application Audience (AUD) Tag** from the app's Overview tab.

   Put it and the team domain into `worker/wrangler.toml`:
   ```toml
   ACCESS_TEAM_DOMAIN = "<team>.cloudflareaccess.com"
   ACCESS_AUD = "<the AUD tag>"
   ```
   and run `npx wrangler deploy` again. Commit the change; neither value is a
   secret.
5. **Service token for the nightly check.** Go to Zero Trust → Access → Service
   credentials → Service Tokens → Create → name `nightly-publish`. Copy the
   **Client ID** and **Client Secret** (the secret is shown once). On the Access
   app, add a second policy with Action *Service Auth* and Include *Service
   Token* = `nightly-publish`.
6. **Secrets for the nightly run.** Add these to the cloud Routine's
   environment (and to `.env` for the dormant Mac pipeline):
   ```
   R2_ACCOUNT_ID=...          # step 2
   R2_ACCESS_KEY_ID=...       # step 2
   R2_SECRET_ACCESS_KEY=...   # step 2
   REPORT_URL=https://stock-report.<sub>.workers.dev/
   CF_ACCESS_CLIENT_ID=...    # step 5
   CF_ACCESS_CLIENT_SECRET=...# step 5
   ```
   The cloud environment's network policy must allow
   `<account>.r2.cloudflarestorage.com` and the workers.dev host.
7. **First upload, by hand**, from a checkout that has a rendered report in
   `output/`:
   ```bash
   mkdir -p /tmp/site && cp output/stock_analysis_results_<date>.html /tmp/site/index.html
   cp output/{prices_meta,hist,details}.json /tmp/site/; cp output/macro.json /tmp/site/ 2>/dev/null
   PAGES_DOCS=/tmp/site python scripts/publish_vol_shards.py
   python scripts/publish_report.py /tmp/site --rundate <date> --dry-run   # plan
   python scripts/publish_report.py /tmp/site --rundate <date>             # upload + verify
   ```

Once the `R2_*` variables are present, nightly step 08 publishes to R2. Until
then it keeps force-pushing the old public `pages-live` branch, so the report
never stops updating mid-migration.

## Checks

```bash
curl -sI https://stock-report.<sub>.workers.dev/                # 302 → <team>.cloudflareaccess.com
curl -sI https://stock-report.<sub>.workers.dev/details.json    # 302 as well, never data
curl -s  -H "CF-Access-Client-Id: $CF_ACCESS_CLIENT_ID" \
         -H "CF-Access-Client-Secret: $CF_ACCESS_CLIENT_SECRET" \
         https://stock-report.<sub>.workers.dev/ | grep -o '20[0-9-]\{8\}' | head -1
```

Then, in a browser: sign in with an allowlisted email and open a ticker popup
(details), a price chart (`px/`), volume (`vol/`), the history views and the
Macro tab. An email that isn't on the list should get no code.

Worker unit tests (Node 20+): `cd cloudflare/worker && npm test`.

## Retiring the public site, then going private

Do this only after a nightly run shows `live: … serves the <date> report` in
`logs/08-publish.log`:

1. Delete `.github/workflows/deploy-pages.yml` on main, and remove the
   `publish_pages_legacy` fallback from `scheduled-tasks/cloud-daily-stock-analysis/run.sh`
   (and the legacy branch of `publish()` in `scripts/run_daily.sh`).
2. Repo Settings → Pages → unpublish, then delete the `pages-live` branch.
3. Settings → General → Danger Zone → **Change visibility → Private**. The
   cloud Routine keeps working (it uses the Claude GitHub App), and so does
   `daily-eod.yml` (it passes `GITHUB_TOKEN`). Actions minutes start counting
   against the plan's free allowance. Anything cloned or forked while the repo
   was public stays out there; check the forks list.

## Later: a custom domain

Add the domain to Cloudflare, then add it under the Worker's Settings → Domains
& Routes → Custom domain, and add that hostname to the same Access
application. Update `REPORT_URL`. No code changes are needed.
