# The report behind a login (Cloudflare Access on Cloudflare Pages)

Nightly step `08b-publish-cloudflare` deploys the report to Cloudflare Pages
(design/supabase-migration.md, P4c). This directory keeps that site private:

```
browser ──► Cloudflare Access: email one-time code, allowlisted emails only
              └─► Pages project (<project>.pages.dev)
                     └─► _worker.js: re-checks the Access JWT ─► static files
nightly 08b: stage _worker.js → wrangler pages deploy → anonymous request must be refused
                                                     → signed-in check (service token) sees today's date
```

* **Access** handles the login. It is free for up to 50 users and needs no
  accounts: people type their email, get a code, and stay signed in for the
  session length.
* **`pages/_worker.js`** runs in front of every file (Pages "advanced mode").
  It verifies the JWT Access adds: RS256 against the team's keys, plus
  audience, issuer and expiry. If the Access application is ever deleted,
  disabled, or doesn't cover a hostname (such as the per-deploy
  `<hash>.<project>.pages.dev` URLs), the site answers 403 instead of serving
  the report. Placeholders left unfilled give a 500.
* **`scripts/stage_pages_worker.py`** copies the Worker into the deploy
  directory with the team domain and AUD tag filled in and validated. If they
  are missing or malformed, step 08b **refuses to deploy**, so the report is
  never published without its login.
* After deploying, step 08b checks that an anonymous request is refused (a
  200 fails the step as "serves the report WITHOUT a login"). It then signs
  in with the service token and looks for today's date.

## One-time setup

Do this before the Pages secrets (`CLOUDFLARE_API_TOKEN`,
`CLOUDFLARE_ACCOUNT_ID`, `CF_PAGES_PROJECT`) go into the Routine's
environment, or at the same time. Everything is in the Cloudflare dashboard.

1. **Pages project.** Follow the P4c runbook in design/supabase-migration.md,
   steps 1–2. That covers the Direct Upload project (say `stock-analysis`) and
   an API token with *Cloudflare Pages → Edit*.
2. **Zero Trust team.** Open Zero Trust from the dashboard sidebar. The first
   time, choose a team name, which gives `<team>.cloudflareaccess.com`, and
   the Free plan.
3. **Access application.** Go to Zero Trust → Access → Applications → Add an
   application → *Self-hosted*.
   * **Name:** `Stock report`.
   * **Public hostnames:** add both `stock-analysis.pages.dev` and
     `*.stock-analysis.pages.dev`, using your project name. The wildcard
     covers the per-deploy preview URLs; the Worker would refuse them anyway,
     but this way people get the login page instead of a bare 403.
   * **Session duration:** 30 days, or whatever you prefer.
   * **Login methods:** *One-time PIN* only.
   * **Policy 1, "Invited":** Action *Allow*, Include → *Emails* → you and each
     person you invite. You can edit this list at any time, with no deploy.
   * Save, then copy the **Application Audience (AUD) Tag** from the app's
     Overview (64 hex characters).
4. **Service token for the nightly check.** Go to Access → Service
   credentials → Service Tokens → Create, name it `nightly-publish`, and copy
   the Client ID and Client Secret (the secret is shown only once). On the
   application, add **Policy 2**: Action *Service Auth*, Include → *Service
   Token* → `nightly-publish`.
5. **Routine environment.** Add these next to the Pages secrets:
   ```
   CF_ACCESS_TEAM_DOMAIN=<team>.cloudflareaccess.com
   CF_ACCESS_AUD=<the 64-hex AUD tag>
   CF_ACCESS_CLIENT_ID=<service token client id>
   CF_ACCESS_CLIENT_SECRET=<service token client secret>
   ```
   The first two aren't secrets, since they only name the team and app. The
   last two are.
6. **The next nightly run deploys.** In `logs/08b-publish-cloudflare.log`,
   expect `stage_pages_worker: wrote …`, then `anonymous request refused as
   expected (302)`, then `live: … serves the <date> report`.

## Checks by hand

```bash
URL=https://stock-analysis.pages.dev/
curl -sI $URL | head -1                       # 302 → <team>.cloudflareaccess.com
curl -sI ${URL}details_index.json | head -1   # 302 as well, never data
printf 'header = "CF-Access-Client-Id: %s"\nheader = "CF-Access-Client-Secret: %s"\n' \
    "$CF_ACCESS_CLIENT_ID" "$CF_ACCESS_CLIENT_SECRET" | curl -s -K - $URL | grep -o '20[0-9-]\{8\}' | head -1
```

Then, in a browser: sign in with an allowlisted email and open a ticker popup,
a price chart, the history views and the Macro tab. An email that isn't on
the list gets no code.

Tests: `cd cloudflare/pages && npm test` (Worker, Node 20+) and
`pytest tests/test_stage_pages_worker.py`.

## Retiring GitHub Pages, then making the repo private

GitHub Pages (`pages-live`) is still the public primary site until it is
retired, and the repo is public, so the report stays readable there until you
do this. Once 08b has been green for a few nights:

1. Follow the P4c runbook's "Retiring GitHub Pages" step. Make 08b blocking,
   drop the `pages-live` push from step 08, delete
   `.github/workflows/deploy-pages.yml`, unpublish Pages in the repo settings,
   and delete the `pages-live` branch.
2. Go to Settings → General → Danger Zone → **Change visibility → Private**.
   * The cloud Routine keeps working, because it uses the Claude GitHub App.
   * `daily-eod.yml` keeps working, because it passes `GITHUB_TOKEN`.
   * Actions minutes start counting against your plan's allowance.
   * Anything cloned or forked while the repo was public stays out there, so
     check the forks list.

## Later: a custom domain

Attach the domain under the Pages project's Custom domains tab, add the same
hostname to the Access application, and set `CF_PAGES_URL`. No code changes
are needed.
