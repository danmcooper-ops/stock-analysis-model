<!-- Build guide for the iPhone/iPad app. Part A (the app feed) is implemented; Part B is the Xcode work; Part C is the data-licence decision that gates inviting anyone else. -->

# Plan: native iPhone/iPad app for the stock analysis model

## Context

The model runs nightly on the Mac mini (Python, ~5 h, SEC + Yahoo). It
publishes an HTML report to Cloudflare Pages behind Cloudflare Access. You want
a native Xcode app that mirrors it, so you can start building at once. Decisions
taken:

- **iPhone + iPad**, one SwiftUI target. A Mac version can come later via
  "Designed for iPad".
- **The app views the nightly output; it never re-runs the models.** It shows
  what the pipeline computed, so it cannot disagree with the report.
- **Audience: you first, then a few invited people (≤ 50, the Access free
  tier).** Each person signs in with their own email through the existing
  Access login (allowlist + one-time code). No shared secret ships in the app,
  and removing an email revokes that person.
- **Data comes from the Cloudflare site**, matching the database plan's rule
  that nothing outside the pipeline queries Postgres (design/supabase-migration.md,
  "Public traffic").
- **Licensing gates the second stage, not the build.** Today's sources (Yahoo,
  the free Finnhub tier, personal FMP and Tiingo plans) are licensed for
  personal use. Finnhub's terms bar sharing even "derived results". So:
  - **Stage 1, the app for you alone:** needs nothing new.
  - **Stage 2, inviting others:** waits until a commercial data licence is
    signed (Part C) and the feed carries only fields that licence and public
    sources cover.
- **Built to grow.** Login sits behind one `AuthProvider` protocol. Going public
  later (Sign in with Apple via Supabase Auth, the Worker checking each user's
  token) swaps one type and one Worker check.

Most of what the app needs is already published beside `index.html`:
- `hist/<T>.json`;
- `px/<T>.json` + `prices_meta.json`;
- `details/<n>.json` + `details_index.json`;
- `macro.json`, `bonds.json`, `currencies.json`, `paper.json`.

The gap is the **rows** (ratings, scores, fair values, gates) and the
**portfolio definitions**. Both are only inline in `index.html`
(`var DATA=_unpackRows({{ chart_data }})` at `templates/report.html:3939`;
`PF_PUB`). So the work has two halves: a small pipeline change that adds an app
feed (Part A), and the Xcode app (Part B).

---

## Part A — Pipeline: publish an app feed (done 2026-11-01)

Implemented as described below: `_write_app_feed` in `scripts/report_html.py`,
the `app/` copy in `scripts/run_daily.sh`'s publish step, and
`tests/test_app_feed.py`. In addition, the old manifest is deleted before
anything is written, so a failed write leaves no feed rather than a stale one.
A4 and A5 are dashboard and app work and remain to do.

**A1. `scripts/report_html.py`: `_write_app_feed(out_dir, chart_records, portfolios_payload, gate_meta_obj, run_date)`**,
called from `build_html` (2208) right after `_extract_details_payload` (2249).
That call already pops the heavy text fields (`_DETAIL_HEAVY_KEYS`: news,
filings, insiders, profile, …) out of each record into `details/`, so the
records are exactly the inline `DATA` and no second drop list is needed.
It writes to `output/app/`:

- **`rows.json`:** those records, column-packed with the existing `pack_rows`
  (91) after `_sanitize`. That is about the inline payload's ~7.8 MiB (CLAUDE.md
  "Report sidecars"), and Pages compresses it over the wire.
  - Add one `APP_EXCLUDE_KEYS` set (empty in Stage 1). Stage 2 lists every
    field the licence does not cover. It is the single switch for what reaches
    other people.
- **`portfolios.json`:** `portfolios_payload`, the same object as `PF_PUB`
  (from `_load_portfolio_payload`, 2240). It holds no positions:
  `portfolio/positions*` is never in it.
- **`gates.json`:** `gate_meta_obj = gate_metadata()` (`scripts/scoring.py:1308`),
  so the app reads gate names, categories, weights and tooltips instead of
  hard-coding them.
- **`manifest.json`, written last, atomically (`os.replace`):**
  ```json
  {"version": 1, "run_date": "…", "generated_at": "…", "build": "<sha256(rows.json)[:12]>",
   "rows": 4412, "files": {"rows.json": "<sha256>", "…": "…"}}
  ```
  The app polls only this file.

  Add `APP_FEED_VERSION = 1` beside `BOND_REPORT_VERSION` and bump it on any
  shape change; the app refuses a version it does not know.

**A2. Publish.** In `scripts/run_daily.sh`'s publish step (~582, where `macro.json`
and `paper.json` are copied), add `app/` with the same rule: copy it when
`output/app/manifest.json` exists, and otherwise delete `$docs/app` so
yesterday's feed never sits under today's HTML. Steps 08 and 08b then carry it.
The feed adds 4 files, so the 20,000-file Pages limit is unaffected, and
`scripts/check_pages_limits.py` already checks file sizes.

**A3. Test, `tests/test_app_feed.py`.** Render through `build_html` the way
`tests/test_report_data_pack.py` does, and assert:
- `unpack_rows(rows.json)` equals the inline `DATA`;
- the manifest hashes match the files and `build` matches `rows.json`;
- `APP_EXCLUDE_KEYS` removes its fields;
- no output carries position numbers. Reuse the real-book check from
  `tests/test_report_portfolio_ui.py`.

**A4. Access (dashboard, no code).** Set the Access application's session
duration to 30 days, so the app is not asking for a code daily. Stage 2 adds
each invitee's email to the allow policy. The Worker
(`cloudflare/pages/_worker.js:48`) checks `Cf-Access-Jwt-Assertion`, which
Access adds to every signed-in request, cookie or header, so it needs no change.
Document both settings in `cloudflare/README.md`.

**A5. Staleness.** The app shows "as of <run_date>" at all times, and a banner
once `generated_at` is more than 26 h old, so a missed night is visible.
Optionally, ping a free uptime check (healthchecks.io) at the end of step 08b.

Also add one CLAUDE.md entry describing the app feed contract.

---

## Part B — The Xcode app (`ios/` in this repo)

Starter sources for M0–M5 are in `ios/StockModel/` and `ios/StockModelTests/`.
They were written without a Swift toolchain, so expect small fixes on the
first build. `ios/README.md` has the exact Xcode steps. Two departures from
the layout below:
- The Mac mini source follows the live-quotes service's `/` → `/b/<build>/`
  redirect, so the app pins an immutable build.
- `FeedClient` re-checks the live build (at most every 5 minutes) before
  fetching a `px/` shard, and reloads instead of mixing builds.

### Day 1 (about an hour)

1. In Xcode, choose File → New → Project → iOS → App:
   - Name it `StockModel`, with SwiftUI, Swift, Storage None, and tests included.
   - Save it into `stock-analysis-model/ios/`.
   - Set the deployment target to **iOS 17**, for `@Observable`, Swift Charts
     and `NavigationSplitView`.
2. Add `ios/.gitignore` (`xcuserdata/`, `DerivedData/`, `*.xcuserstate`). Use no
   third-party packages: URLSession, Charts, WebKit and Security cover it.
3. Serve real data locally before any login work:
   - Get a published build onto your Mac: copy the mini's `docs/` (or a
     `/b/<build>/` copy, `scripts/live_quotes.py`), or run the pipeline locally.
     After Part A it includes `app/`.
   - Serve it:

     ```bash
     python3 -m http.server 8000 -d docs
     ```

   - The Simulator reaches `http://localhost:8000`.
   - Set `NSAppTransportSecurity › NSAllowsLocalNetworking = YES` in Info.plist.
     That covers localhost and `<mini>.local` with no per-host exceptions.
   - Add `NSLocalNetworkUsageDescription` for the mini on a real phone.
4. Put a trimmed feed (about 20 tickers) plus their `hist/`/`px/` shards into
   `StockModelTests/Fixtures/`. Add the Python `unpack_rows` output of it as
   JSON. These serve as the unit-test input, the parity oracle and the SwiftUI
   preview data.

### Project layout

```
ios/StockModel/
  App/       StockModelApp.swift, AppState (@Observable: feed, selection, settings)
  Auth/      AuthProvider.swift  protocol: authorize(_ req: inout URLRequest), signIn(), signOut()
             AccessAuth.swift    sheet with a WKWebView on the site → Access login
                                 (email + code) → read CF_Authorization from
                                 WKHTTPCookieStore → Keychain → send as the
                                 `cf-access-token` header; a 401/403 or redirect to
                                 *.cloudflareaccess.com = expired → sign in again
             NoAuth.swift        local server / mini
  Feed/      FeedClient.swift    URLSession; manifest first, then files it lists
             FeedCache.swift     Application Support/feed/<build>/; keep 2 builds
             RowTable.swift      Swift port of unpack_rows, COLUMNAR: keeps k/c and
                                 a per-key row index; never builds 4k dictionaries
             Manifest.swift, KeychainStore.swift, JSONValue.swift
  Model/     StockRow.swift      a (table, index) view with typed accessors: ticker,
                                 company_name, sector, industry, price, rating,
                                 rating_prev, rating_chg, _composite_score,
                                 _score_valuation/_quality/_moat/_growth/_ownership,
                                 _fv_effective, _fv_source, mos, mc_p10_fv, mc_p90_fv,
                                 roic, wacc, spread, mcap, pe, ev_ebitda, div_yield,
                                 _gates_passed, _rating_cap_reasons, pf; gates via
                                 gates.json keys
             Rating.swift        BUY / LEAN BUY / HOLD / PASS, colours from report.html
             HistShard, PxShard + PricesMeta (dates axis), DetailsPart, Portfolio, Gate
  Features/  Stocks/     search, rating and sector filters, sort by score / MoS / mcap
             Ticker/     header (price, rating, score, gates); fair-value range
                         (P10 / effective / P90 vs price); gate scorecard by
                         category; price chart vs SPY with an FV line (px/ +
                         prices_meta index_px); 10-year record (hist/); description
                         (details/, fetched on demand)
             Portfolios/ cards → members (rows whose pf contains the id)
             Changes/    rating changes today; buy-line crossings first
             Settings/   source (Local / Mini / Cloudflare), sign in/out, as-of, clear cache
  Shared/    formatters (%, $, compact mcap), RatingPill, MetricRow
```

### Rules the app keeps (from the pipeline's contracts)

- **One build at a time.** `px/` shards are offsets into `prices_meta.json`'s
  `dates` and must never mix across deploys. Key every cached file by
  `manifest.build`. On a new build, fetch the new manifest-level files first and
  per-ticker shards lazily. Never pair an old shard with a new `prices_meta`.
- **N/A is not zero.** A missing key, a null and a non-finite value all mean
  N/A and display as "—", the same semantics as the report's `_num`.
- **Flags, not caps.** `_rating_cap_reasons` reads "Flag" (CLAUDE.md, Rating =
  composite bucket).
- **No on-device scoring.** The rating and score are what the snapshot
  recorded.
- **Secrets.** No credential ships in the app. Each user's Access token lives
  only in their Keychain.
- **Distribution.** TestFlight takes up to 100 external testers. The allowlist
  is the gate, not the build link.
- **Cache retention.** Keep at most 2 builds and delete all cached data on sign-out.
  Vendor terms (Part C) require deleting data when the subscription ends.

### Milestones and exit criteria

| # | Scope | Done when |
|---|-------|-----------|
| M0 | Part A merged; Xcode project builds; `RowTable` decodes the fixture | unit test: every row and key equals the Python `unpack_rows` fixture |
| M1 | FeedClient + cache + Settings against the local server | Simulator shows run date and row count; second launch is offline-capable |
| M2 | Stocks list | full feed (~4.4k rows) decodes in < 1 s and scrolls smoothly on a real iPhone (Instruments) |
| M3 | Ticker page | for 5 tickers (one bank, one foreign filer, one without a DCF), FV, MoS, score, gates and chart match the HTML report |
| M4 | AccessAuth against Cloudflare; staleness banner; `BGAppRefreshTask` (best effort, after ~07:00 ET) | sign-in survives relaunch; an expired session re-prompts; unchanged night = one manifest request |
| M5 | Portfolios + Changes; local watchlist (UserDefaults) | membership matches the report's Portfolios view |
| M6 | iPad split view; TestFlight build | installed on your phone. **Stage 1 done.** |
| M7 | Stage 2: licence signed, `APP_EXCLUDE_KEYS` set, invitees allowlisted | a second email signs in on another phone; removing it locks that phone out at the next sign-in |

Later, not v1:
- Macro, Sectors, ETFs, Bonds and Currencies tabs (their sidecars already exist);
- live quotes;
- a watchlist widget;
- push notifications for Action alerts (`output/portfolio_alerts.json`).

---

## Part C — Data licence for Stage 2 (2026-11-01 review)

**First, GitHub Pages.** While `pages-live` publishes `docs/` from this
public repo, the whole report, now including `app/`, is world-readable there
whatever the app does. Retire it first (`cloudflare/README.md`, "Retiring
GitHub Pages"); Stage 2 cannot be licensed while that copy is public.

What has to be licensed is what the app shows others and what the model reads
from Yahoo:
- **Prices:** history and quotes.
- **Company identity:** sector, industry, market cap and shares.
- **Holders and short interest.**
- **Analyst data:** targets and long-term growth.
- **Earnings dates.**

Statements come from SEC XBRL, macro from FRED, banks from FDIC; all public
data, so no licence is needed. The fields that stay unlicensed under any choice
below go in `APP_EXCLUDE_KEYS`:
- Finnhub peers and supply chain (`data/finnhub_supply_client.py`);
- Google News headlines (`data/news_client.py`);
- SSGA holdings.

| | Intrinio Startup | Twelve Data Venture | FMP | Finnhub | Marketstack |
|---|---|---|---|---|---|
| Price | $333/mo × 6, $666 × 6, then $999 (≈ $6k yr 1, $12k after) | $499/mo (≈ $6k/yr; annual −17%), maybe +$2.4k/yr US add-on | personal $29–139/mo; display via sales | sales | $9.99–149.99/mo |
| Display to app users | **published:** "Commercial Use and Display Rights", "No exchange fees"; needs an Order Form | card says "External display"; terms §2.2(e) want an add-on: **confirm** | needs a Display and Licensing Agreement (unpriced) | written approval required | "Commercial Use" only; display not stated |
| Fit for the model | EOD prices 50+ yrs, adjusted, split ratios, bulk CSV; fundamentals, holders, insiders; 2,500 OTC | prices 1 credit; statistics 50; nightly ≈ 1.5 h at 2,584/min, **6–11 h if the limit is really 610** | best: existing client `data/fmp_client.py`, bulk endpoints | good fundamentals; 30 calls/s cap | prices only |
| Gaps | analyst data and ETF holdings Enterprise only ($1,250+/mo); no short interest; mcap/sector fields unconfirmed; rate limits in Order Form | OTC coverage, history depth, holder costs unknown | price unknown | price unknown | no identity, holders, analysts |
| Watch-outs | §2.3: sending its data to any LLM API counts as redistribution, so keep it out of `build_macro_facts` and out of Claude sessions | §2.3(f) "derivative financial products" | — | data deleted on cancellation | US data comes from Tiingo |

**Recommendation: Intrinio Startup.** It is the only vendor whose display rights
are published at a price, with no exchange fees. It covers prices, identity and
holders, and its year-one cost (≈ $6k) matches Twelve Data's.
- **Without analyst data:** price targets are display-only (`models/market.py`);
  the long-term growth estimate is an optional input to
  `models/ddm.estimate_ddm_growth` and the growth diagnostics, which fall back
  without it. Some DDM fair values move; nothing breaks.
- **Choose Twelve Data instead** if, in writing, Venture's display rights are
  confirmed without the add-on and its limit is 2,584/min.
- **Choose FMP** if its display agreement prices near either, since the client
  already exists.

Before signing, send each finalist the same written question: display to ≤ 50
invited users of a private iOS app showing prices, fundamentals and derived
ratings, with on-device caching. Also ask for rate limits, OTC coverage, and
whether derived ratings count as redistribution.

**Engineering (its own plan after the choice, not part of the app):** add a
vendor client that returns the same yfinance-shaped frames and dicts
`YFinanceClient` does (as `data/fmp_client.py` does), so `models/` and scoring
are untouched.
- Pace by the vendor's limit with `data/throttle.py`, credit-weighted where the
  vendor charges credits.
- Use dividend-and-split-adjusted closes, because the price parquets and the NAV
  ledger assume auto-adjusted prices.
- Run it beside yfinance for a week and diff prices, market caps and sectors
  before switching.

The app needs no change for any of this: it shows whatever the pipeline publishes.

---

## Critical files

- `scripts/report_html.py`: `pack_rows`/`unpack_rows` (91), `_extract_details_payload` (called at 2249), `_load_portfolio_payload` (2240), `build_html` (2208); new `_write_app_feed`, `APP_FEED_VERSION`, `APP_EXCLUDE_KEYS`
- `scripts/scoring.py`: `gate_metadata` (1308), reused as is
- `scripts/run_daily.sh`: publish step (~582), copy or remove `app/`
- `tests/test_app_feed.py` (new)
- `cloudflare/README.md`: session duration, app users
- `CLAUDE.md`: app feed contract entry
- `ios/` (new Xcode project)

## Verification

1. `pytest -m "not network and not slow"` (including `tests/test_app_feed.py`) and `ruff check .`.
2. `scripts/run_daily.sh --dry-run` → `output/app/manifest.json` exists, hashes
   match, and `docs/app/` mirrors it after publish.
3. With no login, `curl -sI "$CF_PAGES_URL/app/manifest.json"` is refused (a
   302 to Access or a 403). Signed in as an allowlisted email in a browser, the
   same URL returns the manifest.
4. Xcode: run the unit tests (RowTable parity, StockRow accessors, the
   N/A rules). Run the Simulator against the local server, then a phone against
   Cloudflare, and check M2's timing and M3's five-ticker comparison against
   the HTML report.
