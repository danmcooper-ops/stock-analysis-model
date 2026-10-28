# World Currency Screener

Ranks 30 world currencies rich-to-cheap across five weighted pillars, and
renders the result as a single self-contained HTML page.

Built as a currency analogue of
[stock-analysis-model](https://github.com/danmcooper-ops/stock-analysis-model):
Python computes everything offline, a Jinja2 template inlines the dataset as a
JavaScript array, and GitHub Actions publishes the page to GitHub Pages daily.
No build step, no CDN, no API keys.

```bash
pip install -r requirements.txt
python scripts/analyze_currencies.py      # writes output/
```

## The model

| Pillar | Weight | What it asks | Gates |
|---|---|---|---|
| **Valuation** | 0.30 | Is the real exchange rate cheap against its own history? | REER z-score vs 20y (double weight), REER vs 10y, REER vs 5y |
| **Carry** | 0.25 | What does holding it pay, net of inflation? | Real carry (double weight), nominal carry, real policy rate, carry/vol |
| **External** | 0.20 | Can the country fund itself without depreciating? | Current account, reserves cover, inflation gap, GDP growth |
| **Momentum** | 0.15 | Has the repricing started? | 12m-1m, 3m return, vs 200d MA, REER trend |
| **Stability** | 0.10 | What does the position cost in risk? | Volatility, max drawdown, vol regime, inflation vol |

Nineteen gates in all. Each one is a single entry in `GATES`
(`scripts/scoring.py`) carrying its own threshold, scoring curve, weight and
applicability rule — so the pass/fail matrix, the 0–100 score and the report's
columns can never drift apart. Adding a gate adds a column to the UI with no
template edit.

Ratings are **LONG / LEAN LONG / NEUTRAL / AVOID**, assigned from the
composite and then subject to hard caps (see below).

### Two kinds of N/A

The distinction the whole model rests on:

- **Structurally inapplicable** — the gate *cannot describe* this currency.
  Excluded from the numerator **and** the denominator, so the pillar
  renormalizes over what's left. A hard peg has no meaningful volatility
  signal (it measures the anchor, and inverts the day the peg breaks);
  Singapore has no policy rate because MAS runs exchange-rate policy; USD
  cannot carry against itself.
- **Missing data** — the gate applies but the value is absent. Scores 0 and
  **stays in the denominator**. Sparse coverage stays penalized.

Collapsing the two would either hand pegs a free sweep of the risk pillars or
punish them for a question that doesn't apply.

### Rating caps

Applied after scoring as ordinal ceilings, with `rating_raw` preserved and the
reasons surfaced in the UI. Hong Kong scores well on the pillars that still
apply to it, but a directional call on an administratively fixed rate isn't a
real output — so it's capped.

| Trigger | Cap |
|---|---|
| Base currency (USD) | NEUTRAL |
| Pegged regime (HKD, DKK) | NEUTRAL |
| Inflation > 50% | AVOID |
| Data coverage < 25% | NEUTRAL |
| REER history < 60 months | NEUTRAL |

Caps fail open: a missing metric never triggers one.

## Data sources

All free, no key, no auth. Every endpoint verified from CI.

| Source | Gives | Depth |
|---|---|---|
| [ECB SDMX](https://data.ecb.europa.eu/) | Daily reference rates, 29 currencies | **1999-01-04 →**, 7,000+ observations |
| [BIS SDMX](https://stats.bis.org/) | Real effective exchange rates, 64 economies | Monthly, **1994-01 →** |
| [BIS SDMX](https://stats.bis.org/) | Central bank policy rates, 48 economies | Monthly |
| [World Bank](https://data.worldbank.org/) | Current account, reserves, inflation, GDP growth | Annual, 1–2 yr lag |

Three things worth knowing if you extend this:

**BIS requires `format=csv`.** Without it every request returns `406` with an
SDMX error document. Wildcard the trailing key dimension (`M.R.B.`) to pull
all economies in one call.

**ECB alone gives the full cross-rate matrix.** All series are EUR-based, so
`X/USD = (EUR/USD) ÷ (EUR/X)`. One request covers the universe, and USD-based
history reaches back to 1999 with no second provider.

**World Bank 502s on `country/all`.** Scoping requests to an explicit
semicolon-joined country list works reliably where the wildcard does not. The
client also caches for 30 days and serves an expired entry rather than nothing
when the network fails, flagging it in provenance — annual macro data one
month stale is the same number, and losing a whole pillar to a transient
outage is not.

Coverage from the last run: current account 29/30, reserves 30/30, inflation
30/30, GDP growth 30/30. Government debt (17/30) and external debt (9/30) are
genuinely sparse — external debt is reported mainly by developing economies —
so they're **displayed as context but not scored**.

## Universe

30 currencies: the set with a live ECB daily series, all of which BIS also
covers for REER.

```
EUR USD JPY GBP CHF CAD AUD NZD NOK SEK DKK ISK CZK HUF PLN
RON TRY ILS ZAR CNY HKD SGD KRW THB MYR IDR PHP INR MXN BRL
```

The euro is why `data/currency_meta.py` exists: BIS publishes one euro-area
REER series under the synthetic area `XM`, but twenty countries use the euro.
Without an explicit collapse, joining BIS to a country list yields EUR twenty
times.

**Why not more?** TWD, SAR, AED, KWD, CLP, COP, PEN, ARS, RSD, BGN, MKD, BAM,
MAD, DZD and RUB all have BIS REER coverage, but no free source of *daily
spot history* — which Momentum and Stability need. Including them would add
rows scoring 0 on two of five pillars for want of data rather than on merit.
Adding them requires a daily-history provider; the pillars would then work
unchanged.

## Layout

```
data/       currency_meta (universe + identifier mapping), cache (TTL + retry
            + stale-serve), provenance, and one client per source
models/     valuation, carry, momentum, external, stability, series
            — pure functions, no I/O
scripts/    config (weights/thresholds), scoring (the GATES list),
            analyze_currencies (orchestrator), report_html, rescore_and_render,
            backtest, publish_pages, should_build (the build gate),
            check_site (the post-deploy check)
scheduled-tasks/daily-check/  the Mac mini's post-deploy check (runbook +
            wrapper)
templates/  report.html + world_svg.js (Natural Earth 110m, public domain)
tests/      147 offline tests + 10 live endpoint tests
```

## Working on it

```bash
python -m pytest -q                                    # 147 offline tests, <1s
python -m pytest -m live -q                            # 10 live endpoint tests
python scripts/analyze_currencies.py                   # full run (~1 min warm)
python scripts/analyze_currencies.py --no-macro        # skip World Bank
python scripts/rescore_and_render.py output/results_<date>.json
python scripts/backtest.py --results-dir output
```

`rescore_and_render.py` is the iteration loop for gate and weight changes: it
re-derives everything downstream of `score_and_rate` from a saved snapshot in
under a second, with no API calls. Only `analyze_currencies.py` touches the
network.

`results_<date>.json` is the pivot artifact — the backtester and the
re-scorer both consume it, and it's what makes calibration possible without
re-fetching 27 years of history.

### Sanity check

Japan's real broad effective exchange rate has gone from 164 (1994) to ~65 —
a roughly 60% real depreciation. **JPY should screen cheap on Valuation.** If
it ever doesn't, the REER sign convention has been inverted and the heaviest
pillar is backwards. Pinned in `tests/test_live_endpoints.py`.

## Publishing

`build.yml` runs **weekdays at 16:15 America/New_York** (and on pushes that
touch the model): it runs the tests, rebuilds the report, and deploys it
straight to GitHub Pages via `actions/deploy-pages`. The artifacts are
gitignored — the report is regenerated in full every run, so versioning it on
`main` would grow the repository without ever producing a useful diff.

Two scheduling problems are handled by `scripts/should_build.py` rather than by
cron, which can express neither:

**Daylight saving.** 16:15 New York is 20:15 UTC under EDT and 21:15 UTC under
EST, and GitHub cron is fixed-UTC. Both crons are registered and the gate
discards any run that starts before 16:00 local, or after the site has already
been published that day (read from the `pages-live` commit time), so exactly
one builds in either half of the year with no seasonal editing. The hour is a
floor rather than a window because GitHub often starts scheduled runs hours
late; requiring the 16:00 hour exactly silently stopped every build from
2026-08-26.

**Bank holidays.** Instead of a hard-coded calendar — which would need annual
maintenance and would encode one country's answer to a question about global
data — the gate asks whether the ECB actually published today's reference
rates. No new observation means nothing new to score, so the run is skipped.
That covers TARGET holidays, weekends and outages with one rule, and it fails
in the safe direction: rebuilding on a stale day would republish yesterday's
rates stamped with today's date.

A skipped run is a green run with a notice, not a failure. Missing a day costs
nothing — every build regenerates from full history, so the next run is
identical to the one that would have happened.

The same job also force-pushes a single-commit `pages-live` branch holding
`docs/`. That is **a fallback, not the mechanism**: it lets Pages serve the
site through Settings → Pages → "Deploy from a branch" (`pages-live` / `docs`)
with no Actions run at all. The only thing the workflow reads from it is its
commit time, for the once-a-day check above.

> An earlier version split this across two workflows, with `deploy-pages.yml`
> triggering on pushes to `pages-live`. That could never have fired:
> push-triggered workflows are read from the pushed branch's own tree, and
> `pages-live` contains only `docs/` — no workflow file. Deploying the
> artifact directly from the build job removes the coupling.

**First-time setup.** Actions does not register workflows pushed by some app
tokens, and Pages starts disabled on a new repository. If the Actions tab is
empty, enable Actions under Settings → Actions → General, then run
**Build and publish currency screener** once from the Actions tab. After that
the daily schedule takes over.

### The daily check (Mac mini)

Actions builds and deploys; a separate check confirms the build reached the
site and reports what changed. It runs on the Mac mini from a Claude desktop
app scheduled task (`the-daily-currency-check`, weekdays 18:55 New York)
following `scheduled-tasks/daily-check/SKILL.md`, which launches
`scheduled-tasks/daily-check/mac_run.sh` → `scripts/check_site.py`.

The check waits (up to 110 minutes) for `manifest.json` to show today's
date, because GitHub starts the scheduled build 2–3 hours late. If it never
does, it says why: the ECB published nothing today (a correct skip, the same
question `should_build.py` asks), today's run failed, or it is still late.
It derives the site URL from the git remote — the cloud routine it replaced
hard-coded the pre-rename `Currency-Arbitrage-` URL and reported a healthy
site as a 404. Everything the check keeps lives in one Finder-visible folder,
`~/CurrencyScreener`: each day's snapshot under `history/<date>/`, the latest
`report.md` and `status.json`, and `logs/`. Yesterday is therefore always on
disk for the day-over-day rating diff.

After each check, `scripts/archive_snapshots.py` pushes that history to the
`data/snapshots` branch as `<date>/results.json` and `<date>/manifest.json`,
one `Snapshot: <date>` commit per run. It is the long-term archive: days the
Mac prunes after 400 days stay on the branch, and a lost or replaced Mac
loses nothing. The commit is built with git plumbing on a private index, so
the checkout is never touched, and an unchanged history makes no commit. It
pushes with the Mac's existing GitHub credentials (`gh auth login`).

Mac setup: clone to `~/Projects/Currency-Screener-` (off iCloud) with a
Python ≥ 3.11 installed; there are no dependencies to install. Dry run with
`scheduled-tasks/daily-check/mac_run.sh --wait-min 0`, then create the app
task with the prompt "Follow scheduled-tasks/daily-check/SKILL.md in
~/Projects/Currency-Screener-". Only after its first clean run, disable the
cloud routine "Daily Currency Screener" (disable, not delete, so it can be
turned back on); never run both.

## Caveats

- **Not investment advice.** This ranks currencies on public macro data; it
  models no transaction costs, no position sizing and no execution.
- **Rating thresholds are provisional.** They're set on the composite's
  natural 0–100 scale, not calibrated against forward returns — that needs a
  history of daily snapshots that doesn't exist yet. `backtest.py` is built
  and reports honestly that it has nothing to measure until snapshots
  accumulate.
- **The External pillar is 1–2 years stale by construction.** It's a solvency
  filter, not a timing signal. Stability's realized volatility is what moves
  quickly.
- **Backtest observations overlap** and are therefore not independent; the
  script says so on every run rather than implying clean statistics.

## License

MIT
