# Working in this repo

Currency screener: 30 currencies, 19 gates, 5 weighted pillars, rendered to a
single self-contained HTML page. See README.md for the model itself.

## Stack

Python 3.11+. Dependencies are `jinja2` and `pytest` — nothing else. HTTP is
stdlib `urllib`; there is no pandas, no numpy, no requests. Keep it that way
unless something genuinely needs them: the whole project is 30 rows of data,
and a dependency-free build is why CI takes seconds.

No `__init__.py` files. Scripts insert the repo root on `sys.path` and import
as `from data import ...` / `from models import ...`.

## Architecture

- **`data/`** — one class-free module per upstream source. Every client
  returns plain dicts/lists and **never raises on network failure**; callers
  degrade per-field. `cache.py` owns TTL, retry and stale-serve; clients do
  not implement their own HTTP.
- **`models/`** — pure functions over `(period, value)` series. No I/O, no
  imports from `data/` except `currency_meta`.
- **`scripts/`** — `scoring.py` is the core; everything else orchestrates,
  renders or analyses. `check_site.py` is the post-deploy check the Mac mini
  runs after Actions publishes, and `archive_snapshots.py` pushes its
  history to the `data/snapshots` branch (`scheduled-tasks/daily-check/`;
  README, "The daily check").
- **Publishing** is GitHub Actions only (`build.yml`); the github-pages
  environment refuses deploys from anywhere else. The site URL follows the
  repository name — it moved when the repo was renamed from
  `Currency-Arbitrage-`, and GitHub does not redirect Pages.

## Conventions that matter

**Return `None`, never a plausible default.** A missing FX rate must not
become `1.0`; a short series must not become zero volatility. Downstream code
distinguishes "not applicable", "missing" and "zero", and every one of those
means something different in the scoring layer.

**Never cache a failure.** `cache.py` writes only on successful parse. A
transient 502 must not become "this indicator has no data" for 30 days.

**Intersect dates before computing any spread or return.** FX legs have
different national holiday calendars. `ecb_client.to_usd_series` is the
reference implementation.

**Every gate change touches three places in `scripts/scoring.py`:** the
`GATES` entry itself, a `_GATE_DISPLAY` entry (or the report renders a column
labelled with the raw field name), and a `_GATE_TIPS` entry (or the column
ships with no way to find out what it measures). `tests/test_gates.py`
enforces all three in both directions — no gate without display metadata or a
tooltip, and no orphans of either.

Tooltip copy lives in Python, not the template, so `gate_metadata()` carries
it to the report and the text cannot drift from the gate it describes. Write
what the measure is *and why it is built that way* — the reasoning is
otherwise buried in code comments a report reader never sees. A tooltip that
restates its own column label is worse than none, and the tests reject
anything under 60 characters on that basis.

**Careful with `{` immediately followed by `#` in `templates/report.html`.**
Jinja reads it as a comment opener and the template fails to parse. CSS like
`@media(...){#id{...}}` needs a space after the brace.

**Calibrate score ranges against the observed cross-section**, not textbook
extremes. The Stability pillar originally used a 3–30% volatility range when
the universe actually spans 4–12%; every currency scored 67–96 and the pillar
became a constant offset instead of a discriminator. Check the distribution
before choosing `_score_linear` bounds:

```bash
python scripts/rescore_and_render.py output/results_<date>.json
```

then look at each pillar's mean and standard deviation. A pillar with sd < 10
is not doing any work.

**The two kinds of N/A are load-bearing.** Structurally inapplicable leaves
both numerator and denominator; missing data scores 0 and stays in the
denominator. If you add an `applicable` predicate, it must tolerate a bare
`{'code': 'XXX'}` row — `tests/test_gates.py` calls every predicate that way.

## Testing

```bash
python -m pytest -q          # 154 offline tests, no network, <1s
python -m pytest -m live -q  # 10 tests against real endpoints
```

Offline tests build their own payloads. Anything needing the network goes in
`tests/test_live_endpoints.py` behind the `live` marker, so CI stays fast and
deterministic — but *do* add a live test when you depend on a response shape.
The BIS `format=csv` requirement is the cautionary case: without it every
request 406s, and nothing offline would ever catch a regression.

## The sanity check

JPY must screen cheap on Valuation (BIS REER 164 in 1994 → ~65 now). If it
doesn't, the REER sign convention is inverted and the heaviest pillar is
backwards. Higher REER = stronger real exchange rate = *more expensive*, so
every valuation range in `GATES` runs high→low.

## Iteration loop

`analyze_currencies.py` is the only script that fetches model data (the
build gate and `check_site.py` only ask whether the ECB published). For gate
and weight work, run it once and then iterate with
`rescore_and_render.py output/results_<date>.json` — under a second, no API
calls, rewrites both the JSON and the HTML.

## API notes

No keys, no `.env`, no auth anywhere. Three gotchas, all documented in the
client docstrings:

- BIS returns `406` without `format=csv`.
- World Bank 502s on `country/all`; scope to an explicit `;`-joined list.
- ECB keeps serving terminated series forever, so recency (not presence) is
  how `live_currencies` tells live from dead.
