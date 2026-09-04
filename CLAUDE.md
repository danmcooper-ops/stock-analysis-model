# Stock Analysis Model

## Overview

Python framework for valuing and rating US-listed equities. Runs CAPM/WACC,
two-stage DCF (GGM + exit-multiple + Monte Carlo), DDM, EPV, RIM and NAV
models over SEC XBRL and yfinance data, scores each company against quality
gates, and renders an interactive HTML report plus JSON/Excel snapshots.

## Tech Stack

- **Language:** Python 3.11+
- **Packaging:** `pyproject.toml` (pinned deps; dev extra with pytest, ruff, hypothesis, pytest-cov, pre-commit)
- **Core deps:** yfinance, pandas, numpy, requests, scipy, curl_cffi, jinja2, lxml, pyarrow, openpyxl, nltk, duckdb

## Project Structure

```
data/            - Data clients (SEC XBRL/insider/legal/supply, yfinance, FMP,
                   Tiingo, Finnhub supply, macro, news, culture, FDIC,
                   clinical trials, social sentiment, FX, treasury, US listings)
                   plus shared helpers: throttle.py, yf_session.py,
                   snapshot_cache.py, snapshot_store.py (DuckDB index over
                   the daily results snapshots), provenance.py, validation.py
models/          - Pure model functions: capm, dcf, ddm, epv, rim, nav,
                   ratios (WACC/ROIC), quality (Altman/Beneish/Piotroski),
                   market, macro, narrative, portfolio, valuation_types
scripts/         - Entry points: analyze_stock.py (main pipeline), backtest.py,
                   report_html.py / report_excel.py, scoring.py, config.py,
                   param_set.py, replay.py, ingest_snapshots.py (backfill the
                   snapshot store), plus enrichment/maintenance scripts
tests/           - pytest suite (~750 tests) incl. hypothesis property tests
templates/       - jinja2 report templates
scheduled-tasks/ - Operational runbooks for the nightly analysis + publish
output/          - (gitignored) run artifacts: results JSON, HTML, prices,
                   snapshots.duckdb (derived index over the results JSONs)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install   # runs ruff + the offline test suite before each commit
```

## Running

```bash
python scripts/analyze_stock.py                 # S&P 500 + Dow universe
python scripts/analyze_stock.py --universe us   # all US-listed (~7-8k tickers)
pytest -m "not network and not slow"            # offline suite (CI-equivalent)
ruff check .
```

## Architecture

- **Data layer (`data/`):** class-based clients wrapping third-party APIs,
  returning yfinance-shaped DataFrames/dicts. Every HTTP client shares
  `data/throttle.py` for rate limiting; every raw yfinance call goes through
  `data/yf_session.py` (15s socket timeout) or `YFinanceClient`'s 20s
  wall-clock guard. For US filers, SEC XBRL statements replace yfinance
  frames (`SECXBRLClient.build_yfinance_shape`, newest column first,
  outflows negative).
- **Models layer (`models/`):** pure functions. The fair-value models'
  primary implementations return a `Valuation` envelope
  (value/method/confidence/warnings/inputs_used, see
  `models/valuation_types.py`); same-named legacy wrappers return
  `.value` as float|None. Soft issues are RuntimeWarnings AND recorded
  on the envelope.
- **Snapshot store (`data/snapshot_store.py`):** every run's
  `output/results_<date>.json` (~66 MB, ~2,300 rows x ~270 keys) is mirrored
  into `output/snapshots.duckdb` (tables `runs`, `results`; scalar keys
  become typed columns, dicts/lists become JSON columns, new keys add
  columns on the fly). The JSON stays canonical; the store is a derived
  index that cross-run readers (carry-forward, "yesterday's rating", rating
  history, gate N/A deltas, portfolio alerts) query for a few columns
  instead of re-parsing whole files. Every reader falls back to the JSON
  when the store is absent or does not hold the dates it needs.
  `sync_snapshot_file()` re-mirrors a rewritten file (analyze_stock, the
  enrich_* scripts and rescore_and_render call it); `scripts/
  ingest_snapshots.py` backfills history. `backtest.py` and
  `query_results.py` still read the JSON files directly.
- **Scripts layer (`scripts/`):** `analyze_stock._main()` orchestrates
  13 `_run_*` phase functions (screen → analyze → score → narrate →
  write outputs). `report_html.build_html()` orchestrates the per-row
  context builder and sidecar writers. `scripts/config.py` holds all
  tunable constants; `scripts/param_set.py` validates parameter sets.

## Conventions

- `data/`, `models/`, `scripts/` are packages (`__init__.py`); the editable
  install makes them importable anywhere. Entry-point scripts keep a
  one-line `sys.path.insert(0, <repo root>)` prelude so a bare
  `python scripts/<name>.py` works on an uninstalled checkout.
- Diagnostics use `logging` (module loggers); `print` is reserved for
  genuine CLI/report output. The pipeline entry forces
  `warnings.simplefilter('always', RuntimeWarning)` so model warnings fire
  per ticker, and ends with a run-quality summary counting
  fabricated/fallback inputs.
- Broad `except Exception` handlers must log (WARNING for whole-source
  failures with ticker+source, DEBUG for per-field fallbacks) and never
  swallow silently.
- `open()` always passes `encoding=` (enforced by ruff PLW1514).
- Statement DataFrames: rows = line items, columns = periods newest-first.
- CI (`.github/workflows/ci.yml`) runs ruff + the offline suite with a
  coverage floor (`--cov-fail-under`, a ratchet: raise it when coverage
  rises, never lower it).

## API Keys / Environment

All via environment (never commit secrets; `.env` at repo root is parsed
by analyze_stock and gitignored):
- `SEC_EMAIL` — contact email for SEC EDGAR User-Agent
- `FMP_API_KEY`, `TIINGO_API_KEY`, `FINNHUB_API_KEY` — optional data sources
- `ANTHROPIC_API_KEY` — optional; enables the Claude-generated macro
  narrative on the Macro Outlook tab (skipped cleanly when unset)
- yfinance requires no authentication
