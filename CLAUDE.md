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
                   the daily results snapshots), price_store.py (bulk DuckDB
                   reads over the price parquets), sec_facts_cache.py
                   (on-disk companyfacts blobs), provenance.py, validation.py
models/          - Pure model functions: capm, dcf, ddm, epv, rim, nav,
                   ratios (WACC/ROIC), quality (Altman/Beneish/Piotroski),
                   market, macro, narrative, portfolio, valuation_types
scripts/         - Entry points: analyze_stock.py (main pipeline), backtest.py,
                   report_html.py / report_excel.py, scoring.py, config.py,
                   param_set.py, replay.py, ingest_snapshots.py (backfill the
                   snapshot store), archive_snapshot.py (gzip a run onto the
                   data/snapshots branch, with a size guard), plus
                   enrichment/maintenance scripts
tests/           - pytest suite (~1,300 tests) incl. hypothesis property tests
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
- **SEC companyfacts cache (`data/sec_facts_cache.py`):** companyfacts blobs
  are ~3.9 MB each uncompressed and were re-downloaded every run, so they are
  now kept gzipped (~14x smaller) under `data/cache/sec_facts/{CIK}.json.gz`.
  SEC serves no `ETag`/`Last-Modified` on that endpoint, so freshness is
  driven by filings instead of a TTL: `SECXBRLClient.refresh_stale_facts()`
  walks SEC's daily filing index from its last watermark and evicts the CIKs
  that filed a fact-bearing form (10-K/10-Q/20-F/40-F/6-K — *not* 8-K, whose
  inline XBRL only touches `dei` cover facts). The watermark stops at the
  first day whose index could not be read, so a transient failure re-reads
  that day rather than skipping its filings. `max_age_days` (env
  `SEC_FACTS_CACHE_MAX_AGE_DAYS`, default 30) is only a backstop for when the
  sweep cannot run; entries past it are pruned. Requests send
  `Accept-Encoding: gzip`, which urllib omits by default.
- **Snapshot archive:** `output/results_<date>.json` is the canonical run
  artifact and stays plain JSON locally, but the copy pushed to the
  `data/snapshots` branch is gzipped (`results_<date>.json.gz`, ~87 MiB ->
  ~27 MiB) by `scripts/archive_snapshot.py`. Plain JSON had reached 97.2 MiB
  against GitHub's 100 MiB per-blob cap and one day was already rejected and
  lost. The four discovery/IO helpers in `data/snapshot_store.py` —
  `list_snapshot_files`, `snapshot_date_from_path`, `read_snapshot`,
  `write_snapshot_file` — handle both forms, so the archive can hold a mix and
  every reader (backtest, ingest, query, report) works unchanged; a date
  present in both forms resolves to the plain file exactly once.
  `write_snapshot_file` is the single snapshot writer: compact separators
  (`indent=2` once inflated a file from 67 MB to 107 MB), `default=str` to
  match the historical encoding, atomic via `os.replace`, and deterministic
  when gzipping (`mtime=0`, `filename=''`) so re-archiving does not churn the
  branch. The archive script verifies the round-trip by SHA-256 and fails
  non-zero past an 80 MiB guard.
- **Snapshot store (`data/snapshot_store.py`):** every run's
  `output/results_<date>.json` (~66 MB, ~2,300 rows x ~270 keys) is mirrored
  into `output/snapshots.duckdb` (tables `runs`, `results`; scalar keys
  become typed columns, dicts/lists become JSON columns, new keys add
  columns on the fly). The JSON stays canonical; the store is a derived
  index that cross-run readers (carry-forward, "yesterday's rating", rating
  history, gate N/A deltas, portfolio alerts) query for a few columns
  instead of re-parsing whole files. Every reader falls back to the JSON
  when the store is absent or does not hold the dates it needs.
  `edgar_history` is kept as a slim projection (`years_available` and
  `operating_income_history` — the only sub-keys any scoring path reads), so
  a store row is a drop-in for re-scoring while the other 52 series stay in
  the JSON. The report-only narrative blocks (`news_headlines`,
  `legal_filings`, `insider_transactions`, descriptions, sector head/tailwinds
  …) are dropped entirely via `DEFAULT_EXCLUDE_KEYS`: measured over 84 real
  snapshots they were 2.6 GB of a 3.5 GB store, and nothing outside the HTML
  render reads them. NaN and inf are stored as-is rather than nulled (scoring
  treats a missing value as N/A but a NaN as a failed comparison), and a
  stringified `"Infinity"` does not give a numeric column VARCHAR evidence —
  one such `pe` value used to turn 2,413 floats in a snapshot into strings. `sync_snapshot_file()` re-mirrors a rewritten file (analyze_stock,
  the enrich_* scripts and rescore_and_render call it); `scripts/
  ingest_snapshots.py` backfills history. The store is versioned
  (`SCHEMA_VERSION`): readers ignore a store built at another version and a
  writable open rebuilds it empty, so a stale index degrades to the JSON path
  rather than serving wrong columns.
- **Backtest/query reads:** `backtest.py` loads each snapshot from the store
  when it holds that date (per-date decision, `--no-store` forces JSON) —
  the corpus costs roughly half the RSS of parsing the files, which is what
  made a calibration sweep expensive. Forward returns come from one DuckDB
  scan of `output/prices/*.parquet` (`data/price_store.window_closes`)
  instead of a parquet open per ticker; both paths select the same bar
  (nearest, ties to the later bar, nothing beyond 7 days) so measurements are
  unchanged. `query_results.py` serves `--history` from the store (all
  columns, no full-scan penalty) and offers `--sql` for raw queries, falling
  back to its older `.query_index_v1/` parquet index when the store is absent
  or incomplete.
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
  narrative — the story on the Macro Outlook tab, and, in the Macro Outlook
  section of each Sector Analysis tab, that sector's outlook plus 3-5
  headwind/tailwind bullets drawn from the FRED indicators (skipped cleanly
  when unset: both render without it, minus those blocks). The narrative's
  shape is versioned (`SCHEMA_VERSION` in `data/claude_narrative.py`): the
  day cache is keyed by date alone and a hit skips every post-parse check,
  so a shape change must bump it or the cache replays the old shape.
- yfinance requires no authentication
