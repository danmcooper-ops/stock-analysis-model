# Stock Analysis Model

A Python framework for valuing and rating US-listed equities. It pulls
fundamentals from SEC EDGAR XBRL and yfinance (with optional FMP, Tiingo and
Finnhub enrichment), runs a battery of valuation models — CAPM/WACC,
two-stage DCF with Gordon-Growth and exit-multiple terminal values plus a
Monte Carlo pass, DDM, EPV, RIM and tangible-book NAV — scores every company
against quality gates (Altman Z, Beneish M, Piotroski F, earnings quality),
and renders an interactive HTML report alongside JSON and Excel snapshots.

## Features
- SEC XBRL statement reconstruction (10-K/20-F/40-F) with yfinance fallback
- Six fair-value models cross-checked and blended, each reporting its own
  method, confidence and caveats (`Valuation` envelope)
- Composite scoring and BUY / LEAN BUY / HOLD / PASS ratings with
  sector-relative percentiles
- Macro regime overlay (VIX, yield curve, credit spreads, momentum)
- Forward-return backtesting and walk-forward calibration tooling, with a
  readiness census that says when the snapshot corpus is long enough to trust
- Interactive HTML report with per-ticker deep-dive popups

## Setup
- Python 3.11+
- Runtime only: `pip install -r requirements.txt`
- Development (tests, lint, hooks): `pip install -e ".[dev]"` then
  `pre-commit install` — every commit then runs ruff + the offline test
  suite automatically.

## Usage
```bash
python scripts/analyze_stock.py                 # S&P 500 + Dow universe
python scripts/analyze_stock.py --universe us   # all US-listed equities
python scripts/backtest.py readiness            # how much evidence the snapshots hold
python scripts/backtest.py measure              # forward-return backtest (needs output/prices)
python scripts/backtest.py --help               # calibrate / annotate
pytest -m "not network and not slow"            # offline test suite
```

Outputs land in `output/` (gitignored): `results_<date>.json`,
`stock_analysis_results_<date>.html`, and an Excel workbook. Each run is
also mirrored into `output/snapshots.duckdb`, an embedded DuckDB index over
the daily snapshots that the cross-run readers (yesterday's rating, rating
history, gate coverage deltas, carry-forward) query instead of re-parsing
the JSON files. Backfill it from archived snapshots with
`python scripts/ingest_snapshots.py --results-dir <dir>` (idempotent), and
query it ad hoc:

```python
from data.snapshot_store import SnapshotStore
with SnapshotStore.open_existing('output/snapshots.duckdb') as s:
    print(s.query("SELECT date, count(*) FILTER (WHERE rating = 'BUY') AS buys "
                  "FROM results GROUP BY date ORDER BY date"))
```

## Configuration
API keys are read from the environment (or a gitignored `.env` at the repo
root): `SEC_EMAIL` for EDGAR identification, plus optional `FMP_API_KEY`,
`TIINGO_API_KEY`, `FINNHUB_API_KEY`. yfinance needs no authentication.

## Structure
See `CLAUDE.md` for the full layout and conventions: `data/` (API clients),
`models/` (pure model functions), `scripts/` (pipeline entry points),
`tests/` (~750 tests incl. property-based), `templates/` (report templates).
