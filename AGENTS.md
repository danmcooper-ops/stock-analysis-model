# Stock Analysis Model

## Overview

Python framework for analyzing stock data using CAPM, DCF, and comparison models. Pulls financial data from SEC EDGAR, yfinance, and Finnhub.

## Tech Stack

- **Language:** Python 3.9+
- **Virtual env:** `.venv/` (Python 3.13)
- **Dependencies:** yfinance, requests, pandas, finnhub-python, sec-edgar-downloader

## Project Structure

```
data/           - Data ingestion clients (SEC EDGAR, yFinance, Finnhub)
models/         - Financial models (CAPM, DCF, comparison ratios)
scripts/        - Entry point scripts
sec-edgar-filings/ - Downloaded SEC filing data
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python scripts/analyze_stock.py
```

## Architecture

- **Data layer (`data/`):** Class-based clients (SECEdgarClient, YFinanceClient, FinnhubClient) that wrap third-party APIs and return DataFrames/dicts.
- **Models layer (`models/`):** Pure functions for financial calculations — `calculate_beta`, `expected_return`, `calculate_dcf`, `compute_ratios`.
- **Scripts layer (`scripts/`):** Wires data clients to model functions. Uses `sys.path.append` for imports.

## Conventions

- Data clients are classes; model calculations are standalone functions
- No `__init__.py` files — scripts use `sys.path.append` for module resolution
- Financial data returned as pandas DataFrames or plain dicts
- `numpy` used for numerical computations in models

## API Keys

- Finnhub requires an API key (set via client constructor)
- SEC EDGAR requires an email address for identification
- yfinance requires no authentication
