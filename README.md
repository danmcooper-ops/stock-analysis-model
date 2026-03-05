# Stock Analysis Model

This project provides a reusable Python framework for analyzing stock data using CAPM, DCF, and comparison models. It pulls financial data from SEC EDGAR, yfinance, and Finnhub, and extracts all relevant data from balance sheets, income statements, and cash flow statements.

## Features
- Data ingestion from SEC EDGAR, yfinance, Finnhub
- Extraction of financial statement data
- CAPM, DCF, and comparison model calculations
- Modular and reusable design

## Usage
1. Configure API keys for Finnhub and SEC EDGAR if required.
2. Use the provided scripts to fetch and analyze stock data.
3. Extend modules for custom analysis.

## Setup
- Python 3.9+
- Install dependencies: `pip install -r requirements.txt`

## Structure
- `data/` - Data ingestion modules
- `models/` - Financial models (CAPM, DCF, comparison)
- `scripts/` - Example usage scripts
- `README.md` - Project documentation

## License
MIT
