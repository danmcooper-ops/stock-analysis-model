"""scripts/download_prices.py

Bulk download of full price history. Defaults to S&P 500; use --universe us
to backfill the full SEC EDGAR US-listed universe (matches analyze_stock.py's
--universe us flag).

Writes one Parquet file per ticker to --output-dir (default: output/prices/).
Skips tickers whose file already exists, so the run is safely resumable.

Usage:
    python scripts/download_prices.py                         # S&P 500 (default)
    python scripts/download_prices.py --universe us           # all US-listed
    python scripts/download_prices.py --output-dir output/prices --delay 0.4
    python scripts/download_prices.py --tickers AAPL MSFT GOOG
"""

import argparse
import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze_stock import get_sp500_tickers
from data.us_listings import fetch_us_listed_tickers


def download_ticker(ticker: str, output_dir: str, delay: float) -> str:
    """Download max history for one ticker and save as Parquet.

    Returns 'skipped', 'ok', or an error message string.
    """
    dest = os.path.join(output_dir, f"{ticker}.parquet")
    if os.path.exists(dest):
        return "skipped"

    time.sleep(delay)
    try:
        df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        if df.empty:
            return "empty"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_parquet(dest)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Bulk download price history")
    parser.add_argument("--output-dir", default="output/prices",
                        help="Directory to write per-ticker Parquet files")
    parser.add_argument("--delay", type=float, default=0.35,
                        help="Seconds to wait between requests (default: 0.35)")
    parser.add_argument("--tickers", nargs="+",
                        help="Override ticker list (default: from --universe)")
    parser.add_argument("--universe", choices=["sp500", "us"], default="sp500",
                        help="Ticker universe when --tickers is not given. "
                             "'sp500' = S&P 500 (default), 'us' = all US-listed "
                             "equities from SEC EDGAR (~7-10k tickers).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.tickers:
        tickers = sorted(args.tickers)
    elif args.universe == "us":
        print("Fetching US-listed ticker universe from SEC EDGAR...")
        tickers = sorted(fetch_us_listed_tickers())
    else:
        print("Fetching S&P 500 ticker list...")
        tickers = sorted(get_sp500_tickers())

    total = len(tickers)
    print(f"{total} tickers to process — output: {args.output_dir}\n")

    ok = skipped = empty = errors = 0
    failed = []

    for i, ticker in enumerate(tickers, 1):
        result = download_ticker(ticker, args.output_dir, args.delay)
        if result == "ok":
            ok += 1
        elif result == "skipped":
            skipped += 1
        elif result == "empty":
            empty += 1
            failed.append((ticker, "empty response"))
        else:
            errors += 1
            failed.append((ticker, result))

        print(f"  [{i:>3}/{total}] {ticker:<6} {result}")

    print(f"\n{'='*50}")
    print(f"Done.  ok={ok}  skipped={skipped}  empty={empty}  errors={errors}")

    if failed:
        print("\nFailed tickers:")
        for t, reason in failed:
            print(f"  {t}: {reason}")

    # Report total size on disk
    files = [f for f in os.listdir(args.output_dir) if f.endswith(".parquet")]
    total_mb = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in files
    ) / 1_048_576
    print(f"\n{len(files)} files on disk — {total_mb:.1f} MB total")


if __name__ == "__main__":
    main()
