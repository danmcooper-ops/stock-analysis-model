"""scripts/download_prices.py

Bulk download of full price history. Defaults to S&P 500; use --universe us
to backfill the full SEC EDGAR US-listed universe (matches analyze_stock.py's
--universe us flag).

Writes one Parquet file per ticker to --output-dir (default: output/prices/).
Skips tickers whose file already exists, so the run is safely resumable.

Pass --max-age-days N to also REFRESH files whose most recent bar is older
than N days. Without it the skip is unconditional and nothing on disk is ever
updated — which is how the universe drifted three months stale in 2026-07.

Usage:
    python scripts/download_prices.py                         # S&P 500 (default)
    python scripts/download_prices.py --universe us           # all US-listed
    python scripts/download_prices.py --output-dir output/prices --delay 0.4
    python scripts/download_prices.py --tickers AAPL MSFT GOOG
    python scripts/download_prices.py --max-age-days 7        # refresh stale files
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


def _last_bar_date(path: str):
    """Date of the most recent row in an existing Parquet file, or None."""
    try:
        df = pd.read_parquet(path, columns=["Close"])
        if df.empty:
            return None
        return pd.to_datetime(df.index).tz_localize(None).max()
    except Exception:
        return None


def download_ticker(ticker: str, output_dir: str, delay: float,
                    max_age_days: int = None) -> str:
    """Download max history for one ticker and save as Parquet.

    Returns 'skipped', 'ok', 'refreshed', or an error message string.

    With *max_age_days* set, an existing file is re-downloaded when its last
    BAR is older than that many days. Keying on the last bar rather than the
    file mtime is deliberate: mtime says when we last wrote the file, not how
    current the data inside it is.
    """
    dest = os.path.join(output_dir, f"{ticker}.parquet")
    existed = os.path.exists(dest)
    if existed:
        if max_age_days is None:
            return "skipped"
        last_bar = _last_bar_date(dest)
        if last_bar is not None:
            age = (pd.Timestamp.today().normalize() - last_bar).days
            if age <= max_age_days:
                return "skipped"

    time.sleep(delay)
    try:
        df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        if df.empty:
            return "empty"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        # Write-then-rename so a concurrent reader never sees a torn file —
        # analyze_stock.py reads this directory while downloads may be running.
        tmp = dest + ".tmp"
        df.to_parquet(tmp)
        os.replace(tmp, dest)
        return "refreshed" if existed else "ok"
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
    parser.add_argument("--max-age-days", type=int, default=None,
                        help="Re-download an existing file when its last bar is "
                             "older than N days. Omit to keep the default "
                             "skip-if-exists behaviour (never refreshes).")
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

    ok = skipped = empty = errors = refreshed = 0
    failed = []

    for i, ticker in enumerate(tickers, 1):
        result = download_ticker(ticker, args.output_dir, args.delay,
                                 max_age_days=args.max_age_days)
        if result == "ok":
            ok += 1
        elif result == "refreshed":
            refreshed += 1
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
    print(f"Done.  ok={ok}  refreshed={refreshed}  skipped={skipped}  "
          f"empty={empty}  errors={errors}")

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
