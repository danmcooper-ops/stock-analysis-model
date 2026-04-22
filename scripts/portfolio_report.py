"""scripts/portfolio_report.py

Portfolio-level risk and diversification report for BUY / LEAN BUY rated stocks.

Reads the most recent results_*.json from --results-dir and the per-ticker
Parquet price files from --prices-dir, then prints:

  1. Correlation matrix (condensed heatmap table) + high-correlation flags
  2. Cluster analysis by sector (pairs with r > 0.75)
  3. Concentration summary (sector breakdown)
  4. Drawdown resilience table (2008 / 2020 / 2022 drawdowns + composite score)
  5. One-paragraph portfolio summary

Usage:
    python scripts/portfolio_report.py
    python scripts/portfolio_report.py --results-dir output/ --prices-dir output/prices --rating "BUY,LEAN BUY"
"""

import argparse
import glob
import io
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORR_HIGH = 0.85        # flag as highly correlated
CORR_CLUSTER = 0.75     # include in cluster analysis
DRAWDOWN_WARN = -0.40   # 2020 drawdown warning threshold
LOOKBACK_DAYS = 730     # ~2 years of trading history

SECTION_SEP = "=" * 72
SUBSEP = "-" * 72


# ---------------------------------------------------------------------------
# Tee — write to stdout AND a file simultaneously
# ---------------------------------------------------------------------------

class _Tee:
    """Writes to multiple streams at once (stdout + file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    # Make attribute look-ups (e.g. sys.stdout.encoding) fall through to
    # the first stream so libraries that inspect stdout still work.
    def __getattr__(self, name):
        return getattr(self._streams[0], name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print()
    print(SECTION_SEP)
    print(f"  {title}")
    print(SECTION_SEP)


def find_latest_results(results_dir: str) -> str:
    pattern = os.path.join(results_dir, "results_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"[ERROR] No results_*.json files found in '{results_dir}'")
    return files[-1]


def load_results(path: str) -> tuple[str, list[dict]]:
    """Return (date_str, list_of_stock_dicts)."""
    with open(path) as f:
        data = json.load(f)
    date_str = data.get("date", "unknown")
    results = data.get("results", [])
    return date_str, results


def load_close_series(ticker: str, prices_dir: str, lookback_days: int) -> pd.Series | None:
    """
    Load adjusted Close for the trailing `lookback_days` calendar days.
    Returns a pd.Series with DatetimeIndex, or None if the file is missing/unusable.
    """
    path = os.path.join(prices_dir, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path, columns=["Close"])
        df.index = pd.to_datetime(df.index)
        cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
        series = df.loc[df.index >= cutoff, "Close"].dropna()
        if len(series) < 20:
            return None
        return series.rename(ticker)
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


def compute_returns(series: pd.Series) -> pd.Series:
    """Daily percentage returns, dropping the first NaN."""
    return series.pct_change().dropna()


def pearson_corr_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation on a DataFrame of daily returns.
    Uses numpy for compatibility (scipy not required).
    Rows/columns with insufficient overlap are set to NaN.
    """
    tickers = returns_df.columns.tolist()
    n = len(tickers)
    mat = np.full((n, n), np.nan)

    for i in range(n):
        mat[i, i] = 1.0
        xi = returns_df.iloc[:, i].values
        for j in range(i + 1, n):
            xj = returns_df.iloc[:, j].values
            mask = ~(np.isnan(xi) | np.isnan(xj))
            if mask.sum() < 20:
                continue
            a, b = xi[mask], xj[mask]
            # Pearson r = cov(a,b) / (std(a)*std(b))
            if a.std() == 0 or b.std() == 0:
                continue
            r = np.corrcoef(a, b)[0, 1]
            mat[i, j] = r
            mat[j, i] = r

    return pd.DataFrame(mat, index=tickers, columns=tickers)


def worst_drawdown_in_window(series: pd.Series, start: str, end: str) -> float | None:
    """
    Compute the maximum drawdown (peak-to-trough) within a date window.
    Returns a float in [-1, 0] or None if no data.
    """
    mask = (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
    window = series[mask]
    if len(window) < 5:
        return None
    running_max = window.cummax()
    drawdowns = (window - running_max) / running_max
    return float(drawdowns.min())


def fmt_pct(value, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "  n/a  "
    return f"{value * 100:+.{decimals}f}%"


def fmt_float(value, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "  n/a"
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def print_correlation_matrix(corr: pd.DataFrame, tickers: list[str]) -> list[tuple]:
    """
    Print a condensed heatmap-style table and return list of high-corr pairs.
    """
    section("1. CORRELATION MATRIX  (2-year daily returns, Pearson r)")

    n = len(tickers)
    col_w = 7  # width per correlation cell

    # Header row: print ticker abbreviations across the top
    header = " " * 8
    for t in tickers:
        label = t[:col_w - 1].rjust(col_w - 1)
        header += label + " "
    print(header)
    print(SUBSEP)

    high_pairs = []

    for i, ti in enumerate(tickers):
        row = f"{ti:<7} |"
        for j, tj in enumerate(tickers):
            if i == j:
                cell = "  1.00 "
            elif j < i:
                cell = "       "
            else:
                val = corr.loc[ti, tj]
                if np.isnan(val):
                    cell = f"{'n/a':>{col_w - 1}} "
                else:
                    cell = f"{val:>{col_w - 1}.2f} "
                    if val > CORR_HIGH:
                        high_pairs.append((ti, tj, val))
        row += cell if j >= i else "       "
        # rebuild properly
        row = f"{ti:<7} |"
        for j, tj in enumerate(tickers):
            if i == j:
                row += "  1.00 "
            elif j < i:
                row += "       "
            else:
                val = corr.loc[ti, tj]
                if np.isnan(val):
                    row += f"{'n/a':>{col_w - 1}} "
                else:
                    row += f"{val:>{col_w - 1}.2f} "
        print(row)

    print()
    if high_pairs:
        print(f"  HIGH CORRELATION PAIRS (r > {CORR_HIGH}):")
        for t1, t2, r in sorted(high_pairs, key=lambda x: -x[2]):
            print(f"    *** {t1} <-> {t2}  r = {r:.3f}  — consider overlap / redundancy")
    else:
        print(f"  No pairs exceed r = {CORR_HIGH} threshold.")

    return high_pairs


def print_cluster_analysis(corr: pd.DataFrame, stocks: list[dict]) -> dict[str, list]:
    """
    Group by sector, find correlation clusters within each sector.
    Returns sector -> list of tickers dict.
    """
    section("2. CLUSTER ANALYSIS  (sector groups, pairs with r > 0.75)")

    ticker_to_sector = {s["ticker"]: s.get("sector") or "Unknown" for s in stocks}
    tickers = corr.index.tolist()

    # Group tickers in our corr matrix by sector
    sector_tickers: dict[str, list[str]] = {}
    for t in tickers:
        sec = ticker_to_sector.get(t, "Unknown")
        sector_tickers.setdefault(sec, []).append(t)

    for sector, members in sorted(sector_tickers.items()):
        if len(members) < 2:
            continue

        # Find all pairs with r > CORR_CLUSTER
        high_pairs = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                t1, t2 = members[i], members[j]
                r = corr.loc[t1, t2]
                if not np.isnan(r) and r > CORR_CLUSTER:
                    high_pairs.append((t1, t2, r))

        if not high_pairs:
            print(f"  {sector:<30}  {len(members):>2} tickers  — no clusters above r={CORR_CLUSTER}")
            continue

        # Find unique tickers involved in high-corr pairs
        clustered = set()
        for t1, t2, _ in high_pairs:
            clustered.add(t1)
            clustered.add(t2)

        avg_r = np.mean([r for _, _, r in high_pairs])
        print(f"  Cluster risk: {len(clustered)} tickers in {sector} are highly correlated  "
              f"(avg r={avg_r:.2f})")
        for t1, t2, r in sorted(high_pairs, key=lambda x: -x[2]):
            print(f"    {t1} <-> {t2}  r={r:.2f}")

    return sector_tickers


def print_concentration_summary(stocks: list[dict], total_rated: int) -> None:
    section("3. CONCENTRATION SUMMARY  (BUY + LEAN BUY bucket)")

    sector_counts: dict[str, int] = {}
    for s in stocks:
        sec = s.get("sector") or "Unknown"
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    n = len(stocks)
    print(f"  Total BUY / LEAN BUY: {n} stocks  (out of {total_rated} analyzed)\n")

    by_rating: dict[str, dict[str, int]] = {}
    for s in stocks:
        sec = s.get("sector") or "Unknown"
        rating = s.get("rating", "")
        by_rating.setdefault(sec, {"BUY": 0, "LEAN BUY": 0})
        if rating in by_rating[sec]:
            by_rating[sec][rating] += 1

    print(f"  {'Sector':<32}  {'Count':>5}  {'% of bucket':>11}  {'BUY':>4}  {'LEAN BUY':>9}")
    print("  " + SUBSEP)
    for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
        pct = cnt / n * 100 if n else 0
        b = by_rating.get(sec, {}).get("BUY", 0)
        lb = by_rating.get(sec, {}).get("LEAN BUY", 0)
        bar = "#" * int(pct / 2)  # max ~25 chars for 50%
        print(f"  {sec:<32}  {cnt:>5}  {pct:>9.1f}%  {b:>4}  {lb:>9}   {bar}")


def compute_drawdowns_from_prices(
    stocks: list[dict],
    prices_dir: str,
) -> list[dict]:
    """
    For each stock, read from JSON fields (drawdown_2008, drawdown_2020,
    drawdown_2022, realized_vol).  If those fields are absent or None, compute
    from price data where available.
    Returns list of enriched dicts with keys:
      ticker, sector, composite_score, dd_2008, dd_2020, dd_2022, realized_vol
    """
    WINDOWS = {
        "dd_2008": ("2008-01-01", "2009-03-31"),
        "dd_2020": ("2020-01-15", "2020-04-30"),
        "dd_2022": ("2022-01-01", "2022-10-31"),
    }

    rows = []
    for s in stocks:
        ticker = s["ticker"]

        # Try JSON fields first
        dd_2008 = s.get("drawdown_2008")
        dd_2020 = s.get("drawdown_2020")
        dd_2022 = s.get("drawdown_2022")
        realized_vol = s.get("realized_vol")

        # If not in JSON, compute from full price history
        needs_prices = any(v is None for v in [dd_2008, dd_2020, dd_2022])
        if needs_prices:
            path = os.path.join(prices_dir, f"{ticker}.parquet")
            if os.path.exists(path):
                try:
                    full = pd.read_parquet(path, columns=["Close"])
                    full.index = pd.to_datetime(full.index)
                    close = full["Close"].dropna()

                    if dd_2008 is None:
                        dd_2008 = worst_drawdown_in_window(close, *WINDOWS["dd_2008"])
                    if dd_2020 is None:
                        dd_2020 = worst_drawdown_in_window(close, *WINDOWS["dd_2020"])
                    if dd_2022 is None:
                        dd_2022 = worst_drawdown_in_window(close, *WINDOWS["dd_2022"])

                    if realized_vol is None:
                        # 2-year trailing annualized vol
                        cutoff = close.index.max() - pd.Timedelta(days=730)
                        recent = close[close.index >= cutoff]
                        if len(recent) > 20:
                            daily_ret = recent.pct_change().dropna()
                            realized_vol = float(daily_ret.std() * np.sqrt(252))
                except Exception:
                    pass

        rows.append({
            "ticker": ticker,
            "sector": s.get("sector") or "Unknown",
            "composite_score": s.get("_composite_score"),
            "rating": s.get("rating", ""),
            "dd_2008": dd_2008,
            "dd_2020": dd_2020,
            "dd_2022": dd_2022,
            "realized_vol": realized_vol,
        })

    return rows


def print_drawdown_table(drawdown_rows: list[dict]) -> None:
    section("4. DRAWDOWN RESILIENCE  (ranked by 2020 drawdown, worst first)")

    # Sort by 2020 drawdown ascending (worst first); None goes to the end
    def sort_key(row):
        v = row["dd_2020"]
        return (v is None, v if v is not None else 0)

    sorted_rows = sorted(drawdown_rows, key=sort_key)

    header = (
        f"  {'Ticker':<7}  {'Sector':<28}  {'2020 DD':>8}  "
        f"{'2022 DD':>8}  {'2008 DD':>8}  {'Comp Score':>11}  {'Flag'}"
    )
    print(header)
    print("  " + SUBSEP)

    warned = 0
    for row in sorted_rows:
        flag = ""
        if row["dd_2020"] is not None and row["dd_2020"] < DRAWDOWN_WARN:
            flag = "*** SEVERE DROP"
            warned += 1

        print(
            f"  {row['ticker']:<7}  "
            f"{(row['sector'] or 'Unknown'):<28}  "
            f"{fmt_pct(row['dd_2020']):>8}  "
            f"{fmt_pct(row['dd_2022']):>8}  "
            f"{fmt_pct(row['dd_2008']):>8}  "
            f"{fmt_float(row['composite_score'], 1):>11}  "
            f"{flag}"
        )

    print()
    if warned:
        print(f"  *** = drawdown worse than {DRAWDOWN_WARN * 100:.0f}% in the 2020 COVID crash.")
    else:
        print("  No stocks with 2020 drawdown worse than -40%.")


def print_summary(
    stocks: list[dict],
    high_corr_pairs: list[tuple],
    drawdown_rows: list[dict],
    missing_tickers: list[str],
    results_date: str,
    ratings_filter: list[str],
) -> None:
    section("5. SUMMARY")

    n = len(stocks)
    n_sectors = len({s.get("sector") or "Unknown" for s in stocks})
    n_high_corr = len(high_corr_pairs)
    n_severe_dd = sum(
        1 for r in drawdown_rows
        if r["dd_2020"] is not None and r["dd_2020"] < DRAWDOWN_WARN
    )
    scores = [s["_composite_score"] for s in stocks if s.get("_composite_score") is not None]
    avg_score = np.mean(scores) if scores else None

    sector_counts = {}
    for s in stocks:
        sec = s.get("sector") or "Unknown"
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
    top_sector = max(sector_counts, key=sector_counts.get) if sector_counts else "N/A"
    top_sector_pct = sector_counts.get(top_sector, 0) / n * 100 if n else 0

    corr_note = (
        f"{n_high_corr} highly correlated pair(s) (r > {CORR_HIGH}) were identified, "
        "suggesting some redundancy in the portfolio"
        if n_high_corr
        else f"no pairs exceed the r > {CORR_HIGH} high-correlation threshold"
    )
    dd_note = (
        f"{n_severe_dd} stock(s) experienced drawdowns worse than -40% during the 2020 COVID crash, "
        "warranting extra scrutiny for tail-risk exposure"
        if n_severe_dd
        else "no stocks experienced drawdowns worse than -40% during the 2020 COVID crash"
    )
    missing_note = (
        f"  Note: {len(missing_tickers)} ticker(s) were skipped due to missing price data: "
        f"{', '.join(missing_tickers[:10])}{'...' if len(missing_tickers) > 10 else ''}."
        if missing_tickers
        else ""
    )

    ratings_str = " / ".join(ratings_filter)
    summary = (
        f"  As of {results_date}, the {ratings_str} bucket contains {n} stocks spanning "
        f"{n_sectors} sectors, with an average composite score of "
        f"{f'{avg_score:.1f}' if avg_score is not None else 'n/a'}.  "
        f"The largest sector concentration is {top_sector} at {top_sector_pct:.1f}% of the bucket.  "
        f"Correlation analysis found {corr_note}.  "
        f"On the resilience side, {dd_note}.  "
        f"Investors should monitor sector concentration and correlated positions "
        f"before sizing any single cluster too heavily."
    )

    # Word-wrap at ~80 chars
    words = summary.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > 78:
            print(line)
            line = "  " + word
        else:
            line = (line + " " + word).lstrip()
            if not line.startswith("  "):
                line = "  " + line
    if line.strip():
        print(line)

    if missing_note:
        print()
        print(missing_note)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio diversification and risk report for rated stocks."
    )
    parser.add_argument(
        "--results-dir",
        default="output/",
        help="Directory containing results_YYYY-MM-DD.json files (default: output/)",
    )
    parser.add_argument(
        "--prices-dir",
        default="output/prices",
        help="Directory containing per-ticker Parquet files (default: output/prices)",
    )
    parser.add_argument(
        "--rating",
        default="BUY,LEAN BUY",
        help="Comma-separated ratings to include (default: 'BUY,LEAN BUY')",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help=(
            "Path to save the text report "
            "(default: <results-dir>/portfolio_report_YYYY-MM-DD.txt)"
        ),
    )
    args = parser.parse_args()

    ratings_filter = [r.strip() for r in args.rating.split(",")]

    # ------------------------------------------------------------------
    # 0. Load data
    # ------------------------------------------------------------------
    results_file = find_latest_results(args.results_dir)
    results_date, all_stocks = load_results(results_file)

    # ------------------------------------------------------------------
    # 0a. Set up output file (tee stdout → terminal + file)
    # ------------------------------------------------------------------
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(
            args.results_dir.rstrip("/\\"),
            f"portfolio_report_{results_date}.txt",
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    _out_file = open(out_path, "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _out_file)

    print()
    print(SECTION_SEP)
    print(f"  PORTFOLIO REPORT  —  results date: {results_date}")
    print(f"  Source: {results_file}")
    print(f"  Ratings filter: {', '.join(ratings_filter)}")
    print(SECTION_SEP)

    # Filter to target ratings
    stocks = [s for s in all_stocks if s.get("rating") in ratings_filter]
    if not stocks:
        sys.exit(f"[ERROR] No stocks with ratings {ratings_filter} found in {results_file}")

    total_analyzed = len(all_stocks)

    # ------------------------------------------------------------------
    # 0b. Load price data for correlation analysis
    # ------------------------------------------------------------------
    print(f"\n  Loading price data for {len(stocks)} tickers...")
    returns_dict: dict[str, pd.Series] = {}
    missing_tickers: list[str] = []

    for s in stocks:
        ticker = s["ticker"]
        close = load_close_series(ticker, args.prices_dir, LOOKBACK_DAYS)
        if close is None:
            missing_tickers.append(ticker)
        else:
            rets = compute_returns(close)
            if len(rets) >= 20:
                returns_dict[ticker] = rets

    if missing_tickers:
        print(f"  Skipped (no price file or insufficient data): {', '.join(missing_tickers)}")

    loaded_tickers = list(returns_dict.keys())
    print(f"  Loaded {len(loaded_tickers)} tickers with price data.")

    if len(loaded_tickers) < 2:
        print("\n  [WARN] Fewer than 2 tickers with price data — skipping correlation analysis.")
        corr = pd.DataFrame()
        high_corr_pairs = []
    else:
        # Align on common dates; allow NaN for non-overlapping dates
        returns_df = pd.DataFrame(returns_dict)
        corr = pearson_corr_matrix(returns_df)

        # ------------------------------------------------------------------
        # 1. Correlation matrix
        # ------------------------------------------------------------------
        high_corr_pairs = print_correlation_matrix(corr, loaded_tickers)

        # ------------------------------------------------------------------
        # 2. Cluster analysis
        # ------------------------------------------------------------------
        print_cluster_analysis(corr, stocks)

    # ------------------------------------------------------------------
    # 3. Concentration summary
    # ------------------------------------------------------------------
    print_concentration_summary(stocks, total_analyzed)

    # ------------------------------------------------------------------
    # 4. Drawdown resilience
    # ------------------------------------------------------------------
    print(f"\n  Computing drawdown data for {len(stocks)} tickers...")
    drawdown_rows = compute_drawdowns_from_prices(stocks, args.prices_dir)
    print_drawdown_table(drawdown_rows)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print_summary(
        stocks=stocks,
        high_corr_pairs=high_corr_pairs,
        drawdown_rows=drawdown_rows,
        missing_tickers=missing_tickers,
        results_date=results_date,
        ratings_filter=ratings_filter,
    )

    # ------------------------------------------------------------------
    # Restore stdout and close output file
    # ------------------------------------------------------------------
    sys.stdout = _orig_stdout
    _out_file.close()
    print(f"[Portfolio report saved → {out_path}]")


if __name__ == "__main__":
    main()
