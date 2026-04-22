"""scripts/validate_ratings.py

Validate model ratings against historical price returns.

Loads the most recent results_*.json snapshot, then for each ticker looks up
actual returns from the local parquet price files over 1 trailing horizon
(default 252 trading days ≈ 1 year).

Answers the core question: do stocks the model rates BUY actually beat PASS
stocks over the prior year?  If yes, the model is directionally calibrated.

Usage:
    python scripts/validate_ratings.py
    python scripts/validate_ratings.py --snapshot output/results_2026-04-20.json
    python scripts/validate_ratings.py --horizon 126   # ~6 months
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RATING_ORDER = ['BUY', 'LEAN BUY', 'HOLD', 'PASS']
BENCHMARK    = 'SPY'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_snapshot(path=None, results_dir='output'):
    if path:
        with open(path) as f:
            return json.load(f)
    files = sorted(glob.glob(os.path.join(results_dir, 'results_*.json')))
    if not files:
        raise FileNotFoundError(f"No results_*.json files in {results_dir}")
    with open(files[-1]) as f:
        return json.load(f)


def trailing_return(ticker, as_of, horizon_td, prices_dir):
    """Return (pct_return, start_price, end_price) or None."""
    path = os.path.join(prices_dir, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)[['Close']].sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    end_dt   = pd.Timestamp(as_of)
    start_dt = end_dt - horizon_td

    # Find nearest available trading days
    after_start = df.loc[df.index >= start_dt]
    before_end  = df.loc[df.index <= end_dt]
    if after_start.empty or before_end.empty:
        return None

    start_price = float(after_start['Close'].iloc[0])
    end_price   = float(before_end['Close'].iloc[-1])
    if start_price <= 0:
        return None

    return (end_price - start_price) / start_price, start_price, end_price


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot',    help='Path to a specific results_*.json')
    parser.add_argument('--results-dir', default='output')
    parser.add_argument('--prices-dir',  default='output/prices')
    parser.add_argument('--horizon',     type=int, default=252,
                        help='Trailing trading days to measure (default 252 ≈ 1 yr)')
    args = parser.parse_args()

    snapshot = load_snapshot(args.snapshot, args.results_dir)
    run_date = snapshot.get('run_date') or snapshot.get('date') or \
               snapshot.get('generated_at', '')[:10]
    print(f"Snapshot date : {run_date}")
    print(f"Horizon       : {args.horizon} trading days ≈ {args.horizon/21:.0f} months")

    # Convert trading-day horizon to a calendar window (approx 1.4× factor)
    cal_days  = int(args.horizon * 365 / 252)
    horizon_td = pd.Timedelta(days=cal_days)
    as_of      = pd.Timestamp(run_date) if run_date else pd.Timestamp.today()

    # Benchmark return
    spy_ret = trailing_return(BENCHMARK, as_of, horizon_td, args.prices_dir)
    spy_ret_val = spy_ret[0] if spy_ret else None
    print(f"SPY {args.horizon}d trailing: {spy_ret_val:.1%}" if spy_ret_val else "SPY: N/A")
    print()

    # Pull ratings + scores from snapshot
    stocks = snapshot.get('results') or snapshot.get('stocks') or []
    if not stocks and isinstance(snapshot, list):
        stocks = snapshot

    records = []
    for s in stocks:
        ticker  = s.get('ticker') or s.get('symbol')
        rating  = s.get('rating') or s.get('composite_rating')
        score   = s.get('_composite_score') or s.get('composite_score') or s.get('score')
        if not ticker or not rating:
            continue
        ret = trailing_return(ticker, as_of, horizon_td, args.prices_dir)
        if ret is None:
            continue
        records.append({
            'ticker': ticker,
            'rating': rating,
            'score':  score,
            'ret':    ret[0],
            'excess': (ret[0] - spy_ret_val) if spy_ret_val is not None else None,
        })

    if not records:
        print("No records matched — check snapshot format.")
        return

    df = pd.DataFrame(records)
    print(f"Matched {len(df)} tickers with price data\n")

    # --- Rating bucket performance ---
    print("=" * 58)
    print(f"{'Rating':<12} {'N':>4}  {'Mean Ret':>9}  {'Med Ret':>9}  {'vs SPY':>8}  {'Hit%':>6}")
    print("-" * 58)
    for rating in RATING_ORDER:
        sub = df[df['rating'] == rating]
        if sub.empty:
            continue
        n        = len(sub)
        mean_ret = sub['ret'].mean()
        med_ret  = sub['ret'].median()
        vs_spy   = sub['excess'].mean() if spy_ret_val is not None else float('nan')
        hit_pct  = (sub['ret'] > 0).mean() * 100 if spy_ret_val is None else \
                   (sub['excess'] > 0).mean() * 100
        print(f"{rating:<12} {n:>4}  {mean_ret:>8.1%}  {med_ret:>8.1%}  "
              f"{vs_spy:>+7.1%}  {hit_pct:>5.1f}%")
    print("=" * 58)

    # --- Score correlation ---
    scored = df.dropna(subset=['score', 'ret'])
    if len(scored) > 10:
        spearman_r, spearman_p = scipy_stats.spearmanr(scored['score'], scored['ret'])
        pearson_r,  pearson_p  = scipy_stats.pearsonr(scored['score'],  scored['ret'])
        print(f"\nScore ↔ Return correlation ({len(scored)} stocks):")
        print(f"  Spearman r = {spearman_r:+.3f}  (p={spearman_p:.3f})")
        print(f"  Pearson  r = {pearson_r:+.3f}  (p={pearson_p:.3f})")
        if spearman_r > 0.1 and spearman_p < 0.05:
            print("  ✓ Statistically significant positive correlation")
        elif spearman_p >= 0.05:
            print("  ✗ Not statistically significant")
        else:
            print("  ✗ Negative or weak correlation")

    # --- Top / bottom performers by rating ---
    print("\nTop 5 BUY-rated by 1yr return:")
    top = df[df['rating'] == 'BUY'].nlargest(5, 'ret')[['ticker','ret','score']]
    for _, r in top.iterrows():
        print(f"  {r['ticker']:<6}  {r['ret']:>+7.1%}  score={r['score']:.2f}" if r['score'] else
              f"  {r['ticker']:<6}  {r['ret']:>+7.1%}")

    print("\nBottom 5 PASS-rated by 1yr return (good if negative):")
    bot = df[df['rating'] == 'PASS'].nsmallest(5, 'ret')[['ticker','ret','score']]
    for _, r in bot.iterrows():
        print(f"  {r['ticker']:<6}  {r['ret']:>+7.1%}  score={r['score']:.2f}" if r['score'] else
              f"  {r['ticker']:<6}  {r['ret']:>+7.1%}")


if __name__ == '__main__':
    main()
