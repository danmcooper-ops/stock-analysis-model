# scripts/track_portfolio.py
"""Portfolio tracker — links holdings to the stock analysis model output.

Standalone usage:
    python scripts/track_portfolio.py
    python scripts/track_portfolio.py --holdings portfolio/holdings.json
    python scripts/track_portfolio.py --results output/results_2026-03-22.json

Or called programmatically from analyze_stock.py via run_portfolio_tracker().
"""
import sys
import os
import json
import glob
import argparse
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.yfinance_client import YFinanceClient
from data.portfolio_client import PortfolioClient
from models.portfolio_tracker import (
    enrich_holdings,
    compute_holding_weights,
    compute_portfolio_pnl,
    compute_portfolio_returns,
    detect_alerts,
    summarize_realized_gains,
)
from models.portfolio import concentration_analysis
from scripts.report_portfolio_html import build_portfolio_html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_results(output_dir='output', exclude=None):
    """Find the most recent results_*.json in output_dir."""
    pattern = os.path.join(output_dir, 'results_*.json')
    files = sorted(glob.glob(pattern))
    if exclude:
        files = [f for f in files if f != exclude]
    return files[-1] if files else None


def _load_results_json(path):
    """Load a results JSON and return (meta_dict, results_by_ticker)."""
    if not path or not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    results = data.get('results', [])
    by_ticker = {r['ticker']: r for r in results}
    return data, by_ticker


def _print_summary(portfolio_state):
    """Print a human-readable summary to stdout."""
    pnl = portfolio_state
    print(f"\n{'='*60}")
    print(f"  PORTFOLIO: {portfolio_state['portfolio_name']}")
    print(f"  Date: {portfolio_state['date']}")
    print(f"{'='*60}")

    total_mv = pnl.get('total_market_value')
    total_cost = pnl.get('total_cost_basis')
    upnl = pnl.get('unrealized_pnl')
    upnl_pct = pnl.get('unrealized_pnl_pct')
    port_ytd = pnl.get('portfolio_return_ytd')
    bench_ytd = pnl.get('benchmark_return_ytd')
    alpha_ytd = pnl.get('portfolio_alpha_ytd')

    if total_mv is not None:
        print(f"  Market Value:     ${total_mv:,.2f}")
    if total_cost is not None:
        print(f"  Cost Basis:       ${total_cost:,.2f}")
    if upnl is not None:
        sign = '+' if upnl >= 0 else ''
        pct_str = f" ({sign}{upnl_pct:.1%})" if upnl_pct is not None else ''
        print(f"  Unrealized P&L:   {sign}${upnl:,.2f}{pct_str}")
    if pnl.get('realized_pnl_ytd'):
        print(f"  Realized P&L YTD: ${pnl['realized_pnl_ytd']:,.2f}")

    print()
    if port_ytd is not None:
        print(f"  Portfolio YTD:    {port_ytd:+.1%}")
    if bench_ytd is not None:
        print(f"  {portfolio_state.get('benchmark','SPY')} YTD:          {bench_ytd:+.1%}")
    if alpha_ytd is not None:
        print(f"  Alpha YTD:        {alpha_ytd:+.1%}")

    alerts = portfolio_state.get('alerts', [])
    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            sev_icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(a['severity'], '⚪')
            print(f"    {sev_icon} [{a['severity']}] {a['message']}")

    print()
    print("  HOLDINGS:")
    header = f"  {'Ticker':<8} {'Shares':>7} {'Cost':>8} {'Price':>8} {'Value':>10} {'P&L%':>7} {'Alpha':>7} {'Rating':<12} {'Score':>6}"
    print(header)
    print(f"  {'-'*90}")
    for h in portfolio_state.get('holdings', []):
        ticker = h['ticker']
        shares = h.get('shares', 0)
        cost = h.get('cost_basis', 0)
        price = h.get('current_price')
        mv = h.get('market_value')
        pnl_pct = h.get('unrealized_pnl_pct')
        alpha = h.get('alpha')
        rating = h.get('rating', 'N/A')
        score = h.get('_composite_score')

        price_str = f"${price:.2f}" if price is not None else "N/A"
        mv_str = f"${mv:,.0f}" if mv is not None else "N/A"
        pnl_str = f"{pnl_pct:+.1%}" if pnl_pct is not None else "N/A"
        alpha_str = f"{alpha:+.1%}" if alpha is not None else "N/A"
        score_str = f"{score:.1f}" if score is not None else "N/A"

        print(f"  {ticker:<8} {shares:>7.1f} ${cost:>7.2f} {price_str:>8} {mv_str:>10} {pnl_str:>7} {alpha_str:>7} {rating:<12} {score_str:>6}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Core runner (callable from analyze_stock.py)
# ---------------------------------------------------------------------------

def run_portfolio_tracker(
    results=None,
    json_filename=None,
    risk_free_rate=None,
    holdings_path='portfolio/holdings.json',
    output_dir='output',
    benchmark=None,
):
    """Run the portfolio tracker.

    Can be called with an already-computed results list (from analyze_stock.py)
    or standalone (reads from the latest results JSON).

    Parameters
    ----------
    results : list of dict, optional
        Already-computed analysis results (avoids re-reading the JSON).
    json_filename : str, optional
        Path to today's results JSON (used to find prev-day file).
    risk_free_rate : float, optional
        Risk-free rate from the analysis run.
    holdings_path : str
        Path to portfolio/holdings.json.
    output_dir : str
        Directory to write output files.
    benchmark : str, optional
        Benchmark ticker override (defaults to what's in holdings.json).
    """
    print("\n[Portfolio Tracker] Starting...")

    # --- Load holdings ---
    try:
        pc = PortfolioClient(holdings_path=holdings_path)
        portfolio_data = pc.load_holdings()
    except FileNotFoundError as e:
        print(f"  {e}")
        return None

    holdings = portfolio_data['holdings']
    realized_gains = portfolio_data['realized_gains']
    portfolio_name = portfolio_data['portfolio_name']
    bench_ticker = benchmark or portfolio_data['benchmark']
    tickers = list({h['ticker'] for h in holdings})

    print(f"  Portfolio: {portfolio_name} ({len(tickers)} tickers)")

    # --- Load today's analysis results ---
    if results is not None:
        today_by_ticker = {r['ticker']: r for r in results}
        today_meta = {'date': date.today().isoformat(), 'risk_free_rate': risk_free_rate}
    else:
        latest = json_filename or _find_latest_results(output_dir)
        if not latest:
            print("  No results JSON found. Run analyze_stock.py first.")
            return None
        today_meta, today_by_ticker = _load_results_json(latest)
        json_filename = latest
        print(f"  Using results: {os.path.basename(latest)}")

    # --- Load prior-day results for alert diffing ---
    prev_path = _find_latest_results(output_dir, exclude=json_filename)
    if prev_path:
        _, prev_by_ticker = _load_results_json(prev_path)
        print(f"  Prior results: {os.path.basename(prev_path)}")
    else:
        prev_by_ticker = {}

    # --- Fetch live prices and benchmark ---
    yf_client = YFinanceClient()
    pc_with_yf = PortfolioClient(holdings_path=holdings_path, yf_client=yf_client)

    print(f"  Fetching prices for {len(tickers)} holdings...")
    current_prices = pc_with_yf.fetch_current_prices(tickers)

    print(f"  Fetching benchmark history ({bench_ticker})...")
    benchmark_series = pc_with_yf.fetch_benchmark_history(bench_ticker, period='2y')

    print(f"  Fetching per-holding price histories...")
    ticker_histories = {}
    for ticker in tickers:
        hist = pc_with_yf.fetch_ticker_history(ticker, since_date=None, period='2y')
        if hist is not None and len(hist) > 0:
            ticker_histories[ticker] = hist

    # --- Enrich and compute ---
    print("  Computing P&L and returns...")
    enriched = enrich_holdings(holdings, current_prices, today_by_ticker)
    enriched = compute_holding_weights(enriched)
    pnl_summary = compute_portfolio_pnl(enriched, realized_gains)
    returns_summary = compute_portfolio_returns(enriched, benchmark_series, ticker_histories)
    alerts = detect_alerts(enriched, prev_by_ticker)
    concentration = concentration_analysis(enriched)
    realized_by_year = summarize_realized_gains(realized_gains)

    # --- Assemble state ---
    portfolio_state = {
        'date': date.today().isoformat(),
        'portfolio_name': portfolio_name,
        'benchmark': bench_ticker,
        'risk_free_rate': risk_free_rate or today_meta.get('risk_free_rate'),
        'macro_regime': today_meta.get('macro_regime'),

        # P&L
        **pnl_summary,
        # Returns
        **returns_summary,

        'holdings': enriched,
        'alerts': alerts,
        'concentration': concentration,
        'realized_by_year': realized_by_year,
    }

    # --- Print summary ---
    _print_summary(portfolio_state)

    # --- Save outputs ---
    json_out = pc.save_portfolio_state(portfolio_state, output_dir)
    print(f"  JSON: {json_out}")

    today_str = date.today().isoformat()
    html_out = os.path.join(output_dir, f'portfolio_{today_str}.html')
    try:
        build_portfolio_html(portfolio_state, html_out)
        print(f"  HTML: {html_out}")
    except Exception as e:
        print(f"  HTML report failed: {e}")

    print("[Portfolio Tracker] Done.\n")
    return portfolio_state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(
        description='Portfolio tracker — links holdings to the stock analysis model.'
    )
    parser.add_argument(
        '--holdings', default='portfolio/holdings.json',
        help='Path to holdings JSON file (default: portfolio/holdings.json)'
    )
    parser.add_argument(
        '--results', default=None,
        help='Path to today\'s results JSON (default: latest output/results_*.json)'
    )
    parser.add_argument(
        '--benchmark', default=None,
        help='Benchmark ticker override (default: from holdings.json)'
    )
    parser.add_argument(
        '--output-dir', default='output',
        help='Output directory (default: output)'
    )
    args = parser.parse_args()

    run_portfolio_tracker(
        json_filename=args.results,
        holdings_path=args.holdings,
        output_dir=args.output_dir,
        benchmark=args.benchmark,
    )


if __name__ == '__main__':
    _main()
