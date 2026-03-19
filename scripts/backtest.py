# scripts/backtest.py
"""
Backtesting pipeline: measures whether the model's ratings predict forward returns.

Usage:
    python scripts/backtest.py --results-dir output/
    python scripts/backtest.py --results-dir output/ --horizons 30,90,180

Core question: Do BUY-rated stocks outperform HOLD/PASS stocks?

Loads all results_YYYY-MM-DD.json snapshots.  For each snapshot whose
evaluation date (snapshot + horizon) is in the past, fetches actual prices
and computes:
  1. Rating-bucket returns (mean/median per BUY, LEAN BUY, HOLD, PASS)
  2. Excess return vs SPY benchmark (alpha)
  3. Hit rate (% of stocks that beat SPY)
  4. Gates-passed correlation with forward returns
  5. Fair-value accuracy (MAE, signed error, within ±20%)

Writes:
  - Console summary
  - output/backtest_YYYY-MM-DD.xlsx (Summary, Detail, Stats tabs)
"""
import sys
import os
import json
import glob
import numpy as np
from collections import defaultdict
from datetime import date, datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.utils import rank


# Rating order (best → worst) for display
RATING_ORDER = ['STRONG_BUY', 'BUY', 'LEAN BUY', 'HOLD', 'PASS']
BENCHMARK = 'SPY'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir='output'):
    """Load all results JSON files, sorted by date."""
    pattern = os.path.join(results_dir, 'results_*.json')
    files = sorted(glob.glob(pattern))
    all_results = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            all_results.append(data)
    return all_results


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def fetch_forward_returns(tickers, run_date_str, horizon_days, yf_client):
    """
    Compute forward returns for each ticker + SPY from run_date to
    run_date + horizon_days.

    Returns:
        dict: {ticker: {'ret': float, 'start': float, 'end': float}}
        None for tickers where data is unavailable.
    """
    import pandas as pd

    run_dt = pd.Timestamp(run_date_str)
    eval_dt = run_dt + pd.Timedelta(days=horizon_days)
    returns = {}

    all_tickers = list(set(tickers + [BENCHMARK]))

    for ticker in all_tickers:
        try:
            hist = yf_client.fetch_history(ticker, period="1y")
            if hist is None or len(hist) < 10:
                continue

            # Localise lookup timestamps to match the history index timezone
            tz = hist.index.tz
            run_ts = pd.Timestamp(run_date_str, tz=tz)
            eval_ts = pd.Timestamp(eval_dt, tz=tz) if tz else eval_dt

            # Nearest trading day to run_date and eval_date
            start_idx = hist.index.get_indexer([run_ts], method='nearest')[0]
            end_idx = hist.index.get_indexer([eval_ts], method='nearest')[0]

            start_price = float(hist.iloc[start_idx])
            end_price = float(hist.iloc[end_idx])

            if start_price > 0:
                returns[ticker] = {
                    'ret': (end_price - start_price) / start_price,
                    'start': start_price,
                    'end': end_price,
                }
        except Exception:
            continue

    return returns


# ---------------------------------------------------------------------------
# Single-run analysis
# ---------------------------------------------------------------------------

def analyze_run(run, horizon_days, yf_client):
    """
    Analyze one snapshot at one horizon.

    Returns dict with:
        run_date, horizon, spy_return,
        buckets: {rating: {mean, median, count, alpha, hit_rate}},
        gates_corr: Spearman rho between gates_passed and return,
        fv_metrics: {mae, signed_error, within_20, n},
        details: [{ticker, rating, gates, fv, price, end_price, ret, excess}]
    """
    run_date = run.get('date')
    stocks = run.get('results', [])
    if not run_date or not stocks:
        return None

    tickers = [s['ticker'] for s in stocks if s.get('ticker')]
    returns = fetch_forward_returns(tickers, run_date, horizon_days, yf_client)

    spy = returns.get(BENCHMARK)
    spy_ret = spy['ret'] if spy else 0.0

    # Build detail rows
    details = []
    bucket_returns = defaultdict(list)  # rating -> list of returns
    gates_vals = []
    return_vals = []
    fv_preds = []
    fv_actuals = []

    for s in stocks:
        ticker = s.get('ticker')
        if not ticker or ticker not in returns:
            continue

        r = returns[ticker]
        rating = s.get('rating', 'UNKNOWN')
        gates = s.get('_gates_passed_num', 0)
        fv = s.get('dcf_fv')
        price = s.get('price')

        excess = r['ret'] - spy_ret

        detail = {
            'ticker': ticker,
            'rating': rating,
            'gates_passed': s.get('_gates_passed', 'N/A'),
            'gates_num': gates,
            'dcf_fv': fv,
            'start_price': r['start'],
            'end_price': r['end'],
            'return': r['ret'],
            'spy_return': spy_ret,
            'excess_return': excess,
        }
        details.append(detail)

        bucket_returns[rating].append(r['ret'])

        if isinstance(gates, (int, float)) and gates >= 0:
            gates_vals.append(gates)
            return_vals.append(r['ret'])

        if fv and fv > 0 and r['end'] > 0:
            fv_preds.append(fv)
            fv_actuals.append(r['end'])

    if not details:
        return None

    # --- Rating bucket stats ---
    buckets = {}
    for rating in RATING_ORDER:
        rets = bucket_returns.get(rating, [])
        if not rets:
            continue
        arr = np.array(rets)
        buckets[rating] = {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'count': len(rets),
            'alpha': float(np.mean(arr)) - spy_ret,
            'hit_rate': float(np.sum(arr > spy_ret) / len(arr)),
        }

    # --- Gates-passed correlation ---
    gates_corr = None
    if len(gates_vals) >= 5:
        n = len(gates_vals)
        rank_g = rank(gates_vals)
        rank_r = rank(return_vals)
        d_sq = sum((rg - rr) ** 2 for rg, rr in zip(rank_g, rank_r))
        gates_corr = 1 - (6 * d_sq) / (n * (n ** 2 - 1)) if n > 1 else 0.0

    # --- FV accuracy ---
    fv_metrics = None
    if len(fv_preds) >= 5:
        preds = np.array(fv_preds)
        acts = np.array(fv_actuals)
        pct_err = (preds - acts) / acts
        fv_metrics = {
            'mae': float(np.mean(np.abs(pct_err))),
            'signed_error': float(np.mean(pct_err)),
            'within_20': float(np.sum(np.abs(pct_err) <= 0.20) / len(pct_err)),
            'n': len(pct_err),
        }

    # Store source stock data for enhanced analytics (sector, MoS, targets)
    source_stocks = {}
    for s in stocks:
        t = s.get('ticker')
        if t:
            source_stocks[t] = {
                'sector': s.get('sector'),
                'dcf_fv': s.get('dcf_fv'),
                'target_mean': s.get('target_mean'),
                'rating': s.get('rating'),
            }

    return {
        'run_date': run_date,
        'horizon': horizon_days,
        'n_stocks': len(details),
        'spy_return': spy_ret,
        'buckets': buckets,
        'gates_corr': gates_corr,
        'fv_metrics': fv_metrics,
        'details': details,
        '_source_stocks': source_stocks,
    }


# ---------------------------------------------------------------------------
# Full backtest across all snapshots and horizons
# ---------------------------------------------------------------------------

def run_backtest(results_dir, horizons, yf_client):
    """Run backtest for all snapshots × horizons. Returns list of result dicts."""
    all_results = load_results(results_dir)
    if not all_results:
        print("No results files found in", results_dir)
        return []

    print(f"Loaded {len(all_results)} snapshot(s).")
    for r in all_results:
        print(f"  {r.get('date', '?')}: {r.get('count', 0)} stocks")

    metrics = []
    skipped = 0

    for run in all_results:
        run_date_str = run.get('date')
        if not run_date_str:
            continue
        run_dt = datetime.strptime(run_date_str, '%Y-%m-%d')

        for h in horizons:
            eval_dt = run_dt + timedelta(days=h)
            if eval_dt > datetime.now():
                skipped += 1
                print(f"  {run_date_str} + {h}d → {eval_dt.date()}: skipped (future)")
                continue

            print(f"\n  Analyzing {run_date_str} + {h}d → {eval_dt.date()} ...")
            result = analyze_run(run, h, yf_client)
            if result:
                metrics.append(result)

    if skipped and not metrics:
        print(f"\nAll {skipped} snapshot-horizon pairs have evaluation dates in the future.")
        print("Continue running analyze_stock.py over time — the backtest will activate")
        print("once enough time has passed for forward returns to be measured.")

    return metrics


# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------

def build_backtest_excel(all_metrics, filename):
    """Write backtest results to a styled Excel workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    gray = PatternFill(start_color='D9D9D9', end_color='D9D9D9', fill_type='solid')
    white = PatternFill(start_color='FFFFFF', end_color='FFFFFF', fill_type='solid')
    hdr_font = Font(bold=True, color='000000', size=11)
    hdr_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
    data_font = Font(color='000000')
    green_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
    red_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')

    def write_header(ws, headers):
        for ci, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=ci, value=h)
            cell.font = hdr_font
            cell.fill = gray
            cell.alignment = hdr_align

    def style_row(ws, row, n_cols, frozen=1):
        for ci in range(1, n_cols + 1):
            cell = ws.cell(row=row, column=ci)
            if ci <= frozen:
                cell.fill = gray
                cell.font = Font(bold=True, color='000000')
            else:
                cell.fill = white
                cell.font = data_font

    def auto_width(ws, headers):
        for ci, h in enumerate(headers, 1):
            ws.column_dimensions[get_column_letter(ci)].width = max(len(h) + 3, 12)

    # ---- Summary tab ----
    ws = wb.active
    ws.title = 'Summary'
    summary_headers = [
        'Snapshot', 'Horizon (d)', '# Stocks', 'SPY Return',
        'BUY Mean', 'BUY Alpha', 'BUY Hit Rate', 'BUY #',
        'LEAN BUY Mean', 'LEAN BUY Alpha', 'LEAN BUY #',
        'HOLD Mean', 'HOLD Alpha', 'HOLD #',
        'PASS Mean', 'PASS Alpha', 'PASS #',
        'Gates ρ', 'FV MAE', 'FV Bias', 'FV ±20%',
    ]
    write_header(ws, summary_headers)

    for ri, m in enumerate(all_metrics, 2):
        b = m['buckets']
        fv = m.get('fv_metrics') or {}
        row_data = [
            m['run_date'], m['horizon'], m['n_stocks'],
            m['spy_return'],
            # BUY
            b.get('BUY', {}).get('mean'),
            b.get('BUY', {}).get('alpha'),
            b.get('BUY', {}).get('hit_rate'),
            b.get('BUY', {}).get('count'),
            # LEAN BUY
            b.get('LEAN BUY', {}).get('mean'),
            b.get('LEAN BUY', {}).get('alpha'),
            b.get('LEAN BUY', {}).get('count'),
            # HOLD
            b.get('HOLD', {}).get('mean'),
            b.get('HOLD', {}).get('alpha'),
            b.get('HOLD', {}).get('count'),
            # PASS
            b.get('PASS', {}).get('mean'),
            b.get('PASS', {}).get('alpha'),
            b.get('PASS', {}).get('count'),
            # Correlation & FV
            m.get('gates_corr'),
            fv.get('mae'), fv.get('signed_error'), fv.get('within_20'),
        ]
        for ci, val in enumerate(row_data, 1):
            ws.cell(row=ri, column=ci, value=val)
        style_row(ws, ri, len(summary_headers), frozen=2)

        # Format percentages
        for ci in [4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 19, 20, 21]:
            cell = ws.cell(row=ri, column=ci)
            if cell.value is not None:
                cell.number_format = '0.0%'
        ws.cell(row=ri, column=18).number_format = '0.000'

    auto_width(ws, summary_headers)
    ws.freeze_panes = 'C2'
    ws.row_dimensions[1].height = 13

    # ---- Detail tab ----
    ws2 = wb.create_sheet('Detail')
    detail_headers = [
        'Snapshot', 'Horizon (d)', 'Ticker', 'Rating', 'Gates Passed',
        'DCF Fair Value', 'Start Price', 'End Price',
        'Return', 'SPY Return', 'Excess Return',
    ]
    write_header(ws2, detail_headers)

    row_num = 2
    for m in all_metrics:
        for d in m['details']:
            row_data = [
                m['run_date'], m['horizon'], d['ticker'], d['rating'],
                d['gates_passed'], d.get('dcf_fv'), d['start_price'],
                d['end_price'], d['return'], d['spy_return'],
                d['excess_return'],
            ]
            for ci, val in enumerate(row_data, 1):
                ws2.cell(row=row_num, column=ci, value=val)
            style_row(ws2, row_num, len(detail_headers), frozen=3)

            # Format
            for ci in [6, 7, 8]:
                ws2.cell(row=row_num, column=ci).number_format = '"$"#,##0.00'
            for ci in [9, 10, 11]:
                ws2.cell(row=row_num, column=ci).number_format = '0.0%'

            # Color excess return
            cell = ws2.cell(row=row_num, column=11)
            if cell.value is not None:
                cell.fill = green_fill if cell.value > 0 else red_fill

            row_num += 1

    auto_width(ws2, detail_headers)
    ws2.freeze_panes = 'D2'
    for r in range(1, row_num):
        ws2.row_dimensions[r].height = 13

    # ---- Stats tab (aggregated across all snapshots) ----
    ws3 = wb.create_sheet('Stats')
    ws3.cell(row=1, column=1, value='Aggregated Rating Performance')
    ws3.cell(row=1, column=1).font = Font(bold=True, size=13)

    stats_headers = ['Rating', 'Total Obs', 'Mean Return', 'Median Return',
                     'Mean Alpha', 'Hit Rate', 'Avg Gates']
    for ci, h in enumerate(stats_headers, 1):
        cell = ws3.cell(row=3, column=ci, value=h)
        cell.font = hdr_font
        cell.fill = gray
        cell.alignment = hdr_align

    # Aggregate across all runs
    agg = defaultdict(lambda: {'returns': [], 'alphas': [], 'hits': 0, 'n': 0})
    all_gates = []
    all_returns = []

    for m in all_metrics:
        spy_ret = m['spy_return']
        for d in m['details']:
            rating = d['rating']
            agg[rating]['returns'].append(d['return'])
            agg[rating]['alphas'].append(d['excess_return'])
            if d['return'] > spy_ret:
                agg[rating]['hits'] += 1
            agg[rating]['n'] += 1
            if d.get('gates_num', -1) >= 0:
                all_gates.append(d['gates_num'])
                all_returns.append(d['return'])

    ri = 4
    for rating in RATING_ORDER:
        a = agg.get(rating)
        if not a or not a['returns']:
            continue
        rets = np.array(a['returns'])
        alps = np.array(a['alphas'])
        ws3.cell(row=ri, column=1, value=rating)
        ws3.cell(row=ri, column=2, value=a['n'])
        ws3.cell(row=ri, column=3, value=float(np.mean(rets)))
        ws3.cell(row=ri, column=4, value=float(np.median(rets)))
        ws3.cell(row=ri, column=5, value=float(np.mean(alps)))
        ws3.cell(row=ri, column=6, value=a['hits'] / a['n'] if a['n'] else 0)
        ws3.cell(row=ri, column=7, value='')
        style_row(ws3, ri, 7, frozen=1)
        for ci in [3, 4, 5, 6]:
            ws3.cell(row=ri, column=ci).number_format = '0.0%'
        ri += 1

    # Gates-passed correlation (aggregated)
    ri += 1
    ws3.cell(row=ri, column=1, value='Gates-Passed Correlation')
    ws3.cell(row=ri, column=1).font = Font(bold=True, size=13)
    ri += 1
    if len(all_gates) >= 5:
        n = len(all_gates)
        rank_g = rank(all_gates)
        rank_r = rank(all_returns)
        d_sq = sum((rg - rr) ** 2 for rg, rr in zip(rank_g, rank_r))
        agg_rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1)) if n > 1 else 0.0
        ws3.cell(row=ri, column=1, value='Spearman ρ (gates vs return)')
        ws3.cell(row=ri, column=2, value=round(agg_rho, 4))
        ws3.cell(row=ri, column=2).number_format = '0.0000'
        ri += 1
        ws3.cell(row=ri, column=1, value='Observations')
        ws3.cell(row=ri, column=2, value=n)
    else:
        ws3.cell(row=ri, column=1, value='Not enough data yet')

    # FV accuracy (aggregated)
    ri += 2
    ws3.cell(row=ri, column=1, value='Fair Value Accuracy')
    ws3.cell(row=ri, column=1).font = Font(bold=True, size=13)
    ri += 1
    all_fv_preds = []
    all_fv_acts = []
    for m in all_metrics:
        for d in m['details']:
            fv = d.get('dcf_fv')
            ep = d.get('end_price')
            if fv and fv > 0 and ep and ep > 0:
                all_fv_preds.append(fv)
                all_fv_acts.append(ep)

    if len(all_fv_preds) >= 5:
        preds = np.array(all_fv_preds)
        acts = np.array(all_fv_acts)
        pct_err = (preds - acts) / acts
        metrics_labels = [
            ('Mean Absolute Error', float(np.mean(np.abs(pct_err)))),
            ('Mean Signed Error (bias)', float(np.mean(pct_err))),
            ('Within ±20%', float(np.sum(np.abs(pct_err) <= 0.20) / len(pct_err))),
            ('Observations', len(pct_err)),
        ]
        for label, val in metrics_labels:
            ws3.cell(row=ri, column=1, value=label)
            ws3.cell(row=ri, column=2, value=val)
            if isinstance(val, float):
                ws3.cell(row=ri, column=2).number_format = '0.0%'
            ri += 1
    else:
        ws3.cell(row=ri, column=1, value='Not enough data yet')

    auto_width(ws3, stats_headers)
    ws3.column_dimensions['A'].width = 30
    ws3.freeze_panes = 'A4'
    for r in range(1, ri + 1):
        ws3.row_dimensions[r].height = 13

    # ---- Sector Accuracy tab ----
    ws4 = wb.create_sheet('Sector Accuracy')
    sec_acc = sector_accuracy(all_metrics)
    sec_headers = ['Sector', 'Count', 'Mean Return', 'Mean Alpha', 'Hit Rate']
    write_header(ws4, sec_headers)

    ri = 2
    for sector in sorted(sec_acc, key=lambda s: sec_acc[s]['mean_alpha'], reverse=True):
        sa = sec_acc[sector]
        ws4.cell(row=ri, column=1, value=sector)
        ws4.cell(row=ri, column=2, value=sa['count'])
        ws4.cell(row=ri, column=3, value=sa['mean_return'])
        ws4.cell(row=ri, column=4, value=sa['mean_alpha'])
        ws4.cell(row=ri, column=5, value=sa['hit_rate'])
        style_row(ws4, ri, 5, frozen=1)
        for ci in [3, 4, 5]:
            ws4.cell(row=ri, column=ci).number_format = '0.0%'
        ri += 1

    auto_width(ws4, sec_headers)
    ws4.column_dimensions['A'].width = 25
    ws4.freeze_panes = 'B2'
    for r in range(1, ri):
        ws4.row_dimensions[r].height = 13

    # ---- MoS Buckets tab ----
    ws5 = wb.create_sheet('MoS Buckets')
    mos_acc = mos_bucket_accuracy(all_metrics)
    mos_headers = ['MoS Bucket', 'Count', 'Mean Return', 'Hit Rate']
    write_header(ws5, mos_headers)

    ri = 2
    for bucket in ['>30%', '10-30%', '0-10%', 'Negative']:
        if bucket in mos_acc:
            ma = mos_acc[bucket]
            ws5.cell(row=ri, column=1, value=bucket)
            ws5.cell(row=ri, column=2, value=ma['count'])
            ws5.cell(row=ri, column=3, value=ma['mean_return'])
            ws5.cell(row=ri, column=4, value=ma['hit_rate'])
            style_row(ws5, ri, 4, frozen=1)
            for ci in [3, 4]:
                ws5.cell(row=ri, column=ci).number_format = '0.0%'
            ri += 1

    auto_width(ws5, mos_headers)
    ws5.freeze_panes = 'B2'
    for r in range(1, ri):
        ws5.row_dimensions[r].height = 13

    # ---- Consensus tab ----
    ws6 = wb.create_sheet('Consensus')
    ws6.cell(row=1, column=1, value='Model vs Analyst Targets')
    ws6.cell(row=1, column=1).font = Font(bold=True, size=13)

    consensus = consensus_comparison(all_metrics)
    buy_hit = strong_buy_hit_rate(all_metrics)

    ri = 3
    if buy_hit is not None:
        ws6.cell(row=ri, column=1, value='BUY Hit Rate (beat SPY)')
        ws6.cell(row=ri, column=1).font = hdr_font
        ws6.cell(row=ri, column=2, value=buy_hit)
        ws6.cell(row=ri, column=2).number_format = '0.0%'
        ri += 2

    if consensus:
        ws6.cell(row=ri, column=1, value='Metric')
        ws6.cell(row=ri, column=1).font = hdr_font
        ws6.cell(row=ri, column=1).fill = gray
        ws6.cell(row=ri, column=2, value='Value')
        ws6.cell(row=ri, column=2).font = hdr_font
        ws6.cell(row=ri, column=2).fill = gray
        ri += 1
        metrics_data = [
            ('Mean Bias (Model/Target − 1)', consensus['mean_bias']),
            ('Median Bias', consensus['median_bias']),
            ('# Stocks Compared', consensus['n_stocks']),
        ]
        for label, val in metrics_data:
            ws6.cell(row=ri, column=1, value=label)
            ws6.cell(row=ri, column=2, value=val)
            if isinstance(val, float):
                ws6.cell(row=ri, column=2).number_format = '0.0%'
            style_row(ws6, ri, 2, frozen=1)
            ri += 1
    else:
        ws6.cell(row=ri, column=1, value='Not enough data for consensus comparison')

    ws6.column_dimensions['A'].width = 35
    ws6.column_dimensions['B'].width = 15
    for r in range(1, ri + 1):
        ws6.row_dimensions[r].height = 13

    wb.save(filename)
    return filename


# ---------------------------------------------------------------------------
# Enhanced analytics
# ---------------------------------------------------------------------------

def sector_accuracy(all_metrics):
    """Compute per-sector return and hit rate stats across all runs.

    Returns dict: {sector: {'mean_return': float, 'hit_rate': float,
                             'count': int, 'mean_alpha': float}}
    """
    sector_data = defaultdict(lambda: {'returns': [], 'alphas': []})

    for m in all_metrics:
        spy_ret = m['spy_return']
        stocks = m.get('_source_stocks', {})
        for d in m['details']:
            ticker = d['ticker']
            stock_info = stocks.get(ticker, {})
            sector = stock_info.get('sector', 'Unknown')
            sector_data[sector]['returns'].append(d['return'])
            sector_data[sector]['alphas'].append(d['excess_return'])

    result = {}
    for sector, data in sector_data.items():
        rets = np.array(data['returns'])
        alps = np.array(data['alphas'])
        result[sector] = {
            'mean_return': float(np.mean(rets)),
            'hit_rate': float(np.sum(alps > 0) / len(alps)) if len(alps) > 0 else 0,
            'count': len(rets),
            'mean_alpha': float(np.mean(alps)),
        }
    return result


def mos_bucket_accuracy(all_metrics):
    """Compute accuracy by margin-of-safety buckets.

    Buckets: >30%, 10-30%, 0-10%, negative.
    Returns dict: {bucket_label: {'mean_return': float, 'hit_rate': float, 'count': int}}
    """
    buckets = {
        '>30%': {'returns': [], 'alphas': []},
        '10-30%': {'returns': [], 'alphas': []},
        '0-10%': {'returns': [], 'alphas': []},
        'Negative': {'returns': [], 'alphas': []},
    }

    for m in all_metrics:
        stocks = m.get('_source_stocks', {})
        for d in m['details']:
            ticker = d['ticker']
            stock_info = stocks.get(ticker, {})
            fv = stock_info.get('dcf_fv') or d.get('dcf_fv')
            price = d.get('start_price')
            if fv and fv > 0 and price and price > 0:
                mos = (fv - price) / fv
                if mos > 0.30:
                    key = '>30%'
                elif mos > 0.10:
                    key = '10-30%'
                elif mos >= 0:
                    key = '0-10%'
                else:
                    key = 'Negative'
                buckets[key]['returns'].append(d['return'])
                buckets[key]['alphas'].append(d['excess_return'])

    result = {}
    for label, data in buckets.items():
        if data['returns']:
            rets = np.array(data['returns'])
            alps = np.array(data['alphas'])
            result[label] = {
                'mean_return': float(np.mean(rets)),
                'hit_rate': float(np.sum(alps > 0) / len(alps)) if len(alps) > 0 else 0,
                'count': len(rets),
            }
    return result


def strong_buy_hit_rate(all_metrics):
    """What percentage of BUY-rated stocks actually outperformed SPY?

    Returns float (0-1) or None if no BUY stocks.
    """
    buy_alphas = []
    for m in all_metrics:
        for d in m['details']:
            if d['rating'] == 'BUY':
                buy_alphas.append(d['excess_return'])
    if not buy_alphas:
        return None
    return sum(1 for a in buy_alphas if a > 0) / len(buy_alphas)


def consensus_comparison(all_metrics):
    """Compare model fair values vs analyst target prices.

    Returns dict with: mean_bias, median_bias, n_stocks.
    """
    biases = []
    for m in all_metrics:
        stocks = m.get('_source_stocks', {})
        for d in m['details']:
            ticker = d['ticker']
            stock_info = stocks.get(ticker, {})
            model_fv = stock_info.get('dcf_fv') or d.get('dcf_fv')
            target_mean = stock_info.get('target_mean')
            if model_fv and model_fv > 0 and target_mean and target_mean > 0:
                biases.append(model_fv / target_mean - 1)
    if not biases:
        return None
    arr = np.array(biases)
    return {
        'mean_bias': float(np.mean(arr)),
        'median_bias': float(np.median(arr)),
        'n_stocks': len(arr),
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(all_metrics):
    """Print a concise console summary of backtest results."""
    if not all_metrics:
        return

    print(f"\n{'='*70}")
    print("BACKTEST RESULTS")
    print(f"{'='*70}")

    for m in all_metrics:
        b = m['buckets']
        print(f"\n{m['run_date']} + {m['horizon']}d  ({m['n_stocks']} stocks, SPY {m['spy_return']:+.1%})")
        print(f"  {'Rating':<12s} {'Mean':>8s} {'Alpha':>8s} {'Hit%':>6s} {'#':>4s}")
        print(f"  {'-'*40}")
        for rating in RATING_ORDER:
            if rating in b:
                r = b[rating]
                print(f"  {rating:<12s} {r['mean']:>+7.1%} {r['alpha']:>+7.1%}"
                      f" {r['hit_rate']:>5.0%} {r['count']:>4d}")

        if m.get('gates_corr') is not None:
            print(f"  Gates ρ: {m['gates_corr']:.3f}")
        fv = m.get('fv_metrics')
        if fv:
            print(f"  FV MAE: {fv['mae']:.1%}  Bias: {fv['signed_error']:+.1%}"
                  f"  ±20%: {fv['within_20']:.0%}")

    # --- Enhanced analytics (aggregated across all runs) ---
    print(f"\n{'='*70}")
    print("ENHANCED ANALYTICS")
    print(f"{'='*70}")

    # Strong BUY hit rate
    buy_hit = strong_buy_hit_rate(all_metrics)
    if buy_hit is not None:
        print(f"\n  BUY Hit Rate (beat SPY): {buy_hit:.0%}")

    # Sector accuracy
    sec_acc = sector_accuracy(all_metrics)
    if sec_acc:
        print(f"\n  {'Sector':<25s} {'Mean Ret':>9s} {'Alpha':>8s} {'Hit%':>6s} {'#':>4s}")
        print(f"  {'-'*54}")
        for sector in sorted(sec_acc, key=lambda s: sec_acc[s]['mean_alpha'], reverse=True):
            sa = sec_acc[sector]
            print(f"  {sector:<25s} {sa['mean_return']:>+8.1%} {sa['mean_alpha']:>+7.1%}"
                  f" {sa['hit_rate']:>5.0%} {sa['count']:>4d}")

    # MoS bucket accuracy
    mos_acc = mos_bucket_accuracy(all_metrics)
    if mos_acc:
        print(f"\n  {'MoS Bucket':<12s} {'Mean Ret':>9s} {'Hit%':>6s} {'#':>4s}")
        print(f"  {'-'*33}")
        for bucket in ['>30%', '10-30%', '0-10%', 'Negative']:
            if bucket in mos_acc:
                ma = mos_acc[bucket]
                print(f"  {bucket:<12s} {ma['mean_return']:>+8.1%}"
                      f" {ma['hit_rate']:>5.0%} {ma['count']:>4d}")

    # Consensus comparison
    consensus = consensus_comparison(all_metrics)
    if consensus:
        print(f"\n  Model vs Analyst Targets ({consensus['n_stocks']} stocks):")
        print(f"    Mean bias:   {consensus['mean_bias']:+.1%}")
        print(f"    Median bias: {consensus['median_bias']:+.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backtest stock model predictions')
    parser.add_argument('--results-dir', default='output',
                        help='Directory containing results_*.json files')
    parser.add_argument('--horizons', default='30,90,180',
                        help='Comma-separated horizon days (default: 30,90,180)')
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(',')]

    from data.yfinance_client import YFinanceClient
    yf_client = YFinanceClient()

    all_metrics = run_backtest(args.results_dir, horizons, yf_client)

    if all_metrics:
        print_summary(all_metrics)

        os.makedirs('output', exist_ok=True)
        xlsx = os.path.join('output', f'backtest_{date.today().isoformat()}.xlsx')
        build_backtest_excel(all_metrics, xlsx)
        print(f"\nExcel: {xlsx}")
    else:
        print("\nNo backtest results to report yet.")
        print("As you accumulate snapshots over time, the backtest will activate.")
