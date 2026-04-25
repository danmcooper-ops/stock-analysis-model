"""Backfill edgar_history series into an existing results JSON file.

Usage:
    python scripts/backfill_edgar_hist.py [results_YYYY-MM-DD.json]

If no file is given, uses the most recent results_*.json in output/.

Fetches revenue_history, earnings_history, operating_cf_history,
capex_history, gross_profit_history, shares_history, dividends_paid_history
from SEC EDGAR XBRL for any ticker whose stored edgar_history is missing
those series (i.e., only has 'years_available').

Saves the patched JSON in-place (overwrites) then rebuilds the HTML.
"""
import json
import os
import ssl
import sys
import urllib.request
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.sec_xbrl_client import SECXBRLClient, _SSL_CTX

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

EMAIL = 'nurses.public1y@icloud.com'
CIK_URL = 'https://www.sec.gov/files/company_tickers.json'
REQUEST_DELAY = 0.15   # 6-7 req/sec — well within SEC limit of 10/sec


def _fetch_cik_map():
    print('[backfill] fetching SEC CIK map...')
    req = urllib.request.Request(CIK_URL, headers={'User-Agent': f'StockAnalyzer/1.0 ({EMAIL})'})
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=20) as resp:
        data = json.loads(resp.read())
    cik_map  = {e['ticker']: str(e['cik_str']).zfill(10) for e in data.values()}
    name_map = {e['ticker']: e.get('title', '') for e in data.values()}
    print(f'[backfill] loaded {len(cik_map)} CIKs')
    return cik_map, name_map


def _pick_json():
    if len(sys.argv) > 1:
        return sys.argv[1]
    pattern = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'output', 'results_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print('[backfill] no results_*.json found in output/')
        sys.exit(1)
    return files[-1]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    json_path = _pick_json()
    print(f'[backfill] patching {json_path}')

    with open(json_path) as f:
        doc = json.load(f)

    rows = doc.get('results', doc) if isinstance(doc, dict) else doc
    if not isinstance(rows, list):
        print('[backfill] unexpected JSON structure'); sys.exit(1)

    # Identify tickers that need backfilling
    need = []
    for r in rows:
        tk = r.get('ticker')
        if not tk:
            continue
        eh = r.get('edgar_history') or {}
        # Missing if no revenue_history series (even if years_available is set)
        if not eh.get('revenue_history') and not eh.get('earnings_history'):
            need.append(tk)

    print(f'[backfill] {len(need)}/{len(rows)} tickers need edgar_history backfill')
    if not need:
        print('[backfill] nothing to do'); return

    cik_map, name_map = _fetch_cik_map()
    client = SECXBRLClient(cik_map, name_map, email=EMAIL, request_delay=REQUEST_DELAY)

    ok = err = skip = 0
    row_by_ticker = {r['ticker']: r for r in rows if r.get('ticker')}

    for i, tk in enumerate(need):
        print(f'[backfill] {i+1}/{len(need)} {tk}', end='  ', flush=True)
        try:
            hist = client.fetch_historical_financials(tk, min_years=5)
            if hist:
                row_by_ticker[tk]['edgar_history'] = hist
                ok += 1
                series_found = [k for k in hist if k != 'years_available' and hist[k]]
                print(f'ok ({", ".join(series_found[:4])})')
            else:
                skip += 1
                print('no data')
        except Exception as e:
            err += 1
            print(f'error: {e}')

    print(f'[backfill] done — ok={ok} skip={skip} err={err}')

    # Save patched JSON
    with open(json_path, 'w') as f:
        json.dump(doc, f, indent=2, default=str)
    print(f'[backfill] saved {json_path}')

    # Rebuild HTML
    try:
        from scripts.report_html import build_html
        html_path = json_path.replace('results_', 'stock_analysis_results_').replace('.json', '.html')
        prices_dir = os.path.join(os.path.dirname(os.path.abspath(json_path)), 'prices')
        if not os.path.isdir(prices_dir):
            prices_dir = None
        build_html(rows, html_path, prices_dir=prices_dir)
        print(f'[backfill] HTML rebuilt → {html_path}')
    except Exception as e:
        print(f'[backfill] HTML rebuild failed: {e}')


if __name__ == '__main__':
    main()
