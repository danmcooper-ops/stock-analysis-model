# scripts/crypto_market.py
"""Build Investing › Crypto's market table from CoinGecko.

    python scripts/crypto_market.py                       # today, top 20
    python scripts/crypto_market.py --top 20 --out-dir output/crypto

Writes ``<out-dir>/crypto_report_<date>.json`` (``models/crypto_market``),
which ``report_html._stage_crypto_report`` stages beside the report as
``crypto.json`` for the report's tab and the iOS app. Prices are live at the
moment it runs, so the nightly step runs it only when the run date is today
(``run_daily.sh`` sidemodels). About 21 CoinGecko calls on a warm category
cache, ~5 minutes keyless (``COINGECKO_API_KEY`` makes it ~1).

Exit code: 0 written; 1 nothing written (the market list failed, or no
coin had a history). A coin whose history fails keeps its row with the
history-based columns blank, and the error is listed in the report.
"""
import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.crypto_client import CoinGeckoClient, CoinGeckoError  # noqa: E402
from models import crypto_market as cm  # noqa: E402

logger = logging.getLogger('crypto_market')
NY = ZoneInfo('America/New_York')
# Coins looked at to find TOP_N that are not stablecoins or tokenized claims;
# on 2026-11-22 the 20th kept coin was CoinGecko's #27.
CANDIDATES = 60


def load_spy(prices_dir):
    """SPY's daily closes ``[(date, close)]`` from the price parquet, or None."""
    path = os.path.join(prices_dir, 'SPY.parquet')
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(path, columns=['Close'])
    except Exception as e:  # display only: the S&P column goes blank
        logger.warning('SPY prices unreadable (%s): %s', path, e)
        return None
    s = df['Close'].dropna()
    return [(ts.date(), float(v)) for ts, v in s.items() if v > 0]


def _looks_like_dollar(m):
    """Fallback when a coin's categories cannot be fetched: a price within 2%
    of $1 is treated as a dollar stablecoin rather than shown."""
    p = m.get('current_price')
    return isinstance(p, (int, float)) and abs(p - 1.0) < 0.02


def build(client, as_of, top_n, prices_dir):
    errors = []
    markets = client.markets(CANDIDATES)
    categories, kept = {}, []
    for m in markets:
        if len(kept) >= top_n:
            break
        cid = m.get('id')
        try:
            categories[cid] = client.categories(cid)
        except CoinGeckoError as e:
            errors.append(f'{cid}: categories: {e}')
            if _looks_like_dollar(m):
                categories[cid] = ['Stablecoins (price near $1; categories unavailable)']
        if not cm.excluded_reason(cid, categories.get(cid)):
            kept.append(cid)
    histories = {}
    for cid in (['bitcoin'] if 'bitcoin' not in kept else []) + kept:
        try:
            histories[cid] = client.history(cid)
        except CoinGeckoError as e:
            errors.append(f'{cid}: history: {e}')
            logger.warning('history %s failed: %s', cid, e)
    generated = datetime.now(NY).strftime('%Y-%m-%d %H:%M %Z')
    rep = cm.build_report(markets, histories, categories, as_of, generated,
                          spy=load_spy(prices_dir), top_n=top_n, errors=errors)
    rep['calls'] = client.calls
    return rep


def write_report(rep, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"crypto_report_{rep['run_date']}.json")
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(rep, f, separators=(',', ':'), allow_nan=False)
    os.replace(tmp, path)
    return path


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--run-date', type=date.fromisoformat,
                    default=datetime.now(NY).date())
    ap.add_argument('--top', type=int, default=cm.TOP_N)
    ap.add_argument('--out-dir', default=os.path.join(root, 'output', 'crypto'))
    ap.add_argument('--prices-dir', default=os.path.join(root, 'output', 'prices'))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    client = CoinGeckoClient()
    try:
        rep = build(client, a.run_date, a.top, a.prices_dir)
    except CoinGeckoError as e:
        print(f'crypto: market list failed: {e}')
        return 1
    if not rep['rows'] or not any(r['history_days'] for r in rep['rows']):
        print(f"crypto: no usable rows ({len(rep['rows'])} rows, errors: {rep['errors']})")
        return 1
    path = write_report(rep, a.out_dir)
    print(f"crypto: {rep['total']} coins, {len(rep['excluded'])} skipped "
          f"({', '.join(e['symbol'] for e in rep['excluded'])}), "
          f"{len(rep['errors'])} errors, {rep['calls']} calls -> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
