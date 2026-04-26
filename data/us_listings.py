# data/us_listings.py
"""US-listed ticker universe from SEC EDGAR's company_tickers.json.

The SEC publishes a single authoritative list of every ticker registered
with the Commission. This module pulls that list, filters out non-equity
securities (warrants, rights, units, preferreds), and caches the result
locally so analyze_stock runs don't re-download every time.

Free, no-auth — only requirement is a contact email in the User-Agent.
"""
import csv
import json
import os
import ssl
import urllib.request
from datetime import date, datetime

def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

_SSL_CTX = _ssl_context()

_SEC_TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
_DEFAULT_CACHE = 'data/cache/us_listings.csv'
_DEFAULT_MAX_AGE_DAYS = 7


def _is_excluded(ticker):
    """Reject anything that isn't operating-company common stock.

    Dash-suffixed tickers are preferred series (P*), warrants (W*), units (U*),
    or rights (R*) — except single/short-letter suffixes A/B/C/V which are
    dual-class commons (BRK-B, BF-B, BIO-B, etc.).

    Non-dash 5+ char tickers ending in U/R/W are typically SPAC warrants,
    units, or rights without a dash separator.
    """
    if not ticker:
        return True
    if '-' in ticker:
        suffix = ticker.split('-', 1)[1]
        if not suffix:
            return True
        return suffix[0] in ('P', 'W', 'U', 'R')
    if len(ticker) >= 5 and ticker[-1] in ('U', 'R', 'W'):
        return True
    return False


def _read_cache(cache_path):
    with open(cache_path) as f:
        return [row['ticker'] for row in csv.DictReader(f) if row.get('ticker')]


def _cache_age_days(cache_path):
    mtime = datetime.fromtimestamp(os.path.getmtime(cache_path)).date()
    return (date.today() - mtime).days


def fetch_us_listed_tickers(email='stockanalysis@example.com',
                            cache_path=_DEFAULT_CACHE,
                            max_age_days=_DEFAULT_MAX_AGE_DAYS,
                            force=False):
    """Return a sorted list of US-listed equity tickers.

    Reads from cache_path if present and younger than max_age_days; otherwise
    fetches from SEC EDGAR and writes the cache.

    Parameters
    ----------
    email : str
        Contact email for SEC User-Agent header (SEC requires identification).
    cache_path : str
        CSV cache location. Defaults to data/cache/us_listings.csv.
    max_age_days : int
        Refresh threshold. Default 7 days.
    force : bool
        Skip cache and refetch.
    """
    if not force and os.path.exists(cache_path):
        if _cache_age_days(cache_path) < max_age_days:
            return _read_cache(cache_path)

    ua = f'StockAnalyzer/1.0 ({email})'
    req = urllib.request.Request(_SEC_TICKERS_URL, headers={'User-Agent': ua})
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as resp:
        raw = json.loads(resp.read().decode('utf-8'))

    seen = set()
    rows = []
    for entry in raw.values():
        t = (entry.get('ticker') or '').upper().strip()
        if not t or t in seen or _is_excluded(t):
            continue
        seen.add(t)
        rows.append({
            'ticker': t,
            'cik': str(entry.get('cik_str', '')),
            'name': entry.get('title', ''),
        })
    rows.sort(key=lambda r: r['ticker'])

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ticker', 'cik', 'name'])
        w.writeheader()
        w.writerows(rows)

    return [r['ticker'] for r in rows]
