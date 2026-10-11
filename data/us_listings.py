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
import logging
import os
import ssl
import urllib.request
from datetime import date, datetime

logger = logging.getLogger(__name__)

def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception as e:
        logger.debug(f"us_listings: certifi unavailable, using system trust store: {e}")
        # Fall back to the system trust store — NEVER disable verification:
        # unverified TLS would let a MITM feed fabricated financial data
        # into the pipeline silently.
        return ssl.create_default_context()

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
    with open(cache_path, encoding='utf-8') as f:
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
    with open(cache_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ticker', 'cik', 'name'])
        w.writeheader()
        w.writerows(rows)

    return [r['ticker'] for r in rows]


# ---------------------------------------------------------------------------
# Issuer map: which listings belong to one company
# ---------------------------------------------------------------------------
# company_tickers.json names every ticker registered to a filer, so the
# universe carries one company several times: an OTC line of a foreign
# ordinary share beside its NYSE ADR (NONOF/NVO), an OTC ADR beside the
# ordinary line, preferred series and notes with no dash in the symbol
# (FNMAO, FMCKP, TBB), a filer's exchange-traded notes (BRZL and FNGD under
# BMO), and second share classes (BRK-A, GOOG). On 2026-10-07 that was 226
# of 2,508 rows across 181 companies, and 54 of those companies carried
# different ratings on different lines (MUFG HOLD at +98% MoS, MBFJF PASS
# at -88%). The exchange-tagged list gives each ticker its CIK, its
# exchange and SEC's order, which lists a filer's primary ticker first.
_SEC_TICKERS_EXCHANGE_URL = 'https://www.sec.gov/files/company_tickers_exchange.json'
_DEFAULT_ISSUER_CACHE = 'data/cache/sec_issuers.csv'


def fetch_issuer_map(email='stockanalysis@example.com',
                     cache_path=_DEFAULT_ISSUER_CACHE,
                     max_age_days=_DEFAULT_MAX_AGE_DAYS,
                     force=False):
    """``{ticker: {'cik': str, 'exchange': str|None, 'rank': int}}``.

    ``rank`` is the ticker's position in SEC's list (lower = listed first).
    A failed fetch with no usable cache returns ``{}`` — the caller then
    leaves the universe as it is, never fails a run over it.
    """
    if not force and os.path.exists(cache_path) and \
            _cache_age_days(cache_path) < max_age_days:
        return _read_issuer_cache(cache_path)
    try:
        ua = f'StockAnalyzer/1.0 ({email})'
        req = urllib.request.Request(_SEC_TICKERS_EXCHANGE_URL,
                                     headers={'User-Agent': ua})
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=30) as resp:
            raw = json.loads(resp.read().decode('utf-8'))
        fields = raw['fields']
        records = [dict(zip(fields, rec, strict=False)) for rec in raw['data']]
    except Exception as e:
        logger.warning("us_listings: SEC issuer list unavailable (%s); %s", e,
                       'using the stale cache' if os.path.exists(cache_path)
                       else 'listings will not be collapsed')
        return _read_issuer_cache(cache_path) if os.path.exists(cache_path) else {}
    rows, seen = [], set()
    for i, rec in enumerate(records):
        t = (rec.get('ticker') or '').upper().strip()
        if not t or t in seen or rec.get('cik') is None:
            continue
        seen.add(t)
        rows.append({'ticker': t, 'cik': str(rec['cik']),
                     'exchange': rec.get('exchange') or '', 'rank': i})
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    tmp = cache_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ticker', 'cik', 'exchange', 'rank'])
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, cache_path)
    return {r['ticker']: {'cik': r['cik'], 'exchange': r['exchange'] or None,
                          'rank': r['rank']} for r in rows}


def _read_issuer_cache(cache_path):
    with open(cache_path, encoding='utf-8') as f:
        return {row['ticker']: {'cik': row['cik'],
                                'exchange': row.get('exchange') or None,
                                'rank': int(row.get('rank') or 0)}
                for row in csv.DictReader(f) if row.get('ticker')}
