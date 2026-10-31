"""Daily ETF holdings for the look-through read (``models/etf_lookthrough.py``).

The funds are SSGA's (SPDR) US equity ETFs, listed in ``data/etf_funds.json``
(``scripts/update_etf_list.py`` writes it from SSGA's fund finder, keeping a
fund only when most of its weight is in rows the model scores). v1's 14 —
SPY, DIA and MDY (unit investment trusts, so absent from SEC's mutual-fund
ticker file) and the 11 Select Sector SPDRs — are always included, whatever
the file says (``SPDR_FUNDS``). Rule 6c-11 makes every ETF
publish its full holdings daily; SSGA's are one xlsx per fund with the as-of
date in the header, then Name / Ticker / Identifier (CUSIP) / SEDOL /
Weight (%) / Sector / Shares Held / Local Currency.

Fetched files are parsed and archived as ``<root>/<TICKER>/<as_of>.json.gz``
(deterministic gzip, so a re-fetch of the same day does not churn). Issuer
files carry no history, so the archive is the only record a later replay can
read; ``load_holdings`` picks the newest file dated on or before the run date,
never a later one.
"""
import gzip
import io
import json
import logging
import os
import re
import urllib.request
from datetime import date, datetime

logger = logging.getLogger(__name__)

# Fund -> the report's sector (Yahoo's names, as on the sector pages), None for
# the broad funds. GICS and Yahoo's sectors differ at the edges (GICS IT
# excludes GOOGL/META/AMZN, as Yahoo Technology does), close enough for a
# sector page to show its SPDR beside it.
SPDR_FUNDS = {
    'SPY': None, 'DIA': None, 'MDY': None,
    'XLB': 'Basic Materials', 'XLC': 'Communication Services', 'XLE': 'Energy',
    'XLF': 'Financial Services', 'XLI': 'Industrials', 'XLK': 'Technology',
    'XLP': 'Consumer Defensive', 'XLRE': 'Real Estate', 'XLU': 'Utilities',
    'XLV': 'Healthcare', 'XLY': 'Consumer Cyclical',
}

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'etf_funds.json')
CATEGORIES = ('broad', 'size', 'style', 'dividend', 'factor', 'sector', 'industry', 'thematic')


def load_registry(path=REGISTRY_PATH):
    """Ticker -> ``{ticker, name, category, sector}`` from the registry file,
    with SPDR_FUNDS always present (their ``sector`` from SPDR_FUNDS, so only
    the Select Sectors carry one). A missing or unreadable file leaves just
    the 14, never an empty list."""
    out = {}
    try:
        with open(path, encoding='utf-8') as f:
            for e in json.load(f).get('funds') or []:
                t = normalize_ticker(e.get('ticker'))
                if t:
                    out[t] = {'ticker': t, 'name': e.get('name') or '',
                              'category': e.get('category') or 'broad', 'sector': None}
    except (OSError, ValueError) as e:
        logger.warning('ETF registry %s unreadable (%s): the 14 SPDRs only', path, e)
    for t, sec in SPDR_FUNDS.items():
        e = out.setdefault(t, {'ticker': t, 'name': '', 'category': 'broad'})
        e['sector'] = sec
        if sec:
            e['category'] = 'sector'
    return dict(sorted(out.items()))


def fund_tickers(registry=None):
    """Every fund the nightly fetch and the render read."""
    return list(registry if registry is not None else ETF_FUNDS)


SSGA_URL = ('https://www.ssga.com/us/en/intermediary/library-content/products/'
            'fund-data/etfs/us/holdings-daily-us-en-{t}.xlsx')
DEFAULT_ROOT = os.path.join('output', 'etf_holdings')

# A file whose weights do not sum to ~100% is a partial or malformed download
# (SSGA's weights sum to 99.8-100.2%); refuse it rather than archive it.
WEIGHT_SUM_BOUNDS = (0.95, 1.05)
MIN_HOLDINGS = 10

_CASH_NAME = re.compile(r'MONEY MARKET|DOLLAR|POUND STERLING|\bEURO\b|\bYEN\b|\bCASH\b', re.I)
_CASH_TICKERS = {'CASH_USD', 'USD'}


class HoldingsError(ValueError):
    """A holdings file that cannot be parsed or fails its sanity checks."""


def normalize_ticker(t):
    """SSGA writes class shares with a dot (BRK.B); the pipeline uses BRK-B."""
    t = (str(t).strip().upper() if t is not None else '')
    return t.replace('.', '-').replace('/', '-') if t not in ('', '-', 'NAN') else ''


ETF_FUNDS = load_registry()


def _kind(name, ticker):
    if ticker in _CASH_TICKERS or _CASH_NAME.search(str(name or '')):
        return 'cash'
    return 'security' if ticker else 'other'


def parse_ssga_xlsx(src, fund=None):
    """Parse one SSGA holdings xlsx (path, bytes or file object).

    Returns ``{fund, fund_name, as_of, holdings: [...]}`` where each holding is
    ``{name, ticker, cusip, weight, shares, currency, kind}``, ``weight`` a
    fraction and ``kind`` one of security / cash / other. Raises
    HoldingsError on a file that fails the weight-sum or size checks.
    """
    import pandas as pd
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    try:
        raw = pd.read_excel(src, header=None)
    except Exception as e:
        raise HoldingsError(f'unreadable holdings file: {e}') from e
    head = {str(raw.iloc[i, 0]).strip().rstrip(':'): raw.iloc[i, 1]
            for i in range(min(len(raw), 6))}
    m = re.search(r'(\d{1,2}-[A-Za-z]{3}-\d{4})', str(head.get('Holdings', '')))
    if not m:
        raise HoldingsError('no "As of" date in the header')
    as_of = datetime.strptime(m.group(1), '%d-%b-%Y').date().isoformat()
    hdr = raw.index[raw[0].astype(str).str.strip().eq('Name')]
    if not len(hdr):
        raise HoldingsError('no holdings header row')
    df = raw.iloc[hdr[0] + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr[0]]]
    holdings = []
    for _, h in df.iterrows():
        w = h.get('Weight')
        if not isinstance(w, (int, float)) or w != w:
            continue  # footer text and blank rows
        tk = normalize_ticker(h.get('Ticker'))
        shares = h.get('Shares Held')
        cusip = h.get('Identifier')
        holdings.append({
            'name': str(h.get('Name') or '').strip(),
            'ticker': tk,
            'cusip': str(cusip).strip() if isinstance(cusip, str) else None,
            'weight': float(w) / 100.0,
            'shares': float(shares) if isinstance(shares, (int, float)) and shares == shares else None,
            'currency': str(h.get('Local Currency') or '').strip() or None,
            'kind': _kind(h.get('Name'), tk),
        })
    total = sum(x['weight'] for x in holdings)
    if len(holdings) < MIN_HOLDINGS or not (WEIGHT_SUM_BOUNDS[0] <= total <= WEIGHT_SUM_BOUNDS[1]):
        raise HoldingsError(f'{len(holdings)} holdings summing to {total:.1%}')
    fund_tk = normalize_ticker(head.get('Ticker Symbol')) or (fund or '').upper()
    if fund and fund_tk and fund_tk != fund.upper():
        raise HoldingsError(f'file is for {fund_tk}, expected {fund.upper()}')
    return {'fund': fund_tk, 'fund_name': str(head.get('Fund Name') or '').strip(),
            'as_of': as_of, 'source': 'ssga', 'holdings': holdings}


def fetch_ssga(fund, timeout=30):
    """Download and parse one fund's SSGA holdings file."""
    req = urllib.request.Request(SSGA_URL.format(t=fund.lower()),
                                 headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return parse_ssga_xlsx(body, fund=fund)


def _path(root, fund, as_of):
    return os.path.join(root, fund.upper(), f'{as_of}.json.gz')


def save_holdings(root, parsed):
    """Archive a parsed file; returns its path. Deterministic gzip and an
    atomic replace, so a re-fetch of the same day rewrites identical bytes."""
    path = _path(root, parsed['fund'], parsed['as_of'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = json.dumps(parsed, sort_keys=True, separators=(',', ':')).encode('utf-8')
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as gz:
            gz.write(body)
    os.replace(tmp, path)
    return path


def load_holdings(root, fund, on_or_before=None):
    """The newest archived holdings for *fund* dated on or before
    *on_or_before* (a date or ISO string; None = newest), or None."""
    d = os.path.join(root, fund.upper())
    try:
        names = os.listdir(d)
    except OSError:
        return None
    cut = on_or_before.isoformat() if isinstance(on_or_before, date) else on_or_before
    dates = sorted(n[:10] for n in names
                   if re.fullmatch(r'\d{4}-\d{2}-\d{2}\.json\.gz', n)
                   and (cut is None or n[:10] <= cut))
    if not dates:
        return None
    try:
        with gzip.open(_path(root, fund, dates[-1]), 'rt', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning('ETF holdings %s %s unreadable: %s', fund, dates[-1], e)
        return None
