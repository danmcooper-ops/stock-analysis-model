"""Investing › Crypto: the market table for the largest cryptocurrencies.

Pure functions over CoinGecko's market list and daily closes. No score, no
rating and no gates: a coin's row is its price, size, returns, risk and how
it moves with bitcoin and the S&P 500. Display only.

The universe is the ``TOP_N`` largest coins by market cap, skipping coins
that are a claim on some other asset rather than a cryptocurrency of their
own (``excluded_reason``): stablecoins, wrapped / bridged / liquid-staking
copies of another coin, and tokenized gold, credit or stocks. CoinGecko's
own categories decide it, so a new stablecoin is skipped without a list to
maintain; ``ALWAYS_INCLUDE`` / ``ALWAYS_EXCLUDE`` override by coin id.

Crypto trades every day, so annualised figures use 365 periods. CoinGecko
stamps a daily close at 00:00 UTC (7 or 8 pm in New York, after the US
close), so the S&P correlation pairs each SPY week-ending close with the
coin's close stamped the next day, on weekly returns, which a few hours'
timing gap cannot distort the way daily ones would.
"""

import math
from datetime import date, timedelta

REPORT_VERSION = 1
TOP_N = 20

# Lower-case substrings of CoinGecko category names that mark a coin as a
# claim on another asset. Checked 2026-11-22 against the top 100: USDT, USDC,
# USDS, USDe, DAI, RLUSD, USD1 (stablecoins), XAUT (tokenized gold), Figure
# HELOC (tokenized private credit), WBTC / cbBTC, stETH.
EXCLUDE_PATTERNS = (
    ('stablecoin', 'stablecoin'),
    ('wrapped', 'wrapped copy of another coin'),
    ('bridged', 'bridged copy of another coin'),
    ('liquid staking', 'liquid-staking token'),
    ('liquid restaking', 'liquid-staking token'),
    ('tokenized', 'tokenized asset'),
)
ALWAYS_INCLUDE = frozenset()
ALWAYS_EXCLUDE = frozenset()

YEAR = 365
CORR_MIN_DAILY = 120
CORR_MIN_WEEKLY = 26
VOL_MIN_DAYS = 20


def excluded_reason(coin_id, categories):
    """Why a coin is left out of the table, or None to keep it."""
    if coin_id in ALWAYS_INCLUDE:
        return None
    if coin_id in ALWAYS_EXCLUDE:
        return 'excluded by hand'
    for cat in categories or ():
        c = cat.lower()
        for pat, reason in EXCLUDE_PATTERNS:
            if pat in c:
                return f'{reason} ({cat})'
    return None


# ---------------------------------------------------------------- helpers
def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct(v):
    """CoinGecko's percent (12.3) as a fraction (0.123)."""
    v = _num(v)
    return None if v is None else v / 100.0


def _log_returns(closes):
    out = []
    for a, b in zip(closes, closes[1:], strict=False):
        if a and b and a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def _stdev(xs):
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _corr(xs, ys):
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-20 or syy <= 1e-20:   # a constant series (or rounding noise on one)
        return None
    return sxy / math.sqrt(sxx * syy)


def _window(hist, as_of, days):
    """``(date, close)`` points within *days* before *as_of*, inclusive."""
    lo = as_of - timedelta(days=days)
    return [(d, p) for d, p in hist if lo <= d <= as_of]


# ---------------------------------------------------------------- measures
def volatility(hist, as_of, days):
    """Annualised standard deviation of daily log returns over *days*."""
    w = _window(hist, as_of, days)
    r = _log_returns([p for _, p in w])
    if len(r) < min(VOL_MIN_DAYS, days - 1):
        return None
    s = _stdev(r)
    return None if s is None else s * math.sqrt(YEAR)


def max_drawdown(closes):
    """The deepest peak-to-trough fall, as a negative fraction (0 = none)."""
    peak, worst = None, 0.0
    for p in closes:
        if p is None or p <= 0:
            continue
        peak = p if peak is None or p > peak else peak
        worst = min(worst, p / peak - 1.0)
    return worst if peak is not None else None


def corr_daily(hist, ref, as_of, days=YEAR):
    """Correlation of daily log returns with *ref* (both every day, UTC)."""
    a = dict(_window(hist, as_of, days))
    b = dict(_window(ref, as_of, days))
    ds = sorted(set(a) & set(b))
    xs, ys = [], []
    for d0, d1 in zip(ds, ds[1:], strict=False):
        if (d1 - d0).days != 1:
            continue
        if a[d0] > 0 and a[d1] > 0 and b[d0] > 0 and b[d1] > 0:
            xs.append(math.log(a[d1] / a[d0]))
            ys.append(math.log(b[d1] / b[d0]))
    return _corr(xs, ys) if len(xs) >= CORR_MIN_DAILY else None


def corr_weekly_spy(hist, spy, as_of, days=YEAR):
    """Correlation of weekly log returns with SPY. Each ISO week's last SPY
    close pairs with the coin's close stamped the next day (00:00 UTC, just
    after the US close)."""
    coin = dict(hist)
    lo = as_of - timedelta(days=days)
    last_of_week = {}
    for d, p in spy:
        if lo <= d <= as_of and p and p > 0:
            last_of_week[d.isocalendar()[:2]] = (d, p)
    pairs = []
    for _, (d, p) in sorted(last_of_week.items()):
        c = coin.get(d + timedelta(days=1))
        if c and c > 0:
            pairs.append((p, c))
    xs, ys = [], []
    for (s0, c0), (s1, c1) in zip(pairs, pairs[1:], strict=False):
        xs.append(math.log(c1 / c0))
        ys.append(math.log(s1 / s0))
    return _corr(xs, ys) if len(xs) >= CORR_MIN_WEEKLY else None


def ytd_return(hist, price, as_of):
    """Price against the close stamped 1 January (00:00 UTC)."""
    start = dict(hist).get(date(as_of.year, 1, 1))
    price = _num(price)
    if not start or start <= 0 or price is None:
        return None
    return price / start - 1.0


def _sig(v, n=6):
    return None if v is None else float(f'{v:.{n}g}')


# ---------------------------------------------------------------- rows
def build_row(m, hist, as_of, btc=None, spy=None):
    """One coin's row from its ``/coins/markets`` entry *m* and daily closes."""
    price = _num(m.get('current_price'))
    mcap = _num(m.get('market_cap'))
    vol = _num(m.get('total_volume'))
    circ, mx = _num(m.get('circulating_supply')), _num(m.get('max_supply'))
    year = _window(hist, as_of, YEAR)
    closes = [p for _, p in year] + ([price] if price else [])
    hi = max(closes) if closes else None
    return {
        'rank': m.get('market_cap_rank'),
        'id': m.get('id'),
        'symbol': (m.get('symbol') or '').upper(),
        'name': m.get('name') or m.get('id'),
        'price': price,
        'market_cap': mcap,
        'fdv': _num(m.get('fully_diluted_valuation')),
        'volume_24h': vol,
        'turnover': vol / mcap if vol and mcap else None,
        'circulating_supply': circ,
        'max_supply': mx,
        'supply_issued': circ / mx if circ and mx else None,
        'chg_24h': _pct(m.get('price_change_percentage_24h_in_currency',
                              m.get('price_change_percentage_24h'))),
        'chg_7d': _pct(m.get('price_change_percentage_7d_in_currency')),
        'chg_30d': _pct(m.get('price_change_percentage_30d_in_currency')),
        'chg_1y': _pct(m.get('price_change_percentage_1y_in_currency')),
        'ytd': ytd_return(hist, price, as_of),
        'vol_30d': volatility(hist, as_of, 30),
        'vol_1y': volatility(hist, as_of, YEAR),
        'max_dd_1y': max_drawdown(closes),
        'off_high_1y': price / hi - 1.0 if price and hi else None,
        'ath_off': _pct(m.get('ath_change_percentage')),
        'ath_date': (m.get('ath_date') or '')[:10] or None,
        'corr_btc_1y': corr_daily(hist, btc, as_of) if btc else None,
        'corr_spy_1y': corr_weekly_spy(hist, spy, as_of) if spy else None,
        'history_days': len(year),
        'spark': [[d.isoformat(), _sig(p)] for d, p in year],
    }


UI = {
    'rank': 'Rank by market cap on CoinGecko, stablecoins and other tokenized claims included; the table skips those, so ranks can jump.',
    'symbol': 'The coin’s ticker.',
    'name': 'The coin’s name.',
    'price': 'Last price in US dollars, aggregated across exchanges by CoinGecko, when the nightly run fetched it.',
    'market_cap': 'Circulating supply times price.',
    'fdv': 'Fully diluted value: maximum (or total) supply times price, what the market cap would be with every coin issued.',
    'volume_24h': 'Dollar volume traded over the last 24 hours across exchanges.',
    'turnover': '24-hour volume over market cap: how much of the coin changes hands in a day.',
    'supply_issued': 'Circulating supply as a share of the maximum supply. Blank when the supply has no cap.',
    'chg_24h': 'Price change over the last 24 hours.',
    'chg_7d': 'Price change over the last 7 days.',
    'chg_30d': 'Price change over the last 30 days.',
    'chg_1y': 'Price change over the last year.',
    'ytd': 'Price change since the close stamped 1 January (00:00 UTC).',
    'vol_30d': 'Annualised volatility of daily returns over the last 30 days (365 trading days a year: crypto never closes).',
    'vol_1y': 'Annualised volatility of daily returns over the last year.',
    'max_dd_1y': 'The deepest fall from a peak to a later low within the last year.',
    'off_high_1y': 'How far the price sits below its highest daily close of the last year.',
    'ath_off': 'How far the price sits below its all-time high.',
    'corr_btc_1y': 'Correlation of daily returns with bitcoin over the last year: 1 moves with it, 0 unrelated.',
    'corr_spy_1y': 'Correlation of weekly returns with the S&P 500 (SPY) over the last year. Weekly, because crypto’s daily close (00:00 UTC) falls hours after the US close.',
    'weight': 'Share of the table\u2019s combined market cap.',
    'spark': 'Daily closes over the last year, in US dollars.',
}


def build_report(markets, histories, categories, as_of, generated, spy=None,
                 top_n=TOP_N, errors=None):
    """The report the page and the app draw.

    *markets* is CoinGecko's list in rank order, *histories* ``{id: [(date,
    close)]}``, *categories* ``{id: [names]}``; a coin with no categories
    entry was not checked and is kept. Coins are taken in rank order until
    *top_n* are kept."""
    rows, excluded = [], []
    btc = histories.get('bitcoin')
    for m in markets:
        if len(rows) >= top_n:
            break
        cid = m.get('id')
        why = excluded_reason(cid, categories.get(cid))
        if why:
            excluded.append({'id': cid, 'symbol': (m.get('symbol') or '').upper(),
                             'name': m.get('name'), 'rank': m.get('market_cap_rank'),
                             'reason': why})
            continue
        rows.append(build_row(m, histories.get(cid) or [], as_of, btc=btc, spy=spy))
    total_mcap = sum(r['market_cap'] or 0 for r in rows)
    for r in rows:
        r['weight'] = r['market_cap'] / total_mcap if total_mcap and r['market_cap'] else None
    return {
        'version': REPORT_VERSION,
        'run_date': as_of.isoformat(),
        'generated': generated,
        'total': len(rows),
        'top_n': top_n,
        'source': 'CoinGecko',
        'rows': rows,
        'excluded': excluded,
        'errors': list(errors or []),
        'ui': UI,
    }
