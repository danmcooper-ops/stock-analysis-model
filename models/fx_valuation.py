"""Currencies: where each currency stands against the dollar, whether it is
cheap or dear in real terms, what it pays to hold, and how much of the
universe reports in it.

Display only (Financial Model › Currencies). Nothing here feeds a rating,
``profile_verdict`` or a snapshot row; the conversion the models use is
data/fx_convert.py.

- **Moves** are the currency's value in dollars over 1 month to 5 years
  from the FX store (positive: the currency strengthened against USD).
- **Real value** is the BIS real broad effective exchange rate: the
  currency against a trade-weighted basket, adjusted for relative
  inflation, so a high-inflation currency that falls in step with its
  prices does not read as getting cheaper. Ranked within its own trailing
  ``REER_WINDOW_YEARS``: at or under the ``CHEAP_PCT`` percentile reads
  cheap, at or over ``RICH_PCT`` rich. The series lag about three months
  and are monthly; a history under ``REER_MIN_YEARS`` is not ranked. It
  says where the currency sits against its own past, not that it will
  revert on any schedule.
- **Carry** is the 3-month interbank rate less the US one, in percentage
  points: what holding the currency instead of dollars pays, before any
  move in the rate itself. Where the OECD's 3-month series has gone stale
  (euro area, UK) the overnight rates are compared instead (€STR, SONIA
  and SOFR, or the OECD's immediate rates). A currency with neither shows
  none rather than a guess.
- **Exposure** counts the rows that report in the currency (their
  statements were translated through it), with their market cap and
  median margin of safety.

All rates in the input are USD per unit of the currency.
"""

import math
import statistics
from datetime import date, timedelta

FX_METHOD = 1
REER_WINDOW_YEARS = 20
REER_MIN_YEARS = 10
CHEAP_PCT = 20
RICH_PCT = 80
RATE_MAX_AGE_DAYS = 200   # a 3-month rate older than this is not shown
MOVE_HORIZONS = (('1m', 30), ('3m', 91), ('1y', 365), ('5y', 1826))
SPARK_DAYS = 365
SPARK_STEP_DAYS = 7

# Name and where it is used. Quoted as dollars per unit for the four that
# trade that way by convention; units per dollar for the rest.
CURRENCY_INFO = {
    'EUR': ('Euro', 'Euro area'), 'JPY': ('Japanese yen', 'Japan'),
    'GBP': ('British pound', 'United Kingdom'), 'CHF': ('Swiss franc', 'Switzerland'),
    'CNY': ('Chinese yuan', 'China'), 'CAD': ('Canadian dollar', 'Canada'),
    'AUD': ('Australian dollar', 'Australia'), 'NZD': ('New Zealand dollar', 'New Zealand'),
    'SEK': ('Swedish krona', 'Sweden'), 'NOK': ('Norwegian krone', 'Norway'),
    'DKK': ('Danish krone', 'Denmark'), 'HKD': ('Hong Kong dollar', 'Hong Kong'),
    'SGD': ('Singapore dollar', 'Singapore'), 'KRW': ('South Korean won', 'South Korea'),
    'TWD': ('Taiwan dollar', 'Taiwan'), 'INR': ('Indian rupee', 'India'),
    'IDR': ('Indonesian rupiah', 'Indonesia'), 'THB': ('Thai baht', 'Thailand'),
    'MYR': ('Malaysian ringgit', 'Malaysia'), 'PHP': ('Philippine peso', 'Philippines'),
    'VND': ('Vietnamese dong', 'Vietnam'), 'KZT': ('Kazakhstani tenge', 'Kazakhstan'),
    'MXN': ('Mexican peso', 'Mexico'), 'BRL': ('Brazilian real', 'Brazil'),
    'ARS': ('Argentine peso', 'Argentina'), 'CLP': ('Chilean peso', 'Chile'),
    'COP': ('Colombian peso', 'Colombia'), 'PEN': ('Peruvian sol', 'Peru'),
    'ZAR': ('South African rand', 'South Africa'), 'PLN': ('Polish zloty', 'Poland'),
    'HUF': ('Hungarian forint', 'Hungary'), 'CZK': ('Czech koruna', 'Czechia'),
    'TRY': ('Turkish lira', 'Turkey'), 'ILS': ('Israeli shekel', 'Israel'),
    'USD': ('US dollar', 'United States'),
}
USD_PER_UNIT_QUOTE = {'EUR', 'GBP', 'AUD', 'NZD'}


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _as_of(points, when):
    """Last ``(date, value)`` of sorted *points* on or before *when*."""
    best = None
    for d, v in points:
        if d > when:
            break
        best = (d, v)
    return best


def _points(series):
    """Sorted ``[(date, float)]`` from a ``{date|iso: value}`` mapping or a
    date-indexed pandas Series."""
    if series is None:
        return []
    out = []
    for k, v in series.items():
        v = _num(v)
        if v is None:
            continue
        if isinstance(k, date):
            d = k.date() if callable(getattr(k, 'date', None)) else k
        else:
            d = date.fromisoformat(str(k)[:10])
        out.append((d, v))
    out.sort()
    return out


def moves(spot_points, run_date):
    """``{horizon: change}`` in the currency's dollar value, plus the spot
    as of *run_date* and its bar date."""
    now = _as_of(spot_points, run_date)
    if now is None or (run_date - now[0]).days > 7:
        return None
    out = {'spot': now[1], 'spot_date': now[0].isoformat()}
    for key, days in MOVE_HORIZONS:
        then = _as_of(spot_points, run_date - timedelta(days=days))
        ok = then is not None and (run_date - timedelta(days=days) - then[0]).days <= 7
        out[key] = (now[1] / then[1] - 1) if ok and then[1] > 0 else None
    return out


def real_value(reer_points, run_date):
    """Percentile and z-score of the latest real effective rate within its
    own trailing window, and the cheap/fair/rich read."""
    pts = [(d, v) for d, v in reer_points if d <= run_date]
    if not pts:
        return None
    last_d, last_v = pts[-1]
    lo = last_d - timedelta(days=round(365.25 * REER_WINDOW_YEARS))
    win = [v for d, v in pts if d >= lo]
    years = (last_d - pts[0][0]).days / 365.25
    out = {'level': last_v, 'as_of': last_d.isoformat(),
           'years': round(min(years, REER_WINDOW_YEARS), 1),
           'pct': None, 'z': None, 'read': None}
    if years < REER_MIN_YEARS or len(win) < 24:
        return out
    pct = 100.0 * sum(v <= last_v for v in win) / len(win)
    sd = statistics.pstdev(win)
    out['pct'] = round(pct, 1)
    out['z'] = round((last_v - statistics.fmean(win)) / sd, 2) if sd > 0 else None
    out['read'] = ('cheap' if pct <= CHEAP_PCT else
                   'rich' if pct >= RICH_PCT else 'fair')
    return out


def _carry_pair(rate_points, usd_rate_points, run_date):
    a = _as_of(rate_points, run_date)
    b = _as_of(usd_rate_points, run_date)
    if a is None or b is None:
        return None
    if ((run_date - a[0]).days > RATE_MAX_AGE_DAYS
            or (run_date - b[0]).days > RATE_MAX_AGE_DAYS):
        return None
    return {'rate': a[1], 'usd_rate': b[1], 'spread': a[1] - b[1],
            'as_of': min(a[0], b[0]).isoformat()}


def carry(ind, usd_ind, run_date):
    """The currency's short rate less the US one, in percentage points: the
    3-month pair when both are current, else the overnight pair (the OECD's
    euro-area and UK 3-month series stop at 2026-01). Like for like, never
    a 3-month rate against an overnight one. None when neither pair is
    current."""
    for key, tenor in (('rate3m', '3m'), ('rate_on', 'overnight')):
        got = _carry_pair(_points((ind or {}).get(key)),
                          _points((usd_ind or {}).get(key)), run_date)
        if got:
            got['tenor'] = tenor
            return got
    return None


def exposure(rows, ccy):
    """Rows that report in *ccy*: count, market cap and median MoS."""
    sel = [r for r in rows or () if (r.get('currency_financial') or 'USD') == ccy]
    mcaps = [m for m in (_num(r.get('mcap')) for r in sel) if m and m > 0]
    moss = [m for m in (_num(r.get('mos')) for r in sel) if m is not None]
    return {'n': len(sel), 'mcap': sum(mcaps) if mcaps else None,
            'median_mos': statistics.median(moss) if moss else None,
            'buys': sum(r.get('rating') in ('BUY', 'LEAN BUY') for r in sel)}


def spark(points, run_date, days=SPARK_DAYS, step=SPARK_STEP_DAYS):
    """Weekly values over the last *days*, oldest first."""
    out, d = [], run_date - timedelta(days=days)
    while d <= run_date:
        p = _as_of(points, d)
        if p is not None and (d - p[0]).days <= 7:
            out.append(round(p[1], 8))
        d += timedelta(days=step)
    return out


def currency_board(spot, indicators, rows, run_date, currencies=None):
    """The Currencies tab payload.

    *spot*: ``{ccy: series}`` of USD per unit (dict or pandas Series);
    *indicators*: ``{ccy: {'reer': {...}, 'rate3m': {...}}}`` (USD included);
    *rows*: the rendered snapshot rows; *currencies*: which to show (default:
    every currency with a spot history). Returns ``{method, run_date, usd,
    currencies: [...]}``, currencies ordered by how many rows report in
    them, then by name."""
    run_date = run_date if isinstance(run_date, date) else date.fromisoformat(str(run_date))
    indicators = indicators or {}
    usd_ind = indicators.get('USD') or {}
    usd_rates = _points(usd_ind.get('rate3m'))
    total_mcap = sum(m for m in (_num(r.get('mcap')) for r in rows or ())
                     if m and m > 0) or None
    out = []
    for ccy in sorted(currencies or spot):
        if ccy == 'USD':
            continue
        pts = _points(spot.get(ccy))
        mv = moves(pts, run_date)
        if mv is None:
            continue
        ind = indicators.get(ccy) or {}
        name, where = CURRENCY_INFO.get(ccy, (ccy, ''))
        exp = exposure(rows, ccy)
        exp['mcap_share'] = (exp['mcap'] / total_mcap
                             if exp['mcap'] and total_mcap else None)
        out.append({
            'ccy': ccy, 'name': name, 'where': where,
            'quote': 'usd_per' if ccy in USD_PER_UNIT_QUOTE else 'per_usd',
            **mv,
            'real': real_value(_points(ind.get('reer')), run_date),
            'carry': carry(ind, usd_ind, run_date),
            'exposure': exp,
            'spark': spark(pts, run_date),
        })
    out.sort(key=lambda c: (-c['exposure']['n'], c['ccy']))
    return {
        'method': FX_METHOD, 'run_date': run_date.isoformat(),
        'usd': {'real': real_value(_points(usd_ind.get('reer')), run_date),
                'rate3m': (lambda p: {'rate': p[1], 'as_of': p[0].isoformat()}
                           if p else None)(_as_of(usd_rates, run_date)),
                'exposure': exposure(rows, 'USD')},
        'currencies': out,
    }
