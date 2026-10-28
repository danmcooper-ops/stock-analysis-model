"""BIS SDMX: real effective exchange rates and central bank policy rates.

Endpoints (verified, no key, no auth):
    REER    https://stats.bis.org/api/v2/data/dataflow/BIS/WS_EER/1.0/M.R.B.{AREA}
    Policy  https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.{AREA}

**`format=csv` is mandatory.** Without it BIS answers every request with HTTP
406 and an SDMX error document. (An `Accept: application/vnd.sdmx.data+json`
header works too, but the CSV is a third the size and needs no schema walk.)

Leaving the last key dimension empty wildcards it, so `M.R.B.` returns all 65
economies in one request and `M.` returns all 48 policy-rate series.

REER semantics
--------------
WS_EER key is FREQ.EER_TYPE.EER_BASKET.REF_AREA; `M.R.B` selects Monthly,
Real, Broad basket (64 economies, CPI-deflated, trade-weighted). Index,
2020 = 100.

**Higher REER = stronger currency in real terms = more expensive.** So for the
valuation pillar the sign inverts: a REER far *below* its own long-run average
means the currency is cheap, which is the buy signal. Japan is the sanity
check — 164 in 1994 against ~65 now.

Policy rates are already in percent (4.25 means 4.25%), not decimals.
"""

import csv
import io

from data import currency_meta
from data.cache import fetch

_BASE = 'https://stats.bis.org/api/v2/data/dataflow/BIS'

# BIS revises and republishes; a month-old policy rate is normal, a year-old
# one means the series was discontinued for that economy.
MAX_POLICY_STALENESS_MONTHS = 6


def _parse_csv(text, area_col='REF_AREA'):
    """SDMX csv -> {area: {'YYYY-MM': float}}."""
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        area = row.get(area_col)
        per = row.get('TIME_PERIOD')
        raw = row.get('OBS_VALUE')
        if not area or not per or raw in (None, '', 'NaN'):
            continue
        try:
            out.setdefault(area, {})[per] = float(raw)
        except ValueError:
            continue
    return out


def fetch_reer(ttl_days=7.0, meta=None):
    """Monthly real broad effective exchange rates, all economies, full history.

    TTL is a week: BIS publishes REER monthly, so a daily re-fetch would be
    30 wasted requests for every one that changes anything.

    Returns {bis_area: {'YYYY-MM': index}} or None.
    """
    url = '%s/WS_EER/1.0/M.R.B.?format=csv' % _BASE
    text = fetch(url, 'bis', ttl_days=ttl_days, meta=meta)
    if text is None:
        return None
    return _parse_csv(text) or None


def fetch_policy_rates(ttl_days=1.0, meta=None):
    """Monthly central bank policy rates, all economies, in percent.

    Shorter TTL than REER: policy decisions are the one BIS series that can
    move between two consecutive daily builds.

    Returns {bis_area: {'YYYY-MM': percent}} or None.
    """
    url = '%s/WS_CBPOL/1.0/M.?format=csv' % _BASE
    text = fetch(url, 'bis', ttl_days=ttl_days, meta=meta)
    if text is None:
        return None
    return _parse_csv(text) or None


def _series_for(by_area, code):
    area = currency_meta.bis_area(code)
    if not area:
        return {}
    return by_area.get(area) or {}


def reer_series(by_area, code):
    """Sorted [(YYYY-MM, index)] for a currency, via its BIS area.

    EUR resolves through area 'XM', so the twenty eurozone members share one
    series rather than producing twenty duplicate rows.
    """
    return sorted(_series_for(by_area, code).items())


def policy_series(by_area, code):
    """Sorted [(YYYY-MM, percent)] for a currency."""
    return sorted(_series_for(by_area, code).items())


def latest(series):
    """Most recent (period, value) of a sorted series, or (None, None)."""
    return series[-1] if series else (None, None)


def latest_policy_rate(by_area, code, prov=None):
    """Current policy rate in percent, or None.

    Returns None rather than 0.0 when absent — a missing rate and a zero rate
    are different facts, and conflating them would hand ZIRP-era carry scores
    to economies BIS simply doesn't cover.
    """
    if not currency_meta.has_policy_rate(code):
        # Structural: MAS runs exchange-rate policy, so SGD has no CBPOL
        # series to be missing. Not an error, not an event.
        return None
    series = policy_series(by_area, code)
    if not series:
        if prov:
            prov.record_event('series_missing', code, 'bis',
                              {'slot': 'policy_rate'})
        return None
    period, value = series[-1]
    newest = max((s[-1][0] for s in
                  (policy_series(by_area, c) for c in currency_meta.CODES) if s),
                 default=period)
    if _months_between(period, newest) > MAX_POLICY_STALENESS_MONTHS:
        if prov:
            prov.record_event('series_missing', code, 'bis',
                              {'slot': 'policy_rate', 'last': period,
                               'reason': 'discontinued'})
        return None
    return value


def _months_between(a, b):
    """Whole months from period 'YYYY-MM' a to b; 0 if either is malformed."""
    try:
        ay, am = int(a[:4]), int(a[5:7])
        by, bm = int(b[:4]), int(b[5:7])
        return (by - ay) * 12 + (bm - am)
    except (ValueError, TypeError, IndexError):
        return 0
