"""ECB SDMX daily reference rates -> USD-based daily history.

Endpoint (verified, no key, no auth):
    https://data-api.ecb.europa.eu/service/data/EXR/D.{CCY}.EUR.SP00.A
        ?startPeriod=YYYY-MM-DD&format=csvdata

The whole universe comes back in a single request by wildcarding the currency
dimension (`D..EUR.SP00.A`), which is one HTTP call for 30 currencies x 27
years instead of 30 calls.

Quote convention
----------------
ECB publishes *units of CCY per 1 EUR*. So the raw series for JPY is ~170
(yen per euro) and for USD is ~1.09 (dollars per euro). To get the USD cross:

    USD per 1 CCY  =  (USD per EUR) / (CCY per EUR)

which for JPY gives 1.09/170 = 0.0064 USD per yen. Every downstream metric is
expressed as **USD per unit of currency**, so "up" always means the currency
appreciated. Getting this backwards silently inverts every momentum and
drawdown signal, so `to_usd_series` is the only place the division happens and
tests/test_ecb_client.py pins it against known values.

EUR itself has no EXR series (it is the quote currency); its USD cross is the
USD series read directly.
"""

import csv
import io
from datetime import date, timedelta

from data import currency_meta
from data.cache import fetch

_BASE = 'https://data-api.ecb.europa.eu/service/data/EXR'

# A currency whose newest observation lags the newest observation in the whole
# dataset by more than this is treated as discontinued, not merely untraded.
# ECB keeps serving terminated series (CYP, SIT, HRK...) forever via
# lastNObservations, so recency is the only way to tell live from dead.
MAX_STALENESS_DAYS = 14


def _parse_csv(text):
    """SDMX csvdata -> {currency: {iso_date: float}}."""
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        ccy = row.get('CURRENCY')
        per = row.get('TIME_PERIOD')
        raw = row.get('OBS_VALUE')
        if not ccy or not per or raw in (None, '', 'NaN'):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if val <= 0:
            # A zero or negative reference rate is a data error, not a price;
            # letting it through would produce an infinite cross rate.
            continue
        out.setdefault(ccy, {})[per] = val
    return out


def fetch_reference_rates(start='1999-01-01', ttl_days=0.4, meta=None):
    """All EUR-based daily reference rates from `start`.

    Returns {currency: {iso_date: units_per_EUR}} or None on total failure.
    """
    url = ('%s/D..EUR.SP00.A?startPeriod=%s&format=csvdata' % (_BASE, start))
    text = fetch(url, 'ecb', ttl_days=ttl_days, meta=meta)
    if text is None:
        return None
    parsed = _parse_csv(text)
    return parsed or None


def live_currencies(per_eur):
    """Currencies whose series is still being published.

    Compares each series' last observation against the dataset-wide maximum
    date rather than against today, so a run on a weekend or a TARGET holiday
    doesn't declare the entire universe dead.
    """
    if not per_eur:
        return set()
    last = {c: max(d) for c, d in per_eur.items() if d}
    if not last:
        return set()
    newest = max(last.values())
    try:
        cutoff = (date.fromisoformat(newest) - timedelta(days=MAX_STALENESS_DAYS)).isoformat()
    except ValueError:
        return set(last)
    return {c for c, d in last.items() if d >= cutoff}


def to_usd_series(per_eur, code):
    """USD per one unit of `code`, as {iso_date: rate}.

    Dates are intersected between the currency leg and the USD leg, so a
    national holiday in one country can never be silently compared against a
    different trading day in another. This is the FX analogue of the sibling
    repo's index.intersection() alignment in models/macro.py.
    """
    usd_leg = per_eur.get('USD') or {}
    if not usd_leg:
        return {}

    if code == 'EUR':
        # USD per EUR is exactly the USD series.
        return dict(usd_leg)
    if code == 'USD':
        return {d: 1.0 for d in usd_leg}

    ccy_leg = per_eur.get(code) or {}
    if not ccy_leg:
        return {}

    out = {}
    for d, ccy_per_eur in ccy_leg.items():
        usd_per_eur = usd_leg.get(d)
        if usd_per_eur is None or ccy_per_eur <= 0:
            continue
        out[d] = usd_per_eur / ccy_per_eur
    return out


def build_usd_histories(per_eur, codes=None, prov=None):
    """{code: [(iso_date, usd_per_unit), ...]} sorted ascending.

    Currencies whose series is dead or absent are omitted entirely and, when a
    recorder is supplied, logged as `series_missing` — an empty history must
    not be mistaken for a flat exchange rate.
    """
    codes = codes or currency_meta.CODES
    live = live_currencies(per_eur)
    out = {}
    for code in codes:
        if code not in ('EUR', 'USD') and code not in live:
            if prov:
                prov.record_event('series_missing', code, 'ecb',
                                  {'reason': 'no live EXR series'})
            continue
        series = to_usd_series(per_eur, code)
        if not series:
            if prov:
                prov.record_event('series_missing', code, 'ecb',
                                  {'reason': 'empty after USD cross'})
            continue
        out[code] = sorted(series.items())
    return out
