"""World Bank indicator API — the external-balance and inflation inputs.

    https://api.worldbank.org/v2/country/{codes}/indicator/{id}?format=json

No key, no auth. Annual data, typically one to two years behind.

Reliability
-----------
This is the least reliable source in the project and the design reflects that.
Observed behaviour during development: `FP.CPI.TOTL.ZG` and `NY.GNP.PCAP.PP.CD`
returned HTTP 200 with full data while `BN.CAB.XOKA.GD.ZS`, `FI.RES.TOTL.MO`,
`NY.GDP.MKTP.KD.ZG` and `GC.DOD.TOTL.GD.ZS` returned HTTP 502 on every one of
four retries, in the same session, seconds apart. The API also serves an HTML
error page under HTTP 200 in some states.

Three consequences, all implemented below:

1. **Each indicator is fetched independently.** One 502 costs one column, not
   the pillar. A single batched request would make the whole External pillar
   an all-or-nothing bet on the worst-behaved indicator.
2. **Long TTL (30 days).** The data is annual; a month-old cache entry is the
   same number. This also means a run during an outage usually still has data.
3. **Stale-serve on failure**, handled in cache.fetch, with a provenance event
   so the report can show what was served from a stale cache.

Values are returned per-currency, already resolved through the euro-area
aggregate where applicable (EUR -> EMU).
"""

from data import currency_meta
from data.cache import fetch_json

_BASE = 'https://api.worldbank.org/v2'

# Indicator -> (short field name, human label).
# Signs are as the World Bank reports them: current account and government
# balance are positive for surplus, so higher is better for both.
INDICATORS = {
    'BN.CAB.XOKA.GD.ZS': ('current_account_pct_gdp', 'Current account (% GDP)'),
    'FI.RES.TOTL.MO':    ('reserves_months_imports', 'Reserves (months of imports)'),
    'FP.CPI.TOTL.ZG':    ('inflation_pct',           'Inflation, CPI (annual %)'),
    'NY.GDP.MKTP.KD.ZG': ('gdp_growth_pct',          'GDP growth (annual %)'),
    'GC.DOD.TOTL.GD.ZS': ('gov_debt_pct_gdp',        'Government debt (% GDP)'),
    'DT.DOD.DECT.GN.ZS': ('external_debt_pct_gni',   'External debt (% GNI)'),
}

# How far back to ask for. The API returns the most recent non-null within the
# window, so a wide window is what makes a country with a two-year reporting
# lag resolve at all. Ten years also gives the inflation-volatility metric
# something to chew on.
YEARS = 10

DEFAULT_TTL_DAYS = 30.0


def _fetch_indicator(indicator, wb_codes, ttl_days, meta=None):
    """One indicator for all countries -> {wb_code: [(year, value), ...]}.

    Returns None if the endpoint failed entirely, which the caller must
    distinguish from an empty result (endpoint healthy, no data reported).
    """
    from datetime import date
    end = date.today().year
    codes = ';'.join(sorted(set(wb_codes)))
    url = ('%s/country/%s/indicator/%s?format=json&date=%d:%d&per_page=%d'
           % (_BASE, codes, indicator, end - YEARS, end, len(wb_codes) * (YEARS + 2)))

    payload = fetch_json(url, 'worldbank', ttl_days=ttl_days, meta=meta)
    if payload is None:
        return None
    # Shape is [metadata, rows]; an error document is a bare dict or a
    # one-element list, both of which mean "no usable data".
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return None

    out = {}
    for row in payload[1]:
        try:
            country = (row.get('countryiso3code') or '').strip()
            year = int(row.get('date'))
            value = row.get('value')
        except (AttributeError, TypeError, ValueError):
            continue
        if not country or value is None:
            continue
        out.setdefault(country, []).append((year, float(value)))

    for series in out.values():
        series.sort()
    return out


def fetch_macro(codes=None, ttl_days=DEFAULT_TTL_DAYS, prov=None):
    """All indicators for the currency universe.

    Returns {currency_code: {field: value, field_year: int, ...}} using the
    most recent non-null observation per indicator, plus the full annual
    series under '<field>_series' for trend/volatility metrics.

    A failed indicator is simply absent from every currency's dict, so it
    scores as missing data (0, in the denominator) rather than corrupting
    neighbouring fields.
    """
    codes = codes or currency_meta.CODES
    wb_by_code = {c: currency_meta.wb_code(c) for c in codes}
    wb_codes = [w for w in wb_by_code.values() if w]

    out = {c: {} for c in codes}
    for indicator, (field, _label) in INDICATORS.items():
        meta = {}
        series_by_country = _fetch_indicator(indicator, wb_codes, ttl_days, meta=meta)

        if series_by_country is None:
            if prov:
                prov.record_event('indicator_failed', None, 'worldbank',
                                  {'indicator': indicator, 'field': field,
                                   'error': meta.get('error')})
            continue

        for code in codes:
            wb = wb_by_code.get(code)
            series = series_by_country.get(wb) if wb else None
            if prov:
                prov.record_source(code, 'macro', 'worldbank', meta=meta,
                                   indicator=indicator)
            if not series:
                continue
            year, value = series[-1]
            out[code][field] = value
            out[code][field + '_year'] = year
            out[code][field + '_series'] = series

    return out


def indicator_health(macro_by_code):
    """Per-field coverage, for the run log and the report footer.

    Surfacing this matters because a World Bank outage is invisible in the
    output otherwise — the column just quietly becomes all-N/A.
    """
    out = {}
    for _ind, (field, label) in INDICATORS.items():
        have = sum(1 for m in macro_by_code.values() if m.get(field) is not None)
        out[field] = {'label': label, 'covered': have,
                      'total': len(macro_by_code)}
    return out
