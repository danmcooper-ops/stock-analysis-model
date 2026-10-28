"""Currency universe: ISO-4217 code <-> BIS REF_AREA <-> country ISO3 <-> regime.

This module is the single source of truth for *which* currencies the screener
covers and how each one is addressed in every upstream API. Everything else
(clients, models, scoring, the map in the report) keys off `CURRENCIES`.

Three identifier systems have to be reconciled:

  ISO-4217   'JPY'   what the ECB quotes and what users think in
  BIS area   'JP'    BIS keys by *economy*, not currency
  ISO3       'JPN'   what the Natural Earth basemap keys country shapes by

The euro is the reason this file exists rather than a dict literal inline.
BIS publishes one euro-area REER series under the synthetic area code `XM`,
but twenty sovereign countries use the euro. Without an explicit collapse,
joining BIS to a country list yields EUR twenty times over. `bis_area` is
therefore many-to-one and `map_iso3` is a list, not a scalar.

Universe selection rationale
----------------------------
The universe is the set of currencies with a *live* ECB daily reference-rate
series, because that series is the only free source of multi-decade daily FX
history (back to 1999-01-04) and the Momentum and Stability pillars are
uncomputable without it. All 30 are also covered by BIS REER.

Currencies deliberately excluded despite BIS coverage — TWD, SAR, AED, KWD,
CLP, COP, PEN, ARS, RSD, BGN, MKD, BAM, MAD, DZD, RUB — have no free source
of daily spot history. Including them would produce rows scoring 0 on two of
five pillars for want of data rather than on merit, which is worse than an
honest 30-row table. See README for what adding them would require.
"""

# regime values:
#   'float'   — freely floating; every gate applies
#   'managed' — heavily managed float (authorities lean against moves, but the
#               rate still carries information); gates apply, rating capped
#   'peg'     — hard peg or currency board; valuation/momentum signals are
#               statements about the anchor, not the currency. Structurally
#               inapplicable, and rating-capped to NEUTRAL.
#
# has_policy_rate: False where the central bank does not run a policy-rate
#   regime at all. Singapore's MAS conducts monetary policy through the
#   nominal effective exchange rate band, not an interest rate, so BIS
#   publishes no CBPOL series for SG. That is a structural absence — the
#   carry gates are inapplicable for SGD, not missing.

CURRENCIES = [
    # code   name                    bis    iso3(s)                  regime      policy_rate
    ('EUR', 'Euro',                  'XM', ['AUT', 'BEL', 'CYP', 'DEU', 'ESP', 'EST', 'FIN',
                                            'FRA', 'GRC', 'HRV', 'IRL', 'ITA', 'LTU', 'LUX',
                                            'LVA', 'MLT', 'NLD', 'PRT', 'SVK', 'SVN'],
                                                                     'float',    True),
    ('USD', 'US Dollar',             'US', ['USA'],                  'float',    True),
    ('JPY', 'Japanese Yen',          'JP', ['JPN'],                  'float',    True),
    ('GBP', 'Pound Sterling',        'GB', ['GBR'],                  'float',    True),
    ('CHF', 'Swiss Franc',           'CH', ['CHE'],                  'float',    True),
    ('CAD', 'Canadian Dollar',       'CA', ['CAN'],                  'float',    True),
    ('AUD', 'Australian Dollar',     'AU', ['AUS'],                  'float',    True),
    ('NZD', 'New Zealand Dollar',    'NZ', ['NZL'],                  'float',    True),
    ('NOK', 'Norwegian Krone',       'NO', ['NOR'],                  'float',    True),
    ('SEK', 'Swedish Krona',         'SE', ['SWE'],                  'float',    True),
    ('DKK', 'Danish Krone',          'DK', ['DNK'],                  'peg',      True),
    ('ISK', 'Icelandic Krona',       'IS', ['ISL'],                  'float',    True),
    ('CZK', 'Czech Koruna',          'CZ', ['CZE'],                  'float',    True),
    ('HUF', 'Hungarian Forint',      'HU', ['HUN'],                  'float',    True),
    ('PLN', 'Polish Zloty',          'PL', ['POL'],                  'float',    True),
    ('RON', 'Romanian Leu',          'RO', ['ROU'],                  'managed',  True),
    ('TRY', 'Turkish Lira',          'TR', ['TUR'],                  'float',    True),
    ('ILS', 'Israeli Shekel',        'IL', ['ISR'],                  'float',    True),
    ('ZAR', 'South African Rand',    'ZA', ['ZAF'],                  'float',    True),
    ('CNY', 'Chinese Yuan',          'CN', ['CHN'],                  'managed',  True),
    ('HKD', 'Hong Kong Dollar',      'HK', ['HKG'],                  'peg',      True),
    ('SGD', 'Singapore Dollar',      'SG', ['SGP'],                  'managed',  False),
    ('KRW', 'South Korean Won',      'KR', ['KOR'],                  'float',    True),
    ('THB', 'Thai Baht',             'TH', ['THA'],                  'float',    True),
    ('MYR', 'Malaysian Ringgit',     'MY', ['MYS'],                  'managed',  True),
    ('IDR', 'Indonesian Rupiah',     'ID', ['IDN'],                  'float',    True),
    ('PHP', 'Philippine Peso',       'PH', ['PHL'],                  'float',    True),
    ('INR', 'Indian Rupee',          'IN', ['IND'],                  'managed',  True),
    ('MXN', 'Mexican Peso',          'MX', ['MEX'],                  'float',    True),
    ('BRL', 'Brazilian Real',        'BR', ['BRA'],                  'float',    True),
]

# The currency everything is quoted and scored against. Carry, momentum and
# drawdown are all "versus USD", so USD itself has no self-relative signal and
# is excluded from those gates (see scoring._appl_not_base).
BASE_CURRENCY = 'USD'

# ECB publishes EUR-based rates only, so EUR has no EXR series of its own.
# Its USD cross comes from inverting D.USD.EUR.SP00.A. See ecb_client.
ECB_QUOTE_CURRENCY = 'EUR'


# World Bank country code, where it differs from the single ISO3. The euro
# area has a World Bank *aggregate* ('EMU') that reports a consolidated
# current account and reserves position; using Germany's numbers as a stand-in
# for the euro would be simply wrong, since intra-euro trade nets out of the
# bloc's external balance but not out of any member's.
_WB_OVERRIDE = {'EUR': 'EMU'}


def _build():
    out = {}
    for code, name, bis, iso3s, regime, has_pr in CURRENCIES:
        out[code] = {
            'code': code,
            'name': name,
            'bis_area': bis,
            'map_iso3': list(iso3s),
            'regime': regime,
            'has_policy_rate': has_pr,
            'wb_code': _WB_OVERRIDE.get(code, iso3s[0]),
        }
    return out


META = _build()
CODES = [c[0] for c in CURRENCIES]

# Reverse index. Many-to-one: all 20 eurozone ISO3s resolve to EUR, which is
# what lets the report tint twenty countries from one row.
ISO3_TO_CODE = {iso3: m['code'] for m in META.values() for iso3 in m['map_iso3']}

# BIS area -> currency. Also many-to-one in principle; 'XM' is the only case.
BIS_AREA_TO_CODE = {m['bis_area']: m['code'] for m in META.values()}


def get(code):
    """Return the metadata dict for an ISO-4217 code, or None if unknown."""
    return META.get(code)


def bis_area(code):
    """BIS REF_AREA for a currency ('EUR' -> 'XM')."""
    m = META.get(code)
    return m['bis_area'] if m else None


def is_pegged(code):
    m = META.get(code)
    return bool(m) and m['regime'] == 'peg'


def is_managed(code):
    m = META.get(code)
    return bool(m) and m['regime'] in ('peg', 'managed')


def has_policy_rate(code):
    m = META.get(code)
    return bool(m) and m['has_policy_rate']


def wb_code(code):
    """World Bank country/aggregate code ('EUR' -> 'EMU')."""
    m = META.get(code)
    return m['wb_code'] if m else None


def all_bis_areas():
    """Deduplicated BIS areas to request. Twenty eurozone members collapse to
    one 'XM', so this is shorter than len(CODES) implies."""
    seen, out = set(), []
    for code in CODES:
        a = bis_area(code)
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out
