"""Client parsing and the USD cross-rate convention.

Offline: every test builds its own payload. The one thing worth pinning hard
is the direction of the ECB cross — the raw series is *units per EUR*, and
inverting it silently flips the sign of every momentum, drawdown and
volatility number in the model without raising anything.
"""

from data import bis_client, ecb_client

# ECB quotes units of CCY per 1 EUR. On 1999-01-04 the reference rates were
# 1.1789 USD/EUR and 133.73 JPY/EUR.
_ECB_CSV = '\n'.join([
    'KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,TIME_PERIOD,OBS_VALUE',
    'a,D,USD,EUR,SP00,1999-01-04,1.1789',
    'a,D,USD,EUR,SP00,1999-01-05,1.1790',
    'b,D,JPY,EUR,SP00,1999-01-04,133.73',
    'b,D,JPY,EUR,SP00,1999-01-05,130.96',
    'c,D,DEAD,EUR,SP00,1999-01-04,7.0',
])


def _parsed():
    return ecb_client._parse_csv(_ECB_CSV)


def test_ecb_parse_shape():
    p = _parsed()
    assert p['USD']['1999-01-04'] == 1.1789
    assert p['JPY']['1999-01-05'] == 130.96


def test_ecb_parse_skips_blank_and_non_positive():
    csv = '\n'.join([
        'KEY,CURRENCY,TIME_PERIOD,OBS_VALUE',
        'a,USD,1999-01-04,',
        'a,USD,1999-01-05,NaN',
        'a,USD,1999-01-06,0',
        'a,USD,1999-01-07,-1.5',
        'a,USD,1999-01-08,1.2',
    ])
    out = ecb_client._parse_csv(csv)
    # Only the one good observation survives; a zero rate would produce an
    # infinite cross rate downstream.
    assert out['USD'] == {'1999-01-08': 1.2}


def test_usd_cross_is_usd_per_unit_of_currency():
    p = _parsed()
    jpy = ecb_client.to_usd_series(p, 'JPY')
    # 1.1789 USD/EUR / 133.73 JPY/EUR = 0.008816 USD per yen.
    assert abs(jpy['1999-01-04'] - 1.1789 / 133.73) < 1e-12
    # Sanity on magnitude: a yen is worth well under a cent, not over a dollar.
    assert 0.001 < jpy['1999-01-04'] < 0.05


def test_eur_cross_is_the_usd_series_itself():
    p = _parsed()
    eur = ecb_client.to_usd_series(p, 'EUR')
    assert eur['1999-01-04'] == 1.1789


def test_usd_cross_against_itself_is_unity():
    p = _parsed()
    assert set(ecb_client.to_usd_series(p, 'USD').values()) == {1.0}


def test_cross_intersects_dates():
    """A date present for one leg but not the other must be dropped, not
    silently matched against a different trading day."""
    p = {'USD': {'d1': 1.1, 'd2': 1.2},
         'XXX': {'d1': 10.0, 'd3': 11.0}}
    out = ecb_client.to_usd_series(p, 'XXX')
    assert list(out) == ['d1']


def test_missing_currency_yields_empty_not_flat():
    p = _parsed()
    assert ecb_client.to_usd_series(p, 'NOPE') == {}


def test_live_currencies_excludes_discontinued_series():
    p = {
        'USD': {'2026-07-30': 1.1, '2026-07-31': 1.1},
        'JPY': {'2026-07-31': 170.0},
        'DEAD': {'2007-12-31': 7.0},
    }
    live = ecb_client.live_currencies(p)
    assert 'USD' in live and 'JPY' in live
    # ECB keeps serving terminated series forever; recency is the only signal.
    assert 'DEAD' not in live


def test_build_histories_sorted_and_omits_dead():
    p = {
        'USD': {'2026-07-31': 1.1, '2026-07-30': 1.2},
        'DEAD': {'2007-12-31': 7.0},
    }
    out = ecb_client.build_usd_histories(p, codes=['USD', 'DEAD'])
    assert [d for d, _ in out['USD']] == ['2026-07-30', '2026-07-31']
    assert 'DEAD' not in out


# ------------------------------------------------------------------ BIS

_BIS_CSV = '\n'.join([
    'FREQ,EER_TYPE,EER_BASKET,REF_AREA,TIME_PERIOD,OBS_VALUE',
    'M,R,B,JP,1994-01,164.28',
    'M,R,B,JP,2026-06,65.30',
    'M,R,B,XM,2026-06,102.70',
    'M,R,B,US,2026-06,107.84',
])


def test_bis_parse_and_series_lookup():
    by_area = bis_client._parse_csv(_BIS_CSV)
    jp = bis_client.reer_series(by_area, 'JPY')
    assert jp[0] == ('1994-01', 164.28)
    assert jp[-1] == ('2026-06', 65.30)


def test_bis_resolves_eur_through_the_synthetic_area():
    by_area = bis_client._parse_csv(_BIS_CSV)
    assert bis_client.reer_series(by_area, 'EUR') == [('2026-06', 102.70)]


def test_bis_unknown_currency_is_empty():
    by_area = bis_client._parse_csv(_BIS_CSV)
    assert bis_client.reer_series(by_area, 'ZZZ') == []


def test_singapore_policy_rate_is_structurally_absent():
    """SGD must return None without recording a missing-series event: MAS
    runs exchange-rate policy, so there is no series to be missing."""
    events = []

    class Prov:
        def record_event(self, *a, **k):
            events.append(a)

    assert bis_client.latest_policy_rate({}, 'SGD', Prov()) is None
    assert events == []


def test_months_between():
    assert bis_client._months_between('2026-01', '2026-06') == 5
    assert bis_client._months_between('2025-11', '2026-02') == 3
    assert bis_client._months_between('bad', '2026-02') == 0
