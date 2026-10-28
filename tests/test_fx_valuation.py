"""models/fx_valuation.py (Financial Model › Currencies), the indicators it
reads from data/fx_store.py, and its render payload."""
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from data.fx_store import FXStore
from models import fx_valuation as fv

RUN = date(2026, 10, 26)


def _daily(start, end, fn):
    idx = pd.bdate_range(start, end)
    return pd.Series([fn(d) for d in idx], index=idx, dtype=float)


def _monthly(years, fn, end=date(2026, 7, 1)):
    out, d = {}, end
    for i in range(int(years * 12)):
        out[d.isoformat()] = fn(i)
        d = (pd.Timestamp(d) - pd.DateOffset(months=1)).date()
    return out


# -- moves ---------------------------------------------------------------------

def test_moves_are_the_change_in_dollar_value():
    # 1.00 a year ago, 1.10 now: the currency strengthened 10%.
    s = _daily('2021-01-01', '2026-10-26',
               lambda d: 1.10 if d >= pd.Timestamp('2026-01-01') else 1.00)
    m = fv.moves(fv._points(s), RUN)
    assert m['spot'] == pytest.approx(1.10)
    assert m['1y'] == pytest.approx(0.10)
    assert m['1m'] == pytest.approx(0.0)


def test_moves_need_a_current_bar_and_a_bar_near_each_horizon():
    s = _daily('2026-01-05', '2026-10-01', lambda d: 1.0)
    assert fv.moves(fv._points(s), RUN) is None          # last bar 25 days old
    s = _daily('2026-06-01', '2026-10-26', lambda d: 1.0)
    m = fv.moves(fv._points(s), RUN)
    assert m['3m'] == 0.0 and m['1y'] is None and m['5y'] is None


# -- real value ----------------------------------------------------------------

def test_real_value_ranks_within_its_own_window():
    # 20 years rising 0..239: the latest is the top of its range.
    rv = fv.real_value(fv._points(_monthly(20, lambda i: 240 - i)), RUN)
    assert rv['read'] == 'rich' and rv['pct'] == 100.0 and rv['z'] > 1.5
    rv = fv.real_value(fv._points(_monthly(20, lambda i: i)), RUN)
    assert rv['read'] == 'cheap' and rv['pct'] < 1


def test_real_value_window_ignores_older_history():
    # Very high 25-30 years ago, flat-ish since: those years do not count.
    pts = _monthly(30, lambda i: 1000 if i >= 300 else 100 + (i % 5))
    rv = fv.real_value(fv._points(pts), RUN)
    assert rv['years'] == 20
    assert rv['read'] != 'cheap'


def test_real_value_needs_ten_years():
    rv = fv.real_value(fv._points(_monthly(8, lambda i: i)), RUN)
    assert rv['level'] == 0 and rv['pct'] is None and rv['read'] is None
    assert fv.real_value([], RUN) is None


# -- carry ---------------------------------------------------------------------

def test_carry_prefers_the_3m_pair():
    ind = {'rate3m': {'2026-09-01': 5.0}, 'rate_on': {'2026-10-20': 4.0}}
    usd = {'rate3m': {'2026-09-01': 3.9}, 'rate_on': {'2026-10-20': 3.8}}
    k = fv.carry(ind, usd, RUN)
    assert k['tenor'] == '3m' and k['spread'] == pytest.approx(1.1)


def test_carry_falls_back_to_overnight_when_3m_is_stale():
    # The OECD's euro-area 3-month series stops at 2026-01.
    ind = {'rate3m': {'2026-01-01': 2.0}, 'rate_on': {'2026-10-24': 2.44}}
    usd = {'rate3m': {'2026-09-01': 3.94}, 'rate_on': {'2026-10-24': 3.93}}
    k = fv.carry(ind, usd, RUN)
    assert k['tenor'] == 'overnight'
    assert k['spread'] == pytest.approx(-1.49)


def test_carry_never_mixes_tenors():
    ind = {'rate3m': {'2026-09-01': 5.0}}
    usd = {'rate_on': {'2026-10-24': 3.9}}
    assert fv.carry(ind, usd, RUN) is None
    assert fv.carry({}, usd, RUN) is None


# -- exposure and the board ----------------------------------------------------

ROWS = [
    {'currency_financial': 'JPY', 'mcap': 3e9, 'mos': 0.2, 'rating': 'BUY'},
    {'currency_financial': 'JPY', 'mcap': 1e9, 'mos': -0.1, 'rating': 'HOLD'},
    {'currency_financial': 'USD', 'mcap': 6e9, 'mos': 0.0, 'rating': 'HOLD'},
    {'mcap': 0, 'mos': None},      # no currency: USD
]


def test_exposure():
    e = fv.exposure(ROWS, 'JPY')
    assert e == {'n': 2, 'mcap': 4e9, 'median_mos': pytest.approx(0.05), 'buys': 1}
    assert fv.exposure(ROWS, 'USD')['n'] == 2
    assert fv.exposure(ROWS, 'EUR') == {'n': 0, 'mcap': None, 'median_mos': None, 'buys': 0}


def _board():
    spot = {'JPY': _daily('2021-01-01', '2026-10-26', lambda d: 0.0065),
            'EUR': _daily('2021-01-01', '2026-10-26', lambda d: 1.12),
            'USD': _daily('2021-01-01', '2026-10-26', lambda d: 1.0)}
    ind = {'USD': {'reer': _monthly(20, lambda i: 240 - i),
                   'rate3m': {'2026-09-01': 3.94}},
           'JPY': {'reer': _monthly(20, lambda i: i), 'rate3m': {'2026-08-01': 1.5}}}
    return fv.currency_board(spot, ind, ROWS, RUN)


def test_board_orders_by_exposure_and_quotes_by_convention():
    b = _board()
    assert [c['ccy'] for c in b['currencies']] == ['JPY', 'EUR']
    jpy, eur = b['currencies']
    assert jpy['quote'] == 'per_usd' and eur['quote'] == 'usd_per'
    assert jpy['real']['read'] == 'cheap' and eur['real'] is None
    assert jpy['carry']['spread'] == pytest.approx(-2.44)
    assert jpy['exposure']['mcap_share'] == pytest.approx(0.4)
    assert len(jpy['spark']) >= 50
    assert b['usd']['real']['read'] == 'rich'
    assert b['usd']['exposure']['n'] == 2
    json.dumps(b)    # the payload is plain JSON


def test_every_universe_currency_has_a_name():
    for ccy in ('EUR', 'JPY', 'CAD', 'CNY', 'BRL', 'GBP', 'AUD', 'ARS', 'CHF',
                'KRW', 'MXN', 'HKD', 'SEK', 'ZAR', 'INR', 'IDR', 'THB', 'TWD',
                'PLN', 'TRY', 'NOK', 'DKK', 'CLP', 'PEN', 'COP', 'SGD', 'PHP',
                'NZD', 'ILS', 'KZT', 'VND'):
        assert ccy in fv.CURRENCY_INFO


# -- indicators in the store ---------------------------------------------------

class _Fred:
    def __init__(self, data):
        self.data, self.asked = data, []

    def fetch_series(self, sid, start=None):
        self.asked.append(sid)
        return self.data.get(sid, {})


def test_indicators_round_trip_and_series_ids(tmp_path):
    s = FXStore(str(tmp_path))
    fred = _Fred({'RBXMBIS': {date(2026, 7, 1): 101.5},
                  'ECBESTRVOLWGTTRMDMNRT': {date(2026, 10, 24): 2.44},
                  'SOFR': {date(2026, 10, 24): 3.93},
                  'IR3TIB01USM156N': {date(2026, 9, 1): 3.94}})
    got = s.refresh_indicators(['EUR'], fred_client=fred)
    assert 'IR3TIB01EZM156N' in fred.asked          # OECD's euro-area code
    assert got == {'EUR': ['reer', 'rate_on'], 'USD': ['rate3m', 'rate_on']}
    ind = FXStore(str(tmp_path)).indicators()
    assert ind['EUR']['reer'] == {'2026-07-01': 101.5}
    assert ind['USD']['rate_on'] == {'2026-10-24': 3.93}


def test_indicators_keep_a_series_that_failed_this_time(tmp_path):
    s = FXStore(str(tmp_path))
    s.refresh_indicators(['JPY'], fred_client=_Fred({'RBJPBIS': {date(2026, 7, 1): 70.0}}))
    s.refresh_indicators(['JPY'], fred_client=_Fred({}))
    assert s.indicators()['JPY']['reer'] == {'2026-07-01': 70.0}


# -- render payload and template -------------------------------------------------

def test_render_payload_is_written_and_skipped_without_history(tmp_path):
    from scripts.report_html import _build_fx_payload
    empty = tmp_path / 'fx-empty'
    assert _build_fx_payload(ROWS, str(tmp_path), RUN, fx_dir=str(empty)) is None
    s = FXStore(str(tmp_path / 'fx'))
    s.refresh(['JPY'], yahoo_fetch=lambda c: _daily('2025-01-01', '2026-10-26', lambda d: 0.0065),
              fred_fetch=lambda c: pd.Series(dtype=float))
    p = _build_fx_payload(ROWS, str(tmp_path), RUN, fx_dir=str(tmp_path / 'fx'))
    assert [c['ccy'] for c in p['currencies']] == ['JPY']
    on_disk = json.loads((tmp_path / f'fx_{RUN.isoformat()}.json').read_text(encoding='utf-8'))
    assert on_disk['currencies'][0]['exposure']['n'] == 2


def test_row_context_carries_the_reporting_currency():
    from scripts.report_html import _row_context
    base = {'ticker': 'TM', 'currency_financial': 'JPY'}
    assert _row_context(base, {'gates': []}, set(), {}, {})['ccy'] == 'JPY'
    base['currency_financial'] = 'USD'
    assert _row_context(base, {'gates': []}, set(), {}, {})['ccy'] is None


def _tpl():
    return (Path(__file__).resolve().parents[1] / 'templates' / 'report.html').read_text(encoding='utf-8')


def test_template_drill_filters_on_the_reporting_currency():
    src = _tpl()
    assert "if(f.type==='ccy')return d.ccy===f.ccy;" in src
    assert "if(f.type==='ccy')return 'Reports in '+f.ccy;" in src
    jump = re.search(r'function fxJump\(ccy\)\{.*?\n\}\n', src, re.S).group(0)
    assert "_drillPush('Financial Model · Currencies')" in jump
    assert '_dashJump();' in jump
    assert 'var FX_PUB={{ fx_json }};' in src


def test_template_colours_beat_the_dark_text_rule():
    css = re.search(r'/\* Financial Model › Currencies \(renderFxBoard\).*?\n@media', _tpl(), re.S).group(0)
    for cls in ('.fx-up', '.fx-dn', '.fx-na'):
        rule = re.search(re.escape(cls) + r'[^{]*\{([^}]*)\}', css).group(1)
        assert '!important' in rule, cls


def test_spark_is_weekly_over_a_year():
    pts = fv._points(_daily('2025-01-01', '2026-10-26', lambda d: 1.0))
    sp = fv.spark(pts, RUN)
    assert 50 <= len(sp) <= 54
    assert fv.spark([], RUN) == []
    assert fv.spark(pts, RUN - timedelta(days=900)) == []
