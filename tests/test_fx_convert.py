"""data/fx_convert.py: ASC 830 translation over the FX store, and its use by
the SEC history, the SEC statement shape and the live yfinance path."""
from datetime import date

import pandas as pd
import pytest

from data import fx_convert as fc
from data import sec_xbrl_client as xbrl_mod
from data.fx_store import FXStore
from data.sec_xbrl_client import SECXBRLClient


def _daily(rate_fn, start='2023-01-02', end='2025-12-31'):
    idx = pd.bdate_range(start, end)
    return pd.Series([rate_fn(d) for d in idx], index=idx, dtype=float)


def _jpy(d):
    """Yen that steps from 0.0070 to 0.0060 on 2024-10-01."""
    return 0.0070 if d < pd.Timestamp('2024-10-01') else 0.0060


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = FXStore(str(tmp_path))
    s.refresh(['JPY'], yahoo_fetch=lambda c: _daily(_jpy),
              fred_fetch=lambda c: pd.Series(dtype=float))
    monkeypatch.setattr(fc, '_STORE', s)
    return s


# -- rates --------------------------------------------------------------------

def test_flow_rate_averages_the_fiscal_year_not_the_calendar_year(store):
    # FY2025 ends 2025-03-31: Apr 2024..Mar 2025 is half at each level.
    rates = fc.annual_rates('JPY', [2025], {2025: '2025-03-31'}, store=store)
    flow, stock = rates[2025]
    assert flow == pytest.approx(0.0065, abs=0.00006)
    assert stock == pytest.approx(0.0060)


def test_unknown_fiscal_year_end_is_inferred_from_a_known_one(store):
    rates = fc.annual_rates('JPY', [2024, 2025], {2025: '2025-03-31'}, store=store)
    assert rates[2024] == (pytest.approx(0.0070), pytest.approx(0.0070))


def test_no_known_end_falls_back_to_31_december(store):
    assert fc._resolve_end(2024, {}) == date(2024, 12, 31)
    assert fc._resolve_end(2023, {2024: '2024-02-29'}) == date(2023, 2, 28)


def test_years_before_the_history_have_no_rate(store):
    assert fc.annual_rates('JPY', [2010], store=store)[2010] == (None, None)


def test_usd_is_identity_and_never_touches_the_store(monkeypatch):
    monkeypatch.setattr(fc, 'default_store', lambda: pytest.fail('no store for USD'))
    assert fc.annual_rates('USD', [2024]) == {2024: (1.0, 1.0)}
    assert fc.get_spot_fx_rate('USD') is None


def test_spot_rate_is_as_of_the_run_date(store):
    assert fc.get_spot_fx_rate('JPY', date(2024, 6, 3), store=store) == pytest.approx(0.0070)
    assert fc.get_spot_fx_rate('JPY', date(2025, 6, 3), store=store) == pytest.approx(0.0060)


def test_ensure_fetches_a_missing_currency_once(tmp_path):
    s = FXStore(str(tmp_path))
    calls = []

    def fetch(ccy):
        calls.append(ccy)
        return _daily(lambda d: 0.25)
    s.refresh = lambda ccys: FXStore.refresh(
        s, ccys, yahoo_fetch=fetch, fred_fetch=lambda c: pd.Series(dtype=float))
    assert fc.get_spot_fx_rate('PLN', date(2025, 6, 3), store=s) == 0.25
    assert fc.get_spot_fx_rate('ILS', date(2025, 6, 3), store=s) == 0.25
    # Already tried this process: a date past the history is not refetched.
    assert fc.get_spot_fx_rate('PLN', date(2030, 1, 1), store=s) is None
    assert calls == ['PLN', 'ILS']


# -- statement frames ---------------------------------------------------------

def test_statement_columns_convert_at_their_own_period(store):
    inc = pd.DataFrame({pd.Timestamp('2025-03-31'): [1000.0, 50.0, 10.0],
                        pd.Timestamp('2024-03-31'): [900.0, 45.0, 10.0]},
                       index=['Total Revenue', 'Diluted EPS', 'Diluted Average Shares'])
    out, blanked = fc.convert_statement_df(inc, 'JPY', fc.FLOW, store=store)
    assert blanked == []
    assert out.iloc[0, 0] == pytest.approx(1000 * 0.0065, rel=0.01)
    assert out.iloc[0, 1] == pytest.approx(900 * 0.0070)
    assert out.loc['Diluted EPS'].iloc[0] == pytest.approx(50 * 0.0065, rel=0.01)
    assert list(out.loc['Diluted Average Shares']) == [10.0, 10.0]
    assert inc.iloc[0, 0] == 1000.0                      # input untouched


def test_balance_sheet_columns_take_the_closing_rate(store):
    bs = pd.DataFrame({pd.Timestamp('2025-03-31'): [100.0, 7.0]},
                      index=['Total Assets', 'Ordinary Shares Number'])
    out, _ = fc.convert_statement_df(bs, 'JPY', fc.STOCK, store=store)
    assert out.iloc[0, 0] == pytest.approx(0.60)
    assert out.iloc[1, 0] == 7.0


def test_a_column_without_a_rate_is_blanked(store):
    inc = pd.DataFrame({pd.Timestamp('2025-03-31'): [1000.0],
                        pd.Timestamp('2010-03-31'): [800.0]}, index=['Net Income'])
    out, blanked = fc.convert_statement_df(inc, 'JPY', fc.FLOW, store=store)
    assert blanked == [pd.Timestamp('2010-03-31')]
    assert out.iloc[0, 0] > 0 and pd.isna(out.iloc[0, 1])


@pytest.mark.parametrize('label,is_count', [
    ('Diluted Average Shares', True), ('Ordinary Shares Number', True),
    ('Share Issued', True), ('Basic EPS', False), ('Diluted EPS', False),
    ('Diluted NI Availto Com Stockholders', False),
    ('Issuance Of Capital Stock', False)])
def test_share_count_labels(label, is_count):
    assert fc._is_share_count_label(label) is is_count


# -- SEC history and shape ----------------------------------------------------

def _flow(fy, val, end_md='03-31', form='20-F'):
    return {'form': form, 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy}-06-20', 'start': f'{fy - 1}-04-01',
            'end': f'{fy}-{end_md}'}


def _pit(fy, val, end_md='03-31', form='20-F'):
    return {'form': form, 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy}-06-20', 'end': f'{fy}-{end_md}'}


def _march_filer():
    tags = {'Revenue': [_flow(2024, 1e6), _flow(2025, 1e6)],
            'ProfitLoss': [_flow(2024, 1e5), _flow(2025, 1e5)],
            'Assets': [_pit(2024, 4e6), _pit(2025, 4e6)],
            'Equity': [_pit(2024, 2e6), _pit(2025, 2e6)]}
    facts = {'facts': {'ifrs-full': {
        t: {'units': {'JPY': e}} for t, e in tags.items()}}}
    c = SECXBRLClient.__new__(SECXBRLClient)
    c._cache = {}
    c.fetch_company_facts = lambda tk: facts
    return c, facts


def test_fiscal_year_ends_come_from_the_filing():
    c, facts = _march_filer()
    assert c.fiscal_year_ends(facts, 'JPY') == {2024: '2024-03-31', 2025: '2025-03-31'}


def test_history_flows_average_and_instants_close(tmp_path, monkeypatch):
    s = FXStore(str(tmp_path))    # rates from 2024-06: FY2024 has none
    s.refresh(['JPY'], yahoo_fetch=lambda c: _daily(_jpy, start='2024-06-03'),
              fred_fetch=lambda c: pd.Series(dtype=float))
    monkeypatch.setattr(fc, '_STORE', s)
    c, _ = _march_filer()
    h = c.fetch_historical_financials('TEST')
    assert h['reporting_currency'] == 'JPY'
    assert h['fx_years_missing'] == [2024]
    assert 2024 not in h['revenue_history']  # dropped, not passed through
    # Apr 2024..Mar 2025 with bars from June: 4 months at 0.0070, 6 at 0.0060.
    assert h['revenue_history'][2025] == pytest.approx(1e6 * 0.0064, rel=0.01)
    assert h['total_assets_history'][2025] == pytest.approx(4e6 * 0.0060)


def test_shape_keeps_older_years_without_rates_out(monkeypatch):
    monkeypatch.setattr(xbrl_mod, '_annual_fx_rates',
                        lambda ccy, years, ends=None: {
                            y: ((0.0065, 0.006) if y == 2025 else (None, None))
                            for y in years})
    c, _ = _march_filer()
    shape = c.build_yfinance_shape('TEST')
    assert shape is not None
    inc = shape['income_statement']
    assert inc.columns[0].year == 2025
    assert inc.loc['Total Revenue'].iloc[0] == pytest.approx(6500)
    assert pd.Timestamp('2024-03-31') not in inc.columns or pd.isna(
        inc.loc['Total Revenue', pd.Timestamp('2024-03-31')])


# -- live yfinance path -------------------------------------------------------

def test_live_path_converts_statements_by_column(store):
    from scripts import analyze_stock as az
    inc = pd.DataFrame({pd.Timestamp('2025-03-31'): [1000.0]}, index=['Total Revenue'])
    bs = pd.DataFrame({pd.Timestamp('2025-03-31'): [500.0]}, index=['Total Assets'])
    payload = {'info': {'financialCurrency': 'JPY', 'currency': 'USD',
                        'totalCash': 1000.0},
               'income_statement': inc, 'balance_sheet': bs}
    out, meta = az._convert_financials_to_usd(payload, as_of=date(2025, 6, 2))
    assert out['income_statement'].iloc[0, 0] == pytest.approx(6.5, rel=0.01)
    assert out['balance_sheet'].iloc[0, 0] == pytest.approx(3.0)
    assert out['info']['totalCash'] == pytest.approx(6.0)
    assert meta['fx_rate_financial'] == pytest.approx(0.0060)
    assert meta['fx_converted'] is True and meta['fx_fetch_failed'] is False
