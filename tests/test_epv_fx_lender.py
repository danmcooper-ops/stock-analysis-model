# tests/test_epv_fx_lender.py
"""EPV's inputs on foreign filers and lenders, and the two-model blend.

- yfinance ``info`` totals (totalDebt, totalCash, ...) are in the STATEMENT
  currency. On the SEC path the statements are already USD, so nothing
  converted them, and _debt_levels' fallback handed EPV a yen net debt
  against a USD enterprise value (MFG EPV 9,862/share at $10.67).
- A lender's enterprise bridge measured its deposit-funded cash, not its
  earnings (C $526 at $128), so lenders take EPV at the equity level.
- Two DCF-less models more than 5x apart give no effective fair value.
"""
import pandas as pd
import pytest

import scripts.analyze_stock as az
from models.epv import earnings_power_value_valuation
from scripts.scoring import prepare_scoring_fields

JPY = 0.0067


def _payload(**info):
    bs = pd.DataFrame({pd.Timestamp('2025-12-31'): {'Total Debt': 2.0e9}})
    return {'info': dict({'financialCurrency': 'JPY', 'currency': 'USD',
                          'marketCap': 1.0e10}, **info),
            'balance_sheet': bs}


class TestInfoStatementFields:
    def test_sec_path_converts_statement_currency_totals(self, monkeypatch):
        monkeypatch.setattr(az, 'get_spot_fx_rate', lambda c: JPY if c == 'JPY' else None)
        out, meta = az._convert_financials_to_usd(
            _payload(totalCash=1.0e12, totalDebt=5.0e11), statements_are_usd=True)
        assert out['info']['totalCash'] == pytest.approx(1.0e12 * JPY)
        assert out['info']['totalDebt'] == pytest.approx(5.0e11 * JPY)
        assert out['info']['marketCap'] == 1.0e10           # quote is USD: untouched
        assert meta['fx_rate_info_statement'] == JPY
        assert meta['fx_rate_financial'] is None             # statements not re-converted
        debt, cash, _liab, net, src = az._debt_levels(out)
        assert debt == 2.0e9                                 # statement line wins
        assert cash == pytest.approx(6.7e9) and src == 'statements+yf_info'
        assert net == pytest.approx(2.0e9 - 6.7e9)

    def test_missing_rate_blanks_rather_than_mixes(self, monkeypatch):
        monkeypatch.setattr(az, 'get_spot_fx_rate', lambda c: None)
        out, meta = az._convert_financials_to_usd(
            _payload(totalCash=1.0e12), statements_are_usd=True)
        assert out['info']['totalCash'] is None
        assert meta['fx_fetch_failed'] is False              # nothing is mixed
        _debt, cash, _liab, net, _src = az._debt_levels(out)
        assert cash is None and net is None

    def test_usd_filer_untouched(self, monkeypatch):
        monkeypatch.setattr(az, 'get_spot_fx_rate', lambda c: pytest.fail('no lookup'))
        payload = {'info': {'financialCurrency': 'USD', 'currency': 'USD',
                            'totalCash': 5.0}}
        out, meta = az._convert_financials_to_usd(payload)
        assert out is payload and meta['fx_rate_info_statement'] is None

    def test_does_not_mutate_the_cached_payload(self, monkeypatch):
        monkeypatch.setattr(az, 'get_spot_fx_rate', lambda c: JPY)
        payload = _payload(totalCash=1.0e12)
        az._convert_financials_to_usd(payload, statements_are_usd=True)
        assert payload['info']['totalCash'] == 1.0e12


class TestLenderEpv:
    def test_equity_basis_capitalizes_earnings_at_cost_of_equity(self):
        v = earnings_power_value_valuation(100.0, 0.20, 0.10, 10.0, basis='equity')
        assert v.method == 'epv_equity'
        assert v.value == pytest.approx(100.0 * 0.8 / 0.10 / 10.0)
        assert v.inputs_used['basis'] == 'equity'

    def test_equity_basis_takes_no_bridge(self):
        with pytest.raises(ValueError):
            earnings_power_value_valuation(100.0, 0.2, 0.1, 10.0, total_debt=-5e9,
                                           basis='equity')
        with pytest.raises(ValueError):
            earnings_power_value_valuation(100.0, 0.2, 0.1, 10.0, basis='nope')

    def test_enterprise_basis_unchanged(self):
        v = earnings_power_value_valuation(100.0, 0.20, 0.10, 10.0, total_debt=200.0)
        assert v.method == 'epv_zero_growth'
        assert v.value == pytest.approx((800.0 - 200.0) / 10.0)

    def test_pipeline_routes_lenders_by_net_interest_income(self):
        src = (az.__file__ and open(az.__file__, encoding='utf-8').read())
        assert ("_epv_lender = (sector == 'Financial Services' and (\n"
                "                not _facts_blob\n"
                "                or SECXBRLClient._reports_net_interest_income(_facts_blob)))") in src
        assert "basis='equity')" in src and "'epv_bridge': 'equity' if _epv_lender" in src


class TestBlendConflict:
    def test_two_models_far_apart_give_no_fair_value(self):
        r = {'price': 13.0, 'dcf_fv': None, 'epv_growth_fv': 18.0, 'rim_fv': 154.0}
        prepare_scoring_fields([r])
        assert r['_fv_effective'] is None and r['_fv_source'] == 'conflict'
        assert r.get('mos') is None

    def test_two_models_close_still_blend(self):
        r = {'price': 13.0, 'dcf_fv': None, 'epv_growth_fv': 18.0, 'rim_fv': 30.0}
        prepare_scoring_fields([r])
        assert r['_fv_source'] == 'blend' and r['_fv_effective'] == pytest.approx(24.0)

    def test_three_models_keep_the_median(self):
        r = {'price': 13.0, 'dcf_fv': None, 'epv_growth_fv': 18.0, 'rim_fv': 154.0,
             'ddm_fv': 20.0}
        prepare_scoring_fields([r])
        assert r['_fv_source'] == 'blend' and r['_fv_effective'] == pytest.approx(20.0)
