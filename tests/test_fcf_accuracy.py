# tests/test_fcf_accuracy.py
"""Free-cash-flow accuracy regressions (September 2026 FCF review).

Findings, each verified against live EDGAR companyfacts before the fix:

1. Fiscal-year keying. `_extract_annual_values` keyed each XBRL entry by its
   own 'fy'. For Jan/Feb year-end filers that label a fiscal year by the
   calendar year it STARTS in (HD, LOW, TGT, KR, DG, ROST, CRM, ...), the
   newest 10-K's prior-year comparative carries the newest fy, ties with the
   current-year entry on filed date, and wins on JSON order — so every year
   in the history was the PRIOR year's figure and the "latest" year was
   stale (HD 'FY2025' OCF = $19.81B, the fiscal-2024 figure; actual $16.33B).
2. OCF holes. Filers with discontinued operations (DIS, JCI, DRI, T, HON)
   tag only NetCashProvidedByUsedInOperatingActivitiesContinuingOperations,
   and most large caps used that element for FY2014-2017. Unmapped, those
   years were holes (196 S&P 500 tickers, 909 missing years).
3. Capex under PaymentsToAcquireOtherPropertyPlantAndEquipment (LLY $7.8B,
   ADP) was unmapped — FCF None, DCF skipped.
4. Stale FCF. OCF + Capex exists only for years with both rows; when capex
   is untagged for the newest year(s) the "most recent" FCF was silently an
   old year (ABNB 2022, JCI 2021) priced against today's market cap.
5. Mean-reversion ceiling averaged ALL positive FCF years. yfinance gave
   4-5 years; XBRL gives 10-17, cutting base FCF 40-67% for compounders.
6. The DCF discounted a levered flow (OCF is after interest) at WACC and
   then subtracted net debt — the cost of debt charged twice.
7. CAGRs indexed by list position, so a hole stretched a "5-year" CAGR.
"""

import pandas as pd
import pytest

from data.sec_xbrl_client import SECXBRLClient
from scripts.analyze_stock import (
    _after_tax_interest, _fcf_series_from_cashflow, _fcf_series_is_current,
    derive_edgar_metrics, run_forward_dcf,
)
from scripts.config import YIELD_CEILING_MULT, YIELD_CEILING_WINDOW

OCF_TAG = 'NetCashProvidedByUsedInOperatingActivities'
OCF_CONT_TAG = 'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'


def _client():
    return SECXBRLClient(cik_map={'T': '0000000001'}, name_map={'T': 'T'},
                         email='test@example.com', request_delay=0)


def _facts(tag_entries):
    return {'facts': {'us-gaap': {
        tag: {'units': {'USD': ents}} for tag, ents in tag_entries.items()}}}


def _e(fy, start, end, filed, val, form='10-K', fp='FY'):
    return {'form': form, 'fy': fy, 'fp': fp, 'start': start, 'end': end,
            'filed': filed, 'val': val}


# HD-shaped: fiscal 2025 ends 2026-02-01. Entries in the order companyfacts
# lists them (grouped by period, oldest first), exactly as observed live.
_HD_OCF = [
    _e(2023, '2022-01-31', '2023-01-29', '2024-03-13', 14_615),   # comparative
    _e(2023, '2023-01-30', '2024-01-28', '2024-03-13', 21_172),   # FY2023 current
    _e(2024, '2023-01-30', '2024-01-28', '2025-03-21', 21_172),   # comparative
    _e(2025, '2023-01-30', '2024-01-28', '2026-03-18', 21_172),   # comparative
    _e(2024, '2024-01-29', '2025-02-02', '2025-03-21', 19_810),   # FY2024 current
    _e(2025, '2024-01-29', '2025-02-02', '2026-03-18', 19_810),   # comparative
    _e(2025, '2025-02-03', '2026-02-01', '2026-03-18', 16_325),   # FY2025 current
]


class TestFiscalYearKeying:
    def test_jan_fye_labelled_by_start_year_is_not_shifted(self):
        vals = _client()._extract_annual_values(_facts({OCF_TAG: _HD_OCF}), [OCF_TAG])
        assert vals[2025] == 16_325, 'latest year must be the CURRENT period, not the comparative'
        assert vals[2024] == 19_810
        assert vals[2023] == 21_172

    def test_restated_comparative_value_wins_but_label_stays(self):
        ents = list(_HD_OCF)
        # FY2025 10-K restates fiscal 2024 (end 2025-02-02) upward.
        ents[5] = _e(2025, '2024-01-29', '2025-02-02', '2026-03-18', 19_900)
        vals = _client()._extract_annual_values(_facts({OCF_TAG: ents}), [OCF_TAG])
        assert vals[2024] == 19_900   # latest filed value ...
        assert vals[2025] == 16_325   # ... under the ORIGINAL filing's label

    def test_dec_fye_unchanged(self):
        ents = [
            _e(2023, '2023-01-01', '2023-12-31', '2024-02-15', 100),
            _e(2024, '2023-01-01', '2023-12-31', '2025-02-15', 100),   # comparative
            _e(2024, '2024-01-01', '2024-12-31', '2025-02-15', 110),
        ]
        assert _client()._extract_annual_values(_facts({OCF_TAG: ents}), [OCF_TAG]) == {
            2023: 100, 2024: 110}

    def test_jan_fye_labelled_by_end_year_unchanged(self):
        # WMT / NVDA style: fiscal 2026 ends Jan 2026, SEC fy=2026.
        ents = [
            _e(2025, '2024-02-01', '2025-01-31', '2025-03-15', 50),
            _e(2026, '2024-02-01', '2025-01-31', '2026-03-15', 50),   # comparative, fy-1
            _e(2026, '2025-02-01', '2026-01-31', '2026-03-15', 55),
        ]
        assert _client()._extract_annual_values(_facts({OCF_TAG: ents}), [OCF_TAG]) == {
            2025: 50, 2026: 55}

    def test_point_in_time_balance_sheet_entries(self):
        tag = 'Assets'
        ents = [
            {'form': '10-K', 'fy': 2024, 'fp': 'FY', 'end': '2024-01-28', 'filed': '2024-03-13', 'val': 1},
            {'form': '10-K', 'fy': 2024, 'fp': 'FY', 'end': '2025-02-02', 'filed': '2025-03-21', 'val': 2},
            {'form': '10-K', 'fy': 2025, 'fp': 'FY', 'end': '2025-02-02', 'filed': '2026-03-18', 'val': 2},
            {'form': '10-K', 'fy': 2025, 'fp': 'FY', 'end': '2026-02-01', 'filed': '2026-03-18', 'val': 3},
        ]
        vals = _client()._extract_annual_values(_facts({tag: ents}), [tag])
        assert vals[2025] == 3 and vals[2024] == 2


class TestTagCoverage:
    def test_continuing_ops_ocf_fills_holes_total_wins_when_both(self):
        c = _client()
        facts = _facts({
            OCF_TAG: [_e(2024, '2024-01-01', '2024-12-31', '2025-02-10', 1_000)],
            OCF_CONT_TAG: [
                _e(2023, '2023-01-01', '2023-12-31', '2024-02-10', 900),
                _e(2024, '2024-01-01', '2024-12-31', '2025-02-10', 990),   # same filing
            ],
        })
        vals = c._extract_annual_values(facts, c._XBRL_TAG_MAP['operating_cash_flow'])
        assert vals == {2023: 900, 2024: 1_000}

    def test_other_ppe_capex_tag_mapped(self):
        c = _client()
        assert 'PaymentsToAcquireOtherPropertyPlantAndEquipment' in c._XBRL_TAG_MAP['capex']
        facts = _facts({'PaymentsToAcquireOtherPropertyPlantAndEquipment': [
            _e(2025, '2025-01-01', '2025-12-31', '2026-02-10', 7_841)]})
        assert c._extract_annual_values(facts, c._XBRL_TAG_MAP['capex']) == {2025: 7_841}


def _cf(rows_by_year):
    cols = pd.to_datetime([f'{y}-12-31' for y in rows_by_year])
    return pd.DataFrame({c: rows for c, rows in zip(cols, rows_by_year.values(), strict=False)})


class TestStaleFcf:
    def test_series_ending_before_newest_column_is_not_current(self):
        cf = _cf({2023: {'Operating Cash Flow': 100, 'Capital Expenditure': -20},
                  2024: {'Operating Cash Flow': 110, 'Capital Expenditure': -25},
                  2025: {'Operating Cash Flow': 120, 'Capital Expenditure': None}})
        s = _fcf_series_from_cashflow(cf)
        assert s.index[-1].year == 2024
        assert not _fcf_series_is_current(cf, s)

    def test_derivation_preferred_when_fcf_row_stops_short(self):
        cf = _cf({2024: {'Free Cash Flow': 85, 'Operating Cash Flow': 110, 'Capital Expenditure': -25},
                  2025: {'Free Cash Flow': None, 'Operating Cash Flow': 120, 'Capital Expenditure': -30}})
        s = _fcf_series_from_cashflow(cf)
        assert s.index[-1].year == 2025 and s.iloc[-1] == 90
        assert _fcf_series_is_current(cf, s)

    def test_dcf_skips_stale_base(self):
        cf = _cf({2022: {'Operating Cash Flow': 1_000, 'Capital Expenditure': -200},
                  2023: {'Operating Cash Flow': 1_100, 'Capital Expenditure': -200},
                  2024: {'Operating Cash Flow': 1_200, 'Capital Expenditure': None},
                  2025: {'Operating Cash Flow': 1_300, 'Capital Expenditure': None}})
        inc = pd.DataFrame({c: {'Total Revenue': 10_000.0} for c in cf.columns})
        out = run_forward_dcf({'cash_flow': cf, 'income_statement': inc,
                               'info': {'marketCap': 20_000.0, 'sharesOutstanding': 1_000.0}},
                              wacc=0.10)
        assert out == (None, None, None, {}, None)


def _yf_data(fcf_by_year, interest=None, tax=None, pretax=None, capex=200.0, da=300.0):
    years = pd.to_datetime([f'{2010 + i}-12-31' for i in range(len(fcf_by_year))])
    cf = pd.DataFrame({y: {'Free Cash Flow': v, 'Operating Cash Flow': v + capex,
                           'Depreciation And Amortization': da,
                           'Capital Expenditure': -capex, 'Stock Based Compensation': 0.0}
                       for y, v in zip(years, fcf_by_year, strict=False)})
    inc_rows = {'Total Revenue': 10_000.0}
    if interest is not None:
        inc_rows['Interest Expense'] = interest
    if tax is not None:
        inc_rows['Tax Provision'] = tax
    if pretax is not None:
        inc_rows['Pretax Income'] = pretax
    inc = pd.DataFrame({y: inc_rows for y in years})
    info = {'marketCap': 20_000.0, 'sharesOutstanding': 1_000.0,
            'totalDebt': 0.0, 'totalCash': 0.0, 'currentPrice': 20.0}
    return {'cash_flow': cf, 'income_statement': inc, 'info': info}


def _base(**kw):
    sector = kw.pop('sector', None)
    _, _, _, diag, _ = run_forward_dcf(_yf_data(**kw), wacc=0.10, sector=sector)
    return diag


class TestCeilingWindow:
    def test_ceiling_uses_trailing_window_not_whole_history(self):
        fcf = [1000.0 * 1.2 ** i for i in range(12)]   # 12 years, +20%/yr
        diag = _base(fcf_by_year=fcf)
        window = fcf[-YIELD_CEILING_WINDOW:]
        expected = min(fcf[-1], YIELD_CEILING_MULT * sum(window) / len(window))
        assert diag['base_fcf'] == pytest.approx(expected)
        whole = YIELD_CEILING_MULT * sum(fcf) / len(fcf)
        assert diag['base_fcf'] > whole, 'the old whole-history cap must no longer bind'

    def test_flat_history_uncapped(self):
        diag = _base(fcf_by_year=[1000.0] * 8)
        assert diag['base_fcf'] == pytest.approx(1000.0)


class TestInterestAddBack:
    def test_after_tax_interest_added_to_base(self):
        plain = _base(fcf_by_year=[1000.0] * 5)
        lev = _base(fcf_by_year=[1000.0] * 5, interest=100.0, tax=20.0, pretax=100.0)
        assert lev['interest_addback'] == pytest.approx(80.0)
        assert lev['base_fcf'] == pytest.approx(plain['base_fcf'] + 80.0)

    def test_financials_excluded(self):
        lev = _base(fcf_by_year=[1000.0] * 5, interest=100.0, tax=20.0, pretax=100.0,
                    sector='Financial Services')
        assert lev['interest_addback'] == 0.0

    def test_default_rate_when_pretax_nonpositive_and_clamp(self):
        yf = _yf_data([1000.0] * 3, interest=-100.0, tax=50.0, pretax=-10.0)
        assert _after_tax_interest(yf) == pytest.approx(100.0 * 0.79)
        yf = _yf_data([1000.0] * 3, interest=100.0, tax=90.0, pretax=100.0)
        assert _after_tax_interest(yf) == pytest.approx(100.0 * 0.65)
        assert _after_tax_interest(_yf_data([1000.0] * 3)) == 0.0


class TestGapAwareEdgarMetrics:
    def test_fcf_cagr_keyed_by_year_across_a_hole(self):
        ocf = {y: 100.0 * 1.1 ** (y - 2015) for y in range(2015, 2026) if y != 2021}
        cap = {y: 10.0 for y in ocf}
        out = derive_edgar_metrics({'operating_cf_history': ocf, 'capex_history': cap})
        fcf = {y: ocf[y] - cap[y] for y in ocf}
        assert out['fcf_cagr_5y'] == pytest.approx((fcf[2025] / fcf[2020]) ** 0.2 - 1)
        assert out['fcf_cagr_10y'] == pytest.approx((fcf[2025] / fcf[2015]) ** 0.1 - 1)
        assert out['fcf_edgar'] == pytest.approx(fcf[2025])

    def test_hole_at_base_year_yields_none_not_a_stretched_cagr(self):
        ocf = {y: 100.0 for y in range(2015, 2026) if y != 2020}
        cap = {y: 10.0 for y in ocf}
        out = derive_edgar_metrics({'operating_cf_history': ocf, 'capex_history': cap})
        assert out['fcf_cagr_5y'] is None

    def test_fcf_edgar_none_when_latest_capex_missing(self):
        ocf = {2023: 100.0, 2024: 110.0, 2025: 120.0}
        cap = {2023: 10.0, 2024: 10.0}
        out = derive_edgar_metrics({'operating_cf_history': ocf, 'capex_history': cap})
        assert out['fcf_edgar'] is None

    def test_rev_cagr_keyed_by_year(self):
        rev = {y: 100.0 * 1.05 ** (y - 2015) for y in range(2015, 2026) if y != 2022}
        out = derive_edgar_metrics({'revenue_history': rev})
        assert out['rev_cagr_5y'] == pytest.approx(0.05)
        assert out['rev_cagr_10y'] == pytest.approx(0.05)
