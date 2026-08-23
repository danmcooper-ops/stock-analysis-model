# tests/test_xbrl_cashflow_rows.py
"""SBC + shareholder-return rows in the XBRL cash-flow reconstruction.

On the primary (US-filer) path the pipeline swaps yfinance's cash-flow
frame for build_yfinance_shape's reconstruction. Until that frame carried
'Stock Based Compensation', 'Common Stock Dividend Paid',
'Repurchase Of Capital Stock' and 'Issuance Of Capital Stock', the
owner-earnings SBC haircut in run_forward_dcf and the buyback half of
_compute_shareholder_yield silently never fired for US companies.

Synthetic companyfacts stubs; no live SEC API calls.
"""

import pandas as pd
import pytest


from data.sec_xbrl_client import SECXBRLClient
from scripts.analyze_stock import _compute_shareholder_yield, run_forward_dcf


def _make_client():
    return SECXBRLClient(
        cik_map={'TEST': '0000000001'},
        name_map={'TEST': 'Test Co'},
        email='test@example.com',
        request_delay=0,
    )


def _annual_entries(years_values, form='10-K'):
    return [{
        'form': form, 'fy': fy, 'fp': 'FY', 'val': val,
        'filed': f'{fy + 1}-01-15',
        'start': f'{fy}-01-01', 'end': f'{fy}-12-31',
    } for fy, val in years_values.items()]


def _company_facts(concepts):
    """Build a companyfacts stub: {tag: {fy: value}} -> raw JSON shape."""
    return {'facts': {'us-gaap': {
        tag: {'units': {'USD': _annual_entries(years_values)}}
        for tag, years_values in concepts.items()
    }}}


YEARS = (2021, 2022, 2023, 2024)


def _built_shape():
    """A US filer with SBC, dividends, buybacks and issuance tagged."""
    c = _make_client()
    c._cache['TEST'] = _company_facts({
        'Revenues':                    {y: 1000.0 + 50 * i for i, y in enumerate(YEARS)},
        'NetIncomeLoss':               {y: 150.0 for y in YEARS},
        'OperatingIncomeLoss':         {y: 200.0 for y in YEARS},
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest':
                                       {y: 190.0 for y in YEARS},
        'StockholdersEquity':          {y: 800.0 for y in YEARS},
        'Assets':                      {y: 2000.0 for y in YEARS},
        'NetCashProvidedByUsedInOperatingActivities':
                                       {y: 300.0 for y in YEARS},
        'PaymentsToAcquirePropertyPlantAndEquipment':
                                       {y: 50.0 for y in YEARS},
        'DepreciationDepletionAndAmortization':
                                       {y: 40.0 for y in YEARS},
        'ShareBasedCompensation':      {y: 60.0 for y in YEARS},
        'PaymentsOfDividendsCommonStock':
                                       {y: 30.0 for y in YEARS},
        'PaymentsForRepurchaseOfCommonStock':
                                       {y: 100.0 for y in YEARS},
        'ProceedsFromIssuanceOfCommonStock':
                                       {y: 20.0 for y in YEARS},
    })
    shape = c.build_yfinance_shape('TEST')
    assert shape is not None
    return shape


class TestCashFlowRows:
    def test_all_rows_present(self):
        cf = _built_shape()['cash_flow']
        for row in ('Operating Cash Flow', 'Capital Expenditure',
                    'Depreciation And Amortization', 'Stock Based Compensation',
                    'Common Stock Dividend Paid', 'Repurchase Of Capital Stock',
                    'Issuance Of Capital Stock'):
            assert row in cf.index, f'missing row: {row}'

    def test_yfinance_sign_conventions(self):
        """Outflows negative (capex, dividends, buybacks); SBC and issuance
        proceeds positive — matching native yfinance frames so every
        consumer's abs()/addition logic reads identically on both paths."""
        latest = _built_shape()['cash_flow'].iloc[:, 0]
        assert latest['Capital Expenditure'] == -50.0
        assert latest['Common Stock Dividend Paid'] == -30.0
        assert latest['Repurchase Of Capital Stock'] == -100.0
        assert latest['Stock Based Compensation'] == 60.0
        assert latest['Issuance Of Capital Stock'] == 20.0

    def test_newest_column_first(self):
        cf = _built_shape()['cash_flow']
        assert cf.columns[0] == pd.Timestamp(year=2024, month=12, day=31)

    def test_untagged_rows_are_nan_not_absent(self):
        """A filer that tags none of the new concepts still gets the rows
        (as NaN), and consumers' pd.notna guards skip them safely."""
        c = _make_client()
        c._cache['TEST'] = _company_facts({
            'Revenues':      {y: 1000.0 for y in YEARS},
            'NetIncomeLoss': {y: 150.0 for y in YEARS},
        })
        cf = c.build_yfinance_shape('TEST')['cash_flow']
        assert 'Stock Based Compensation' in cf.index
        assert cf.loc['Stock Based Compensation'].isna().all()


class TestShareholderYieldOnXbrlPath:
    def test_buybacks_and_issuance_flow_through(self):
        """Buyback rate must reflect net repurchases (100 - 20 issuance),
        not the fabricated zero the 3-row frame used to produce."""
        shape = _built_shape()
        mcap = 10_000.0
        result = _compute_shareholder_yield(shape, mcap)
        assert result is not None
        assert result['buyback_rate'] == pytest.approx((100.0 - 20.0) / mcap)
        assert result['shareholder_yield'] == pytest.approx(
            (30.0 + 100.0 - 20.0) / mcap)


class TestSbcHaircutOnXbrlPath:
    def test_sbc_reduces_fair_value(self):
        """run_forward_dcf must value the same company lower when the frame
        carries its SBC than when SBC is untagged — the haircut used to be
        dead on the XBRL path because the row didn't exist."""
        shape = _built_shape()
        info = {'sharesOutstanding': 100.0, 'marketCap': 10_000.0,
                'currentPrice': 100.0}

        with_sbc = dict(shape, info=info)
        fv_with_sbc, _, _, _, _ = run_forward_dcf(
            with_sbc, wacc=0.10, sector='Technology')

        no_sbc_cf = shape['cash_flow'].copy()
        no_sbc_cf.loc['Stock Based Compensation'] = float('nan')
        without_sbc = dict(shape, cash_flow=no_sbc_cf, info=info)
        fv_without_sbc, _, _, _, _ = run_forward_dcf(
            without_sbc, wacc=0.10, sector='Technology')

        assert fv_with_sbc is not None and fv_without_sbc is not None
        assert fv_with_sbc < fv_without_sbc
