# tests/test_xbrl_fiscal_year_match.py
"""Fiscal-year bucketing in SECXBRLClient._extract_annual_values.

Filers whose fiscal year ends in late January / early February label it
by its STARTING calendar year (Home Depot: the year ending 2026-02-01 is
fy=2025). The matcher accepts end_year == fy + 1 for them, but that also
admits the same 10-K's prior-year comparative (end 2025-02-02, fy=2025),
which shares the filed date. Before the end-date tie-break the comparative
won, so every year came out shifted back one and the latest fiscal year was
dropped (verified live on HD, LOW, TGT, DG). Walmart-style filers label by
END year and were never affected; they must stay correct.

Synthetic companyfacts stubs; no live SEC API calls.
"""

from data.sec_xbrl_client import SECXBRLClient
from models.ratios import calculate_roic


def _make_client():
    return SECXBRLClient(
        cik_map={'TEST': '0000000001'},
        name_map={'TEST': 'Test Co'},
        email='test@example.com',
        request_delay=0,
    )


def _entry(fy, start, end, filed, val, form='10-K'):
    e = {'form': form, 'fy': fy, 'fp': 'FY', 'val': val, 'filed': filed, 'end': end}
    if start:
        e['start'] = start
    return e


def _hd_style_facts():
    """fy labelled by starting year; each 10-K carries the prior-year
    comparative FIRST (as EDGAR orders them) with the same filed date."""
    op = [
        # FY2024 10-K (filed 2025-03-21): comparative FY2023 then current
        _entry(2024, '2023-01-30', '2024-01-28', '2025-03-21', 21_689.0),
        _entry(2024, '2024-01-29', '2025-02-02', '2025-03-21', 21_526.0),
        # FY2025 10-K (filed 2026-03-18)
        _entry(2025, '2024-01-29', '2025-02-02', '2026-03-18', 21_526.0),
        _entry(2025, '2025-02-03', '2026-02-01', '2026-03-18', 20_890.0),
    ]
    eq = [
        _entry(2024, None, '2024-01-28', '2025-03-21', 1_044.0),
        _entry(2024, None, '2025-02-02', '2025-03-21', 6_640.0),
        _entry(2025, None, '2025-02-02', '2026-03-18', 6_640.0),
        _entry(2025, None, '2026-02-01', '2026-03-18', 12_813.0),
    ]
    return {'facts': {'us-gaap': {
        'OperatingIncomeLoss': {'units': {'USD': op}},
        'StockholdersEquity': {'units': {'USD': eq}},
        'Revenues': {'units': {'USD': [
            _entry(2024, '2024-01-29', '2025-02-02', '2025-03-21', 159_514.0),
            _entry(2025, '2025-02-03', '2026-02-01', '2026-03-18', 163_000.0),
        ]}},
    }}}


def _wmt_style_facts():
    """fy labelled by END year: the comparative fails the year match."""
    op = [
        _entry(2025, '2024-02-01', '2025-01-31', '2025-03-14', 29_348.0),
        _entry(2026, '2024-02-01', '2025-01-31', '2026-03-13', 29_348.0),
        _entry(2026, '2025-02-01', '2026-01-31', '2026-03-13', 29_820.0),
    ]
    return {'facts': {'us-gaap': {'OperatingIncomeLoss': {'units': {'USD': op}}}}}


class TestStartYearLabelledFilers:
    def test_current_period_beats_prior_year_comparative(self):
        c = _make_client()
        got = c._extract_annual_values(_hd_style_facts(), ['OperatingIncomeLoss'])
        assert got == {2024: 21_526.0, 2025: 20_890.0}

    def test_point_in_time_balance_uses_period_end_not_comparative(self):
        c = _make_client()
        got = c._extract_annual_values(_hd_style_facts(), ['StockholdersEquity'])
        assert got == {2024: 6_640.0, 2025: 12_813.0}

    def test_latest_fiscal_year_reaches_roic(self):
        """End to end: the newest column must carry the newest filing's
        current-period figures, not last year's."""
        c = _make_client()
        c.fetch_company_facts = lambda tk: _hd_style_facts()
        shape = c.build_yfinance_shape('TEST')
        latest = shape['income_statement'].iloc[:, 0]
        assert latest['Operating Income'] == 20_890.0
        assert shape['balance_sheet'].iloc[:, 0]['Stockholders Equity'] == 12_813.0
        roic = calculate_roic(shape)
        assert set(roic['roic_by_year']) == {'2024', '2025'}


class TestEndYearLabelledFilers:
    def test_unaffected(self):
        c = _make_client()
        got = c._extract_annual_values(_wmt_style_facts(), ['OperatingIncomeLoss'])
        assert got == {2025: 29_348.0, 2026: 29_820.0}


class TestRestatements:
    def test_later_filing_still_wins_for_same_period(self):
        """A restated value for the SAME period end filed later replaces the
        original (the pre-existing latest-filed rule is preserved)."""
        c = _make_client()
        facts = {'facts': {'us-gaap': {'OperatingIncomeLoss': {'units': {'USD': [
            _entry(2023, '2023-01-01', '2023-12-31', '2024-02-20', 100.0),
            _entry(2023, '2023-01-01', '2023-12-31', '2024-09-01', 95.0, form='10-K'),
        ]}}}}}
        got = c._extract_annual_values(facts, ['OperatingIncomeLoss'])
        assert got == {2023: 95.0}

    def test_first_tag_keeps_tie(self):
        """Two tags carrying the same period from the same filing: the
        earlier tag in the list is the preferred one and must win the tie."""
        c = _make_client()
        facts = {'facts': {'us-gaap': {
            'StockholdersEquity': {'units': {'USD': [
                _entry(2023, None, '2023-12-31', '2024-02-20', 500.0)]}},
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest':
                {'units': {'USD': [_entry(2023, None, '2023-12-31', '2024-02-20', 520.0)]}},
        }}}
        got = c._extract_annual_values(
            facts, ['StockholdersEquity',
                    'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'])
        assert got == {2023: 500.0}


class TestCombinedCashRow:
    def test_short_term_investments_added_when_tagged(self):
        facts = _hd_style_facts()
        facts['facts']['us-gaap']['CashAndCashEquivalentsAtCarryingValue'] = {'units': {'USD': [
            _entry(2025, None, '2026-02-01', '2026-03-18', 1_389.0)]}}
        facts['facts']['us-gaap']['ShortTermInvestments'] = {'units': {'USD': [
            _entry(2025, None, '2026-02-01', '2026-03-18', 611.0)]}}
        c = _make_client()
        c.fetch_company_facts = lambda tk: facts
        bs = c.build_yfinance_shape('TEST')['balance_sheet']
        assert bs.iloc[:, 0]['Cash And Cash Equivalents'] == 1_389.0
        assert bs.iloc[:, 0]['Cash Cash Equivalents And Short Term Investments'] == 2_000.0
        # A year without a short-term-investments tag leaves the combined
        # row None so consumers fall back to plain cash.
        assert bs.iloc[:, 1].isna()['Cash Cash Equivalents And Short Term Investments']
