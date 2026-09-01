# tests/test_nav_tangible.py
"""Tangible book on the SEC XBRL path + signed TBV for scoring.

Two regressions:

1. build_yfinance_shape replaces the yfinance balance sheet for every US
   filer, and until it carried 'Goodwill' / 'Other Intangible Assets' the
   NAV model found nothing to strip — nav_fv / p_tbv silently degraded to
   plain book value for the whole SEC cohort.
2. tangible_book_value_per_share returned None for negative tangible
   equity, so scoring's "negative tangible book -> P/TBV inapplicable"
   exemption could never fire; buyback-rich compounders were scored as
   a failed valuation test on "missing" data instead.

Synthetic companyfacts stubs; no live SEC API calls.
"""

import pandas as pd
import pytest

from data.sec_xbrl_client import SECXBRLClient
from models.nav import tangible_book_value_per_share, tangible_equity_per_share
from scripts.scoring import _appl_positive_tbv


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
    return {'facts': {'us-gaap': {
        tag: {'units': {'USD': _annual_entries(years_values)}}
        for tag, years_values in concepts.items()
    }}}


YEARS = (2022, 2023, 2024)


def _built_shape(goodwill=300.0, intangibles=100.0, equity=800.0):
    c = _make_client()
    concepts = {
        'Revenues':            {y: 1000.0 for y in YEARS},
        'NetIncomeLoss':       {y: 150.0 for y in YEARS},
        'OperatingIncomeLoss': {y: 200.0 for y in YEARS},
        'StockholdersEquity':  {y: equity for y in YEARS},
        'Assets':              {y: 2000.0 for y in YEARS},
    }
    if goodwill is not None:
        concepts['Goodwill'] = {y: goodwill for y in YEARS}
    if intangibles is not None:
        concepts['IntangibleAssetsNetExcludingGoodwill'] = {y: intangibles for y in YEARS}
    c._cache['TEST'] = _company_facts(concepts)
    shape = c.build_yfinance_shape('TEST')
    assert shape is not None
    return shape


class TestXbrlBalanceSheetRows:
    def test_goodwill_and_intangibles_rows_present(self):
        latest = _built_shape()['balance_sheet'].iloc[:, 0]
        assert latest['Goodwill'] == 300.0
        assert latest['Other Intangible Assets'] == 100.0

    def test_untagged_rows_are_null_not_missing(self):
        """A filer with no goodwill still gets the row (NaN), so consumers
        see the same index on every XBRL frame."""
        bs = _built_shape(goodwill=None, intangibles=None)['balance_sheet']
        assert 'Goodwill' in bs.index
        assert 'Other Intangible Assets' in bs.index
        assert pd.isna(bs.iloc[:, 0]['Goodwill'])

    def test_nav_strips_intangibles_on_xbrl_path(self):
        """The regression: TBV/share must be below plain BV/share."""
        shape = _built_shape()
        shape['info'] = {'sharesOutstanding': 100.0}
        tbv = tangible_book_value_per_share(shape)
        assert tbv == pytest.approx((800.0 - 300.0 - 100.0) / 100.0)
        assert tbv < 800.0 / 100.0

    def test_nav_equals_book_when_filer_has_no_intangibles(self):
        shape = _built_shape(goodwill=None, intangibles=None)
        shape['info'] = {'sharesOutstanding': 100.0}
        assert tangible_book_value_per_share(shape) == pytest.approx(8.0)


def _financials(equity, goodwill=0.0, intangibles=0.0, shares=100.0):
    bs = pd.DataFrame({pd.Timestamp('2024-12-31'): {
        'Stockholders Equity': equity,
        'Goodwill': goodwill,
        'Other Intangible Assets': intangibles,
        'Total Assets': 2000.0,
    }})
    return {'balance_sheet': bs, 'info': {'sharesOutstanding': shares}}


class TestSignedTangibleEquity:
    def test_negative_tangible_book_keeps_sign(self):
        """Goodwill exceeding equity: signed value negative, floor None."""
        fin = _financials(equity=800.0, goodwill=1000.0)
        assert tangible_equity_per_share(fin) == pytest.approx(-2.0)
        assert tangible_book_value_per_share(fin) is None

    def test_negative_equity_keeps_sign(self):
        """Buyback compounders with negative book: signed value negative."""
        fin = _financials(equity=-500.0, goodwill=100.0)
        assert tangible_equity_per_share(fin) == pytest.approx(-6.0)
        assert tangible_book_value_per_share(fin) is None

    def test_positive_case_agrees_with_floor(self):
        fin = _financials(equity=800.0, goodwill=300.0, intangibles=100.0)
        assert tangible_equity_per_share(fin) == pytest.approx(4.0)
        assert tangible_book_value_per_share(fin) == pytest.approx(4.0)

    def test_missing_balance_sheet_is_none(self):
        assert tangible_equity_per_share({}) is None
        assert tangible_equity_per_share({'balance_sheet': pd.DataFrame()}) is None

    def test_missing_shares_is_none(self):
        fin = _financials(equity=800.0)
        fin['info'] = {}
        assert tangible_equity_per_share(fin) is None

    def test_intangible_heavy_warning_only_for_positive_equity(self):
        with pytest.warns(RuntimeWarning, match='treat as floor only'):
            tangible_equity_per_share(_financials(equity=800.0, goodwill=500.0))
        with pytest.warns(RuntimeWarning, match='treat as floor only'):
            tangible_equity_per_share(_financials(equity=800.0, goodwill=1000.0))
        # Negative equity: the ratio is meaningless, so no warning.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            tangible_equity_per_share(_financials(equity=-500.0, goodwill=100.0))


class TestScoringExemptionWiring:
    """End-to-end: the signed value must reach _appl_positive_tbv."""

    def test_negative_tbv_makes_ptbv_inapplicable(self):
        fin = _financials(equity=800.0, goodwill=1000.0)
        row = {'tangible_book_ps': tangible_equity_per_share(fin), 'p_tbv': None}
        assert _appl_positive_tbv(row) is False

    def test_missing_balance_sheet_stays_applicable(self):
        row = {'tangible_book_ps': tangible_equity_per_share({}), 'p_tbv': None}
        assert _appl_positive_tbv(row) is True

    def test_positive_tbv_stays_applicable(self):
        fin = _financials(equity=800.0, goodwill=300.0)
        row = {'tangible_book_ps': tangible_equity_per_share(fin), 'p_tbv': 2.0}
        assert _appl_positive_tbv(row) is True


# ---------------------------------------------------------------------------
# Parent-attributable equity + period-matched share count on the XBRL path
# ---------------------------------------------------------------------------

def _facts_with_units(concepts):
    """Like _company_facts but each concept picks its own units key."""
    return {'facts': {'us-gaap': {
        tag: {'units': {units: _annual_entries(years_values)}}
        for tag, (units, years_values) in concepts.items()
    }}}


def _shape_from(concepts):
    c = _make_client()
    base = {
        'Revenues':            ('USD', {y: 1000.0 for y in YEARS}),
        'NetIncomeLoss':       ('USD', {y: 150.0 for y in YEARS}),
        'OperatingIncomeLoss': ('USD', {y: 200.0 for y in YEARS}),
        'Assets':              ('USD', {y: 2000.0 for y in YEARS}),
    }
    base.update(concepts)
    c._cache['TEST'] = _facts_with_units(base)
    shape = c.build_yfinance_shape('TEST')
    assert shape is not None
    return shape


_INCL = 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'


class TestParentEquityResolution:
    def test_parent_tag_wins_when_both_present(self):
        latest = _shape_from({
            'StockholdersEquity': ('USD', {y: 800.0 for y in YEARS}),
            _INCL:                ('USD', {y: 950.0 for y in YEARS}),
            'MinorityInterest':   ('USD', {y: 150.0 for y in YEARS}),
        })['balance_sheet'].iloc[:, 0]
        assert latest['Stockholders Equity'] == 800.0

    def test_inclusive_only_subtracts_minority_interest(self):
        latest = _shape_from({
            _INCL:              ('USD', {y: 950.0 for y in YEARS}),
            'MinorityInterest': ('USD', {y: 150.0 for y in YEARS}),
        })['balance_sheet'].iloc[:, 0]
        assert latest['Stockholders Equity'] == 800.0

    def test_inclusive_only_without_nci_tag_uses_total(self):
        latest = _shape_from({
            _INCL: ('USD', {y: 950.0 for y in YEARS}),
        })['balance_sheet'].iloc[:, 0]
        assert latest['Stockholders Equity'] == 950.0

    def test_per_year_fallback(self):
        """Parent tag for recent years, inclusive-only for an older one."""
        latest_year = YEARS[-1]
        old_year = YEARS[0]
        bs = _shape_from({
            'StockholdersEquity': ('USD', {latest_year: 800.0}),
            _INCL:                ('USD', {y: 950.0 for y in YEARS}),
            'MinorityInterest':   ('USD', {y: 150.0 for y in YEARS}),
        })['balance_sheet']
        assert bs.iloc[:, 0]['Stockholders Equity'] == 800.0
        old_col = pd.Timestamp(year=old_year, month=12, day=31)
        assert bs[old_col]['Stockholders Equity'] == 800.0

    def test_history_path_uses_same_resolution(self):
        c = _make_client()
        c._cache['TEST'] = _facts_with_units({
            'Revenues':            ('USD', {y: 1000.0 for y in YEARS}),
            'NetIncomeLoss':       ('USD', {y: 150.0 for y in YEARS}),
            _INCL:                 ('USD', {y: 950.0 for y in YEARS}),
            'MinorityInterest':    ('USD', {y: 150.0 for y in YEARS}),
        })
        vals, _ccy = c._resolve_equity_annual(c._cache['TEST'])
        assert vals[YEARS[-1]] == 800.0


class TestPeriodMatchedShareCount:
    def test_shares_row_emitted_from_instant_tag(self):
        latest = _shape_from({
            'StockholdersEquity':          ('USD', {y: 800.0 for y in YEARS}),
            'CommonStockSharesOutstanding': ('shares', {y: 100.0 for y in YEARS}),
        })['balance_sheet'].iloc[:, 0]
        assert latest['Ordinary Shares Number'] == 100.0

    def test_nav_prefers_period_end_count(self):
        """Year-end 100 shares vs 90 live after buybacks: divide by 100."""
        shape = _shape_from({
            'StockholdersEquity':          ('USD', {y: 800.0 for y in YEARS}),
            'Goodwill':                    ('USD', {y: 300.0 for y in YEARS}),
            'CommonStockSharesOutstanding': ('shares', {y: 100.0 for y in YEARS}),
        })
        shape['info'] = {'sharesOutstanding': 90.0}
        assert tangible_equity_per_share(shape) == pytest.approx(5.0)

    def test_implausible_period_count_falls_back_to_live(self):
        """A single share class (1/3 of the live count) is not the base."""
        fin = _financials(equity=900.0, goodwill=0.0, shares=300.0)
        fin['balance_sheet'].loc['Ordinary Shares Number'] = 100.0
        assert tangible_equity_per_share(fin) == pytest.approx(3.0)

    def test_period_count_used_when_live_missing(self):
        fin = _financials(equity=900.0, goodwill=0.0)
        fin['balance_sheet'].loc['Ordinary Shares Number'] = 100.0
        fin['info'] = {}
        assert tangible_equity_per_share(fin) == pytest.approx(9.0)

    def test_live_count_when_row_absent(self):
        fin = _financials(equity=900.0, goodwill=0.0, shares=300.0)
        assert tangible_equity_per_share(fin) == pytest.approx(3.0)
