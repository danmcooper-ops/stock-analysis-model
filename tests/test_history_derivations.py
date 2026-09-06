# tests/test_history_derivations.py
"""Derived income-statement series in fetch_historical_financials, and the
Fund Growth gate's independence from the DCF path.

build_yfinance_shape has long filled the row-level statements from
accounting identities (operating = pretax + interest expense) for filers that
never tag OperatingIncomeLoss. The multi-year series behind Mult vs Hist,
Pool Share, Margin vs Hist, the int_cov_edgar fallback and the gross-margin
trend did not, so those gates read N/A for HCA, ZTS, ADP, MRK, LLY, COP and
~120 other filers whose point-in-time statements resolved fine. Gross profit
likewise: ~380 filers tag CostOfRevenue but no GrossProfit.

Synthetic companyfacts stubs; no live SEC API calls.
"""

import pytest


from data.sec_xbrl_client import SECXBRLClient
from scripts.analyze_stock import _ensure_fundamental_growth, run_forward_dcf

PRETAX_TAG = ('IncomeLossFromContinuingOperationsBeforeIncomeTaxes'
              'ExtraordinaryItemsNoncontrollingInterest')
YEARS = (2021, 2022, 2023, 2024)


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


def _history(concepts):
    c = _make_client()
    c._cache['TEST'] = _company_facts(concepts)
    return c.fetch_historical_financials('TEST')


class TestOperatingIncomeDerivation:
    def test_derived_from_pretax_plus_interest_when_untagged(self):
        """HCA-style filer: pretax income and interest expense tagged, no
        OperatingIncomeLoss at all — the series is now pretax + interest,
        exactly what build_yfinance_shape puts in the row-level statement."""
        h = _history({
            'Revenues':        {y: 1000.0 for y in YEARS},
            'NetIncomeLoss':   {y: 150.0 for y in YEARS},
            PRETAX_TAG:        {y: 190.0 + i for i, y in enumerate(YEARS)},
            'InterestExpense': {y: 10.0 for y in YEARS},
        })
        assert h['operating_income_history'] == {
            2021: 200.0, 2022: 201.0, 2023: 202.0, 2024: 203.0}
        assert list(h['operating_income_history']) == sorted(YEARS)

    def test_missing_interest_expense_treated_as_zero(self):
        h = _history({
            'Revenues':      {y: 1000.0 for y in YEARS},
            'NetIncomeLoss': {y: 150.0 for y in YEARS},
            PRETAX_TAG:      {y: 190.0 for y in YEARS},
        })
        assert h['operating_income_history'] == {y: 190.0 for y in YEARS}

    def test_tagged_years_never_overwritten_holes_filled(self):
        """A filer that tagged OperatingIncomeLoss for some years keeps those
        values; only the untagged years are derived."""
        h = _history({
            'Revenues':            {y: 1000.0 for y in YEARS},
            'NetIncomeLoss':       {y: 150.0 for y in YEARS},
            'OperatingIncomeLoss': {2023: 500.0, 2024: 520.0},
            PRETAX_TAG:            {y: 190.0 for y in YEARS},
            'InterestExpense':     {y: 10.0 for y in YEARS},
        })
        assert h['operating_income_history'] == {
            2021: 200.0, 2022: 200.0, 2023: 500.0, 2024: 520.0}

    def test_pretax_last_resort_from_net_income_plus_tax(self):
        """No pretax tag either: pretax = NI + tax provision, then operating
        follows — the same two-step chain as the shape builder."""
        h = _history({
            'Revenues':                {y: 1000.0 for y in YEARS},
            'NetIncomeLoss':           {y: 150.0 for y in YEARS},
            'IncomeTaxExpenseBenefit': {y: 40.0 for y in YEARS},
            'InterestExpense':         {y: 10.0 for y in YEARS},
        })
        assert h['pretax_income_history'] == {y: 190.0 for y in YEARS}
        assert h['operating_income_history'] == {y: 200.0 for y in YEARS}

    def test_nothing_derived_without_inputs(self):
        h = _history({
            'Revenues':      {y: 1000.0 for y in YEARS},
            'NetIncomeLoss': {y: 150.0 for y in YEARS},
        })
        assert h['operating_income_history'] == {}
        assert h['pretax_income_history'] == {}

    def test_years_available_not_inflated_by_derivation(self):
        """years_available gates backfill refetch upstream; a derived series
        cannot exceed the net-income series it is built from."""
        h = _history({
            'Revenues':      {2023: 1000.0, 2024: 1000.0},
            'NetIncomeLoss': {2023: 150.0, 2024: 150.0},
            PRETAX_TAG:      {y: 190.0 for y in YEARS},
            'InterestExpense': {y: 10.0 for y in YEARS},
        })
        assert h['years_available'] == 4  # the tagged pretax span, not more


class TestGrossProfitDerivation:
    def test_revenue_minus_cost_of_revenue(self):
        h = _history({
            'Revenues':      {y: 1000.0 for y in YEARS},
            'NetIncomeLoss': {y: 150.0 for y in YEARS},
            'CostOfRevenue': {y: 600.0 + 10 * i for i, y in enumerate(YEARS)},
        })
        assert h['gross_profit_history'] == {
            2021: 400.0, 2022: 390.0, 2023: 380.0, 2024: 370.0}

    def test_tagged_gross_profit_wins(self):
        h = _history({
            'Revenues':      {y: 1000.0 for y in YEARS},
            'NetIncomeLoss': {y: 150.0 for y in YEARS},
            'CostOfRevenue': {y: 600.0 for y in YEARS},
            'GrossProfit':   {2024: 425.0},
        })
        assert h['gross_profit_history'][2024] == 425.0
        assert h['gross_profit_history'][2021] == 400.0

    def test_no_cost_of_revenue_leaves_series_empty(self):
        """Services filers with no COGS line stay N/A on the margin trend —
        a derived zero-cost gross margin would be fiction."""
        h = _history({
            'Revenues':      {y: 1000.0 for y in YEARS},
            'NetIncomeLoss': {y: 150.0 for y in YEARS},
        })
        assert h['gross_profit_history'] == {}

    def test_derived_series_follows_fx_conversion(self, monkeypatch):
        """Derivation runs in the reporting currency, before the FX pass, so
        the derived gross profit converts like the tagged lines."""
        import data.sec_xbrl_client as xbrl_mod
        c = _make_client()
        facts = {'facts': {'us-gaap': {
            tag: {'units': {'EUR': _annual_entries(vals, form='20-F')}}
            for tag, vals in {
                'Revenues':      {2023: 1000.0, 2024: 1000.0},
                'NetIncomeLoss': {2023: 150.0, 2024: 150.0},
                'CostOfRevenue': {2023: 600.0, 2024: 600.0},
            }.items()
        }}}
        c._cache['TEST'] = facts
        monkeypatch.setattr(xbrl_mod, '_get_fx_rates_to_usd',
                            lambda ccy: {2023: 1.10, 2024: 1.05})
        h = c.fetch_historical_financials('TEST')
        assert h['fx_converted'] is True
        assert h['gross_profit_history'][2023] == pytest.approx(440.0)
        assert h['gross_profit_history'][2024] == pytest.approx(420.0)


def _negative_fcf_shape():
    """A profitable, positive-ROIC filer whose capex exceeds operating cash
    flow: the DCF exits on a non-positive base FCF before its growth-signal
    block, so its diagnostic never carried fundamental_growth."""
    c = _make_client()
    c._cache['TEST'] = _company_facts({
        'Revenues':            {y: 1000.0 + 50 * i for i, y in enumerate(YEARS)},
        'NetIncomeLoss':       {y: 150.0 for y in YEARS},
        'OperatingIncomeLoss': {y: 200.0 for y in YEARS},
        PRETAX_TAG:            {y: 190.0 for y in YEARS},
        'IncomeTaxExpenseBenefit': {y: 40.0 for y in YEARS},
        'StockholdersEquity':  {y: 800.0 for y in YEARS},
        'Assets':              {y: 2000.0 for y in YEARS},
        'NetCashProvidedByUsedInOperatingActivities':
                               {y: 100.0 for y in YEARS},
        'PaymentsToAcquirePropertyPlantAndEquipment':
                               {y: 250.0 for y in YEARS},
        'DepreciationDepletionAndAmortization':
                               {y: 40.0 for y in YEARS},
    })
    shape = c.build_yfinance_shape('TEST')
    assert shape is not None
    return shape


class TestFundGrowthIndependentOfDcf:
    def test_dcf_skips_but_fund_growth_still_computed(self):
        shape = _negative_fcf_shape()
        fv, _sens, _g, diag, _mc = run_forward_dcf(shape, wacc=0.09)
        assert fv is None
        assert diag.get('fundamental_growth') is None   # the DCF path bailed
        roic_data = {'roic_median_5y': 0.20}
        out = _ensure_fundamental_growth(diag, shape, roic_data)
        assert out['fundamental_growth'] is not None
        assert out['fundamental_growth'] > 0
        # Capex (250) - D&A (40) against NOPAT ~150 clamps reinvestment at 1.0,
        # so growth = ROIC.
        assert out['reinvestment_rate'] == pytest.approx(1.0)
        assert out['fundamental_growth'] == pytest.approx(0.20)
        assert diag.get('fundamental_growth') is None   # input not mutated

    def test_existing_value_left_alone(self):
        diag = {'fundamental_growth': 0.07, 'reinvestment_rate': 0.5}
        assert _ensure_fundamental_growth(diag, {}, None) is diag

    def test_uncomputable_stays_na(self):
        out = _ensure_fundamental_growth({}, {'income_statement': None,
                                              'cash_flow': None}, None)
        assert out.get('fundamental_growth') is None


def _entries(years_values, filed_offset='01-15'):
    return [{
        'form': '10-K', 'fy': fy, 'fp': 'FY', 'val': val,
        'filed': f'{fy + 1}-{filed_offset}',
        'start': f'{fy}-01-01', 'end': f'{fy}-12-31',
    } for fy, val in years_values.items()]


def _facts(tag_years):
    return {'facts': {'us-gaap': {
        tag: {'units': {'USD': _entries(vals)}} for tag, vals in tag_years.items()
    }}}


class TestRevenueTagStitching:
    def test_pre_2018_services_total_extends_history(self):
        """HCA-shaped: SalesRevenueServicesNet through FY2017, the ASC 606 tag
        from FY2018 — one continuous series instead of one starting 2018."""
        c = _make_client()
        facts = _facts({
            'SalesRevenueServicesNet': {2013: 34.0, 2014: 37.0, 2015: 40.0,
                                        2016: 41.0, 2017: 44.0},
            'RevenueFromContractWithCustomerExcludingAssessedTax':
                {2018: 47.0, 2019: 51.0, 2020: 52.0},
        })
        rev, _t, _c = c._extract_concept_annual(facts, 'revenue')
        assert rev == {2013: 34.0, 2014: 37.0, 2015: 40.0, 2016: 41.0,
                       2017: 44.0, 2018: 47.0, 2019: 51.0, 2020: 52.0}

    def test_same_filing_tie_keeps_whole_company_total(self):
        """NJR-shaped: the operating-revenues total and the ASC 606 slice are
        tagged in the same filing for the same year. The larger one is the
        company's revenue, whatever its position in the tag list."""
        c = _make_client()
        facts = _facts({
            'RevenueFromContractWithCustomerExcludingAssessedTax':
                {2019: 782.0, 2020: 834.0},
            'RegulatedAndUnregulatedOperatingRevenue':
                {2018: 2915.0, 2019: 2592.0, 2020: 1954.0},
        })
        rev, _t, _c = c._extract_concept_annual(facts, 'revenue')
        assert rev == {2018: 2915.0, 2019: 2592.0, 2020: 1954.0}

    def test_later_filing_still_supersedes_regardless_of_size(self):
        """A restatement filed later wins even when it is smaller: the max
        rule only breaks exact ties on filed date."""
        c = _make_client()
        facts = _facts({'Revenues': {2020: 1000.0}})
        facts['facts']['us-gaap']['Revenues']['units']['USD'].append({
            'form': '10-K', 'fy': 2020, 'fp': 'FY', 'val': 950.0,
            'filed': '2021-06-30', 'start': '2020-01-01', 'end': '2020-12-31'})
        rev, _t, _c = c._extract_concept_annual(facts, 'revenue')
        assert rev == {2020: 950.0}

    def test_explicit_gaap_total_beats_larger_alias(self):
        """EQT-shaped: Revenues (after derivative losses) is smaller than
        the gas-sales alias tagged in the same filing. The explicit total is
        the revenue figure; the size rule only arbitrates among aliases."""
        c = _make_client()
        facts = _facts({
            'Revenues':        {2021: 3065.0},
            'OilAndGasRevenue': {2021: 6804.0},
        })
        rev, _t, _c = c._extract_concept_annual(facts, 'revenue')
        assert rev == {2021: 3065.0}

    def test_first_tag_rule_unchanged_for_other_concepts(self):
        """Net income keeps first-tag-wins: the NCI-inclusive ProfitLoss is
        larger than the parent-only NetIncomeLoss the models want."""
        c = _make_client()
        facts = _facts({
            'NetIncomeLoss': {2020: 100.0},
            'ProfitLoss':    {2020: 120.0},
        })
        ni, _t, _c = c._extract_concept_annual(facts, 'net_income')
        assert ni == {2020: 100.0}

    def test_gross_and_component_health_care_tags_not_in_list(self):
        """Guard against re-adding the tags that fabricate a growth step at
        the 2018 boundary (see the revenue alias comment)."""
        tags = SECXBRLClient._XBRL_TAG_MAP['revenue']
        assert 'SalesRevenueServicesNet' in tags
        for bad in ('HealthCareOrganizationPatientServiceRevenue',
                    'SalesRevenueServicesGross',
                    'OperatingLeasesIncomeStatementLeaseRevenue',
                    'FoodAndBeverageRevenue', 'AdvertisingRevenue',
                    'HomeBuildingRevenue'):
            assert bad not in tags
