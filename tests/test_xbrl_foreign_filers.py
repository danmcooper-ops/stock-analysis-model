"""Foreign private issuers must build usable statement shapes.

Regression coverage for the 2026-09-08 run, which dropped 235 foreign-listed
companies (ASML, AZN, BHP, BP, BTI, ...) from the universe on ROIC N/A.

build_yfinance_shape read every concept through _extract_annual_values, whose
defaults are units_key='USD' and taxonomy_key='us-gaap'. A foreign filer tags
its statements in its own currency and often under the IFRS taxonomy, so every
one of those reads came back empty. Revenue and equity survived only because
they have their own currency-aware resolvers — which is precisely what made
the failure silent: revenue alone cleared the "is this data usable" guard, so
the method returned a shape whose sole populated income row was revenue, and
analyze_stock replaces yfinance's statements with that shape unconditionally.

Before the fix these filers had working yfinance-sourced ROIC; the sparse XBRL
shape blanked it.
"""

import pytest

from data.sec_xbrl_client import SECXBRLClient


def _flow(fy, val):
    return {'form': '20-F', 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-02-01', 'start': f'{fy}-01-01', 'end': f'{fy}-12-31'}


def _pit(fy, val):
    return {'form': '20-F', 'fy': fy, 'fp': 'FY', 'val': val,
            'filed': f'{fy + 1}-02-01', 'end': f'{fy}-12-31'}


def _client(taxonomies, currency='USD'):
    """Client whose TEST filer tags `taxonomies` = {taxonomy: {tag: entries}}."""
    c = SECXBRLClient(cik_map={'TEST': '0000000001'},
                      name_map={'TEST': 'Test Co'},
                      email='test@example.com', request_delay=0)
    c._cache['TEST'] = {'facts': {
        taxo: {tag: {'units': {currency: entries}} for tag, entries in tags.items()}
        for taxo, tags in taxonomies.items()}}
    return c


_US_GAAP_STATEMENT = {
    'Revenues': [_flow(2023, 1000.0), _flow(2024, 1200.0)],
    'NetIncomeLoss': [_flow(2023, 90.0), _flow(2024, 120.0)],
    'OperatingIncomeLoss': [_flow(2023, 150.0), _flow(2024, 200.0)],
    'Assets': [_pit(2023, 4000.0), _pit(2024, 5000.0)],
    'StockholdersEquity': [_pit(2023, 2000.0), _pit(2024, 2500.0)],
    'LongTermDebtNoncurrent': [_pit(2023, 800.0), _pit(2024, 900.0)],
    'CashAndCashEquivalentsAtCarryingValue': [_pit(2023, 300.0), _pit(2024, 400.0)],
}

_IFRS_STATEMENT = {
    'Revenue': [_flow(2023, 1000.0), _flow(2024, 1200.0)],
    'ProfitLoss': [_flow(2023, 90.0), _flow(2024, 120.0)],
    'ProfitLossFromOperatingActivities': [_flow(2023, 150.0), _flow(2024, 200.0)],
    'Assets': [_pit(2023, 4000.0), _pit(2024, 5000.0)],
    'Equity': [_pit(2023, 2000.0), _pit(2024, 2500.0)],
}


@pytest.fixture(autouse=True)
def _fixed_fx(monkeypatch):
    """Pin FX so conversion is deterministic and offline (2024 EUR->USD 1.25)."""
    monkeypatch.setattr('data.sec_xbrl_client._get_fx_rates_to_usd',
                        lambda ccy: {2023: 1.10, 2024: 1.25})


class TestForeignCurrencyStatements:
    def test_eur_filer_populates_the_income_statement(self):
        """The bug: EUR-tagged concepts read through a USD-only default."""
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='EUR')
        shape = c.build_yfinance_shape('TEST')
        assert shape is not None
        inc = shape['income_statement']
        latest = inc.columns[0]
        # Operating Income was NaN before the fix while Total Revenue (which
        # has its own currency-aware resolver) came through.
        assert pd_notna(inc.loc['Operating Income', latest])
        assert pd_notna(inc.loc['Net Income', latest])

    def test_eur_values_are_converted_to_usd(self):
        """Frames merge with yfinance's USD `info`, so they must be USD."""
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='EUR')
        inc = c.build_yfinance_shape('TEST')['income_statement']
        latest = inc.columns[0]
        assert inc.loc['Total Revenue', latest] == pytest.approx(1200.0 * 1.25)
        assert inc.loc['Operating Income', latest] == pytest.approx(200.0 * 1.25)

    def test_usd_filer_is_not_converted(self):
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='USD')
        inc = c.build_yfinance_shape('TEST')['income_statement']
        latest = inc.columns[0]
        assert inc.loc['Total Revenue', latest] == pytest.approx(1200.0)
        assert inc.loc['Operating Income', latest] == pytest.approx(200.0)

    def test_ifrs_taxonomy_populates_the_income_statement(self):
        """AZN and BP tag zero us-gaap concepts — everything is ifrs-full."""
        c = _client({'ifrs-full': _IFRS_STATEMENT}, currency='GBP')
        shape = c.build_yfinance_shape('TEST')
        assert shape is not None
        inc = shape['income_statement']
        latest = inc.columns[0]
        assert pd_notna(inc.loc['Operating Income', latest])
        assert pd_notna(inc.loc['Net Income', latest])

    def test_foreign_debt_is_not_silently_zero(self):
        """Total debt resolved through a USD-only reader read as unlevered."""
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='EUR')
        bs = c.build_yfinance_shape('TEST')['balance_sheet']
        assert bs.loc['Total Debt', bs.columns[0]] == pytest.approx(900.0 * 1.25)


class TestXbrlShapeUsability:
    def test_revenue_without_any_earnings_line_is_declined(self):
        """A shape this thin would blank the caller's yfinance statements."""
        c = _client({'us-gaap': {
            'Revenues': [_flow(2024, 1000.0)],
            'Assets': [_pit(2024, 5000.0)],
        }})
        assert c.build_yfinance_shape('TEST') is None

    def test_revenue_with_operating_income_is_accepted(self):
        c = _client({'us-gaap': {
            'Revenues': [_flow(2024, 1000.0)],
            'OperatingIncomeLoss': [_flow(2024, 200.0)],
            'Assets': [_pit(2024, 5000.0)],
        }})
        assert c.build_yfinance_shape('TEST') is not None


class TestReportingCurrency:
    """One currency per filer, chosen by majority over the primary concepts.

    Asking concept by concept is not enough: _detect_currency returns USD
    whenever a concept carries any USD units, so the ~30 China-, Japan- and
    India-domiciled 20-F filers that tag a USD convenience translation beside
    their real statements (BABA, JD, BIDU, NTES, ZTO, QFIN, HTHT, VIPS, IX,
    UMC, HDB, ASX ...) read as USD-reporting. Their frames then mixed units —
    revenue in USD, net income in CNY — which no single FX rate can fix.
    Measured against yfinance's financialCurrency over 255 cached filers, the
    majority agrees 244 times and the per-concept detector 212.
    """

    def test_single_currency_filer(self):
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='EUR')
        assert c.reporting_currency(c._cache['TEST']) == 'EUR'

    def test_convenience_translation_does_not_win(self):
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='CNY')
        # A USD translation of revenue only, as such a filer tags it.
        c._cache['TEST']['facts']['us-gaap']['Revenues']['units']['USD'] = [
            _flow(2023, 140.0), _flow(2024, 168.0)]
        assert c.reporting_currency(c._cache['TEST']) == 'CNY'

    def test_shape_reads_every_row_in_the_majority_currency(self):
        """The row tagged in both currencies must be read in the filer's own
        and converted, not passed through as though it were already USD."""
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='CNY')
        c._cache['TEST']['facts']['us-gaap']['Revenues']['units']['USD'] = [
            _flow(2023, 140.0), _flow(2024, 168.0)]
        shape = c.build_yfinance_shape('TEST')
        assert shape['reporting_currency'] == 'CNY'
        inc = shape['income_statement']
        latest = inc.columns[0]
        assert inc.loc['Total Revenue', latest] == pytest.approx(1200.0 * 1.25)
        assert inc.loc['Operating Income', latest] == pytest.approx(200.0 * 1.25)

    def test_ties_resolve_to_usd_deterministically(self):
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='EUR')
        for tag in c._cache['TEST']['facts']['us-gaap'].values():
            tag['units']['USD'] = list(next(iter(tag['units'].values())))
        assert c.reporting_currency(c._cache['TEST']) == 'USD'

    def test_no_currency_units_defaults_to_usd(self):
        c = _client({'us-gaap': _US_GAAP_STATEMENT})
        assert c.reporting_currency({'facts': {}}) == 'USD'
        assert c.reporting_currency(None) == 'USD'


class TestMissingFxRates:
    def test_no_rates_declines_the_shape(self, monkeypatch):
        """_apply_fx_annual passes values through unchanged on an empty rate
        table, so converting anyway would publish EUR magnitudes as dollars
        with nothing to flag it. yfinance's statements at least carry a
        financialCurrency the FX layer can act on."""
        monkeypatch.setattr('data.sec_xbrl_client._get_fx_rates_to_usd',
                            lambda ccy: {})
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='EUR')
        assert c.build_yfinance_shape('TEST') is None

    def test_a_usd_filer_never_asks_for_rates(self, monkeypatch):
        monkeypatch.setattr(
            'data.sec_xbrl_client._get_fx_rates_to_usd',
            lambda ccy: pytest.fail('USD filer must not fetch FX rates'))
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='USD')
        assert c.build_yfinance_shape('TEST') is not None


class TestHistoryUsesTheSameCurrency:
    """fetch_historical_financials must read what it converts.

    It picked the first non-USD currency among per-concept votes, so for a
    dual-tagged filer it read revenue in USD, net income in CNY, decided the
    filer was CNY, and then scaled the USD revenue by the CNY rate too.
    """

    def test_dual_tagged_filer_reads_and_converts_in_one_currency(self):
        c = _client({'us-gaap': _US_GAAP_STATEMENT}, currency='CNY')
        c._cache['TEST']['facts']['us-gaap']['Revenues']['units']['USD'] = [
            _flow(2023, 140.0), _flow(2024, 168.0)]
        h = c.fetch_historical_financials('TEST')
        assert h['reporting_currency'] == 'CNY'
        assert h['fx_converted'] is True
        assert h['revenue_history'][2024] == pytest.approx(1200.0 * 1.25)
        assert h['earnings_history'][2024] == pytest.approx(120.0 * 1.25)


def pd_notna(v):
    import pandas as pd
    return pd.notna(v)
