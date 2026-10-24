"""Phase 2 keeps tickers without a computable ROIC, except SPACs and funds."""
from scripts.analyze_stock import _phase2_non_operating


def _yf(industry):
    return {'info': {'industry': industry}}


def test_spac_is_dropped():
    assert _phase2_non_operating(_yf('Shell Companies'), 'sec_xbrl+yfinance')


def test_closed_end_fund_without_sec_statements_is_dropped():
    assert _phase2_non_operating(_yf('Asset Management'), 'yfinance')


def test_bdc_with_sec_statements_is_kept():
    # ARCC: Asset Management on Yahoo, but it files a 10-K.
    assert _phase2_non_operating(_yf('Asset Management'), 'sec_xbrl+yfinance') is None


def test_operating_companies_are_kept():
    for industry in ('Banks - Diversified', 'Residential Construction',
                     'Oil & Gas Midstream', 'Software - Infrastructure'):
        assert _phase2_non_operating(_yf(industry), 'yfinance') is None


def test_missing_info_is_kept():
    assert _phase2_non_operating({}, 'sec_xbrl') is None
    assert _phase2_non_operating(None, None) is None
