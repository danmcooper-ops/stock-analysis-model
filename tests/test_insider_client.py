# tests/test_insider_client.py
"""Form 4 discovery window and availability semantics in SECInsiderClient.

The client parses the newest max_form4_files Form 4s. An annual grant cycle
files one Form 4 per insider on the same day (PPG and GWW: 8+ award-only
filings on 2026-09-01), which used to exhaust the budget and, because a
filing with no open-market trade parsed to an empty list indistinguishable
from a failed fetch, the company read as "no insider data" and the Insider
Buying gate went N/A. Three guarantees:

  * at most _MAX_FILES_PER_DAY filings are taken from one filing date, so
    the budget reaches the rest of the year;
  * a filer whose Form 4s carry only awards/exercises is AVAILABLE with zero
    open-market counts, and the result says how many filings were in the
    window vs parsed;
  * a submissions or document outage is flagged fetch_failed and never
    cached.

No live SEC calls: the HTTP layer is monkeypatched.
"""

from datetime import datetime, timedelta

from data.sec_insider_client import SECInsiderClient


def _client():
    return SECInsiderClient(cik_map={'TEST': '0000000001'},
                            name_map={'TEST': 'Test Co'},
                            email='test@example.com', request_delay=0,
                            max_form4_files=6)


def _days_ago(n):
    return (datetime.now() - timedelta(days=n)).strftime('%Y-%m-%d')


def _submissions(dated_forms):
    """Build a submissions JSON stub from [(form, filing_date), ...],
    newest first."""
    return {'filings': {'recent': {
        'form': [f for f, _ in dated_forms],
        'filingDate': [d for _, d in dated_forms],
        'accessionNumber': [f'0000000001-26-{i:06d}' for i in range(len(dated_forms))],
        'primaryDocument': [f'xslF345X06/wk-form4_{i}.xml' for i in range(len(dated_forms))],
    }}}


def _award(date):
    return {'date': date, 'insider_name': 'A', 'title': 'CEO',
            'is_officer': True, 'is_director': False,
            'transaction_code': 'A', 'transaction_type': 'award',
            'shares': 100.0, 'price_per_share': 0.0, 'dollar_value': 0.0,
            'shares_after': 1000.0}


def _sale(date):
    return dict(_award(date), transaction_code='S', transaction_type='sell',
                price_per_share=50.0, dollar_value=5000.0)


class TestFilingWindow:
    def test_per_day_cap_spreads_budget_across_the_year(self, monkeypatch):
        grant_day = _days_ago(5)
        dated = [('4', grant_day)] * 10 + [('4', _days_ago(40)), ('10-Q', _days_ago(45)),
                                           ('4', _days_ago(120)), ('4', _days_ago(300)),
                                           ('4', _days_ago(400))]   # outside window
        c = _client()
        monkeypatch.setattr(c, '_request_json', lambda url: _submissions(dated))
        filings, total = c._find_form4_filings('0000000001', days_back=365)
        dates = [d for _, _, d in filings]
        assert dates.count(grant_day) == SECInsiderClient._MAX_FILES_PER_DAY
        assert _days_ago(40) in dates and _days_ago(120) in dates and _days_ago(300) in dates
        assert _days_ago(400) not in dates
        assert len(filings) == 6            # budget honoured
        assert total == 13                  # every Form 4 inside the window

    def test_budget_still_caps_total_files(self, monkeypatch):
        dated = [('4', _days_ago(10 * i)) for i in range(1, 12)]
        c = _client()
        monkeypatch.setattr(c, '_request_json', lambda url: _submissions(dated))
        filings, total = c._find_form4_filings('0000000001', days_back=365)
        assert len(filings) == 6
        assert total == 11

    def test_submissions_failure_is_none(self, monkeypatch):
        c = _client()
        monkeypatch.setattr(c, '_request_json', lambda url: None)
        assert c._find_form4_filings('0000000001') is None


class TestAvailability:
    def test_awards_only_is_available_with_zero_counts(self, monkeypatch):
        c = _client()
        monkeypatch.setattr(c, '_find_form4_filings',
                            lambda cik, days_back=365: ([('a', 'd1', _days_ago(3)),
                                                         ('b', 'd2', _days_ago(9))], 14))
        monkeypatch.setattr(c, '_parse_form4_xml',
                            lambda cik, acc, doc: [_award(_days_ago(3))] if acc == 'a' else [])
        r = c.fetch_insider_activity('TEST')
        assert r['available'] is True
        assert r['fetch_failed'] is False
        assert r['buy_count_365d'] == 0 and r['sell_count_365d'] == 0
        assert r['insider_buy_ratio'] is None
        assert r['form4_total_365d'] == 14
        assert r['form4_parsed'] == 2
        assert 'TEST' in c._cache

    def test_open_market_sales_counted(self, monkeypatch):
        c = _client()
        monkeypatch.setattr(c, '_find_form4_filings',
                            lambda cik, days_back=365: ([('a', 'd1', _days_ago(3))], 1))
        monkeypatch.setattr(c, '_parse_form4_xml',
                            lambda cik, acc, doc: [_sale(_days_ago(3)), _award(_days_ago(3))])
        r = c.fetch_insider_activity('TEST')
        assert r['available'] is True
        assert r['sell_count_365d'] == 1 and r['buy_count_365d'] == 0
        assert r['insider_buy_ratio'] == 0.0

    def test_submissions_outage_flagged_and_not_cached(self, monkeypatch):
        c = _client()
        monkeypatch.setattr(c, '_find_form4_filings', lambda cik, days_back=365: None)
        r = c.fetch_insider_activity('TEST')
        assert r['available'] is False
        assert r['fetch_failed'] is True
        assert r['buy_count_365d'] == 0
        assert 'TEST' not in c._cache

    def test_all_documents_unreadable_flagged_and_not_cached(self, monkeypatch):
        c = _client()
        monkeypatch.setattr(c, '_find_form4_filings',
                            lambda cik, days_back=365: ([('a', 'd1', _days_ago(3)),
                                                         ('b', 'd2', _days_ago(9))], 2))
        monkeypatch.setattr(c, '_parse_form4_xml', lambda cik, acc, doc: None)
        r = c.fetch_insider_activity('TEST')
        assert r['available'] is False
        assert r['fetch_failed'] is True
        assert r['form4_total_365d'] == 2
        assert 'TEST' not in c._cache

    def test_no_form4s_in_window_is_unavailable_but_not_failed(self, monkeypatch):
        c = _client()
        monkeypatch.setattr(c, '_find_form4_filings', lambda cik, days_back=365: ([], 0))
        r = c.fetch_insider_activity('TEST')
        assert r['available'] is False
        assert r['fetch_failed'] is False
        assert 'TEST' in c._cache

    def test_unknown_ticker_is_unavailable(self):
        c = _client()
        r = c.fetch_insider_activity('NOPE')
        assert r['available'] is False and r['fetch_failed'] is False
