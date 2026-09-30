"""NewsClient fetch/parse behaviour — offline, with urlopen stubbed."""
import urllib.error
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

from data.news_client import NewsClient


def _rss(titles, age_days=1):
    when = format_datetime(datetime.now(timezone.utc) - timedelta(days=age_days))
    items = ''.join(
        f'<item><title>{t}</title><link>https://x/{i}</link>'
        f'<pubDate>{when}</pubDate><source>Wire</source></item>'
        for i, t in enumerate(titles)
    )
    return f'<?xml version="1.0"?><rss><channel>{items}</channel></rss>'.encode()


class _Resp:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def client():
    # delay 0 so the tests are instant; the throttle itself is tested via
    # its own penalize/relax API below.
    return NewsClient(request_delay=0, delay_max=2.0, breaker_after=3)


def test_parses_rss_fields(client):
    with mock.patch('urllib.request.urlopen', return_value=_Resp(_rss(['Apple beats']))):
        out = client.fetch_company_news('AAPL', 'Apple Inc.')
    assert len(out) == 1
    assert out[0]['title'] == 'Apple beats'
    assert out[0]['origin'] == 'google_news_ticker'
    assert out[0]['source'] == 'Wire'
    assert out[0]['timestamp'] > 0


def test_drops_items_older_than_max_age(client):
    client._max_age_days = 30
    with mock.patch('urllib.request.urlopen', return_value=_Resp(_rss(['Old'], age_days=90))):
        assert client.fetch_company_news('AAPL', 'Apple Inc.') == []


def test_respects_max_items(client):
    body = _rss([f'Apple story {i}' for i in range(20)])
    with mock.patch('urllib.request.urlopen', return_value=_Resp(body)):
        assert len(client.fetch_company_news('AAPL', 'Apple Inc.', max_items=5)) == 5


def test_company_query_is_cached_on_ticker(client):
    """The prefetch pool and the main loop must not both pay for the query."""
    with mock.patch('urllib.request.urlopen',
                    return_value=_Resp(_rss(['Apple beats']))) as u:
        client.fetch_company_news('AAPL', 'Apple Inc.')
        # Main loop calls again, possibly with a differently-derived name.
        client.fetch_company_news('AAPL', 'Apple Incorporated')
    assert u.call_count == 1


def test_unidentifiable_company_makes_no_request(client):
    with mock.patch('urllib.request.urlopen') as u:
        assert client.fetch_company_news('', '') == []
    u.assert_not_called()


def test_rate_limit_widens_the_interval():
    # Back-off is multiplicative, so it needs a non-zero base to grow from —
    # NEWS_REQUEST_DELAY=0 deliberately means "no throttle, no back-off",
    # the same semantics the yfinance interval has.
    client = NewsClient(request_delay=0.01, delay_max=2.0)
    err = urllib.error.HTTPError('u', 429, 'Too Many Requests', {}, None)
    before = client._throttle.delay
    with mock.patch('urllib.request.urlopen', side_effect=err):
        assert client.fetch_company_news('AAPL', 'Apple Inc.') == []
    assert client._throttle.delay > before
    assert client._throttle.penalties == 1


def test_breaker_stops_company_queries_after_repeated_429s(client):
    err = urllib.error.HTTPError('u', 429, 'Too Many Requests', {}, None)
    with mock.patch('urllib.request.urlopen', side_effect=err) as u:
        for i in range(10):
            client.fetch_company_news(f'T{i}', f'Company {i} Inc.')
        assert client.company_rss_enabled is False
        # breaker_after=3, so requests stop at 3 despite 10 tickers asked for.
        assert u.call_count == 3


def test_breaker_leaves_sector_news_working(client):
    err = urllib.error.HTTPError('u', 429, 'Too Many Requests', {}, None)
    with mock.patch('urllib.request.urlopen', side_effect=err):
        for i in range(5):
            client.fetch_company_news(f'T{i}', f'Company {i} Inc.')
    assert client.company_rss_enabled is False
    with mock.patch('urllib.request.urlopen', return_value=_Resp(_rss(['Tech rallies']))):
        assert len(client.fetch_sector_news('Technology')) == 1


def test_healthy_response_relaxes_a_penalised_interval():
    client = NewsClient(request_delay=0.01, delay_max=2.0)
    client._throttle.penalize(2.0, cap=2.0)
    penalised = client._throttle.delay
    with mock.patch('urllib.request.urlopen', return_value=_Resp(_rss(['Apple beats']))):
        client.fetch_company_news('AAPL', 'Apple Inc.')
    assert client._throttle.delay < penalised


def test_malformed_xml_returns_empty_not_raises(client):
    with mock.patch('urllib.request.urlopen', return_value=_Resp(b'not xml')):
        assert client.fetch_company_news('AAPL', 'Apple Inc.') == []


def test_combined_news_puts_company_first_and_caps_sector(client):
    company = _rss([f'Apple story {i}' for i in range(4)])
    sector = _rss([f'Technology stocks move {i}' for i in range(8)])
    with mock.patch('urllib.request.urlopen', side_effect=[_Resp(company), _Resp(sector)]):
        with mock.patch.object(client, 'fetch_ticker_news', return_value=[]):
            out = client.get_combined_news('AAPL', 'Technology',
                                           company_name='Apple Inc.', max_sector=3)
    assert [h['scope'] for h in out] == ['company'] * 4 + ['sector'] * 3
    assert all('tags' in h and 'tier' in h for h in out)
