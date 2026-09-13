# tests/test_yfinance_evict_ticker.py
"""YFinanceClient.evict_ticker: Phase 1 of analyze_stock drops a
non-qualifying ticker's financials and every history/dividend series the
moment its screen is over, instead of holding ~9k tickers' worth until the
end of the sweep.  Other tickers' entries must be untouched."""

import pandas as pd

from data.yfinance_client import YFinanceClient


def _client_with_two_tickers():
    client = YFinanceClient(request_delay=0)
    client._financials_cache['DROP'] = {'info': {'marketCap': 1}}
    client._financials_cache['KEEP'] = {'info': {'marketCap': 2}}
    client._history_cache[('DROP', '5y')] = pd.Series([1.0])
    client._history_cache[('DROP', '5y', 'dividends')] = pd.Series([0.1])
    client._history_cache[('DROP', 'max')] = pd.Series([1.0, 2.0])
    client._history_cache[('KEEP', '5y')] = pd.Series([3.0])
    return client


def test_evict_ticker_drops_financials_and_every_history_key():
    client = _client_with_two_tickers()
    client.evict_ticker('DROP')
    assert 'DROP' not in client._financials_cache
    assert not [k for k in client._history_cache if k[0] == 'DROP']


def test_evict_ticker_leaves_other_tickers_alone():
    client = _client_with_two_tickers()
    client.evict_ticker('DROP')
    assert client._financials_cache['KEEP'] == {'info': {'marketCap': 2}}
    assert list(client._history_cache) == [('KEEP', '5y')]


def test_evict_ticker_unknown_ticker_is_a_noop():
    client = _client_with_two_tickers()
    before = (dict(client._financials_cache), dict(client._history_cache))
    client.evict_ticker('NEVER_SEEN')
    assert (client._financials_cache, client._history_cache) == before
