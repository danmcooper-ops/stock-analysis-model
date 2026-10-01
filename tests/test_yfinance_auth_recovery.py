# tests/test_yfinance_auth_recovery.py
"""Recovery from a poisoned yfinance crumb (2026-09-25/26 nights).

yfinance 1.7.0 caches the getcrumb response text before validating it, so a
rate-limited getcrumb leaves "Too Many Requests" as the crumb and every
`.info` call answers HTTP 401, which yfinance logs and swallows. ~60% of the
universe then went through the screen with no price, sector or EV.
YFinanceClient must see the 401, clear the crumb, pause and retry on a fresh
Ticker — and never end up with less data than before the check existed."""

import logging
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from data import yf_session
from data import yfinance_client as yc
from data.yf_session import auth_errors_this_thread, crumb_is_valid, reset_crumb
from data.yfinance_client import YahooAuthError, YFinanceClient

_STMT = pd.DataFrame({'2025-12-31': [1.0]}, index=['Total Revenue'])
_GOOD_INFO = {'symbol': 'TEST', 'shortName': 'Test Co', 'currentPrice': 10.0,
              'marketCap': 1e9, 'sharesOutstanding': 1e8, 'sector': 'Technology',
              'currency': 'USD'}


def _log_401():
    logging.getLogger('yfinance').error(
        'HTTP Error 401: {"finance":{"result":null,"error":{"code":"Unauthorized",'
        '"description":"Invalid Crumb"}}}')


def _fake_ticker_factory(fail_attempts):
    """yf.Ticker stand-in whose first `fail_attempts` instances 401 on .info."""
    made = []

    class _FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker
            self._fail = len(made) < fail_attempts
            made.append(self)
            self.balance_sheet = _STMT
            self.financials = _STMT
            self.cashflow = _STMT
            self.growth_estimates = None
            self.earnings_history = None
            self.fast_info = SimpleNamespace(shares=1e8, market_cap=1e9, last_price=10.0)

        @property
        def info(self):
            if self._fail:
                _log_401()
                return {'trailingPegRatio': None}
            return dict(_GOOD_INFO)

    return _FakeTicker, made


def _client(**kw):
    kw.setdefault('auth_pause', 0.01)
    return YFinanceClient(request_delay=0, prices_dir=None, **kw)


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    # _retry sleeps 1s/2s between attempts; the pause logic is under test,
    # not the fixed backoff.
    monkeypatch.setattr(yc.time, 'sleep', lambda s: None)


# --- crumb helpers ----------------------------------------------------------

@pytest.mark.parametrize('crumb', ['5Yx1rgmPdFj', 'a/b.c-d_E'])
def test_crumb_is_valid_accepts_real_tokens(crumb):
    assert crumb_is_valid(crumb)


@pytest.mark.parametrize('crumb', ['Too Many Requests\r\n', '<html><body>err</body></html>',
                                   '', '   ', None, 'x' * 65])
def test_crumb_is_valid_rejects_error_text(crumb):
    assert not crumb_is_valid(crumb)


def test_reset_crumb_clears_only_a_poisoned_crumb():
    from yfinance.data import YfData
    yd = YfData()
    saved = yd._crumb
    try:
        yd._crumb = 'Too Many Requests\r\n'
        assert reset_crumb() is True
        assert yd._crumb is None
        yd._crumb = '5Yx1rgmPdFj'
        assert reset_crumb() is False
        assert yd._crumb == '5Yx1rgmPdFj'
    finally:
        yd._crumb = saved


def test_auth_error_count_is_per_thread():
    before = auth_errors_this_thread()
    _log_401()
    assert auth_errors_this_thread() == before + 1

    other = {}

    def _worker():
        other['start'] = auth_errors_this_thread()
        _log_401()
        _log_401()
        other['end'] = auth_errors_this_thread()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert other['end'] - other['start'] == 2
    assert auth_errors_this_thread() == before + 1


def test_auth_counter_ignores_other_errors():
    before = auth_errors_this_thread()
    logging.getLogger('yfinance').error('HTTP Error 404: Not Found')
    assert auth_errors_this_thread() == before
    assert yf_session._AUTH_COUNTER in logging.getLogger('yfinance').handlers


# --- fetch_financials -------------------------------------------------------

@pytest.mark.parametrize('fetch_timeout', [None, 20])
def test_401_is_retried_on_a_fresh_ticker_and_recovers(monkeypatch, fetch_timeout):
    fake, made = _fake_ticker_factory(fail_attempts=1)
    monkeypatch.setattr(yc.yf, 'Ticker', fake)
    client = _client(fetch_timeout=fetch_timeout)

    out = client.fetch_financials('TEST')

    assert out['info']['currentPrice'] == 10.0
    assert '_info_auth_failed' not in out
    assert len(made) == 2  # a new Ticker per attempt, not the cached failure
    assert client.stats['auth_failures'] == 1
    assert client.stats['empty_attempts'] == 1  # the Phase-1 valve sees it
    assert client.stats['retries'] == 1
    assert client.stats['errors'] == 0


def test_exhausted_retries_keep_the_statements(monkeypatch):
    fake, made = _fake_ticker_factory(fail_attempts=99)
    monkeypatch.setattr(yc.yf, 'Ticker', fake)
    client = _client(fetch_timeout=None)

    out = client.fetch_financials('TEST')

    # Never less than before the check: statements survive, quote fields don't.
    assert out['_info_auth_failed'] is True
    assert not out['income_statement'].empty
    assert out['info'].get('currentPrice') is None
    assert len(made) == 3
    assert client.stats['auth_failures'] == 3
    assert client.stats['errors'] == 1
    assert client._financials_cache['TEST'] is out


def test_spent_pause_budget_stops_retrying(monkeypatch):
    fake, made = _fake_ticker_factory(fail_attempts=99)
    monkeypatch.setattr(yc.yf, 'Ticker', fake)
    client = _client(fetch_timeout=None, auth_pause_budget=0)

    out = client.fetch_financials('TEST')

    assert out['_info_auth_failed'] is True
    assert len(made) == 1  # an all-night outage costs one attempt per ticker
    assert client.stats['retries'] == 0


def test_all_empty_with_401_raises_auth_error(monkeypatch):
    class _Empty:
        def __init__(self, ticker):
            self.balance_sheet = self.financials = self.cashflow = pd.DataFrame()

        @property
        def info(self):
            _log_401()
            return {}

    monkeypatch.setattr(yc.yf, 'Ticker', _Empty)
    client = _client(fetch_timeout=None)
    with pytest.raises(YahooAuthError):
        client.fetch_financials('TEST')


# --- shared pause -----------------------------------------------------------

def test_pause_doubles_to_the_cap_and_resets_on_success(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(yc.time, 'monotonic', lambda: now[0])
    client = _client(auth_pause=10, auth_pause_max=40, auth_pause_budget=1000)

    pauses = []
    for _ in range(4):
        assert client._note_auth_failure()
        pauses.append(client._auth_pause_until - now[0])
        now[0] = client._auth_pause_until  # let the pause lapse
    assert pauses == [10, 20, 40, 40]

    client._retry(lambda: 'ok')                        # a history call ...
    assert client._auth_pause_next == 40               # ... does not reset it
    client._retry(lambda: 'ok', resets_backoff=True)   # a healthy .info does
    assert client._auth_pause_next == 10


# --- HTTP 429 circuit breaker (2026-09-30 night) -----------------------------

def _rl_client(monkeypatch, now, **kw):
    monkeypatch.setattr(yc.time, 'monotonic', lambda: now[0])
    kw.setdefault('rate_limit_pause', 60)
    kw.setdefault('rate_limit_pause_max', 240)
    kw.setdefault('rate_limit_budget', 1000)
    kw.setdefault('rate_limit_probe_interval', 100)
    return _client(**kw)


def _limited():
    raise RuntimeError('Too Many Requests. Rate limited. Try after a while.')


def test_429_pauses_double_and_only_an_info_success_resets(monkeypatch):
    now = [1000.0]
    client = _rl_client(monkeypatch, now)
    pauses = []
    for _ in range(4):
        assert client._note_rate_limit()
        pauses.append(client._auth_pause_until - now[0])
        now[0] = client._auth_pause_until
    assert pauses == [60, 120, 240, 240]
    assert client.stats['rate_limit_pauses'] == 4
    client._retry(lambda: 'ok')                        # history: no reset
    assert client._rl_pause_next == 240
    client._retry(lambda: 'ok', resets_backoff=True)   # .info: reset
    assert client._rl_pause_next == 60


def test_429s_inside_one_pause_share_it(monkeypatch):
    now = [1000.0]
    client = _rl_client(monkeypatch, now)
    client._note_rate_limit()
    until = client._auth_pause_until
    client._note_rate_limit()                          # another worker
    assert client._auth_pause_until == until
    assert client._rl_pause_next == 120                # doubled once
    assert client.stats['rate_limited'] == 2 and client.stats['rate_limit_pauses'] == 1


def test_breaker_opens_when_the_budget_is_spent_and_short_circuits(monkeypatch):
    now = [1000.0]
    client = _rl_client(monkeypatch, now, rate_limit_budget=150)
    assert client._note_rate_limit()                   # 60s, 90 left
    now[0] = client._auth_pause_until
    assert client._note_rate_limit()                   # 90s (capped), 0 left
    now[0] = client._auth_pause_until
    assert client._note_rate_limit() is False          # breaker opens
    assert client.rate_limited_out and client.stats['breaker_opened'] == 1

    # Open: every fetch fails without a request — and as the throttle's
    # exception, so Phase 1 re-queues and falls back to SEC data.
    calls = []
    with pytest.raises(yc.YahooRateLimitError, match='breaker open'):
        client._retry(lambda: calls.append(1))
    assert calls == [] and client.stats['breaker_short_circuits'] == 1
    assert client.stats['retries'] == 0                # no attempts burned


def test_breaker_lets_one_info_probe_through_and_closes_on_success(monkeypatch):
    now = [1000.0]
    client = _rl_client(monkeypatch, now, rate_limit_budget=60)
    client._note_rate_limit()
    now[0] = client._auth_pause_until
    assert client._note_rate_limit() is False
    opened_at = now[0]

    # Before the probe interval: even an .info fetch is refused.
    with pytest.raises(yc.YahooRateLimitError):
        client._retry(lambda: 'ok', resets_backoff=True)
    # A history call never probes, however long it waits.
    now[0] = opened_at + 1000
    with pytest.raises(yc.YahooRateLimitError):
        client._retry(lambda: 'ok')
    # The probe goes out, fails, and the breaker stays open ...
    with pytest.raises(yc.YahooRateLimitError):
        client._retry(_limited, resets_backoff=True)
    assert client.stats['breaker_probes'] == 1 and client.rate_limited_out
    # ... until one succeeds: closed, budget and escalation restored.
    now[0] += 1000
    assert client._retry(lambda: 'ok', resets_backoff=True) == 'ok'
    assert not client.rate_limited_out
    assert client._rl_budget == 60 and client._rl_pause_next == 60
    assert client._retry(lambda: 'ok') == 'ok'          # ordinary fetches resume


def test_probe_yahoo_classifies_the_answer(monkeypatch):
    from yfinance.exceptions import YFRateLimitError

    class _T:
        answer = None

        def __init__(self, symbol):
            pass

        @property
        def info(self):
            if isinstance(_T.answer, Exception):
                raise _T.answer
            return _T.answer

    monkeypatch.setattr(yc.yf, 'Ticker', _T)
    _T.answer = dict(_GOOD_INFO)
    assert yc.probe_yahoo(timeout=None) == 'ok'
    _T.answer = YFRateLimitError()
    assert yc.probe_yahoo(timeout=None) == 'rate_limited'
    _T.answer = {'trailingPegRatio': None}
    assert yc.probe_yahoo(timeout=None) == 'empty'
    _T.answer = ConnectionError('reset')
    with pytest.raises(ConnectionError):
        yc.probe_yahoo(timeout=None)


def test_fetch_financials_surfaces_a_429_as_an_empty_response(monkeypatch):
    """End to end: a 429 from yfinance reaches the Phase-1 caller as the
    throttle exception, is never cached, and widens the interval."""
    from yfinance.exceptions import YFRateLimitError

    class _Limited:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def balance_sheet(self):
            raise YFRateLimitError()

    monkeypatch.setattr(yc.yf, 'Ticker', _Limited)
    client = YFinanceClient(request_delay=0.1, prices_dir=None, fetch_timeout=None,
                            rate_limit_pause=0, rate_limit_budget=1e9)
    with pytest.raises(yc.EmptyYahooResponseError):
        client.fetch_financials('LIM')
    assert 'LIM' not in client._financials_cache
    assert client.stats['rate_limited'] == 3 and client._throttle.delay > 0.1


def test_failures_inside_one_pause_share_it(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(yc.time, 'monotonic', lambda: now[0])
    client = _client(auth_pause=10, auth_pause_budget=1000)

    client._note_auth_failure()
    until = client._auth_pause_until
    client._note_auth_failure()  # another worker, same pause
    assert client._auth_pause_until == until
    assert client._auth_pause_next == 20  # doubled once, not twice
    assert client.stats['auth_failures'] == 2


def test_pause_budget_is_capped_and_then_refuses(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(yc.time, 'monotonic', lambda: now[0])
    client = _client(auth_pause=10, auth_pause_max=120, auth_pause_budget=25)

    assert client._note_auth_failure()          # 10s, 15 left
    now[0] = client._auth_pause_until
    assert client._note_auth_failure()          # 15s (capped by budget), 0 left
    assert client._auth_pause_until - now[0] == 15
    now[0] = client._auth_pause_until
    assert client._note_auth_failure() is False  # budget spent
