"""Phase-1 beta history served from the local price parquets.

run.sh downloads those closes immediately before the analysis, then Phase 1
used to re-fetch the same 5y window over a 1s-throttled yfinance call. These
cover the freshness gate that decides when the local copy may stand in, and
that a stale/short/missing file still falls through to the network.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.analyze_stock as A  # noqa: E402
from scripts.analyze_stock import (_fresh_local_prices,  # noqa: E402
                                   select_cost_of_equity)

TODAY = date(2026, 9, 15)


def _write_parquet(tmp_path, ticker, last_day=TODAY, days=900, close=100.0):
    """A daily Close series ending on *last_day* (business days only)."""
    idx = pd.bdate_range(end=pd.Timestamp(last_day), periods=days)
    df = pd.DataFrame({'Close': [close + i * 0.01 for i in range(len(idx))]},
                      index=idx)
    df.to_parquet(tmp_path / f"{ticker}.parquet")
    return df


# --- the freshness gate -----------------------------------------------------

def test_fresh_parquet_is_used(tmp_path):
    _write_parquet(tmp_path, 'AAA')
    s = _fresh_local_prices('AAA', str(tmp_path), TODAY)
    assert s is not None and len(s) > 60


def test_stale_parquet_is_rejected(tmp_path):
    _write_parquet(tmp_path, 'AAA', last_day=TODAY - timedelta(days=30))
    assert _fresh_local_prices('AAA', str(tmp_path), TODAY) is None


def test_a_long_weekend_is_still_fresh(tmp_path):
    _write_parquet(tmp_path, 'AAA', last_day=TODAY - timedelta(days=4))
    assert _fresh_local_prices('AAA', str(tmp_path), TODAY) is not None


def test_future_dated_parquet_is_rejected(tmp_path):
    """A --run-date re-run of a past session must not use tomorrow's bars."""
    _write_parquet(tmp_path, 'AAA', last_day=TODAY)
    assert _fresh_local_prices('AAA', str(tmp_path), TODAY - timedelta(days=10)) is None


def test_missing_short_and_unset_all_fall_through(tmp_path):
    assert _fresh_local_prices('NOPE', str(tmp_path), TODAY) is None
    _write_parquet(tmp_path, 'SHORT', days=30)
    assert _fresh_local_prices('SHORT', str(tmp_path), TODAY) is None
    assert _fresh_local_prices('AAA', None, TODAY) is None
    assert _fresh_local_prices('AAA', str(tmp_path), None) is None


def test_window_is_sliced_to_five_years(tmp_path):
    """The parquets hold full history; yfinance period='5y' does not.

    Feeding the whole file would move the headline beta, which is
    stock_ret[-260:], for names whose 5y window is slightly short.
    """
    _write_parquet(tmp_path, 'AAA', days=4000)      # ~16 years
    s = _fresh_local_prices('AAA', str(tmp_path), TODAY)
    span_days = (s.index[-1] - s.index[0]).days
    assert 5 * 365 - 10 <= span_days <= 5 * 365 + 10
    assert 1200 < len(s) < 1350                      # ~5y of business days


# --- select_cost_of_equity wiring -------------------------------------------

class _NoNetworkClient:
    """Any fetch is a test failure: the local path must not touch the wire."""

    def __init__(self):
        self.calls = []

    def fetch_history(self, ticker, period="5y"):
        self.calls.append(ticker)
        raise AssertionError(f"unexpected network fetch for {ticker}")


class _RecordingClient:
    def __init__(self, series):
        self.series = series
        self.calls = []

    def fetch_history(self, ticker, period="5y"):
        self.calls.append(ticker)
        return self.series


def _financials():
    return {'info': {'beta': 1.0, 'sector': 'Technology'}}


def test_local_prices_skip_the_network(tmp_path):
    stock = _write_parquet(tmp_path, 'AAA')['Close']
    market = _write_parquet(tmp_path, 'SPY', close=400.0)['Close']
    client = _NoNetworkClient()
    re, method, diag = select_cost_of_equity(
        _financials(), 0.04, client, 'AAA',
        local_prices=stock, local_market_prices=market)
    assert client.calls == []
    assert re is not None and method


def test_one_sided_local_series_falls_back_to_the_network(tmp_path):
    """Never regress a local stock series against a fetched market series."""
    stock = _write_parquet(tmp_path, 'AAA')['Close']
    client = _RecordingClient(stock)
    select_cost_of_equity(_financials(), 0.04, client, 'AAA',
                          local_prices=stock, local_market_prices=None)
    assert client.calls == ['AAA', 'SPY']


def test_no_local_series_uses_the_network(tmp_path):
    stock = _write_parquet(tmp_path, 'AAA')['Close']
    client = _RecordingClient(stock)
    select_cost_of_equity(_financials(), 0.04, client, 'AAA')
    assert client.calls == ['AAA', 'SPY']


def test_local_and_network_betas_agree(tmp_path):
    """Same bars in, same beta out — the local path is a source change only.

    (Against live data the two windows differ by one weekly observation and
    the shrunk beta moves by <0.001; measured on 10 large caps.)
    """
    stock = _write_parquet(tmp_path, 'AAA', days=1300)['Close']
    market = _write_parquet(tmp_path, 'SPY', days=1300, close=400.0)['Close']

    local = select_cost_of_equity(_financials(), 0.04, _NoNetworkClient(), 'AAA',
                                  local_prices=stock, local_market_prices=market)

    class _Both:
        calls = []

        def fetch_history(self, ticker, period="5y"):
            return market if ticker == 'SPY' else stock

    network = select_cost_of_equity(_financials(), 0.04, _Both(), 'AAA')
    assert local[0] == pytest.approx(network[0])
    assert local[1] == network[1]


# --- --prices-dir must reach the client ------------------------------------

class _StubSECLegal:
    """_run_build_clients loads the real SEC CIK map; these tests are about
    argument threading, so stub the network out and keep them offline."""

    def __init__(self, *a, **k):
        self._cik_map = {}
        self._name_map = {}

    def _load_cik_map(self):
        pass


class _StubSECXBRL:
    def __init__(self, *a, **k):
        pass

    def refresh_stale_facts(self):
        return {}


@pytest.fixture
def _offline_clients(monkeypatch):
    monkeypatch.setattr(A, 'SECLegalClient', _StubSECLegal)
    monkeypatch.setattr(A, 'SECXBRLClient', _StubSECXBRL)


def test_build_clients_threads_prices_dir_to_the_client(tmp_path, _offline_clients):
    """The client's write-through dir has its own default, so a run pointed
    elsewhere used to read from --prices-dir but write stubs to
    output/prices — seeding files in a tree nothing was reading."""
    clients = A._run_build_clients(date(2026, 9, 22), yf_delay=0,
                                   prices_dir=str(tmp_path / 'custom'))
    assert clients['yf_client']._prices_dir == str(tmp_path / 'custom')


def test_build_clients_keeps_the_default_when_unset(_offline_clients):
    clients = A._run_build_clients(date(2026, 9, 22), yf_delay=0)
    assert clients['yf_client']._prices_dir == 'output/prices'


def test_write_through_lands_in_the_configured_dir(tmp_path):
    """End of the same path: the stub is written where the run was told."""
    from data.yfinance_client import YFinanceClient
    import pandas as pd
    d = tmp_path / 'custom'
    client = YFinanceClient(request_delay=0, prices_dir=str(d))
    idx = pd.bdate_range(end=pd.Timestamp('2026-09-22'), periods=80)
    client._maybe_persist_prices('AAA', pd.DataFrame({'Close': range(80)}, index=idx))
    assert (d / 'AAA.parquet').exists()
