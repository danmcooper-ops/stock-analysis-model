"""The 2026-09-25 sector dropout: Yahoo throttled .info while statements kept
answering, so 1,548 rows arrived with no price, sector or name.

Covers the three layers of the fix: the fetch treats an identity-less .info
as a soft throttle, Phase 1 re-queues a Yahoo failure even for a US filer,
and a ticker still empty on the retry keeps the prior snapshot's identity.
"""
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.yfinance_client as Y  # noqa: E402
import scripts.analyze_stock as A  # noqa: E402
from tests.test_phase1_prefetch import _args, _FakeSEC, _FakeYF, _Prov  # noqa: E402

FRAME = pd.DataFrame({'2025-12-31': [1.0]}, index=['Total Revenue'])


class _Ticker:
    """yf.Ticker stand-in with real statements and a configurable .info."""
    info_by = {}

    def __init__(self, t):
        self.ticker = t
        self.balance_sheet = self.financials = self.cashflow = FRAME
        self.info = dict(self.info_by.get(t, {}))
        self.growth_estimates = self.earnings_history = None
        self.fast_info = None


@pytest.fixture
def yf_client(monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    monkeypatch.setattr(Y.yf, 'Ticker', _Ticker)
    return Y.YFinanceClient(request_delay=0, fetch_timeout=None)


class TestDetector:
    def test_statements_with_empty_info_is_a_soft_throttle(self, yf_client):
        _Ticker.info_by = {'THR': {'trailingPegRatio': None}}   # yfinance's throttled shape
        with pytest.raises(Y.EmptyYahooResponseError, match='statements but an empty .info'):
            yf_client.fetch_financials('THR')
        # Retried, counted per attempt, and never cached as a good response.
        assert yf_client.stats['empty_attempts'] == 3
        assert 'THR' not in yf_client._financials_cache

    def test_info_with_identity_passes(self, yf_client):
        _Ticker.info_by = {'OK': {'symbol': 'OK', 'marketCap': 5e9, 'sharesOutstanding': 1e8,
                                  'currentPrice': 50.0, 'sector': 'Technology'}}
        d = yf_client.fetch_financials('OK')
        assert d['info']['sector'] == 'Technology' and yf_client.stats['empty_attempts'] == 0


class TestIdentityFill:
    def test_fills_only_blank_identity_and_never_price(self):
        info = {'sector': '', 'currentPrice': None}
        got = A.fill_identity_from_prior(info, {'company_name': 'Broadcom Inc.', 'sector': 'Technology',
                                                'industry': 'Semiconductors', 'country': 'United States',
                                                'price': 344.72})
        assert sorted(got) == ['company_name', 'country', 'industry', 'sector']
        assert info['shortName'] == 'Broadcom Inc.' and info['sector'] == 'Technology'
        assert info['currentPrice'] is None                   # price is never carried

    def test_keeps_what_yahoo_did_send(self):
        info = {'longName': 'Live Name', 'sector': 'Energy'}
        assert A.fill_identity_from_prior(info, {'company_name': 'Old', 'sector': 'Tech',
                                                 'industry': 'Oil'}) == ['industry']
        assert info['sector'] == 'Energy' and 'shortName' not in info
        assert A.fill_identity_from_prior({}, None) == []


# ---------------------------------------------------------------- Phase 1

class _SEC(_FakeSEC):
    """SEC covers every ticker with (dummy) statements."""
    def build_yfinance_shape(self, ticker):
        return {'balance_sheet': FRAME, 'income_statement': FRAME, 'cash_flow': FRAME,
                'info': {}}


class _FlakyYF(_FakeYF):
    """Empty for the first *fails* fetches of each ticker in *flaky*."""
    def __init__(self, flaky, fails=1, **kw):
        super().__init__(**kw)
        self._left = {t: fails for t in flaky}

    def fetch_financials(self, ticker):
        with self._lock:
            if self._left.get(ticker, 0) > 0:
                self._left[ticker] -= 1
                self.fetched.append(ticker)
                raise Y.EmptyYahooResponseError(ticker)
        return super().fetch_financials(ticker)


class _RecProv(_Prov):
    def __init__(self):
        self.events = []

    def record_event(self, typ, ticker, source, detail=None):
        self.events.append((typ, ticker, source, detail))


@pytest.fixture
def phase1(monkeypatch, tmp_path):
    from data.screen_skip_cache import ScreenSkipCache
    monkeypatch.setattr(A, 'calculate_roic', lambda d: {'roic_median_5y': 0.15})
    monkeypatch.setattr(A, 'calculate_wacc', lambda *a, **k: 0.08)
    monkeypatch.setattr(A, 'select_cost_of_equity', lambda *a, **k: (0.09, 'capm', None))
    monkeypatch.setattr(A, '_convert_financials_to_usd', lambda d, **k: (d, {}))
    monkeypatch.setattr(A, '_fresh_local_prices', lambda *a, **k: None)
    monkeypatch.setattr(A, '_drop_stopped_carry_forwards', lambda *a, **k: set())
    prior = [{'ticker': 'AVGO', 'shares_out': 1, 'mcap': 1e12, 'company_name': 'Broadcom Inc.',
              'sector': 'Technology', 'industry': 'Semiconductors', 'country': 'United States'}]
    monkeypatch.setattr(A, 'prior_snapshot_file', lambda *a, **k: ('2026-09-24', 'x'))
    monkeypatch.setattr(A, '_load_carry_forward_rows', lambda d, p: prior)
    cache = ScreenSkipCache(path=str(tmp_path / 's.json'), today=None)
    monkeypatch.setattr(A, 'ScreenSkipCache', lambda **k: cache)

    def run(yf, tickers, sec=None, workers=1):
        prov = _RecProv()
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = A._run_phase1_screen(_args(no_screen_cache=False), prov, list(tickers),
                                       {t: 'quality' for t in tickers}, yf, None,
                                       sec or _SEC(ciks=tickers), 0.04, 0.045,
                                       prices_dir=None, phase1_workers=workers)
        return out, buf.getvalue(), prov, cache
    return run


def test_us_filer_is_requeued_and_recovers(phase1):
    """The outage clears by the end of the pass: the retry gets real .info."""
    yf = _FlakyYF(flaky={'AVGO'}, fails=1)
    out, text, prov, _ = phase1(yf, ['AVGO', 'NVDA'])
    assert 'AVGO - yfinance empty (likely throttled) — re-queued for retry' in text
    assert yf.fetched.count('AVGO') == 2
    assert out['screen_cache']['AVGO']['data_source'] == 'sec_xbrl+yfinance'
    assert 'Fetch-failure retry: 1 re-queued, 1 recovered, 0 still empty on Yahoo' in text
    assert not [e for e in prov.events if e[2] == 'identity']


@pytest.mark.parametrize('workers', [1, 4])
def test_still_empty_keeps_sec_data_and_prior_identity(phase1, workers):
    yf = _FlakyYF(flaky={'AVGO'}, fails=99)
    out, text, prov, cache = phase1(yf, ['AVGO', 'NVDA'], workers=workers)
    entry = out['screen_cache']['AVGO']
    assert entry['data_source'] == 'sec_xbrl'
    info = entry['yf_data']['info']
    assert (info['sector'], info['industry'], info['shortName']) == \
        ('Technology', 'Semiconductors', 'Broadcom Inc.')
    ids = [e for e in prov.events if e[2] == 'identity']
    assert ids and ids[0][1] == 'AVGO' and 'sector' in ids[0][3]['fields']
    assert '1 still empty on Yahoo (1 kept on SEC data, 1 with the prior snapshot' in text
    assert cache.skip_reason('AVGO', 1e9) is None          # throttled is not dead


# ------------------------------------------------- HTTP 429 (2026-09-30)

class _RateLimitedYF(_FlakyYF):
    """Like _FlakyYF, but Yahoo's answer is the explicit HTTP 429."""
    def fetch_financials(self, ticker):
        with self._lock:
            if self._left.get(ticker, 0) > 0:
                self._left[ticker] -= 1
                self.fetched.append(ticker)
                raise Y.YahooRateLimitError('Too Many Requests. Rate limited.')
        return _FakeYF.fetch_financials(self, ticker)


def test_rate_limited_us_filer_is_requeued_with_the_cause(phase1):
    """The 2026-09-30 failure: 3,655 tickers raised the 429 and only the 10
    soft-throttle empties were re-queued. A 429 must take the same path."""
    yf = _RateLimitedYF(flaky={'AVGO'}, fails=1)
    out, text, prov, _ = phase1(yf, ['AVGO', 'NVDA'])
    assert 'AVGO - yfinance rate-limited (HTTP 429) — re-queued for retry' in text
    assert 'AVGO - error:' not in text
    assert yf.fetched.count('AVGO') == 2
    assert out['screen_cache']['AVGO']['data_source'] == 'sec_xbrl+yfinance'
    assert 'Rate-limited (HTTP 429): 1 ticker(s)' in text


@pytest.mark.parametrize('workers', [1, 4])
def test_still_rate_limited_us_filer_keeps_sec_data_and_identity(phase1, workers):
    yf = _RateLimitedYF(flaky={'AVGO'}, fails=99)
    out, text, prov, cache = phase1(yf, ['AVGO', 'NVDA'], workers=workers)
    entry = out['screen_cache']['AVGO']
    assert entry['data_source'] == 'sec_xbrl'
    assert entry['yf_data']['info']['shortName'] == 'Broadcom Inc.'
    assert [e for e in prov.events if e[2] == 'identity'][0][1] == 'AVGO'
    assert cache.skip_reason('AVGO', 1e9) is None          # rate-limited is not dead


def test_open_breaker_disables_the_prefetch_pool(phase1):
    yf = _FakeYF()
    yf.rate_limited_out = True
    _, text, _, _ = phase1(yf, ['AVGO', 'NVDA'], workers=4)
    assert '[!] Phase 1: yfinance rate-limit breaker is open — prefetch disabled' in text


# ------------------------------------------------- startup gate

class TestStartupGate:
    def _run(self, answers, max_wait=300, interval=60):
        seq = list(answers)
        slept = []
        now = [0.0]

        def probe():
            a = seq.pop(0)
            if isinstance(a, Exception):
                raise a
            return a

        def sleep(s):
            slept.append(s)
            now[0] += s

        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = A._yahoo_startup_gate(probe=probe, max_wait_s=max_wait, interval_s=interval,
                                       sleep=sleep, clock=lambda: now[0])
        return ok, slept, buf.getvalue()

    def test_passes_at_once_when_yahoo_answers(self):
        ok, slept, text = self._run(['ok'])
        assert ok and slept == [] and text == ''

    def test_waits_out_a_short_limit(self):
        ok, slept, text = self._run(['rate_limited', 'empty', ConnectionError('x'), 'ok'])
        assert ok and slept == [60, 60, 60]
        assert 'rate_limited at startup — waiting 60s before probe 2' in text
        assert 'Yahoo answered after 180s (4 probes)' in text

    def test_gives_up_after_the_cooldown(self):
        ok, slept, _ = self._run(['rate_limited'] * 10, max_wait=300, interval=60)
        assert ok is False and slept == [60] * 5     # probes at 0..300s, then stop

    def test_require_exits_3_when_the_gate_fails(self, monkeypatch, capsys):
        monkeypatch.delenv('YF_STARTUP_GATE', raising=False)
        monkeypatch.setattr(A, '_yahoo_startup_gate', lambda: False)
        with pytest.raises(SystemExit) as ei:
            A._require_yahoo_or_exit()
        assert ei.value.code == 3
        assert 'checkpoint is kept' in capsys.readouterr().out

    def test_require_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv('YF_STARTUP_GATE', '0')
        monkeypatch.setattr(A, '_yahoo_startup_gate', lambda: pytest.fail('probed'))
        A._require_yahoo_or_exit()


def test_throttled_non_filer_is_not_remembered_as_dead(phase1):
    yf = _FlakyYF(flaky={'FRGN'}, fails=99)
    _, text, _, cache = phase1(yf, ['FRGN', 'NVDA'], sec=_FakeSEC(ciks=['NVDA']))
    assert 'FRGN - error: yfinance empty AND no SEC XBRL coverage (retry also failed)' in text
    assert cache.skip_reason('FRGN', 1e9) is None


def test_run_quality_flags_an_info_outage(caplog):
    rows = [{'ticker': f'T{i}', 'price': None if i < 3 else 10.0, 'company_name': 'X', 'sector': 'S'}
            for i in range(10)]

    class P:
        events = [{'type': 'source_fallback', 'source': 'identity', 'ticker': 'T0'}]

    class W:
        fabricated = total = 0

    with caplog.at_level('WARNING', logger='analyze_stock'):
        A._run_quality_summary(0.04, 'fred', W(), P(), results=rows)
    text = caplog.text
    assert 'RUN QUALITY: 3 of 10 rows (30%) have no price or identity' in text
    assert "1 ticker(s) had an empty Yahoo .info even on the retry" in text
    caplog.clear()
    with caplog.at_level('WARNING', logger='analyze_stock'):
        A._run_quality_summary(0.04, 'fred', W(), None, results=rows[3:])
    assert 'no price or identity' not in caplog.text
