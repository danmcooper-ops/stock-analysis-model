# tests/test_treasury_rate.py
"""The risk-free rate's fallback chain: ^TNX -> FRED DGS10 -> the prior
snapshot's measured rate -> hardcoded 4%.

On 2026-09-30 only the first and last rungs existed. Yahoo rate-limited the
host before the first ticker, so every discount rate in a seven-hour run was
built on 4.00% while the 10-year was at 5.29% — with FRED_API_KEY set and
the prior day's measured 5.25% sitting in output/."""
import json
import logging
from datetime import date

import pytest

from data import treasury_rate as tr

RUN = date(2026, 9, 30)


@pytest.fixture(autouse=True)
def _fresh_module(monkeypatch):
    monkeypatch.setattr(tr, '_cached_rate', None)
    monkeypatch.setattr(tr, 'last_rate_source', None)
    monkeypatch.setattr(tr, 'last_rate_detail', None)


def _yahoo(monkeypatch, price):
    """^TNX answering *price* (a percent), or raising when it is an Exception."""
    class _T:
        def __init__(self, symbol, session=None):
            pass

        @property
        def info(self):
            if isinstance(price, Exception):
                raise price
            return {'regularMarketPrice': price}
    monkeypatch.setattr(tr.yf, 'Ticker', _T)


def _fred(monkeypatch, obs):
    """FREDClient whose DGS10 series is *obs* ({date: percent}) or raises."""
    import data.fred_client as fc

    class _C:
        def __init__(self, *a, **k):
            pass

        def fetch_series(self, series_id, start=None, end=None, force=False):
            assert series_id == 'DGS10'
            if isinstance(obs, Exception):
                raise obs
            return obs
        _as_of_value = staticmethod(fc.FREDClient._as_of_value)
    monkeypatch.setattr(fc, 'FREDClient', _C)


def _prior(tmp_path, day, rate, source):
    (tmp_path / f'results_{day}.json').write_text(json.dumps(
        {'date': day, 'risk_free_rate': rate, 'risk_free_rate_source': source,
         'count': 0, 'results': []}), encoding='utf-8')
    return str(tmp_path)


def test_live_tnx_wins(monkeypatch):
    _yahoo(monkeypatch, 5.29)
    _fred(monkeypatch, {RUN: 5.30})
    assert tr.fetch_risk_free_rate(run_date=RUN) == pytest.approx(0.0529)
    assert tr.last_rate_source == 'live' and tr.last_rate_detail is None


def test_fred_stands_in_when_yahoo_is_rate_limited(monkeypatch, caplog):
    _yahoo(monkeypatch, RuntimeError('Too Many Requests. Rate limited.'))
    _fred(monkeypatch, {date(2026, 9, 26): 5.21, date(2026, 9, 29): 5.25})
    with caplog.at_level(logging.WARNING):
        rate = tr.fetch_risk_free_rate(run_date=RUN)
    assert rate == pytest.approx(0.0525)                   # the newest on/before the run
    assert (tr.last_rate_source, tr.last_rate_detail) == ('fred', '2026-09-29')
    assert 'using the fred 10-year (2026-09-29) of 5.25%' in caplog.text


def test_fred_ignores_a_stale_observation(monkeypatch, tmp_path):
    _yahoo(monkeypatch, None)                              # empty .info
    _fred(monkeypatch, {date(2026, 8, 1): 5.0})            # 60 days old
    results = _prior(tmp_path, '2026-09-29', 0.0525, 'live')
    assert tr.fetch_risk_free_rate(run_date=RUN, results_dir=results) == pytest.approx(0.0525)
    assert tr.last_rate_source == 'prior_snapshot'


def test_prior_snapshot_is_borrowed_only_when_it_measured_its_rate(monkeypatch, tmp_path):
    _yahoo(monkeypatch, ConnectionError('down'))
    _fred(monkeypatch, {})
    # The 2026-09-30 snapshot itself: a fabricated 4.00% must not propagate.
    results = _prior(tmp_path, '2026-09-29', 0.04, 'fallback')
    assert tr.fetch_risk_free_rate(run_date=RUN, results_dir=results) == 0.04
    assert tr.last_rate_source == 'fallback'


def test_prior_snapshot_too_old_is_not_borrowed(monkeypatch, tmp_path):
    _yahoo(monkeypatch, ConnectionError('down'))
    _fred(monkeypatch, ConnectionError('down'))
    results = _prior(tmp_path, '2026-09-10', 0.0525, 'live')   # 20 days
    assert tr.fetch_risk_free_rate(run_date=RUN, results_dir=results) == 0.04
    assert tr.last_rate_source == 'fallback'


def test_prior_snapshot_is_the_one_before_the_run_date(monkeypatch, tmp_path):
    _yahoo(monkeypatch, 50.0)                              # implausible: rejected
    _fred(monkeypatch, {})
    _prior(tmp_path, '2026-09-29', 0.0525, 'live')
    results = _prior(tmp_path, '2026-09-30', 0.04, 'fallback')   # today's own re-run
    assert tr.fetch_risk_free_rate(run_date=RUN, results_dir=results) == pytest.approx(0.0525)
    assert (tr.last_rate_source, tr.last_rate_detail) == ('prior_snapshot', '2026-09-29')


def test_every_source_failing_is_the_loud_hardcoded_fallback(monkeypatch, tmp_path, caplog):
    _yahoo(monkeypatch, ConnectionError('down'))
    _fred(monkeypatch, ConnectionError('down'))
    with caplog.at_level(logging.WARNING):
        assert tr.fetch_risk_free_rate(run_date=RUN, results_dir=str(tmp_path)) == 0.04
    assert tr.last_rate_source == 'fallback'
    assert 'hardcoded fallback risk-free rate of 4.00%' in caplog.text
    assert 'live source failed' in caplog.text and 'fred source failed' in caplog.text


def test_result_is_cached_until_refreshed(monkeypatch):
    _yahoo(monkeypatch, 5.0)
    assert tr.fetch_risk_free_rate() == 0.05
    _yahoo(monkeypatch, 6.0)
    assert tr.fetch_risk_free_rate() == 0.05
    assert tr.fetch_risk_free_rate(refresh=True) == 0.06
