# tests/test_download_prices_pool.py
"""The bulk price download's worker pool.

scripts/download_prices.py fetched one period="max" history per ticker in a
sequential loop with an inline sleep. On a cold cache — every night, since the
stateless cloud container starts with an empty output/prices/ — nothing
short-circuits and the universe pays the interval plus a full request each.

The pool that replaced it must change only scheduling: same parquets, same
stdout in the same order, one shared interval rather than one per worker, and
a write that cannot leave a truncated file behind (the freshness check reads a
parquet's index, so a half-written file could read as current).
"""
import os

import numpy as np
import pandas as pd
import pytest

from data.throttle import Throttle
from scripts import download_prices as dp


def _history(periods=300):
    idx = pd.bdate_range(end=pd.Timestamp('2026-08-28'), periods=periods)
    close = np.linspace(10.0, 20.0, len(idx))
    return pd.DataFrame(
        {'Open': close, 'High': close, 'Low': close, 'Close': close,
         'Volume': np.full(len(idx), 1e6)}, index=idx)


@pytest.fixture
def fake_yahoo(monkeypatch):
    """yf.Ticker stand-in: EMPTY* return no bars, BOOM* raise, rest succeed."""
    calls = []

    class _Ticker:
        def __init__(self, ticker, session=None):
            self.ticker = ticker

        def history(self, **_kw):
            calls.append(self.ticker)
            if self.ticker.startswith('BOOM'):
                raise RuntimeError('yahoo said no')
            if self.ticker.startswith('EMPTY'):
                return pd.DataFrame()
            return _history()

    monkeypatch.setattr(dp.yf, 'Ticker', _Ticker)
    monkeypatch.setattr(dp, '_yf_session', lambda: None)
    return calls


def _run_main(monkeypatch, tmp_path, tickers, workers):
    monkeypatch.setattr('sys.argv', [
        'download_prices.py', '--output-dir', str(tmp_path),
        '--delay', '0', '--price-workers', str(workers),
        '--tickers', *tickers])
    dp.main()


def test_pool_matches_sequential_output_and_files(monkeypatch, tmp_path, capsys,
                                                  fake_yahoo):
    # A mixed bag: successes, an empty response and a hard failure.
    tickers = ['AAA', 'BBB', 'EMPTY1', 'CCC', 'BOOM1', 'DDD']

    seq_dir = tmp_path / 'seq'
    par_dir = tmp_path / 'par'
    seq_dir.mkdir()
    par_dir.mkdir()

    _run_main(monkeypatch, seq_dir, tickers, workers=1)
    sequential = capsys.readouterr().out
    _run_main(monkeypatch, par_dir, tickers, workers=4)
    parallel = capsys.readouterr().out

    # Per-ticker lines are identical and in submission order — the pool does
    # network only; tallying and printing stay in the main thread.
    def _rows(out):
        return [ln for ln in out.splitlines() if ln.startswith('  [')]
    assert _rows(parallel) == _rows(sequential)
    # main() sorts the ticker list, so submission order is alphabetical.
    assert _rows(parallel) == [
        f'  [{i:>3}/6] {t:<6} {r}' for i, (t, r) in enumerate([
            ('AAA', 'ok'), ('BBB', 'ok'), ('BOOM1', 'error: yahoo said no'),
            ('CCC', 'ok'), ('DDD', 'ok'), ('EMPTY1', 'empty')], 1)]

    assert 'ok=4' in parallel and 'empty=1' in parallel and 'errors=1' in parallel
    assert (sorted(os.listdir(seq_dir)) == sorted(os.listdir(par_dir))
            == ['AAA.parquet', 'BBB.parquet', 'CCC.parquet', 'DDD.parquet'])


def test_worker_count_does_not_change_the_request_rate(monkeypatch, tmp_path,
                                                       fake_yahoo):
    # The throttle is per-process, so N workers must not mean N intervals.
    # Every ticker that actually fetches ticks it exactly once.
    ticks = []
    real_init = Throttle.__init__

    def _spy_init(self, delay):
        real_init(self, delay)
        ticks.append(self)
    monkeypatch.setattr(Throttle, '__init__', _spy_init)

    _run_main(monkeypatch, tmp_path, ['AAA', 'BBB', 'CCC', 'EMPTY1'], workers=4)
    assert len(ticks) == 1, 'the run must share exactly one throttle'
    assert ticks[0].stats()['calls'] == 4


def test_fresh_files_never_tick_the_throttle(tmp_path, fake_yahoo):
    # The short-circuits sit before the throttle: a current file costs a local
    # parquet read, not a slot in the request budget.
    _history().to_parquet(tmp_path / 'AAA.parquet')
    throttle = Throttle(0)
    assert dp.download_ticker('AAA', str(tmp_path), delay=0, refresh=False,
                              throttle=throttle) == 'skipped'
    assert throttle.stats()['calls'] == 0

    assert dp.download_ticker('NEW', str(tmp_path), delay=0, refresh=False,
                              throttle=throttle) == 'ok'
    assert throttle.stats()['calls'] == 1


def test_failed_write_leaves_no_partial_or_temp_file(monkeypatch, tmp_path,
                                                     fake_yahoo):
    def _boom(self, *a, **k):
        raise OSError('disk full')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', _boom)

    result = dp.download_ticker('AAA', str(tmp_path), delay=0)
    assert result.startswith('error:')
    # Neither a truncated AAA.parquet (which the freshness check would read as
    # current) nor an orphaned temp file.
    assert os.listdir(tmp_path) == []


class TestEmptyStreakGovernor:
    """Empty means 'soft-throttled' or 'delisted'; only the first is our fault."""

    def test_scattered_empties_do_not_widen_the_interval(self):
        throttle = Throttle(0.4)
        gov = dp._EmptyStreakGovernor(throttle)
        for _ in range(10):
            gov.record('empty')
            gov.record('ok')
        assert gov.penalties == 0
        assert throttle.delay == pytest.approx(0.4)

    def test_a_burst_widens_it(self):
        throttle = Throttle(0.4)
        gov = dp._EmptyStreakGovernor(throttle, streak=3)
        gov.record('empty')
        gov.record('empty')
        assert gov.penalties == 0
        gov.record('empty')
        assert gov.penalties == 1
        assert throttle.delay > 0.4

    def test_success_walks_a_widened_interval_back_down(self):
        throttle = Throttle(0.4)
        gov = dp._EmptyStreakGovernor(throttle, streak=1)
        gov.record('empty')
        widened = throttle.delay
        assert widened > 0.4
        for _ in range(500):
            gov.record('ok')
        # relax() floors at the configured base, never below it.
        assert throttle.delay == pytest.approx(0.4)

    def test_the_interval_is_capped(self):
        throttle = Throttle(0.4)
        gov = dp._EmptyStreakGovernor(throttle, streak=1)
        for _ in range(50):
            gov.record('empty')
        assert throttle.delay <= dp.YF_REQUEST_DELAY_MAX


def test_interrupt_cancels_the_queue_instead_of_draining_it(monkeypatch,
                                                            tmp_path, capsys):
    """Ctrl-C must not sit through every queued ticker before exiting.

    The whole universe is submitted up front, so the executor's default
    shutdown would wait out ~2,300 downloads — minutes of apparently hung
    terminal for an operator who asked it to stop.
    """
    seen = []

    class _Ticker:
        def __init__(self, ticker, session=None):
            self.ticker = ticker

        def history(self, **_kw):
            seen.append(self.ticker)
            if self.ticker == 'T002':
                raise KeyboardInterrupt
            return _history()

    monkeypatch.setattr(dp.yf, 'Ticker', _Ticker)
    monkeypatch.setattr(dp, '_yf_session', lambda: None)

    tickers = [f'T{i:03d}' for i in range(60)]
    _run_main(monkeypatch, tmp_path, tickers, workers=1)
    out = capsys.readouterr().out

    assert 'pending downloads cancelled' in out
    # One worker, cancelled at the third ticker: the remaining ~57 never ran.
    assert len(seen) < 10, f'queue was drained anyway: {len(seen)} fetched'
    # The summary still reports what actually completed.
    assert 'ok=2' in out
