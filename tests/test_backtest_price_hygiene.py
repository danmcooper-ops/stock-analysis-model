# tests/test_backtest_price_hygiene.py
"""Bar rules that keep bad prices out of the frozen return sidecars.

The 2026-10-16 adversarial review found four faults that a sidecar froze for
good: a trailing NaN-close row on SPY read as a settled eval date, VSCO was
measured to a bar two days short of its eval date, GRKZF's one-day $0.001
print became a -99.99% return, and forward-filled dead quotes entered as 0%.
SIDECAR_METHOD 4 applies data.price_store.select_window's rules and recomputes
every older settled sidecar once.
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import scripts.backtest as bt
import scripts.download_prices as dp

RUN = '2026-07-08'


def _px(prices, ticker, start, end, closes=None, drift=0.001, base=100.0):
    idx = pd.date_range(start, end, freq='B', name='Date')
    vals = closes if closes is not None else base * (1 + drift) ** np.arange(len(idx))
    pd.DataFrame({'Close': vals}, index=idx).to_parquet(prices / f'{ticker}.parquet')
    return idx


@pytest.fixture
def prices(tmp_path):
    p = tmp_path / 'prices'
    p.mkdir()
    bt._BAR_CACHE.clear()
    bt._MANIFEST_CACHE.clear()
    return p


def _snap(*tickers):
    return {'date': RUN, 'results': [{'ticker': t, 'price': 100.0} for t in tickers]}


def test_trailing_nan_benchmark_row_defers_instead_of_unmeasuring(prices, tmp_path):
    """SPY's 2026-10-06 row had no close: the pair is deferred, not reported
    as 'no benchmark return'."""
    idx = pd.date_range('2026-07-01', '2026-10-06', freq='B')
    closes = list(740 + np.arange(len(idx)) * 0.3)
    closes[-1] = float('nan')
    _px(prices, 'SPY', '2026-07-01', '2026-10-06', closes=closes)
    _px(prices, 'AAA', '2026-07-01', '2026-10-06')
    snap = _snap('AAA')
    out = bt.annotate_snapshot_returns(snap, [90], None, prices_dir=str(prices),
                                       cache_dir=str(tmp_path / 'r'),
                                       today=date(2026, 10, 7))
    assert out == {90: None}
    stats = snap['_fwd_stats'][90]
    assert stats.get('unsettled') is True and 'no_benchmark' not in stats
    assert stats['bench_last_bar'] == '2026-10-05'


def test_a_series_short_of_the_eval_date_is_pending_then_measured(prices, tmp_path):
    """VSCO: Tiingo's series ended 10-02 against a 10-04 eval date and was
    frozen at the 10-02 close. Now it waits, and the sidecar stays open."""
    cache = tmp_path / 'r'
    _px(prices, 'SPY', '2026-07-01', '2026-09-30')
    _px(prices, 'AAA', '2026-07-01', '2026-09-30')
    _px(prices, 'SHORT', '2026-07-01', '2026-08-05', drift=0.002)   # eval 08-07
    snap = _snap('AAA', 'SHORT')
    bt.annotate_snapshot_returns(snap, [30], None, prices_dir=str(prices),
                                 cache_dir=str(cache), today=date(2026, 9, 30))
    body = json.loads((cache / f'{RUN}_h30.json').read_text(encoding='utf-8'))
    assert 'SHORT' not in body['tickers'] and body['pending'] == ['SHORT']
    assert not bt.sidecar_is_complete(body, ['AAA', 'SHORT'])

    _px(prices, 'SHORT', '2026-07-01', '2026-09-30', drift=0.002)  # refreshed
    bt._BAR_CACHE.clear()
    snap = _snap('AAA', 'SHORT')
    bt.annotate_snapshot_returns(snap, [30], None, prices_dir=str(prices),
                                 cache_dir=str(cache), today=date(2026, 10, 7))
    body = json.loads((cache / f'{RUN}_h30.json').read_text(encoding='utf-8'))
    assert 'SHORT' in body['tickers'] and body['pending'] == []
    assert bt.sidecar_is_complete(body, ['AAA', 'SHORT'])


def test_a_file_that_ended_before_the_snapshot_is_not_pending(prices, tmp_path):
    _px(prices, 'SPY', '2026-07-01', '2026-09-30')
    _px(prices, 'AAA', '2026-07-01', '2026-09-30')
    _px(prices, 'GONE', '2026-05-01', '2026-07-01')
    assert bt.pending_tickers(str(prices), ['GONE', 'NOFILE'], RUN, 30) == []


def test_an_older_method_sidecar_is_recomputed_not_topped_up(prices, tmp_path):
    """A settled method-3 file froze GRKZF's -99.99%: method 4 recomputes it,
    so the spike leaves and the honest return replaces any frozen one."""
    cache = tmp_path / 'r'
    cache.mkdir()
    _px(prices, 'SPY', '2026-07-01', '2026-09-30')
    _px(prices, 'AAA', '2026-07-01', '2026-09-30')
    idx = pd.date_range('2026-07-01', '2026-09-30', freq='B')
    spike = [15.5 + 0.01 * (i % 3) for i in range(len(idx))]
    spike[list(idx).index(pd.Timestamp('2026-08-07'))] = 0.001        # eval bar
    _px(prices, 'GRKZF', '2026-07-01', '2026-09-30', closes=spike)
    frozen = {'run_date': RUN, 'horizon_days': 30, 'method': 3,
              'eval_date': '2026-08-07', 'bench_last_bar': '2026-09-30',
              'spy_return': 0.03, 'coverage': 1.0, 'n_requested': 2, 'n_priced': 2,
              'tickers': {
                  'AAA': {'ret': 0.5, 'excess_return': 0.47, 'start_price': 1.0,
                          'end_price': 1.5, 'spy_return': 0.03},
                  'GRKZF': {'ret': -0.9999, 'excess_return': -1.03,
                            'start_price': 15.5, 'end_price': 0.001,
                            'spy_return': 0.03}}}
    path = cache / f'{RUN}_h30.json'
    path.write_text(json.dumps(frozen), encoding='utf-8')
    snap = _snap('AAA', 'GRKZF')
    bt.annotate_snapshot_returns(snap, [30], None, prices_dir=str(prices),
                                 cache_dir=str(cache), today=date(2026, 9, 30))
    body = json.loads(path.read_text(encoding='utf-8'))
    assert body['method'] == bt.SIDECAR_METHOD == 4
    assert 'GRKZF' not in body['tickers']
    want = bt.fetch_forward_returns(['AAA'], RUN, 30, None, prices_dir=str(prices))
    assert body['tickers']['AAA']['ret'] == pytest.approx(want['AAA']['ret'])
    assert body['tickers']['AAA']['ret'] != pytest.approx(0.5)


def test_delisting_inside_the_last_week_gets_carry(prices):
    """A name delisted within MAX_SNAP_GAP_DAYS of the eval date no longer
    reaches it, so it goes through terminal_returns (proceeds ride SPY) rather
    than being measured to its last bar without carry."""
    _px(prices, 'SPY', '2026-07-01', '2026-09-30',
        closes=[100.0 if d <= pd.Timestamp('2026-08-04') else 110.0
                for d in pd.date_range('2026-07-01', '2026-09-30', freq='B')])
    _px(prices, 'DEAL', '2026-07-01', '2026-08-04',
        closes=[40.0 if d <= pd.Timestamp(RUN) else 60.0
                for d in pd.date_range('2026-07-01', '2026-08-04', freq='B')])
    (prices / bt.BACKFILL_MANIFEST).write_text(json.dumps({'delisted': {
        'DEAL': {'last_date': '2026-08-04', 'last_close': 60.0}}}), encoding='utf-8')
    got = bt.fetch_forward_returns(['DEAL'], RUN, 30, None, prices_dir=str(prices))
    assert got['DEAL']['ret'] == pytest.approx(60 / 40 * 1.1 - 1)
    assert got['DEAL']['delisted']['date'] == '2026-08-04'


def test_download_trims_an_unfinished_trailing_row(monkeypatch, tmp_path):
    idx = pd.date_range('2026-10-01', '2026-10-06', freq='B')
    df = pd.DataFrame({'Open': 1.0, 'High': 1.0, 'Low': 1.0,
                       'Close': [10.0, 11.0, 12.0, float('nan')],
                       'Volume': 100}, index=idx)

    class _Ticker:
        def __init__(self, *a, **k):
            pass

        def history(self, **k):
            return df.copy()

    monkeypatch.setattr(dp.yf, 'Ticker', _Ticker)
    monkeypatch.setattr(dp, '_yf_session', lambda: None)
    assert dp.download_ticker('SPY', str(tmp_path), 0) == 'ok'
    path = tmp_path / 'SPY.parquet'
    assert pd.read_parquet(path).index.max() == pd.Timestamp('2026-10-05')
    assert dp._parquet_max_date(str(path)) == date(2026, 10, 5)


def test_freshness_ignores_a_trailing_nan_row(tmp_path):
    """A kept file whose last row has no close is not current: it is the
    file that kept SPY's unfinished 10-06 row through the next run."""
    idx = pd.date_range('2026-10-01', '2026-10-06', freq='B')
    pd.DataFrame({'Close': [10.0, 11.0, 12.0, float('nan')], 'Volume': 1},
                 index=idx.rename('Date')).to_parquet(tmp_path / 'SPY.parquet')
    assert dp._parquet_max_date(str(tmp_path / 'SPY.parquet')) == date(2026, 10, 5)


def test_an_older_sidecar_kept_without_prices_keeps_its_method(tmp_path):
    """With no local benchmark to judge settlement by, an older file is topped
    up as before, but it must not be stamped method 4: a later run with local
    prices still owes it a recompute."""
    cache = tmp_path / 'r'
    cache.mkdir()
    path = cache / f'{RUN}_h30.json'
    path.write_text(json.dumps({
        'run_date': RUN, 'horizon_days': 30, 'method': 3, 'window_unverified': True,
        'spy_return': 0.03, 'coverage': 0.5, 'n_requested': 2, 'n_priced': 1,
        'tickers': {'AAA': {'ret': 0.1, 'excess_return': 0.07, 'start_price': 1.0,
                            'end_price': 1.1, 'spy_return': 0.03}}}), encoding='utf-8')
    idx = pd.date_range('2026-07-01', '2026-09-30', freq='B')

    class _Client:
        def fetch_history(self, ticker, period=None):
            return pd.Series(50 * 1.001 ** np.arange(len(idx)), index=idx)

    bt.annotate_snapshot_returns(_snap('AAA', 'BBB'), [30], _Client(), prices_dir=None,
                                 cache_dir=str(cache), today=date(2026, 9, 30))
    body = json.loads(path.read_text(encoding='utf-8'))
    assert 'BBB' in body['tickers'] and body['method'] == 3
