# tests/test_price_store.py
"""Tests for the bulk DuckDB price-parquet reader (data/price_store.py).

The bulk query replaces a per-ticker pandas read in the backtester's
forward-return pass, so the tests that matter are parity ones: it must pick
exactly the bar the pandas path picks, including the awkward cases (ties,
stale files, NaN closes, Close-only stubs, the gap boundary).
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from data.price_store import parquet_paths, window_closes
from scripts.backtest import MAX_SNAP_GAP_DAYS, _nearest_bar


def _write(dirpath, ticker, index, closes, ohlcv=True):
    df = pd.DataFrame({'Close': closes}, index=pd.to_datetime(index))
    if ohlcv:
        df['Open'] = df['High'] = df['Low'] = df['Close']
        df['Volume'] = 1000
    df.index.name = 'Date'
    df.to_parquet(os.path.join(dirpath, f'{ticker}.parquet'))


def _pandas_window(dirpath, ticker, start, end):
    """The per-ticker path from backtest.fetch_forward_returns, verbatim."""
    path = os.path.join(dirpath, f'{ticker}.parquet')
    if not os.path.exists(path):
        return None
    hist = pd.read_parquet(path)[['Close']].sort_index()['Close']
    si = _nearest_bar(hist.index, pd.Timestamp(start))
    ei = _nearest_bar(hist.index, pd.Timestamp(end))
    if si is None or ei is None:
        return None
    sp, ep = float(hist.iloc[si]), float(hist.iloc[ei])
    if sp > 0 and math.isfinite(sp) and math.isfinite(ep):
        return {'start': sp, 'end': ep}
    return None


@pytest.fixture
def prices(tmp_path):
    d = tmp_path / 'prices'
    d.mkdir()
    bdays = pd.date_range('2025-01-01', '2026-06-30', freq='B')
    rng = np.random.default_rng(11)
    _write(d, 'FULL', bdays, 100 + rng.standard_normal(len(bdays)).cumsum())
    # Close-only write-through stub (no Volume column)
    _write(d, 'STUB', bdays, 50 + rng.standard_normal(len(bdays)).cumsum(),
           ohlcv=False)
    # Stale file: stops early, so late targets fall outside the gap window
    half = bdays[:len(bdays) // 2]
    _write(d, 'STALE', half, 30 + rng.standard_normal(len(half)).cumsum())
    # Equidistant pair around 2025-06-08 (Sun): Fri 06-06 and Tue 06-10
    _write(d, 'TIE', ['2025-06-06', '2025-06-10'], [10.0, 20.0])
    # NaN close exactly on a target date
    _write(d, 'NANC', ['2025-03-03', '2025-03-04', '2025-03-05', '2025-06-02'],
           [10.0, float('nan'), 12.0, 20.0])
    # Non-positive start price must be rejected by both paths
    _write(d, 'ZERO', ['2025-03-04', '2025-06-02'], [0.0, 15.0])
    return str(d)


ALL = ['FULL', 'STUB', 'STALE', 'TIE', 'NANC', 'ZERO', 'NOFILE']


@pytest.mark.parametrize('start,end', [
    ('2025-06-08', '2025-09-08'),   # tie date as the start target
    ('2025-03-04', '2025-06-02'),   # NaN / zero bars land on the targets
    ('2025-01-15', '2025-04-15'),
    ('2026-06-29', '2026-06-30'),   # near the end of the full series
    ('2026-09-01', '2026-12-01'),   # beyond every file: nothing matches
])
def test_matches_the_per_ticker_pandas_path(prices, start, end):
    bulk = window_closes(prices, start, end, tickers=ALL) or {}
    expected = {t: w for t in ALL
                if (w := _pandas_window(prices, t, start, end)) is not None}
    assert set(bulk) == set(expected)
    for t, want in expected.items():
        assert bulk[t]['start'] == pytest.approx(want['start'], abs=1e-12)
        assert bulk[t]['end'] == pytest.approx(want['end'], abs=1e-12)


def test_tie_breaks_toward_the_later_bar(prices):
    """pandas get_indexer(method='nearest') resolves a tie to the later bar."""
    got = window_closes(prices, '2025-06-08', '2025-06-10', tickers=['TIE'])
    assert got['TIE']['start'] == 20.0        # 06-10, not 06-06
    assert _pandas_window(prices, 'TIE', '2025-06-08', '2025-06-10')['start'] == 20.0


def test_nan_close_is_rejected_not_skipped_to_a_neighbour(prices):
    """A NaN bar on the target drops the ticker; it never snaps past it to a
    neighbouring bar, which would silently add an observation."""
    assert _pandas_window(prices, 'NANC', '2025-03-04', '2025-06-02') is None
    got = window_closes(prices, '2025-03-04', '2025-06-02', tickers=['NANC'])
    assert 'NANC' not in got


def test_non_positive_start_is_dropped(prices):
    assert _pandas_window(prices, 'ZERO', '2025-03-04', '2025-06-02') is None
    assert 'ZERO' not in window_closes(prices, '2025-03-04', '2025-06-02',
                                       tickers=['ZERO'])


def test_gap_boundary_is_inclusive(prices):
    """A target exactly MAX_SNAP_GAP_DAYS from the nearest bar still matches;
    one day further does not."""
    on_edge = (pd.Timestamp('2025-06-10') + pd.Timedelta(days=MAX_SNAP_GAP_DAYS))
    past = on_edge + pd.Timedelta(days=1)
    assert window_closes(prices, '2025-06-06', on_edge.date(), tickers=['TIE'])
    assert not window_closes(prices, '2025-06-06', past.date(), tickers=['TIE'])


def test_stale_file_drops_only_that_ticker(prices):
    got = window_closes(prices, '2026-06-01', '2026-06-29',
                        tickers=['FULL', 'STALE'])
    assert 'FULL' in got and 'STALE' not in got


def test_returns_none_when_it_cannot_answer(prices, tmp_path):
    assert window_closes(str(tmp_path / 'missing'), '2025-01-01', '2025-02-01') is None
    assert window_closes(prices, '2025-01-01', '2025-02-01', tickers=['NOPE']) is None
    assert window_closes(prices, '2025-01-01', '2025-02-01', tickers=[]) is None


def test_parquet_paths_selection(prices):
    assert len(parquet_paths(prices)) == 6           # every file
    assert parquet_paths(prices, ['FULL', 'NOFILE']) == [
        os.path.join(prices, 'FULL.parquet')]
    assert parquet_paths(prices, ['FULL', 'FULL']) == [
        os.path.join(prices, 'FULL.parquet')]        # de-duped
    assert parquet_paths(prices, [None, '']) == []
    assert parquet_paths(str(prices) + '_nope') == []
