# tests/test_price_store.py
"""Tests for the bulk DuckDB price-parquet reader (data/price_store.py).

The bulk query replaces a per-ticker pandas read in the backtester's
forward-return pass, so the tests that matter are parity ones: it must pick
exactly the bars the per-ticker reference (select_window) picks, including
the awkward cases (ties, stale files, NaN closes, Close-only stubs, the gap
boundary, spikes, dead quotes, series that stop short).
"""

import os

import numpy as np
import pandas as pd
import pytest

from data.price_store import (last_valid_dates, parquet_paths, select_window,
                              trim_trailing_invalid, valid_span, window_closes)
from scripts.backtest import MAX_SNAP_GAP_DAYS


def _write(dirpath, ticker, index, closes, ohlcv=True):
    df = pd.DataFrame({'Close': closes}, index=pd.to_datetime(index))
    if ohlcv:
        df['Open'] = df['High'] = df['Low'] = df['Close']
        df['Volume'] = 1000
    df.index.name = 'Date'
    df.to_parquet(os.path.join(dirpath, f'{ticker}.parquet'))


def _pandas_window(dirpath, ticker, start, end):
    """The per-ticker path from backtest.fetch_forward_returns."""
    path = os.path.join(dirpath, f'{ticker}.parquet')
    if not os.path.exists(path):
        return None
    hist = pd.read_parquet(path)[['Close']].sort_index()['Close']
    return select_window(hist, start, end, max_gap_days=MAX_SNAP_GAP_DAYS)


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
    # One-day bad print between steady closes (GRKZF's $0.001 on 2026-08-14)
    spike = pd.date_range('2025-05-26', '2025-06-13', freq='B')
    vals = [15.5 + 0.01 * i for i in range(len(spike))]
    vals[list(spike).index(pd.Timestamp('2025-06-02'))] = 0.001
    _write(d, 'SPIKE', list(pd.date_range('2025-03-03', '2025-03-07', freq='B'))
           + list(spike), [15.0] * 5 + vals)
    # Dead quote: one close repeated for weeks (forward-filled Tiingo series)
    _write(d, 'FLAT', bdays, [19.21] * len(bdays))
    # Sparse OTC line: the bar nearest the start date is AFTER it
    _write(d, 'SPARSE', ['2025-02-27', '2025-03-07', '2025-06-02', '2025-06-05'],
           [5.0, 9.0, 7.0, 7.5])
    return str(d)


ALL = ['FULL', 'STUB', 'STALE', 'TIE', 'NANC', 'ZERO', 'SPIKE', 'FLAT',
       'SPARSE', 'NOFILE']


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


def test_end_tie_breaks_toward_the_later_bar(prices):
    got = window_closes(prices, '2025-06-06', '2025-06-08', tickers=['TIE'])
    assert got['TIE']['end'] == 20.0          # 06-10, not 06-06
    assert select_window(pd.read_parquet(os.path.join(prices, 'TIE.parquet'))['Close'],
                         '2025-06-06', '2025-06-08')['end'] == 20.0


def test_start_is_as_of_never_a_later_bar(prices):
    """The nearest bar to a start date can lie after it; the snapshot did not
    have that price, so the start is the last bar on or before the date."""
    got = window_closes(prices, '2025-03-04', '2025-06-03', tickers=['SPARSE'])
    assert got['SPARSE']['start'] == 5.0      # 02-27, not the nearer 03-07
    assert _pandas_window(prices, 'SPARSE', '2025-03-04', '2025-06-03')['start'] == 5.0


def test_nan_close_is_not_a_bar(prices):
    """A NaN close is a missing print, not a price: the nearest valid close
    within the gap answers instead. Dropping the ticker used to leave a
    permanent hole once a >=90%-coverage sidecar froze without it."""
    got = window_closes(prices, '2025-03-04', '2025-06-02', tickers=['NANC'])
    assert got['NANC'] == {'start': 10.0, 'end': 20.0}     # 03-03, as of 03-04
    assert _pandas_window(prices, 'NANC', '2025-03-04', '2025-06-02') == got['NANC']


def test_a_series_must_reach_the_end_date(prices, tmp_path):
    """A file whose valid closes stop before the end date is not measured to
    its last bar, even within the gap (VSCO's Tiingo series ended 10-02 and
    was frozen as the 10-04 close)."""
    d = tmp_path / 'short'
    d.mkdir()
    _write(d, 'SHORT', pd.date_range('2025-01-01', '2025-03-28', freq='B'),
           np.linspace(10, 20, 63))
    assert not window_closes(str(d), '2025-01-15', '2025-03-30', tickers=['SHORT'])
    assert window_closes(str(d), '2025-01-15', '2025-03-30', tickers=['SHORT'],
                         require_reach=False)
    assert _pandas_window(str(d), 'SHORT', '2025-01-15', '2025-03-30') is None


def test_trailing_nan_row_does_not_count_as_reaching(tmp_path):
    """SPY's 2026-10-06 row had a NaN close: the file reaches 10-05, not 10-06."""
    d = tmp_path / 'p'
    d.mkdir()
    idx = pd.date_range('2026-07-01', '2026-10-06', freq='B')
    closes = list(np.linspace(740, 775, len(idx)))
    closes[-1] = float('nan')
    _write(d, 'SPY', idx, closes)
    assert not window_closes(str(d), '2026-07-08', '2026-10-06', tickers=['SPY'])
    assert window_closes(str(d), '2026-07-07', '2026-10-05', tickers=['SPY'])
    path = os.path.join(str(d), 'SPY.parquet')
    assert valid_span(path)[1] == pd.Timestamp('2026-10-05').date()
    assert last_valid_dates(str(d), ['SPY'])['SPY'] == pd.Timestamp('2026-10-05').date()


def test_spike_bar_is_rejected_at_either_end(prices):
    assert 'SPIKE' not in window_closes(prices, '2025-03-05', '2025-06-02',
                                        tickers=['SPIKE'])
    assert 'SPIKE' in window_closes(prices, '2025-03-05', '2025-06-04',
                                    tickers=['SPIKE'])


def test_dead_quote_window_is_rejected(prices):
    assert 'FLAT' not in window_closes(prices, '2025-03-04', '2025-06-02',
                                       tickers=['FLAT'])
    # Too few bars to call it dead: a quiet two-day window is a measurement
    assert 'FLAT' in window_closes(prices, '2025-03-04', '2025-03-05',
                                   tickers=['FLAT'])


def test_trim_trailing_invalid_keeps_interior_rows():
    df = pd.DataFrame({'Close': [1.0, float('nan'), 2.0, float('nan'), 0.0]},
                      index=pd.date_range('2026-01-01', periods=5))
    assert list(trim_trailing_invalid(df)['Close'].fillna(-1)) == [1.0, -1, 2.0]



def test_non_positive_start_is_dropped(prices):
    assert _pandas_window(prices, 'ZERO', '2025-03-04', '2025-06-02') is None
    assert 'ZERO' not in window_closes(prices, '2025-03-04', '2025-06-02',
                                       tickers=['ZERO'])


def test_gap_boundary_is_inclusive(prices):
    """A target exactly MAX_SNAP_GAP_DAYS from the nearest bar still matches;
    one day further does not."""
    on_edge = (pd.Timestamp('2025-06-10') + pd.Timedelta(days=MAX_SNAP_GAP_DAYS))
    past = on_edge + pd.Timedelta(days=1)
    # (require_reach off: these targets lie past TIE's last bar on purpose)
    assert window_closes(prices, '2025-06-06', on_edge.date(), tickers=['TIE'],
                         require_reach=False)
    assert not window_closes(prices, '2025-06-06', past.date(), tickers=['TIE'],
                             require_reach=False)


def test_stale_file_drops_only_that_ticker(prices):
    got = window_closes(prices, '2026-06-01', '2026-06-29',
                        tickers=['FULL', 'STALE'])
    assert 'FULL' in got and 'STALE' not in got


def test_returns_none_when_it_cannot_answer(prices, tmp_path):
    assert window_closes(str(tmp_path / 'missing'), '2025-01-01', '2025-02-01') is None
    assert window_closes(prices, '2025-01-01', '2025-02-01', tickers=['NOPE']) is None
    assert window_closes(prices, '2025-01-01', '2025-02-01', tickers=[]) is None


def test_parquet_paths_selection(prices):
    assert len(parquet_paths(prices)) == 9           # every file
    assert parquet_paths(prices, ['FULL', 'NOFILE']) == [
        os.path.join(prices, 'FULL.parquet')]
    assert parquet_paths(prices, ['FULL', 'FULL']) == [
        os.path.join(prices, 'FULL.parquet')]        # de-duped
    assert parquet_paths(prices, [None, '']) == []
    assert parquet_paths(str(prices) + '_nope') == []
