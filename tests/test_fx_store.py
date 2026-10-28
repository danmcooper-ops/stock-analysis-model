"""data/fx_store.py: cleaning, source merge, as-of lookups and refresh."""
from datetime import date

import pandas as pd
import pytest

from data import fx_store as fs
from data.fx_store import FXStore
# Bound at import, before conftest's autouse stub replaces the module's.
from data.fx_store import fetch_fred as real_fetch_fred


def _s(values, start='2026-01-05'):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)),
                     dtype=float)


# -- cleaning ----------------------------------------------------------------

def test_clean_drops_non_finite_and_non_positive():
    s, dropped = fs.clean_series(_s([1.0, float('nan'), 0.0, -1.0, float('inf'), 1.1]))
    assert list(s) == [1.0, 1.1]
    assert dropped == 4


def test_clean_drops_isolated_decimal_shift():
    # COP 2013-07-12: 0.27 between two days at 0.00052.
    s, dropped = fs.clean_series(_s([0.00052, 0.27, 0.00053, 0.00052]))
    assert dropped == 1
    assert s.max() < 0.001


def test_clean_keeps_a_devaluation():
    # A level shift moves the following bars too, so it is not a spike.
    s, dropped = fs.clean_series(_s([0.0028, 0.0028, 0.00125, 0.00124, 0.00124]))
    assert dropped == 0
    assert len(s) == 5


def test_clean_dedupes_and_sorts():
    raw = pd.Series([2.0, 1.0, 1.5],
                    index=pd.to_datetime(['2026-01-07', '2026-01-05', '2026-01-07']))
    s, _ = fs.clean_series(raw)
    assert list(s.index.date) == [date(2026, 1, 5), date(2026, 1, 7)]
    assert s.iloc[-1] == 1.5


# -- merge -------------------------------------------------------------------

def test_merge_prefers_yahoo_fills_from_fred_and_overrides_bad_prints():
    yahoo = _s([1.10, 1.11, 1.50], start='2026-01-06')   # 01-06..01-08
    fred = _s([1.09, 1.105, 1.112, 1.12])                 # 01-05..01-08
    frame, st = fs.merge_sources(yahoo, fred)
    assert frame.loc['2026-01-05', 'source'] == 'fred'    # Yahoo lacks it
    assert frame.loc['2026-01-06', 'source'] == 'yahoo'   # within tolerance
    assert frame.loc['2026-01-08', 'rate'] == pytest.approx(1.12)
    assert frame.loc['2026-01-08', 'source'] == 'fred'    # 34% off: overridden
    assert st == {'bars': 4, 'yahoo': 2, 'fred': 2, 'overridden': 1}


def test_merge_without_fred_is_yahoo():
    frame, st = fs.merge_sources(_s([1.0, 1.1]), None)
    assert set(frame['source']) == {'yahoo'}
    assert st['overridden'] == 0


def test_fetch_fred_inverts_units_per_usd():
    class Fred:
        def fetch_series(self, sid, start=None):
            assert sid == 'DEXJPUS'
            return {date(2026, 1, 5): 150.0}
    s = real_fetch_fred('JPY', Fred())
    assert s.iloc[0] == pytest.approx(1 / 150)


def test_fetch_fred_keeps_usd_per_unit_series():
    class Fred:
        def fetch_series(self, sid, start=None):
            assert sid == 'DEXUSEU'
            return {date(2026, 1, 5): 1.10}
    assert real_fetch_fred('EUR', Fred()).iloc[0] == pytest.approx(1.10)


def test_fetch_fred_unknown_currency_is_empty():
    assert real_fetch_fred('ARS', object()).empty


# -- store -------------------------------------------------------------------

def _store(tmp_path, ccy='EUR', values=None, start='2026-01-05'):
    store = FXStore(str(tmp_path))
    yahoo = _s(values or [1.10, 1.12, 1.14, 1.16, 1.18], start=start)
    store.refresh([ccy], yahoo_fetch=lambda c: yahoo,
                  fred_fetch=lambda c: pd.Series(dtype=float))
    return FXStore(str(tmp_path))   # a fresh reader, from disk


def test_rate_is_as_of_not_latest(tmp_path):
    store = _store(tmp_path)                       # 01-05..01-09
    assert store.rate('EUR', date(2026, 1, 6)) == pytest.approx(1.12)
    assert store.rate('eur', '2026-01-09') == pytest.approx(1.18)
    # A weekend takes Friday's bar.
    assert store.rate_info('EUR', date(2026, 1, 11)) == (
        pytest.approx(1.18), date(2026, 1, 9), 'yahoo')


def test_rate_is_none_before_history_or_past_lookback(tmp_path):
    store = _store(tmp_path)
    assert store.rate('EUR', date(2026, 1, 2)) is None
    assert store.rate('EUR', date(2026, 1, 20)) is None
    assert store.rate('EUR', date(2026, 1, 20), max_lookback_days=30) == pytest.approx(1.18)


def test_unknown_currency_is_none_and_usd_is_one(tmp_path):
    store = FXStore(str(tmp_path))
    assert store.rate('XYZ', date(2026, 1, 5)) is None
    assert store.avg_rate('XYZ', date(2026, 1, 1), date(2026, 1, 31)) is None
    assert store.rate('USD', date(2026, 1, 5)) == 1.0
    assert store.avg_rate('USD', date(2026, 1, 1), date(2026, 1, 31)) == 1.0


def test_avg_rate_and_coverage_floor(tmp_path):
    store = _store(tmp_path)
    assert store.avg_rate('EUR', date(2026, 1, 5), date(2026, 1, 9)) == pytest.approx(1.14)
    # Five bars cannot stand for a month of weekdays.
    assert store.avg_rate('EUR', date(2026, 1, 1), date(2026, 1, 31)) is None
    assert store.avg_rate('EUR', date(2026, 1, 9), date(2026, 1, 5)) is None


def test_close_rate_is_the_as_of_rate(tmp_path):
    store = _store(tmp_path)
    assert store.close_rate('EUR', date(2026, 1, 7)) == store.rate('EUR', date(2026, 1, 7))


def test_refresh_keeps_history_when_the_refetch_is_short(tmp_path):
    _store(tmp_path, values=[1.0 + i / 1000 for i in range(100)])
    store = FXStore(str(tmp_path))
    st = store.refresh(['EUR'], yahoo_fetch=lambda c: _s([1.2, 1.2]),
                       fred_fetch=lambda c: pd.Series(dtype=float))
    assert st['EUR']['status'] == 'kept'
    assert len(FXStore(str(tmp_path)).frame('EUR')) == 100


def test_refresh_survives_source_errors(tmp_path):
    def boom(c):
        raise RuntimeError('429')
    store = FXStore(str(tmp_path))
    st = store.refresh(['EUR', 'USD', ''], yahoo_fetch=boom, fred_fetch=boom)
    assert st == {'EUR': {'bars': 0, 'yahoo': 0, 'fred': 0, 'overridden': 0,
                          'spikes_dropped': 0, 'status': 'missing'}}
    assert store.currencies() == []


def test_refresh_falls_back_to_fred_alone(tmp_path):
    store = FXStore(str(tmp_path))
    st = store.refresh(['JPY'], yahoo_fetch=lambda c: pd.Series(dtype=float),
                       fred_fetch=lambda c: _s([0.0066, 0.0067]))
    assert st['JPY']['status'] == 'ok'
    assert store.rate_info('JPY', date(2026, 1, 6))[2] == 'fred'
    assert store.currencies() == ['JPY']
