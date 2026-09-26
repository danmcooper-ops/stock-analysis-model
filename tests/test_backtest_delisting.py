# tests/test_backtest_delisting.py
"""Delisted names are measured to their last close, not dropped.

On the 2026-07..09 corpus every unpriced forward return traced back to 50
tickers, and the bulk of them were acquisitions (EA, TMHC, NFBK, LEG...):
Yahoo drops a delisted symbol's history, so the backtest silently left out
exactly the names that had been taken over. scripts/backtest_cloud.py
backfill-prices recovers the series from Tiingo and lists the confirmed
delistings; scripts/backtest.py measures them to the last close with the
proceeds reinvested in SPY.
"""

import json
from datetime import date, datetime

import pandas as pd
import pytest

import scripts.backtest as bt
from scripts import backtest_cloud as bc

RUN = '2026-07-06'
TODAY = date(2026, 9, 26)


def _write_px(prices, ticker, start, end, value):
    idx = pd.date_range(start, end, freq='B', name='Date')
    vals = value(idx) if callable(value) else [value] * len(idx)
    pd.DataFrame({'Close': vals}, index=idx).to_parquet(prices / f'{ticker}.parquet')


def _manifest(prices, delisted, backfilled=()):
    (prices / bt.BACKFILL_MANIFEST).write_text(json.dumps(
        {'delisted': delisted, 'backfilled': list(backfilled)}), encoding='utf-8')


@pytest.fixture
def prices(tmp_path):
    p = tmp_path / 'prices'
    p.mkdir()
    # SPY: 100 until the deal closes on 07-20, then 110 to the eval date.
    _write_px(p, 'SPY', '2026-06-01', '2026-09-25',
              lambda idx: [100.0 if d <= pd.Timestamp('2026-07-20') else 110.0 for d in idx])
    _write_px(p, 'LIVE', '2026-06-01', '2026-09-25', 50.0)
    _write_px(p, 'DEAL', '2026-06-01', '2026-07-20',          # take-out at 60
              lambda idx: [40.0 if d <= pd.Timestamp('2026-07-06') else 60.0 for d in idx])
    _write_px(p, 'BUST', '2026-06-01', '2026-07-20',          # collapsed to 5
              lambda idx: [40.0 if d <= pd.Timestamp('2026-07-06') else 5.0 for d in idx])
    _write_px(p, 'GONE', '2026-05-01', '2026-06-15', 30.0)    # delisted before the snapshot
    _write_px(p, 'STALE', '2026-06-01', '2026-07-20', 20.0)   # ends early, not confirmed
    _manifest(p, {
        'DEAL': {'last_date': '2026-07-20', 'last_close': 60.0, 'source': 'tiingo'},
        'BUST': {'last_date': '2026-07-20', 'last_close': 5.0, 'source': 'tiingo'},
        'GONE': {'last_date': '2026-06-15', 'last_close': 30.0, 'source': 'tiingo'},
    })
    bt._MANIFEST_CACHE.clear()
    return p


def test_merger_measured_to_last_close_with_proceeds_in_spy(prices):
    rets = bt.terminal_returns(str(prices), ['DEAL'], datetime(2026, 7, 6),
                               datetime(2026, 8, 5),
                               bt.load_backfill_manifest(str(prices))['delisted'])
    # 40 -> 60 (+50%), then cash rides SPY 100 -> 110 (+10%): 1.5 * 1.1 - 1
    assert rets['DEAL']['ret'] == pytest.approx(0.65)
    assert rets['DEAL']['delisted'] == {'date': '2026-07-20', 'kind': 'merger'}


def test_performance_delisting_takes_the_shumway_haircut(prices):
    rets = bt.terminal_returns(str(prices), ['BUST'], datetime(2026, 7, 6),
                               datetime(2026, 8, 5),
                               bt.load_backfill_manifest(str(prices))['delisted'])
    # 5 < 0.5 * 40: performance. 5 * 0.7 / 40 * 1.1 - 1
    assert rets['BUST']['ret'] == pytest.approx(5 * 0.7 / 40 * 1.1 - 1)
    assert rets['BUST']['delisted']['kind'] == 'performance'


def test_delisting_kind_thresholds():
    assert bt.delisting_kind(40.0, 60.0) == 'merger'
    assert bt.delisting_kind(40.0, 21.0) == 'merger'
    assert bt.delisting_kind(40.0, 19.0) == 'performance'
    assert bt.delisting_kind(1.5, 0.9) == 'performance'


def test_fetch_forward_returns_measures_confirmed_delistings_only(prices):
    got = bt.fetch_forward_returns(['LIVE', 'DEAL', 'GONE', 'STALE'], RUN, 30, None,
                                   prices_dir=str(prices))
    assert 'LIVE' in got and 'DEAL' in got
    assert 'GONE' not in got      # was already gone before the snapshot
    assert 'STALE' not in got     # ends early but no delisting confirmed


def test_analyze_run_reports_delistings_and_stale_rows(prices, tmp_path):
    snap = {'date': RUN, 'results': [
        {'ticker': 'LIVE', 'price': 50.0, 'rating': 'BUY'},
        {'ticker': 'DEAL', 'price': 40.0, 'rating': 'PASS'},
        {'ticker': 'BUST', 'price': 40.0, 'rating': 'PASS'},
        {'ticker': 'GONE', 'price': 30.0, 'rating': 'HOLD'},
        {'ticker': 'STALE', 'price': 20.0, 'rating': 'HOLD'},
    ]}
    m = bt.analyze_run(snap, 30, None, prices_dir=str(prices),
                       cache_dir=str(tmp_path / 'returns'), today=TODAY)
    cov = m['coverage']
    assert cov['delisted_by_rating'] == {'PASS': {'merger': 1, 'performance': 1}}
    assert cov['gone_before_snapshot'] == 1
    assert cov['unpriced_by_rating'] == {'HOLD': 1}
    detail = {d['ticker']: d for d in m['details']}
    assert detail['DEAL']['delisted'] == 'merger' and detail['LIVE']['delisted'] is None
    body = json.loads((tmp_path / 'returns' / f'{RUN}_h30.json').read_text(encoding='utf-8'))
    assert body['method'] == bt.SIDECAR_METHOD
    assert body['tickers']['DEAL']['delisted']['kind'] == 'merger'


def _fwd(ret):
    return {'excess_return': ret, 'ret': ret, 'start_price': 50.0,
            'end_price': 50.0 * (1 + ret), 'spy_return': 0.0}


def test_method_1_sidecar_is_topped_up_once_keeping_frozen_returns(prices, tmp_path):
    cache = tmp_path / 'returns'
    cache.mkdir()
    path = cache / f'{RUN}_h30.json'
    path.write_text(json.dumps({                    # complete by coverage, old method
        'run_date': RUN, 'horizon_days': 30, 'spy_return': 0.1, 'coverage': 0.95,
        'n_requested': 2, 'n_priced': 1, 'tickers': {'LIVE': _fwd(0.123)}}),
        encoding='utf-8')
    snap = {'date': RUN, 'results': [{'ticker': 'LIVE', 'price': 50.0},
                                     {'ticker': 'DEAL', 'price': 40.0}]}
    out = bt.annotate_snapshot_returns(snap, [30], None, prices_dir=str(prices),
                                       cache_dir=str(cache), today=TODAY)
    assert out == {30: 2}
    body = json.loads(path.read_text(encoding='utf-8'))
    assert body['method'] == bt.SIDECAR_METHOD
    assert body['tickers']['LIVE']['ret'] == 0.123        # frozen, not recomputed
    assert body['tickers']['DEAL']['delisted']['kind'] == 'merger'
    assert snap['_fwd_stats'][30]['cached'] is False
    # ...and from now on it is complete
    snap2 = {'date': RUN, 'results': [{'ticker': 'LIVE', 'price': 50.0},
                                      {'ticker': 'DEAL', 'price': 40.0}]}
    bt.annotate_snapshot_returns(snap2, [30], None, prices_dir=str(prices),
                                 cache_dir=str(cache), today=TODAY)
    assert snap2['_fwd_stats'][30]['cached'] is True


def test_complete_sidecar_reopens_for_a_newly_confirmed_delisting(prices):
    cached = {'spy_return': 0.1, 'coverage': 0.95, 'method': bt.SIDECAR_METHOD,
              'tickers': {'LIVE': _fwd(0.1)}}
    man = bt.load_backfill_manifest(str(prices))
    assert bt.sidecar_is_complete(cached, ['LIVE'], man)
    assert not bt.sidecar_is_complete(cached, ['LIVE', 'DEAL'], man)
    assert not bt.sidecar_is_complete(cached, ['LIVE', 'VSCO'],
                                      {'delisted': {}, 'backfilled': ['VSCO']})
    assert bt.sidecar_is_complete(cached, ['LIVE', 'STALE'], man)


# ---------------------------------------------------------------------------
# backfill-prices (scripts/backtest_cloud.py)
# ---------------------------------------------------------------------------

class _FakeTiingo:
    def __init__(self, series, limit_after=None):
        self.series = series
        self.calls = []
        self.limit_after = limit_after
        self.rate_limited = False

    def fetch_closes(self, ticker, start):
        self.calls.append(ticker)
        if self.limit_after is not None and len(self.calls) > self.limit_after:
            self.rate_limited = True
            return None
        return self.series.get(ticker, pd.Series(dtype=float))


def _closes(start, end, value):
    idx = pd.date_range(start, end, freq='B')
    return pd.Series([float(value)] * len(idx), index=idx)


def test_price_problems_classification(tmp_path):
    p = tmp_path / 'prices'
    p.mkdir()
    _write_px(p, 'OK', '2026-06-01', '2026-09-25', 1.0)
    _write_px(p, 'STUB', '2026-08-14', '2026-08-17', 1.0)   # Yahoo's stub of a delisted name
    _write_px(p, 'EARLY', '2026-06-01', '2026-07-20', 1.0)
    dates = {'OK': [RUN], 'STUB': [RUN, '2026-07-07'], 'EARLY': [RUN],
             'NOFILE': [RUN], 'SPY': [RUN]}
    got = bc.price_problems(dates, str(p), today=TODAY)
    assert {t: v['reason'] for t, v in got.items()} == {
        'STUB': 'starts_late', 'EARLY': 'ends_early', 'NOFILE': 'missing'}
    assert got['STUB']['rows'] == 2 and got['STUB']['first'] == date(2026, 7, 6)


def test_backfill_writes_prices_caches_and_confirms_delistings(tmp_path):
    p, cache = tmp_path / 'prices', tmp_path / 'cache'
    p.mkdir()
    _write_px(p, 'STUB', '2026-08-14', '2026-08-17', 1.0)
    client = _FakeTiingo({
        'STUB': _closes('2026-06-26', '2026-08-17', 30.0),   # delisted 08-17
        'GAP': _closes('2026-06-26', '2026-09-25', 80.0),    # Yahoo gap, still trading
    })
    problems = {'STUB': {'first': date(2026, 7, 6), 'rows': 5, 'reason': 'starts_late'},
                'GAP': {'first': date(2026, 7, 6), 'rows': 9, 'reason': 'missing'},
                'NOPE': {'first': date(2026, 7, 6), 'rows': 1, 'reason': 'missing'}}
    stats = bc.backfill_prices(problems, str(p), str(cache), client, '2026-06-26',
                               today=TODAY, log=lambda *_: None)
    assert client.calls == ['GAP', 'STUB', 'NOPE']          # most rows first
    assert stats['written'] == 2 and stats['unknown'] == 1
    man = json.loads((p / '_backfill.json').read_text(encoding='utf-8'))
    assert set(man['delisted']) == {'STUB'}
    assert man['delisted']['STUB']['last_date'] == '2026-08-17'
    assert man['backfilled'] == ['GAP', 'STUB']
    assert bc._bar_span(str(p / 'STUB.parquet'))[0] == date(2026, 6, 26)

    # Next week: every answer comes from the cache (NOPE's 404 included).
    client2 = _FakeTiingo({})
    bc.backfill_prices(problems, str(p), str(cache), client2, '2026-06-26',
                       today=date(2026, 9, 30), log=lambda *_: None)
    assert client2.calls == []


def test_backfill_respects_the_call_cap_and_rate_limit(tmp_path):
    p, cache = tmp_path / 'prices', tmp_path / 'cache'
    p.mkdir()
    series = {t: _closes('2026-06-26', '2026-08-17', 10.0) for t in 'ABCD'}
    problems = {t: {'first': date(2026, 7, 6), 'rows': 10 - i, 'reason': 'missing'}
                for i, t in enumerate('ABCD')}
    capped = _FakeTiingo(series)
    stats = bc.backfill_prices(problems, str(p), str(cache), capped, '2026-06-26',
                               today=TODAY, max_calls=2, log=lambda *_: None)
    assert capped.calls == ['A', 'B'] and stats['deferred'] == 2
    limited = _FakeTiingo(series, limit_after=0)
    stats = bc.backfill_prices(problems, str(p), str(cache), limited, '2026-06-26',
                               today=TODAY, log=lambda *_: None)
    assert limited.calls == ['C'] and stats['deferred'] == 2   # C failed, D never tried
