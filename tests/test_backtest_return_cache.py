# tests/test_backtest_return_cache.py
"""Forward-return sidecars once they are persisted between runs.

The cloud backtest starts every week from a cold price cache and keeps the
return sidecars on the data/snapshots branch, so whatever one run writes is
reused by every later one. A sidecar must therefore never freeze a return
measured against a missing benchmark, and a sidecar that priced only part of
the snapshot must be topped up rather than trusted.
"""

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

import scripts.backtest as bt

RUN = '2026-07-06'
TODAY = date(2026, 9, 1)
TICKERS = [f'T{i:02d}' for i in range(10)]


def _series(start=100.0, drift=0.001):
    idx = pd.date_range('2026-06-01', '2026-09-01', freq='B')
    return pd.Series(start * (1 + drift) ** np.arange(len(idx)), index=idx)


class _Client:
    """fetch_history answers from a dict; records every ticker asked for."""

    def __init__(self, have):
        self.have = have
        self.asked = []

    def fetch_history(self, ticker, period=None):
        self.asked.append(ticker)
        return self.have.get(ticker)


class _Forbidden:
    def fetch_history(self, *a, **k):
        raise AssertionError('a complete sidecar must not trigger a fetch')


def _snapshot():
    return {'date': RUN, 'results': [
        {'ticker': t, 'price': 100.0, 'rating': 'PASS' if i % 2 else 'BUY'}
        for i, t in enumerate(TICKERS)]}


def _fwd(ret=0.05, spy=0.02):
    return {'excess_return': ret - spy, 'ret': ret, 'start_price': 100.0,
            'end_price': 100.0 * (1 + ret), 'spy_return': spy}


def _write(cache, body):
    cache.mkdir(exist_ok=True)
    path = cache / f'{RUN}_h30.json'
    path.write_text(json.dumps(body), encoding='utf-8')
    return path


def test_missing_benchmark_is_never_a_zero_return(tmp_path):
    snap = _snapshot()
    client = _Client({t: _series() for t in TICKERS})    # no SPY
    out = bt.annotate_snapshot_returns(snap, [30], client, cache_dir=str(tmp_path),
                                       today=TODAY)
    assert out == {30: None}
    assert not any('_fwd' in r for r in snap['results'])
    assert not list(tmp_path.iterdir())                 # nothing frozen
    assert snap['_fwd_stats'][30]['no_benchmark'] is True


def test_fresh_computation_records_coverage(tmp_path):
    snap = _snapshot()
    have = {t: _series() for t in TICKERS[:8]}
    have['SPY'] = _series(drift=0.0005)
    out = bt.annotate_snapshot_returns(snap, [30], _Client(have),
                                       cache_dir=str(tmp_path), today=TODAY)
    assert out == {30: 8}
    body = json.loads((tmp_path / f'{RUN}_h30.json').read_text(encoding='utf-8'))
    assert body['n_requested'] == 10 and body['n_priced'] == 8
    assert body['coverage'] == pytest.approx(0.8)
    assert body['spy_return'] == pytest.approx(snap['results'][0]['_fwd'][30]['spy_return'])
    assert snap['_fwd_stats'][30] == {'requested': 10, 'priced': 8, 'implausible': 0,
                                      'coverage': 0.8, 'cached': False}


def test_complete_sidecar_is_reused_without_fetching(tmp_path):
    _write(tmp_path, {'run_date': RUN, 'horizon_days': 30, 'spy_return': 0.02,
                      'coverage': 1.0, 'n_requested': 10, 'n_priced': 10,
                      'tickers': {t: _fwd() for t in TICKERS}})
    snap = _snapshot()
    out = bt.annotate_snapshot_returns(snap, [30], _Forbidden(),
                                       cache_dir=str(tmp_path), today=TODAY)
    assert out == {30: 10}
    assert snap['_fwd_stats'][30]['cached'] is True


def test_low_coverage_sidecar_is_topped_up_and_keeps_frozen_returns(tmp_path):
    frozen = {t: _fwd(ret=0.5) for t in TICKERS[:5]}      # differs from today's prices
    path = _write(tmp_path, {'run_date': RUN, 'horizon_days': 30, 'spy_return': 0.02,
                             'coverage': 0.5, 'computed_at': '2026-08-10',
                             'tickers': frozen})
    have = {t: _series() for t in TICKERS}
    have['SPY'] = _series(drift=0.0005)
    client = _Client(have)
    snap = _snapshot()
    out = bt.annotate_snapshot_returns(snap, [30], client, cache_dir=str(tmp_path),
                                       today=TODAY)
    assert out == {30: 10}
    # only the missing tickers (plus the benchmark) were fetched
    assert sorted(client.asked) == sorted(TICKERS[5:] + ['SPY'])
    rows = {r['ticker']: r for r in snap['results']}
    assert rows['T00']['_fwd'][30]['ret'] == 0.5          # frozen value kept
    body = json.loads(path.read_text(encoding='utf-8'))
    assert body['coverage'] == 1.0 and body['n_priced'] == 10
    assert body['computed_at'] == '2026-08-10' and body['updated_at'] == TODAY.isoformat()


def test_legacy_sidecar_without_coverage_is_filled_once(tmp_path):
    path = _write(tmp_path, {'run_date': RUN, 'horizon_days': 30, 'spy_return': 0.02,
                             'tickers': {t: _fwd() for t in TICKERS[:9]}})
    have = {'T09': _series(), 'SPY': _series(drift=0.0005)}
    snap = _snapshot()
    bt.annotate_snapshot_returns(snap, [30], _Client(have), cache_dir=str(tmp_path),
                                 today=TODAY)
    body = json.loads(path.read_text(encoding='utf-8'))
    assert body['coverage'] == 1.0
    # ...and from then on it is complete: no further fetch
    bt.annotate_snapshot_returns(_snapshot(), [30], _Forbidden(),
                                 cache_dir=str(tmp_path), today=TODAY)


def test_legacy_sidecar_keeps_its_benchmark_when_spy_cannot_be_refetched(tmp_path):
    _write(tmp_path, {'run_date': RUN, 'horizon_days': 30, 'spy_return': 0.02,
                      'tickers': {t: _fwd() for t in TICKERS}})
    snap = _snapshot()
    out = bt.annotate_snapshot_returns(snap, [30], None, cache_dir=str(tmp_path),
                                       today=TODAY)
    assert out == {30: 10}


def test_local_prices_only_never_calls_the_network(tmp_path):
    prices = tmp_path / 'prices'
    prices.mkdir()
    for t in ['SPY', 'T00']:
        s = _series()
        pd.DataFrame({'Close': s.values}, index=s.index.rename('Date')).to_parquet(
            prices / f'{t}.parquet')
    snap = _snapshot()
    out = bt.annotate_snapshot_returns(snap, [30], None, prices_dir=str(prices),
                                       cache_dir=str(tmp_path / 'returns'), today=TODAY)
    assert out == {30: 1}
    assert snap['_fwd_stats'][30]['coverage'] == 0.1


def test_analyze_run_reports_attrition_by_rating(tmp_path):
    have = {t: _series() for t in TICKERS[:6]}
    have['SPY'] = _series(drift=0.0005)
    m = bt.analyze_run(_snapshot(), 30, _Client(have), cache_dir=str(tmp_path),
                       today=TODAY)
    cov = m['coverage']
    assert cov['rows'] == 10 and cov['measured'] == 6 and cov['priced'] == 6
    # T06..T09 are unpriced: T06, T08 are BUY; T07, T09 are PASS
    assert cov['unpriced_by_rating'] == {'BUY': 2, 'PASS': 2}


def test_summary_carries_provenance_coverage_and_skips():
    metrics = [{'run_date': RUN, 'horizon': 30, 'spy_return': 0.01, 'buckets': {},
                'details': [], 'coverage': {'rows': 10, 'measured': 9,
                                            'coverage': 0.9}}]
    report = {'loaded': 3, 'usable': 1,
              'skipped': [['2026-07-01', 'dated before 2026-07-06']],
              'unmeasured': [{'run_date': '2026-07-07', 'horizon': 30,
                              'reason': 'no benchmark return'}]}
    s = bt.build_measure_summary(metrics, [30], date(2026, 7, 6), '2026-09-27',
                                 {}, report)
    assert s['date'] == '2026-09-27'
    assert s['provenance']['snapshots_loaded'] == 3
    assert s['provenance']['min_return_coverage'] == bt.MIN_RETURN_COVERAGE
    assert 'git_sha' in s['provenance']
    assert s['skipped_snapshots'] == report['skipped']
    assert s['unmeasured'] == report['unmeasured']
    assert s['coverage'] == [{'run_date': RUN, 'horizon': 30, 'rows': 10,
                              'measured': 9, 'coverage': 0.9}]


def test_measure_cli_names_outputs_by_stamp(tmp_path, monkeypatch):
    # The rows carry no sector: measure once crashed formatting a None sector.
    from scripts.scoring import GATES
    rows = [dict({g.field: 1.0 for g in GATES}, ticker=t, price=100.0,
                 rating='BUY', _composite_score=float(i))
            for i, t in enumerate(TICKERS)]
    res = tmp_path / 'res'
    res.mkdir()
    (res / f'results_{RUN}.json').write_text(
        json.dumps({'date': RUN, 'count': len(rows), 'results': rows}), encoding='utf-8')
    prices = tmp_path / 'prices'
    prices.mkdir()
    for i, t in enumerate(TICKERS + ['SPY']):
        s = _series(drift=0.0005 * (i % 4))
        pd.DataFrame({'Close': s.values}, index=s.index.rename('Date')).to_parquet(
            prices / f'{t}.parquet')
    monkeypatch.setattr(bt, 'USE_SNAPSHOT_STORE', False)
    args = argparse.Namespace(
        signals=None, horizons='30', since=None, results_dir=str(res),
        prices_dir=str(prices), cache_dir=str(tmp_path / 'returns'),
        output_dir=str(tmp_path / 'out'), stamp='2026-09-27', cohort=None,
        exclude_capped=False, local_prices_only=True)
    bt._cli_measure(args)
    out = tmp_path / 'out'
    assert (out / 'backtest_2026-09-27.xlsx').exists()
    summary = json.loads((out / 'backtest_summary_2026-09-27.json').read_text(encoding='utf-8'))
    assert summary['date'] == '2026-09-27'
    assert summary['coverage'][0]['coverage'] == 1.0
    assert (tmp_path / 'returns' / f'{RUN}_h30.json').exists()


def test_topup_that_finds_nothing_leaves_the_sidecar_untouched(tmp_path):
    body = {'run_date': RUN, 'horizon_days': 30, 'spy_return': 0.02,
            'coverage': 0.5, 'n_requested': 10, 'n_priced': 5,
            'updated_at': '2026-08-10', 'tickers': {t: _fwd() for t in TICKERS[:5]}}
    path = _write(tmp_path, body)
    before = path.read_text(encoding='utf-8')
    have = {'SPY': _series(drift=0.0005)}            # still nothing for T05..T09
    out = bt.annotate_snapshot_returns(_snapshot(), [30], _Client(have),
                                       cache_dir=str(tmp_path), today=TODAY)
    assert out == {30: 5}
    assert path.read_text(encoding='utf-8') == before
