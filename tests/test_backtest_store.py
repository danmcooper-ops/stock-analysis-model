# tests/test_backtest_store.py
"""The backtester reading snapshots from the DuckDB store instead of the JSONs.

Every assertion here is a parity one: the store path exists only to save RAM
and I/O, so a measurement or a re-score taken through it must equal the one
taken by parsing the files.
"""

import json

import numpy as np
import pandas as pd
import pytest

import scripts.backtest as bt
from scripts.ingest_snapshots import ingest_dir
from scripts.param_set import default_params
from scripts.scoring import GATES

DATES = ['2026-01-05', '2026-02-05', '2026-03-05']
TICKERS = [f'B{i:02d}' for i in range(12)]


def _row(ticker, seed):
    rng = np.random.default_rng(seed)
    r = {
        'ticker': ticker,
        'sector': 'Technology' if seed % 3 else 'Financial Services',
        'rating': ['BUY', 'LEAN BUY', 'HOLD', 'PASS'][seed % 4],
        'price': 100.0, 'dcf_fv': 120.0, 'mos': float(rng.uniform(-0.3, 0.5)),
        '_composite_score': float(rng.uniform(10, 80)),
        '_gates_passed': '11/17', '_gates_passed_num': 11,
        'shares_out': 1e8, 'mcap': 1e10, 'revenue': 1e9, 'fcf': 1e8,
        'altman_z_zone': 'safe', 'beneish_flag': False,
        'edgar_quality_score': 80, '_data_coverage_score': 70,
        'avg_dollar_volume_3m': 5e6, 'tangible_book_ps': 10.0,
        # Both projected sub-keys matter: years_available drives the
        # thin-history rating cap, operating_income_history the pool-share
        # signal. The other series must NOT be needed by any scoring path.
        'edgar_history': {
            'years_available': 11,
            'operating_income_history': {str(y): 1e8 + y for y in range(2015, 2026)},
            'revenue_history': {str(y): 1e9 for y in range(2015, 2026)},
        },
    }
    for f in sorted({g.field for g in GATES}):
        r.setdefault(f, float(rng.uniform(0, 1)))
    return r


@pytest.fixture
def corpus(tmp_path):
    """Three snapshots plus price parquets, with the store already built."""
    d = tmp_path / 'out'
    (d / 'prices').mkdir(parents=True)
    bdays = pd.date_range('2025-06-01', '2026-09-01', freq='B')
    rng = np.random.default_rng(5)
    for t in TICKERS + ['SPY']:
        close = 100 + rng.standard_normal(len(bdays)).cumsum()
        df = pd.DataFrame({'Open': close, 'High': close, 'Low': close,
                           'Close': close, 'Volume': 1000}, index=bdays)
        df.index.name = 'Date'
        df.to_parquet(d / 'prices' / f'{t}.parquet')
    for n, day in enumerate(DATES):
        snap = {'date': day, 'risk_free_rate': 0.04,
                'risk_free_rate_source': 'fred', 'count': len(TICKERS),
                'provenance': {'run': day},
                'results': [_row(t, n * 100 + i) for i, t in enumerate(TICKERS)]}
        (d / f'results_{day}.json').write_text(json.dumps(snap), encoding='utf-8')
    ingest_dir(str(d))
    return d


@pytest.fixture(autouse=True)
def _restore_store_flag():
    original = bt.USE_SNAPSHOT_STORE
    yield
    bt.USE_SNAPSHOT_STORE = original


def _load(results_dir, use_store):
    bt.USE_SNAPSHOT_STORE = use_store
    return bt.load_corpus(str(results_dir))


def test_corpus_from_store_carries_the_same_scoring_fields(corpus):
    store_snaps = _load(corpus, True)
    json_snaps = _load(corpus, False)
    assert [s['date'] for s in store_snaps] == DATES
    assert [s['date'] for s in json_snaps] == DATES
    assert store_snaps[0]['risk_free_rate'] == 0.04
    s_row = {r['ticker']: r for r in store_snaps[0]['results']}['B00']
    j_row = {r['ticker']: r for r in json_snaps[0]['results']}['B00']
    for field in sorted({g.field for g in GATES}) + ['rating', 'mos', 'price',
                                                     '_composite_score', 'sector']:
        assert s_row[field] == pytest.approx(j_row[field]) if isinstance(
            j_row[field], float) else s_row[field] == j_row[field]
    # edgar_history arrives slim but with everything the caps/signals read
    assert s_row['edgar_history'] == {
        'years_available': 11,
        'operating_income_history': j_row['edgar_history']['operating_income_history'],
    }
    assert 'revenue_history' in j_row['edgar_history']


def test_rescoring_is_identical_through_either_path(corpus):
    """The calibration hot path: same params, same ratings and composites."""
    params = default_params()
    scored, caps = {}, {}
    for use_store in (True, False):
        snaps = _load(corpus, use_store)
        metrics = bt._evaluate_params_on_snapshots(snaps, params, [30])
        scored[use_store] = [
            (m['run_date'], m['horizon'],
             sorted((d['ticker'], d['rating'], d['_composite_score'])
                    for d in m['details']))
            for m in metrics]
        # the edgar_history-driven cap must fire the same way on both paths
        caps[use_store] = sorted(
            (r['ticker'], tuple(r.get('_rating_cap_reasons') or ()))
            for r in snaps[0]['results'])
    assert scored[True] and scored[True] == scored[False]
    assert caps[True] == caps[False]


def test_backtest_measurement_is_identical_through_either_path(corpus):
    class _NoNet:
        def fetch_history(self, *a, **k):
            return None

    runs = {}
    for use_store in (True, False):
        bt.USE_SNAPSHOT_STORE = use_store
        metrics = bt.run_backtest(str(corpus), [30, 90], _NoNet(),
                                  prices_dir=str(corpus / 'prices'), since=None)
        runs[use_store] = [
            (m['run_date'], m['horizon'], m['spy_return'], m['buckets'],
             sorted((d['ticker'], d['return'], d['excess_return'])
                    for d in m['details']))
            for m in sorted(metrics, key=lambda m: (m['run_date'], m['horizon']))]
    assert runs[True] and runs[True] == runs[False]


def test_falls_back_per_date_when_the_store_lags(corpus):
    """A store missing a date serves the ones it has and the JSON does the rest."""
    (corpus / 'results_2026-04-06.json').write_text(json.dumps({
        'date': '2026-04-06', 'count': 1,
        'results': [_row('B00', 7)]}), encoding='utf-8')
    snaps = _load(corpus, True)
    assert [s['date'] for s in snaps] == DATES + ['2026-04-06']
    assert len(snaps[-1]['results']) == 1


def test_no_store_flag_and_missing_store_use_json(corpus, tmp_path):
    assert [s['date'] for s in _load(corpus, False)] == DATES
    (corpus / 'snapshots.duckdb').unlink()
    assert [s['date'] for s in _load(corpus, True)] == DATES


def test_discover_dates_skips_replay_copies(corpus):
    (corpus / 'results_2026-02-05_replay.json').write_text('{}', encoding='utf-8')
    from datetime import date
    assert bt._discover_snapshot_dates(str(corpus)) == [
        date.fromisoformat(d) for d in DATES]


def test_forward_returns_bulk_matches_per_ticker(corpus):
    """The one-scan parquet path and the per-ticker path agree exactly."""
    class _NoNet:
        def fetch_history(self, *a, **k):
            return None

    pdir = str(corpus / 'prices')
    bulk = bt.fetch_forward_returns(TICKERS, '2026-01-05', 30, _NoNet(),
                                    prices_dir=pdir)
    import data.price_store as ps
    original = ps.window_closes
    try:                       # force the per-ticker fallback
        bt.window_closes = lambda *a, **k: None
        per_ticker = bt.fetch_forward_returns(TICKERS, '2026-01-05', 30,
                                              _NoNet(), prices_dir=pdir)
    finally:
        bt.window_closes = original
    assert set(bulk) == set(per_ticker) and len(bulk) > 5
    for t, v in per_ticker.items():
        assert bulk[t]['ret'] == pytest.approx(v['ret'], abs=1e-12)
        assert bulk[t]['start'] == pytest.approx(v['start'], abs=1e-12)
        assert bulk[t]['end'] == pytest.approx(v['end'], abs=1e-12)


class _RecordingClient:
    """yfinance stand-in that records which tickers reached the network path."""

    def __init__(self):
        self.asked = []

    def fetch_history(self, ticker, **kwargs):
        self.asked.append(ticker)
        return None


def test_local_parquet_is_settled_by_the_scan_never_refetched(corpus):
    """Stale prices (no bar near the eval date) must not send tickers that
    have a local parquet to yfinance — the per-ticker path never did."""
    client = _RecordingClient()
    # An eval date far past the last bar: the scan answers for nobody.
    got = bt.fetch_forward_returns(TICKERS, '2026-08-25', 365, client,
                                   prices_dir=str(corpus / 'prices'))
    assert got == {}
    assert client.asked == []


def test_ticker_without_a_parquet_still_falls_through_to_yfinance(corpus):
    client = _RecordingClient()
    bt.fetch_forward_returns(['NOPARQUET'], '2026-01-05', 30, client,
                             prices_dir=str(corpus / 'prices'))
    assert 'NOPARQUET' in client.asked


def test_unusable_bulk_query_falls_back_to_the_per_ticker_path(corpus, monkeypatch):
    """If the scan itself cannot run, every ticker takes the old path."""
    monkeypatch.setattr(bt, 'window_closes', lambda *a, **k: None)
    got = bt.fetch_forward_returns(TICKERS, '2026-01-05', 30, _RecordingClient(),
                                   prices_dir=str(corpus / 'prices'))
    assert len(got) > 5
