"""Resampling inference for the backtest (models/inference.py) and its wiring."""
import copy
from datetime import date, timedelta

import numpy as np
import pytest

from models.inference import (block_length, bootstrap_mean, ic_permutation_test,
                              permutation_p, stationary_bootstrap_indices)
from scripts import backtest as bt
from scripts.backtest_cloud import significance_notices


def _dates(n, start=date(2026, 7, 1)):
    return [start + timedelta(days=k) for k in range(n)]


class TestBlockBootstrap:

    def test_block_length_is_snapshots_per_period(self):
        # 90 daily snapshots over a 30-day horizon: 89 // 30 + 1 = 3 periods.
        assert block_length(_dates(90), 30) == 30
        assert block_length(_dates(1), 30) == 1

    def test_indices_continue_runs(self):
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(50, 10, 200, rng)
        assert idx.shape == (200, 50) and idx.min() >= 0 and idx.max() < 50
        steps = (idx[:, 1:] - idx[:, :-1]) % 50
        # Mean block 10 -> a jump on ~10% of steps, a continuation otherwise.
        assert 0.8 < np.mean(steps == 1) < 0.95

    def test_ci_covers_mean_and_is_reproducible(self):
        x = np.random.default_rng(1).normal(0.02, 0.05, 120)
        a = bootstrap_mean(x, 5, seed=3)
        assert a == bootstrap_mean(x, 5, seed=3)
        assert a['ci_low'] < a['mean'] < a['ci_high']
        assert a['mean'] == pytest.approx(float(x.mean()))
        assert a['n_blocks'] == 24

    def test_persistent_series_widens_the_interval(self):
        # The same values in long runs: an iid bootstrap would read 120
        # observations; blocks of 30 read four, and the interval widens.
        rng = np.random.default_rng(2)
        x = np.repeat(rng.normal(0, 0.05, 4), 30)
        iid = bootstrap_mean(x, 1, seed=0)
        blk = bootstrap_mean(x, 30, seed=0)
        assert (blk['ci_high'] - blk['ci_low']) > 3 * (iid['ci_high'] - iid['ci_low'])

    def test_clear_signal_is_significant_and_noise_is_not(self):
        rng = np.random.default_rng(4)
        assert bootstrap_mean(rng.normal(0.05, 0.02, 100), 2)['p_value'] < 0.01
        assert bootstrap_mean(rng.normal(0.0, 0.02, 100), 2)['p_value'] > 0.05

    def test_too_few_blocks(self):
        b = bootstrap_mean(np.ones(60) * 0.05, 30)
        assert b['n_blocks'] == 2 and b['ci_low'] is None and b['mean'] == pytest.approx(0.05)

    def test_degenerate_inputs(self):
        assert bootstrap_mean([0.1], 1) is None
        assert bootstrap_mean([None, float('nan'), 0.1], 1) is None


def _panels(n_dates, n_tk, signal, seed=0):
    """Scores fixed per ticker; returns = signal × score + noise per date."""
    rng = np.random.default_rng(seed)
    tickers = [f'T{k:03d}' for k in range(n_tk)]
    score = rng.normal(size=n_tk)
    out = []
    for _ in range(n_dates):
        ret = signal * score + rng.normal(size=n_tk)
        out.append((tickers, list(score), list(ret)))
    return out


class TestPermutation:

    def test_observed_matches_backtest_spearman(self):
        panels = _panels(3, 40, 0.3)
        expected = np.mean([bt._spearman(s, r) for _, s, r in panels])
        res = ic_permutation_test(panels, n_perm=20)
        assert res['observed'] == pytest.approx(expected)

    def test_signal_rejects_and_noise_does_not(self):
        assert ic_permutation_test(_panels(5, 200, 0.4), n_perm=99)['p_value'] <= 0.02
        noise = ic_permutation_test(_panels(5, 200, 0.0, seed=7), n_perm=99)
        assert noise['p_value'] > 0.05
        assert abs(noise['null_mean']) < 0.02

    def test_null_keeps_cross_date_dependence(self):
        # Fixed scores and returns repeated on every date: the IC series is
        # one observation copied. A per-date shuffle would give a null sd near
        # 1/sqrt(n·dates); the ticker permutation keeps it near 1/sqrt(n).
        rng = np.random.default_rng(5)
        tk = [f'T{k}' for k in range(100)]
        s, r = list(rng.normal(size=100)), list(rng.normal(size=100))
        res = ic_permutation_test([(tk, s, r)] * 20, n_perm=200)
        assert res['null_sd'] == pytest.approx(1 / np.sqrt(99), rel=0.3)

    def test_thin_and_empty(self):
        assert ic_permutation_test([]) is None
        assert ic_permutation_test([(['A', 'B'], [1, 2], [0.1, 0.2])]) is None
        assert permutation_p(0.5, [0.1, 0.6, 0.2]) == pytest.approx(2 / 4)
        assert permutation_p(None, [0.1]) is None


def _metric(run_date, tickers, scores, excess, ratings, horizon=30):
    return {'run_date': run_date, 'horizon': horizon, 'details': [
        {'ticker': t, '_composite_score': s, 'excess_return': e, 'rating': r}
        for t, s, e, r in zip(tickers, scores, excess, ratings, strict=True)]}


def _metrics(n_dates=60, n_tk=60, signal=0.5, seed=0):
    rng = np.random.default_rng(seed)
    tk = [f'T{k}' for k in range(n_tk)]
    score = rng.uniform(0, 100, n_tk)
    ratings = ['BUY' if s >= 57 else 'LEAN BUY' if s >= 39 else
               'HOLD' if s >= 25 else 'PASS' for s in score]
    out = []
    for d in _dates(n_dates):
        ex = signal * (score - 50) / 1000 + rng.normal(0, 0.02, n_tk)
        out.append(_metric(d.isoformat(), tk, list(score), list(ex), ratings))
    return out


class TestBacktestWiring:

    def test_composite_ic_carries_bootstrap(self):
        s = bt.composite_ic_summary(_metrics(n_dates=120))[30]
        b = s['bootstrap']
        assert b['mean'] == pytest.approx(s['mean_ic'])
        assert b['block'] == block_length(_dates(120), 30) and b['n_blocks'] == 4
        assert b['ci_low'] < s['mean_ic'] < b['ci_high']

    def test_one_period_gets_no_interval(self, capsys):
        # 16 daily snapshots at 90 days are one independent period: the
        # bootstrap would only reshuffle that episode, so it reports nothing.
        s = bt.composite_ic_summary([dict(m, horizon=90) for m in _metrics(n_dates=16)])[90]
        b = s['bootstrap']
        assert b['n_blocks'] == 1 and b['ci_low'] is None and b['p_value'] is None
        assert 'independent period' in b['reason']
        bt.print_composite_ic({90: s})
        assert 'n/a' in capsys.readouterr().out

    def test_permutation_by_horizon(self):
        res = bt.composite_ic_permutation(_metrics(n_dates=10), n_perm=30)
        assert set(res) == {30} and res[30]['p_value'] <= 0.05
        assert bt.composite_ic_permutation(_metrics(), n_perm=0) == {}

    def test_rating_inference_spread(self):
        inf = bt.rating_bucket_inference(_metrics())[30]
        assert inf['spread']['mean'] > 0
        assert inf['spread']['n_snapshots'] == 60
        assert set(inf) >= {'BUY', 'PASS', 'spread'}

    def test_summary_and_printers(self, capsys):
        m = _metrics(n_dates=10)
        perm = bt.composite_ic_permutation(m, n_perm=10)
        summary = bt.build_measure_summary(m, [30], None, '2026-10-28', {}, {},
                                           permutation=perm)
        assert summary['composite_ic_permutation']['30']['n_perm'] == 10
        assert 'spread' in summary['rating_bucket_inference']['30']
        bt.print_composite_ic(bt.composite_ic_summary(m), perm)
        bt.print_rating_inference(summary['rating_bucket_inference'])
        out = capsys.readouterr().out
        assert '90% CI (boot)' in out and 'BUY+LEAN BUY - PASS' in out


class TestNullCalibration:

    def _snaps(self):
        snaps = {}
        for k, d in enumerate(_dates(3)):
            snaps[d] = {'date': d.isoformat(), 'results': [
                {'ticker': f'T{j}', '_fwd': {30: {'excess_return': j + 10 * k}}}
                for j in range(5)]}
        return snaps

    def test_permute_keeps_paths_and_restores(self):
        snaps = self._snaps()
        before = copy.deepcopy(snaps)
        saved = bt._permute_forward_returns(snaps, np.random.default_rng(1))
        # One permutation for every date: the partner of T0 is the same ticker
        # on each, so the return path moves as a whole.
        partner = {(snaps[d]['results'][0]['_fwd'][30]['excess_return'] - 10 * k)
                   for k, d in enumerate(sorted(snaps))}
        assert len(partner) == 1
        assert sorted(r['_fwd'][30]['excess_return']
                      for r in snaps[min(snaps)]['results']) == [0, 1, 2, 3, 4]
        bt._restore_forward_returns(saved)
        assert snaps == before

    def test_null_reports_p_values(self, capsys):
        snaps = self._snaps()
        calls = []

        def run_windows():
            calls.append(1)
            return [{'train_objective': 0.1, 'test_objective': 0.0}], 0

        null = bt._null_calibration(snaps, run_windows, 4, 0,
                                    {'mean_train_objective': 0.2,
                                     'mean_test_objective': -0.1})
        assert len(calls) == 4 and null['runs'] == 4
        assert null['p_train'] == pytest.approx(1 / 5)
        assert null['p_test'] == pytest.approx(1.0)
        assert null['train_objective']['mean'] == 0.1


def test_compare_notices_a_significance_change():
    def summ(lo, hi):
        return {'composite_ic': {'30': {'mean_ic': (lo + hi) / 2, 'bootstrap': {
            'ci_low': lo, 'ci_high': hi, 'n_blocks': 6}}}}
    assert significance_notices(summ(0.01, 0.05), summ(-0.01, 0.04)) == [
        '30d composite IC went inconclusive -> positive (mean +0.030, 90% CI '
        '[+0.010, +0.050] over 6 independent period(s))']
    assert significance_notices(summ(0.01, 0.05), summ(0.02, 0.06)) == []
    assert significance_notices(summ(0.01, 0.05), {'composite_ic': {'30': {}}}) == []
    assert significance_notices(summ(0.01, 0.05), None) == []
    one = {'composite_ic': {'30': {'mean_ic': 0.1, 'bootstrap': {
        'ci_low': None, 'ci_high': None, 'n_blocks': 1}}}}
    assert significance_notices(one, summ(-0.01, 0.04)) == []
