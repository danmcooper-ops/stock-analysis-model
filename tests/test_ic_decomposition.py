"""Within- vs between-sector split of the composite IC (models/ic_decomposition.py)."""
from datetime import date, timedelta

import numpy as np
import pytest

from models.ic_decomposition import (MIN_GROUP_ROWS, decompose, snapshot_components,
                                     within_permutation_test)
from scripts import backtest as bt
from scripts.backtest_cloud import sector_ic_notices

SECTORS = [f'S{k}' for k in range(8)]


def _snapshot(rng, kind, per=40):
    """One cross-section where the score's skill is known by construction.

    ``within``: the score orders names inside each sector, and sector levels
    of score and return are unrelated. ``between``: the score is the sector's
    level alone, which also drives that sector's returns. ``noise``: neither.
    """
    sec, ind, s, r = [], [], [], []
    score_lvl = rng.normal(0, 10, len(SECTORS))
    ret_lvl = rng.normal(0, 0.05, len(SECTORS))
    for k, g in enumerate(SECTORS):
        x = rng.normal(0, 1, per)
        if kind == 'within':
            scores = score_lvl[k] + x
            rets = ret_lvl[k] + 0.02 * x + rng.normal(0, 0.01, per)
        elif kind == 'between':
            scores = np.full(per, 10.0 * k) + rng.normal(0, 0.01, per)
            rets = 0.01 * k + rng.normal(0, 0.03, per)
        else:
            scores = score_lvl[k] + x
            rets = ret_lvl[k] + rng.normal(0, 0.03, per)
        sec += [g] * per
        ind += [f'{g}-a'] * (per // 2) + [f'{g}-b'] * (per - per // 2)
        s += list(scores)
        r += list(rets)
    return sec, ind, s, r


class TestSnapshotComponents:

    def test_within_signal_shows_within_not_between(self):
        c = snapshot_components(*_snapshot(np.random.default_rng(0), 'within'))
        assert c['within_sector'] > 0.7 and c['within_industry'] > 0.7
        # Sector levels were drawn independently, so the neutral score beats
        # the raw one on the same raw returns.
        assert c['sector_neutral'] > c['total']
        assert c['neutral_minus_total'] == pytest.approx(c['sector_neutral'] - c['total'])
        assert set(c['per_sector']) == set(SECTORS)

    def test_between_signal_shows_between_not_within(self):
        c = snapshot_components(*_snapshot(np.random.default_rng(1), 'between'))
        # Sector return levels step 0.01 against ~0.005 of noise in a sector
        # mean, so neighbours can swap: near 1, not exactly 1.
        assert c['between_sector'] > 0.9
        assert abs(c['within_sector']) < 0.2
        assert c['total'] > 0.3 and c['neutral_minus_total'] < -0.2

    def test_small_groups_are_left_out_and_industries_fall_back(self):
        rng = np.random.default_rng(2)
        sec, ind, s, r = _snapshot(rng, 'within', per=MIN_GROUP_ROWS * 2)
        # Split one sector's rows into tiny industries: they fall back to the
        # sector, so within_industry still covers them.
        ind = [f'tiny{i}' if g == 'S0' else x for i, (g, x) in enumerate(zip(sec, ind, strict=True))]
        c = snapshot_components(sec, ind, s, r)
        assert c['within_industry'] is not None
        few = snapshot_components(['A'] * 12, ['a'] * 12, list(range(12)), list(range(12)))
        assert few['within_sector'] is None and few['between_sector'] is None
        assert few['total'] == pytest.approx(1.0)

    def test_missing_sector_and_too_few_rows(self):
        assert snapshot_components(['A'] * 5, [None] * 5, range(5), range(5)) is None
        c = snapshot_components([None] * 20, [None] * 20, range(20), range(20))
        assert c['within_sector'] is None and c['between_sector'] is None
        assert c['sector_neutral'] == pytest.approx(c['total'])


def _series(kinds, start=date(2026, 7, 1), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k, kind in enumerate(kinds):
        sec, ind, s, r = _snapshot(rng, kind)
        out.append((start + timedelta(days=k), sec, ind, s, r))
    return out


class TestDecompose:

    def test_periods_and_bootstrap(self):
        snaps = _series(['within'] * 45 + ['noise'] * 45)
        res = decompose([(d, snapshot_components(*x)) for d, *x in snaps], 30)
        assert [p['n_snapshots'] for p in res['by_period']] == [30, 30, 30]
        # The signal lives in the first 45 days, so period means fall off.
        p = [q['within_sector'] for q in res['by_period']]
        assert p[0] > 0.7 and p[2] < 0.2
        w = res['components']['within_sector']
        assert w['n_snapshots'] == 90 and w['n_blocks'] == 3
        assert set(res['per_sector']) == set(SECTORS)

    def test_empty(self):
        assert decompose([], 30) is None
        assert decompose([(date(2026, 7, 1), None)], 30) is None


class TestWithinPermutation:

    def _panels(self, kind, n=20):
        out = []
        for _, sec, _, s, r in _series([kind] * n, seed=5):
            tickers = [f'{g}{i % 40}' for i, g in enumerate(sec)]
            out.append((tickers, sec, s, r))
        return out

    def test_signal_beats_null_noise_does_not(self):
        sig = within_permutation_test(self._panels('within'), n_perm=40, seed=1)
        assert sig['observed'] > 0.7 and sig['p_value'] == pytest.approx(1 / 41)
        noise = within_permutation_test(self._panels('noise'), n_perm=40, seed=1)
        assert noise['p_value'] > 0.05

    def test_degenerate(self):
        assert within_permutation_test([], n_perm=10) is None
        assert within_permutation_test(self._panels('within', 2), n_perm=0) is None


def _metrics(kinds, horizon=30):
    out = []
    for d, sec, ind, s, r in _series(kinds, seed=9):
        out.append({'run_date': d.isoformat(), 'horizon': horizon, 'details': [
            {'ticker': f'{g}{i % 40}', 'sector': g, 'industry': x,
             '_composite_score': sc, 'excess_return': ex, 'rating': 'HOLD'}
            for i, (g, x, sc, ex) in enumerate(zip(sec, ind, s, r, strict=True))]})
    return out


class TestBacktestWiring:

    def test_breakdown_and_summary(self):
        m = _metrics(['within'] * 61)
        sic = bt.sector_ic_breakdown(m, n_perm=20)
        assert sic[30]['components']['within_sector']['mean'] > 0.7
        assert sic[30]['within_permutation']['n_perm'] == 20
        summary = bt.build_measure_summary(m, [30], None, '2026-11-04', {}, {})
        assert summary['sector_ic']['30']['components']['total']['n_snapshots'] == 61
        assert 'within_permutation' not in summary['sector_ic']['30']
        bt.print_sector_ic(sic)   # smoke: every branch formats

    def test_rows_without_score_or_return_are_dropped(self):
        m = _metrics(['within'] * 3)
        m[0]['details'][0].update(_composite_score=None)
        m[1]['details'][0].update(excess_return=float('nan'))
        sic = bt.sector_ic_breakdown(m)
        assert sic[30]['components']['total']['n_snapshots'] == 3


def test_compare_notices_a_sector_ic_change():
    def summ(lo, hi, key='between_sector'):
        return {'sector_ic': {'30': {'components': {key: {
            'mean': (lo + hi) / 2, 'ci_low': lo, 'ci_high': hi, 'n_blocks': 4}}}}}
    assert sector_ic_notices(summ(0.01, 0.05), summ(-0.01, 0.04)) == [
        '30d between-sector IC went inconclusive -> positive (mean +0.030, '
        '90% CI [+0.010, +0.050] over 4 independent period(s))']
    assert sector_ic_notices(summ(0.01, 0.05), summ(0.02, 0.06)) == []
    assert sector_ic_notices(summ(0.01, 0.05), None) == []
    assert sector_ic_notices(summ(-0.05, -0.01, 'neutral_minus_total'),
                             summ(-0.01, 0.04, 'neutral_minus_total'))[0].startswith(
        '30d sector-neutral gain over the headline IC went inconclusive -> negative')
    no_ci = {'sector_ic': {'30': {'components': {'between_sector': {
        'mean': 0.1, 'ci_low': None, 'ci_high': None, 'n_blocks': 1}}}}}
    assert sector_ic_notices(no_ci, summ(-0.01, 0.04)) == []
