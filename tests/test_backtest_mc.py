"""DCF Monte Carlo diagnostics in the backtest: the two new signals and the
fair-value range check (mc_band_coverage)."""
import json
import math

import pytest

from models.montecarlo import QUANTILE_LEVELS
from scripts import backtest as bt

# A fair-value distribution 50..150 linear in the quantile level:
# P5 = 55, P25 = 75, P50 = 100, P75 = 125, P95 = 145.
Q = [50 + 100 * lvl for lvl in QUANTILE_LEVELS]


def _metrics(rows, horizon=30, run_date='2026-11-13'):
    """rows: (ticker, start, end, info) → one analyze_run-shaped result."""
    return {
        'run_date': run_date,
        'horizon': horizon,
        'details': [{'ticker': t, 'start_price': p0, 'end_price': p1,
                     'excess_return': p1 / p0 - 1} for t, p0, p1, _ in rows],
        '_source_stocks': {t: info for t, _, _, info in rows},
    }


def test_signals_registered():
    keys = [k for k, _ in bt.SIGNAL_SPECS]
    assert 'mc_price_pctile' in keys and 'implied_growth_z' in keys


def test_band_shares_and_pull():
    info = {'mc_quantiles': Q}
    rows = [
        ('A', 100, 100, info),   # inside both bands at start and end
        ('B', 70, 80, info),     # 90% only → both bands; moved toward median
        ('C', 140, 160, info),   # 90% only → outside both; moved away
        ('D', 40, 45, info),     # below both, pinned at the 1% tail: not moved
    ]
    cov = bt.mc_band_coverage([_metrics(rows)])[30]
    assert cov['n_obs'] == 4 and cov['n_snapshots'] == 1
    assert cov['start_in'] == {'50': 0.25, '90': 0.75}
    assert cov['end_in'] == {'50': 0.5, '90': 0.5}
    assert cov['toward_median'] == 0.25
    assert cov['comparable_to_nominal'] is False
    assert cov['nominal']['50'] == pytest.approx(0.5)
    assert cov['nominal']['90'] == pytest.approx(0.9)
    assert cov['evidence'] is None   # no sigmas on these rows


def test_rows_without_quantiles_are_skipped():
    rows = [('A', 100, 110, {'mc_quantiles': None}),
            ('B', 100, 110, {'mc_quantiles': Q[:5]}),
            ('C', 100, 110, {'mc_quantiles': Q[:-1] + [float('nan')]})]
    assert bt.mc_band_coverage([_metrics(rows)]) == {}


def test_store_json_text_is_decoded():
    info = {'mc_quantiles': json.dumps(Q)}
    cov = bt.mc_band_coverage([_metrics([('A', 100, 100, info)])])
    assert cov[30]['n_obs'] == 1


def test_evidence_width_factor():
    # Growth carries all the variance: the band scales with the sigma ratio.
    assert bt.evidence_width_factor({'mc_growth_sigma': 0.02,
                                     'mc_growth_sigma_evidence': 0.05,
                                     'mc_drivers': {'growth': 1.0}}) == pytest.approx(2.5)
    # Growth carries none: nothing changes.
    assert bt.evidence_width_factor({'mc_growth_sigma': 0.02,
                                     'mc_growth_sigma_evidence': 0.05,
                                     'mc_drivers': {'growth': 0.0}}) == pytest.approx(1.0)
    # Half: sqrt(0.5 + 0.5 * 6.25).
    assert bt.evidence_width_factor({'mc_growth_sigma': 0.02,
                                     'mc_growth_sigma_evidence': 0.05,
                                     'mc_drivers': json.dumps({'growth': 0.5})}
                                    ) == pytest.approx(math.sqrt(3.625))
    assert bt.evidence_width_factor({'mc_growth_sigma': 0.02,
                                     'mc_drivers': {'growth': 0.5}}) is None


def test_evidence_band_is_wider():
    info = {'mc_quantiles': Q, 'mc_growth_sigma': 0.02,
            'mc_growth_sigma_evidence': 0.04, 'mc_drivers': {'growth': 1.0}}
    # k = 2: the 50% band becomes 50..150, the 90% band 10..190.
    rows = [('A', 140, 180, info), ('B', 100, 20, info)]
    cov = bt.mc_band_coverage([_metrics(rows, horizon=365)])[365]
    assert cov['comparable_to_nominal'] is True
    assert cov['end_in'] == {'50': 0.0, '90': 0.0}
    ev = cov['evidence']
    assert ev['n_obs'] == 2 and ev['median_width_factor'] == pytest.approx(2.0)
    assert ev['start_in'] == {'50': 1.0, '90': 1.0}
    assert ev['end_in'] == {'50': 0.0, '90': 1.0}


def test_widened_floors_at_zero():
    w = bt._widened(Q, 5.0)
    assert min(w) == 0.0 and w[bt._MC_Q[0.5]] == pytest.approx(100.0)


def test_summary_carries_signals_and_band(monkeypatch):
    info_lo = {'mc_quantiles': Q, 'mc_price_pctile': 0.1, 'implied_growth_z': -1.0}
    info_hi = {'mc_quantiles': Q, 'mc_price_pctile': 0.9, 'implied_growth_z': 1.0}
    rows = [(f'L{i}', 100, 110, info_lo) for i in range(12)] + \
           [(f'H{i}', 100, 95, info_hi) for i in range(12)]
    metrics = [_metrics(rows)]
    sig = bt.signal_summary(metrics)['30']
    assert sig['mc_price_pctile']['spread'] < 0
    assert sig['implied_growth_z']['spread'] < 0
    band = bt.mc_band_coverage(metrics)
    assert band[30]['n_obs'] == 24
    bt.print_mc_band_coverage(band)       # smoke: formats without error
    bt.print_mc_band_coverage({})


# --- DCF growth model v1 / v2 are measured apart ------------------------------

def test_growth_model_of_reads_dcf_inputs_in_any_form():
    assert bt.growth_model_of({}) == 'v1'
    assert bt.growth_model_of({'dcf_inputs': {'model': 'v2'}}) == 'v2'
    assert bt.growth_model_of({'dcf_inputs': json.dumps({'model': 'v2'})}) == 'v2'
    assert bt.growth_model_of({'dcf_inputs': 'not json'}) == 'v1'
    assert bt.growth_model_of({'dcf_inputs': {'model': None}}) == 'v1'


def _mixed_metrics():
    # v1 rows: price inside the band; v2 rows (same quantiles here) end far
    # outside it, so pooling would blend the two.
    v1 = {'mc_quantiles': Q, 'mc_price_pctile': 0.1, 'implied_growth_z': -1.0}
    v2 = {'mc_quantiles': Q, 'mc_price_pctile': 0.9, 'implied_growth_z': 1.0,
          'dcf_growth_model': 'v2'}
    rows = [(f'A{i}', 100, 100, v1) for i in range(24)] + \
           [(f'B{i}', 100, 300, v2) for i in range(24)]
    return [_metrics(rows)]


def test_band_coverage_is_per_model():
    m = _mixed_metrics()
    v1, v2 = bt.mc_band_coverage(m)[30], bt.mc_band_coverage(m, 'v2')[30]
    assert v1['n_obs'] == 24 and v1['end_in']['50'] == 1.0
    assert v2['n_obs'] == 24 and v2['end_in']['90'] == 0.0
    assert bt.growth_models(m) == ['v1', 'v2']


def test_evidence_widening_is_v1_only():
    info = {'mc_quantiles': Q, 'mc_growth_sigma': 0.02, 'mc_growth_sigma_evidence': 0.06,
            'mc_drivers': {'growth': 0.6}, 'dcf_growth_model': 'v2'}
    cov = bt.mc_band_coverage([_metrics([('A', 100, 100, info)])], 'v2')
    assert cov[30]['evidence'] is None


def test_signals_and_summary_split_by_model():
    m = _mixed_metrics()
    sig = bt.signal_summary(m)['30']
    assert {'mc_price_pctile', 'mc_price_pctile_v2', 'implied_growth_z_v2'} <= set(sig)
    assert sig['mc_price_pctile']['n_obs'] == 24 and sig['mc_price_pctile_v2']['n_obs'] == 24
    # v1-only history (no v2 rows): the summary keeps its old shape.
    only_v1 = [_metrics([(f'A{i}', 100, 100, {'mc_quantiles': Q, 'mc_price_pctile': 0.5})
                         for i in range(24)])]
    assert 'mc_price_pctile_v2' not in bt.signal_summary(only_v1)['30']
    bt.print_signal_tables(m, 'smoke')    # formats both models without error
    for model in bt.growth_models(m):
        bt.print_mc_band_coverage(bt.mc_band_coverage(m, model), model)


def test_weekly_job_loads_the_growth_model():
    import pathlib
    sh = pathlib.Path(__file__).resolve().parents[1] / 'scheduled-tasks/weekly-backtest/mac_run.sh'
    line = next(x for x in sh.read_text(encoding='utf-8').splitlines()
                if x.startswith('WEEKLY_ENV_KEYS='))
    assert 'DCF_GROWTH_MODEL' in line and 'TIINGO_API_KEY' in line
