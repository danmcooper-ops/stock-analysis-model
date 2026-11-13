"""The DCF simulation's diagnostics (2026-11-13): evidence-based growth sigma,
fair-value quantiles with the price's position in them, and attribution of
the spread to its inputs."""
import numpy as np
import pytest

from models.dcf import MC_DCF_DRIVERS, monte_carlo_dcf
from models.montecarlo import (
    GROWTH_SIGMA_CAP, GROWTH_SIGMA_FLOOR, QUANTILE_LEVELS,
    evidence_growth_sigma, fv_quantiles, growth_sigma, quantile_rank, sobol_indices,
)

BENIGN = (1e9, 0.10, 0.09, 0.03, 5e9, 1e9)   # base_fcf, g, wacc, tg, net_debt, shares


class TestEvidenceGrowthSigma:
    def test_disagreement_is_standard_error_of_weighted_mean(self):
        sigma, basis, parts = evidence_growth_sigma(0.08, [0.04, 0.12], [1, 1])
        assert basis == 'disagreement'
        assert sigma == pytest.approx(0.04 / np.sqrt(2))
        assert parts == {'disagreement': pytest.approx(0.04 / np.sqrt(2))}
        # Unequal weights: fewer effective signals, larger standard error.
        lopsided, _, _ = evidence_growth_sigma(0.08, [0.04, 0.12], [9, 1])
        spread = np.sqrt(0.9 * 0.1) * 0.08
        assert lopsided == pytest.approx(spread * np.sqrt(0.82))

    def test_history_is_revenue_vol_over_sqrt_years(self):
        sigma, basis, _ = evidence_growth_sigma(0.08, rev_growth_vol=0.15, years=5)
        assert basis == 'history'
        assert sigma == pytest.approx(0.15 / np.sqrt(5))

    def test_parts_combine_in_quadrature(self):
        sigma, basis, _ = evidence_growth_sigma(0.08, [0.02, 0.08], [1, 1], rev_growth_vol=0.04 * np.sqrt(5))
        assert basis == 'disagreement+history'
        assert sigma == pytest.approx(np.hypot(0.03 / np.sqrt(2), 0.04))

    def test_independent_of_growth_level(self):
        """The old ratio rule scaled with |g|; evidence does not."""
        lo, _, _ = evidence_growth_sigma(0.02, [0.00, 0.04], [1, 1], rev_growth_vol=0.1)
        hi, _, _ = evidence_growth_sigma(0.30, [0.28, 0.32], [1, 1], rev_growth_vol=0.1)
        assert lo == pytest.approx(hi)

    def test_floor_cap_and_outlier_history(self):
        assert evidence_growth_sigma(0.05, [0.05, 0.05], [1, 1])[0] == GROWTH_SIGMA_FLOOR
        # A 1,400% revenue-growth volatility (acquisition jump) is capped.
        assert evidence_growth_sigma(0.05, rev_growth_vol=14.0)[0] == GROWTH_SIGMA_CAP
        assert evidence_growth_sigma(0.05, [-0.9, 0.9], [1, 1])[0] == GROWTH_SIGMA_CAP

    def test_falls_back_to_ratio_rule_without_evidence(self):
        for signals in (None, [0.1], [0.1, None]):
            sigma, basis, parts = evidence_growth_sigma(0.25, signals, [1, 1])
            assert basis == 'ratio' and sigma == growth_sigma(0.25) and parts == {}

    def test_ignores_non_finite_and_zero_weight_signals(self):
        sigma, basis, _ = evidence_growth_sigma(0.08, [0.04, 0.12, float('nan'), 0.9], [1, 1, 1, 0])
        assert basis == 'disagreement' and sigma == pytest.approx(0.04 / np.sqrt(2))


class TestQuantileRank:
    Q = [float(i) for i in range(21)]   # value i at QUANTILE_LEVELS[i]

    def test_exact_and_interpolated(self):
        assert quantile_rank(self.Q, 10.0) == pytest.approx(0.50)
        assert quantile_rank(self.Q, 10.5) == pytest.approx(0.525)
        assert quantile_rank(self.Q, 0.5) == pytest.approx(0.03)

    def test_tails_report_the_outermost_level(self):
        assert quantile_rank(self.Q, -5.0) == 0.01
        assert quantile_rank(self.Q, 99.0) == 0.99

    def test_flat_run_returns_its_middle(self):
        q = [0.0] * 5 + [float(i) for i in range(1, 17)]   # 1%..20% clamped to zero
        assert quantile_rank(q, 0.0) == pytest.approx(np.mean(QUANTILE_LEVELS[:5]))

    def test_missing_inputs(self):
        assert quantile_rank(None, 1.0) is None
        assert quantile_rank(self.Q, None) is None
        assert quantile_rank(self.Q[:-1], 1.0) is None

    def test_round_trip_with_fv_quantiles(self):
        vals = np.random.default_rng(0).lognormal(3, 0.5, 10_000)
        q = fv_quantiles(vals)
        for x in (15.0, 20.0, 30.0):
            assert quantile_rank(q, x) == pytest.approx(np.mean(vals < x), abs=0.01)


class TestSobolIndices:
    def test_additive_function_splits_by_variance(self):
        first, total = sobol_indices(lambda u: u[:, 0] + 2 * u[:, 1] + 0 * u[:, 2], 4096, 3, seed=1)
        assert first == pytest.approx([0.2, 0.8, 0.0], abs=0.02)
        assert total == pytest.approx([0.2, 0.8, 0.0], abs=0.02)

    def test_interaction_shows_in_total_not_first_order(self):
        first, total = sobol_indices(lambda u: (u[:, 0] - .5) * (u[:, 1] - .5), 4096, 2, seed=1)
        assert first == pytest.approx([0, 0], abs=0.05)
        assert total == pytest.approx([1, 1], abs=0.05)

    def test_constant_output(self):
        assert sobol_indices(lambda u: np.ones(len(u)), 256, 2) == (None, None)


class TestMonteCarloDcfDiagnostics:
    def test_attribution_leaves_main_draws_unchanged(self):
        kw = dict(n_iterations=1024, seed=7, base_ebitda=1.5e9, exit_multiple=10)
        off = monte_carlo_dcf(*BENIGN, **kw)
        on = monte_carlo_dcf(*BENIGN, attribution=True, **kw)
        assert off['drivers'] is None and on['drivers'] is not None
        for k in ('median_fv', 'p10_fv', 'p90_fv', 'cv', 'quantiles', 'clip_rate'):
            assert on[k] == off[k], k

    def test_drivers_sum_to_one_and_name_the_inputs(self):
        with_exit = monte_carlo_dcf(*BENIGN, base_ebitda=1.5e9, exit_multiple=10, attribution=True)
        assert set(with_exit['drivers']) == set(MC_DCF_DRIVERS)
        assert sum(with_exit['drivers'].values()) == pytest.approx(1.0)
        ggm = monte_carlo_dcf(*BENIGN, attribution=True)
        assert set(ggm['drivers']) == set(MC_DCF_DRIVERS[:3])

    def test_dominant_input_is_attributed(self):
        res = monte_carlo_dcf(*BENIGN, growth_sigma=0.08, wacc_sigma=1e-9, tg_sigma=1e-9,
                              attribution=True)
        assert max(res['drivers'], key=res['drivers'].get) == 'growth'
        assert res['drivers']['growth'] > 0.9
        res = monte_carlo_dcf(*BENIGN, growth_sigma=1e-9, wacc_sigma=0.02, tg_sigma=1e-9,
                              attribution=True)
        assert max(res['drivers'], key=res['drivers'].get) == 'discount_rate'

    def test_quantiles_are_ordered_and_match_the_percentiles(self):
        res = monte_carlo_dcf(*BENIGN, n_iterations=1024)
        q = res['quantiles']
        assert len(q) == len(QUANTILE_LEVELS) and q == sorted(q)
        assert q[QUANTILE_LEVELS.index(0.5)] == pytest.approx(res['median_fv'])
        assert q[QUANTILE_LEVELS.index(0.1)] == pytest.approx(res['p10_fv'])
        assert q[QUANTILE_LEVELS.index(0.9)] == pytest.approx(res['p90_fv'])
        assert res['growth_sigma'] == pytest.approx(growth_sigma(0.10))


class TestPipelineFields:
    def test_blend_rescales_quantiles(self):
        from scripts.analyze_stock import _rescale_fv_band
        row = {'mc_p10_fv': 10.0, 'mc_p90_fv': 30.0, 'mc_quantiles': [float(i) for i in range(21)]}
        _rescale_fv_band(row, 0.5)
        assert row['mc_quantiles'] == [i * 0.5 for i in range(21)]
        assert row['mc_p10_fv'] == 5.0

    def test_price_position_and_implied_growth_z(self):
        from scripts.analyze_stock import _set_mc_price_position
        row = {'price': 10.0, 'mc_quantiles': [float(i) for i in range(21)],
               'implied_vs_estimated': 0.06, 'mc_growth_sigma': 0.04}
        _set_mc_price_position(row)
        assert row['mc_price_pctile'] == pytest.approx(0.5)
        assert row['mc_prob_above_price'] == pytest.approx(0.5)
        assert row['implied_growth_z'] == pytest.approx(1.5)

    def test_price_position_without_simulation(self):
        from scripts.analyze_stock import _set_mc_price_position
        row = {'price': 10.0}
        _set_mc_price_position(row)
        assert row['mc_price_pctile'] is None and row['mc_prob_above_price'] is None
        assert row['implied_growth_z'] is None

    def test_forward_dcf_reports_sigma_basis_and_drivers(self):
        from scripts.analyze_stock import run_forward_dcf
        from tests.test_monte_carlo import _dcf_yf_data
        _, _, _, _, mc = run_forward_dcf(_dcf_yf_data(), wacc=0.10, rev_growth_vol=0.2)
        assert mc['growth_sigma_basis'] in ('history', 'disagreement+history')
        assert mc['drivers'] and sum(mc['drivers'].values()) == pytest.approx(1.0)

    def test_evidence_sigma_is_recorded_but_off_by_default(self):
        from scripts.analyze_stock import run_forward_dcf
        from scripts.config import MC_GROWTH_SIGMA_EVIDENCE
        from tests.test_monte_carlo import _dcf_yf_data
        assert MC_GROWTH_SIGMA_EVIDENCE is False
        _, _, g, _, mc = run_forward_dcf(_dcf_yf_data(), wacc=0.10, rev_growth_vol=0.2)
        assert mc['growth_sigma'] == pytest.approx(growth_sigma(g))
        assert mc['growth_sigma_evidence'] != pytest.approx(mc['growth_sigma'])
        assert 'history' in mc['growth_sigma_parts']

    def test_evidence_sigma_sets_the_simulation_when_on(self, monkeypatch):
        import scripts.analyze_stock as az
        from tests.test_monte_carlo import _dcf_yf_data
        monkeypatch.setattr(az, 'MC_GROWTH_SIGMA_EVIDENCE', True)
        _, _, _, _, mc = az.run_forward_dcf(_dcf_yf_data(), wacc=0.10, rev_growth_vol=0.2)
        assert mc['growth_sigma'] == pytest.approx(mc['growth_sigma_evidence'])
