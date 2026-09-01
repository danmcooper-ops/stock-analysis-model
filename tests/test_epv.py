# tests/test_epv.py
import pytest


import math

import pandas as pd

from models.epv import (earnings_power_value, earnings_power_value_valuation,
                        epv_with_growth_premium)
from models.ratios import effective_tax_rate


# ---------------------------------------------------------------------------
# earnings_power_value
# ---------------------------------------------------------------------------

class TestEarningsPowerValue:
    def test_basic_positive(self):
        """EBIT > 0, valid WACC -> positive EPV."""
        fv = earnings_power_value(ebit=12e9, tax_rate=0.20,
                                   cost_of_capital=0.10,
                                   shares_outstanding=1e9)
        assert fv is not None
        assert fv > 0

    def test_known_value(self):
        """EBIT=10, tax=20%, WACC=10%, 1 share, no cash -> 80."""
        fv = earnings_power_value(ebit=10, tax_rate=0.20,
                                   cost_of_capital=0.10,
                                   shares_outstanding=1)
        assert fv == pytest.approx(80.0)

    def test_none_on_negative_ebit(self):
        """Negative EBIT -> None."""
        assert earnings_power_value(-5e9, 0.20, 0.10, 1e9) is None

    def test_none_on_zero_ebit(self):
        """Zero EBIT -> None."""
        assert earnings_power_value(0, 0.20, 0.10, 1e9) is None

    def test_none_on_zero_shares(self):
        """Zero shares -> None."""
        assert earnings_power_value(10e9, 0.20, 0.10, 0) is None

    def test_none_on_zero_wacc(self):
        """Zero cost of capital -> None."""
        assert earnings_power_value(10e9, 0.20, 0, 1e9) is None

    def test_higher_ebit_gives_higher_epv(self):
        """Higher EBIT -> higher EPV."""
        fv_low = earnings_power_value(5e9, 0.20, 0.10, 1e9)
        fv_high = earnings_power_value(10e9, 0.20, 0.10, 1e9)
        assert fv_high > fv_low

    def test_excess_cash_increases_value(self):
        """Excess cash adds to EPV."""
        fv_no_cash = earnings_power_value(10e9, 0.20, 0.10, 1e9, excess_cash=0)
        fv_with_cash = earnings_power_value(10e9, 0.20, 0.10, 1e9, excess_cash=5e9)
        assert fv_with_cash > fv_no_cash

    def test_tax_rate_none_defaults(self):
        """None tax rate defaults to 21%."""
        fv = earnings_power_value(10, None, 0.10, 1)
        expected = 10 * (1 - 0.21) / 0.10
        assert fv == pytest.approx(expected)

    def test_tax_rate_clamped(self):
        """Extreme tax rates clamped to [0, 0.50]."""
        fv = earnings_power_value(10, 0.80, 0.10, 1)
        # clamped to 0.50
        expected = 10 * (1 - 0.50) / 0.10
        assert fv == pytest.approx(expected)


# ---------------------------------------------------------------------------
# epv_with_growth_premium
# ---------------------------------------------------------------------------

class TestEPVGrowthPremium:
    def test_return_above_cost_gives_premium(self):
        """Return on capital > its cost -> growth premium -> value above base."""
        base = 100.0
        result = epv_with_growth_premium(base, return_rate=0.20, discount_rate=0.10)
        assert result > base

    def test_return_below_cost_returns_base(self):
        """Return < its cost -> no growth premium, returns base EPV."""
        base = 100.0
        result = epv_with_growth_premium(base, return_rate=0.05, discount_rate=0.10)
        assert result == base  # multiplier floored at 1.0

    def test_none_on_none_epv(self):
        """None base EPV -> None."""
        assert epv_with_growth_premium(None, 0.15, 0.10) is None

    def test_negative_roe_returns_base(self):
        """Negative ROE -> returns base."""
        result = epv_with_growth_premium(100.0, return_rate=-0.10, discount_rate=0.10)
        assert result == 100.0

    def test_none_on_zero_re(self):
        """Zero cost of equity -> None."""
        assert epv_with_growth_premium(100.0, 0.15, 0) is None

    def test_growth_premium_capped(self):
        """Very high ROE -> multiplier capped at 2x.

        Tightened from 3× to 2× — a sustained 3× ROE-to-Re ratio is rare
        moat territory that EPV's zero-growth premise can't model honestly,
        so the cap holds the model to its "floor" purpose.
        """
        result = epv_with_growth_premium(100.0, return_rate=0.50, discount_rate=0.10)
        assert result == pytest.approx(200.0)  # capped at 2x


# ---------------------------------------------------------------------------
# EPV input selection — normalized (through-cycle) vs point-in-time EBIT
# ---------------------------------------------------------------------------

class TestSelectEpvEbit:
    def _select(self, *args, **kwargs):
        from scripts.analyze_stock import _select_epv_ebit
        return _select_epv_ebit(*args, **kwargs)

    def test_normalized_with_enough_history(self):
        ebit, source = self._select(point_ebit=200.0, yf_revenue=1000.0,
                                    op_margin_avg_10y=0.15,
                                    op_margin_hist_years=8)
        assert source == 'normalized'
        assert ebit == pytest.approx(150.0)  # 15% avg margin × current rev

    def test_point_when_history_thin(self):
        ebit, source = self._select(point_ebit=200.0, yf_revenue=1000.0,
                                    op_margin_avg_10y=0.15,
                                    op_margin_hist_years=4)
        assert source == 'point'
        assert ebit == 200.0

    def test_point_when_no_revenue(self):
        """No yfinance revenue → can't normalize in the quote currency;
        EDGAR USD revenue must never be substituted."""
        ebit, source = self._select(point_ebit=200.0, yf_revenue=None,
                                    op_margin_avg_10y=0.15,
                                    op_margin_hist_years=10)
        assert source == 'point'
        assert ebit == 200.0

    def test_normalized_used_even_when_point_ebit_missing(self):
        """Missing point EBIT must NOT drop EPV when the through-cycle path is
        available — the consensus fallback needs exactly these sparse names."""
        ebit, source = self._select(point_ebit=None, yf_revenue=1000.0,
                                    op_margin_avg_10y=0.15,
                                    op_margin_hist_years=10)
        assert source == 'normalized'
        assert ebit == pytest.approx(150.0)

    def test_none_when_no_point_ebit_and_no_history(self):
        ebit, source = self._select(point_ebit=None, yf_revenue=1000.0,
                                    op_margin_avg_10y=None,
                                    op_margin_hist_years=0)
        assert ebit is None
        assert source is None

    def test_negative_normalized_ebit_flows_through(self):
        """A negative through-cycle margin produces negative normalized EBIT;
        earnings_power_value then correctly returns None downstream."""
        ebit, source = self._select(point_ebit=50.0, yf_revenue=1000.0,
                                    op_margin_avg_10y=-0.05,
                                    op_margin_hist_years=10)
        assert source == 'normalized'
        assert ebit == pytest.approx(-50.0)
        assert earnings_power_value(ebit, 0.21, 0.10, 1e6) is None


class TestEquityBridge:
    """NOPAT/WACC is enterprise value; the cash/debt bridge converts to equity."""

    def test_debt_decreases_value(self):
        fv_no_debt = earnings_power_value(10e9, 0.20, 0.10, 1e9, total_debt=0)
        fv_with_debt = earnings_power_value(10e9, 0.20, 0.10, 1e9,
                                            total_debt=20e9)
        assert fv_with_debt == pytest.approx(fv_no_debt - 20.0)

    def test_known_value(self):
        # NOPAT = 10*0.8 = 8; EV = 8/0.10 = 80; equity = 80 + 5 - 30 = 55
        fv = earnings_power_value(10, 0.20, 0.10, 1,
                                  excess_cash=5, total_debt=30)
        assert fv == pytest.approx(55.0)

    def test_none_when_debt_exceeds_enterprise_value(self):
        assert earnings_power_value(10, 0.20, 0.10, 1, total_debt=100) is None


    def test_nan_inputs_return_none(self):
        """A NaN return or discount rate must not propagate NaN downstream."""
        assert epv_with_growth_premium(100.0, float('nan'), 0.10) is None
        assert epv_with_growth_premium(100.0, 0.15, float('nan')) is None


# ---------------------------------------------------------------------------
# Envelope: input validation, caveats and confidence
# ---------------------------------------------------------------------------

class TestEnvelopeValidation:
    def test_nan_tax_rate_is_rejected_not_zeroed(self):
        """max(0, min(nan, 0.5)) silently read as 0% tax before."""
        v = earnings_power_value_valuation(10, float('nan'), 0.10, 1)
        assert v.value is None
        assert v.confidence == 0.0
        assert any('tax_rate' in w for w in v.warnings)

    def test_nan_debt_is_rejected_not_propagated(self):
        v = earnings_power_value_valuation(10, 0.2, 0.10, 1, total_debt=float('nan'))
        assert v.value is None
        assert any('total_debt' in w for w in v.warnings)

    def test_nan_cash_is_rejected(self):
        v = earnings_power_value_valuation(10, 0.2, 0.10, 1, excess_cash=float('nan'))
        assert v.value is None

    def test_value_is_never_nan(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            for kw in ('tax_rate', 'excess_cash', 'total_debt'):
                kwargs = {'tax_rate': 0.2, 'excess_cash': 0, 'total_debt': 0}
                kwargs[kw] = bad
                v = earnings_power_value_valuation(10, kwargs['tax_rate'], 0.10, 1,
                                                   excess_cash=kwargs['excess_cash'],
                                                   total_debt=kwargs['total_debt'])
                assert v.value is None or math.isfinite(v.value)

    def test_negative_net_debt_is_a_valid_bridge(self):
        """Net-cash firms pass a negative net debt through total_debt."""
        v = earnings_power_value_valuation(10, 0.2, 0.10, 1, total_debt=-20)
        assert v.value == pytest.approx(100.0)
        assert v.confidence == 1.0
        assert v.warnings == ()

    def test_missing_tax_rate_is_a_caveat(self):
        v = earnings_power_value_valuation(10, None, 0.10, 1)
        assert v.value == pytest.approx(10 * 0.79 / 0.10)
        assert len(v.warnings) == 1
        assert 'statutory' in v.warnings[0]
        assert v.confidence == pytest.approx(0.85)

    def test_clamped_tax_rate_is_a_caveat(self):
        v = earnings_power_value_valuation(10, 0.80, 0.10, 1)
        assert v.value == pytest.approx(10 * 0.5 / 0.10)
        assert v.inputs_used['tax_rate'] == 0.50
        assert any('clamped' in w for w in v.warnings)
        assert v.confidence == pytest.approx(0.85)

    def test_point_ebit_is_a_caveat(self):
        v = earnings_power_value_valuation(10, 0.2, 0.10, 1, ebit_source='point')
        assert v.value == pytest.approx(80.0)
        assert any('point-in-time' in w for w in v.warnings)
        assert v.confidence == pytest.approx(0.85)
        assert v.inputs_used['ebit_source'] == 'point'

    def test_normalized_ebit_is_not_a_caveat(self):
        v = earnings_power_value_valuation(10, 0.2, 0.10, 1, ebit_source='normalized')
        assert v.warnings == ()
        assert v.confidence == 1.0

    def test_confidence_steps_and_floors(self):
        v = earnings_power_value_valuation(10, None, 0.10, 1, ebit_source='point')
        assert len(v.warnings) == 2
        assert v.confidence == pytest.approx(0.70)
        # Never below the floor however many caveats stack up.
        assert v.confidence >= 0.5


# ---------------------------------------------------------------------------
# effective_tax_rate — the through-cycle rate EPV capitalizes with
# ---------------------------------------------------------------------------

def _inc(*periods):
    """Income statement with columns newest-first from (pretax, provision)."""
    cols = {}
    for i, (pretax, prov) in enumerate(periods):
        cols[pd.Timestamp(year=2024 - i, month=12, day=31)] = {
            'Pretax Income': pretax, 'Tax Provision': prov}
    return pd.DataFrame(cols)


class TestEffectiveTaxRate:
    def test_median_of_three_years(self):
        rate, src = effective_tax_rate(_inc((100, 20), (100, 25), (100, 22)))
        assert rate == pytest.approx(0.22)
        assert src == 'median'

    def test_one_off_year_is_absorbed(self):
        """A deferred-tax revaluation year no longer drives the rate."""
        rate, src = effective_tax_rate(_inc((100, 45), (100, 21), (100, 20)))
        assert rate == pytest.approx(0.21)
        assert src == 'median'

    def test_loss_years_are_skipped(self):
        """Positive-EBIT firm with a pretax loss and a positive provision used
        to produce a negative rate, clamped to 0% -> NOPAT = EBIT."""
        rate, src = effective_tax_rate(_inc((-50, 5), (100, 23)))
        assert rate == pytest.approx(0.23)
        assert src == 'median'

    def test_all_loss_years_default_to_statutory(self):
        rate, src = effective_tax_rate(_inc((-50, 5), (-10, -2)))
        assert rate == 0.21
        assert src == 'default'

    def test_band_clamps_extremes(self):
        rate, src = effective_tax_rate(_inc((1, 0.9)))   # 90% on a breakeven year
        assert rate == 0.40
        assert src == 'clamped'
        rate, src = effective_tax_rate(_inc((100, -10)))  # net benefit
        assert rate == 0.05
        assert src == 'clamped'

    def test_only_the_recent_window_counts(self):
        rate, _ = effective_tax_rate(_inc((100, 20), (100, 20), (100, 20), (100, 40)))
        assert rate == pytest.approx(0.20)

    def test_missing_statement(self):
        assert effective_tax_rate(None) == (0.21, 'default')
        assert effective_tax_rate(pd.DataFrame()) == (0.21, 'default')

    def test_missing_lines(self):
        df = pd.DataFrame({pd.Timestamp('2024-12-31'): {'Total Revenue': 100.0}})
        assert effective_tax_rate(df) == (0.21, 'default')


# ---------------------------------------------------------------------------
# Pipeline wiring (source inspection, mirrors tests/test_debt_levels.py)
# ---------------------------------------------------------------------------

class TestPipelineWiring:
    def _src(self):
        import inspect
        import scripts.analyze_stock as mod
        return inspect.getsource(mod._run_phase2_analysis)

    def test_bridge_uses_row_net_debt_first(self):
        src = self._src()
        assert '_epv_net_debt = (net_debt_val if net_debt_val is not None' in src

    def test_tax_rate_derived_independently_of_point_ebit(self):
        src = self._src()
        assert 'effective_tax_rate(inc_stmt)' in src
        assert '_epv_eff_tax = 0.21' not in src

    def test_ebit_source_reaches_the_envelope(self):
        assert 'ebit_source=_epv_ebit_source' in self._src()

    def test_growth_premium_prefers_roic_vs_wacc(self):
        src = self._src()
        assert 'epv_with_growth_premium(epv_fv, _epv_roic, wacc)' in src
        assert "sector != 'Financial Services'" in src
