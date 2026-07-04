# tests/test_epv.py
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.epv import earnings_power_value, epv_with_growth_premium


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
    def test_roe_above_re_gives_premium(self):
        """ROE > Re -> growth premium -> value above base."""
        base = 100.0
        result = epv_with_growth_premium(base, roe=0.20, cost_of_equity=0.10)
        assert result > base

    def test_roe_below_re_returns_base(self):
        """ROE < Re -> no growth premium, returns base EPV."""
        base = 100.0
        result = epv_with_growth_premium(base, roe=0.05, cost_of_equity=0.10)
        assert result == base  # multiplier floored at 1.0

    def test_none_on_none_epv(self):
        """None base EPV -> None."""
        assert epv_with_growth_premium(None, 0.15, 0.10) is None

    def test_negative_roe_returns_base(self):
        """Negative ROE -> returns base."""
        result = epv_with_growth_premium(100.0, roe=-0.10, cost_of_equity=0.10)
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
        result = epv_with_growth_premium(100.0, roe=0.50, cost_of_equity=0.10)
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
