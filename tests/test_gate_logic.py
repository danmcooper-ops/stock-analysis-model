# tests/test_gate_logic.py
"""Tests for screening-gate behavior — correctness of pass/fail logic
and the upstream fields they consume."""
import pandas as pd


from scripts.scoring import GATES
from scripts.analyze_stock import _compute_shareholder_yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate(name):
    """Return the test_fn for a gate by display name."""
    for gate in GATES:
        if gate.name == name:
            return gate.test_fn
    raise KeyError(name)


def _cf_frame(values):
    """Build a one-column cash-flow DataFrame keyed by row labels."""
    return pd.DataFrame({'2024-12-31': values})


# ---------------------------------------------------------------------------
# Quality: Accruals — pass on negative (good) accruals, fail only on
# large positive accruals (Sloan 1996 anomaly is one-sided).
# ---------------------------------------------------------------------------

class TestAccrualsGate:
    def setup_method(self):
        self.fn = _gate('Quality: Accruals')

    def test_strongly_negative_accruals_pass(self):
        """CFO >> NI: conservative accounting / strong cash gen — passes."""
        assert self.fn(-0.20, {}) is True
        assert self.fn(-0.10, {}) is True

    def test_small_accruals_pass(self):
        assert self.fn(0.0, {}) is True
        assert self.fn(0.05, {}) is True

    def test_large_positive_accruals_fail(self):
        """High positive accruals (NI >> CFO) — Sloan's red flag, fails."""
        assert self.fn(0.10, {}) is False
        assert self.fn(0.50, {}) is False

    def test_boundary(self):
        """Threshold is strict <: 0.08 itself fails."""
        assert self.fn(0.08, {}) is False
        assert self.fn(0.0799, {}) is True

    def test_none_returns_none(self):
        assert self.fn(None, {}) is None


# ---------------------------------------------------------------------------
# _compute_shareholder_yield — net dilution must produce a NEGATIVE
# buyback_rate and reduced (or negative) shareholder_yield. Flooring
# at zero would mask the diluter.
# ---------------------------------------------------------------------------

class TestShareholderYield:
    MCAP = 10_000_000_000.0  # $10B

    def test_pure_dividend(self):
        """No buybacks, no issuance, $200M dividend → 2% yield."""
        cf = _cf_frame({'Cash Dividends Paid': -200_000_000})
        out = _compute_shareholder_yield({'cash_flow': cf}, self.MCAP)
        assert out['shareholder_yield'] == 0.02
        assert out['buyback_rate'] == 0.0

    def test_buybacks_only(self):
        """$300M buybacks, no issuance, no dividend → 3% yield."""
        cf = _cf_frame({'Repurchase Of Capital Stock': -300_000_000})
        out = _compute_shareholder_yield({'cash_flow': cf}, self.MCAP)
        assert out['shareholder_yield'] == 0.03
        assert out['buyback_rate'] == 0.03

    def test_net_dilution_produces_negative_buyback_rate(self):
        """Issuance > buybacks → buyback_rate negative (NOT floored at 0)."""
        cf = _cf_frame({
            'Repurchase Of Capital Stock': -100_000_000,
            'Issuance Of Capital Stock': -300_000_000,
        })
        out = _compute_shareholder_yield({'cash_flow': cf}, self.MCAP)
        assert out['buyback_rate'] == -0.02
        assert out['shareholder_yield'] == -0.02

    def test_dividend_offset_by_dilution(self):
        """3% dividend, 4% net dilution → -1% true shareholder yield."""
        cf = _cf_frame({
            'Cash Dividends Paid': -300_000_000,
            'Issuance Of Capital Stock': -400_000_000,
        })
        out = _compute_shareholder_yield({'cash_flow': cf}, self.MCAP)
        assert out['shareholder_yield'] == -0.01
        assert out['buyback_rate'] == -0.04

    def test_extreme_negative_yield_is_capped_to_none(self):
        """|yield| > 50% indicates a data error — both fields nulled."""
        cf = _cf_frame({'Issuance Of Capital Stock': -10_000_000_000})  # 100% dilution
        out = _compute_shareholder_yield({'cash_flow': cf}, self.MCAP)
        assert out['shareholder_yield'] is None
        assert out['buyback_rate'] is None

    def test_extreme_positive_yield_is_capped_to_none(self):
        """Pre-existing sanity cap on the upper side — preserved."""
        cf = _cf_frame({'Repurchase Of Capital Stock': -10_000_000_000})  # 100% buyback
        out = _compute_shareholder_yield({'cash_flow': cf}, self.MCAP)
        assert out['shareholder_yield'] is None
        assert out['buyback_rate'] is None

    def test_zero_mcap_returns_none(self):
        cf = _cf_frame({'Cash Dividends Paid': -100_000_000})
        assert _compute_shareholder_yield({'cash_flow': cf}, 0) is None
        assert _compute_shareholder_yield({'cash_flow': cf}, None) is None

    def test_no_cash_flow_returns_none(self):
        assert _compute_shareholder_yield({}, self.MCAP) is None

    def test_dividend_rate_fallback_when_cf_missing(self):
        """No cash-flow at all, but info.dividendRate set → use rate × shares."""
        info = {'dividendRate': 2.0, 'sharesOutstanding': 100_000_000}
        out = _compute_shareholder_yield({'info': info}, self.MCAP)
        # 2.0 × 100M = $200M dividend / $10B mcap = 2.0%
        assert out['shareholder_yield'] == 0.02
        assert out['buyback_rate'] == 0.0

    def test_dividend_rate_fallback_when_cf_lacks_dividend_row(self):
        """CF present (e.g. has buyback row) but no Dividend Paid line → fall back."""
        cf = _cf_frame({'Repurchase Of Capital Stock': -100_000_000})
        info = {'dividendRate': 2.0, 'sharesOutstanding': 100_000_000}
        out = _compute_shareholder_yield({'cash_flow': cf, 'info': info}, self.MCAP)
        # 1% buyback + 2% dividend (from fallback) = 3% total
        assert out['shareholder_yield'] == 0.03
        assert out['buyback_rate'] == 0.01

    def test_cf_dividend_preferred_over_info_rate(self):
        """When CF has Dividend Paid AND info.dividendRate is set, CF wins."""
        cf = _cf_frame({'Cash Dividends Paid': -150_000_000})
        info = {'dividendRate': 5.0, 'sharesOutstanding': 100_000_000}
        out = _compute_shareholder_yield({'cash_flow': cf, 'info': info}, self.MCAP)
        # CF says $150M = 1.5%, NOT info's $500M = 5%
        assert out['shareholder_yield'] == 0.015

    def test_genuine_no_return_company_stays_zero(self):
        """CF present, no dividend / buyback rows, no dividendRate → 0 (not None).
        A young growth co that legitimately returns nothing should read as 0."""
        cf = _cf_frame({'Net Income': 100_000_000})  # any non-dividend row
        out = _compute_shareholder_yield({'cash_flow': cf, 'info': {}}, self.MCAP)
        assert out['shareholder_yield'] == 0.0
        assert out['buyback_rate'] == 0.0


# ---------------------------------------------------------------------------
# Downstream gate behavior: the buyback gate now correctly fails diluters
# (negative rate fails the > 0.01 threshold).
# ---------------------------------------------------------------------------

class TestNewGateThresholds:
    """Threshold logic for the gates added in the 2026-07 rebalance."""

    def test_ebit_ev(self):
        fn = _gate('Valuation: EBIT/EV')
        assert fn(0.10, {}) is True      # cheap: 10x EV/EBIT
        assert fn(0.08, {}) is False     # boundary is strict >
        assert fn(0.05, {}) is False     # 20x EV/EBIT
        assert fn(None, {}) is None

    def test_incremental_roic(self):
        fn = _gate('Moat: Incr ROIC')
        assert fn(0.20, {}) is True
        assert fn(0.10, {}) is False     # boundary is strict >
        assert fn(-0.02, {}) is False
        assert fn(None, {}) is None

    def test_margin_vs_hist(self):
        fn = _gate('Quality: Margin vs Hist')
        assert fn(0.00, {}) is True      # at own historical average
        assert fn(-0.03, {}) is True     # below history — no over-earning
        assert fn(0.05, {}) is False     # 5pp+ above history fails
        assert fn(0.10, {}) is False
        assert fn(None, {}) is None

    def test_insider_buying(self):
        fn = _gate('Ownership: Insider Buying')
        assert fn(0.75, {}) is True
        assert fn(0.50, {}) is True      # boundary is >=
        assert fn(0.25, {}) is False
        assert fn(None, {}) is None

    def test_pool_share(self):
        fn = _gate('Moat: Pool Share')
        assert fn(0.05, {}) is True      # gaining share of the sector pool
        assert fn(0.0, {}) is False      # boundary is strict >
        assert fn(-0.03, {}) is False    # losing share
        assert fn(None, {}) is None
