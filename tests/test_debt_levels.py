# tests/test_debt_levels.py
"""Row-level debt levels: statements-first with per-field .info fallback.

Guards two fixed defects: the reverse-DCF branch used to clobber the row's
net_debt with `get_net_debt(...) or 0` (coercing unknown leverage to zero
and letting the report show Net Debt != Total Debt - Cash), and
debt_source labeled the row 'yf_info' even when only one of the two
fields actually fell back.
"""
import pandas as pd

from scripts.analyze_stock import _debt_levels


def _bs_frame(rows):
    """Single newest-column balance-sheet frame from {line item: value}."""
    col = pd.Timestamp('2025-12-31')
    return pd.DataFrame({col: rows})


class TestDebtLevels:
    def test_statements_only(self):
        yf_data = {
            'balance_sheet': _bs_frame({
                'Total Debt': 500.0,
                'Cash And Cash Equivalents': 200.0,
                'Total Liabilities Net Minority Interest': 900.0,
            }),
            'info': {'totalDebt': 999.0, 'totalCash': 999.0},  # must be ignored
        }
        debt, cash, liabs, net, source = _debt_levels(yf_data)
        assert (debt, cash, liabs) == (500.0, 200.0, 900.0)
        assert net == 300.0  # the invariant the clobber used to break
        assert source is None

    def test_full_info_fallback(self):
        yf_data = {'balance_sheet': None,
                   'info': {'totalDebt': 400.0, 'totalCash': 150.0}}
        debt, cash, liabs, net, source = _debt_levels(yf_data)
        assert (debt, cash, net) == (400.0, 150.0, 250.0)
        assert liabs is None
        assert source == 'yf_info'

    def test_mixed_sources_labeled_mixed(self):
        """Statement debt + info cash used to be mislabeled plain 'yf_info'."""
        yf_data = {
            'balance_sheet': _bs_frame({'Total Debt': 500.0}),
            'info': {'totalCash': 120.0},
        }
        debt, cash, _liabs, net, source = _debt_levels(yf_data)
        assert (debt, cash, net) == (500.0, 120.0, 380.0)
        assert source == 'statements+yf_info'

    def test_single_known_field_from_info_is_yf_info(self):
        """Only one field known and it came from .info -> plain 'yf_info'
        (there is no statement contribution to acknowledge)."""
        yf_data = {'balance_sheet': None, 'info': {'totalDebt': 300.0}}
        debt, cash, _liabs, net, source = _debt_levels(yf_data)
        assert debt == 300.0
        assert cash is None
        assert net is None  # one side unknown -> net unknown, never coerced
        assert source == 'yf_info'

    def test_unknown_stays_none_not_zero(self):
        yf_data = {'balance_sheet': None, 'info': {}}
        debt, cash, liabs, net, source = _debt_levels(yf_data)
        assert debt is None and cash is None and liabs is None and net is None
        assert source is None

    def test_reverse_dcf_branch_no_longer_clobbers(self):
        """The reverse-DCF block must use a branch-local variable, not the
        row's net_debt_val."""
        import inspect
        import scripts.analyze_stock as mod
        src = inspect.getsource(mod._run_phase2_analysis)
        assert 'net_debt_val = get_net_debt' not in src
        assert '_rev_net_debt = get_net_debt' in src
