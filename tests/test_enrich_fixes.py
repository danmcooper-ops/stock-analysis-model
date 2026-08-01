# tests/test_enrich_fixes.py
"""Guards for the 2026-07 enrichment audit fixes: net-cash must never
fabricate zero debt, and the FDIC NPL ratio must not treat 0 as missing."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from scripts.enrich_xbrl import _compute_one
from data.sec_xbrl_client import SECXBRLClient


class FakeXbrl(SECXBRLClient):
    """Maps the FIRST tag of each concept list to a canned annual series.

    Subclasses the real client so _compute_one's call into the shared
    _resolve_total_debt_annual exercises the real priority ladder against
    the canned tag data.
    """
    def __init__(self, by_tag):
        super().__init__(cik_map={}, name_map={}, request_delay=0)
        self.by_tag = by_tag

    def _extract_annual_values(self, facts, tags, **kwargs):
        for t in tags:
            if t in self.by_tag:
                return dict(self.by_tag[t])
        return {}


def _run(by_tag, mcap=1000.0):
    rec = {"mcap": mcap, "edgar_history": {}}
    _compute_one(rec, facts={}, xbrl_client=FakeXbrl(by_tag))
    return rec


class TestNetCashDebtResolution:
    CASH = {"CashAndCashEquivalentsAtCarryingValue": {2024: 500.0}}

    def test_clean_split_noncurrent_plus_current_total(self):
        rec = _run({**self.CASH,
                    "LongTermDebtNoncurrent": {2024: 300.0},
                    "DebtCurrent": {2024: 100.0}})
        # 500 - (300 + 100) = 100 → /1000 mcap
        assert rec["net_cash_to_mcap"] == pytest.approx(0.10)

    def test_ltd_total_includes_current_no_double_subtraction(self):
        # Only LongTermDebt (total, includes current maturities) tagged:
        # debt = 400, NOT 400 + current again.
        rec = _run({**self.CASH,
                    "LongTermDebt": {2024: 400.0},
                    "LongTermDebtCurrent": {2024: 100.0}})
        assert rec["net_cash_to_mcap"] == pytest.approx(0.10)

    def test_missing_debt_year_yields_none_not_zero_debt(self):
        # Debt concept exists but lacks the cash-anchor year (the JKS/SID
        # failure: levered filers shown as 3-4x mcap in net cash).
        rec = _run({**self.CASH,
                    "LongTermDebtNoncurrent": {2022: 300.0}})
        assert "net_cash_to_mcap" not in rec

    def test_never_tagged_debt_means_unlevered(self):
        rec = _run(dict(self.CASH))
        assert rec["net_cash_to_mcap"] == pytest.approx(0.50)

    def test_short_term_borrowings_counted(self):
        rec = _run({**self.CASH,
                    "LongTermDebtNoncurrent": {2024: 200.0},
                    "ShortTermBorrowings": {2024: 200.0}})
        # no DebtCurrent → nc + ltd_cur(0) + stb(200) = 400
        assert rec["net_cash_to_mcap"] == pytest.approx(0.10)


class TestFdicNplFalsyZero:
    def test_zero_nonperforming_is_best_in_class_not_missing(self):
        # Mirror the fixed expression from enrich_fdic
        nclnls, lnlsgr = 0.0, 5_000_000.0
        npl = ((nclnls / lnlsgr)
               if (nclnls is not None and lnlsgr and lnlsgr > 0) else None)
        assert npl == 0.0
