"""Base-FCF pipeline of run_forward_dcf: SBC monotone floor + ceiling basis.

Regression tests for two July-audit findings:

1. SBC cliff: the old guard (`if sbc > 0 and base_fcf - sbc > 0`) skipped the
   deduction entirely whenever SBC >= FCF, so the heaviest SBC issuers paid NO
   penalty and fair value JUMPED UP as SBC crossed above FCF. The deduction
   must be monotone: more SBC never means a higher fair value.

2. Ceiling basis: the mean-reversion ceiling (1.25x trailing avg positive FCF)
   was applied AFTER the owner-earnings adjustments, comparing adjusted FCF
   against a raw-FCF average. Whenever the ceiling bound, it erased the SBC
   deduction and the growth-capex add-back entirely. The ceiling now caps the
   raw base first; adjustments apply on top.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'scripts'))

from scripts.analyze_stock import run_forward_dcf  # noqa: E402


def _yf_data(fcf_by_year, sbc=0.0, capex=200.0, da=300.0, ocf=None):
    """Minimal yf_data for run_forward_dcf. Flat revenue keeps growth quiet."""
    years = pd.to_datetime([f'{2021 + i}-12-31' for i in range(len(fcf_by_year))])
    cf = pd.DataFrame(
        {y: {'Free Cash Flow': v,
             'Operating Cash Flow': ocf if ocf is not None else v + capex,
             'Depreciation And Amortization': da,
             'Capital Expenditure': -capex,
             'Stock Based Compensation': sbc}
         for y, v in zip(years, fcf_by_year)}
    )
    inc = pd.DataFrame({y: {'Total Revenue': 10_000.0} for y in years})
    info = {'marketCap': 20_000.0, 'sharesOutstanding': 1_000.0,
            'totalDebt': 0.0, 'totalCash': 0.0, 'currentPrice': 20.0}
    return {'cash_flow': cf, 'income_statement': inc, 'info': info}


def _fv(**kw):
    fv, _, _, _, _ = run_forward_dcf(_yf_data(**kw), wacc=0.10)
    return fv


class TestSbcMonotoneFloor:
    def test_more_sbc_never_raises_fair_value(self):
        """Sweep SBC through the old cliff point (SBC == FCF): FV must be
        non-increasing across the whole sweep."""
        flat = [1000.0] * 4
        fvs = [_fv(fcf_by_year=flat, sbc=s)
               for s in (0.0, 400.0, 800.0, 999.0, 1001.0, 1500.0)]
        assert all(v is not None for v in fvs)
        for lo, hi in zip(fvs[1:], fvs):
            assert lo <= hi + 1e-9, f'FV rose as SBC increased: {fvs}'

    def test_extreme_sbc_haircuts_not_aborts(self):
        """SBC far above FCF floors the base at 25% of pre-SBC FCF: a valuation
        still comes out, at a deep haircut vs the SBC-free case."""
        flat = [1000.0] * 4
        fv_heavy = _fv(fcf_by_year=flat, sbc=5000.0)
        fv_clean = _fv(fcf_by_year=flat, sbc=0.0)
        assert fv_heavy is not None
        assert fv_heavy < fv_clean * 0.5


class TestCeilingBasis:
    def test_sbc_survives_a_binding_ceiling(self):
        """Peak-cycle spike (last year 3x the norm) makes the ceiling bind.
        Under the old ordering the ceiling was applied after the SBC
        deduction, so with-SBC and without-SBC both landed exactly on the
        ceiling — SBC became invisible for every firm at peak cycle. The
        deduction must still separate them."""
        spike = [1000.0, 1000.0, 1000.0, 3000.0]
        fv_sbc = _fv(fcf_by_year=spike, sbc=800.0)
        fv_no_sbc = _fv(fcf_by_year=spike, sbc=0.0)
        assert fv_sbc is not None and fv_no_sbc is not None
        assert fv_sbc < fv_no_sbc - 1e-9

    def test_ceiling_still_caps_peak_extrapolation(self):
        """The reorder must not disable the ceiling itself: a peak-year base
        is still pulled back toward the trailing average. Both series start
        and end at 3000 (identical FCF CAGR and revenue), so the only
        difference is the trough years dragging the trailing average — and
        with it the ceiling — down for the cyclical."""
        cyclical = [3000.0, 1000.0, 1000.0, 3000.0]
        steady = [3000.0] * 4
        fv_cyclical = _fv(fcf_by_year=cyclical)
        fv_steady = _fv(fcf_by_year=steady)
        assert fv_cyclical is not None and fv_steady is not None
        assert fv_cyclical < fv_steady
