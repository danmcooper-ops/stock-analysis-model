"""models/ratios.py: calculate_fundamental_growth column selection."""

import numpy as np
import pandas as pd
import pytest

from models.ratios import calculate_fundamental_growth

COLS = [pd.Timestamp('2025-12-31'), pd.Timestamp('2024-12-31'), pd.Timestamp('2023-12-31')]
NAN = np.nan


def _fin(op, capex, da, tax=(21.0, 21.0, 21.0), pretax=(100.0, 100.0, 100.0),
         ca=(500.0, 450.0, 400.0), cl=(300.0, 280.0, 260.0)):
    n = len(op)
    cols = COLS[:n]
    inc = pd.DataFrame({c: {'Operating Income': op[i], 'Tax Provision': tax[i],
                            'Pretax Income': pretax[i]} for i, c in enumerate(cols)})
    cf = pd.DataFrame({c: {'Capital Expenditure': capex[i],
                           'Depreciation And Amortization': da[i]} for i, c in enumerate(cols)})
    bs = pd.DataFrame({c: {'Current Assets': ca[i], 'Current Liabilities': cl[i]}
                       for i, c in enumerate(cols)})
    return {'income_statement': inc, 'cash_flow': cf, 'balance_sheet': bs}


def test_complete_latest_column_is_used():
    r = calculate_fundamental_growth(_fin([100.0, 90.0], [-40.0, -30.0], [20.0, 15.0]),
                                     roic_override=0.2)
    # NOPAT 79; reinvestment 40 - 20 + (200 - 170) = 50 -> 50/79
    assert r['reinvestment_rate'] == pytest.approx(50 / 79)
    assert r['fundamental_growth'] == pytest.approx(0.2 * 50 / 79)
    assert r['growth_basis_year'] == 2025


def test_nan_capex_in_latest_falls_back_to_prior_column():
    r = calculate_fundamental_growth(_fin([100.0, 90.0, 80.0], [NAN, -30.0, -25.0],
                                          [20.0, 15.0, 10.0]),
                                     roic_override=0.2)
    # column 1: NOPAT 90 * 0.79 = 71.1; reinvestment 30 - 15 + (170 - 140) = 45
    assert r['growth_basis_year'] == 2024
    assert r['reinvestment_rate'] == pytest.approx(45 / 71.1)


def test_nan_da_in_latest_falls_back_too():
    r = calculate_fundamental_growth(_fin([100.0, 90.0], [-40.0, -30.0], [NAN, 15.0]),
                                     roic_override=0.2)
    assert r['growth_basis_year'] == 2024


def test_basis_column_without_prior_balance_sheet_uses_zero_delta_wc():
    r = calculate_fundamental_growth(_fin([100.0, 90.0], [NAN, -30.0], [20.0, 15.0]),
                                     roic_override=0.2)
    assert r['growth_basis_year'] == 2024
    assert r['reinvestment_rate'] == pytest.approx(15 / 71.1)


def test_both_columns_nan_returns_empty():
    assert calculate_fundamental_growth(_fin([100.0, 90.0], [NAN, NAN], [20.0, 15.0]),
                                        roic_override=0.2) == {}


def test_lookback_is_bounded():
    fin = _fin([100.0, 90.0, 80.0], [NAN, NAN, -25.0], [20.0, 15.0, 10.0])
    assert calculate_fundamental_growth(fin, roic_override=0.2)['growth_basis_year'] == 2023
    # a fourth column is never reached
    for k in ('income_statement', 'cash_flow', 'balance_sheet'):
        fin[k][pd.Timestamp('2022-12-31')] = fin[k][COLS[2]]
        fin[k][COLS[2]] = NAN
    assert calculate_fundamental_growth(fin, roic_override=0.2) == {}


def test_latest_operating_loss_is_not_skipped():
    """The lookback skips missing data, never a bad year."""
    assert calculate_fundamental_growth(_fin([-10.0, 90.0], [NAN, -30.0], [20.0, 15.0]),
                                        roic_override=0.2) == {}


def test_missing_latest_operating_income_walks_back():
    r = calculate_fundamental_growth(_fin([NAN, 90.0], [-40.0, -30.0], [20.0, 15.0]),
                                     roic_override=0.2)
    assert r['growth_basis_year'] == 2024


def test_non_positive_roic_returns_empty():
    fin = _fin([100.0, 90.0], [-40.0, -30.0], [20.0, 15.0])
    assert calculate_fundamental_growth(fin, roic_override=0.0) == {}
    assert calculate_fundamental_growth(fin, roic_override=-0.1) == {}


def test_empty_statements_return_empty():
    assert calculate_fundamental_growth({}) == {}
    assert calculate_fundamental_growth({'income_statement': pd.DataFrame(),
                                         'cash_flow': pd.DataFrame()}) == {}
