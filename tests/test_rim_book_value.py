"""RIM book value per share must sit on the same basis as the price.

Regression for the 2026-09-09 snapshot, where yfinance's info['bookValue']
put RIM fair values at 505x price for EC (a COP-reporting ADR), 70,000x for
PIFMF (IDR) and 1,280x for BRK-B (bookValue per A-share equivalent against a
B-share price, rated LEAN BUY).
"""

import math

import pandas as pd

from scripts.analyze_stock import (RIM_BOOK_BASIS_MAX_RATIO,
                                   _rim_book_value_per_share)


def _bs(equity, key='Stockholders Equity'):
    return pd.DataFrame({'2025-12-31': {key: equity}})


USD = {'currency_financial': 'USD', 'currency_quote': 'USD'}


def test_adr_uses_converted_statement_equity_not_local_book_value():
    # EC: bookValue 1963 COP per ordinary share vs a USD ADS price; the
    # balance sheet is already converted to USD.
    fx = {'currency_financial': 'COP', 'currency_quote': 'USD'}
    bv, src = _rim_book_value_per_share({'bookValue': 1963.0}, _bs(19_000_000_000.0),
                                        1_000_000_000, fx)
    assert src == 'statement_ccy_mismatch'
    assert math.isclose(bv, 19.0)


def test_adr_without_statement_equity_gets_no_book_value():
    fx = {'currency_financial': 'IDR', 'currency_quote': 'USD'}
    assert _rim_book_value_per_share({'bookValue': 2500.0}, None, 1e9, fx) == (None, None)


def test_dual_class_basis_mismatch_falls_back_to_statement():
    # BRK-B: bookValue per A-share equivalent, ~1,500x the per-B-share book.
    bv, src = _rim_book_value_per_share({'bookValue': 450_000.0}, _bs(650_000_000_000.0),
                                        2_150_000_000, USD)
    assert src == 'statement_basis_mismatch'
    assert 290 < bv < 310


def test_agreeing_estimates_keep_info_book_value():
    # Within the basis tolerance the reported figure is kept as before.
    bv, src = _rim_book_value_per_share({'bookValue': 31.0}, _bs(30_000_000_000.0),
                                        1_000_000_000, USD)
    assert (bv, src) == (31.0, 'info')
    edge = 30.0 * RIM_BOOK_BASIS_MAX_RATIO * 0.99
    assert _rim_book_value_per_share({'bookValue': edge}, _bs(30e9), 1e9, USD)[1] == 'info'


def test_missing_book_value_uses_statement_and_other_equity_keys():
    bv, src = _rim_book_value_per_share({}, _bs(5e9, key='Common Stock Equity'), 1e8, USD)
    assert (bv, src) == (50.0, 'statement')
    assert _rim_book_value_per_share({'bookValue': float('nan')}, _bs(5e9), 1e8, USD) == (50.0, 'statement')


def test_nothing_usable_returns_none():
    assert _rim_book_value_per_share({}, None, None, USD) == (None, None)
    assert _rim_book_value_per_share({}, _bs(5e9), 0, USD) == (None, None)
    assert _rim_book_value_per_share(None, pd.DataFrame(), 1e8, None) == (None, None)


def test_negative_equity_does_not_trigger_basis_switch():
    # A buyback-heavy firm with negative statement equity: no ratio test,
    # the reported bookValue stands (the RIM handles its own sign rules).
    bv, src = _rim_book_value_per_share({'bookValue': -3.0}, _bs(-2e9), 1e9, USD)
    assert (bv, src) == (-3.0, 'info')
