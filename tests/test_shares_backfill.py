# tests/test_shares_backfill.py
import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.yfinance_client import (_backfill_shares_and_mcap,
                                  _null_uncorroborated_shares)


class _FastInfo:
    """Stand-in for yfinance's fast_info accessor."""

    def __init__(self, market_cap=None, shares=None, last_price=None,
                 raises=False):
        self._raises = raises
        if not raises:
            self.market_cap = market_cap
            self.shares = shares
            self.last_price = last_price

    def __getattr__(self, name):
        # Only reached for attributes not set in __init__ (the raises case).
        raise RuntimeError('fast_info unavailable')


class _Stock:
    def __init__(self, fast_info=None, shares_full=None,
                 fast_info_raises=False, shares_full_raises=False):
        self._fast_info = fast_info
        self._shares_full = shares_full
        self._fast_info_raises = fast_info_raises
        self._shares_full_raises = shares_full_raises

    @property
    def fast_info(self):
        if self._fast_info_raises:
            raise RuntimeError('no fast_info')
        return self._fast_info

    def get_shares_full(self, start=None):
        if self._shares_full_raises:
            raise RuntimeError('no shares series')
        return self._shares_full


class TestBackfillSharesAndMcap:
    def test_healthy_info_is_left_alone(self):
        """The common path must not touch the network or the values."""
        info = {'marketCap': 183057252352, 'sharesOutstanding': 1730383296}
        stock = _Stock(fast_info_raises=True, shares_full_raises=True)
        assert _backfill_shares_and_mcap(stock, info) == []
        assert info['marketCap'] == 183057252352
        assert info['sharesOutstanding'] == 1730383296

    def test_recovers_both_fields_from_fast_info(self):
        """The observed failure: .info drops both, fast_info still has them."""
        info = {'marketCap': None, 'sharesOutstanding': None,
                'currentPrice': 574.34, 'floatShares': 875799481}
        stock = _Stock(_FastInfo(market_cap=507473152776.0, shares=883583855,
                                 last_price=574.335))
        recovered = _backfill_shares_and_mcap(stock, info)
        assert set(recovered) == {'marketCap', 'sharesOutstanding'}
        assert info['sharesOutstanding'] == 883583855
        assert info['marketCap'] == pytest.approx(507473152776.0)

    def test_derives_mcap_from_price_when_fast_info_lacks_it(self):
        """fast_info's market_cap is price x shares, so deriving it is the
        same figure by another route when only that value is absent."""
        info = {'marketCap': None, 'sharesOutstanding': None,
                'currentPrice': 100.0}
        stock = _Stock(_FastInfo(market_cap=None, shares=1_000_000))
        recovered = _backfill_shares_and_mcap(stock, info)
        assert set(recovered) == {'marketCap', 'sharesOutstanding'}
        assert info['marketCap'] == pytest.approx(100_000_000.0)

    def test_falls_back_to_shares_full_series(self):
        """Second fallback for share count when fast_info comes back bare."""
        info = {'marketCap': None, 'sharesOutstanding': None,
                'regularMarketPrice': 50.0}
        series = pd.Series([900_000, 950_000, 1_000_000])
        stock = _Stock(_FastInfo(market_cap=None, shares=None), shares_full=series)
        recovered = _backfill_shares_and_mcap(stock, info)
        assert info['sharesOutstanding'] == 1_000_000
        assert info['marketCap'] == pytest.approx(50_000_000.0)
        assert set(recovered) == {'marketCap', 'sharesOutstanding'}

    def test_recovers_shares_only_when_mcap_already_present(self):
        info = {'marketCap': 5_000_000, 'sharesOutstanding': None}
        stock = _Stock(_FastInfo(market_cap=None, shares=250_000))
        assert _backfill_shares_and_mcap(stock, info) == ['sharesOutstanding']
        assert info['marketCap'] == 5_000_000

    def test_all_sources_failing_degrades_quietly(self):
        """A genuinely dataless ticker must not raise — it just stays None."""
        info = {'marketCap': None, 'sharesOutstanding': None}
        stock = _Stock(fast_info_raises=True, shares_full_raises=True)
        assert _backfill_shares_and_mcap(stock, info) == []
        assert info['marketCap'] is None
        assert info['sharesOutstanding'] is None

    def test_zero_and_negative_values_are_rejected(self):
        """Zero shares would make every per-share figure a division blow-up."""
        info = {'marketCap': None, 'sharesOutstanding': None,
                'currentPrice': 10.0}
        stock = _Stock(_FastInfo(market_cap=0, shares=0), shares_full=pd.Series([0]))
        assert _backfill_shares_and_mcap(stock, info) == []
        assert info['sharesOutstanding'] is None
        assert info['marketCap'] is None

    def test_no_price_anywhere_leaves_mcap_none(self):
        info = {'marketCap': None, 'sharesOutstanding': None}
        stock = _Stock(_FastInfo(market_cap=None, shares=1_000, last_price=None))
        recovered = _backfill_shares_and_mcap(stock, info)
        assert recovered == ['sharesOutstanding']
        assert info['marketCap'] is None


class TestNullUncorroboratedShares:
    """Yahoo hands preferred / secondary OTC lines the parent's common share
    count with no marketCap from ANY source (verified live for FNMAG, FMCKP,
    AGNCM, VLYPO, AILLO on 2026-08-14) — the count must be rejected before
    the backfill derives a phantom cap from it."""

    def test_contaminated_preferred_line_is_nulled(self):
        """The FNMAG shape: packaged shares, no cap anywhere, fast_info and
        the shares series both blank."""
        info = {'marketCap': None, 'sharesOutstanding': 5_738_840_064,
                'regularMarketPrice': 15.65}
        stock = _Stock(_FastInfo(market_cap=None, shares=None))
        assert (_null_uncorroborated_shares(stock, info)
                == ['sharesOutstanding_uncorroborated'])
        assert info['sharesOutstanding'] is None
        # End-to-end: the backfill must now find nothing to derive from.
        assert _backfill_shares_and_mcap(stock, info) == []
        assert info['marketCap'] is None

    def test_packaged_mcap_disengages_the_check(self):
        info = {'marketCap': 7_191_723_520, 'sharesOutstanding': 1_158_087_567}
        stock = _Stock(fast_info_raises=True, shares_full_raises=True)
        assert _null_uncorroborated_shares(stock, info) == []
        assert info['sharesOutstanding'] == 1_158_087_567

    def test_both_fields_missing_leaves_nothing_to_distrust(self):
        """The MA/LLY sticky-drop case is the backfill's job, not this one's."""
        info = {'marketCap': None, 'sharesOutstanding': None}
        stock = _Stock(fast_info_raises=True, shares_full_raises=True)
        assert _null_uncorroborated_shares(stock, info) == []

    def test_agreeing_fast_info_count_corroborates(self):
        info = {'marketCap': None, 'sharesOutstanding': 1_000_000,
                'currentPrice': 10.0}
        stock = _Stock(_FastInfo(market_cap=None, shares=1_010_000))
        assert _null_uncorroborated_shares(stock, info) == []
        assert info['sharesOutstanding'] == 1_000_000

    def test_fast_info_mcap_implied_count_corroborates(self):
        info = {'marketCap': None, 'sharesOutstanding': 1_000_000,
                'currentPrice': 10.0}
        stock = _Stock(_FastInfo(market_cap=10_500_000, shares=None))
        assert _null_uncorroborated_shares(stock, info) == []

    def test_shares_series_corroborates(self):
        info = {'marketCap': None, 'sharesOutstanding': 1_000_000,
                'currentPrice': 10.0}
        series = pd.Series([980_000, 990_000])
        stock = _Stock(_FastInfo(market_cap=None, shares=None),
                       shares_full=series)
        assert _null_uncorroborated_shares(stock, info) == []

    def test_wildly_disagreeing_source_nulls_the_packaged_count(self):
        """When fast_info knows a different security-level count, the
        packaged one is the poison — null it and let the backfill refill
        from the trusted source."""
        info = {'marketCap': None, 'sharesOutstanding': 3_221_329_920,
                'currentPrice': 12.08}
        stock = _Stock(_FastInfo(market_cap=None, shares=45_000_000))
        assert (_null_uncorroborated_shares(stock, info)
                == ['sharesOutstanding_uncorroborated'])
        recovered = _backfill_shares_and_mcap(stock, info)
        assert info['sharesOutstanding'] == 45_000_000
        assert set(recovered) == {'marketCap', 'sharesOutstanding'}
