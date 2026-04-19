# tests/test_portfolio_tracker.py
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.portfolio_tracker import (
    enrich_holdings,
    compute_holding_weights,
    compute_portfolio_pnl,
    detect_alerts,
    summarize_realized_gains,
    _aggregate_lots,
    _compute_return_since,
)
import pandas as pd
import numpy as np
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_holdings():
    return [
        {'ticker': 'AAPL', 'shares': 10, 'cost_basis': 150.0, 'purchase_date': '2024-01-15'},
        {'ticker': 'MSFT', 'shares': 5, 'cost_basis': 300.0, 'purchase_date': '2024-06-01'},
        {'ticker': 'NVDA', 'shares': 8, 'cost_basis': 500.0, 'purchase_date': '2023-09-10'},
    ]


@pytest.fixture
def sample_prices():
    return {
        'AAPL': 180.0,
        'MSFT': 420.0,
        'NVDA': 800.0,
    }


@pytest.fixture
def sample_results():
    return {
        'AAPL': {
            'ticker': 'AAPL', 'rating': 'LEAN BUY', 'price': 180.0,
            'dcf_fv': 160.0, 'mos': 0.12, '_composite_score': 62.0,
            '_score_valuation': 55.0, '_score_quality': 70.0,
            '_score_moat': 75.0, '_score_growth': 58.0, '_score_ownership': 68.0,
            'sector': 'Technology', 'company_name': 'Apple Inc.',
            'spread': 0.15, 'mc_confidence': 'MEDIUM (20%)',
        },
        'MSFT': {
            'ticker': 'MSFT', 'rating': 'BUY', 'price': 420.0,
            'dcf_fv': 450.0, 'mos': 0.065, '_composite_score': 78.5,
            '_score_valuation': 72.0, '_score_quality': 85.0,
            '_score_moat': 88.0, '_score_growth': 70.0, '_score_ownership': 60.0,
            'sector': 'Technology', 'company_name': 'Microsoft Corp.',
            'spread': 0.22, 'mc_confidence': 'HIGH (12%)',
        },
        # NVDA intentionally absent to test not-in-universe path
    }


@pytest.fixture
def sample_benchmark_series():
    """Fake benchmark series: starts at 100, grows 10% over 2 years."""
    idx = pd.date_range(start='2023-01-01', periods=500, freq='B', tz='UTC')
    prices = [100 * (1 + 0.10 * i / 500) for i in range(500)]
    return pd.Series(prices, index=idx)


# ---------------------------------------------------------------------------
# _aggregate_lots
# ---------------------------------------------------------------------------

class TestAggregateLots:
    def test_single_lot(self, sample_holdings):
        result = _aggregate_lots(sample_holdings)
        assert len(result) == 3

    def test_multiple_lots_same_ticker(self):
        holdings = [
            {'ticker': 'AAPL', 'shares': 10, 'cost_basis': 100.0, 'purchase_date': '2024-01-01'},
            {'ticker': 'AAPL', 'shares': 10, 'cost_basis': 200.0, 'purchase_date': '2024-06-01'},
        ]
        result = _aggregate_lots(holdings)
        assert len(result) == 1
        aapl = result[0]
        assert aapl['shares'] == 20.0
        assert aapl['cost_basis'] == pytest.approx(150.0)  # weighted average
        assert aapl['cost_value'] == pytest.approx(3000.0)
        assert aapl['purchase_date'] == '2024-01-01'  # earliest

    def test_empty(self):
        assert _aggregate_lots([]) == []


# ---------------------------------------------------------------------------
# enrich_holdings
# ---------------------------------------------------------------------------

class TestEnrichHoldings:
    def test_basic_enrichment(self, sample_holdings, sample_prices, sample_results):
        result = enrich_holdings(sample_holdings, sample_prices, sample_results)
        assert len(result) == 3

    def test_in_universe_flag(self, sample_holdings, sample_prices, sample_results):
        result = enrich_holdings(sample_holdings, sample_prices, sample_results)
        by_ticker = {h['ticker']: h for h in result}
        assert by_ticker['AAPL']['in_universe'] is True
        assert by_ticker['MSFT']['in_universe'] is True
        assert by_ticker['NVDA']['in_universe'] is False

    def test_not_in_universe_rating(self, sample_holdings, sample_prices, sample_results):
        result = enrich_holdings(sample_holdings, sample_prices, sample_results)
        nvda = next(h for h in result if h['ticker'] == 'NVDA')
        assert nvda['rating'] == 'NOT IN UNIVERSE'

    def test_market_value_computed(self, sample_holdings, sample_prices, sample_results):
        result = enrich_holdings(sample_holdings, sample_prices, sample_results)
        aapl = next(h for h in result if h['ticker'] == 'AAPL')
        assert aapl['market_value'] == pytest.approx(10 * 180.0)

    def test_unrealized_pnl(self, sample_holdings, sample_prices, sample_results):
        result = enrich_holdings(sample_holdings, sample_prices, sample_results)
        aapl = next(h for h in result if h['ticker'] == 'AAPL')
        # cost: 10 * 150 = 1500, mv: 10 * 180 = 1800
        assert aapl['unrealized_pnl'] == pytest.approx(300.0)
        assert aapl['unrealized_pnl_pct'] == pytest.approx(0.20)

    def test_missing_price_handled(self, sample_holdings, sample_results):
        prices = {'AAPL': None, 'MSFT': 420.0, 'NVDA': None}
        result = enrich_holdings(sample_holdings, prices, sample_results)
        aapl = next(h for h in result if h['ticker'] == 'AAPL')
        assert aapl['market_value'] is None
        assert aapl['unrealized_pnl'] is None

    def test_valuation_gap_computed(self, sample_holdings, sample_prices, sample_results):
        result = enrich_holdings(sample_holdings, sample_prices, sample_results)
        # AAPL: price=180, dcf_fv=160 → gap = (180-160)/160 = 0.125
        aapl = next(h for h in result if h['ticker'] == 'AAPL')
        assert aapl['valuation_gap_pct'] == pytest.approx(0.125)

    def test_analysis_fields_joined(self, sample_holdings, sample_prices, sample_results):
        result = enrich_holdings(sample_holdings, sample_prices, sample_results)
        msft = next(h for h in result if h['ticker'] == 'MSFT')
        assert msft['rating'] == 'BUY'
        assert msft['_composite_score'] == pytest.approx(78.5)
        assert msft['sector'] == 'Technology'


# ---------------------------------------------------------------------------
# compute_holding_weights
# ---------------------------------------------------------------------------

class TestComputeHoldingWeights:
    def test_weights_sum_to_one(self, sample_holdings, sample_prices, sample_results):
        enriched = enrich_holdings(sample_holdings, sample_prices, sample_results)
        enriched = compute_holding_weights(enriched)
        weights = [h['position_weight'] for h in enriched if h.get('position_weight') is not None]
        assert sum(weights) == pytest.approx(1.0)

    def test_no_price_excluded_from_denom(self):
        holdings = [
            {'ticker': 'A', 'shares': 10, 'cost_basis': 100.0, 'purchase_date': '2024-01-01'},
            {'ticker': 'B', 'shares': 10, 'cost_basis': 100.0, 'purchase_date': '2024-01-01'},
        ]
        prices = {'A': 200.0, 'B': None}
        enriched = enrich_holdings(holdings, prices, {})
        enriched = compute_holding_weights(enriched)
        a = next(h for h in enriched if h['ticker'] == 'A')
        b = next(h for h in enriched if h['ticker'] == 'B')
        assert a['position_weight'] == pytest.approx(1.0)
        assert b['position_weight'] is None

    def test_equal_value_equal_weight(self):
        holdings = [
            {'ticker': 'A', 'shares': 10, 'cost_basis': 100.0, 'purchase_date': '2024-01-01'},
            {'ticker': 'B', 'shares': 10, 'cost_basis': 100.0, 'purchase_date': '2024-01-01'},
        ]
        prices = {'A': 100.0, 'B': 100.0}
        enriched = enrich_holdings(holdings, prices, {})
        enriched = compute_holding_weights(enriched)
        for h in enriched:
            assert h['position_weight'] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_portfolio_pnl
# ---------------------------------------------------------------------------

class TestComputePortfolioPnl:
    def test_total_values(self, sample_holdings, sample_prices, sample_results):
        enriched = enrich_holdings(sample_holdings, sample_prices, sample_results)
        pnl = compute_portfolio_pnl(enriched)
        # cost: 10*150 + 5*300 + 8*500 = 1500 + 1500 + 4000 = 7000
        assert pnl['total_cost_basis'] == pytest.approx(7000.0)
        # mv: 10*180 + 5*420 + 8*800 = 1800 + 2100 + 6400 = 10300
        assert pnl['total_market_value'] == pytest.approx(10300.0)
        assert pnl['unrealized_pnl'] == pytest.approx(3300.0)
        assert pnl['unrealized_pnl_pct'] == pytest.approx(3300.0 / 7000.0)

    def test_realized_pnl_ytd(self):
        this_year = str(date.today().year)
        realized = [
            {'ticker': 'X', 'shares_sold': 10, 'cost_basis': 100.0,
             'sale_price': 150.0, 'sale_date': f'{this_year}-03-01'},
            {'ticker': 'Y', 'shares_sold': 5, 'cost_basis': 200.0,
             'sale_price': 180.0, 'sale_date': f'{this_year}-06-15'},
            # Prior year — should be excluded
            {'ticker': 'Z', 'shares_sold': 20, 'cost_basis': 50.0,
             'sale_price': 80.0, 'sale_date': f'{int(this_year)-1}-12-01'},
        ]
        pnl = compute_portfolio_pnl([], realized)
        # 10*(150-100) + 5*(180-200) = 500 - 100 = 400
        assert pnl['realized_pnl_ytd'] == pytest.approx(400.0)

    def test_empty_holdings(self):
        pnl = compute_portfolio_pnl([])
        assert pnl['total_cost_basis'] == 0
        assert pnl['total_market_value'] == 0


# ---------------------------------------------------------------------------
# detect_alerts
# ---------------------------------------------------------------------------

class TestDetectAlerts:
    def _make_enriched(self, ticker, rating, score, dcf_fv, price, in_universe=True):
        gap = (price - dcf_fv) / dcf_fv if dcf_fv and price and dcf_fv > 0 else None
        return {
            'ticker': ticker,
            'rating': rating,
            '_composite_score': score,
            'dcf_fv': dcf_fv,
            'current_price': price,
            'valuation_gap_pct': gap,
            'valuation_gap_alert': False,
            'in_universe': in_universe,
        }

    def test_no_alerts_stable(self):
        holdings = [self._make_enriched('AAPL', 'BUY', 75.0, 200.0, 195.0)]
        prev = {'AAPL': {'rating': 'BUY', '_composite_score': 75.0}}
        alerts = detect_alerts(holdings, prev)
        assert alerts == []

    def test_rating_downgrade_alert(self):
        holdings = [self._make_enriched('AAPL', 'HOLD', 55.0, 200.0, 195.0)]
        prev = {'AAPL': {'rating': 'BUY', '_composite_score': 70.0}}
        alerts = detect_alerts(holdings, prev)
        downgrade_alerts = [a for a in alerts if a['alert_type'] == 'rating_downgrade']
        assert len(downgrade_alerts) == 1
        assert downgrade_alerts[0]['severity'] == 'HIGH'
        assert downgrade_alerts[0]['ticker'] == 'AAPL'

    def test_rating_upgrade_alert(self):
        holdings = [self._make_enriched('MSFT', 'BUY', 80.0, 500.0, 480.0)]
        prev = {'MSFT': {'rating': 'HOLD', '_composite_score': 55.0}}
        alerts = detect_alerts(holdings, prev)
        upgrade_alerts = [a for a in alerts if a['alert_type'] == 'rating_upgrade']
        assert len(upgrade_alerts) == 1
        assert upgrade_alerts[0]['severity'] == 'LOW'

    def test_valuation_gap_alert_above(self):
        # Price 30% above DCF FV → alert
        holdings = [self._make_enriched('TSLA', 'HOLD', 50.0, 100.0, 130.0)]
        alerts = detect_alerts(holdings, {}, valuation_gap_threshold=0.20)
        gap_alerts = [a for a in alerts if a['alert_type'] == 'valuation_gap']
        assert len(gap_alerts) == 1
        assert gap_alerts[0]['severity'] == 'MEDIUM'

    def test_valuation_gap_below_threshold_no_alert(self):
        # 10% gap, threshold 20% → no alert
        holdings = [self._make_enriched('NVDA', 'BUY', 80.0, 500.0, 550.0)]
        alerts = detect_alerts(holdings, {}, valuation_gap_threshold=0.20)
        gap_alerts = [a for a in alerts if a['alert_type'] == 'valuation_gap']
        assert gap_alerts == []

    def test_score_drop_alert(self):
        holdings = [self._make_enriched('CF', 'LEAN BUY', 45.0, 130.0, 125.0)]
        prev = {'CF': {'rating': 'LEAN BUY', '_composite_score': 70.0}}
        alerts = detect_alerts(holdings, prev, score_drop_threshold=10.0)
        score_alerts = [a for a in alerts if a['alert_type'] == 'score_drop']
        assert len(score_alerts) == 1
        assert score_alerts[0]['severity'] == 'MEDIUM'

    def test_not_in_universe_alert(self):
        holdings = [self._make_enriched('XYZ', 'NOT IN UNIVERSE', None, None, None, in_universe=False)]
        alerts = detect_alerts(holdings, {})
        univ_alerts = [a for a in alerts if a['alert_type'] == 'not_in_universe']
        assert len(univ_alerts) == 1
        assert univ_alerts[0]['severity'] == 'MEDIUM'

    def test_no_prev_results_no_crash(self):
        holdings = [self._make_enriched('AAPL', 'BUY', 75.0, 200.0, 195.0)]
        alerts = detect_alerts(holdings, {})  # empty prev
        assert isinstance(alerts, list)

    def test_high_alerts_sorted_first(self):
        holdings = [
            self._make_enriched('A', 'HOLD', 40.0, 100.0, 135.0),  # valuation gap (MEDIUM)
            self._make_enriched('B', 'PASS', 30.0, 100.0, 98.0),   # rating downgrade (HIGH)
        ]
        prev = {
            'A': {'rating': 'HOLD', '_composite_score': 40.0},
            'B': {'rating': 'BUY', '_composite_score': 70.0},
        }
        alerts = detect_alerts(holdings, prev, valuation_gap_threshold=0.20)
        assert alerts[0]['severity'] == 'HIGH'


# ---------------------------------------------------------------------------
# summarize_realized_gains
# ---------------------------------------------------------------------------

class TestSummarizeRealizedGains:
    def test_basic(self):
        realized = [
            {'ticker': 'A', 'shares_sold': 10, 'cost_basis': 100.0,
             'sale_price': 150.0, 'sale_date': '2025-03-01'},
            {'ticker': 'B', 'shares_sold': 5, 'cost_basis': 200.0,
             'sale_price': 250.0, 'sale_date': '2025-07-10'},
        ]
        result = summarize_realized_gains(realized)
        assert '2025' in result
        assert result['2025'] == pytest.approx(10 * 50 + 5 * 50)

    def test_multi_year(self):
        realized = [
            {'ticker': 'A', 'shares_sold': 10, 'cost_basis': 100.0,
             'sale_price': 120.0, 'sale_date': '2024-01-01'},
            {'ticker': 'B', 'shares_sold': 10, 'cost_basis': 100.0,
             'sale_price': 130.0, 'sale_date': '2025-01-01'},
        ]
        result = summarize_realized_gains(realized)
        assert result['2024'] == pytest.approx(200.0)
        assert result['2025'] == pytest.approx(300.0)

    def test_empty(self):
        assert summarize_realized_gains([]) == {}
        assert summarize_realized_gains(None) == {}


# ---------------------------------------------------------------------------
# _compute_return_since
# ---------------------------------------------------------------------------

class TestComputeReturnSince:
    def test_positive_return(self):
        idx = pd.date_range('2024-01-01', periods=100, freq='B', tz='UTC')
        series = pd.Series([100 + i for i in range(100)], index=idx)
        ret = _compute_return_since(series, '2024-01-01')
        assert ret == pytest.approx(99 / 100)

    def test_returns_none_for_empty_series(self):
        assert _compute_return_since(pd.Series(dtype=float), '2024-01-01') is None

    def test_returns_none_for_none(self):
        assert _compute_return_since(None, '2024-01-01') is None

    def test_future_since_date_returns_none(self):
        idx = pd.date_range('2024-01-01', periods=50, freq='B', tz='UTC')
        series = pd.Series(range(50), index=idx)
        ret = _compute_return_since(series, '2030-01-01')
        assert ret is None
