# tests/test_cost_of_equity.py
"""select_cost_of_equity: the four-level hierarchy with a fake price client."""
import numpy as np
import pandas as pd
import pytest

from models.capm import calculate_beta, weekly_returns, ROLLING_BETA_WINDOWS
from scripts.analyze_stock import select_cost_of_equity
from scripts.config import BETA_PRIOR_MEAN, BETA_PRIOR_SD, TERMINAL_GROWTH_RATE

RF = 0.04
ERP_T = 0.045


class _FakeYF:
    """Stands in for YFinanceClient: fetch_history(ticker, period) -> Series."""

    def __init__(self, series_by_ticker):
        self._s = series_by_ticker
        self.calls = []

    def fetch_history(self, ticker, period="5y"):
        self.calls.append((ticker, period))
        return self._s.get(ticker)


def _prices(seed=7, n_days=1300, beta=1.3, noise_sd=0.008):
    """Daily close series for a stock and market, stock = beta*market + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2021-01-04', periods=n_days)
    m = rng.normal(0.0004, 0.01, n_days)
    s = beta * m + rng.normal(0.0, noise_sd, n_days)
    market = pd.Series(100 * np.cumprod(1 + m), index=idx)
    stock = pd.Series(50 * np.cumprod(1 + s), index=idx)
    return stock, market


class TestLocalBetaPath:
    def test_capm_from_local_weekly_beta(self):
        stock, market = _prices()
        yf = _FakeYF({'TEST': stock, 'SPY': market})
        re, label, diag = select_cost_of_equity(
            {'info': {'beta': 1.2}}, RF, yf_client=yf, ticker='TEST', erp=ERP_T)

        assert label == 'capm'
        assert diag['beta_source'] == 'weekly_5y'
        assert diag['raw_beta'] == pytest.approx(1.3, abs=0.2)
        assert 0.5 < diag['shrink_weight'] <= 1.0
        assert re == pytest.approx(RF + diag['shrunk_beta'] * ERP_T)
        assert diag['n_observations'] <= ROLLING_BETA_WINDOWS['5y']
        assert ('TEST', '5y') in yf.calls and ('SPY', '5y') in yf.calls

    def test_headline_equals_rolling_5y_and_direct_regression(self):
        stock, market = _prices()
        yf = _FakeYF({'TEST': stock, 'SPY': market})
        _, _, diag = select_cost_of_equity(
            {'info': {}}, RF, yf_client=yf, ticker='TEST', erp=ERP_T)

        rb = diag['rolling_betas']
        assert set(rb) >= {'1y', '3y', '5y', 'stability'}
        assert rb['5y']['beta'] == pytest.approx(diag['raw_beta'], abs=1e-4)
        assert rb['5y']['n'] == diag['n_observations']
        assert rb['stability'] is not None

        s_ret, m_ret, _ = weekly_returns(stock, market)
        n5 = ROLLING_BETA_WINDOWS['5y']
        direct = calculate_beta(s_ret[-n5:], m_ret[-n5:],
                                prior_mean=BETA_PRIOR_MEAN, prior_sd=BETA_PRIOR_SD)
        assert diag['raw_beta'] == pytest.approx(direct['raw_beta'])
        assert diag['shrunk_beta'] == pytest.approx(direct['shrunk_beta'])

    def test_noisy_beta_is_shrunk_and_warns(self):
        stock, market = _prices(beta=1.0, noise_sd=0.05)
        yf = _FakeYF({'TEST': stock, 'SPY': market})
        with pytest.warns(RuntimeWarning, match=r'TEST: beta:'):
            _, label, diag = select_cost_of_equity(
                {'info': {}}, RF, yf_client=yf, ticker='TEST', erp=ERP_T)
        assert label == 'capm'
        assert diag['shrink_weight'] < 0.5
        # Shrunk beta sits between the raw estimate and the prior.
        lo, hi = sorted([diag['raw_beta'], BETA_PRIOR_MEAN])
        assert lo <= diag['shrunk_beta'] <= hi
        assert any('shrunk' in w for w in diag['warnings'])


class TestFallbacks:
    def test_yahoo_beta_is_blume_adjusted(self):
        re, label, diag = select_cost_of_equity(
            {'info': {'beta': 1.2}}, RF, yf_client=None, ticker='TEST', erp=ERP_T)
        expected_beta = (2 / 3) * 1.2 + (1 / 3) * BETA_PRIOR_MEAN
        assert label == 'capm (yahoo beta)'
        assert re == pytest.approx(RF + expected_beta * ERP_T)
        assert diag['beta_source'] == 'yahoo'
        assert diag['raw_beta'] == pytest.approx(1.2)
        assert diag['shrunk_beta'] == pytest.approx(expected_beta)
        assert diag['se_beta'] is None

    def test_local_path_failure_falls_through_to_yahoo(self):
        yf = _FakeYF({})  # no price history for anyone
        re, label, diag = select_cost_of_equity(
            {'info': {'beta': 0.8}}, RF, yf_client=yf, ticker='TEST', erp=ERP_T)
        assert label == 'capm (yahoo beta)'
        assert diag['beta_source'] == 'yahoo'

    def test_ggm_uses_forward_dividend(self):
        info = {'dividendRate': 2.0, 'currentPrice': 150.0}
        re, label, diag = select_cost_of_equity(
            {'info': info}, RF, yf_client=None, ticker='TEST', erp=ERP_T)
        assert label == 'ggm'
        assert re == pytest.approx(2.0 / 150.0 + TERMINAL_GROWTH_RATE)
        assert diag is None

    def test_buildup_last_resort(self):
        re, label, diag = select_cost_of_equity(
            {'info': {}}, RF, yf_client=None, ticker='TEST', erp=ERP_T)
        assert label == 'buildup'
        assert re == pytest.approx(RF + ERP_T)
        assert diag is None
