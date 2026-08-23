"""Macro-economic indicator fetcher.

Fetches five market-regime indicators from yfinance:
  1. VIX (implied volatility)
  2. Yield curve slope (10yr − 3mo Treasury)
  3. Credit spread proxy (HYG vs LQD 3-month returns)
  4. SPY momentum (price / 200-day SMA)
  5. Industrial relative strength (XLI vs SPY 3-month returns)

Also fetches sector ETF data for headwind/tailwind analysis.

Each indicator is fetched independently with try/except; missing data
returns None.  Results are session-cached to avoid re-fetching.
"""

import logging
import time
import numpy as np
import yfinance as yf

from data.yf_session import make_yf_session

logger = logging.getLogger(__name__)

_YF_SESSION = None


def _yf_session():
    """Shared curl_cffi session so every Yahoo call has a hard 15s timeout."""
    global _YF_SESSION
    if _YF_SESSION is None:
        _YF_SESSION = make_yf_session()
    return _YF_SESSION


# GICS sector → SPDR sector ETF
SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Consumer Cyclical': 'XLY',
    'Consumer Defensive': 'XLP',
    'Communication Services': 'XLC',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Basic Materials': 'XLB',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
}


class MacroClient:
    """Fetch macro-economic indicators from yfinance."""

    def __init__(self, request_delay=0.5):
        self._delay = request_delay
        self._last_req = 0
        self._cache = None
        self._sector_cache = None
        # yfinance calls share the module-level curl_cffi session (_yf_session)
        # so every request carries a hard socket timeout.

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_req = time.time()

    # ------------------------------------------------------------------
    # Individual indicator fetchers
    # ------------------------------------------------------------------

    def _fetch_vix(self):
        """Current VIX level (e.g. 18.5)."""
        self._throttle()
        try:
            price = yf.Ticker('^VIX', session=_yf_session()).info.get('regularMarketPrice')
            if price and 5 < price < 90:
                return round(float(price), 2)
        except Exception as e:
            logger.warning(f'macro: VIX fetch failed for ^VIX: {e}')
        return None

    def _fetch_yield_curve_slope(self):
        """10yr − 3mo Treasury yield spread as decimal (e.g. 0.012 = 1.2%)."""
        self._throttle()
        try:
            tnx = yf.Ticker('^TNX', session=_yf_session()).info.get('regularMarketPrice')
            self._throttle()
            irx = yf.Ticker('^IRX', session=_yf_session()).info.get('regularMarketPrice')
            if tnx is not None and irx is not None:
                slope = (float(tnx) - float(irx)) / 100.0
                if -0.06 < slope < 0.06:
                    return round(slope, 4)
        except Exception as e:
            logger.warning(f'macro: yield curve fetch failed for ^TNX/^IRX: {e}')
        return None

    def _fetch_credit_spread(self):
        """LQD 3m return − HYG 3m return.  Positive = stress (HYG lagging)."""
        try:
            self._throttle()
            hyg = yf.Ticker('HYG', session=_yf_session()).history(period='3mo')
            self._throttle()
            lqd = yf.Ticker('LQD', session=_yf_session()).history(period='3mo')
            if hyg is not None and lqd is not None and len(hyg) > 5 and len(lqd) > 5:
                hyg_ret = float(hyg['Close'].iloc[-1] / hyg['Close'].iloc[0]) - 1
                lqd_ret = float(lqd['Close'].iloc[-1] / lqd['Close'].iloc[0]) - 1
                return round(lqd_ret - hyg_ret, 4)
        except Exception as e:
            logger.warning(f'macro: credit spread fetch failed for HYG/LQD: {e}')
        return None

    def _fetch_spy_momentum(self):
        """SPY current price / 200-day SMA ratio (e.g. 1.04)."""
        try:
            self._throttle()
            hist = yf.Ticker('SPY', session=_yf_session()).history(period='1y')
            if hist is not None and len(hist) >= 200:
                current = float(hist['Close'].iloc[-1])
                sma200 = float(hist['Close'].iloc[-200:].mean())
                if sma200 > 0:
                    return round(current / sma200, 4)
        except Exception as e:
            logger.warning(f'macro: SPY momentum fetch failed for SPY: {e}')
        return None

    def _fetch_industrial_rs(self):
        """XLI 3m return − SPY 3m return.  Positive = cyclical strength."""
        try:
            self._throttle()
            xli = yf.Ticker('XLI', session=_yf_session()).history(period='3mo')
            self._throttle()
            spy = yf.Ticker('SPY', session=_yf_session()).history(period='3mo')
            if (xli is not None and spy is not None
                    and len(xli) > 5 and len(spy) > 5):
                xli_ret = float(xli['Close'].iloc[-1] / xli['Close'].iloc[0]) - 1
                spy_ret = float(spy['Close'].iloc[-1] / spy['Close'].iloc[0]) - 1
                return round(xli_ret - spy_ret, 4)
        except Exception as e:
            logger.warning(f'macro: industrial RS fetch failed for XLI/SPY: {e}')
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_macro_indicators(self):
        """Fetch all macro indicators.

        Returns:
            dict with keys: vix, yield_curve_slope, credit_spread_3m,
            spy_sma200_ratio, xli_rel_strength_3m.
            Values are floats or None on failure.
        """
        if self._cache is not None:
            return self._cache

        logger.info('Fetching macro indicators...')
        self._cache = {
            'vix': self._fetch_vix(),
            'yield_curve_slope': self._fetch_yield_curve_slope(),
            'credit_spread_3m': self._fetch_credit_spread(),
            'spy_sma200_ratio': self._fetch_spy_momentum(),
            'xli_rel_strength_3m': self._fetch_industrial_rs(),
        }
        return self._cache

    def fetch_sector_data(self):
        """Fetch sector ETF metrics for headwind/tailwind analysis.

        Returns:
            dict[sector_name → dict] with keys per sector:
              return_3m, return_6m, rel_strength_3m, sma200_ratio, volatility_30d
        """
        if self._sector_cache is not None:
            return self._sector_cache

        logger.info('Fetching sector ETF data...')
        # Fetch SPY as benchmark (1 year)
        self._throttle()
        try:
            spy_hist = yf.Ticker('SPY', session=_yf_session()).history(period='1y')
        except Exception as e:
            logger.warning(f'macro: benchmark fetch failed for SPY: {e}')
            spy_hist = None

        if spy_hist is None or len(spy_hist) < 60:
            self._sector_cache = {}
            return self._sector_cache

        spy_close = spy_hist['Close']
        spy_3m_ret = float(spy_close.iloc[-1] / spy_close.iloc[-63]) - 1 if len(spy_close) >= 63 else 0.0

        result = {}
        for sector, etf in SECTOR_ETF_MAP.items():
            self._throttle()
            try:
                hist = yf.Ticker(etf, session=_yf_session()).history(period='1y')
                if hist is None or len(hist) < 60:
                    continue
                close = hist['Close']
                n = len(close)

                # 3-month return (~63 trading days)
                ret_3m = float(close.iloc[-1] / close.iloc[-min(63, n)]) - 1
                # 6-month return (~126 trading days)
                # only when ~126 bars actually exist; otherwise a shorter
                # span would be mislabeled as a 6-month return
                ret_6m = float(close.iloc[-1] / close.iloc[-126]) - 1 if n >= 126 else None
                # Relative strength vs SPY (3m)
                rel_3m = ret_3m - spy_3m_ret
                # Price vs 200-day SMA
                sma200 = float(close.iloc[-min(200, n):].mean())
                sma_ratio = float(close.iloc[-1]) / sma200 if sma200 > 0 else None
                # 30-day annualized volatility
                daily_returns = close.pct_change().dropna().iloc[-30:]
                vol_30d = float(np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) >= 20 else None

                result[sector] = {
                    'etf': etf,
                    'return_3m': round(ret_3m, 4),
                    'return_6m': round(ret_6m, 4) if ret_6m is not None else None,
                    'rel_strength_3m': round(rel_3m, 4),
                    'sma200_ratio': round(sma_ratio, 4) if sma_ratio else None,
                    'volatility_30d': round(vol_30d, 4) if vol_30d else None,
                }
            except Exception as e:
                logger.warning(f'macro: sector ETF fetch failed for {etf}: {e}')
                continue

        self._sector_cache = result
        return self._sector_cache

    def fetch_commodity_data(self):
        """Fetch commodity and cross-sector proxies for narrative enrichment.

        Returns:
            dict with keys: 'oil', 'gold', 'bonds', 'consumer_sentiment'
            Each contains 'return_3m' and optionally 'return_6m'.
        """
        if hasattr(self, '_commodity_cache') and self._commodity_cache is not None:
            return self._commodity_cache

        logger.info('Fetching commodity & cross-sector data...')
        result = {}

        tickers = {
            'oil':   'CL=F',   # WTI Crude Oil futures
            'gold':  'GLD',    # Gold ETF
            'bonds': 'TLT',    # 20+ Year Treasury Bond ETF
        }

        for key, symbol in tickers.items():
            self._throttle()
            try:
                hist = yf.Ticker(symbol, session=_yf_session()).history(period='1y')
                if hist is not None and len(hist) >= 60:
                    close = hist['Close']
                    n = len(close)
                    ret_3m = float(close.iloc[-1] / close.iloc[-min(63, n)]) - 1
                    ret_6m = float(close.iloc[-1] / close.iloc[-126]) - 1 if n >= 126 else None
                    result[key] = {
                        'return_3m': round(ret_3m, 4),
                        'return_6m': round(ret_6m, 4) if ret_6m is not None else None,
                    }
            except Exception as e:
                logger.warning(f'macro: commodity fetch failed for {symbol}: {e}')
                continue

        # Consumer sentiment proxy: XLY/XLP ratio (discretionary vs staples)
        self._throttle()
        try:
            xly = yf.Ticker('XLY', session=_yf_session()).history(period='6mo')
            self._throttle()
            xlp = yf.Ticker('XLP', session=_yf_session()).history(period='6mo')
            if (xly is not None and xlp is not None
                    and len(xly) >= 60 and len(xlp) >= 60):
                # Current ratio vs 3-month-ago ratio
                ratio_now = float(xly['Close'].iloc[-1] / xlp['Close'].iloc[-1])
                ratio_3m = float(xly['Close'].iloc[-63] / xlp['Close'].iloc[-63])
                result['consumer_sentiment'] = {
                    'xly_xlp_ratio': round(ratio_now / ratio_3m, 4) if ratio_3m > 0 else None,
                }
        except Exception as e:
            logger.warning(f'macro: consumer sentiment fetch failed for XLY/XLP: {e}')

        self._commodity_cache = result
        return self._commodity_cache
