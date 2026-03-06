# data/yfinance_client.py
import time
import yfinance as yf
import pandas as pd


class YFinanceClient:
    def __init__(self, request_delay=0.25):
        self._financials_cache = {}
        self._history_cache = {}
        self._request_delay = request_delay
        self._last_request_time = 0

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_delay:
            time.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

    def _retry(self, func, max_retries=2):
        for attempt in range(max_retries + 1):
            try:
                self._throttle()
                return func()
            except Exception:
                if attempt == max_retries:
                    raise
                time.sleep(1.0 * (attempt + 1))

    def fetch_financials(self, ticker):
        if ticker in self._financials_cache:
            return self._financials_cache[ticker]
        stock = yf.Ticker(ticker)

        def _fetch():
            return {
                'balance_sheet': stock.balance_sheet,
                'income_statement': stock.financials,
                'cash_flow': stock.cashflow,
            }

        financials = self._retry(_fetch)
        self._financials_cache[ticker] = financials
        return financials

    def fetch_shares_outstanding(self, ticker):
        stock = yf.Ticker(ticker)

        def _fetch():
            info = stock.info
            return info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')

        return self._retry(_fetch)

    def fetch_market_price(self, ticker):
        stock = yf.Ticker(ticker)

        def _fetch():
            info = stock.info
            return info.get('currentPrice') or info.get('regularMarketPrice')

        return self._retry(_fetch)

    def fetch_history(self, ticker, period="5y"):
        cache_key = (ticker, period)
        if cache_key in self._history_cache:
            return self._history_cache[cache_key]
        stock = yf.Ticker(ticker)

        def _fetch():
            return stock.history(period=period)['Close']

        history = self._retry(_fetch)
        self._history_cache[cache_key] = history
        return history
