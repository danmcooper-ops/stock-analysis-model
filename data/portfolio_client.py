# data/portfolio_client.py
"""Portfolio client — loads holdings and fetches live price data."""
import json
import os
from datetime import date, datetime

import pandas as pd


class PortfolioClient:
    """Loads a portfolio holdings file and fetches current prices via yfinance.

    Parameters
    ----------
    holdings_path : str
        Path to the holdings JSON file (default: portfolio/holdings.json).
    yf_client : YFinanceClient, optional
        Existing YFinanceClient instance to reuse (avoids duplicate throttling).
    """

    def __init__(self, holdings_path='portfolio/holdings.json', yf_client=None):
        self._holdings_path = holdings_path
        self._yf = yf_client
        self._price_cache = {}

    # ------------------------------------------------------------------
    # Holdings I/O
    # ------------------------------------------------------------------

    def load_holdings(self):
        """Parse the holdings JSON file.

        Returns
        -------
        dict
            {
                'portfolio_name': str,
                'benchmark': str,
                'holdings': list of dict,
                'realized_gains': list of dict,
            }
        Raises FileNotFoundError if the file does not exist.
        """
        if not os.path.exists(self._holdings_path):
            raise FileNotFoundError(
                f"Holdings file not found: {self._holdings_path}. "
                "Create portfolio/holdings.json to use the portfolio tracker."
            )
        with open(self._holdings_path) as f:
            data = json.load(f)

        # Validate required fields
        for i, h in enumerate(data.get('holdings', [])):
            for field in ('ticker', 'shares', 'cost_basis', 'purchase_date'):
                if field not in h:
                    raise ValueError(
                        f"Holding #{i} is missing required field '{field}'"
                    )
            # Normalize purchase_date to ISO string
            h['purchase_date'] = str(h['purchase_date'])

        return {
            'portfolio_name': data.get('portfolio_name', 'My Portfolio'),
            'benchmark': data.get('benchmark', 'SPY'),
            'holdings': data.get('holdings', []),
            'realized_gains': data.get('realized_gains', []),
        }

    def save_portfolio_state(self, portfolio_state, output_dir='output'):
        """Write the enriched portfolio state to output/portfolio_{date}.json."""
        os.makedirs(output_dir, exist_ok=True)
        today_str = date.today().isoformat()
        filename = os.path.join(output_dir, f'portfolio_{today_str}.json')
        with open(filename, 'w') as f:
            json.dump(portfolio_state, f, indent=2, default=str)
        return filename

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def fetch_current_prices(self, tickers):
        """Fetch the most recent closing price for each ticker.

        Uses YFinanceClient.fetch_history with a 5-day window and takes the
        last available close. Falls back to None for any ticker that fails.

        Parameters
        ----------
        tickers : list of str

        Returns
        -------
        dict
            {ticker: float or None}
        """
        prices = {}
        for ticker in tickers:
            if ticker in self._price_cache:
                prices[ticker] = self._price_cache[ticker]
                continue
            try:
                if self._yf is not None:
                    history = self._yf.fetch_history(ticker, period='5d')
                else:
                    import yfinance as yf
                    history = yf.Ticker(ticker).history(period='5d')['Close']

                if history is not None and len(history) > 0:
                    price = float(history.iloc[-1])
                else:
                    price = None
            except Exception as e:
                print(f"  [portfolio] price fetch failed for {ticker}: {e}")
                price = None
            self._price_cache[ticker] = price
            prices[ticker] = price
        return prices

    def fetch_benchmark_history(self, benchmark='SPY', period='2y'):
        """Fetch benchmark Close price series for return comparison.

        Parameters
        ----------
        benchmark : str
            Ticker for the benchmark (default 'SPY').
        period : str
            yfinance period string (default '2y').

        Returns
        -------
        pandas.Series
            DatetimeIndex → Close price, or empty Series on failure.
        """
        try:
            if self._yf is not None:
                return self._yf.fetch_history(benchmark, period=period)
            import yfinance as yf
            return yf.Ticker(benchmark).history(period=period)['Close']
        except Exception as e:
            print(f"  [portfolio] benchmark fetch failed for {benchmark}: {e}")
            return pd.Series(dtype=float)

    def fetch_ticker_history(self, ticker, since_date, period='2y'):
        """Fetch Close series for a ticker for alpha calculation.

        Parameters
        ----------
        ticker : str
        since_date : str
            ISO date string for the purchase date (used for period selection).
        period : str
            yfinance period string.

        Returns
        -------
        pandas.Series
            DatetimeIndex → Close price, or empty Series on failure.
        """
        try:
            if self._yf is not None:
                return self._yf.fetch_history(ticker, period=period)
            import yfinance as yf
            return yf.Ticker(ticker).history(period=period)['Close']
        except Exception as e:
            print(f"  [portfolio] history fetch failed for {ticker}: {e}")
            return pd.Series(dtype=float)
