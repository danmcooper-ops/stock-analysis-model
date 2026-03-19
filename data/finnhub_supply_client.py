"""Finnhub peers and supply chain fetcher.

Uses the Finnhub API to retrieve peer companies and (if premium)
supply chain relationships for each company.

Requires FINNHUB_API_KEY environment variable (or pass via constructor).
Falls back to empty results when unavailable. Uses only stdlib.
"""

import json
import os
import time
import urllib.error
import urllib.request


class FinnhubSupplyClient:
    """Fetch peers and supply chain data from Finnhub API."""

    _BASE_URL = 'https://finnhub.io/api/v1'

    def __init__(self, api_key=None, request_delay=1.0):
        self._api_key = api_key or os.environ.get('FINNHUB_API_KEY', '')
        self._delay = request_delay
        self._last_req = 0
        self._peers_cache = {}    # ticker -> list of peer tickers
        self._supply_cache = {}   # ticker -> result dict
        self._supply_disabled = False  # set True after first 403

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_req = time.time()

    def _request(self, url, timeout=10):
        """Make a GET request. Returns parsed JSON or None.

        Sets self._last_http_code for callers that need to detect 403, etc.
        """
        self._throttle()
        self._last_http_code = 0
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'StockAnalyzer/1.0',
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                self._last_http_code = resp.status
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            self._last_http_code = e.code
            return None
        except Exception:
            return None

    @property
    def available(self):
        """True if an API key is configured."""
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # Peers (free tier)
    # ------------------------------------------------------------------

    def fetch_peers(self, ticker, max_peers=10):
        """Fetch peer tickers from Finnhub (free tier).

        Args:
            ticker: Stock ticker symbol.
            max_peers: Maximum peers to return.

        Returns:
            list[str]: Peer ticker symbols (excludes the input ticker).
        """
        if ticker in self._peers_cache:
            return self._peers_cache[ticker]

        if not self._api_key:
            self._peers_cache[ticker] = []
            return []

        url = (f'{self._BASE_URL}/stock/peers'
               f'?symbol={ticker}&token={self._api_key}')
        data = self._request(url)

        if not data or not isinstance(data, list):
            self._peers_cache[ticker] = []
            return []

        # Finnhub includes the ticker itself in the list — remove it
        peers = [t for t in data if t != ticker][:max_peers]
        self._peers_cache[ticker] = peers
        return peers

    # ------------------------------------------------------------------
    # Supply chain (premium tier — graceful fallback)
    # ------------------------------------------------------------------

    def fetch_supply_chain(self, ticker, max_suppliers=10, max_customers=10):
        """Fetch supplier and customer lists for a ticker.

        Args:
            ticker: Stock ticker symbol.
            max_suppliers: Maximum suppliers to return.
            max_customers: Maximum customers to return.

        Returns:
            dict with keys:
                suppliers (list[dict]): Each has symbol, name, country.
                customers (list[dict]): Same structure.
                available (bool): Whether the API was reachable.
        """
        if ticker in self._supply_cache:
            return self._supply_cache[ticker]

        empty = {'suppliers': [], 'customers': [], 'available': False}

        if not self._api_key or self._supply_disabled:
            self._supply_cache[ticker] = empty
            return empty

        url = (f'{self._BASE_URL}/stock/supply-chain'
               f'?symbol={ticker}&token={self._api_key}')
        data = self._request(url)

        # 403 means premium-only — disable for all future tickers
        if self._last_http_code == 403:
            self._supply_disabled = True
            print('Finnhub supply-chain requires premium tier — skipping.')
            self._supply_cache[ticker] = empty
            return empty

        if not data or not isinstance(data, dict):
            self._supply_cache[ticker] = empty
            return empty

        relationships = data.get('data', [])
        suppliers = []
        customers = []
        for rel in relationships:
            entry = {
                'symbol': rel.get('symbol', ''),
                'name': rel.get('name', ''),
                'country': rel.get('country', ''),
            }
            if rel.get('supplier'):
                suppliers.append(entry)
            if rel.get('customer'):
                customers.append(entry)

        result = {
            'suppliers': suppliers[:max_suppliers],
            'customers': customers[:max_customers],
            'available': True,
        }
        self._supply_cache[ticker] = result
        return result
