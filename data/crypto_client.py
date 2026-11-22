"""CoinGecko: the market table behind Investing › Crypto.

    /coins/markets          price, market cap, volume and the 24h/7d/30d/1y
                            changes for the largest coins, in one call
    /coins/{id}             a coin's categories (decides what is excluded)
    /coins/{id}/market_chart  daily USD closes for the volatility, drawdown
                            and correlation columns

No key is needed. The public tier allows only a handful of calls a minute
and answers 429 past it, so every call goes through one slow ``Throttle``
and a 429 waits out the minute and retries. ``COINGECKO_API_KEY`` (a free
"demo" key) lifts the limit; with it the interval drops to two seconds.

A coin's categories change rarely, so they are kept on disk
(``data/cache/crypto/coin_categories.json``) for ``CATEGORY_TTL_DAYS``. A
night then costs about 21 calls: the market list and one history per coin.
"""

import json
import logging
import os
import time
from datetime import date, datetime, timezone

import requests

from data.throttle import Throttle

logger = logging.getLogger(__name__)

_BASE = 'https://api.coingecko.com/api/v3'
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'crypto')
CATEGORY_TTL_DAYS = 30
# Measured 2026-11-22: about five keyless calls a minute before 429s begin.
PUBLIC_INTERVAL = 13.0
KEYED_INTERVAL = 2.0
RETRIES_429 = 4
WAIT_429 = 65.0


class CoinGeckoError(RuntimeError):
    pass


class CoinGeckoClient:
    def __init__(self, api_key=None, cache_dir=_CACHE, interval=None, session=None,
                 sleep=time.sleep):
        self.api_key = api_key if api_key is not None else os.environ.get('COINGECKO_API_KEY')
        self.throttle = Throttle(interval if interval is not None else
                                 (KEYED_INTERVAL if self.api_key else PUBLIC_INTERVAL))
        self.cache_dir = cache_dir
        self.session = session or requests.Session()
        self.sleep = sleep
        self.calls = 0

    # ------------------------------------------------------------ transport
    def _get(self, path, params=None):
        headers = {'Accept': 'application/json'}
        if self.api_key:
            headers['x-cg-demo-api-key'] = self.api_key
        for attempt in range(RETRIES_429 + 1):
            self.throttle()
            self.calls += 1
            try:
                r = self.session.get(_BASE + path, params=params, headers=headers, timeout=30)
            except requests.RequestException as e:
                if attempt >= RETRIES_429:
                    raise CoinGeckoError(f'{path}: {e}') from e
                self.sleep(10)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt >= RETRIES_429:
                    raise CoinGeckoError(f'{path}: HTTP {r.status_code} after {attempt + 1} tries')
                wait = WAIT_429
                try:
                    wait = max(float(r.headers.get('Retry-After', WAIT_429)), 5.0)
                except ValueError:
                    pass
                logger.info('coingecko %s: HTTP %s, waiting %.0fs', path, r.status_code, wait)
                self.sleep(wait)
                continue
            if r.status_code != 200:
                raise CoinGeckoError(f'{path}: HTTP {r.status_code}')
            try:
                return r.json()
            except ValueError as e:
                raise CoinGeckoError(f'{path}: not JSON') from e
        raise CoinGeckoError(f'{path}: gave up')  # pragma: no cover

    # ------------------------------------------------------------ endpoints
    def markets(self, n=60):
        """The *n* largest coins by market cap, CoinGecko's rank order."""
        d = self._get('/coins/markets', {
            'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': n, 'page': 1,
            'sparkline': 'false', 'price_change_percentage': '24h,7d,30d,1y'})
        if not isinstance(d, list):
            raise CoinGeckoError('/coins/markets: unexpected shape')
        return d

    def history(self, coin_id, days=365):
        """Daily ``[(date, close)]``, oldest first. CoinGecko stamps each day at
        00:00 UTC; the last point is the live price, which is dropped so every
        point is a whole day. Keyless, CoinGecko serves at most 365 days and
        answers 401 past it (2026-11-22)."""
        d = self._get(f'/coins/{coin_id}/market_chart',
                      {'vs_currency': 'usd', 'days': days, 'interval': 'daily'})
        out = {}
        for ms, px in (d or {}).get('prices') or []:
            if px is None:
                continue
            t = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
            if t.hour or t.minute:          # the live, intraday point
                continue
            out[t.date()] = float(px)
        return sorted(out.items())

    def categories(self, coin_id, today=None):
        """A coin's category names, from the disk cache when fresh."""
        today = today or date.today()
        cache = self._load_cats()
        hit = cache.get(coin_id)
        if hit and (today - date.fromisoformat(hit['fetched'])).days < CATEGORY_TTL_DAYS:
            return hit['categories']
        d = self._get(f'/coins/{coin_id}', {
            'localization': 'false', 'tickers': 'false', 'market_data': 'false',
            'community_data': 'false', 'developer_data': 'false', 'sparkline': 'false'})
        cats = [c for c in (d or {}).get('categories') or [] if c]
        cache[coin_id] = {'fetched': today.isoformat(), 'categories': cats}
        self._save_cats(cache)
        return cats

    # ------------------------------------------------------------ cache
    def _cats_path(self):
        return os.path.join(self.cache_dir, 'coin_categories.json')

    def _load_cats(self):
        try:
            with open(self._cats_path(), encoding='utf-8') as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_cats(self, cache):
        os.makedirs(self.cache_dir, exist_ok=True)
        tmp = self._cats_path() + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=0, sort_keys=True)
        os.replace(tmp, self._cats_path())
