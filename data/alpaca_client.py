"""Alpaca paper-trading client (scripts/paper_trade.py).

Plain REST over ``requests``, like the other clients in ``data/``: the
handful of endpoints the paper bot needs did not justify ``alpaca-py``.

**Paper only.** The client refuses to build against anything but
``paper-api.alpaca.markets``. There is no flag that relaxes this: trading
real money from this repo is a separate decision, made in code review, not
an environment variable.

Keys come from ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` (the paper
account's keys). Market data (latest trades) uses the free IEX feed.
"""
from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import requests

from data.throttle import Throttle

log = logging.getLogger(__name__)

PAPER_BASE = 'https://paper-api.alpaca.markets'
DATA_BASE = 'https://data.alpaca.markets'
# Alpaca allows 200 requests a minute on the free plan.
REQUEST_INTERVAL = 0.35
TIMEOUT = 20
QUOTE_BATCH = 200


class LiveEndpointError(RuntimeError):
    """Raised when the client is pointed at anything but the paper API."""


class AlpacaError(RuntimeError):
    pass


def to_alpaca_symbol(ticker):
    """Yahoo-style share classes (BRK-A) are dotted at Alpaca (BRK.A)."""
    return str(ticker).upper().replace('-', '.')


def from_alpaca_symbol(symbol):
    return str(symbol).upper().replace('.', '-')


def format_qty(qty):
    """Plain decimal for the order API: never exponent notation, never
    rounded up past what *qty* holds (``f'{q:g}'`` sent 1.31027e+06 and
    turned 999.9995 into 1000)."""
    return f'{qty:.9f}'.rstrip('0').rstrip('.')


def _check_paper(base_url):
    host = urlparse(base_url).hostname or ''
    if host != urlparse(PAPER_BASE).hostname:
        raise LiveEndpointError(
            f'refusing to trade against {base_url!r}: only {PAPER_BASE} is allowed')


class AlpacaPaperClient:
    def __init__(self, key=None, secret=None, base_url=None, session=None,
                 throttle=None):
        base_url = base_url or os.environ.get('ALPACA_BASE_URL') or PAPER_BASE
        _check_paper(base_url)
        key = key or os.environ.get('ALPACA_API_KEY')
        secret = secret or os.environ.get('ALPACA_SECRET_KEY')
        if not key or not secret:
            raise AlpacaError('ALPACA_API_KEY and ALPACA_SECRET_KEY must be set')
        self.base_url = base_url.rstrip('/')
        self.session = session or requests.Session()
        self.session.headers.update({'APCA-API-KEY-ID': key,
                                     'APCA-API-SECRET-KEY': secret})
        self.throttle = throttle or Throttle(REQUEST_INTERVAL)

    def _req(self, method, url, **kw):
        self.throttle()
        try:
            r = self.session.request(method, url, timeout=TIMEOUT, **kw)
        except requests.RequestException as e:
            # One error type for callers: a timed-out order is recorded as
            # failed (its client_order_id stops a retry from doubling it).
            raise AlpacaError(f'{method} {url}: {e}') from e
        if r.status_code >= 400:
            raise AlpacaError(f'{method} {url}: HTTP {r.status_code} {r.text[:300]}')
        return r.json() if r.content else None

    def _api(self, method, path, **kw):
        return self._req(method, self.base_url + path, **kw)

    # -- account -----------------------------------------------------------
    def account(self):
        a = self._api('GET', '/v2/account')
        return {'equity': float(a['equity']), 'last_equity': float(a['last_equity']),
                'cash': float(a['cash']), 'buying_power': float(a['buying_power']),
                'trading_blocked': bool(a.get('trading_blocked')),
                'account_blocked': bool(a.get('account_blocked'))}

    def positions(self):
        """{ticker (Yahoo form): qty} of every open position."""
        return {t: p['qty'] for t, p in self.positions_detail().items()}

    def positions_detail(self):
        """{ticker: {qty, avg_cost, value, price, pl}} of every open position;
        ``pl`` is the unrealized return on cost."""
        def num(p, k):
            try:
                return float(p[k])
            except (KeyError, TypeError, ValueError):
                return None
        return {from_alpaca_symbol(p['symbol']): {
                    'qty': num(p, 'qty'), 'avg_cost': num(p, 'avg_entry_price'),
                    'value': num(p, 'market_value'), 'price': num(p, 'current_price'),
                    'pl': num(p, 'unrealized_plpc')}
                for p in self._api('GET', '/v2/positions') or []}

    def clock(self):
        c = self._api('GET', '/v2/clock')
        return {'is_open': bool(c['is_open']), 'timestamp': c.get('timestamp')}

    def assets(self):
        """{ticker: {'tradable', 'fractionable'}} over active US equities."""
        out = {}
        for a in self._api('GET', '/v2/assets',
                           params={'status': 'active', 'asset_class': 'us_equity'}) or []:
            out[from_alpaca_symbol(a['symbol'])] = {
                'tradable': bool(a.get('tradable')),
                'fractionable': bool(a.get('fractionable'))}
        return out

    # -- orders ------------------------------------------------------------
    def open_orders(self):
        return self._api('GET', '/v2/orders', params={'status': 'open', 'limit': 500}) or []

    def cancel_open_orders(self):
        return self._api('DELETE', '/v2/orders')

    def submit_order(self, ticker, qty, side, limit_price, client_order_id=None):
        body = {'symbol': to_alpaca_symbol(ticker), 'qty': format_qty(qty), 'side': side,
                'type': 'limit', 'time_in_force': 'day',
                'limit_price': f'{limit_price:.4f}' if limit_price < 1 else f'{limit_price:.2f}'}
        if client_order_id:
            body['client_order_id'] = client_order_id
        return self._api('POST', '/v2/orders', json=body)

    def orders_since(self, after_iso):
        """Every order submitted after *after_iso*, oldest first. Paged: one
        call returns at most 500, and a day with many exits sends more."""
        out, after, seen = [], after_iso, set()
        while True:
            page = self._api('GET', '/v2/orders',
                             params={'status': 'all', 'after': after, 'limit': 500,
                                     'direction': 'asc'}) or []
            new = [o for o in page if o['id'] not in seen]
            out.extend(new)
            seen.update(o['id'] for o in new)
            if len(page) < 500 or not new:
                return out
            after = page[-1]['submitted_at']

    # -- market data -------------------------------------------------------
    def latest_prices(self, tickers, since=None):
        """{ticker: last trade price} from the IEX feed. Tickers with no
        trade, or (with *since*, an ISO date) whose last IEX trade is older
        than that date, are left out: IEX carries a small share of volume,
        and a thin name's last print there can be days old."""
        out = {}
        tickers = list(dict.fromkeys(tickers))
        for i in range(0, len(tickers), QUOTE_BATCH):
            chunk = tickers[i:i + QUOTE_BATCH]
            syms = ','.join(to_alpaca_symbol(t) for t in chunk)
            try:
                data = self._req('GET', DATA_BASE + '/v2/stocks/trades/latest',
                                 params={'symbols': syms, 'feed': 'iex'})
            except AlpacaError as e:
                log.warning('alpaca latest trades failed for %d symbols: %s', len(chunk), e)
                continue
            for sym, t in ((data or {}).get('trades') or {}).items():
                px = t.get('p')
                if since and str(t.get('t') or '')[:10] < since:
                    continue
                if px and px > 0:
                    out[from_alpaca_symbol(sym)] = float(px)
        return out
