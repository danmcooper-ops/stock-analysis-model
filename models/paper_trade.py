"""Paper-trading rebalance: from a portfolio's members to a list of orders.

Pure functions; ``scripts/paper_trade.py`` does the I/O and talks to the
broker (``data/alpaca_client.py``).

The book is a portfolio grouping (``models/portfolio_groups.py``), resolved
against a snapshot and weighted exactly as the NAV ledger weighs it
(``data/portfolio_nav.entry_weights``), so the paper account's realized
return and the ledger's hypothetical one measure the same holdings and their
gap is the cost of trading.

Limits (``scripts/config.py``, ``PAPER_*``):

* a position trades only when its gap exceeds ``band`` of its target value
  (relative, because ~950 equal weights are ~0.1% each) and ``min_order``
  dollars; a member leaving the book is sold whatever its size;
* no position above ``max_position`` of equity;
* at most ``max_turnover`` of equity traded a day, exits first, then the
  largest gaps;
* sells before buys, buys scaled down to the cash the day has;
* a quote more than ``quote_max_gap`` from the snapshot close is skipped
  (bad print or news the model has not seen), and so is a ticker the broker
  cannot trade.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from data import portfolio_nav as pn
from models import portfolio_groups as pg


@dataclass(frozen=True)
class Limits:
    band: float = 0.25
    min_order: float = 1.0
    max_turnover: float = 0.25
    max_position: float = 0.10
    limit_slip: float = 0.005
    quote_max_gap: float = 0.15

    @classmethod
    def from_config(cls, config):
        return cls(band=config.PAPER_REBAL_BAND, min_order=config.PAPER_MIN_ORDER,
                   max_turnover=config.PAPER_MAX_TURNOVER,
                   max_position=config.PAPER_MAX_POSITION,
                   limit_slip=config.PAPER_LIMIT_SLIP,
                   quote_max_gap=config.PAPER_QUOTE_MAX_GAP)


def target_weights(portfolio, rows):
    """{ticker: weight} the book should hold, summing to 1 (cash, when a
    ``target`` weighting leaves some, under ``portfolio_nav.CASH``).

    Members without a usable price are dropped first: the NAV ledger leaves
    them unpriced too, and an order cannot be sized without one."""
    by_tk = pg.rows_by_ticker(rows)
    members = [t for t in pg.resolve_members(portfolio, by_tk)['members']
               if not pg.data_missing(by_tk.get(t))]
    if not members:
        return {}
    w = pn.entry_weights(portfolio, members, by_tk, cash=False)
    if w is None:
        return {t: 1.0 / len(members) for t in members}
    return w


def tradable_only(targets, assets):
    """*targets* restricted to what the broker can trade, renormalized.

    Alpaca trades no OTC lines (41 of the 954 BUY/LEAN BUY names on
    2026-10-28); left in, their ~4% would sit in cash and read as a
    shortfall against the hypothetical NAV, which holds them. Returns
    ``(weights, dropped_weight)``; cash (``portfolio_nav.CASH``) is kept."""
    keep = {t: w for t, w in targets.items()
            if t == pn.CASH or (assets.get(t) or {}).get('tradable')}
    dropped = sum(targets.values()) - sum(keep.values())
    tot = sum(keep.values())
    if tot <= 0:
        return {}, round(dropped, 6)
    return {t: w / tot for t, w in keep.items()}, round(dropped, 6)


def hold_set(positions, rows):
    """Held tickers the snapshot says nothing usable about, which must be
    kept rather than sold: a row with no price or identity (a quote outage
    below the systemic threshold dropped 1,548 rows' prices on 2026-09-25),
    or no row at all that is not a listing folded into a kept one. Selling
    them would turn a data gap into a round trip; a real exit is a row that
    is present and no longer a member."""
    by_tk = pg.rows_by_ticker(rows)
    aliases = pg.listing_aliases(by_tk)
    return {t for t in positions
            if (t in by_tk and pg.data_missing(by_tk[t]))
            or (t not in by_tk and t not in aliases)}


def _limit_price(px, side, slip):
    lp = px * (1 + slip) if side == 'buy' else px * (1 - slip)
    return round(lp, 4) if lp < 1 else round(lp, 2)


def _qty(dollars, px, fractionable):
    q = dollars / px
    return math.floor(q * 1e6) / 1e6 if fractionable else float(math.floor(q))


def plan_orders(targets, positions, account, prices, closes, assets, limits=None,
                hold=()):
    """The day's orders and what was skipped.

    *targets* {ticker: weight}; *positions* {ticker: qty} held; *account*
    needs ``equity`` and ``cash``; *prices* {ticker: live price};
    *closes* {ticker: snapshot close}; *assets* {ticker: {'tradable',
    'fractionable'}} from the broker; *hold* tickers (``hold_set``) are never
    sold out of the book.

    Returns ``{'orders': [...], 'skipped': [...], 'turnover': $,
    'buy_scale': x}``; each order is ``{ticker, side, qty, limit_price,
    dollars, reason}``, sells first.
    """
    limits = limits or Limits()
    equity = float(account['equity'])
    skipped = []
    cands = []
    for t in sorted(set(targets) | set(positions)):
        if t == pn.CASH:
            continue
        held = float(positions.get(t, 0.0))
        w = float(targets.get(t, 0.0))
        if held == 0 and w <= 0:
            continue
        if w <= 0 and t in hold:
            skipped.append({'ticker': t, 'reason': 'held: no usable row in the snapshot'})
            continue
        px = prices.get(t)
        if not px or px <= 0:
            skipped.append({'ticker': t, 'reason': 'no live price'})
            continue
        close = closes.get(t)
        if w > 0 and close and abs(px / close - 1) > limits.quote_max_gap:
            skipped.append({'ticker': t, 'reason': f'quote {px:g} is {px / close - 1:+.0%} '
                                                   f'from the snapshot close {close:g}'})
            continue
        a = assets.get(t) or {}
        if not a.get('tradable'):
            if held:
                skipped.append({'ticker': t, 'reason': 'held but not tradable at the broker'})
            else:
                skipped.append({'ticker': t, 'reason': 'not tradable at the broker'})
            continue
        tgt = min(w, limits.max_position) * equity
        cur = held * px
        gap = tgt - cur
        exit_ = w <= 0 and held > 0
        if not exit_:
            if abs(gap) < limits.min_order or (tgt > 0 and abs(gap) < limits.band * tgt):
                continue
        side = 'buy' if gap > 0 else 'sell'
        if exit_:
            qty = held
        elif side == 'sell':
            qty = min(held, _qty(-gap, px, a.get('fractionable')))
        else:
            qty = _qty(gap, px, a.get('fractionable'))
        if qty <= 0:
            skipped.append({'ticker': t, 'reason': 'rounds to zero shares (not fractionable)'})
            continue
        cands.append({'ticker': t, 'side': side, 'qty': qty, 'px': px,
                      'dollars': qty * px, 'exit': exit_,
                      'reason': 'left the book' if exit_ else
                      f'{cur:,.0f} -> {tgt:,.0f}'})

    # Turnover cap: exits first, then the largest gaps.
    cands.sort(key=lambda o: (not o['exit'], -o['dollars']))
    budget = limits.max_turnover * equity
    chosen, used = [], 0.0
    for o in cands:
        if used + o['dollars'] > budget:
            skipped.append({'ticker': o['ticker'], 'reason': 'daily turnover cap'})
            continue
        chosen.append(o)
        used += o['dollars']

    sells = [o for o in chosen if o['side'] == 'sell']
    buys = [o for o in chosen if o['side'] == 'buy']
    cash = float(account['cash']) + sum(o['dollars'] for o in sells)
    want = sum(o['dollars'] for o in buys)
    scale = 1.0
    if want > 0 and want > cash:
        scale = max(cash, 0.0) / want
    orders = []
    for o in sells:
        orders.append(_order(o, o['qty'], limits))
    for o in buys:
        frac = (assets.get(o['ticker']) or {}).get('fractionable')
        qty = o['qty'] if scale == 1.0 else _qty(o['dollars'] * scale, o['px'], frac)
        if qty <= 0 or qty * o['px'] < limits.min_order:
            skipped.append({'ticker': o['ticker'], 'reason': 'not enough cash'})
            continue
        orders.append(_order(o, qty, limits))
    return {'orders': orders, 'skipped': skipped,
            'turnover': round(sum(x['dollars'] for x in orders), 2),
            'buy_scale': round(scale, 4)}


def _order(o, qty, limits):
    return {'ticker': o['ticker'], 'side': o['side'], 'qty': qty,
            'limit_price': _limit_price(o['px'], o['side'], limits.limit_slip),
            'dollars': round(qty * o['px'], 2), 'reason': o['reason']}


def safety_checks(*, stop_file, coverage, systemic, snapshot_age, max_age,
                  account, loss_cap, market_open, open_orders, in_window):
    """Reasons to trade nothing today; empty means go.

    Every check fails closed: a missing account or clock answer is a reason
    to stop, and so is anything the snapshot says about its own quality."""
    out = []
    if stop_file:
        out.append('kill switch is set (portfolio/paper/STOP)')
    if not in_window:
        out.append('outside the trading window')
    if not market_open:
        out.append('the market is closed')
    if coverage and coverage.get('degraded'):
        out.append(f"snapshot is coverage-degraded ({coverage.get('rows')} rows "
                   f"vs {coverage.get('prior_rows')})")
    msg = pg.systemic_message(systemic) if systemic else None
    if msg:
        out.append(f'systemic day: {msg}')
    if snapshot_age is None or snapshot_age > max_age:
        out.append(f'snapshot is {snapshot_age} trading days old (max {max_age})')
    if not account:
        out.append('no account data')
    else:
        if account.get('trading_blocked') or account.get('account_blocked'):
            out.append('the broker has blocked trading on this account')
        last = account.get('last_equity') or 0
        if last > 0 and account['equity'] / last - 1 <= -loss_cap:
            out.append(f"equity {account['equity']:,.0f} is "
                       f"{account['equity'] / last - 1:.1%} since the prior close "
                       f"(cap -{loss_cap:.0%})")
    if open_orders:
        out.append(f'{open_orders} order(s) still open from an earlier run')
    return out
