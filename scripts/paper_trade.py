#!/usr/bin/env python3
"""Paper-trade a portfolio grouping in an Alpaca paper account.

    python scripts/paper_trade.py rebalance             # dry run: print the orders
    python scripts/paper_trade.py rebalance --execute   # send them
    python scripts/paper_trade.py reconcile             # record fills and equity
    python scripts/paper_trade.py status                # paper vs hypothetical vs SPY
    python scripts/paper_trade.py stop | resume         # kill switch

The book is ``PAPER_PORTFOLIO`` (default ``paper``) in
``portfolio/portfolios.json``, resolved against the newest *finished*
snapshot (one whose ``run_summary_<date>.json`` exists, or the newest when no
nightly run holds its lock). The launchd agent
``scheduled-tasks/mac-mini/com.stockmodel.papertrade.plist`` runs ``tick``
at 9:45, 10:20 and 16:10 ET: the first that falls in the trading window
rebalances, the rest reconcile.

Everything the account says (orders, fills, equity) is written under
``portfolio/paper/``, gitignored and owner-only: the repo and its report are
public. The public NAV ledger keeps the portfolio's hypothetical series.

Exit status: 0 done (including a halt for a closed market or a safety
check, which is logged); 1 an error; 2 bad arguments.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root (simple key=value parser, as analyze_stock does).
_env_path = os.path.join(REPO, '.env')
if os.path.exists(_env_path):
    try:
        with open(_env_path, encoding='utf-8') as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
    except (OSError, UnicodeDecodeError) as _e:
        print(f"[WARN] Could not read .env ({_e}); continuing with process env only.",
              file=sys.stderr)

from data import portfolio_nav as pn  # noqa: E402
from data.alpaca_client import AlpacaError, AlpacaPaperClient  # noqa: E402
from data.snapshot_store import (  # noqa: E402
    list_snapshot_files, read_snapshot, snapshot_date_from_path, split_snapshot)
from models import paper_trade as pt  # noqa: E402
from models import portfolio_groups as pg  # noqa: E402
from scripts import config  # noqa: E402
from scripts.live_quotes import RUN_LOCK, run_lock_active  # noqa: E402
from scripts.market_open import market_status  # noqa: E402

log = logging.getLogger('paper_trade')
ET = ZoneInfo('America/New_York')
PAPER_DIR = os.environ.get('PAPER_DIR', os.path.join(REPO, 'portfolio', 'paper'))
OFFLINE_EQUITY = 100_000.0


def stop_path():
    return os.path.join(PAPER_DIR, 'STOP')


def ledger_path():
    return os.path.join(PAPER_DIR, 'ledger.jsonl')


def nav_path():
    return os.path.join(PAPER_DIR, 'nav.json')


def _private_dir():
    os.makedirs(PAPER_DIR, mode=0o700, exist_ok=True)


def append_ledger(record):
    _private_dir()
    fd = os.open(ledger_path(), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, separators=(',', ':'), default=str) + '\n')


def read_ledger():
    try:
        with open(ledger_path(), encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def load_nav():
    try:
        with open(nav_path(), encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_nav(series):
    _private_dir()
    tmp = nav_path() + '.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(series, f, indent=1)
    os.replace(tmp, nav_path())


# -- snapshot ---------------------------------------------------------------

def finished_snapshot(results_dir, lock_path=RUN_LOCK):
    """``(date, path)`` of the newest snapshot the nightly run has finished.

    A run marks its end by writing ``run_summary_<date>.json``; until then
    the enrich and render steps may still be rewriting the snapshot. With no
    run in progress (a dev box, or a run that died) the newest is taken."""
    files = list_snapshot_files(results_dir)
    locked = run_lock_active(lock_path)
    for d, p in reversed(files):
        if os.path.exists(os.path.join(results_dir, f'run_summary_{d}.json')):
            return d, p
        if not locked:
            return d, p
    return None


def trading_days_between(d0, d1):
    """Open sessions in (d0, d1]."""
    n, d = 0, d0 + timedelta(days=1)
    while d <= d1:
        if market_status(d)[0]:
            n += 1
        d += timedelta(days=1)
    return n


def in_window(now_et):
    (h0, m0), (h1, m1) = config.PAPER_WINDOW_ET
    return now_et.weekday() < 5 and (h0, m0) <= (now_et.hour, now_et.minute) < (h1, m1)


def _portfolio(pid, path=None):
    for p in pg.load_portfolios(path)['portfolios']:
        if p['id'] == pid:
            return p
    raise SystemExit(f"portfolio '{pid}' is not in portfolio/portfolios.json — create it with "
                     "scripts/portfolios.py create (see scheduled-tasks/MAC-MINI-SETUP.md §9)")


class OfflineBroker:
    """A dry run without keys: an empty $100k account, snapshot closes as
    quotes, every ticker tradable. Never used with --execute."""

    def __init__(self, closes):
        self.closes = closes

    def account(self):
        return {'equity': OFFLINE_EQUITY, 'last_equity': OFFLINE_EQUITY,
                'cash': OFFLINE_EQUITY, 'buying_power': OFFLINE_EQUITY,
                'trading_blocked': False, 'account_blocked': False}

    def positions(self):
        return {}

    def clock(self):
        return {'is_open': True}

    def open_orders(self):
        return []

    def assets(self):
        return {t: {'tradable': True, 'fractionable': True} for t in self.closes}

    def latest_prices(self, tickers, since=None):
        return {t: self.closes[t] for t in tickers if t in self.closes}


# -- commands ---------------------------------------------------------------

def cmd_rebalance(args, broker=None, now=None):
    now = now or datetime.now(ET)
    today = now.date()
    if args.snapshot:
        d = snapshot_date_from_path(args.snapshot)
        if not d:
            print(f'not a results_<date>.json snapshot: {args.snapshot}')
            return 2
        snap = (d, args.snapshot)
    else:
        snap = finished_snapshot(args.results_dir)
    if not snap:
        print('no finished snapshot found')
        return 1
    snap_date, snap_path = snap
    meta, rows = split_snapshot(read_snapshot(snap_path))
    by_tk = pg.rows_by_ticker(rows)
    prior = [(d, p) for d, p in list_snapshot_files(args.results_dir) if d < snap_date]
    systemic = None
    if prior:
        _pm, prev_rows = split_snapshot(read_snapshot(prior[-1][1]))
        systemic = pg.universe_stats(by_tk, pg.rows_by_ticker(prev_rows))
    portfolio = _portfolio(args.portfolio, args.portfolios_file)
    targets = pt.target_weights(portfolio, rows)
    closes = {t: pg.js_num(r.get('price')) for t, r in by_tk.items()
              if pg.js_num(r.get('price'))}

    offline = False
    if broker is None:
        try:
            broker = AlpacaPaperClient()
        except AlpacaError as e:
            if args.execute:
                print(f'cannot execute: {e}')
                return 1
            print(f'[offline dry run: {e}; using an empty ${OFFLINE_EQUITY:,.0f} account '
                  'and snapshot closes]')
            broker = OfflineBroker(closes)
            offline = True
    account = broker.account()
    clock = broker.clock()
    positions = broker.positions()
    open_orders = broker.open_orders()
    assets = broker.assets()
    targets, untradable = pt.tradable_only(targets, assets)
    wanted = sorted(set(targets) | set(positions))
    prices = broker.latest_prices(wanted, since=snap_date)
    # No IEX print since the snapshot: price the limit off the snapshot close.
    # A day limit order at close ± slip either fills near the model's price
    # or expires; it never chases a stale quote.
    at_close = [t for t in wanted if t not in prices and closes.get(t)]
    prices.update({t: closes[t] for t in at_close})

    reasons = pt.safety_checks(
        stop_file=os.path.exists(stop_path()),
        coverage=(meta.get('provenance') or {}).get('coverage'),
        systemic=systemic,
        snapshot_age=trading_days_between(date.fromisoformat(snap_date), today),
        max_age=config.PAPER_MAX_SNAPSHOT_AGE,
        account=account, loss_cap=config.PAPER_DAILY_LOSS_CAP,
        market_open=clock.get('is_open') and market_status(today)[0],
        open_orders=len(open_orders),
        in_window=offline or not args.execute or in_window(now))
    plan = pt.plan_orders(targets, positions, account, prices, closes, assets,
                          pt.Limits.from_config(config), hold=pt.hold_set(positions, rows))

    print(f"paper rebalance {today} from results_{snap_date} — portfolio '{portfolio['id']}': "
          f"{len(targets)} targets, {len(positions)} held, equity ${account['equity']:,.2f}, "
          f"cash ${account['cash']:,.2f}")
    print(f"  {len(plan['orders'])} orders, turnover ${plan['turnover']:,.2f} "
          f"({plan['turnover'] / account['equity']:.1%} of equity), "
          f"buy scale {plan['buy_scale']:.2f}, {len(plan['skipped'])} skipped, "
          f"{len(at_close)} priced at the snapshot close, "
          f"{untradable:.1%} of the book untradable at the broker (spread over the rest)")
    sells = [o for o in plan['orders'] if o['side'] == 'sell']
    buys = [o for o in plan['orders'] if o['side'] == 'buy']
    print(f"  sells {len(sells)} (${sum(o['dollars'] for o in sells):,.2f}), "
          f"buys {len(buys)} (${sum(o['dollars'] for o in buys):,.2f})")
    for o in sorted(plan['orders'], key=lambda o: -o['dollars'])[:args.show]:
        print(f"    {o['side']:4} {o['ticker']:7} {o['qty']:>12g} @ {o['limit_price']:<10g} "
              f"${o['dollars']:>10,.2f}  {o['reason']}")
    skip_kinds = {}
    for s in plan['skipped']:
        k = s['reason'].split(' ')[0] if s['reason'].startswith('quote') else s['reason']
        skip_kinds[k] = skip_kinds.get(k, 0) + 1
    for k, n in sorted(skip_kinds.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {n:>4}: {k}")

    record = {'type': 'rebalance', 'date': today.isoformat(), 'at': now.isoformat(),
              'snapshot': snap_date, 'portfolio': portfolio['id'],
              'execute': bool(args.execute), 'offline': offline,
              'equity': account['equity'], 'cash': account['cash'],
              'targets': len(targets), 'halted': reasons,
              'orders': [dict(o, close=closes.get(o['ticker'])) for o in plan['orders']],
              'skipped': plan['skipped'], 'turnover': plan['turnover'],
              'priced_at_close': len(at_close), 'untradable_weight': untradable}
    if reasons:
        print('  HALTED — no orders sent:')
        for r in reasons:
            print(f'    - {r}')
    if not args.execute or reasons:
        if not args.execute:
            print('  dry run: nothing sent (pass --execute to trade)')
        if args.execute:
            append_ledger(record)
        return 0

    sent, failed = [], []
    for o in plan['orders']:
        cid = f"paper-{today:%Y%m%d}-{o['side'][0]}-{o['ticker']}"[:48]
        try:
            res = broker.submit_order(o['ticker'], o['qty'], o['side'], o['limit_price'],
                                      client_order_id=cid)
            sent.append({'ticker': o['ticker'], 'id': (res or {}).get('id'), 'cid': cid})
        except AlpacaError as e:
            log.warning('order %s %s failed: %s', o['side'], o['ticker'], e)
            failed.append({'ticker': o['ticker'], 'error': str(e)[:300]})
    record.update(sent=sent, failed=failed)
    append_ledger(record)
    print(f"  sent {len(sent)}, failed {len(failed)}")
    return 0


def cmd_reconcile(args, broker=None, now=None):
    now = now or datetime.now(ET)
    today = now.date().isoformat()
    broker = broker or AlpacaPaperClient()
    start = datetime.combine(now.date(), datetime.min.time(), ET).isoformat()
    orders = broker.orders_since(start)
    planned = {}
    for r in read_ledger():
        if r.get('type') == 'rebalance' and r.get('date') == today and r.get('sent'):
            planned = {(o['ticker'], o['side']): o for o in r.get('orders') or []}
    fills = []
    for o in orders:
        if not str(o.get('client_order_id', '')).startswith('paper-'):
            continue
        qty = float(o.get('filled_qty') or 0)
        avg = float(o['filled_avg_price']) if o.get('filled_avg_price') else None
        t = o['symbol'].replace('.', '-')
        p = planned.get((t, o['side'])) or {}
        f = {'ticker': t, 'side': o['side'], 'status': o.get('status'), 'qty': qty,
             'ordered': float(o.get('qty') or 0), 'avg': avg,
             'limit': p.get('limit_price'), 'close': p.get('close')}
        if avg and p.get('close'):
            sign = 1 if o['side'] == 'buy' else -1
            f['slip_vs_close'] = round(sign * (avg / p['close'] - 1), 5)
        fills.append(f)
    account = broker.account()
    filled = [f for f in fills if f['qty'] > 0]
    slips = [f['slip_vs_close'] for f in filled if 'slip_vs_close' in f]
    rec = {'type': 'reconcile', 'date': today, 'at': now.isoformat(),
           'equity': account['equity'], 'cash': account['cash'],
           'orders': len(fills), 'filled': len(filled),
           'filled_dollars': round(sum(f['qty'] * (f['avg'] or 0) for f in filled), 2),
           'mean_slip_vs_close': round(sum(slips) / len(slips), 5) if slips else None,
           'fills': fills}
    append_ledger(rec)
    series = [e for e in load_nav() if e['d'] != today]
    series.append({'d': today, 'equity': account['equity'],
                   'last_equity': account['last_equity']})
    save_nav(sorted(series, key=lambda e: e['d']))
    print(f"paper reconcile {today}: {len(filled)}/{len(fills)} orders filled "
          f"(${rec['filled_dollars']:,.2f}), equity ${account['equity']:,.2f}"
          + (f", mean cost vs snapshot close {rec['mean_slip_vs_close']:+.2%}" if slips else ''))
    return 0


def _ret_since(series, d0, key):
    pts = [e for e in series if e['d'] >= d0]
    if len(pts) < 2 or not pts[0][key]:
        return None
    return pts[-1][key] / pts[0][key] - 1


def cmd_status(args):
    nav = load_nav()
    print(f"kill switch: {'SET' if os.path.exists(stop_path()) else 'off'}")
    if not nav:
        print('no paper NAV recorded yet (reconcile writes one a day)')
        return 0
    d0, d1 = nav[0]['d'], nav[-1]['d']
    led = pn.load_ledger(pn.ledger_path(args.results_dir))
    hyp = [{'d': e['d'], 'nav': e['nav']} for e in led['portfolios'].get(args.portfolio, [])]
    spy = [{'d': e['d'], 'nav': e['nav']} for e in led['bench'].get('spy', [])]
    print(f"{d0} .. {d1} ({len(nav)} days), equity ${nav[-1]['equity']:,.2f}")
    for label, series, key in (('paper (realized)', nav, 'equity'),
                               (f"'{args.portfolio}' hypothetical", hyp, 'nav'),
                               ('SPY', spy, 'nav')):
        r = _ret_since(series, d0, key)
        print(f"  {label:28} {'n/a' if r is None else f'{r:+.2%}'}")
    recs = [r for r in read_ledger() if r.get('type') == 'reconcile']
    slips = [r['mean_slip_vs_close'] for r in recs if r.get('mean_slip_vs_close') is not None]
    if slips:
        print(f"  mean fill cost vs snapshot close: {sum(slips) / len(slips):+.2%} "
              f"over {len(slips)} days")
    halts = [r for r in read_ledger() if r.get('type') == 'rebalance' and r.get('halted')]
    if halts:
        print(f"  last halt {halts[-1]['date']}: {'; '.join(halts[-1]['halted'])}")
    return 0


def cmd_tick(args, broker=None, now=None):
    """What the launchd agent runs (9:45, 10:20 and 16:10 ET): rebalance
    once a day, inside the window, until one goes through; otherwise
    reconcile. The 16:10 reconcile records the closing equity the NAV uses."""
    now = now or datetime.now(ET)
    if not market_status(now.date())[0]:
        print(f'{now.date()}: market closed, nothing to do')
        return 0
    # Done once a rebalance went through: not halted, and not one whose every
    # order the broker refused (an outage at 9:45 is retried at 10:20).
    done = any(r.get('type') == 'rebalance' and r.get('date') == now.date().isoformat()
               and r.get('execute') and not r.get('halted')
               and (r.get('sent') or not r.get('orders')) for r in read_ledger())
    if not done and in_window(now):
        args.execute, args.snapshot = True, None
        return cmd_rebalance(args, broker=broker, now=now)
    return cmd_reconcile(args, broker=broker, now=now)


def cmd_stop(args):
    _private_dir()
    with open(stop_path(), 'w', encoding='utf-8') as f:
        f.write(f'set {datetime.now(ET).isoformat()}\n')
    print(f'kill switch set ({stop_path()}): rebalance will send nothing')
    return 0


def cmd_resume(args):
    if os.path.exists(stop_path()):
        os.remove(stop_path())
    print('kill switch cleared')
    return 0


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--results-dir', default=os.path.join(REPO, 'output'))
    ap.add_argument('--portfolio', default=config.PAPER_PORTFOLIO)
    ap.add_argument('--portfolios-file', default=None)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sp = sub.add_parser('rebalance', help='plan (and with --execute, send) the day\'s orders')
    sp.add_argument('--execute', action='store_true', help='send the orders (default: dry run)')
    sp.add_argument('--snapshot', help='a results_<date>.json to trade from (default: newest finished)')
    sp.add_argument('--show', type=int, default=15, help='largest orders to print')
    sub.add_parser('reconcile', help="record today's fills and equity")
    sp = sub.add_parser('tick', help='scheduled entry: rebalance once a day, else reconcile')
    sp.add_argument('--show', type=int, default=15)
    sub.add_parser('status', help='paper vs hypothetical vs SPY')
    sub.add_parser('stop', help='set the kill switch')
    sub.add_parser('resume', help='clear the kill switch')
    args = ap.parse_args(argv)
    try:
        return {'rebalance': cmd_rebalance, 'reconcile': cmd_reconcile, 'status': cmd_status,
                'tick': cmd_tick, 'stop': cmd_stop, 'resume': cmd_resume}[args.cmd](args)
    except AlpacaError as e:
        print(f'broker error: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
