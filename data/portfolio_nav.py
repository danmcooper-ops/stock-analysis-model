"""Point-in-time NAV ledger for portfolio groupings.

Each run appends one entry per portfolio to ``output/portfolio_nav.json``:
the members *as resolved that day* and a NAV that compounds the
equal-weighted return of the *previous* entry's members from the previous
market date to this one (rebalanced daily). Membership is recorded, not
re-derived, so a rule portfolio's history is what it actually held — no
look-ahead, and a later edit to the definition does not rewrite the past
(each entry carries the definition's hash; the report marks the changes).

Two benchmarks ride along with the same mechanics: SPY and the
equal-weighted universe (every ticker in the previous snapshot).

Prices. Both ends of every step are read from the *current* price parquets
(``data.price_store.asof_closes``), as of recorded market dates — never a
close remembered from an earlier run. The parquets are re-downloaded and
re-adjusted for splits and dividends, so a remembered close could turn a
2:1 split into a -50% day. The market date of a run is SPY's last bar on or
before the snapshot date; if the data lags, the next step simply spans the
missing bar, so no move is lost. A member with no parquet close falls back
to the two snapshots' ``price`` fields when their ratio is plausible (0.5-2,
a split guard); ``cov`` records the share of held members that were priced.

``rebuild`` replays the *current* definitions over archived snapshots
(entries flagged ``bf``, backfilled/hypothetical) and splices them in front
of the live history, rescaling the live NAVs so they continue the
backfilled path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile

from data.price_store import asof_closes
from models import portfolio_groups as pg

logger = logging.getLogger(__name__)

LEDGER_VERSION = 1
LEDGER_NAME = 'portfolio_nav.json'
MARKET_TICKER = 'SPY'
START_NAV = 100.0
# Snapshot-price fallback accepts a day's move only inside this ratio band.
FALLBACK_RATIO = (0.5, 2.0)
# Any single member's move between two runs outside this band, or starting
# below MIN_PRICE, is treated as bad data and left unpriced: yfinance
# history carries junk prints (GRKZF went $0.001 -> $15.52 on 2026-08-17,
# which alone moved the equal-weighted universe +672%).
PLAUSIBLE_RATIO = (0.25, 4.0)
MIN_PRICE = 0.01
LOW_COVERAGE = 0.8


def ledger_path(results_dir):
    return os.path.join(results_dir, LEDGER_NAME)


def empty_ledger():
    return {'version': LEDGER_VERSION, 'portfolios': {},
            'bench': {'spy': [], 'universe': []}}


def load_ledger(path):
    try:
        with open(path, encoding='utf-8') as f:
            led = json.load(f)
    except FileNotFoundError:
        return empty_ledger()
    except (OSError, ValueError) as e:
        logger.warning("portfolio NAV ledger %s unreadable (%s); starting a new one", path, e)
        return empty_ledger()
    if not isinstance(led, dict) or led.get('version') != LEDGER_VERSION:
        logger.warning("portfolio NAV ledger %s has an unknown version; starting a new one", path)
        return empty_ledger()
    led.setdefault('portfolios', {})
    led.setdefault('bench', {}).setdefault('spy', [])
    led['bench'].setdefault('universe', [])
    return led


def save_ledger(path, ledger):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.portfolio_nav.', suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(ledger, f, separators=(',', ':'))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def def_hash(p):
    """Hash of what decides membership (not name/color/description)."""
    key = {k: p.get(k) for k in ('tickers', 'exclude', 'rule')}
    blob = json.dumps(key, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:10]


def prices_closes_fn(prices_dir):
    """A ``closes(tickers, dates)`` backed by the price parquets."""
    def closes(tickers, dates):
        return asof_closes(prices_dir, dates, tickers) or {}
    return closes


def _step_return(held, m0, m1, closes, rows0, rows1):
    """Equal-weighted return of *held* from market date m0 to m1.

    Returns ``(ret, coverage)``; ``(0.0, 1.0)`` for an empty holding (cash)
    and ``(None, 0.0)`` when no member could be priced.
    """
    if not held:
        return 0.0, 1.0
    c0, c1 = closes.get(m0) or {}, closes.get(m1) or {}
    rets = []
    for t in held:
        a, b = c0.get(t), c1.get(t)
        if a and b:
            if a[1] >= MIN_PRICE and PLAUSIBLE_RATIO[0] <= b[1] / a[1] <= PLAUSIBLE_RATIO[1]:
                rets.append(b[1] / a[1] - 1.0)
            continue
        p0 = pg.js_num((rows0.get(t) or {}).get('price'))
        p1 = pg.js_num((rows1.get(t) or {}).get('price'))
        if p0 and p1 and p0 >= MIN_PRICE and FALLBACK_RATIO[0] <= p1 / p0 <= FALLBACK_RATIO[1]:
            rets.append(p1 / p0 - 1.0)
    if not rets:
        return None, 0.0
    return sum(rets) / len(rets), len(rets) / len(held)


def _advance(series, day, m1, held_for, closes, rows0, rows1, extra):
    """Append (or, for a re-run of the same day, replace) *day*'s entry.

    ``held_for(prev_entry)`` names what was held over the step. Returns the
    new entry, or None when *day* is older than the series' last entry.
    """
    if series and series[-1]['d'] > day:
        return None
    if series and series[-1]['d'] == day:
        series.pop()
    if not series:
        e = {'d': day, 'm': m1, 'nav': START_NAV, 'ret': None, 'cov': None}
    else:
        prev = series[-1]
        ret, cov = _step_return(held_for(prev), prev['m'], m1, closes, rows0, rows1)
        nav = prev['nav'] * (1.0 + ret) if ret is not None else prev['nav']
        e = {'d': day, 'm': m1, 'nav': round(nav, 6),
             'ret': None if ret is None else round(ret, 8),
             'cov': round(cov, 4)}
    e.update(extra)
    series.append(e)
    return e


def update(ledger, portfolios, day, rows, closes_fn, prev_rows=None,
           backfilled=False, benchmarks=True):
    """Advance every series in *ledger* to snapshot *day* (idempotent).

    *rows* are that day's snapshot rows; *prev_rows* the previous
    snapshot's (the universe benchmark's holding and the price fallback).
    Returns the market date used, or None when SPY has no bar near *day*
    (nothing is written).
    """
    mk = (closes_fn([MARKET_TICKER], [day]) or {}).get(day, {}).get(MARKET_TICKER)
    if not mk:
        logger.warning("portfolio NAV: no %s bar on or before %s; not updated", MARKET_TICKER, day)
        return None
    m1 = mk[0]
    by_tk = pg.rows_by_ticker(rows)
    prev_by = pg.rows_by_ticker(prev_rows or [])
    universe_held = sorted(prev_by)

    def prior(series):
        if not series:
            return None
        return series[-2] if series[-1]['d'] == day and len(series) > 1 else \
            (series[-1] if series[-1]['d'] < day else None)

    # One price query for every series' step.
    series_list = [ledger['portfolios'].setdefault(p['id'], []) for p in portfolios]
    need_t, need_d = {MARKET_TICKER}, {m1}
    for s in series_list:
        pe = prior(s)
        if pe:
            need_t.update(pe.get('members') or ())
            need_d.add(pe['m'])
    if benchmarks:
        for s in (ledger['bench']['spy'], ledger['bench']['universe']):
            pe = prior(s)
            if pe:
                need_d.add(pe['m'])
        if prior(ledger['bench']['universe']):
            need_t.update(universe_held)
    closes = closes_fn(sorted(need_t), sorted(need_d)) or {}

    flag = {'bf': True} if backfilled else {}
    for p, s in zip(portfolios, series_list, strict=True):
        members = pg.resolve_members(p, by_tk)['members']
        _advance(s, day, m1, lambda pe: pe.get('members') or [], closes, prev_by, by_tk,
                 dict(flag, n=len(members), members=members, h=def_hash(p)))
    if benchmarks:
        _advance(ledger['bench']['spy'], day, m1, lambda pe: [MARKET_TICKER],
                 closes, {}, {}, dict(flag))
        _advance(ledger['bench']['universe'], day, m1, lambda pe: universe_held,
                 closes, prev_by, by_tk, dict(flag, n=len(by_tk)))
    return m1


def rebuild(portfolios, snapshots, prices_dir, closes_fn=None):
    """Replay *portfolios* (today's definitions) over *snapshots*, a list of
    ``(date, rows)`` ascending, into a fresh ledger of backfilled entries.

    Prices come from two queries: SPY as of every snapshot date (the market
    dates), then every ticker as of those market dates.
    """
    led = empty_ledger()
    if not snapshots:
        return led
    if closes_fn is None:
        days = [d for d, _ in snapshots]
        mk = asof_closes(prices_dir, days, [MARKET_TICKER]) or {}
        mdates = sorted({v[MARKET_TICKER][0] for v in mk.values() if MARKET_TICKER in v})
        tickers = sorted({t for _, rows in snapshots for t in pg.rows_by_ticker(rows)} | {MARKET_TICKER})
        allc = asof_closes(prices_dir, mdates, tickers) or {}
        for d in days:            # SPY as of each snapshot date, for update()'s market lookup
            if MARKET_TICKER in mk.get(d, {}):
                allc.setdefault(d, {})[MARKET_TICKER] = mk[d][MARKET_TICKER]

        def closes_fn(tks, ds):
            want = set(tks)
            return {d: {t: v for t, v in (allc.get(d) or {}).items() if t in want} for d in ds}
    prev = None
    for day, rows in snapshots:
        update(led, portfolios, day, rows, closes_fn, prev_rows=prev, backfilled=True)
        prev = rows
    return led


def splice(live, backfilled):
    """Backfilled entries before the live history begins, then the live
    entries rescaled so they continue the backfilled NAV path."""
    if not backfilled:
        return live
    if not live:
        return backfilled
    first = live[0]['d']
    head = [e for e in backfilled if e['d'] < first]
    at = next((e for e in backfilled if e['d'] == first), None)
    factor = at['nav'] / live[0]['nav'] if at and live[0]['nav'] else None
    if factor is None and head:
        # No backfilled point on the live start date: continue from the last
        # backfilled NAV (a one-step gap in the path, not a jump).
        factor = head[-1]['nav'] / live[0]['nav']
    tail = []
    for e in live:
        e = dict(e)
        if factor:
            e['nav'] = round(e['nav'] * factor, 6)
        tail.append(e)
    if at and tail:
        tail[0]['ret'], tail[0]['cov'] = at['ret'], at['cov']
    return head + tail


def merge_rebuild(ledger, rebuilt, ids=None):
    """Splice *rebuilt* into *ledger* for the portfolio *ids* (all rebuilt
    ones when None) and the benchmarks."""
    for pid, bf in rebuilt['portfolios'].items():
        if ids is None or pid in ids:
            live = [e for e in ledger['portfolios'].get(pid, []) if not e.get('bf')]
            ledger['portfolios'][pid] = splice(live, bf)
    for k in ('spy', 'universe'):
        live = [e for e in ledger['bench'][k] if not e.get('bf')]
        ledger['bench'][k] = splice(live, rebuilt['bench'][k])
    return ledger


def payload(ledger, ids):
    """Compact series for the report: ``{'spy': [[d, nav]], 'universe': …,
    'pf': {id: [[d, nav, flags]]}}``. Flags: 1 backfilled, 2 definition
    changed at this entry, 4 priced coverage below LOW_COVERAGE, 8 held no
    members (the rule matched nothing; the step is cash)."""
    out = {'spy': [[e['d'], e['nav']] for e in ledger['bench']['spy']],
           'universe': [[e['d'], e['nav']] for e in ledger['bench']['universe']],
           'pf': {}}
    for pid in ids:
        rows, last_h = [], None
        for e in ledger['portfolios'].get(pid, []):
            f = (1 if e.get('bf') else 0)
            if last_h is not None and e.get('h') != last_h:
                f |= 2
            if e.get('cov') is not None and e['cov'] < LOW_COVERAGE:
                f |= 4
            if e.get('n') == 0:
                f |= 8
            last_h = e.get('h')
            rows.append([e['d'], e['nav'], f])
        if rows:
            out['pf'][pid] = rows
    return out


def window_return(series, days):
    """Return over the last *days* calendar days of a ``[{d, nav}]`` or
    ``[[d, nav, …]]`` series: from the last point on or before the window
    start to the latest point. None when the series doesn't reach back."""
    from datetime import date, timedelta
    pts = [(e['d'], e['nav']) if isinstance(e, dict) else (e[0], e[1]) for e in series]
    if len(pts) < 2:
        return None
    end_d, end_nav = pts[-1]
    if days is None:
        return end_nav / pts[0][1] - 1.0
    start = (date.fromisoformat(end_d) - timedelta(days=days)).isoformat()
    base = [p for p in pts if p[0] <= start]
    return end_nav / base[-1][1] - 1.0 if base else None


def rebased(series, start_d):
    """{date: nav/nav(start_d)} for points on or after *start_d*."""
    pts = [(e['d'], e['nav']) for e in series if e['d'] >= start_d]
    if not pts:
        return {}
    b = pts[0][1]
    return {d: n / b for d, n in pts}
