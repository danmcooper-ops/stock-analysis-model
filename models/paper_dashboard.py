"""The Paper account dashboard: the Alpaca paper account against the model.

Built at render time by ``report_html._stage_paper_dashboard`` from the
paper bot's private records (``data/paper_account``) and the public NAV
ledger, and shipped beside the report as ``paper.json``. The portfolio's
page in Portfolios draws it as its Paper account sub-tab. It is **public**
(the user's choice, 2026-10-29): equity, positions and fills of a paper
account, never keys.

"Model" is the portfolio's hypothetical NAV (``output/portfolio_nav.json``),
the book the account is trying to hold, at no cost and fully invested.
"Paper" is the account's closing equity. Both are rebased to 100 on the
first day the account recorded equity, so their gap is everything trading
costs: slippage, the days the book took to build, cash, untradable names
and skipped orders. ``build`` says how much of that gap is the build.

Pure: callers pass the records in.
"""
from __future__ import annotations

PAPER_DASHBOARD_VERSION = 1
ACTIVITY_DAYS = 15


def _at_or_before(series, d, key):
    """The *key* value of the last entry dated on or before *d*."""
    v = None
    for e in series:
        if e['d'] > d:
            break
        if e.get(key) is not None:
            v = e[key]
    return v


def _rebased(series, dates, d0, key):
    base = _at_or_before(series, d0, key)
    if not base:
        return [None] * len(dates)
    out = []
    for d in dates:
        v = _at_or_before(series, d, key)
        out.append(round(100 * v / base, 4) if v else None)
    return out


def _last(vals):
    for v in reversed(vals):
        if v is not None:
            return v
    return None


def build(portfolio_id, records, nav, model_series, spy_series, run_date=None):
    """The dashboard payload, or None before the account has a day of equity.

    *records* are ``ledger.jsonl``'s, *nav* ``nav.json``'s; *model_series*
    and *spy_series* the NAV ledger's entries (``{d, nav}``) for the
    portfolio and SPY. Only what is dated on or before *run_date* is used,
    so a render never pairs the account's later equity with an older
    model point."""
    cut = run_date or '9999-12-31'
    nav = sorted((e for e in nav if e.get('d') and e['d'] <= cut and e.get('equity')),
                 key=lambda e: e['d'])
    if not nav:
        return None
    records = [r for r in records if (r.get('date') or '') <= cut
               and r.get('portfolio', portfolio_id) == portfolio_id]
    model_series = sorted(model_series or [], key=lambda e: e['d'])
    spy_series = sorted(spy_series or [], key=lambda e: e['d'])
    dates = [e['d'] for e in nav]
    d0 = dates[0]
    paper = _rebased(nav, dates, d0, 'equity')
    model = _rebased(model_series, dates, d0, 'nav')
    spy = _rebased(spy_series, dates, d0, 'nav')

    def ret(vals):
        v = _last(vals)
        return None if v is None else round(v / 100 - 1, 6)

    rets = {'paper': ret(paper), 'model': ret(model), 'spy': ret(spy)}
    rets['delta'] = (None if rets['paper'] is None or rets['model'] is None
                     else round(rets['paper'] - rets['model'], 6))
    rets['vs_spy'] = (None if rets['paper'] is None or rets['spy'] is None
                      else round(rets['paper'] - rets['spy'], 6))

    rebs = [r for r in records if r.get('type') == 'rebalance' and r.get('execute')]
    recs = [r for r in records if r.get('type') == 'reconcile']
    last_reb = next((r for r in reversed(rebs) if r.get('weights')), None)
    last_rec = next((r for r in reversed(recs) if r.get('positions') is not None), None)
    equity = nav[-1]['equity']
    targets = (last_reb or {}).get('weights') or {}
    positions = (last_rec or {}).get('positions') or {}

    holdings = []
    for t in sorted(set(targets) | set(positions)):
        if t.startswith('$'):
            continue
        p = positions.get(t) or {}
        tw = targets.get(t, 0.0)
        aw = (p.get('value') or 0.0) / equity if equity else 0.0
        status = ('held' if p and tw else 'to sell' if p else 'not yet bought')
        holdings.append({'t': t, 'target': round(tw, 6), 'actual': round(aw, 6),
                         'delta': round(aw - tw, 6), 'qty': p.get('qty'),
                         'value': p.get('value'), 'cost': p.get('avg_cost'),
                         'pl': p.get('pl'), 'status': status})
    held_w = sum(min(h['actual'], h['target']) for h in holdings if h['target'] > 0)
    invested = sum(h['actual'] for h in holdings)
    build_ = {'targets': sum(1 for h in holdings if h['target'] > 0),
              'held': sum(1 for h in holdings if h['status'] == 'held'),
              'pending': sum(1 for h in holdings if h['status'] == 'not yet bought'),
              'to_sell': sum(1 for h in holdings if h['status'] == 'to sell'),
              'target_held': round(held_w, 4), 'invested': round(invested, 4),
              'cash': round(1 - invested, 4)}

    by_day = {}
    for r in rebs:
        by_day.setdefault(r['date'], {})['reb'] = r
    for r in recs:
        by_day.setdefault(r['date'], {})['rec'] = r
    activity = []
    for d in sorted(by_day, reverse=True)[:ACTIVITY_DAYS]:
        reb, rec = by_day[d].get('reb') or {}, by_day[d].get('rec') or {}
        activity.append({
            'd': d, 'orders': len(reb.get('orders') or []) if reb else None,
            'sent': len(reb.get('sent') or []) if reb else None,
            'failed': len(reb.get('failed') or []) if reb else None,
            'filled': rec.get('filled'), 'filled_dollars': rec.get('filled_dollars'),
            'slip': rec.get('mean_slip_vs_close'), 'turnover': reb.get('turnover'),
            'snapshot': reb.get('snapshot'), 'halted': reb.get('halted') or []})
    skipped = {}
    for s in (last_reb or {}).get('skipped') or []:
        k = 'quote moved since the snapshot' if s['reason'].startswith('quote') else s['reason']
        skipped[k] = skipped.get(k, 0) + 1

    return {'version': PAPER_DASHBOARD_VERSION, 'portfolio': portfolio_id,
            'as_of': dates[-1], 'start': d0, 'snapshot': (last_reb or {}).get('snapshot'),
            'account': {'equity': equity, 'start_equity': nav[0]['equity'],
                        'cash': (last_rec or {}).get('cash')},
            'returns': rets,
            'series': {'d': dates, 'paper': paper, 'model': model, 'spy': spy},
            'build': build_, 'holdings': holdings, 'activity': activity,
            'skipped': skipped,
            'untradable': (last_reb or {}).get('untradable_weight')}
