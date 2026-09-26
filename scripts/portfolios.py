# scripts/portfolios.py
"""Manage portfolio groupings (portfolio/portfolios.json).

A portfolio is a named set of tickers — hand-picked, rule-driven, or both —
and a ticker may belong to any number of them. See models/portfolio_groups.py
for the file format and the rule semantics.

    python scripts/portfolios.py list
    python scripts/portfolios.py create semis --name Semiconductors --tickers NVDA,AMD,TSM
    python scripts/portfolios.py create energy-buys --name "Energy BUYs" \\
        --sector Energy --rating "BUY,LEAN BUY" --min mcap=2e9
    python scripts/portfolios.py add semis AVGO MU
    python scripts/portfolios.py remove energy-buys XOM      # rule member -> exclude
    python scripts/portfolios.py show energy-buys            # against the latest snapshot
    python scripts/portfolios.py import ~/Downloads/portfolios.json --dry-run
    python scripts/portfolios.py import 'https://…/#pf=eyJpZCI6…'

Rule min/max on percent columns are fractions (--min mos=0.2 means MoS >= 20%).
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.snapshot_store import SnapshotStore, list_snapshot_files, load_snapshot_file  # noqa: E402
from models import portfolio_groups as pg  # noqa: E402

logger = logging.getLogger('portfolios')


def _split_csv(s):
    return [x.strip() for x in s.split(',') if x.strip()] if s else []


def _parse_kv(items, what):
    out = {}
    for it in items or ():
        k, sep, v = it.partition('=')
        if not sep or not k:
            raise SystemExit(f"--{what} expects KEY=VALUE, got {it!r}")
        out[k.strip()] = v.strip()
    return out


def _rule_from_args(a):
    """Build a rule from the create/set-rule flags; None when none given."""
    if a.rule_json:
        return json.loads(a.rule_json)
    mins = {k: float(v) for k, v in _parse_kv(a.min, 'min').items()}
    maxs = {k: float(v) for k, v in _parse_kv(a.max, 'max').items()}
    txts = _parse_kv(a.contains, 'contains')
    cf = [{'key': k, 'min': mins.get(k), 'max': maxs.get(k)}
          for k in sorted(set(mins) | set(maxs))]
    cf += [{'key': k, 'txt': v} for k, v in sorted(txts.items())]
    rule = {'ratings': _split_csv(a.rating) or None,
            'sectors': a.sector or None,
            'countries': a.country or None,
            'cf': cf}
    if rule['ratings'] is None and rule['sectors'] is None \
            and rule['countries'] is None and not cf:
        return None
    return rule


def _add_rule_flags(sp):
    sp.add_argument('--rating', help='comma list, e.g. "BUY,LEAN BUY"')
    sp.add_argument('--sector', action='append',
                    help='sector name (repeat for several)')
    sp.add_argument('--country', action='append',
                    help='country name (repeat for several)')
    sp.add_argument('--min', action='append', metavar='KEY=V',
                    help='column lower bound (fractions for %% columns)')
    sp.add_argument('--max', action='append', metavar='KEY=V',
                    help='column upper bound')
    sp.add_argument('--contains', action='append', metavar='KEY=TEXT',
                    help='case-insensitive text match on a column')
    sp.add_argument('--rule-json', help='the whole rule as JSON')


def _find(doc, pid):
    for p in doc['portfolios']:
        if p['id'] == pid:
            return p
    raise SystemExit(f"no portfolio {pid!r} (have: "
                     f"{', '.join(p['id'] for p in doc['portfolios']) or 'none'})")


def _latest_rows(results_dir, day=None):
    files = list_snapshot_files(results_dir)
    if day:
        files = [f for f in files if f[0] == day]
    if not files:
        raise SystemExit(f"no results snapshot{' for ' + day if day else ''} in {results_dir}")
    d, path = files[-1]
    _, rows = load_snapshot_file(path)
    return d, rows


def prior_rows(results_dir, before, columns):
    """``(date, rows)`` of the newest snapshot strictly before *before*
    (``YYYY-MM-DD``), or ``(None, [])``.

    Served from the DuckDB snapshot store when it holds that date — only
    *columns* are read — else by parsing the JSON. The store answers an
    unknown column with NULL, which a rule would read as a present-but-N/A
    ``_gate_<key>`` and so drop every row; gate columns the store lacks are
    therefore never requested.
    """
    files = [(d, p) for d, p in list_snapshot_files(results_dir) if d < before]
    if not files:
        return None, []
    day, path = files[-1]
    try:
        store = SnapshotStore.for_results_dir(results_dir)
        if store is not None:
            with store:
                if store.has_date(day):
                    have = {c.lower() for c in store.columns()}
                    cols = [c for c in columns
                            if not c.startswith('_gate_') or c.lower() in have]
                    return day, store.rows(day, cols)
    except Exception as e:  # the JSON is canonical; the store is a shortcut
        logger.warning("portfolios: snapshot store read for %s failed (%s); "
                       "parsing the JSON", day, e)
    _, rows = load_snapshot_file(path)
    return day, rows


def _fmt_pct(v):
    return '—' if not isinstance(v, (int, float)) else f"{v:+.0%}"


def _fmt_num(v, spec='.0f'):
    return '—' if not isinstance(v, (int, float)) else format(v, spec)


# ---------------------------------------------------------------- commands

def cmd_list(doc, a):
    if not doc['portfolios']:
        print("No portfolios yet. Create one with: portfolios.py create <id> --name ...")
        return
    for p in pg.with_colors(doc['portfolios']):
        kind = ('rule + ' if p['rule'] else '') + f"{len(p['tickers'])} picked"
        excl = f", {len(p['exclude'])} excluded" if p['exclude'] else ''
        print(f"{p['id']:<24} {p['name']:<28} {kind}{excl}  {p['color']}")
        if p['rule']:
            print(f"{'':<24} rule: {pg._rule_summary(p['rule'])}")


def cmd_create(doc, a):
    pid = a.id or pg.slugify(a.name)
    if any(p['id'] == pid for p in doc['portfolios']):
        raise SystemExit(f"portfolio {pid!r} already exists")
    doc['portfolios'].append({
        'id': pid, 'name': a.name or pid, 'color': a.color,
        'description': a.description or '',
        'tickers': _split_csv(a.tickers), 'exclude': [],
        'rule': _rule_from_args(a)})
    return f"created {pid}"


def cmd_set_rule(doc, a):
    p = _find(doc, a.id)
    p['rule'] = None if a.clear else _rule_from_args(a)
    return f"{a.id}: rule {'cleared' if p['rule'] is None else 'set'}"


def cmd_edit(doc, a):
    p = _find(doc, a.id)
    if a.name:
        p['name'] = a.name
    if a.color:
        p['color'] = a.color
    if a.description is not None:
        p['description'] = a.description
    return f"updated {a.id}"


def cmd_delete(doc, a):
    _find(doc, a.id)
    doc['portfolios'] = [p for p in doc['portfolios'] if p['id'] != a.id]
    return f"deleted {a.id}"


def cmd_add(doc, a):
    p = _find(doc, a.id)
    tks = [t.upper() for t in a.tickers]
    p['exclude'] = [t for t in p['exclude'] if t not in tks]
    p['tickers'] = sorted(set(p['tickers']) | set(tks))
    return f"{a.id}: added {' '.join(tks)}"


def cmd_remove(doc, a):
    p = _find(doc, a.id)
    msgs = []
    for t in (x.upper() for x in a.tickers):
        hit = t in p['tickers']
        if hit:
            p['tickers'] = [x for x in p['tickers'] if x != t]
            msgs.append(f"removed {t}")
        # A rule could (re)admit it, so pin it out explicitly.
        if p['rule'] and t not in p['exclude']:
            p['exclude'] = sorted(set(p['exclude']) | {t})
            msgs.append(f"excluded {t} from the rule")
        elif not hit and not p['rule']:
            msgs.append(f"{t} was not in {a.id}")
    return f"{a.id}: " + ', '.join(msgs)


def cmd_show(doc, a):
    targets = [_find(doc, a.id)] if a.id else doc['portfolios']
    day, rows = _latest_rows(a.results_dir, a.date)
    by_tk = pg.rows_by_ticker(rows)
    print(f"Snapshot {day} ({len(by_tk)} tickers)")
    for p in targets:
        res = pg.resolve_members(p, by_tk)
        print(f"\n{p['name']} [{p['id']}] — {len(res['members'])} members"
              + (f", {len(res['missing'])} not in universe" if res['missing'] else ''))
        if p['rule']:
            print(f"  rule: {pg._rule_summary(p['rule'])}")
            for c in p['rule']['cf']:
                if not any(c['key'] in r for r in rows):
                    print(f"  warning: no row has a {c['key']!r} field, so this rule matches nothing")
        print('  ' + _stats_line(pg.portfolio_stats(res['members'], by_tk)))
        manual, ruled = set(res['manual']), set(res['ruled'])
        print(f"  {'Ticker':<8} {'Rating':<9} {'MoS':>6} {'Score':>6}  {'Src':<5} Sector")
        for t in res['members']:
            r = by_tk[t]
            src = 'both' if t in manual and t in ruled else ('rule' if t in ruled else 'pick')
            print(f"  {t:<8} {str(r.get('rating') or '—'):<9} {_fmt_pct(r.get('mos')):>6} "
                  f"{_fmt_num(r.get('_composite_score')):>6}  {src:<5} {r.get('sector') or '—'}")
        if res['missing']:
            print(f"  not in universe: {' '.join(res['missing'])}")


def _stats_line(st):
    mix = ' '.join(f"{k}:{v}" for k, v in st['ratings'].items() if v) or 'no ratings'
    conc = ''
    if st['top_sector']:
        conc = f"  top sector {st['top_sector']} {st['top_sector_weight']:.0%}"
        if st['concentrated']:
            conc += ' (concentrated)'
    return (f"{mix}  median MoS {_fmt_pct(st['median_mos'])}  "
            f"score {_fmt_num(st['median_score'])}  "
            f"spread {_fmt_pct(st['median_spread'])}{conc}")


def build_alerts_report(portfolios, rows, prev_rows, day, prev_day):
    """The text of output/portfolio_alerts_<date>.txt."""
    by_tk, prev_by_tk = pg.rows_by_ticker(rows), pg.rows_by_ticker(prev_rows)
    names = {p['id']: p['name'] for p in portfolios}
    alerts = pg.portfolio_alerts(portfolios, by_tk, prev_by_tk, run_date=day)
    lines = [f"Portfolio alerts {day} (vs {prev_day or 'no prior snapshot'})", '']
    if not portfolios:
        return '\n'.join(lines + ['No portfolios defined.']) + '\n'
    for p in portfolios:
        res = pg.resolve_members(p, by_tk)
        st = pg.portfolio_stats(res['members'], by_tk, prev_by_tk)
        chg = f"  today: +{st['upgrades']} up / -{st['downgrades']} down" \
            if prev_day else ''
        lines.append(f"{p['name']} [{p['id']}]: {st['n']} stocks{chg}")
        lines.append('  ' + _stats_line(st))
    lines += ['', f"{len(alerts)} alert(s)" + ('' if prev_day else
              ' — no prior snapshot, so no change alerts')]
    for al in alerts:
        pfs = ', '.join(names.get(i, i) for i in al['portfolios'])
        lines.append(f"  [{al['severity']:<6}] {al['message']}  — {pfs}")
    return '\n'.join(lines) + '\n'


def cmd_alerts(doc, a):
    day, rows = _latest_rows(a.results_dir, a.date)
    prev_day, prev = prior_rows(a.results_dir, day, pg.rule_columns(doc['portfolios']))
    text = build_alerts_report(doc['portfolios'], rows, prev, day, prev_day)
    print(text, end='')
    if a.out:
        out = a.out.replace('{date}', day)
        with open(out, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"wrote {out}")


def load_snapshots(results_dir, columns, since=None):
    """``[(date, rows)]`` ascending for every snapshot (optionally from
    *since*), reading only *columns* from the snapshot store for the dates it
    holds and parsing the JSON for the rest."""
    files = [(d, p) for d, p in list_snapshot_files(results_dir) if not since or d >= since]
    out, store = [], None
    try:
        store = SnapshotStore.for_results_dir(results_dir)
    except Exception as e:
        logger.warning("portfolios: snapshot store unavailable (%s); parsing JSON", e)
    try:
        have = {c.lower() for c in store.columns()} if store is not None else set()
        cols = [c for c in columns if not c.startswith('_gate_') or c.lower() in have]
        for d, path in files:
            if store is not None and store.has_date(d):
                out.append((d, store.rows(d, cols)))
            else:
                out.append((d, load_snapshot_file(path)[1]))
    finally:
        if store is not None:
            store.close()
    return out


def _nav_table(led, portfolios):
    from data import portfolio_nav as pn
    spy, uni = led['bench']['spy'], led['bench']['universe']
    lines = [f"{'Portfolio':<26} {'Since':<11} {'Total':>7} {'vs SPY':>7} {'vs EW':>7} "
             f"{'1M':>6} {'3M':>6}  Notes"]
    for p in portfolios:
        s = led['portfolios'].get(p['id']) or []
        if not s:
            lines.append(f"{p['name'][:26]:<26} (no history yet)")
            continue
        start = s[0]['d']
        tot = pn.window_return(s, None)
        rs, ru = pn.rebased(spy, start), pn.rebased(uni, start)
        last = s[-1]['d']
        vs_spy = (tot - (rs[last] - 1)) if tot is not None and last in rs else None
        vs_ew = (tot - (ru[last] - 1)) if tot is not None and last in ru else None
        notes = []
        bf = sum(1 for e in s if e.get('bf'))
        if bf:
            notes.append(f"{bf} backfilled day(s)")
        low = sum(1 for e in s if e.get('cov') is not None and e['cov'] < pn.LOW_COVERAGE)
        if low:
            notes.append(f"{low} low-coverage day(s)")
        if len({e.get('h') for e in s}) > 1:
            notes.append('definition changed')
        lines.append(f"{p['name'][:26]:<26} {start:<11} {_fmt_pct(tot):>7} {_fmt_pct(vs_spy):>7} "
                     f"{_fmt_pct(vs_ew):>7} {_fmt_pct(pn.window_return(s, 30)):>6} "
                     f"{_fmt_pct(pn.window_return(s, 91)):>6}  {', '.join(notes)}")
    return '\n'.join(lines)


def cmd_nav(doc, a):
    from data import portfolio_nav as pn
    path = pn.ledger_path(a.results_dir)
    led = pn.load_ledger(path)
    pfs = doc['portfolios']
    if a.id:
        pfs = [_find(doc, i) for i in a.id]
    if a.rebuild:
        snaps = load_snapshots(a.results_dir, pg.rule_columns(pfs), since=a.since)
        print(f"replaying {len(pfs)} portfolio(s) over {len(snaps)} snapshot(s)"
              + (f" ({snaps[0][0]} .. {snaps[-1][0]})" if snaps else ''))
        rebuilt = pn.rebuild(pfs, snaps, a.prices_dir)
        pn.merge_rebuild(led, rebuilt, ids={p['id'] for p in pfs})
        pn.save_ledger(path, led)
        print(f"wrote {path}")
    print(_nav_table(led, pfs))


def cmd_import(doc, a):
    """Bring in a report export or share link.

    An export whose ``base_rev`` is the file's current revision holds the
    browser's edits on top of exactly this file, so it is taken whole —
    deletions included ("fast-forward"). Anything else (a share link, an
    export from an older file) merges: new ids are added, and ids that
    differ from the file are refused unless --overwrite.
    """
    src = a.source
    text = open(src, encoding='utf-8').read() if os.path.exists(src) else src
    try:
        inc_doc, base_rev = pg.decode_share(text)
    except (ValueError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read {src!r}: {e}") from None
    incoming = inc_doc['portfolios']
    cur_rev = pg.revision(doc)
    if a.replace or (base_rev == cur_rev and not a.merge):
        if not a.replace:
            print(f"fast-forward: the export was made from this file (rev {cur_rev})")
        new = incoming
    else:
        if base_rev:
            print(f"merge: the export was made from rev {base_rev}; the file is now {cur_rev}")
        try:
            new = pg.merge_portfolios(doc['portfolios'], incoming, overwrite=a.overwrite)
        except ValueError as e:
            raise SystemExit(str(e)) from None
    new = pg.with_colors(pg.normalize({'version': pg.SCHEMA_VERSION,
                                       'portfolios': new})['portfolios'])
    lines = pg.diff_portfolios(pg.with_colors(doc['portfolios']), new)
    print('\n'.join(lines) if lines else 'no changes')
    if a.dry_run or not lines:
        return None
    doc['portfolios'] = new
    return f"imported ({len(lines)} change line(s))"


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file', default=pg.DEFAULT_PORTFOLIOS_PATH,
                    help='definitions file (default: portfolio/portfolios.json)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('list', help='list portfolios')

    sp = sub.add_parser('create', help='create a portfolio')
    sp.add_argument('id', nargs='?', help='slug id (default: from --name)')
    sp.add_argument('--name', required=True)
    sp.add_argument('--color', help='#rrggbb (default: palette)')
    sp.add_argument('--description')
    sp.add_argument('--tickers', help='comma list of hand-picked tickers')
    _add_rule_flags(sp)

    sp = sub.add_parser('set-rule', help='replace (or --clear) a portfolio rule')
    sp.add_argument('id')
    sp.add_argument('--clear', action='store_true')
    _add_rule_flags(sp)

    sp = sub.add_parser('edit', help='rename / recolor / describe')
    sp.add_argument('id')
    sp.add_argument('--name')
    sp.add_argument('--color')
    sp.add_argument('--description')

    sp = sub.add_parser('delete', help='delete a portfolio')
    sp.add_argument('id')

    for name, hlp in (('add', 'add hand-picked tickers'),
                      ('remove', 'remove tickers (rule members are excluded)')):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument('id')
        sp.add_argument('tickers', nargs='+')

    sp = sub.add_parser('show', help='resolve members against a snapshot')
    sp.add_argument('id', nargs='?')
    sp.add_argument('--results-dir', default='output')
    sp.add_argument('--date', help='snapshot date (default: latest)')

    sp = sub.add_parser('alerts', help='per-portfolio stats and change alerts vs the prior run')
    sp.add_argument('--results-dir', default='output')
    sp.add_argument('--date', help='snapshot date (default: latest)')
    sp.add_argument('--out', help='also write the report here ({date} is substituted)')

    sp = sub.add_parser('nav', help='NAV history per portfolio; --rebuild backfills it')
    sp.add_argument('--id', action='append', help='portfolio id (repeat; default: all)')
    sp.add_argument('--rebuild', action='store_true',
                    help='replay the current definitions over archived snapshots '
                         '(marked backfilled) in front of the live history')
    sp.add_argument('--since', help='first snapshot date for --rebuild')
    sp.add_argument('--results-dir', default='output')
    sp.add_argument('--prices-dir', default='output/prices')

    sp = sub.add_parser('import', help="merge a report export or share link")
    sp.add_argument('source', help='exported JSON file, JSON text, or a #pf= share link')
    sp.add_argument('--overwrite', action='store_true',
                    help='imported versions win on id collisions')
    sp.add_argument('--merge', action='store_true',
                    help='merge even when the export is a fast-forward of the file')
    sp.add_argument('--replace', action='store_true',
                    help='the import becomes the whole file')
    sp.add_argument('--dry-run', action='store_true', help='show the diff only')
    return ap


COMMANDS = {'list': cmd_list, 'create': cmd_create, 'set-rule': cmd_set_rule,
            'edit': cmd_edit, 'delete': cmd_delete, 'add': cmd_add,
            'remove': cmd_remove, 'show': cmd_show, 'alerts': cmd_alerts, 'nav': cmd_nav,
            'import': cmd_import}


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        doc = pg.load_portfolios(a.file)
    except ValueError as e:
        raise SystemExit(f"{a.file}: {e}") from None
    msg = COMMANDS[a.cmd](doc, a)
    if msg:
        try:
            pg.save_portfolios(doc, a.file)
        except ValueError as e:
            raise SystemExit(f"not saved: {e}") from None
        print(msg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
