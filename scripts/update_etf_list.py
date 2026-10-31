# scripts/update_etf_list.py
"""Refresh data/etf_funds.json, the funds the ETF look-through reads.

SSGA's fund finder (the JSON behind ssga.com's ETF screener, no auth) lists
every SPDR ETF with its asset-class and strategy tags. The candidates are its
equity and alternative-sector funds (bond, cash and multi-asset funds hold
no stocks the model scores). Each candidate's daily holdings file is fetched,
and a fund is kept only when it parses and at least ``--min-matched`` (80%,
the look-through's own ``MIN_COVERAGE``) of its weight is cash or a holding
with a row in the newest snapshot — so a listed fund can read, rather than
sit INSUFFICIENT every night (an international fund matches almost nothing),
and at most ``--max-cash`` (50%) is cash: CERY, a commodity-futures fund,
matched 80% on its T-bills alone.

Run by hand, or monthly, never nightly; a fund that later stops matching only
reads INSUFFICIENT until the next refresh:

    python scripts/update_etf_list.py
    python scripts/update_etf_list.py --results output/results_2026-10-30.json

SPY, DIA, MDY and the 11 Select Sector SPDRs are kept whatever they match
(``etf_holdings.SPDR_FUNDS``: the sector pages read them).
"""
import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import etf_holdings  # noqa: E402
from data.snapshot_store import list_snapshot_files, load_snapshot_file  # noqa: E402

logger = logging.getLogger('update_etf_list')

FINDER_URL = ('https://www.ssga.com/bin/v1/ssmp/fund/fundfinder?country=us&language=en'
              '&role=intermediary&product=etfs&ui=fund-finder')
CANDIDATE_CLASSES = ('equity', 'alternative')
MIN_FUNDS = 20  # fewer kept than this is a broken fetch, not a fund list


def _get_json(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _tickers(s):
    return {etf_holdings.normalize_ticker(t.replace('®', '')) for t in (s or '').split('|') if t}


def _walk(cats, path=()):
    """Yield ``(path_of_keys, funds)`` for every node of the finder's category tree."""
    for c in cats or []:
        p = path + (c.get('key') or '',)
        yield p, _tickers(c.get('funds'))
        yield from _walk(c.get('subCategories'), p)


def parse_finder(finder):
    """``(names, tags)`` from the finder JSON: ticker -> fund name, and
    ticker -> the set of every category key on the path to it."""
    etfs = finder['data']['funds']['etfs']
    names = {}
    for f in etfs.get('datas') or []:
        t = etf_holdings.normalize_ticker(str(f.get('fundTicker') or '').replace('®', ''))
        if t:
            names[t] = str(f.get('fundName') or '').replace('®', '').replace('℠', '').strip()
    tags = {}
    for path, funds in _walk(etfs.get('categories')):
        for t in funds:
            tags.setdefault(t, set()).update(path)
    return names, tags


def categorize(ticker, name, tags):
    """One of ``etf_holdings.CATEGORIES`` from the finder's tags and the name."""
    n = name.upper()
    if etf_holdings.SPDR_FUNDS.get(ticker):
        return 'sector'
    if 'Dividend' in tags or 'DIVIDEND' in n or 'YIELD' in n:
        return 'dividend'
    if 'smart-beta' in tags:
        return 'factor'
    if 'Sectors---Industries' in tags or 'alternative' in tags:
        return 'industry'
    if 'GROWTH' in n or 'VALUE' in n:
        return 'style'
    # The finder tags the digital-asset and tech-theme funds Growth too.
    if tags & {'New-Economies', 'Environmental--Social---Governance', 'equity-style'}:
        return 'thematic'
    if 'Mid-Cap' in tags or 'Small-Cap' in tags:
        return 'size'
    return 'broad'


def matched_weight(holdings, rows_by_ticker, aliases):
    """``(matched, cash)``: the shares of a fund's weight that are cash or a
    holding with a row, and that are cash alone."""
    total = matched = cash = 0.0
    for x in holdings['holdings']:
        w = x.get('weight') or 0.0
        total += w
        if x.get('kind') == 'cash':
            cash += w
            matched += w
        elif aliases.get(x['ticker'], x['ticker']) in rows_by_ticker:
            matched += w
    return (matched / total, cash / total) if total > 0 else (0.0, 0.0)


def _snapshot_rows(path):
    _, rows = load_snapshot_file(path)
    by_tk = {r['ticker']: r for r in rows if isinstance(r, dict) and r.get('ticker')}
    aliases = {a: t for t, r in by_tk.items() for a in (r.get('listing_aliases') or [])}
    return by_tk, aliases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--results', help='snapshot to match holdings against (default: newest in output/)')
    ap.add_argument('--min-matched', type=float, default=0.80)
    ap.add_argument('--max-cash', type=float, default=0.50)
    ap.add_argument('--delay', type=float, default=1.0, help='seconds between holdings requests')
    ap.add_argument('--out', default=etf_holdings.REGISTRY_PATH)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

    path = args.results
    if not path:
        files = list_snapshot_files('output')
        if not files:
            raise SystemExit('No snapshot in output/; pass --results')
        path = files[-1][1]
    rows_by_ticker, aliases = _snapshot_rows(path)
    print(f'Matching against {path} ({len(rows_by_ticker)} rows)')

    names, tags = parse_finder(_get_json(FINDER_URL))
    candidates = sorted({t for t, k in tags.items()
                         if 'assetclass' in k and k & set(CANDIDATE_CLASSES)}
                        | set(etf_holdings.SPDR_FUNDS))
    print(f'{len(names)} SPDR ETFs listed, {len(candidates)} equity candidates')

    kept = []
    for i, t in enumerate(candidates):
        if i:
            time.sleep(args.delay)
        try:
            h = etf_holdings.fetch_ssga(t)
        except Exception as e:
            logger.warning('%s: holdings fetch failed: %s', t, e)
            if t in etf_holdings.SPDR_FUNDS:
                kept.append(t)
            continue
        m, cash = matched_weight(h, rows_by_ticker, aliases)
        keep = (m >= args.min_matched and cash <= args.max_cash) or t in etf_holdings.SPDR_FUNDS
        print(f'{t:5} {m:6.1%} matched  {cash:6.1%} cash  {len(h["holdings"]):5} holdings  '
              f'{"keep" if keep else "drop"}')
        if keep:
            kept.append(t)
            names.setdefault(t, h.get('fund_name') or '')

    if len(kept) < MIN_FUNDS:
        raise SystemExit(f'Only {len(kept)} funds kept; refusing to overwrite {args.out}')
    funds = [{'ticker': t, 'name': names.get(t, ''),
              'category': categorize(t, names.get(t, ''), tags.get(t, set())),
              'sector': etf_holdings.SPDR_FUNDS.get(t)}
             for t in sorted(kept)]
    body = {'source': 'ssga fund finder', 'updated': date.today().isoformat(),
            'matched_against': os.path.basename(path), 'min_matched': args.min_matched,
            'max_cash': args.max_cash,
            'funds': funds}
    tmp = args.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(body, f, indent=1, ensure_ascii=False)
        f.write('\n')
    os.replace(tmp, args.out)
    print(f'Wrote {len(funds)} funds to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
