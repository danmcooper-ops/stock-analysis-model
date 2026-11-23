"""Fetch EDGAR histories for companies that have LEFT the listed universe,
so the growth hindcast and its base rates stop resting on survivors alone.

Every snapshot row is a company listed today, so a hindcast over them
conditions on survival to 2026: acquired, bankrupt and delisted companies
are missing, and the realised-growth base rates read high (6-14%, which is
why models/growth_v2.BASE_RATE_BOUNDS clips them). SEC keeps every filer's
XBRL facts, delisted or not. This script:

  1. reads EDGAR's quarterly form indexes for the chosen years (cached under
     data/cache/edgar_index/) and collects the CIKs that filed a 10-K;
  2. drops every CIK among today's listed issuers (data/cache/sec_issuers.csv,
     SEC's company_tickers_exchange list);
  3. samples --limit of the rest (seeded, so a re-run extends the same set)
     and builds each one's edgar_history with the pipeline's own extractor
     (SECXBRLClient.fetch_historical_financials; companyfacts cached in
     output/delisted/sec_facts, apart from the pipeline's synced cache);
  4. appends {ticker: 'CIK<cik>', cik, name, delisted: true, edgar_history}
     to output/delisted/histories.jsonl (one line per company; a CIK
     already there is skipped).

Prices are not fetched: SEC rarely keeps a delisted company's old ticker,
so the valuation backtest stays on survivors. The growth hindcast reads
these rows with --extra-rows. A company that stopped filing before a
window's fifth year has no outcome for it, so the remaining bias is in
windows cut short by failure; complete windows of later-delisted companies
are what this adds.

Usage (SEC_EMAIL from the environment or .env, as the pipeline):
  python scripts/fetch_delisted.py --years 2010 2011 2012 2013 --limit 500
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(ROOT, 'data', 'cache', 'edgar_index')
OUT = os.path.join(ROOT, 'output', 'delisted', 'histories.jsonl')
# Its own companyfacts cache: the pipeline's data/cache/sec_facts is synced to
# Supabase Storage nightly (scripts/sec_cache.py), and delisted filers the
# pipeline never reads would ride along every night.
FACTS_DIR = os.path.join(ROOT, 'output', 'delisted', 'sec_facts')
INDEX_URL = 'https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx'
_LINE = re.compile(r'^10-K\s{2,}(?P<name>.+?)\s{2,}(?P<cik>\d+)\s{2,}\d{4}-\d{2}-\d{2}\s')


def _load_dotenv():
    p = os.path.join(ROOT, '.env')
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _get(url, email):
    req = urllib.request.Request(url, headers={'User-Agent': f'StockAnalyzer/1.0 ({email})'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode('latin-1')


def ten_k_filers(years, email):
    """{cik (10-digit str): company name} of every 10-K filer in `years`."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    out = {}
    for y in years:
        for q in (1, 2, 3, 4):
            path = os.path.join(INDEX_DIR, f'{y}Q{q}.idx')
            if not os.path.exists(path):
                text = _get(INDEX_URL.format(y=y, q=q), email)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text)
            with open(path, encoding='utf-8') as f:
                for line in f:
                    m = _LINE.match(line)
                    if m:
                        out[m['cik'].zfill(10)] = m['name'].strip()
    return out


ISSUERS = os.path.join(ROOT, 'data', 'cache', 'sec_issuers.csv')


def listed_ciks(path=ISSUERS):
    with open(path, encoding='utf-8') as f:
        return {str(r['cik']).zfill(10) for r in csv.DictReader(f)}


def already_fetched(path=OUT):
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        return {json.loads(line)['cik'] for line in f if line.strip()}


def candidates(years, email, seed=0):
    filers = ten_k_filers(years, email)
    gone = sorted(set(filers) - listed_ciks())
    random.Random(seed).shuffle(gone)
    return [(c, filers[c]) for c in gone]


def main(argv=None):
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--years', type=int, nargs='+', default=[2010, 2011, 2012, 2013])
    ap.add_argument('--limit', type=int, default=300, help='companies to fetch this run')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args(argv)
    email = os.environ.get('SEC_EMAIL')
    if not email:
        raise SystemExit('SEC_EMAIL is not set (SEC requires a contact in the User-Agent)')

    from data.sec_xbrl_client import SECXBRLClient
    from data.throttle import SEC_MIN_INTERVAL, Throttle

    cands = candidates(a.years, email, a.seed)
    done = already_fetched(a.out)
    todo = [(c, n) for c, n in cands if c not in done][:a.limit]
    print(f'[delisted] {len(cands)} 10-K filers of {a.years} are not listed today; '
          f'{len(done)} already fetched, fetching {len(todo)}')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    keyed = {f'CIK{c}': c for c, _ in todo}
    client = SECXBRLClient(cik_map=keyed, name_map={k: n for (k, (_, n)) in zip(keyed, todo, strict=True)},
                           email=email, throttle=Throttle(SEC_MIN_INTERVAL), facts_cache=FACTS_DIR)
    kept = 0
    with open(a.out, 'a', encoding='utf-8') as f:
        for i, (cik, name) in enumerate(todo, 1):
            key = f'CIK{cik}'
            try:
                eh = client.fetch_historical_financials(key)
            except Exception as e:  # one bad filer never ends the sweep
                print(f'  {key} {name}: failed ({type(e).__name__}: {e})', file=sys.stderr)
                eh = None
            if eh and eh.get('revenue_history'):
                f.write(json.dumps({'ticker': key, 'cik': cik, 'name': name, 'sector': None,
                                    'delisted': True, 'edgar_history': eh}, default=str) + '\n')
                kept += 1
            else:
                # Record the miss so a re-run does not fetch it again.
                f.write(json.dumps({'ticker': key, 'cik': cik, 'name': name, 'sector': None,
                                    'delisted': True, 'edgar_history': None}) + '\n')
            if i % 50 == 0:
                print(f'  {i}/{len(todo)} ({kept} with history)')
    print(f'[delisted] wrote {kept} histories ({len(todo) - kept} without XBRL history) to {a.out}')


if __name__ == '__main__':
    main()
