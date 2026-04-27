"""Build data/wikidata_founders.json from Wikidata's SPARQL endpoint.

Queries Wikidata for every company tagged with a CIK (SEC Central Index Key)
and at least one human founder. Cross-references CIKs against SEC's
company_tickers.json to map → US ticker symbols. Filters out non-person
"founders" (parent companies, predecessor entities) by requiring the founder
to be a human (P31 = Q5).

Output: data/wikidata_founders.json mapping {ticker: [founder names]}.

Consumed at analyze time: a company is flagged founder-led if any
Wikidata-listed founder appears in the current yfinance companyOfficers list.

Usage:
    python scripts/build_wikidata_founders.py

Re-run periodically (monthly is plenty) to pick up new entries.
"""
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()

WIKIDATA_SPARQL = 'https://query.wikidata.org/sparql'
SEC_TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
EMAIL = 'nurses.public1y@icloud.com'

# Companies tagged with a CIK + a human founder. CIK is far better-tagged on
# Wikidata than the ticker symbol (P249), and uniquely identifies SEC-filed
# companies.
SPARQL = """
SELECT ?cik ?companyLabel ?founderLabel WHERE {
  ?company wdt:P5531 ?cik .
  ?company wdt:P112 ?founder .
  ?founder wdt:P31 wd:Q5 .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def _fetch_wikidata():
    url = WIKIDATA_SPARQL + '?' + urllib.parse.urlencode({'query': SPARQL, 'format': 'json'})
    req = urllib.request.Request(url, headers={
        'User-Agent': 'StockAnalyzer/1.0 (founder-led detection seed)',
        'Accept': 'application/sparql-results+json',
    })
    print('[wikidata] fetching SPARQL results (CIK + founder pairs)...')
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=240) as resp:
        return json.loads(resp.read())


def _fetch_cik_to_ticker():
    """Returns {cik_int: ticker_str} from SEC's company_tickers.json."""
    req = urllib.request.Request(
        SEC_TICKERS_URL,
        headers={'User-Agent': f'StockAnalyzer/1.0 ({EMAIL})'},
    )
    print('[sec] fetching CIK→ticker map from company_tickers.json...')
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=60) as resp:
        data = json.loads(resp.read())
    out = {}
    for entry in data.values():
        cik = entry.get('cik_str')
        tk = (entry.get('ticker') or '').upper().strip()
        if cik is None or not tk:
            continue
        # SEC publishes one CIK per filer, but a filer can have multiple
        # tickers (preferred / common share classes). Keep the first ticker —
        # we'll match by CIK so any class works.
        if int(cik) not in out:
            out[int(cik)] = tk
    return out


def main():
    cik_to_ticker = _fetch_cik_to_ticker()
    print(f'[sec] loaded {len(cik_to_ticker)} CIK→ticker pairs')

    data = _fetch_wikidata()
    bindings = data.get('results', {}).get('bindings', [])
    print(f'[wikidata] {len(bindings)} (cik, founder) rows returned')

    # Group by ticker, dedupe founder names per ticker
    by_ticker = {}
    cik_skipped = 0
    for b in bindings:
        cik_str = (b.get('cik', {}).get('value') or '').strip()
        founder_name = (b.get('founderLabel', {}).get('value') or '').strip()
        if not cik_str or not founder_name:
            continue
        try:
            cik_int = int(cik_str)
        except ValueError:
            continue
        ticker = cik_to_ticker.get(cik_int)
        if not ticker:
            cik_skipped += 1
            continue
        # Drop bare Q-id labels (Wikidata fallback when no English label exists)
        if founder_name.startswith('Q') and founder_name[1:].isdigit():
            continue
        by_ticker.setdefault(ticker, set()).add(founder_name)

    out = {tk: sorted(names) for tk, names in sorted(by_ticker.items())}
    print(f'[wikidata] resolved to {len(out)} US tickers; '
          f'{cik_skipped} CIKs had no SEC ticker match')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'data', 'wikidata_founders.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True, ensure_ascii=False)
    print(f'[wikidata] wrote {out_path}')

    # Spot-check a few well-known tickers
    sample = ['DELL', 'ORCL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NKE', 'SBUX',
              'BLK', 'BX', 'AAPL', 'GOOGL', 'GOOG', 'NFLX', 'COST', 'WMT']
    print()
    print('Spot-check:')
    for tk in sample:
        if tk in out:
            print(f'  {tk}: {out[tk]}')
        else:
            print(f'  {tk}: (not in Wikidata)')


if __name__ == '__main__':
    main()
