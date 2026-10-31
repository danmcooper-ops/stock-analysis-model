# scripts/fetch_etf_holdings.py
"""Fetch and archive the SPDR funds' daily holdings (data/etf_holdings.py).

    python scripts/fetch_etf_holdings.py                 # every registry fund
    python scripts/fetch_etf_holdings.py --funds SPY XLK

The funds are data/etf_funds.json's (scripts/update_etf_list.py refreshes it)
plus the 14 SPDR_FUNDS, about 70 at 1 s apart.

Each file lands as output/etf_holdings/<FUND>/<as_of>.json.gz, so the render's
look-through (report_html._build_etf_payload) reads the newest file dated on
or before the run. One fund failing never stops the rest. Exit 0 when at least
one fund was archived, 1 when none was; run_daily.sh runs it as a soft step,
so a failure costs only the night's ETF read, never the report.
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import etf_holdings  # noqa: E402

logger = logging.getLogger('fetch_etf_holdings')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--funds', nargs='+', default=etf_holdings.fund_tickers())
    ap.add_argument('--out', default=etf_holdings.DEFAULT_ROOT)
    ap.add_argument('--delay', type=float, default=1.0,
                    help='seconds between requests (default 1.0)')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    saved = 0
    for i, fund in enumerate(args.funds):
        if i:
            time.sleep(args.delay)
        try:
            parsed = etf_holdings.fetch_ssga(fund.upper())
        except Exception as e:
            logger.warning('%s: holdings fetch failed: %s', fund.upper(), e)
            continue
        path = etf_holdings.save_holdings(args.out, parsed)
        saved += 1
        print(f"{parsed['fund']:5} {parsed['as_of']}  {len(parsed['holdings']):4} holdings  -> {path}")
    print(f'Archived {saved}/{len(args.funds)} funds')
    return 0 if saved else 1


if __name__ == '__main__':
    sys.exit(main())
