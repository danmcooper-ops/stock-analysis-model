# scripts/fx_rates.py
"""Refresh and inspect the FX rate store (data/fx_store.py).

    python scripts/fx_rates.py refresh                    # nightly, before the analysis
    python scripts/fx_rates.py refresh --currencies TRY ARS
    python scripts/fx_rates.py show EUR --date 2026-10-23

``refresh`` covers the majors, every currency already stored, and the
reporting and quote currencies of the newest snapshot in --results-dir.

Exit code: 0 done; 1 a currency the newest snapshot reports in has no rate
as of --run-date (the store cannot serve it). The nightly step is soft: the
analysis fetches a currency the store cannot serve once, on first use.
"""
import argparse
import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fx_store import MAJORS, FXStore, snapshot_currencies  # noqa: E402


def _refresh(store, a):
    wanted = set(snapshot_currencies(a.results_dir, before=None))
    ccys = set(MAJORS) | set(store.currencies()) | wanted | set(a.currencies or [])
    stats = store.refresh(ccys)
    unserved = []
    for ccy, st in stats.items():
        info = store.rate_info(ccy, a.run_date)
        when = (f'{info[0]:.6g} on {info[1]} ({info[2]})' if info
                else 'NO RATE as of the run date')
        print(f"{ccy}  {st.get('status', '?'):7s} bars={st.get('bars', '-')} "
              f"fred={st.get('fred', '-')} overridden={st.get('overridden', '-')} "
              f"spikes={st.get('spikes_dropped', '-')}  {when}")
        if info is None and ccy in wanted:
            unserved.append(ccy)
    print(f'{len(stats)} currencies refreshed into {store.fx_dir}')
    if unserved:
        print(f"no rate as of {a.run_date} for: {' '.join(sorted(unserved))}")
        return 1
    return 0


def _show(store, a):
    ccy = a.currency.upper()
    df = store.frame(ccy)
    if df is None:
        print(f'{ccy}: no stored history')
        return 1
    info = store.rate_info(ccy, a.run_date)
    print(f'{ccy}: {len(df)} bars {df.index[0].date()}..{df.index[-1].date()}, '
          f"{int((df['source'] == 'fred').sum())} from FRED")
    print(f'as of {a.run_date}: ' + (f'{info[0]:.6g} USD per {ccy} '
                                     f'(bar {info[1]}, {info[2]})' if info else 'no rate'))
    print(df.tail(a.tail).to_string())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='action', required=True)
    r = sub.add_parser('refresh')
    r.add_argument('--currencies', nargs='*', help='extra ISO codes to refresh')
    r.add_argument('--results-dir', default='output')
    s = sub.add_parser('show')
    s.add_argument('currency')
    s.add_argument('--tail', type=int, default=5)
    for p in (r, s):
        p.add_argument('--fx-dir', default=None)
        p.add_argument('--run-date', '--date', dest='run_date',
                       type=date.fromisoformat, default=date.today())
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    store = FXStore(a.fx_dir)
    return _refresh(store, a) if a.action == 'refresh' else _show(store, a)


if __name__ == '__main__':
    sys.exit(main())
