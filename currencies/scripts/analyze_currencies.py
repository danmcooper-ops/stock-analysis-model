"""Currency screener entry point.

    python scripts/analyze_currencies.py [--output-dir output] [--no-macro]

Stages: fetch -> derive -> score -> render. Writes
`output/results_<date>.json` (the pivot artifact every other script consumes)
and `output/currency_screener_<date>.html`.

The results JSON is deliberately the interchange format rather than an
in-memory handoff: `rescore_and_render.py` can re-score a snapshot in seconds
after a gate change, without re-hitting a single API.
"""

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import bis_client, currency_meta, ecb_client, provenance
from data import worldbank_client as wb
from models import carry, external, momentum, stability, valuation
from models import series as sr
from scripts.config import MIN_SPOT_DAYS
from scripts.report_html import build_html
from scripts.scoring import score_and_rate

# The report inlines a price sparkline per currency. Daily data back to 1999
# is ~7,000 points per currency; at 30 currencies that is 210k numbers for a
# chart a few hundred pixels wide. Downsampling to weekly over 10 years keeps
# the page small enough to stay a genuinely single self-contained file.
SPARK_YEARS = 10
SPARK_STRIDE = 5


def _spark(spot):
    """Weekly-sampled recent history as [[iso_date, rate], ...]."""
    window = spot[-(SPARK_YEARS * sr.TRADING_DAYS_YEAR):] if spot else []
    sampled = window[::SPARK_STRIDE]
    # Always keep the true latest observation; striding can drop it.
    if window and sampled and sampled[-1] != window[-1]:
        sampled.append(window[-1])
    return [[d, round(v, 8)] for d, v in sampled]


def build_rows(spot_hist, reer_by_area, policy_by_area, macro_by_code, prov):
    """Assemble one scored-input dict per currency."""
    base = currency_meta.BASE_CURRENCY
    base_macro = macro_by_code.get(base, {})
    base_policy = bis_client.latest_policy_rate(policy_by_area, base, prov)
    base_inflation = base_macro.get('inflation_pct')

    if base_policy is None:
        # Every carry number in the model is a spread against this. Without it
        # the entire Carry pillar would silently score 0 for all 30 rows.
        prov.record_event('source_failed', base, 'bis',
                          {'slot': 'policy_rate',
                           'detail': 'base policy rate missing; carry unavailable'})

    rows = []
    for code in currency_meta.CODES:
        meta = currency_meta.get(code)
        spot = spot_hist.get(code) or []
        reer = bis_client.reer_series(reer_by_area, code)
        macro = macro_by_code.get(code, {})

        if not spot:
            prov.record_event('series_missing', code, 'ecb', {'slot': 'spot'})
            continue
        if spot:
            prov.record_source(code, 'spot', 'ecb', observations=len(spot))
        if reer:
            prov.record_source(code, 'reer', 'bis', observations=len(reer))
        else:
            prov.record_event('series_missing', code, 'bis', {'slot': 'reer'})

        if len(spot) < MIN_SPOT_DAYS:
            prov.record_event('short_history', code, 'ecb',
                              {'days': len(spot), 'required': MIN_SPOT_DAYS})

        policy_rate = bis_client.latest_policy_rate(policy_by_area, code, prov)
        if policy_rate is not None:
            prov.record_source(code, 'policy_rate', 'bis')

        row = {
            'code': code,
            'name': meta['name'],
            'regime': meta['regime'],
            'map_iso3': meta['map_iso3'],
            'bis_area': meta['bis_area'],
        }

        stab = stability.compute(spot, external.inflation_volatility(macro))

        row.update(valuation.compute(reer))
        row.update(momentum.compute(spot, reer))
        row.update(external.compute(macro))
        row.update(stab)
        row.update(carry.compute(
            policy_rate_pct=policy_rate,
            inflation_pct=macro.get('inflation_pct'),
            base_policy_rate_pct=base_policy,
            base_inflation_pct=base_inflation,
            vol_1y=stab.get('vol_1y'),
        ))

        row['spark'] = _spark(spot)
        row['_provenance'] = prov.currency_block(code)
        rows.append(row)

    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', default='output')
    ap.add_argument('--no-macro', action='store_true',
                    help='Skip World Bank (fast iteration; External pillar '
                         'will score as missing data)')
    ap.add_argument('--macro-ttl', type=float, default=wb.DEFAULT_TTL_DAYS)
    args = ap.parse_args(argv)

    run_date = date.today()
    prov = provenance.ProvenanceRecorder(run_date)
    print('Currency screener — run %s' % run_date)

    print('  [1/4] ECB daily reference rates...')
    meta = {}
    per_eur = ecb_client.fetch_reference_rates(meta=meta)
    if not per_eur:
        print('FATAL: ECB reference rates unavailable and no cache. '
              'Spot history is the spine of the model; aborting.',
              file=sys.stderr)
        return 2
    spot_hist = ecb_client.build_usd_histories(per_eur, prov=prov)
    print('        %d currencies, %d daily observations each'
          % (len(spot_hist), max((len(s) for s in spot_hist.values()), default=0)))

    print('  [2/4] BIS REER and policy rates...')
    reer_meta, pol_meta = {}, {}
    reer_by_area = bis_client.fetch_reer(meta=reer_meta) or {}
    policy_by_area = bis_client.fetch_policy_rates(meta=pol_meta) or {}
    if not reer_by_area:
        prov.record_event('source_failed', None, 'bis', {'slot': 'reer'})
        print('        WARNING: no REER — the Valuation pillar will be empty.')
    print('        %d REER areas, %d policy-rate areas'
          % (len(reer_by_area), len(policy_by_area)))

    macro_by_code = {}
    if args.no_macro:
        print('  [3/4] World Bank macro... SKIPPED (--no-macro)')
    else:
        print('  [3/4] World Bank macro indicators...')
        macro_by_code = wb.fetch_macro(ttl_days=args.macro_ttl, prov=prov)
        for field, h in wb.indicator_health(macro_by_code).items():
            flag = '' if h['covered'] >= h['total'] * 0.8 else '   <-- sparse'
            print('        %-26s %2d/%d%s' % (field, h['covered'], h['total'], flag))

    print('  [4/4] Scoring and rendering...')
    rows = build_rows(spot_hist, reer_by_area, policy_by_area, macro_by_code, prov)
    score_and_rate(rows)
    rows.sort(key=lambda r: (r.get('_composite_score') is None,
                             -(r.get('_composite_score') or 0)))

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = run_date.isoformat()
    run_prov = prov.run_block(rows)

    json_path = os.path.join(args.output_dir, 'results_%s.json' % stamp)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'date': stamp, 'count': len(rows),
                   'provenance': run_prov, 'results': rows}, f, indent=1)

    html_path = os.path.join(args.output_dir,
                             'currency_screener_%s.html' % stamp)
    build_html(rows, html_path, run_date=run_date, run_provenance=run_prov)
    prov.write_events(args.output_dir)

    print()
    print('  %-5s %-22s %6s  %-10s %s'
          % ('CCY', 'Name', 'Score', 'Rating', 'Gates'))
    for r in rows:
        print('  %-5s %-22s %6s  %-10s %s'
              % (r['code'], r['name'][:22],
                 r.get('_composite_score', '--'),
                 r.get('rating', '--'), r.get('_gates_passed', '')))

    print()
    print('  events: %s' % (prov.event_counts() or 'none'))
    print('  wrote %s' % json_path)
    print('  wrote %s' % html_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
