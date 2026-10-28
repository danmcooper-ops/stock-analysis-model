"""Re-score an existing results snapshot and re-render the report.

    python scripts/rescore_and_render.py output/results_2026-08-03.json

Runs in under a second against a saved snapshot, with no API calls. This is
the iteration loop for gate and weight changes: `analyze_currencies.py` is the
slow path that talks to four upstream services, and everything downstream of
`score_and_rate` can be re-derived from the JSON it leaves behind.

Overwrites the HTML for that snapshot's date and rewrites the JSON with the
new scores, so a gate change is immediately visible in both artifacts.
"""

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.report_html import build_html, load_results
from scripts.scoring import score_and_rate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('snapshot', help='path to a results_<date>.json')
    ap.add_argument('--output-dir', default=None,
                    help='defaults to the snapshot\'s own directory')
    ap.add_argument('--no-write-json', action='store_true',
                    help='re-render the HTML but leave the snapshot untouched')
    args = ap.parse_args(argv)

    rows, payload = load_results(args.snapshot)
    if not rows:
        print('No rows in %s' % args.snapshot, file=sys.stderr)
        return 2

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.snapshot))
    stamp = payload.get('date') or date.today().isoformat()

    score_and_rate(rows)
    rows.sort(key=lambda r: (r.get('_composite_score') is None,
                             -(r.get('_composite_score') or 0)))

    run_date = date.fromisoformat(stamp)
    html_path = os.path.join(out_dir, 'currency_screener_%s.html' % stamp)
    build_html(rows, html_path, run_date=run_date,
               run_provenance=payload.get('provenance'))

    if not args.no_write_json:
        payload['results'] = rows
        with open(args.snapshot, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=1)

    print('%-5s %6s %6s %6s %6s %6s %6s  %-10s'
          % ('CCY', 'Comp', 'Val', 'Carry', 'Extrn', 'Mom', 'Stab', 'Rating'))
    for r in rows:
        print('%-5s %6s %6s %6s %6s %6s %6s  %-10s' % (
            r['code'], r.get('_composite_score'),
            r.get('_score_pillar_valuation'), r.get('_score_pillar_carry'),
            r.get('_score_pillar_external'), r.get('_score_pillar_momentum'),
            r.get('_score_pillar_stability'), r.get('rating')))
    print('\nwrote %s' % html_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
