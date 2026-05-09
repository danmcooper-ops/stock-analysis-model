"""Re-score a saved snapshot JSON and regenerate the HTML report.

Use this after editing scoring.py / report_html.py to refresh today's HTML
without re-running the 3-6h analysis. Loads the JSON, re-applies the canonical
scoring pipeline, and writes a new HTML next to the JSON.

Usage:
    python scripts/rescore_and_render.py output/results_YYYY-MM-DD.json
"""
import argparse
import json
import os
import sys
from datetime import date

# Ensure repo root is on sys.path so `scripts.*` imports resolve when invoked
# directly (i.e. `python scripts/rescore_and_render.py ...`).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.scoring import score_and_rate
from scripts.report_html import build_html


def rescore_and_render(json_path, prices_dir='output/prices'):
    with open(json_path) as f:
        snap = json.load(f)

    # Snapshot may be a bare list or wrapped in a dict with 'results' key
    if isinstance(snap, dict) and 'results' in snap:
        results = snap['results']
        snap_date = snap.get('date') or date.today().isoformat()
    else:
        results = snap
        snap_date = date.today().isoformat()

    score_and_rate(results)

    # Write new HTML alongside the JSON.
    html_path = os.path.join(os.path.dirname(json_path) or '.',
                             f'stock_analysis_results_{snap_date}.html')
    build_html(results, html_path, prices_dir=prices_dir)
    print(f'Wrote {html_path}')

    # Persist the rescored JSON back so the snapshot is consistent with the HTML
    if isinstance(snap, dict) and 'results' in snap:
        snap['results'] = results
        out = snap
    else:
        out = results
    with open(json_path, 'w') as f:
        json.dump(out, f, default=str)
    print(f'Updated {json_path}')

    # Quick sanity summary
    n = len(results)
    cs = [r['_composite_score'] for r in results if r.get('_composite_score') is not None]
    denoms = set(r['_gates_passed'].split('/')[1] for r in results if r.get('_gates_passed'))
    rating_dist = {}
    for r in results:
        rating_dist[r.get('rating', '?')] = rating_dist.get(r.get('rating', '?'), 0) + 1
    print(f'\nSummary:')
    print(f'  Tickers: {n}')
    print(f'  Gate denominators: {denoms}')
    if cs:
        print(f'  Composite score: min={min(cs):.1f}, '
              f'median={sorted(cs)[len(cs)//2]:.1f}, max={max(cs):.1f}')
    print(f'  Rating distribution: {rating_dist}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('json_path', help='Path to results_YYYY-MM-DD.json')
    ap.add_argument('--prices-dir', default='output/prices')
    args = ap.parse_args()
    rescore_and_render(args.json_path, prices_dir=args.prices_dir)
