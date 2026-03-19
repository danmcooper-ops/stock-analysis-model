# scripts/replay.py
"""Replay the analysis pipeline for a historical date using cached data.

Loads financial snapshots from the disk cache, applies point-in-time slicing,
re-runs the scoring/rating pipeline with optional parameter overrides, and
saves the results JSON in the same schema as live runs.

Usage:
    python scripts/replay.py --as-of 2026-03-08
    python scripts/replay.py --as-of 2026-03-08 --params '{"erp": 0.06}'
    python scripts/replay.py --as-of 2026-03-08 --cache-dir data/cache
"""

import sys
import os
import json
import argparse
from datetime import date

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.snapshot_cache import SnapshotCache
from data.time_slice import slice_financials_as_of
from scripts.param_set import default_params, merge_params, validate_params


def replay_analysis(as_of_date, cache_dir='data/cache', output_dir='output',
                    param_overrides=None, write_output=True):
    """Replay the full analysis pipeline using cached data for a historical date.

    This is a lightweight replay that re-uses existing result snapshots
    and re-scores them with different parameters.  It does NOT re-run
    the full DCF pipeline (which would require live yfinance data).

    For a *full* replay that re-computes fair values, the ``as_of`` path
    in ``YFinanceClient.fetch_financials()`` should be used with the
    ``analyze_stock.py`` pipeline.

    Args:
        as_of_date: The historical date to replay.
        cache_dir: Path to the snapshot cache directory.
        output_dir: Where to find existing results JSONs and write output.
        param_overrides: Optional dict of parameter overrides.
        write_output: Whether to write the results JSON to disk.

    Returns:
        dict: The full results JSON structure (same schema as live runs),
        or None if no data is available for the date.
    """
    # Resolve parameters
    params = merge_params(param_overrides)
    errors = validate_params(params)
    if errors:
        raise ValueError(f"Invalid parameters: {'; '.join(errors)}")

    # Try to load an existing results snapshot for re-scoring
    results_data = _load_closest_results(output_dir, as_of_date)
    if results_data is None:
        print(f"No results snapshot found for {as_of_date}")
        return None

    # Re-score with the given params
    from scripts.scoring import (compute_continuous_scores,
                                 apply_screening_matrix,
                                 apply_composite_rating_override)
    from models.market import compute_rating
    import copy

    results = copy.deepcopy(results_data.get('results', []))

    # Re-run the rating assignment using original fair values
    for r in results:
        # Recompute base rating from MoS (original dcf_fv and price unchanged)
        mos = r.get('mos')
        if mos is not None:
            r['rating'] = compute_rating(mos)

    # Re-run screening matrix (pass/fail gates)
    apply_screening_matrix(results)

    # Re-score with (potentially different) category weights
    compute_continuous_scores(results, params=params)

    # Apply composite rating override
    apply_composite_rating_override(results, params=params)

    # Build output
    output = {
        'date': as_of_date.isoformat(),
        'risk_free_rate': results_data.get('risk_free_rate'),
        'count': len(results),
        'replay': True,
        'params_override': param_overrides,
        'results': results,
    }

    if write_output:
        suffix = '_replay' if param_overrides else ''
        out_path = os.path.join(
            output_dir,
            f'results_{as_of_date.isoformat()}{suffix}.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, default=str)
        print(f"Replay results written to {out_path} ({len(results)} stocks)")

    return output


def _load_closest_results(results_dir, as_of):
    """Load the results JSON closest to (but not after) as_of."""
    if not os.path.isdir(results_dir):
        return None

    candidates = []
    for fname in os.listdir(results_dir):
        if fname.startswith('results_') and fname.endswith('.json'):
            date_str = fname.replace('results_', '').replace('.json', '')
            try:
                d = date.fromisoformat(date_str)
                if d <= as_of:
                    candidates.append((d, fname))
            except ValueError:
                continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    best_fname = candidates[-1][1]
    path = os.path.join(results_dir, best_fname)
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Replay analysis for a historical date')
    parser.add_argument('--as-of', required=True,
                        help='Historical date (YYYY-MM-DD)')
    parser.add_argument('--cache-dir', default='data/cache',
                        help='Snapshot cache directory')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory')
    parser.add_argument('--params', default=None,
                        help='JSON string of parameter overrides')
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of)
    overrides = json.loads(args.params) if args.params else None

    result = replay_analysis(as_of, cache_dir=args.cache_dir,
                             output_dir=args.output_dir,
                             param_overrides=overrides)
    if result:
        print(f"Replay complete: {result['count']} stocks, "
              f"date={result['date']}")


if __name__ == '__main__':
    main()
