# scripts/calibrate.py
"""Walk-forward parameter calibration for the stock analysis pipeline.

Splits historical result snapshots into rolling train/test windows.
For each window:
  1. TRAIN — evaluate candidate ParamSets against backtest metrics
  2. TEST  — measure out-of-sample performance with the best params
  3. RECORD — log train vs test objective per window

Usage:
    python scripts/calibrate.py --results-dir output/ --horizons 90
    python scripts/calibrate.py --objective alpha --max-evals 300
"""

import sys
import os
import json
import math
import itertools
from datetime import date, datetime
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from scripts.param_set import default_params, merge_params, validate_params


# ======================================================================
# Window splitting
# ======================================================================

def generate_windows(snapshot_dates, train_size=3, test_size=1, step=1):
    """Generate rolling train / test windows from sorted snapshot dates.

    Args:
        snapshot_dates: Sorted list of date objects with available snapshots.
        train_size: Number of snapshots in each training window.
        test_size: Number of snapshots in each test window.
        step: How many snapshots to advance between windows.

    Returns:
        List of ``{'train_dates': [...], 'test_dates': [...]}`` dicts.
        Empty list if there aren't enough snapshots.
    """
    total_needed = train_size + test_size
    if len(snapshot_dates) < total_needed:
        return []

    windows = []
    i = 0
    while i + total_needed <= len(snapshot_dates):
        train = snapshot_dates[i:i + train_size]
        test = snapshot_dates[i + train_size:i + total_needed]
        windows.append({'train_dates': train, 'test_dates': test})
        i += step
    return windows


# ======================================================================
# Objective functions
# ======================================================================

def compute_objective(backtest_metrics, objective='hit_rate'):
    """Compute a scalar objective from backtest metrics.

    Args:
        backtest_metrics: List of per-snapshot metric dicts from
            ``backtest.run_backtest()`` or compatible structure.
        objective: 'hit_rate' | 'alpha' | 'information_ratio' | 'composite'

    Returns:
        float: Objective value (higher is better).  Returns 0.0 if
        insufficient data.
    """
    fn = {
        'hit_rate': hit_rate_objective,
        'alpha': alpha_objective,
        'information_ratio': information_ratio_objective,
        'composite': composite_objective,
    }.get(objective, hit_rate_objective)
    return fn(backtest_metrics)


def hit_rate_objective(metrics):
    """Fraction of BUY-rated stocks that outperformed SPY.

    Aggregates across all snapshots in *metrics*.  BUYs with no measurable
    forward return are excluded from numerator AND denominator rather than
    counted as misses (or crashing on ``None > 0``).
    """
    beats = 0
    total = 0
    for m in metrics:
        for detail in m.get('details', []):
            if detail.get('rating') == 'BUY':
                er = detail.get('excess_return')
                if er is None:
                    continue
                total += 1
                if er > 0:
                    beats += 1
    return beats / total if total > 0 else 0.0


def alpha_objective(metrics):
    """Mean excess return of BUY-rated stocks across all snapshots."""
    excess = []
    for m in metrics:
        for detail in m.get('details', []):
            if detail.get('rating') == 'BUY':
                er = detail.get('excess_return')
                if er is not None:
                    excess.append(er)
    return float(np.mean(excess)) if excess else 0.0


def information_ratio_objective(metrics):
    """Alpha / tracking error for the BUY bucket."""
    excess = []
    for m in metrics:
        for detail in m.get('details', []):
            if detail.get('rating') == 'BUY':
                er = detail.get('excess_return')
                if er is not None:
                    excess.append(er)
    if len(excess) < 3:
        return 0.0
    mu = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=1))
    return mu / sigma if sigma > 0 else 0.0


def composite_objective(metrics):
    """Blended objective: 0.4 * hit_rate + 0.3 * norm_alpha + 0.3 * fv_accuracy."""
    hr = hit_rate_objective(metrics)
    alpha = alpha_objective(metrics)
    # Normalise alpha to [0, 1] (cap at 10%)
    norm_alpha = max(0.0, min(1.0, alpha / 0.10))

    # Fair-value accuracy: fraction within ±20% of actual price
    within = 0
    fv_total = 0
    for m in metrics:
        for detail in m.get('details', []):
            fv = detail.get('dcf_fv')
            actual = detail.get('end_price')
            if fv and actual and actual > 0:
                fv_total += 1
                if abs(fv - actual) / actual <= 0.20:
                    within += 1
    fv_acc = within / fv_total if fv_total > 0 else 0.0

    return 0.4 * hr + 0.3 * norm_alpha + 0.3 * fv_acc


# ======================================================================
# Parameter search space
# ======================================================================

# Only parameters that the snapshot re-score path (scoring.score_and_rate)
# actually consumes belong here.  DCF-stage knobs (erp, blend_trigger,
# blend_dcf_weight, growth_weight_*, analyst_haircut,
# margin_trend_sensitivity) have NO effect when re-scoring frozen snapshots
# — fair values are not recomputed — so searching them just burns the
# evaluation budget and exports arbitrary "calibrated" values.  Re-add them
# only if calibration switches to a full pipeline replay.
SEARCH_SPACE = {
    # Scoring weights (3 free, growth derived as 1 - sum - ownership)
    'score_weight_valuation': (0.15, 0.45, 0.05),
    'score_weight_quality':   (0.10, 0.40, 0.05),
    'score_weight_moat':      (0.10, 0.40, 0.05),
    # score_weight_growth = 1.0 - sum(above three) - ownership default

    # Composite-score rating thresholds (validate_params enforces
    # buy > lean > pass; pass threshold stays at its default)
    'rating_threshold_buy':  (52, 67, 5),
    'rating_threshold_lean': (34, 44, 5),
}


def _generate_grid(search_space):
    """Generate all combinations from the search space.

    Each entry is ``(min, max, step)``.  Values are generated with
    ``numpy.arange`` and rounded to avoid floating-point drift.

    Returns:
        list[dict]: Every parameter combination as a dict.
    """
    keys = sorted(search_space.keys())
    ranges = []
    for k in keys:
        lo, hi, step = search_space[k]
        vals = np.arange(lo, hi + step * 0.5, step)
        vals = [round(float(v), 6) for v in vals]
        ranges.append(vals)

    grid = []
    for combo in itertools.product(*ranges):
        d = dict(zip(keys, combo))
        grid.append(d)
    return grid


def _apply_derived_params(candidate):
    """Compute derived parameters and merge with defaults.

    - ``score_weight_growth`` = 1.0 - sum of other three scoring weights
      (only computed when at least one scoring weight is in the candidate)
    - ``blend_mult_weight`` = 1.0 - ``blend_dcf_weight``

    Returns:
        dict: Full ParamSet (defaults + candidate + derived), or None
        if constraints violated.
    """
    derived = dict(candidate)

    # Derive score_weight_growth only when scoring weights are being tuned.
    # Ownership weight is fixed at its default; growth absorbs the remainder.
    sw_keys = ('score_weight_valuation', 'score_weight_quality', 'score_weight_moat')
    if any(k in candidate for k in sw_keys):
        defs = default_params()
        sw_sum = (candidate.get('score_weight_valuation', defs['score_weight_valuation'])
                  + candidate.get('score_weight_quality', defs['score_weight_quality'])
                  + candidate.get('score_weight_moat', defs['score_weight_moat'])
                  + candidate.get('score_weight_ownership', defs['score_weight_ownership']))
        sw_growth = round(1.0 - sw_sum, 4)
        if sw_growth < 0.05:
            return None  # Constraint violation
        derived['score_weight_growth'] = sw_growth

    # Derive blend_mult_weight
    if 'blend_dcf_weight' in derived:
        derived['blend_mult_weight'] = round(1.0 - derived['blend_dcf_weight'], 4)

    try:
        params = merge_params(derived)
    except ValueError:
        return None

    errors = validate_params(params)
    if errors:
        return None
    return params


def _sample_grid(full_grid, n, seed=42):
    """Stratified random sampling from a large grid.

    Uses numpy to select *n* samples that are approximately
    evenly distributed across the grid.
    """
    rng = np.random.default_rng(seed)
    if n >= len(full_grid):
        return full_grid
    indices = rng.choice(len(full_grid), size=n, replace=False)
    return [full_grid[i] for i in sorted(indices)]


def grid_search(evaluate_fn, search_space=None, max_evaluations=500):
    """Search over parameter space to maximise the objective.

    If the full grid exceeds *max_evaluations*, samples a subset.

    Args:
        evaluate_fn: Callable(params_dict) → float (objective value).
        search_space: ``{name: (min, max, step)}``.  Defaults to
            module-level ``SEARCH_SPACE``.
        max_evaluations: Cap on evaluations.

    Returns:
        List of ``{'params': dict, 'objective': float}`` sorted by
        objective descending.
    """
    if search_space is None:
        search_space = SEARCH_SPACE

    raw_grid = _generate_grid(search_space)
    if len(raw_grid) > max_evaluations:
        raw_grid = _sample_grid(raw_grid, max_evaluations)

    results = []
    for candidate in raw_grid:
        params = _apply_derived_params(candidate)
        if params is None:
            continue
        obj = evaluate_fn(params)
        results.append({'params': params, 'objective': obj})

    results.sort(key=lambda x: x['objective'], reverse=True)
    return results


# ======================================================================
# Overfitting prevention
# ======================================================================

def regularized_objective(base_obj, params, lambda_reg=0.05):
    """Penalise large deviations from default parameter values.

    Args:
        base_obj: Raw objective value (higher is better).
        params: Candidate ParamSet dict.
        lambda_reg: Regularisation strength.

    Returns:
        float: Penalised objective.
    """
    defaults = default_params()
    deviation = 0.0
    for k, v in params.items():
        dv = defaults.get(k)
        if dv is not None and isinstance(v, (int, float)) and isinstance(dv, (int, float)):
            # Normalise by default value to make deviations comparable
            denom = abs(dv) if dv != 0 else 1.0
            deviation += ((v - dv) / denom) ** 2
    return base_obj - lambda_reg * deviation


def compute_stability(window_results):
    """Measure how much each parameter varies across windows.

    Args:
        window_results: List of window dicts with 'best_params'.

    Returns:
        dict: ``{param_name: std_dev}`` for each parameter.
    """
    if not window_results:
        return {}

    all_params = [w['best_params'] for w in window_results if 'best_params' in w]
    if not all_params:
        return {}

    stability = {}
    for key in all_params[0]:
        vals = [p[key] for p in all_params if isinstance(p.get(key), (int, float))]
        if len(vals) >= 2:
            stability[key] = float(np.std(vals, ddof=1))
    return stability


# ======================================================================
# Walk-forward calibration loop
# ======================================================================

def walk_forward_calibrate(results_dir='output', horizons=None,
                           train_size=3, test_size=1, step=1,
                           objective='hit_rate', max_evaluations=200,
                           lambda_reg=0.05, yf_client=None,
                           prices_dir=None):
    """Run the full walk-forward calibration.

    For each rolling window:
      1. Load train-window results JSONs
      2. For each candidate ParamSet: re-score → measure backtest objective
      3. Select best ParamSet from train
      4. Apply best ParamSet to test window → record OOS performance
      5. Roll forward

    Args:
        results_dir: Directory containing results_YYYY-MM-DD.json files.
        horizons: List of forward-return horizons in days.  Defaults to [90].
        train_size: Snapshots per training window.
        test_size: Snapshots per test window.
        step: Roll-forward step in snapshots.
        objective: Objective function name.
        max_evaluations: Max parameter combinations per window.
        lambda_reg: Regularisation strength (0 = disabled).
        yf_client: YFinanceClient for forward-return fetching.  Created on
            demand when None.
        prices_dir: Local parquet price dir passed to fetch_forward_returns.

    Returns:
        dict with 'windows', 'overall', 'recommended_params'.
    """
    if horizons is None:
        horizons = [90]

    # Discover available snapshot dates
    snapshot_dates = _discover_snapshot_dates(results_dir)
    windows = generate_windows(snapshot_dates, train_size, test_size, step)

    if not windows:
        return {
            'windows': [],
            'overall': {'error': 'Insufficient snapshots for walk-forward'},
            'recommended_params': default_params(),
        }

    # Load each snapshot once (windows overlap heavily; re-reading per
    # window would re-parse the same multi-MB JSONs repeatedly)
    snapshots_by_date = {}
    for d in snapshot_dates:
        loaded = _load_snapshots(results_dir, [d])
        if loaded:
            snapshots_by_date[d] = loaded[0]

    # Forward returns are params-independent: compute them ONCE per
    # snapshot×horizon, not per grid candidate.  Without this join the
    # objectives have no outcomes to score against — the details previously
    # read `_excess_return` keys that nothing ever wrote.
    from scripts.backtest import fetch_forward_returns
    if yf_client is None:
        from data.yfinance_client import YFinanceClient
        yf_client = YFinanceClient()

    forward_returns = {}
    for d, snap in snapshots_by_date.items():
        run_date = snap.get('date', d.isoformat())
        tickers = [r.get('ticker') for r in snap.get('results', [])
                   if r.get('ticker')]
        for h in horizons:
            forward_returns[(run_date, h)] = fetch_forward_returns(
                tickers, run_date, h, yf_client, prices_dir=prices_dir)

    measurable = sum(1 for v in forward_returns.values() if v)
    if measurable == 0:
        return {
            'windows': [],
            'overall': {'error': 'No forward returns available for any '
                                 'snapshot/horizon — horizons may not have '
                                 'elapsed yet'},
            'recommended_params': default_params(),
        }

    window_results = []
    for win in windows:
        train_results = [snapshots_by_date[d] for d in win['train_dates']
                         if d in snapshots_by_date]
        test_results = [snapshots_by_date[d] for d in win['test_dates']
                        if d in snapshots_by_date]

        # For each candidate, re-score with the candidate's weights and
        # measure backtest metrics on the train window
        def evaluate(params):
            metrics = _evaluate_params_on_snapshots(
                train_results, params, horizons,
                forward_returns=forward_returns)
            raw = compute_objective(metrics, objective)
            if lambda_reg > 0:
                return regularized_objective(raw, params, lambda_reg)
            return raw

        search_results = grid_search(evaluate, max_evaluations=max_evaluations)

        if not search_results:
            continue

        best = search_results[0]
        best_params = best['params']
        train_obj = best['objective']

        # Out-of-sample: apply best params to test window
        test_metrics = _evaluate_params_on_snapshots(
            test_results, best_params, horizons,
            forward_returns=forward_returns)
        test_obj = compute_objective(test_metrics, objective)

        window_results.append({
            'train_dates': [d.isoformat() for d in win['train_dates']],
            'test_dates': [d.isoformat() for d in win['test_dates']],
            'best_params': best_params,
            'train_objective': round(train_obj, 4),
            'test_objective': round(test_obj, 4),
        })

    # Aggregate results
    stability = compute_stability(window_results)
    train_objs = [w['train_objective'] for w in window_results]
    test_objs = [w['test_objective'] for w in window_results]

    # Recommended params: from the most recent window (standard walk-forward
    # deployment).  Never select by test objective — that turns the
    # out-of-sample windows into a selection set and biases the reported
    # OOS performance upward; mean_test_objective stays honest only if the
    # test windows weren't used for selection.
    if window_results:
        recommended = window_results[-1]['best_params']
    else:
        recommended = default_params()

    overall = {
        'mean_train_objective': round(float(np.mean(train_objs)), 4) if train_objs else None,
        'mean_test_objective': round(float(np.mean(test_objs)), 4) if test_objs else None,
        'overfit_gap': round(
            float(np.mean(train_objs)) - float(np.mean(test_objs)), 4
        ) if train_objs and test_objs else None,
        'param_stability': {k: round(v, 6) for k, v in stability.items()},
    }

    return {
        'date': date.today().isoformat(),
        'objective': objective,
        'horizon_days': horizons,
        'n_windows': len(window_results),
        'n_evaluations_per_window': max_evaluations,
        'windows': window_results,
        'overall': overall,
        'recommended_params': recommended,
    }


# ======================================================================
# Internal helpers
# ======================================================================

def _discover_snapshot_dates(results_dir):
    """Find all results_YYYY-MM-DD.json files and return sorted dates."""
    dates = []
    if not os.path.isdir(results_dir):
        return dates
    for fname in os.listdir(results_dir):
        if fname.startswith('results_') and fname.endswith('.json'):
            date_str = fname.replace('results_', '').replace('.json', '')
            try:
                dates.append(date.fromisoformat(date_str))
            except ValueError:
                continue
    return sorted(dates)


def _load_snapshots(results_dir, dates):
    """Load results JSON files for the given dates.

    Returns:
        list[dict]: Each dict is the full JSON structure with 'date' and 'results'.
    """
    snapshots = []
    for d in dates:
        path = os.path.join(results_dir, f'results_{d.isoformat()}.json')
        if os.path.exists(path):
            with open(path) as f:
                snapshots.append(json.load(f))
    return snapshots


def _evaluate_params_on_snapshots(snapshots, params, horizons,
                                  forward_returns=None):
    """Re-score snapshot results with *params* and compute backtest metrics.

    This performs a lightweight canonical re-score rather than a full DCF
    re-run. The fair values from the original snapshot are preserved; only
    derived scoring fields, composite scores, rating caps, and ratings change.

    Args:
        snapshots: Loaded results JSON dicts.
        params: Candidate ParamSet for re-scoring.
        horizons: Forward-return horizons in days.
        forward_returns: ``{(run_date_str, horizon): {ticker: {'ret', ...}}}``
            from ``backtest.fetch_forward_returns`` (params-independent, so
            computed once by the caller, not per candidate).  Without it every
            detail's excess_return is None and the objectives are meaningless.

    Returns:
        list[dict]: Backtest-compatible metric dicts, one per snapshot×horizon.
    """
    from scripts.scoring import score_and_rate
    from scripts.backtest import BENCHMARK
    import copy

    metrics = []
    for snap in snapshots:
        results = copy.deepcopy(snap.get('results', []))
        run_date = snap.get('date', '')

        # Re-score with candidate params through the canonical live workflow.
        score_and_rate(results, params=params)

        # Build a lightweight metric dict compatible with objective functions
        for h in horizons:
            fwd = (forward_returns or {}).get((run_date, h), {})
            spy = fwd.get(BENCHMARK)
            details = []
            for r in results:
                ticker = r.get('ticker')
                tr = fwd.get(ticker)
                excess = (tr['ret'] - spy['ret']) if (tr and spy) else None
                details.append({
                    'ticker': ticker,
                    'rating': r.get('rating'),
                    'dcf_fv': r.get('dcf_fv'),
                    'price': r.get('price'),
                    'mos': r.get('mos'),
                    '_composite_score': r.get('_composite_score'),
                    'excess_return': excess,
                    'end_price': tr['end'] if tr else None,
                })
            metrics.append({
                'run_date': run_date,
                'horizon': h,
                'details': details,
            })
    return metrics


# ======================================================================
# Weight calibration via Cohen's d
# ======================================================================

def _cohens_d(group_a, group_b):
    """Compute Cohen's d effect size between two score lists."""
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (len(group_a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (len(group_b) - 1)
    pooled_sd = math.sqrt((var_a + var_b) / 2)
    return (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0


def optimize_weights(results_json_path, output_path=None):
    """Grid search over category weight combinations to maximize Cohen's d.

    Loads a results JSON, generates all weight combos over Valuation /
    Quality / Moat / Growth that — together with Ownership held at its
    config default — sum to 1.0, then re-scores each combo and computes
    Cohen's d between the quality/poor groups.

    This matches the 5-category scheme used by ``score_and_rate`` and the
    walk-forward calibrator: every category contributes its full weight,
    Ownership is treated as fixed (it has its own small set of gates that
    tuning over here can't meaningfully separate), and Growth absorbs the
    residual when V+Q+M are pinned.

    Args:
        results_json_path: Path to results_YYYY-MM-DD.json.
        output_path: Optional path to write calibration results JSON.

    Returns:
        dict with best weights, Cohen's d, and top-10 results.
    """
    from scripts.scoring import score_and_rate
    import copy

    with open(results_json_path) as f:
        data = json.load(f)
    all_results = data.get('results', data) if isinstance(data, dict) else data

    # Check we have quality/poor labels
    quality = [r for r in all_results if r.get('source_group') == 'quality']
    poor = [r for r in all_results if r.get('source_group') == 'poor']
    if len(quality) < 3 or len(poor) < 3:
        print(f"Insufficient labelled data: {len(quality)} quality, {len(poor)} poor")
        return None

    # Hold Ownership at its config default; let Growth absorb the residual.
    # This mirrors `_apply_derived_params` in the walk-forward path so a
    # weight combo discovered here can be dropped straight into config.py.
    defs = default_params()
    w_own = defs['score_weight_ownership']

    # Generate weight grid over V/Q/M; Growth = 1 - V - Q - M - Own.
    steps = [round(v, 2) for v in np.arange(0.10, 0.40, 0.05)]
    grid = []
    for wv in steps:
        for wq in steps:
            for wm in steps:
                wg = round(1.0 - wv - wq - wm - w_own, 2)
                if 0.05 <= wg <= 0.40:
                    grid.append((wv, wq, wm, wg))

    print(f"Optimizing weights: {len(grid)} combos, "
          f"{len(quality)} quality, {len(poor)} poor stocks "
          f"(Ownership held at {w_own:.0%})")

    # Compute baseline Cohen's d with the live config defaults
    baseline_results = copy.deepcopy(all_results)
    score_and_rate(baseline_results)
    q_baseline = [r['_composite_score'] for r in baseline_results
                  if r.get('source_group') == 'quality' and r.get('_composite_score') is not None]
    p_baseline = [r['_composite_score'] for r in baseline_results
                  if r.get('source_group') == 'poor' and r.get('_composite_score') is not None]
    baseline_d = _cohens_d(q_baseline, p_baseline)
    print(f"Baseline Cohen's d: {baseline_d:.3f} "
          f"(weights: V={defs['score_weight_valuation']:.0%} "
          f"Q={defs['score_weight_quality']:.0%} "
          f"M={defs['score_weight_moat']:.0%} "
          f"G={defs['score_weight_growth']:.0%} "
          f"Own={defs['score_weight_ownership']:.0%})")

    results_list = []
    for wv, wq, wm, wg in grid:
        trial = copy.deepcopy(all_results)
        params = {
            'score_weight_valuation': wv,
            'score_weight_quality': wq,
            'score_weight_moat': wm,
            'score_weight_growth': wg,
            'score_weight_ownership': w_own,
        }
        score_and_rate(trial, params=params)

        q_scores = [r['_composite_score'] for r in trial
                    if r.get('source_group') == 'quality' and r.get('_composite_score') is not None]
        p_scores = [r['_composite_score'] for r in trial
                    if r.get('source_group') == 'poor' and r.get('_composite_score') is not None]
        d = _cohens_d(q_scores, p_scores)
        q_mean = sum(q_scores) / len(q_scores) if q_scores else 0
        p_mean = sum(p_scores) / len(p_scores) if p_scores else 0

        results_list.append({
            'weights': {'valuation': wv, 'quality': wq, 'moat': wm,
                        'growth': wg, 'ownership': w_own},
            'cohens_d': round(d, 4),
            'quality_mean': round(q_mean, 1),
            'poor_mean': round(p_mean, 1),
        })

    results_list.sort(key=lambda x: x['cohens_d'], reverse=True)
    best = results_list[0]

    print(f"\n--- Best Weights ---")
    w = best['weights']
    print(f"  Valuation: {w['valuation']:.0%}  Quality: {w['quality']:.0%}  "
          f"Moat: {w['moat']:.0%}  Growth: {w['growth']:.0%}  "
          f"Ownership: {w['ownership']:.0%}")
    print(f"  Cohen's d: {best['cohens_d']:.3f} (baseline: {baseline_d:.3f})")
    print(f"  Quality mean: {best['quality_mean']}  Poor mean: {best['poor_mean']}")

    print(f"\n--- Top 10 ---")
    for i, r in enumerate(results_list[:10]):
        w = r['weights']
        print(f"  {i+1}. V={w['valuation']:.0%} Q={w['quality']:.0%} "
              f"M={w['moat']:.0%} G={w['growth']:.0%} Own={w['ownership']:.0%}  "
              f"d={r['cohens_d']:.3f}  q={r['quality_mean']} p={r['poor_mean']}")

    output = {
        'date': date.today().isoformat(),
        'baseline_cohens_d': round(baseline_d, 4),
        'baseline_weights': {
            'valuation': defs['score_weight_valuation'],
            'quality': defs['score_weight_quality'],
            'moat': defs['score_weight_moat'],
            'growth': defs['score_weight_growth'],
            'ownership': defs['score_weight_ownership'],
        },
        'best': best,
        'top_10': results_list[:10],
        'n_quality': len(quality),
        'n_poor': len(poor),
        'grid_size': len(grid),
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {output_path}")

    return output


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Walk-forward parameter calibration')
    parser.add_argument('--results-dir', default='output',
                        help='Directory with results_*.json files')
    parser.add_argument('--horizons', default='90',
                        help='Comma-separated forward return horizons in days')
    parser.add_argument('--train-size', type=int, default=3)
    parser.add_argument('--test-size', type=int, default=1)
    parser.add_argument('--objective', default='hit_rate',
                        choices=['hit_rate', 'alpha', 'information_ratio',
                                 'composite'])
    parser.add_argument('--max-evals', type=int, default=200)
    parser.add_argument('--prices-dir', default='output/prices',
                        help='Local parquet price dir for forward returns')
    parser.add_argument('--lambda-reg', type=float, default=0.05)
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: output/calibration_DATE.json)')
    parser.add_argument('--optimize-weights', default=None,
                        help='Path to results JSON for weight optimization via Cohen\'s d')
    args = parser.parse_args()

    # Weight optimization mode
    if args.optimize_weights:
        out = args.output or os.path.join(
            os.path.dirname(args.optimize_weights),
            f'weight_calibration_{date.today().isoformat()}.json')
        optimize_weights(args.optimize_weights, output_path=out)
        return

    horizons = [int(h) for h in args.horizons.split(',')]

    prices_dir = args.prices_dir if os.path.isdir(args.prices_dir) else None

    result = walk_forward_calibrate(
        results_dir=args.results_dir,
        horizons=horizons,
        train_size=args.train_size,
        test_size=args.test_size,
        objective=args.objective,
        max_evaluations=args.max_evals,
        lambda_reg=args.lambda_reg,
        prices_dir=prices_dir,
    )

    out_path = args.output or os.path.join(
        args.results_dir,
        f'calibration_{date.today().isoformat()}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'Calibration results written to {out_path}')

    overall = result.get('overall', {})
    print(f"\nWindows: {result['n_windows']}")
    print(f"Mean train objective: {overall.get('mean_train_objective')}")
    print(f"Mean test objective:  {overall.get('mean_test_objective')}")
    print(f"Overfit gap:          {overall.get('overfit_gap')}")


if __name__ == '__main__':
    main()
