# scripts/scoring.py
"""Scoring, screening, and validation functions for the stock analysis pipeline."""

from scripts.config import (SCORE_WEIGHT_VALUATION, SCORE_WEIGHT_QUALITY,
                             SCORE_WEIGHT_MOAT, SCORE_WEIGHT_GROWTH,
                             SCORE_WEIGHT_OWNERSHIP)

MIN_SECTOR_SCORING = 5  # Min stocks per sector for sector-relative percentile
RATING_RANK = {'PASS': 0, 'HOLD': 1, 'LEAN BUY': 2, 'BUY': 3}
RATING_BY_RANK = {v: k for k, v in RATING_RANK.items()}


def _gate_short(gate_name):
    """Return the stable suffix used by _gate_* / _score_* fields."""
    return gate_name.split(': ')[1].lower().replace(' ', '_').replace('/', '_')


def _gate_key(gate_name):
    return '_gate_' + _gate_short(gate_name)


def _gp_key(gate_name):
    return '_gp_' + _gate_short(gate_name)


def _score_key(gate_name):
    return '_score_' + _gate_short(gate_name)


def _cap_rating(rating, cap):
    """Return rating capped at cap, preserving None/UNRATED inputs."""
    if rating not in RATING_RANK or cap not in RATING_RANK:
        return rating
    return RATING_BY_RANK[min(RATING_RANK[rating], RATING_RANK[cap])]


def _mc_confidence_label(cv):
    """Convert coefficient of variation to a confidence label with CV%."""
    if cv is None:
        return None
    pct = round(cv * 100)
    if cv < 0.20:
        return f'HIGH ({pct}%)'
    elif cv < 0.40:
        return f'MEDIUM ({pct}%)'
    else:
        return f'LOW ({pct}%)'


def _score_linear(value, worst, best):
    """Map value linearly from [worst, best] to [0, 100], clamped."""
    if value is None:
        return None
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100
    return max(0.0, min(100.0, score))


def _ranked_percentiles(items, higher_better=True):
    """Assign average-rank percentiles so equal values receive equal scores.

    Args:
        items: list of (row_index, value) pairs.
        higher_better: Whether higher raw values should get higher percentiles.

    Returns:
        dict: {row_index: percentile_0_to_100}
    """
    if not items:
        return {}
    sorted_items = sorted(items, key=lambda x: x[1])
    n = len(sorted_items)
    if n == 1:
        return {sorted_items[0][0]: 50.0}

    out = {}
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_items[j][1] == sorted_items[i][1]:
            j += 1
        avg_rank = (i + (j - 1)) / 2.0
        pctile = (avg_rank / (n - 1)) * 100
        if not higher_better:
            pctile = 100 - pctile
        for k in range(i, j):
            out[sorted_items[k][0]] = pctile
        i = j
    return out


# Gate definitions: (display_name, field_or_callable, pass_test)
# pass_test is a callable: (value, row) -> bool | None (None = skip gate)
SCREENING_GATES = [
    ('Valuation: MoS',
     'mos',
     lambda v, r: v > 0.10 if v is not None else None),
    ('Valuation: Price/FV',
     'price',
     lambda v, r: (v / r['dcf_fv']) < 1.0
     if v and r.get('dcf_fv') else None),
    ('Valuation: P/FCF',
     'pfcf',
     lambda v, r: 0 < v <= 20 if v is not None else None),
    ('Quality: Int Coverage',
     'int_cov',
     lambda v, r: v > 3.0 if v is not None else None),
    ('Quality: Accruals',
     'accruals',
     lambda v, r: abs(v) < 0.08 if v is not None else None),
    ('Ownership: Shrhldr Yield',
     'shareholder_yield',
     lambda v, r: v > 0.02 if v is not None else None),
    ('Ownership: Insider Own',
     'insider_pct',
     lambda v, r: v >= 0.05 if v is not None else None),
    ('Ownership: Buyback Rate',
     'share_buyback_rate',
     lambda v, r: v > 0.01 if v is not None else None),
    ('Moat: ROIC Consistency',
     'roic_cv',
     lambda v, r: v < 0.30 if v is not None else None),
    ('Moat: Spread > 5%',
     'spread',
     lambda v, r: v > 0.07 if v is not None else None),
    ('Moat: Gross Margin',
     'gross_margin_avg_5y',
     lambda v, r: v > 0.35 if v is not None else None),
    ('Growth: Fund Growth',
     'fundamental_growth',
     lambda v, r: v > 0.03 if v is not None else None),
    ('Growth: Margins',
     'gross_margin_trend',
     lambda v, r: v >= 0 if v is not None else None),
    ('Quality: ROE',
     'roe',
     lambda v, r: v > 0.20 if v is not None else None),
    # Buffett additions — balance sheet conservatism + owner earnings quality
    ('Quality: Net Debt/EBITDA',
     'nd_ebitda',
     lambda v, r: v <= 1.5 if v is not None else None),
    ('Quality: Cash Conv',
     'cash_conv',
     lambda v, r: v >= 0.85 if v is not None else None),
    ('Growth: Rev Durability',
     'rev_cagr_10y',
     lambda v, r: v > 0.02 if v is not None else None),
    ('Ownership: SBC Dilution',
     'sbc_pct_rev',
     lambda v, r: v <= 0.02 if v is not None else None),
    ('Valuation: Price/Book',
     'pb',
     lambda v, r: v <= 5.0 if v is not None else None),
    ('Moat: FCF Margin',
     'fcf_margin',
     lambda v, r: v > 0.12 if v is not None else None),
    ('Growth: FCF Durability',
     'fcf_cagr_5y',
     lambda v, r: v > 0.05 if v is not None else None),
    ('Ownership: Share Shrink',
     'shares_cagr_5y',
     lambda v, r: v < 0 if v is not None else None),
    # Migrated from compute_rating
    ('Quality: Piotroski',
     'piotroski',
     lambda v, r: v >= 7 if v is not None else None),
    ('Moat: ROIC Trend',
     'roic_trend_slope',
     lambda v, r: v > 0.005 if v is not None else None),
    ('Valuation: EPV Floor',
     'epv_floor_ratio',
     lambda v, r: v >= 1.0 if v is not None else None),
    ('Valuation: RIM MoS',
     'rim_mos',
     lambda v, r: v > 0.10 if v is not None else None),
]


def prepare_scoring_fields(results):
    """Populate derived fields shared by gates and continuous scoring."""
    for r in results:
        sbc = r.get('sbc')
        rev = r.get('revenue')
        fcf = r.get('fcf')
        price = r.get('price')
        fv = r.get('dcf_fv')
        r['sbc_pct_rev'] = (sbc / rev) if (sbc is not None and rev and rev > 0) else None
        r['fcf_margin'] = (fcf / rev) if (fcf is not None and rev and rev > 0) else None
        r['_price_fv'] = (price / fv) if (price and fv and fv > 0) else None

        # ROIC trend slope (last-year minus first-year ROIC)
        roic_by_year = r.get('roic_by_year')
        if roic_by_year and len(roic_by_year) >= 2:
            sorted_years = sorted(roic_by_year.keys())
            r['roic_trend_slope'] = roic_by_year[sorted_years[-1]] - roic_by_year[sorted_years[0]]
        else:
            r['roic_trend_slope'] = None

        # EPV floor ratio (epv_fv / price) - >=1 means trading below zero-growth value
        epv = r.get('epv_fv')
        r['epv_floor_ratio'] = (epv / price) if (epv is not None and price and price > 0) else None


def apply_screening_matrix(results):
    """Evaluate each stock against pass/fail gates.

    Stores per-gate actual data values and pass/fail booleans in each row dict.
    _gate_* fields hold the raw metric value (number), _gp_* fields hold
    True (pass) / False (fail) / None (N/A) for colour formatting.
    """
    prepare_scoring_fields(results)

    # Fixed denominator: every ticker is graded against the full gate list.
    # Gates with missing data count as fail (no credit) rather than being
    # excluded from the count, so "X / N" is comparable across tickers.
    total_gates = len(SCREENING_GATES)
    for r in results:
        r['rating_raw'] = r.get('rating')
        passed = 0
        for gate_name, field, test_fn in SCREENING_GATES:
            val = r.get(field)
            result = test_fn(val, r)
            gate_key = _gate_key(gate_name)
            gp_key = _gp_key(gate_name)
            if result is None:
                # N/A — keep _gate_* / _gp_* as None so the cell still
                # renders as "N/A" visually, but the gate counts as a fail
                # (passed not incremented; denominator is fixed below).
                r[gate_key] = None
                r[gp_key] = None
            else:
                # Store actual metric value instead of PASS/FAIL
                if field == 'price':
                    # Price/FV gate: display the ratio, not the raw price
                    r[gate_key] = val / r['dcf_fv']
                else:
                    r[gate_key] = val
                r[gp_key] = bool(result)
                if result:
                    passed += 1

        r['_gates_passed'] = f'{passed}/{total_gates}'
        r['_gates_passed_num'] = passed


def _print_validation_stats(results, screen_outcomes):
    """Print validation statistics comparing quality vs poor performer groups."""
    from statistics import median as _med

    quality = [r for r in results if r.get('source_group') == 'quality']
    poor = [r for r in results if r.get('source_group') == 'poor']

    print("\n" + "=" * 70)
    print("VALIDATION: Quality vs Poor Performer Separation")
    print("=" * 70)

    # 1. Screen pass rates
    print("\n--- Screen Pass Rates ---")
    for grp in ('quality', 'poor'):
        o = screen_outcomes[grp]
        rate = o['passed'] / o['total'] if o['total'] > 0 else 0
        print(f"  {grp:>8}: {o['passed']:>3}/{o['total']:<3} passed ({rate:.0%})")

    # 2. Rating distribution
    print("\n--- Rating Distribution ---")
    for grp_name, grp_data in [('quality', quality), ('poor', poor)]:
        counts = {}
        for r in grp_data:
            rating = r.get('rating', 'N/A')
            counts[rating] = counts.get(rating, 0) + 1
        print(f"  {grp_name:>8} (n={len(grp_data)}): ", end='')
        for rating in ['BUY', 'LEAN BUY', 'HOLD', 'PASS']:
            print(f"{rating}={counts.get(rating, 0)} ", end='')
        print()

    # 3. Composite score distributions
    print("\n--- Composite Score Distribution ---")
    for grp_name, grp_data in [('quality', quality), ('poor', poor)]:
        scores = [r['_composite_score'] for r in grp_data
                  if r.get('_composite_score') is not None]
        if scores:
            scores_sorted = sorted(scores)
            n = len(scores_sorted)
            p10 = scores_sorted[max(0, int(n * 0.10))]
            p90 = scores_sorted[min(n - 1, int(n * 0.90))]
            print(f"  {grp_name:>8} (n={n}): mean={sum(scores)/n:.1f}  "
                  f"median={_med(scores):.1f}  p10={p10:.1f}  p90={p90:.1f}")
        else:
            print(f"  {grp_name:>8}: no scores available")

    # 4. Cohen's d effect size
    q_scores = [r['_composite_score'] for r in quality
                if r.get('_composite_score') is not None]
    p_scores = [r['_composite_score'] for r in poor
                if r.get('_composite_score') is not None]
    if len(q_scores) >= 2 and len(p_scores) >= 2:
        import math
        q_mean = sum(q_scores) / len(q_scores)
        p_mean = sum(p_scores) / len(p_scores)
        q_var = sum((x - q_mean) ** 2 for x in q_scores) / (len(q_scores) - 1)
        p_var = sum((x - p_mean) ** 2 for x in p_scores) / (len(p_scores) - 1)
        pooled_sd = math.sqrt((q_var + p_var) / 2)
        cohens_d = (q_mean - p_mean) / pooled_sd if pooled_sd > 0 else 0
        effect = ('large' if abs(cohens_d) >= 0.8
                  else 'medium' if abs(cohens_d) >= 0.5
                  else 'small')
        print(f"\n--- Separation Test ---")
        print(f"  Quality mean: {q_mean:.1f}   Poor mean: {p_mean:.1f}")
        print(f"  Cohen's d: {cohens_d:.2f} ({effect} effect)")

    # 5. MoS distribution
    print("\n--- Margin of Safety Distribution ---")
    for grp_name, grp_data in [('quality', quality), ('poor', poor)]:
        mos_vals = [r['mos'] for r in grp_data if r.get('mos') is not None]
        if mos_vals:
            print(f"  {grp_name:>8} (n={len(mos_vals)}): "
                  f"mean={sum(mos_vals)/len(mos_vals):.1%}  "
                  f"median={_med(mos_vals):.1%}  "
                  f"positive={sum(1 for m in mos_vals if m > 0)}/{len(mos_vals)}")
        else:
            print(f"  {grp_name:>8}: no MoS data")

    print("=" * 70)


# (name, field, category, score_fn, relative_mode, higher_better)
# relative_mode: False=absolute, 'global'=global percentile, 'sector'=sector percentile
# higher_better: True=higher value→higher percentile, False=lower→higher
# score_fn: (value, row, percentile_or_None) -> 0-100
SCORING_GATES = [
    ('Valuation: MoS', 'mos', 'Valuation',
     lambda v, r, pct: _score_linear(v, -0.10, 0.40), False, True),   # tightened: worst raised -20%→-10%
    ('Valuation: Price/FV', '_price_fv', 'Valuation',
     lambda v, r, pct: _score_linear(v, 1.2, 0.7), False, True),      # tightened: worst 1.5→1.2
    ('Valuation: P/FCF', 'pfcf', 'Valuation',
     lambda v, r, pct: _score_linear(v, 40.0, 8.0) if v is not None and v > 0 else 0.0,
     False, True),   # tightened: worst 50→40, best 10→8
    ('Quality: Int Coverage', 'int_cov', 'Quality',
     lambda v, r, pct: _score_linear(min(v, 40) if v is not None else None, 1.0, 20.0),
     False, True),
    ('Quality: Accruals', 'accruals', 'Quality',
     lambda v, r, pct: pct, 'sector', False),
    ('Ownership: Shrhldr Yield', 'shareholder_yield', 'Ownership',
     lambda v, r, pct: _score_linear(v, -0.01, 0.08), False, True),
    ('Ownership: Insider Own', 'insider_pct', 'Ownership',
     lambda v, r, pct: _score_linear(v, 0.0, 0.15), False, True),
    ('Ownership: Buyback Rate', 'share_buyback_rate', 'Ownership',
     lambda v, r, pct: _score_linear(v, 0.0, 0.05), False, True),     # tightened: worst -1%→0%
    ('Moat: ROIC Consistency', 'roic_cv', 'Moat',
     lambda v, r, pct: _score_linear(v, 0.60, 0.0), False, True),
    ('Moat: Spread', 'spread', 'Moat',
     lambda v, r, pct: _score_linear(v, 0.0, 0.20), False, True),     # tightened: best 25%→20%
    ('Moat: Gross Margin', 'gross_margin_avg_5y', 'Moat',
     lambda v, r, pct: pct, 'sector', True),
    ('Growth: Fund Growth', 'fundamental_growth', 'Growth',
     lambda v, r, pct: _score_linear(v, 0.0, 0.10), False, True),
    ('Growth: Margins', 'gross_margin_trend', 'Growth',
     lambda v, r, pct: _score_linear(v, -0.05, 0.05), False, True),
    ('Quality: ROE', 'roe', 'Quality',
     lambda v, r, pct: _score_linear(v, 0.0, 0.35), False, True),     # tightened: best 30%→35%
    # Buffett additions
    ('Quality: Net Debt/EBITDA', 'nd_ebitda', 'Quality',
     lambda v, r, pct: _score_linear(v, 4.0, -0.5), False, True),     # tightened: worst 5→4, best 0→-0.5 (net cash rewarded)
    ('Quality: Cash Conv', 'cash_conv', 'Quality',
     lambda v, r, pct: _score_linear(v, 0.0, 1.5), False, True),
    ('Growth: Rev Durability', 'rev_cagr_10y', 'Growth',
     lambda v, r, pct: _score_linear(v, -0.05, 0.15), False, True),
    ('Ownership: SBC Dilution', 'sbc_pct_rev', 'Ownership',
     lambda v, r, pct: _score_linear(v, 0.06, 0.0), False, True),     # tightened: worst 10%→6%
    ('Valuation: Price/Book', 'pb', 'Valuation',
     lambda v, r, pct: _score_linear(v, 15.0, 0.5), False, True),
    ('Moat: FCF Margin', 'fcf_margin', 'Moat',
     lambda v, r, pct: _score_linear(v, 0.05, 0.25), False, True),    # tightened: worst 0%→5%, best 20%→25%
    ('Growth: FCF Durability', 'fcf_cagr_5y', 'Growth',
     lambda v, r, pct: _score_linear(v, -0.05, 0.15), False, True),
    ('Ownership: Share Shrink', 'shares_cagr_5y', 'Ownership',
     lambda v, r, pct: _score_linear(v, 0.04, -0.04), False, True),
    # Migrated from compute_rating
    ('Quality: Piotroski', 'piotroski', 'Quality',
     lambda v, r, pct: _score_linear(v, 0, 9), False, True),
    ('Moat: ROIC Trend', 'roic_trend_slope', 'Moat',
     lambda v, r, pct: _score_linear(v, -0.05, 0.05), False, True),
    ('Valuation: EPV Floor', 'epv_floor_ratio', 'Valuation',
     lambda v, r, pct: _score_linear(v, 0.5, 1.2), False, True),
    ('Valuation: RIM MoS', 'rim_mos', 'Valuation',
     lambda v, r, pct: _score_linear(v, -0.20, 0.20), False, True),
]


def compute_continuous_scores(results, params=None):
    """Score each stock on all gates using continuous 0-100 scoring.

    Computes per-gate scores, category averages, and weighted composite score.
    Integrates MC confidence as a penalty.
    Supports three scoring modes: absolute, global percentile, and sector percentile.

    Args:
        results: List of stock result dicts.
        params: Optional ParamSet dict.  When provided, category weights
                are read from params instead of module-level constants.
    """
    # Step 1: Pre-compute percentile ranks for relative metrics
    for gate_name, field, category, score_fn, relative_mode, higher_better in SCORING_GATES:
        if not relative_mode:
            continue

        # Collect all values with sector info
        all_vals = [(i, r.get(field), r.get('sector') or '_unknown')
                    for i, r in enumerate(results) if r.get(field) is not None]
        if len(all_vals) < 2:
            continue

        pctile_key = f'{gate_name}_{field}'

        if relative_mode == 'sector':
            # Group by sector
            sector_groups = {}
            for i, val, sector in all_vals:
                sector_groups.setdefault(sector, []).append((i, val))

            for sector, group in sector_groups.items():
                # Fallback to global pool if sector too small
                pool = group if len(group) >= MIN_SECTOR_SCORING else [(i, v) for i, v, _ in all_vals]
                pctiles = _ranked_percentiles(pool, higher_better=higher_better)
                # Only assign percentiles to stocks in this sector group.
                group_indices = set(i for i, _ in group)
                for orig_idx in group_indices:
                    results[orig_idx].setdefault('_pctile', {})[pctile_key] = pctiles[orig_idx]

        else:  # 'global'
            pctiles = _ranked_percentiles([(i, v) for i, v, _ in all_vals],
                                           higher_better=higher_better)
            for orig_idx, pctile in pctiles.items():
                results[orig_idx].setdefault('_pctile', {})[pctile_key] = pctile

    # Step 2: Compute individual gate scores and category averages
    p = params or {}
    cat_weights = {
        'Valuation': p.get('score_weight_valuation', SCORE_WEIGHT_VALUATION),
        'Quality': p.get('score_weight_quality', SCORE_WEIGHT_QUALITY),
        'Moat': p.get('score_weight_moat', SCORE_WEIGHT_MOAT),
        'Growth': p.get('score_weight_growth', SCORE_WEIGHT_GROWTH),
        'Ownership': p.get('score_weight_ownership', SCORE_WEIGHT_OWNERSHIP),
    }
    # Fixed per-category denominators: every gate contributes to its category
    # average, with N/A scoring as 0 (worst). This prevents sparse-data tickers
    # from being graded only on the gates they happen to have.
    gates_per_category = {}
    for gate_name, field, category, *_ in SCORING_GATES:
        gates_per_category[category] = gates_per_category.get(category, 0) + 1

    for r in results:
        category_sums = {cat: 0.0 for cat in gates_per_category}
        for gate_name, field, category, score_fn, relative_mode, higher_better in SCORING_GATES:
            val = r.get(field)
            if val is None:
                # N/A → 0 (counts as worst, included in denominator)
                score = 0.0
            else:
                pct = r.get('_pctile', {}).get(f'{gate_name}_{field}', 50) if relative_mode else None
                s = score_fn(val, r, pct)
                score = s if s is not None else 0.0
            # Store per-gate score (always — even when N/A)
            r[_score_key(gate_name)] = round(score, 1)
            category_sums[category] += score

        # Category averages over fixed denominator (gates_per_category)
        cat_avgs = {cat: category_sums[cat] / gates_per_category[cat]
                    for cat in gates_per_category}
        r['_score_valuation'] = round(cat_avgs.get('Valuation', 0), 1)
        r['_score_quality'] = round(cat_avgs.get('Quality', 0), 1)
        r['_score_moat'] = round(cat_avgs.get('Moat', 0), 1)
        r['_score_growth'] = round(cat_avgs.get('Growth', 0), 1)
        r['_score_ownership'] = round(cat_avgs.get('Ownership', 0), 1)

        # Weighted composite — every category always contributes its full weight
        weighted_sum = 0.0
        weight_total = 0.0
        for cat in gates_per_category:
            w = cat_weights.get(cat, 0)
            weighted_sum += cat_avgs[cat] * w
            weight_total += w
        composite = weighted_sum / weight_total if weight_total > 0 else None

        # Store raw composite before MC penalty
        r['_composite_score_raw'] = round(composite, 1) if composite is not None else None

        # MC confidence penalty
        mc_cv = r.get('mc_cv')
        if composite is not None and mc_cv is not None:
            if mc_cv > 0.40:
                composite *= 0.85
            elif mc_cv > 0.30:
                composite *= 0.93

        r['_composite_score'] = round(composite, 1) if composite is not None else None
        present = sum(1 for _, field, *_ in SCORING_GATES if r.get(field) is not None)
        r['_data_coverage_score'] = round(present / len(SCORING_GATES) * 100, 1)

        # Clean up temp
        r.pop('_pctile', None)


def rating_from_composite(composite, params=None):
    """Map a 0-100 composite score to a rating bucket.

      BUY       composite >= 60
      LEAN BUY  composite >= 43
      HOLD      composite >= 29
      PASS      composite <  29

    Thresholds calibrated against the 2026-05-08 universe (n=1735) to
    produce ~0.6% BUY / ~25% LEAN / ~49% HOLD / ~24% PASS.

    Returns None when composite is None. Thresholds tunable via params.
    """
    if composite is None:
        return None
    p = params or {}
    if composite >= p.get('rating_threshold_buy', 60):
        return 'BUY'
    if composite >= p.get('rating_threshold_lean', 43):
        return 'LEAN BUY'
    if composite >= p.get('rating_threshold_pass', 29):
        return 'HOLD'
    return 'PASS'


def apply_composite_rating_override(results, params=None):
    """Set rating on each row from its composite score.

    Sole rating producer now that compute_rating is gone. Name preserved for
    backward compatibility with existing callers (analyze_stock, calibrate,
    rescore_and_render, replay).
    """
    for r in results:
        rating = rating_from_composite(r.get('_composite_score'), params)
        if rating is not None:
            r['rating'] = rating


_GATE_DISPLAY = {
    'mos': {'label': 'MoS', 'threshold': 'MoS > 10%', 'fmt': 'pct1'},
    'price_fv': {'label': 'Price/FV', 'threshold': 'P/FV < 1.0', 'fmt': 'ratio'},
    'p_fcf': {'label': 'P/FCF', 'threshold': 'P/FCF <= 20x', 'fmt': 'ratio'},
    'int_coverage': {'label': 'Int Cov', 'threshold': 'IC > 3x', 'fmt': 'ratio'},
    'accruals': {'label': 'Accruals', 'threshold': '|Acr| < 8%', 'fmt': 'pct1'},
    'shrhldr_yield': {'label': 'Shrhldr Yld', 'threshold': 'Yield > 2%', 'fmt': 'pct1'},
    'insider_own': {'label': 'Insider %', 'threshold': 'Insider >= 5%', 'fmt': 'pct1'},
    'buyback_rate': {'label': 'Buyback', 'threshold': 'Buyback > 1%', 'fmt': 'pct1'},
    'roic_consistency': {'label': 'ROIC CV', 'threshold': 'CV < 30%', 'fmt': 'pct1'},
    'spread_>_5%': {'label': 'Spread', 'threshold': 'Spread > 7%', 'fmt': 'pct1'},
    'gross_margin': {'label': 'Gross Mgn', 'threshold': 'GM > 35%', 'fmt': 'pct1'},
    'fund_growth': {'label': 'Fund Growth', 'threshold': 'FG > 3%', 'fmt': 'pct1'},
    'margins': {'label': 'Margins', 'threshold': 'Margin >= 0', 'fmt': 'pct1'},
    'roe': {'label': 'ROE', 'threshold': 'ROE > 20%', 'fmt': 'pct1'},
    'net_debt_ebitda': {'label': 'ND/EBITDA', 'threshold': 'ND/EBITDA <= 1.5x', 'fmt': 'ratio'},
    'cash_conv': {'label': 'Cash Conv', 'threshold': 'CashConv >= 0.85x', 'fmt': 'ratio'},
    'rev_durability': {'label': '10Y Rev CAGR', 'threshold': '10Y RevCAGR > 2%', 'fmt': 'pct1'},
    'sbc_dilution': {'label': 'SBC/Rev', 'threshold': 'SBC/Rev <= 2%', 'fmt': 'pct1'},
    'price_book': {'label': 'P/B', 'threshold': 'P/B <= 5x', 'fmt': 'ratio'},
    'fcf_margin': {'label': 'FCF Margin', 'threshold': 'FCF Margin > 12%', 'fmt': 'pct1'},
    'fcf_durability': {'label': '5Y FCF CAGR', 'threshold': '5Y FCF CAGR > 5%', 'fmt': 'pct1'},
    'share_shrink': {'label': 'Share Shrink', 'threshold': '5Y Shares CAGR < 0', 'fmt': 'pct1'},
    'piotroski': {'label': 'Piotroski', 'threshold': 'F-Score >= 7', 'fmt': 'int'},
    'roic_trend': {'label': 'ROIC Trend', 'threshold': 'ROIC trend > 0.5pp', 'fmt': 'pct1'},
    'epv_floor': {'label': 'EPV Floor', 'threshold': 'EPV/Price >= 1.0', 'fmt': 'ratio'},
    'rim_mos': {'label': 'RIM MoS', 'threshold': 'RIM MoS > 10%', 'fmt': 'pct1'},
}

_CATEGORY_DISPLAY = {
    'Valuation': {'dark': '#2F5496', 'light': '#D6E4F0', 'weight_key': 'score_weight_valuation'},
    'Quality': {'dark': '#548235', 'light': '#E2EFDA', 'weight_key': 'score_weight_quality'},
    'Moat': {'dark': '#C55A11', 'light': '#FCE4CC', 'weight_key': 'score_weight_moat'},
    'Growth': {'dark': '#7030A0', 'light': '#E4CCEF', 'weight_key': 'score_weight_growth'},
    'Ownership': {'dark': '#BF8F00', 'light': '#FFF2CC', 'weight_key': 'score_weight_ownership'},
}


def gate_metadata(params=None):
    """Return Matrix/report metadata derived from the active gate definitions."""
    p = params or {}
    gate_categories = {gate_name: category for gate_name, _, category, *_ in SCORING_GATES}
    gates = []
    for gate_name, *_ in SCREENING_GATES:
        short = _gate_short(gate_name)
        display = _GATE_DISPLAY.get(short, {})
        gates.append({
            'key': _gate_key(gate_name),
            'label': display.get('label', gate_name.split(': ')[1]),
            'gpKey': _gp_key(gate_name),
            'scoreKey': _score_key(gate_name),
            'threshold': display.get('threshold', ''),
            'category': gate_categories.get(gate_name, gate_name.split(': ')[0]),
            'fmt': display.get('fmt', 'ratio'),
        })

    categories = []
    for name in ('Valuation', 'Quality', 'Moat', 'Growth', 'Ownership'):
        display = _CATEGORY_DISPLAY[name]
        categories.append({
            'name': name,
            'weight': p.get(display['weight_key'], globals()[display['weight_key'].upper()]),
            'dark': display['dark'],
            'light': display['light'],
            'scoreKey': '_score_' + name.lower(),
        })
    return {'gates': gates, 'categories': categories}


def _rating_cap_for_row(row, params=None):
    """Return (cap, reasons) for critical investability failures only."""
    reasons = []
    cap = None

    def add(new_cap, reason):
        nonlocal cap
        if cap is None or RATING_RANK[new_cap] < RATING_RANK[cap]:
            cap = new_cap
        reasons.append(reason)

    price_fv = row.get('_price_fv')
    mos = row.get('mos')

    if row.get('price') is None or row.get('dcf_fv') is None:
        add('HOLD', 'missing price or fair value')
    elif price_fv is not None:
        if price_fv >= 1.20:
            add('PASS', 'price/fair value >= 1.20')
        elif price_fv >= 1.00:
            add('HOLD', 'price/fair value >= 1.00')
    elif mos is not None:
        if mos <= -0.10:
            add('PASS', 'margin of safety <= -10%')
        elif mos <= 0:
            add('HOLD', 'non-positive margin of safety')

    if row.get('beneish_flag') is True:
        add('HOLD', 'Beneish manipulation flag')
    if row.get('altman_z_zone') == 'distress':
        add('HOLD', 'Altman Z distress zone')
    edgar_q = row.get('edgar_quality_score')
    if edgar_q is not None and edgar_q < 40:
        add('HOLD', 'low EDGAR data quality')
    if row.get('_data_coverage_score') is not None and row['_data_coverage_score'] < 25:
        add('HOLD', 'low scoring data coverage')

    return cap, reasons


def apply_rating_caps(results, params=None):
    """Apply critical-only rating caps and expose raw/final rating diagnostics."""
    for r in results:
        raw = rating_from_composite(r.get('_composite_score'), params)
        r['_rating_from_score'] = raw
        r['rating_raw'] = raw
        cap, reasons = _rating_cap_for_row(r, params=params)
        r['_rating_cap'] = cap
        r['_rating_cap_reasons'] = reasons
        r['rating'] = _cap_rating(raw, cap) if cap and raw else raw


def score_and_rate(results, params=None):
    """Run the canonical scoring workflow used by live, replay, and rescore paths."""
    apply_screening_matrix(results)
    compute_continuous_scores(results, params=params)
    apply_rating_caps(results, params=params)
    return results
