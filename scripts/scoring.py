# scripts/scoring.py
"""Scoring, screening, and validation functions for the stock analysis pipeline."""

from scripts.config import (SCORE_WEIGHT_VALUATION, SCORE_WEIGHT_QUALITY,
                             SCORE_WEIGHT_MOAT, SCORE_WEIGHT_GROWTH,
                             SCORE_WEIGHT_OWNERSHIP)

MIN_SECTOR_SCORING = 5  # Min stocks per sector for sector-relative percentile


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
     'gross_margin',
     lambda v, r: v > 0.35 if v is not None else None),
    ('Growth: Fund Growth',
     'fundamental_growth',
     lambda v, r: v > 0.03 if v is not None else None),
    ('Growth: Margins',
     'margin_trend',
     lambda v, r: v >= 0 if v is not None else None),
    ('Growth: ROE',
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
]


def apply_screening_matrix(results):
    """Evaluate each stock against pass/fail gates and override ratings.

    Stores per-gate actual data values and pass/fail booleans in each row dict.
    _gate_* fields hold the raw metric value (number), _gp_* fields hold
    True (pass) / False (fail) / None (N/A) for colour formatting.
    Only downgrades ratings — never upgrades.
    """
    # Pre-compute derived gate fields
    for r in results:
        sbc = r.get('sbc')
        rev = r.get('revenue')
        fcf = r.get('fcf')
        r['sbc_pct_rev'] = (sbc / rev) if (sbc is not None and rev and rev > 0) else None
        r['fcf_margin'] = (fcf / rev) if (fcf is not None and rev and rev > 0) else None

    for r in results:
        r['rating_raw'] = r['rating']
        passed = 0
        total = 0
        for gate_name, field, test_fn in SCREENING_GATES:
            val = r.get(field)
            result = test_fn(val, r)
            gate_key = '_gate_' + gate_name.split(': ')[1].lower().replace(' ', '_').replace('/', '_')
            gp_key = '_gp' + gate_key[5:]  # _gate_mos → _gp_mos
            if result is None:
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
                total += 1

        r['_gates_passed'] = f'{passed}/{total}' if total > 0 else 'N/A'
        r['_gates_passed_num'] = passed if total > 0 else -1


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
    ('Moat: Gross Margin', 'gross_margin', 'Moat',
     lambda v, r, pct: pct, 'sector', True),
    ('Growth: Fund Growth', 'fundamental_growth', 'Growth',
     lambda v, r, pct: _score_linear(v, 0.0, 0.10), False, True),
    ('Growth: Margins', 'margin_trend', 'Growth',
     lambda v, r, pct: _score_linear(v, -0.05, 0.05), False, True),
    ('Growth: ROE', 'roe', 'Growth',
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
    # 12-minus-1 month price momentum (skips last month to avoid short-term reversal).
    # Scored against the Growth category — not a pure fundamental signal, but confirms
    # that business trajectory is visible in the market.  Scored as a weak signal
    # (moderate range) so it doesn't dominate fundamentals.
    ('Growth: Momentum', 'momentum_12_1', 'Growth',
     lambda v, r, pct: _score_linear(v, -0.25, 0.35), False, True),   # -25%→0, +35%→100
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
                sorted_pool = sorted(pool, key=lambda x: x[1])
                n = len(sorted_pool)
                # Only assign percentiles to stocks in this sector group
                group_indices = set(i for i, _ in group)
                for rank_idx, (orig_idx, val) in enumerate(sorted_pool):
                    if orig_idx in group_indices:
                        pctile = (rank_idx / (n - 1)) * 100 if n > 1 else 50
                        if not higher_better:
                            pctile = 100 - pctile
                        results[orig_idx].setdefault('_pctile', {})[pctile_key] = pctile

        else:  # 'global'
            sorted_vals = sorted(all_vals, key=lambda x: x[1])
            n = len(sorted_vals)
            for rank_idx, (orig_idx, val, _) in enumerate(sorted_vals):
                pctile = (rank_idx / (n - 1)) * 100 if n > 1 else 50
                if not higher_better:
                    pctile = 100 - pctile
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
    for r in results:
        category_scores = {}
        for gate_name, field, category, score_fn, relative_mode, higher_better in SCORING_GATES:
            val = r.get(field)
            if val is None:
                continue
            pct = r.get('_pctile', {}).get(f'{gate_name}_{field}', 50) if relative_mode else None
            score = score_fn(val, r, pct)
            if score is not None:
                # Store per-gate score
                short = gate_name.split(': ')[1].lower().replace(' ', '_').replace('/', '_')
                r[f'_score_{short}'] = round(score, 1)
                category_scores.setdefault(category, []).append(score)

        # Category averages
        cat_avgs = {}
        for cat, scores in category_scores.items():
            cat_avgs[cat] = sum(scores) / len(scores) if scores else 0
        r['_score_valuation'] = round(cat_avgs.get('Valuation', 0), 1)
        r['_score_quality'] = round(cat_avgs.get('Quality', 0), 1)
        r['_score_moat'] = round(cat_avgs.get('Moat', 0), 1)
        r['_score_growth'] = round(cat_avgs.get('Growth', 0), 1)
        r['_score_ownership'] = round(cat_avgs.get('Ownership', 0), 1)

        # Weighted composite score
        weighted_sum = 0
        weight_total = 0
        for cat, avg in cat_avgs.items():
            w = cat_weights.get(cat, 0)
            weighted_sum += avg * w
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

        # Clean up temp
        r.pop('_pctile', None)


def apply_composite_rating_override(results, params=None):
    """Override ratings using the weighted composite score.

    Sole arbiter of rating overrides — only downgrades, never upgrades.
    Thresholds are calibrated so that rating, composite, and gate pass ratio
    move together directionally:

      BUY        requires composite >= 60  (strong across weighted categories)
      LEAN BUY   requires composite >= 40  (solid but not exceptional)
      HOLD       composite < 40            (meaningful weakness somewhere)

    Args:
        results: List of stock result dicts.
        params: Optional ParamSet dict (reserved for future threshold tuning).
    """
    for r in results:
        cs = r.get('_composite_score')
        if cs is None:
            continue
        rating = r['rating']
        if rating == 'BUY' and cs < 40:
            r['rating'] = 'HOLD'
        elif rating == 'BUY' and cs < 60:
            r['rating'] = 'LEAN BUY'
        elif rating == 'LEAN BUY' and cs < 40:
            r['rating'] = 'HOLD'
