# scripts/scoring.py
"""Scoring, screening, and validation functions for the stock analysis pipeline."""

import statistics
from collections import namedtuple

from scripts.config import (SCORE_WEIGHT_VALUATION, SCORE_WEIGHT_QUALITY,
                             SCORE_WEIGHT_MOAT, SCORE_WEIGHT_GROWTH,
                             SCORE_WEIGHT_OWNERSHIP)

# Gate specs. `weight` scales a gate's contribution within its category
# (default 1.0). `applicable` is an optional row-predicate: when it returns
# False the gate is STRUCTURALLY INAPPLICABLE for that ticker (a bank has no
# meaningful FCF yield; negative tangible book makes P/TBV undefined) and is
# excluded from numerator AND denominator — unlike missing data, which still
# scores 0 (worst). `applicable=None` means always applicable.
ScoringGate = namedtuple(
    'ScoringGate',
    ['name', 'field', 'category', 'score_fn', 'relative_mode', 'higher_better',
     'weight', 'applicable'],
    defaults=(1.0, None))
ScreeningGate = namedtuple(
    'ScreeningGate', ['name', 'field', 'test_fn', 'applicable'],
    defaults=(None,))


def _gate_applicable(gate, row):
    """True unless the gate declares an applicability predicate that fails."""
    return gate.applicable is None or gate.applicable(row)


# ---------------------------------------------------------------------------
# Applicability predicates (shared by SCREENING_GATES and SCORING_GATES)
# ---------------------------------------------------------------------------

def _appl_non_financial(r):
    """FCF- and EV-based gates are structurally meaningless for banks,
    insurers, and brokers: operating cash flow reflects deposit/loan/float
    movements and enterprise value is distorted by deposit funding."""
    return r.get('sector') != 'Financial Services'


def _appl_positive_tbv(r):
    """P/TBV is undefined for negative tangible book (buyback-rich
    compounders). Only exempt when the raw TBV is PRESENT and <= 0 —
    a missing balance sheet stays applicable so it scores worst."""
    tbv = r.get('tangible_book_ps')
    return not (isinstance(tbv, (int, float)) and tbv <= 0)


def _appl_insider_activity(r):
    """Insider buy-ratio only means something with a minimum of open-market
    activity; quiet insiders (or a failed Form 4 fetch — both counts None)
    are no-signal, not bearish."""
    total = (r.get('insider_buy_count_365d') or 0) + \
            (r.get('insider_sell_count_365d') or 0)
    return total >= 4


def _appl_margin_history(r):
    """Margin-vs-history needs a real through-cycle baseline."""
    return (r.get('op_margin_hist_years') or 0) >= 5


def _appl_incr_roic(r):
    """Incremental ROIC is undefined (not bad) when the capital base shrank —
    a capital-light compounder returning cash must not score 0."""
    return not r.get('_incr_roic_undefined')

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
#
# Order matters — the Gate Matrix renders columns within each category in the
# order they appear here. Grouping is intentional:
#   Valuation:  DCF-derived → multiples → alternative models
#   Quality:    leverage → earnings quality → returns → composite
#   Moat:       ROIC family (spread, consistency, trend) → margin family
#   Growth:     top-line → bottom-line → margin trend → composite
#   Ownership:  share-count direction (yield, buyback, shrink) → dilution → alignment
SCREENING_GATES = [
    # ---- Valuation ----
    ('Valuation: MoS',
     'mos',
     lambda v, r: v > 0.10 if v is not None else None),
    ('Valuation: FV Dispersion',
     'fv_dispersion',
     # MAD/median statistic (see prepare_scoring_fields); 0.15 preserves the
     # ~29% pass rate the old (max−min)/median @0.50 produced. Provisional.
     lambda v, r: v <= 0.15 if v is not None else None),
    # EBIT/EV earnings yield: capital-structure-neutral market multiple
    # (replaced P/FCF, which was the exact reciprocal of FCF Yield — the
    # same signal counted twice). Absolute threshold ≈ ≤12.5x EV/EBIT.
    ('Valuation: EBIT/EV',
     'ebit_ev',
     lambda v, r: v > 0.08 if v is not None else None,
     _appl_non_financial),
    ('Valuation: P/TBV',
     'p_tbv',
     lambda v, r: 0 < v <= 2.5 if v is not None else None,
     _appl_positive_tbv),
    ('Valuation: EPV Floor',
     'epv_floor_ratio',
     lambda v, r: v >= 1.0 if v is not None else None),
    # FCF yield vs the risk-free rate: an absolute, model-independent value
    # floor (replaced RIM MoS, a third price-vs-intrinsic-value gate that
    # triangulated the same signal as MoS + EPV Floor).
    ('Valuation: FCF Yield',
     'fcf_yield',
     lambda v, r: v > (r.get('_risk_free_rate') or 0.045) if v is not None else None,
     _appl_non_financial),

    # ---- Quality ----
    ('Quality: Int Coverage',
     'int_cov',
     lambda v, r: v > 3.0 if v is not None else None),
    ('Quality: Net Debt/EBITDA',
     'nd_ebitda',
     lambda v, r: v <= 1.5 if v is not None else None),
    ('Quality: Accruals',
     'accruals',
     # Accruals = (NI - CFO) / Assets. Sloan (1996): high *positive* accruals
     # predict negative future returns — earnings exceed cash, often via
     # aggressive recognition. Negative accruals (CFO > NI) indicate
     # conservative accounting and strong cash generation; do not penalize.
     lambda v, r: v < 0.08 if v is not None else None),
    # (Cash Conv removed: CFO-vs-NI is the accruals signal inverted, and
    # Piotroski's internals check it again — three gates, one signal.)
    # Revenue-growth volatility (lower = steadier). Replaced the ROE gate,
    # which double-counted returns-on-capital with the Moat/ROIC block and is
    # distortable by leverage and buybacks.
    ('Quality: Rev Volatility',
     'rev_growth_vol',
     lambda v, r: v < 0.12 if v is not None else None),
    # Over-earning guard: current operating margin vs the company's own
    # ~10y average. All four FV models eat the same point-in-time earnings,
    # so at a cyclical margin peak FV Dispersion is tight and MoS inflated
    # exactly when they shouldn't be — this is the orthogonal check.
    ('Quality: Margin vs Hist',
     'margin_vs_hist',
     lambda v, r: v < 0.05 if v is not None else None,
     _appl_margin_history),
    ('Quality: Piotroski',
     'piotroski',
     lambda v, r: v >= 7 if v is not None else None),

    # ---- Moat ----
    # Name must match the SCORING_GATES entry exactly — the Matrix scoreKey
    # is derived from this name and a mismatch blanks the score column.
    ('Moat: Spread',
     'spread',
     lambda v, r: v > 0.07 if v is not None else None),
    ('Moat: ROIC Consistency',
     'roic_cv',
     lambda v, r: v < 0.30 if v is not None else None),
    # Incremental ROIC (ΔNOPAT/ΔIC over the statement window): the moat-
    # TRAJECTORY test — is each new dollar of capital still earning above
    # the cost of capital? The better replacement for the retired ROIC
    # Trend gate.
    ('Moat: Incr ROIC',
     'incremental_roic',
     lambda v, r: v > 0.10 if v is not None else None,
     _appl_incr_roic),
    # Operating margin vs the sector median: a structural competitive-position
    # signal orthogonal to ROIC. Also carries the margin-vs-sector signal of
    # the retired Gross Margin percentile gate (same axis, operating level).
    ('Moat: Margin Advantage',
     'margin_advantage',
     lambda v, r: v > 0.05 if v is not None else None),
    ('Moat: FCF Margin',
     'fcf_margin',
     lambda v, r: v > 0.12 if v is not None else None,
     _appl_non_financial),

    # ---- Growth ----
    ('Growth: Rev Durability',
     'rev_cagr_10y',
     lambda v, r: v > 0.02 if v is not None else None),
    ('Growth: FCF Durability',
     'fcf_cagr_5y',
     lambda v, r: v > 0.05 if v is not None else None,
     _appl_non_financial),
    ('Growth: Margins',
     'gross_margin_trend',
     lambda v, r: v >= 0 if v is not None else None),
    ('Growth: Fund Growth',
     'fundamental_growth',
     lambda v, r: v > 0.03 if v is not None else None),

    # ---- Ownership ----
    ('Ownership: Shrhldr Yield',
     'shareholder_yield',
     lambda v, r: v > 0.02 if v is not None else None),
    # (Buyback Rate removed: already inside Shareholder Yield, and Share
    # Shrink is its 5y integral — three gates on share-count direction.)
    ('Ownership: Share Shrink',
     'shares_cagr_5y',
     lambda v, r: v < 0 if v is not None else None),
    ('Ownership: SBC Dilution',
     'sbc_pct_rev',
     lambda v, r: v <= 0.02 if v is not None else None),
    ('Ownership: Insider Own',
     'insider_pct',
     lambda v, r: v >= 0.05 if v is not None else None),
    # Open-market insider net buying (Form 4 P vs S transactions): the
    # classic conviction signal — collected by the SEC insider client but
    # never scored until now.
    ('Ownership: Insider Buying',
     'insider_buy_ratio',
     lambda v, r: v >= 0.5 if v is not None else None,
     _appl_insider_activity),
]

# Normalize to gate specs (adds weight/applicable defaults).
SCREENING_GATES = [ScreeningGate(*g) for g in SCREENING_GATES]


def prepare_scoring_fields(results):
    """Populate derived fields shared by gates and continuous scoring."""
    for r in results:
        sbc = r.get('sbc')
        rev = r.get('revenue')
        fcf = r.get('fcf')
        # Fallback: when yfinance surfaced no cash-flow statement (common
        # across the expanded EDGAR universe), use the FCF derived from EDGAR
        # history (OCF − capex, latest fiscal year). edgar_history is already
        # USD-normalized to the same basis as revenue/mcap, so the fcf_margin
        # and pfcf ratios below stay FX-consistent. See derive_edgar_metrics.
        # Financial Services (banks/insurers/brokers) are excluded: their
        # operating cash flow reflects deposit/loan/trading/float movements,
        # so OCF − capex is not a valid FCF proxy — the same reason FCF Margin
        # and Gross Margin are legitimately N/A for financials (they're scored
        # on FDIC KPIs / combined ratios instead).
        if fcf is None and r.get('sector') != 'Financial Services':
            fcf_edgar = r.get('fcf_edgar')
            if fcf_edgar is not None:
                fcf = fcf_edgar
                r['fcf'] = fcf_edgar
                r['_fcf_source'] = 'edgar'
        price = r.get('price')
        # Effective fair value: prefer the DCF; when it's absent (no yfinance
        # cash flow — ~75% of the expanded universe) fall back to a robust
        # consensus (median) of the intrinsic-value models that did resolve.
        # This lets Price/FV and MoS — and the valuation gates and rating caps
        # that key off them — populate for companies the DCF alone couldn't
        # value. Require >=2 models so the consensus never rests on a single
        # estimate. All model FVs are per-share in the quote currency, so
        # price / fv is unit-consistent.
        #
        # Model choice matters for parity with the DCF cohort (median P/FV
        # ~0.90): the fallback must be GROWTH-inclusive so DCF-less names
        # aren't rated more harshly just because of which model resolved.
        #   • epv_growth_fv (EPV with growth) — median P/FV ~1.00, the closest
        #     DCF proxy; used instead of bare epv_fv (growth-agnostic, ~1.35).
        #   • rim_fv, ddm_fv — legitimate conservative cross-checks.
        #   • nav_fv is EXCLUDED: it's an asset floor (median P/FV ~2.45), not
        #     a fair value for operating companies; REITs/financials are valued
        #     on their own NAV/FFO tracks elsewhere.
        dcf_fv = r.get('dcf_fv')
        if dcf_fv is not None and dcf_fv > 0:
            fv_eff, fv_src = dcf_fv, 'dcf'
        else:
            alt = [r.get(k) for k in ('epv_growth_fv', 'rim_fv', 'ddm_fv')]
            alt = [v for v in alt
                   if isinstance(v, (int, float)) and 0 < v < float('inf')]
            if len(alt) >= 2:
                fv_eff, fv_src = statistics.median(alt), 'blend'
            else:
                fv_eff, fv_src = None, None
        r['_fv_effective'] = fv_eff
        r['_fv_source'] = fv_src
        r['sbc_pct_rev'] = (sbc / rev) if (sbc is not None and rev and rev > 0) else None
        r['fcf_margin'] = (fcf / rev) if (fcf is not None and rev and rev > 0) else None
        # _price_fv is retained (= 1 − MoS) for the Excel export and raw
        # consumers, but it is no longer a scored gate or matrix column — MoS
        # carries the same signal (see fv_dispersion below, which replaced the
        # Price/FV gate in the Valuation category).
        r['_price_fv'] = (price / fv_eff) if (price and fv_eff and fv_eff > 0) else None
        r['mos'] = ((fv_eff - price) / fv_eff) if (price and fv_eff and fv_eff > 0) else None
        # Cross-model fair-value agreement: how tightly the comparable intrinsic
        # models corroborate one another. Low dispersion → the fair value (and
        # the MoS derived from it) is trustworthy; wide dispersion → the
        # estimate is noisy and a large MoS may be a value trap. Two fixes vs
        # the naive version:
        #   • Uses the PRE-BLEND DCF (_dcf_fv_preblend) so the DDM leg — which
        #     is blended into dcf_fv upstream — isn't counted twice, which used
        #     to mute the gate for exactly the payers where DCF and DDM diverge.
        #   • Statistic is median-absolute-deviation about the median ÷ median,
        #     a robust spread comparable across 2/3/4 resolving models, rather
        #     than (max−min)/median which grows mechanically with model count.
        # NAV is excluded (asset floor) and bare EPV omitted in favor of the
        # growth-inclusive epv_growth_fv. >=2 models.
        _dcf_disp = r.get('_dcf_fv_preblend', r.get('dcf_fv'))
        fv_models = [_dcf_disp, r.get('epv_growth_fv'), r.get('rim_fv'), r.get('ddm_fv')]
        fv_models = [v for v in fv_models
                     if isinstance(v, (int, float)) and 0 < v < float('inf')]
        if len(fv_models) >= 2:
            _fvm = statistics.median(fv_models)
            if _fvm > 0:
                _mad = statistics.median([abs(v - _fvm) for v in fv_models])
                r['fv_dispersion'] = _mad / _fvm
            else:
                r['fv_dispersion'] = None
        else:
            r['fv_dispersion'] = None
        # Recompute P/FCF from the (possibly EDGAR-derived) FCF when it wasn't
        # set upstream. Only meaningful for positive FCF (mirrors market.py).
        mcap = r.get('mcap')
        if r.get('pfcf') is None:
            r['pfcf'] = (mcap / fcf) if (mcap and fcf and fcf > 0) else None
        # FCF yield (Valuation: FCF Yield gate) — the earnings-power yield at
        # today's price, gated against the run's risk-free rate. Uses the same
        # (possibly EDGAR-derived) fcf; None for financials where fcf is N/A.
        r['fcf_yield'] = (fcf / mcap) if (fcf is not None and mcap and mcap > 0) else None
        # Margin advantage vs sector (Moat: Margin Advantage gate) — operating
        # margin minus the sector median: a company-specific competitive-
        # position signal orthogonal to ROIC. Prefer the precomputed profit-
        # pool value; fall back to operating_margin − the sector median.
        ma = r.get('pp_margin_advantage')
        if ma is None:
            om, sm = r.get('operating_margin'), r.get('_sector_median_opm')
            if isinstance(om, (int, float)) and isinstance(sm, (int, float)):
                ma = om - sm
        r['margin_advantage'] = ma

        # EBIT/EV earnings yield (Valuation: EBIT/EV gate) — capital-
        # structure-neutral counterpart to the equity-based FCF yield.
        op_inc = r.get('operating_income')
        ev = r.get('enterprise_value')
        r['ebit_ev'] = (op_inc / ev) if (
            isinstance(op_inc, (int, float)) and
            isinstance(ev, (int, float)) and ev > 0) else None

        # Incremental ROIC (Moat: Incr ROIC gate) — ΔNOPAT/ΔIC between the
        # first and last common statement years. When the capital base
        # SHRANK (ΔIC ≤ materiality floor) the ratio is undefined, not bad:
        # flag it so the gate goes inapplicable instead of scoring 0.
        r['incremental_roic'] = None
        r['_incr_roic_undefined'] = False
        nopat_by = r.get('_nopat_by_year') or {}
        ic_by = r.get('_ic_by_year') or {}
        common = sorted(set(nopat_by) & set(ic_by))
        if len(common) >= 2:
            first_y, last_y = common[0], common[-1]
            d_nopat = nopat_by[last_y] - nopat_by[first_y]
            d_ic = ic_by[last_y] - ic_by[first_y]
            ic_floor = 0.02 * abs(ic_by[last_y])
            if d_ic > max(0.0, ic_floor):
                # Clamp for sane display; the score range is far narrower.
                r['incremental_roic'] = max(-1.0, min(1.0, d_nopat / d_ic))
            else:
                r['_incr_roic_undefined'] = True

        # Margin vs own history (Quality: Margin vs Hist gate) — current
        # operating margin minus the company's ~10y average; the over-
        # earning guard for cyclical peaks.
        om_now = r.get('operating_margin')
        om_avg = r.get('op_margin_avg_10y')
        r['margin_vs_hist'] = (om_now - om_avg) if (
            isinstance(om_now, (int, float)) and
            isinstance(om_avg, (int, float))) else None

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

    # Per-row denominator over APPLICABLE gates only. Two kinds of N/A:
    #   • Structurally inapplicable (gate.applicable(row) is False — e.g.
    #     FCF gates for banks, P/TBV on negative tangible book): excluded
    #     from numerator AND denominator, so a bank isn't mechanically
    #     failed on gates that cannot describe it.
    #   • Missing data (test_fn returns None): still counts as a fail
    #     against the denominator — sparse data stays penalized.
    for r in results:
        r['rating_raw'] = r.get('rating')
        passed = 0
        applicable_total = 0
        inapplicable = 0
        for gate in SCREENING_GATES:
            gate_key = _gate_key(gate.name)
            gp_key = _gp_key(gate.name)
            if not _gate_applicable(gate, r):
                # Structurally inapplicable — renders N/A, counts nowhere.
                r[gate_key] = None
                r[gp_key] = None
                inapplicable += 1
                continue
            applicable_total += 1
            val = r.get(gate.field)
            result = gate.test_fn(val, r)
            if result is None:
                # Missing data — renders N/A but counts as a fail
                # (passed not incremented; still in the denominator).
                r[gate_key] = None
                r[gp_key] = None
            else:
                r[gate_key] = val
                r[gp_key] = bool(result)
                if result:
                    passed += 1

        r['_gates_passed'] = f'{passed}/{applicable_total}'
        r['_gates_passed_num'] = passed
        r['_gates_inapplicable'] = inapplicable


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


# (name, field, category, score_fn, relative_mode, higher_better[, weight[, applicable]])
# relative_mode: False=absolute, 'global'=global percentile, 'sector'=sector percentile
# higher_better: True=higher value→higher percentile, False=lower→higher
# score_fn: (value, row, percentile_or_None) -> 0-100
# weight: within-category weight (default 1.0)
# applicable: optional row-predicate; False = structurally inapplicable
SCORING_GATES = [
    # MoS is the headline price-vs-value signal; double weight within
    # Valuation so it isn't diluted to 1/6 of a 0.30-weight category.
    ('Valuation: MoS', 'mos', 'Valuation',
     lambda v, r, pct: _score_linear(v, -0.10, 0.40), False, True, 2.0),  # tightened: worst raised -20%→-10%
    ('Valuation: FV Dispersion', 'fv_dispersion', 'Valuation',
     lambda v, r, pct: _score_linear(v, 0.70, 0.0), False, True),     # lower is better; range rescaled for the MAD/median statistic
    # EBIT/EV earnings yield (replaced P/FCF — exact reciprocal of FCF
    # Yield). Absolute range, not rf-relative: FCF Yield already carries
    # the rate-regime beta for the category. 6% ≈ market-average EV/EBIT.
    ('Valuation: EBIT/EV', 'ebit_ev', 'Valuation',
     lambda v, r, pct: _score_linear(v, 0.0, 0.12), False, True, 1.0,
     _appl_non_financial),
    ('Quality: Int Coverage', 'int_cov', 'Quality',
     lambda v, r, pct: _score_linear(min(v, 40) if v is not None else None, 1.0, 20.0),
     False, True),
    ('Quality: Accruals', 'accruals', 'Quality',
     lambda v, r, pct: pct, 'sector', False),
    ('Ownership: Shrhldr Yield', 'shareholder_yield', 'Ownership',
     lambda v, r, pct: _score_linear(v, -0.01, 0.08), False, True),
    ('Ownership: Insider Own', 'insider_pct', 'Ownership',
     lambda v, r, pct: _score_linear(v, 0.0, 0.15), False, True),
    # Open-market insider net buying; neutral 0.5 buy-ratio scores 50.
    # Inapplicable (not zero) below 4 open-market transactions/365d.
    ('Ownership: Insider Buying', 'insider_buy_ratio', 'Ownership',
     lambda v, r, pct: _score_linear(v, 0.0, 1.0), False, True, 1.0,
     _appl_insider_activity),
    ('Moat: ROIC Consistency', 'roic_cv', 'Moat',
     lambda v, r, pct: _score_linear(v, 0.60, 0.0), False, True),
    ('Moat: Spread', 'spread', 'Moat',
     lambda v, r, pct: _score_linear(v, 0.0, 0.20), False, True),     # tightened: best 25%→20%
    # Incremental ROIC: moat trajectory — return on each NEW dollar of
    # invested capital over the statement window.
    ('Moat: Incr ROIC', 'incremental_roic', 'Moat',
     lambda v, r, pct: _score_linear(v, -0.05, 0.25), False, True, 1.0,
     _appl_incr_roic),
    ('Growth: Fund Growth', 'fundamental_growth', 'Growth',
     lambda v, r, pct: _score_linear(v, 0.0, 0.10), False, True),
    ('Growth: Margins', 'gross_margin_trend', 'Growth',
     lambda v, r, pct: _score_linear(v, -0.05, 0.05), False, True),
    ('Quality: Rev Volatility', 'rev_growth_vol', 'Quality',
     lambda v, r, pct: _score_linear(v, 0.40, 0.05), False, True),    # lower is better: steady top line scores high
    # Over-earning guard (one-sided): current op margin 8pp+ above the
    # company's own ~10y average scores 0; at/below history clamps to 100.
    # Margin DETERIORATION is already penalized by Growth: Margins.
    ('Quality: Margin vs Hist', 'margin_vs_hist', 'Quality',
     lambda v, r, pct: _score_linear(v, 0.08, 0.0), False, True, 1.0,
     _appl_margin_history),
    # Buffett additions
    ('Quality: Net Debt/EBITDA', 'nd_ebitda', 'Quality',
     lambda v, r, pct: _score_linear(v, 4.0, -0.5), False, True),     # tightened: worst 5→4, best 0→-0.5 (net cash rewarded)
    ('Growth: Rev Durability', 'rev_cagr_10y', 'Growth',
     lambda v, r, pct: _score_linear(v, -0.05, 0.15), False, True),
    ('Ownership: SBC Dilution', 'sbc_pct_rev', 'Ownership',
     lambda v, r, pct: _score_linear(v, 0.06, 0.0), False, True),     # tightened: worst 10%→6%
    ('Valuation: P/TBV', 'p_tbv', 'Valuation',
     lambda v, r, pct: _score_linear(v, 5.0, 1.0) if v is not None and v > 0 else None,
     False, True, 1.0, _appl_positive_tbv),
    ('Moat: FCF Margin', 'fcf_margin', 'Moat',
     lambda v, r, pct: _score_linear(v, 0.05, 0.25), False, True, 1.0,
     _appl_non_financial),                                            # tightened: worst 0%→5%, best 20%→25%
    ('Growth: FCF Durability', 'fcf_cagr_5y', 'Growth',
     lambda v, r, pct: _score_linear(v, -0.05, 0.15), False, True, 1.0,
     _appl_non_financial),
    ('Ownership: Share Shrink', 'shares_cagr_5y', 'Ownership',
     lambda v, r, pct: _score_linear(v, 0.04, -0.04), False, True),
    # Migrated from compute_rating
    ('Quality: Piotroski', 'piotroski', 'Quality',
     lambda v, r, pct: _score_linear(v, 0, 9), False, True),
    ('Moat: Margin Advantage', 'margin_advantage', 'Moat',
     lambda v, r, pct: _score_linear(v, -0.10, 0.20), False, True),   # opm vs sector: sector-median→33, +20pp→100
    ('Valuation: EPV Floor', 'epv_floor_ratio', 'Valuation',
     lambda v, r, pct: _score_linear(v, 0.5, 1.2), False, True),
    ('Valuation: FCF Yield', 'fcf_yield', 'Valuation',
     lambda v, r, pct: (_score_linear(v - (r.get('_risk_free_rate') or 0.045), -0.03, 0.08)
                        if v is not None else None), False, True, 1.0,
     _appl_non_financial),
]

# Normalize to gate specs (adds weight/applicable defaults).
SCORING_GATES = [ScoringGate(*g) for g in SCORING_GATES]


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
    for gate in SCORING_GATES:
        if not gate.relative_mode:
            continue
        gate_name, field, higher_better = gate.name, gate.field, gate.higher_better
        relative_mode = gate.relative_mode

        # Collect all values with sector info; rows where the gate is
        # structurally inapplicable stay out of the ranking pools.
        all_vals = [(i, r.get(field), r.get('sector') or '_unknown')
                    for i, r in enumerate(results)
                    if r.get(field) is not None and _gate_applicable(gate, r)]
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
    all_categories = []
    for gate in SCORING_GATES:
        if gate.category not in all_categories:
            all_categories.append(gate.category)

    for r in results:
        # Per-row weighted category averages over APPLICABLE gates:
        #   • Structurally inapplicable gates (gate.applicable(row) False)
        #     are excluded from numerator AND denominator — the category
        #     renormalizes over the gates that can describe this business.
        #   • Missing data still scores 0 (worst) against its full weight,
        #     so sparse-data tickers stay penalized.
        cat_score_sums = {cat: 0.0 for cat in all_categories}
        cat_weight_sums = {cat: 0.0 for cat in all_categories}
        applicable_gates = 0
        covered_gates = 0
        for gate in SCORING_GATES:
            if not _gate_applicable(gate, r):
                # Inapplicable → score stays None so matrix cells render
                # N/A rather than a misleading 0.0.
                r[_score_key(gate.name)] = None
                continue
            applicable_gates += 1
            val = r.get(gate.field)
            if val is None:
                # N/A → 0 (counts as worst, included in denominator)
                score = 0.0
            else:
                covered_gates += 1
                pct = (r.get('_pctile', {}).get(f'{gate.name}_{gate.field}', 50)
                       if gate.relative_mode else None)
                s = gate.score_fn(val, r, pct)
                score = s if s is not None else 0.0
            r[_score_key(gate.name)] = round(score, 1)
            cat_score_sums[gate.category] += score * gate.weight
            cat_weight_sums[gate.category] += gate.weight

        # Category averages over the applicable weight mass. A category with
        # zero applicable weight scores None and drops out of the composite
        # (unreachable with current masks — defensive only).
        cat_avgs = {
            cat: (cat_score_sums[cat] / cat_weight_sums[cat]
                  if cat_weight_sums[cat] > 0 else None)
            for cat in all_categories
        }
        for cat in all_categories:
            key = '_score_' + cat.lower()
            r[key] = round(cat_avgs[cat], 1) if cat_avgs[cat] is not None else None

        # Weighted composite over categories that scored
        weighted_sum = 0.0
        weight_total = 0.0
        for cat in all_categories:
            if cat_avgs[cat] is None:
                continue
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
        # Coverage over applicable gates only: a bank shouldn't read as
        # low-coverage because gates that can't describe it are absent.
        r['_data_coverage_score'] = (
            round(covered_gates / applicable_gates * 100, 1)
            if applicable_gates > 0 else None)

        # Clean up temp
        r.pop('_pctile', None)


def rating_from_composite(composite, params=None):
    """Map a 0-100 composite score to a rating bucket.

      BUY       composite >= 57
      LEAN BUY  composite >= 39
      HOLD      composite >= 25
      PASS      composite <  25

    Thresholds quantile-matched against the rescored 2026-07-02 universe
    (n=2211) after the 2026-07 gate rebalance, reproducing the target
    ~0.6% BUY / ~25% LEAN / ~49% HOLD / ~24% PASS distribution.
    PROVISIONAL: old snapshots carry no per-year NOPAT/invested-capital,
    so Incr ROIC scores 0 on rescores; the first live run will lift
    composites slightly — revisit then (the weekly calibrate job also
    searches these thresholds against forward returns).

    Returns None when composite is None. Thresholds tunable via params.
    """
    if composite is None:
        return None
    p = params or {}
    if composite >= p.get('rating_threshold_buy', 57):
        return 'BUY'
    if composite >= p.get('rating_threshold_lean', 39):
        return 'LEAN BUY'
    if composite >= p.get('rating_threshold_pass', 25):
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
    'fv_dispersion': {'label': 'FV Dispersion', 'threshold': 'Model MAD <= 15%', 'fmt': 'pct1'},
    'ebit_ev': {'label': 'EBIT/EV', 'threshold': 'EBIT/EV > 8%', 'fmt': 'pct1'},
    'int_coverage': {'label': 'Int Cov', 'threshold': 'IC > 3x', 'fmt': 'ratio'},
    'accruals': {'label': 'Accruals', 'threshold': '|Acr| < 8%', 'fmt': 'pct1'},
    'shrhldr_yield': {'label': 'Shrhldr Yld', 'threshold': 'Yield > 2%', 'fmt': 'pct1'},
    'insider_own': {'label': 'Insider %', 'threshold': 'Insider >= 5%', 'fmt': 'pct1'},
    'insider_buying': {'label': 'Insider Buys', 'threshold': 'Buy ratio >= 50%', 'fmt': 'pct1'},
    'roic_consistency': {'label': 'ROIC CV', 'threshold': 'CV < 30%', 'fmt': 'pct1'},
    'spread': {'label': 'Spread', 'threshold': 'Spread > 7%', 'fmt': 'pct1'},
    'incr_roic': {'label': 'Incr ROIC', 'threshold': 'ΔNOPAT/ΔIC > 10%', 'fmt': 'pct1'},
    'fund_growth': {'label': 'Fund Growth', 'threshold': 'FG > 3%', 'fmt': 'pct1'},
    'margins': {'label': 'Margins', 'threshold': 'Margin >= 0', 'fmt': 'pct1'},
    'rev_volatility': {'label': 'Rev Vol', 'threshold': 'Rev growth σ < 12%', 'fmt': 'pct1'},
    'margin_vs_hist': {'label': 'Mgn vs Hist', 'threshold': 'OpM < hist avg + 5pp', 'fmt': 'pct1'},
    'net_debt_ebitda': {'label': 'ND/EBITDA', 'threshold': 'ND/EBITDA <= 1.5x', 'fmt': 'ratio'},
    'rev_durability': {'label': '10Y Rev CAGR', 'threshold': '10Y RevCAGR > 2%', 'fmt': 'pct1'},
    'sbc_dilution': {'label': 'SBC/Rev', 'threshold': 'SBC/Rev <= 2%', 'fmt': 'pct1'},
    'p_tbv': {'label': 'P/TBV', 'threshold': 'P/TBV <= 2.5x', 'fmt': 'ratio'},
    'fcf_margin': {'label': 'FCF Margin', 'threshold': 'FCF Margin > 12%', 'fmt': 'pct1'},
    'fcf_durability': {'label': '5Y FCF CAGR', 'threshold': '5Y FCF CAGR > 5%', 'fmt': 'pct1'},
    'share_shrink': {'label': 'Share Shrink', 'threshold': '5Y Shares CAGR < 0', 'fmt': 'pct1'},
    'piotroski': {'label': 'Piotroski', 'threshold': 'F-Score >= 7', 'fmt': 'int'},
    'margin_advantage': {'label': 'Margin Adv', 'threshold': 'OpM vs sector > 5pp', 'fmt': 'pct1'},
    'epv_floor': {'label': 'EPV Floor', 'threshold': 'EPV/Price >= 1.0', 'fmt': 'ratio'},
    'fcf_yield': {'label': 'FCF Yield', 'threshold': 'FCF Yield > risk-free', 'fmt': 'pct1'},
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
    gate_categories = {g.name: g.category for g in SCORING_GATES}
    gate_weights = {g.name: g.weight for g in SCORING_GATES}
    gates = []
    for gate in SCREENING_GATES:
        gate_name = gate.name
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
            'weight': gate_weights.get(gate_name, 1.0),
        })

    categories = []
    # Order matters — driver of left-to-right column order in the Financial
    # Model matrix and the per-stock detail popup. Moat leads for historical
    # continuity (ties broken by the historical Val→Qual and Growth→Own
    # ordering so a sector reads consistently across releases).
    for name in ('Moat', 'Valuation', 'Quality', 'Growth', 'Ownership'):
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

    mos = row.get('mos')

    # Key on the EFFECTIVE fair value (DCF, or the multi-model consensus
    # fallback when the DCF didn't resolve) — keying on dcf_fv alone
    # unconditionally capped every DCF-less name at HOLD, defeating the
    # consensus fallback's documented purpose.
    if row.get('price') is None or row.get('_fv_effective') is None:
        add('HOLD', 'missing price or fair value')
    elif mos is not None:
        # MoS carries the price-vs-fair-value signal that Price/FV used to
        # (MoS = 1 − P/FV); thresholds mirror the retired P/FV caps
        # (P/FV >= 1.20 ⇔ MoS <= −20%, P/FV >= 1.00 ⇔ MoS <= 0).
        if mos <= -0.20:
            add('PASS', 'margin of safety <= -20%')
        elif mos <= 0:
            add('HOLD', 'non-positive margin of safety')

    if row.get('beneish_flag') is True:
        add('HOLD', 'Beneish manipulation flag')
    if row.get('altman_z_zone') == 'distress':
        add('HOLD', 'Altman Z distress zone')
    if row.get('fx_fetch_failed') is True:
        # A failed FX lookup means the statements feeding every FV model are in
        # mixed currencies — the fair value (and MoS) can't be trusted for a BUY.
        add('HOLD', 'FX conversion failed (mixed-currency inputs)')
    edgar_q = row.get('edgar_quality_score')
    if edgar_q is not None and edgar_q < 40:
        add('HOLD', 'low EDGAR data quality')
    if row.get('_data_coverage_score') is not None and row['_data_coverage_score'] < 25:
        add('HOLD', 'low scoring data coverage')

    eh = row.get('edgar_history') or {}
    years = eh.get('years_available', 0) or 0
    if years < 5:
        add('HOLD', f'thin EDGAR history ({years}y)')

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


def _purge_stale_gate_fields(results):
    """Drop _gate_*/_gp_*/_score_* keys for gates that no longer exist.

    Snapshot rows round-trip through re-scoring: without this, fields from
    retired gates persist forever and leak into the report payloads.
    """
    valid = set()
    for g in SCREENING_GATES:
        valid.add(_gate_key(g.name))
        valid.add(_gp_key(g.name))
    for g in SCORING_GATES:
        valid.add(_score_key(g.name))
        valid.add('_score_' + g.category.lower())  # category averages
    for r in results:
        stale = [k for k in r
                 if (k.startswith('_gate_') or k.startswith('_gp_') or
                     k.startswith('_score_')) and k not in valid]
        for k in stale:
            del r[k]


def score_and_rate(results, params=None):
    """Run the canonical scoring workflow used by live, replay, and rescore paths."""
    _purge_stale_gate_fields(results)
    apply_screening_matrix(results)
    compute_continuous_scores(results, params=params)
    apply_rating_caps(results, params=params)
    return results
