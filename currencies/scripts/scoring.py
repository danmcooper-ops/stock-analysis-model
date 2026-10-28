"""Screening and scoring for the currency model.

Architecture is lifted from the sibling stock-analysis-model: a single `Gate`
list drives the pass/fail matrix, the continuous 0-100 score, and the report's
column metadata simultaneously, so a threshold and its scoring curve live on
the same line and cannot drift apart.

The N/A policy is the part worth reading carefully, because it matters more
for currencies than it did for equities:

  * **Structurally inapplicable** (`gate.applicable(row)` is False) — excluded
    from numerator AND denominator. The pillar renormalizes over the gates
    that can describe this currency. A hard-pegged currency has no meaningful
    volatility signal (it measures the anchor); Singapore has no policy rate
    because MAS runs exchange-rate policy; USD cannot carry against itself.
  * **Missing data** (value is None but the gate applies) — scores 0.0 and
    stays in the denominator. Sparse coverage stays penalized.

Collapsing these two would either hand pegs a free sweep of the Stability
pillar or punish them for a question that does not apply. Both are wrong in
opposite directions.
"""

from collections import namedtuple

from data import currency_meta
from scripts.config import (
    HYPERINFLATION_PCT, MIN_DATA_COVERAGE, MIN_REER_MONTHS,
    PILLAR_COLORS, PILLAR_ORDER, PILLAR_WEIGHTS,
    RATING_THRESHOLD_LEAN, RATING_THRESHOLD_LONG, RATING_THRESHOLD_NEUTRAL,
)


class Gate(namedtuple('Gate',
                      ['name', 'field', 'test_fn', 'score_fn',
                       'relative_mode', 'higher_better', 'weight',
                       'applicable'],
                      defaults=(False, True, 1.0, None))):
    """One screening/scoring criterion.

    test_fn:  (value, row) -> bool | None   None = missing data, renders N/A
    score_fn: (value, row, percentile_or_None) -> 0-100
    relative_mode: False = absolute, 'global' = percentile-ranked
    higher_better: percentile direction (relative gates only)
    weight: contribution within its pillar (default 1.0)
    applicable: row predicate; False = structurally inapplicable
    """
    __slots__ = ()

    @property
    def category(self):
        return self.name.split(': ')[0]


def _gate_applicable(gate, row):
    return gate.applicable is None or gate.applicable(row)


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def _score_linear(value, worst, best):
    """Map value linearly from [worst, best] to [0, 100], clamped.

    Direction is encoded by argument order: _score_linear(v, 2.0, -2.0) scores
    a lower-is-better metric. Clamping is what handles fat tails — there is no
    separate winsorization step.
    """
    if value is None:
        return None
    if best == worst:
        return 50.0
    score = (value - worst) / (best - worst) * 100
    return max(0.0, min(100.0, score))


def _ranked_percentiles(items, higher_better=True):
    """Average-rank percentiles, so equal values receive equal scores.

    Args:
        items: list of (row_index, value) pairs.
    Returns:
        {row_index: percentile_0_to_100}
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


# ---------------------------------------------------------------------------
# Applicability predicates
# ---------------------------------------------------------------------------

def _appl_not_base(r):
    """Every carry, momentum and drawdown measure in this model is expressed
    versus USD. The base currency has no spread, no return and no drawdown
    against itself — those gates are undefined for it, not zero."""
    return r.get('code') != currency_meta.BASE_CURRENCY


def _appl_carry(r):
    """Carry needs a policy rate on both legs. Singapore is the structural
    case: MAS targets the nominal effective exchange rate band rather than an
    interest rate, so BIS publishes no policy-rate series for SG at all. That
    is an absence of the concept, not an absence of the data."""
    return _appl_not_base(r) and currency_meta.has_policy_rate(r.get('code'))


def _appl_market_risk(r):
    """Volatility, drawdown and momentum gates are inapplicable to hard pegs.

    A pegged currency posts near-zero realized volatility and near-zero
    drawdown by construction. Scoring that would rank the peg top of the
    Stability pillar for a property that belongs to its anchor, and that
    inverts violently the day the peg breaks. Managed floats stay applicable —
    they are leaned against, not fixed."""
    return _appl_not_base(r) and not currency_meta.is_pegged(r.get('code'))


def _appl_reer(r):
    """Valuation needs enough real-exchange-rate history to have a baseline."""
    return (r.get('reer_months') or 0) >= MIN_REER_MONTHS


def _appl_reer_float(r):
    """Real-terms valuation on a hard peg describes the anchor currency's
    trajectory, not an independent judgement about this one."""
    return _appl_reer(r) and not currency_meta.is_pegged(r.get('code'))


# ---------------------------------------------------------------------------
# Gate definitions
#
# One list drives the matrix, the score, and the report columns. Order within
# a pillar is the column order in the Gate Matrix.
#
# Scoring ranges are stated as (worst, best) pairs of economically meaningful
# levels, not as distribution quantiles, so each one can be argued with.
# ---------------------------------------------------------------------------

GATES = [
    # ---- Valuation: cheap in real terms, versus its own history ----
    # Lower REER = cheaper currency, so every range below runs high->low.
    #
    # Double weight: this is the model's primary thesis. Normalizing by each
    # currency's own volatility is what makes a -1.5 z comparable across CHF
    # and TRY, where a raw percentage deviation would not be.
    Gate('Valuation: REER Z-Score', 'reer_z20',
         lambda v, r: v < -0.5 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 2.0, -2.0),
         weight=2.0, applicable=_appl_reer_float),

    Gate('Valuation: REER vs 10y', 'reer_dev10',
         lambda v, r: v < -0.05 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.25, -0.25),
         applicable=_appl_reer_float),

    # Shorter horizon. Not redundant with the 10-year measure: the two
    # disagree exactly when a currency's regime has shifted within the decade,
    # and that disagreement is informative.
    Gate('Valuation: REER vs 5y', 'reer_dev5',
         lambda v, r: v < -0.03 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.20, -0.20),
         applicable=_appl_reer_float),

    # ---- Carry: what holding it pays, net of inflation ----
    # Real carry is double-weighted over nominal deliberately. Ranking on
    # nominal carry alone puts every high-inflation currency at the top of the
    # book, which is the classic way carry strategies lose money.
    Gate('Carry: Real Carry', 'real_carry',
         lambda v, r: v > 1.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -5.0, 8.0),
         weight=2.0, applicable=_appl_carry),

    Gate('Carry: Nominal Carry', 'nominal_carry',
         lambda v, r: v > 0.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -3.0, 8.0),
         applicable=_appl_carry),

    Gate('Carry: Real Rate', 'real_policy_rate',
         lambda v, r: v > 0.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -4.0, 6.0),
         applicable=_appl_carry),

    # Risk-adjusted carry has no natural absolute scale — "good" depends
    # entirely on the volatility regime of the moment — so this one is scored
    # by rank within the universe rather than against a fixed range.
    Gate('Carry: Carry/Vol', 'carry_to_vol',
         lambda v, r: v > 0.15 if v is not None else None,
         lambda v, r, pct: pct,
         relative_mode='global', higher_better=True,
         applicable=_appl_carry),

    # ---- External: can the country fund itself without depreciating? ----
    # Best set at +12 rather than +8 so the large-surplus economies in this
    # universe (SGD, CHF, NOK all run double digits) still separate from each
    # other instead of all clamping to 100.
    Gate('External: Current Account', 'current_account_pct_gdp',
         lambda v, r: v > 0.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -8.0, 12.0)),

    # Three months of import cover is the conventional adequacy floor.
    Gate('External: Reserves Cover', 'reserves_months_imports',
         lambda v, r: v > 3.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.0, 12.0)),

    # Distance outside the 1-3% target band, so deflation is penalized like
    # inflation rather than scoring as maximal price stability.
    Gate('External: Inflation Gap', 'inflation_gap',
         lambda v, r: v < 2.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 10.0, 0.0)),

    Gate('External: GDP Growth', 'gdp_growth_pct',
         lambda v, r: v > 1.5 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -2.0, 6.0)),

    # ---- Momentum: has the repricing started? ----
    Gate('Momentum: 12m-1m', 'ret_12m_1m',
         lambda v, r: v > 0.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.15, 0.15),
         applicable=_appl_market_risk),

    Gate('Momentum: 3m Return', 'ret_3m',
         lambda v, r: v > 0.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.08, 0.08),
         applicable=_appl_market_risk),

    Gate('Momentum: vs 200d MA', 'spot_vs_ma200',
         lambda v, r: v > 0.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.10, 0.10),
         applicable=_appl_market_risk),

    # Real-terms trend. Nominal spot can drift on an inflation differential
    # alone; a rising REER means real purchasing power is actually gaining.
    Gate('Momentum: REER Trend', 'reer_trend_6m',
         lambda v, r: v > 0.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.08, 0.08),
         applicable=_appl_reer_float),

    # ---- Stability: what the position costs in risk ----
    # Ranges here are calibrated against the observed cross-section of
    # floating currencies rather than against textbook extremes. Annualized
    # FX volatility for this universe runs roughly 4-12%, not 3-30%; scoring
    # against the wider range compressed every currency into 67-96 and made
    # the whole pillar a near-constant offset instead of a discriminator.
    Gate('Stability: Volatility', 'vol_1y',
         lambda v, r: v < 0.08 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.15, 0.03),
         applicable=_appl_market_risk),

    Gate('Stability: Max Drawdown', 'max_dd_3y',
         lambda v, r: v < 0.15 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.25, 0.05),
         applicable=_appl_market_risk),

    # Recent volatility against the currency's own baseline. Above 1 means a
    # regime is destabilizing — and this fires months before the annual World
    # Bank indicators in the External pillar would show anything.
    Gate('Stability: Vol Regime', 'vol_ratio',
         lambda v, r: v < 1.0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 1.4, 0.6),
         applicable=_appl_market_risk),

    Gate('Stability: Inflation Vol', 'inflation_vol',
         lambda v, r: v < 2.5 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 6.0, 0.8)),
]


# ---------------------------------------------------------------------------
# Field-key helpers
# ---------------------------------------------------------------------------

def _gate_short(gate_name):
    """Stable suffix for the _gate_* / _gp_* / _score_* field families."""
    return (gate_name.split(': ')[1].lower()
            .replace(' ', '_').replace('/', '_').replace('-', '_'))


def _gate_key(name):
    return '_gate_' + _gate_short(name)


def _gp_key(name):
    return '_gp_' + _gate_short(name)


def _score_key(name):
    return '_score_' + _gate_short(name)


RATING_RANK = {'AVOID': 0, 'NEUTRAL': 1, 'LEAN LONG': 2, 'LONG': 3}
RATING_BY_RANK = {v: k for k, v in RATING_RANK.items()}


def _cap_rating(rating, cap):
    if rating not in RATING_RANK or cap not in RATING_RANK:
        return rating
    return RATING_BY_RANK[min(RATING_RANK[rating], RATING_RANK[cap])]


# ---------------------------------------------------------------------------
# Screening matrix
# ---------------------------------------------------------------------------

def apply_screening_matrix(results):
    """Evaluate every currency against every gate.

    Writes per row: _gate_<k> (raw value), _gp_<k> (True/False/None),
    _gates_passed ("n/m"), _gates_passed_num, _gates_inapplicable.
    """
    for r in results:
        passed = 0
        applicable_total = 0
        inapplicable = 0

        for gate in GATES:
            gk, pk = _gate_key(gate.name), _gp_key(gate.name)

            if not _gate_applicable(gate, r):
                # Structurally inapplicable — renders N/A, counts nowhere.
                r[gk] = None
                r[pk] = None
                inapplicable += 1
                continue

            applicable_total += 1
            val = r.get(gate.field)
            outcome = gate.test_fn(val, r)
            if outcome is None:
                # Missing data — renders N/A but stays in the denominator.
                r[gk] = None
                r[pk] = None
            else:
                r[gk] = val
                r[pk] = bool(outcome)
                if outcome:
                    passed += 1

        r['_gates_passed'] = '%d/%d' % (passed, applicable_total)
        r['_gates_passed_num'] = passed
        r['_gates_applicable'] = applicable_total
        r['_gates_inapplicable'] = inapplicable


# ---------------------------------------------------------------------------
# Continuous scoring
# ---------------------------------------------------------------------------

def compute_continuous_scores(results, params=None):
    """Per-gate scores, pillar averages, and the weighted composite."""
    p = params or {}
    cat_weights = {cat: p.get('score_weight_' + cat.lower(), w)
                   for cat, w in PILLAR_WEIGHTS.items()}

    # --- Pass 1: percentile pools for relative gates ---
    # Rows where the gate is structurally inapplicable stay out of the pool,
    # so a peg cannot shift the ranking of the currencies it is not competing
    # with.
    for gate in GATES:
        if not gate.relative_mode:
            continue
        pool = [(i, r.get(gate.field)) for i, r in enumerate(results)
                if r.get(gate.field) is not None and _gate_applicable(gate, r)]
        if len(pool) < 2:
            continue
        pctiles = _ranked_percentiles(pool, higher_better=gate.higher_better)
        key = '%s_%s' % (gate.name, gate.field)
        for idx, pctile in pctiles.items():
            results[idx].setdefault('_pctile', {})[key] = pctile

    # --- Pass 2: score each row ---
    categories = PILLAR_ORDER

    for r in results:
        cat_score_sums = {c: 0.0 for c in categories}
        cat_weight_sums = {c: 0.0 for c in categories}
        applicable_gates = 0
        covered_gates = 0

        for gate in GATES:
            if not _gate_applicable(gate, r):
                # None, not 0.0 — the matrix must render N/A rather than a
                # misleading zero.
                r[_score_key(gate.name)] = None
                continue

            applicable_gates += 1
            val = r.get(gate.field)
            if val is None:
                score = 0.0          # missing data scores worst, stays in denom
            else:
                covered_gates += 1
                pct = (r.get('_pctile', {}).get('%s_%s' % (gate.name, gate.field), 50.0)
                       if gate.relative_mode else None)
                s = gate.score_fn(val, r, pct)
                score = s if s is not None else 0.0

            r[_score_key(gate.name)] = round(score, 1)
            cat_score_sums[gate.category] += score * gate.weight
            cat_weight_sums[gate.category] += gate.weight

        # Pillar averages over applicable weight mass only.
        cat_avgs = {
            c: (cat_score_sums[c] / cat_weight_sums[c]
                if cat_weight_sums[c] > 0 else None)
            for c in categories
        }
        for c in categories:
            r['_score_pillar_' + c.lower()] = (
                round(cat_avgs[c], 1) if cat_avgs[c] is not None else None)

        # Weighted composite over pillars that scored. The denominator
        # renormalizes, so a pillar dropping out (USD's carry, a peg's
        # momentum) rescales the rest instead of zeroing the composite.
        weighted_sum = 0.0
        weight_total = 0.0
        for c in categories:
            if cat_avgs[c] is None:
                continue
            w = cat_weights.get(c, 0.0)
            weighted_sum += cat_avgs[c] * w
            weight_total += w

        composite = weighted_sum / weight_total if weight_total > 0 else None
        r['_composite_score'] = round(composite, 1) if composite is not None else None
        r['_data_coverage_score'] = (
            round(covered_gates / applicable_gates * 100, 1)
            if applicable_gates > 0 else None)

        r.pop('_pctile', None)


# ---------------------------------------------------------------------------
# Rating
# ---------------------------------------------------------------------------

def rating_from_composite(composite, params=None):
    p = params or {}
    if composite is None:
        return 'UNRATED'
    if composite >= p.get('rating_threshold_long', RATING_THRESHOLD_LONG):
        return 'LONG'
    if composite >= p.get('rating_threshold_lean', RATING_THRESHOLD_LEAN):
        return 'LEAN LONG'
    if composite >= p.get('rating_threshold_neutral', RATING_THRESHOLD_NEUTRAL):
        return 'NEUTRAL'
    return 'AVOID'


def _rating_cap_for_row(r, params=None):
    """Hard investability ceilings applied after scoring.

    These are ordinal caps, not score deductions: a currency can score 80 and
    still be capped at NEUTRAL because the score is answering a question that
    does not apply to it. `rating_raw` is preserved so the UI can show both.

    Caps deliberately fail open — a missing metric never triggers one.
    """
    cap = None
    reasons = []

    def add(new_cap, reason):
        nonlocal cap
        if cap is None or RATING_RANK[new_cap] < RATING_RANK[cap]:
            cap = new_cap
        reasons.append(reason)

    code = r.get('code')

    # The numeraire. Every relative signal in the model is "versus USD", so a
    # recommendation to be long USD against USD is not a coherent output.
    if code == currency_meta.BASE_CURRENCY:
        add('NEUTRAL', 'Base currency — all signals are measured against it')

    # A peg's score describes its anchor. Valuation and momentum gates are
    # already masked off; this stops the remaining pillars from producing a
    # directional call on a currency that is administratively fixed.
    if currency_meta.is_pegged(code):
        add('NEUTRAL', 'Pegged regime — real-rate signals describe the anchor')

    infl = r.get('inflation_pct')
    if infl is not None and infl > HYPERINFLATION_PCT:
        add('AVOID', 'Inflation %.0f%% — monetary regime, not a carry trade' % infl)

    cov = r.get('_data_coverage_score')
    if cov is not None and cov < MIN_DATA_COVERAGE:
        add('NEUTRAL', 'Data coverage %.0f%% below %.0f%% minimum'
            % (cov, MIN_DATA_COVERAGE))

    months = r.get('reer_months')
    if months is not None and months < MIN_REER_MONTHS:
        add('NEUTRAL', 'Only %d months of REER history' % months)

    return cap, reasons


def apply_rating_caps(results, params=None):
    for r in results:
        raw = rating_from_composite(r.get('_composite_score'), params)
        r['rating_raw'] = raw
        cap, reasons = _rating_cap_for_row(r, params)
        r['_rating_cap'] = cap
        r['_rating_cap_reasons'] = reasons
        r['rating'] = _cap_rating(raw, cap) if cap else raw


def score_and_rate(results, params=None):
    """The canonical scoring workflow. Mutates and returns `results`."""
    _purge_stale_gate_fields(results)
    apply_screening_matrix(results)
    compute_continuous_scores(results, params=params)
    apply_rating_caps(results, params=params)
    return results


def _purge_stale_gate_fields(results):
    """Drop _gate_*/_gp_*/_score_* keys for gates that no longer exist.

    Without this, re-scoring an older snapshot leaks retired gates into the
    report, where they render as columns with no metadata behind them.
    """
    live = set()
    for g in GATES:
        live.update((_gate_key(g.name), _gp_key(g.name), _score_key(g.name)))
    live.update('_score_pillar_' + c.lower() for c in PILLAR_ORDER)

    for r in results:
        for k in [k for k in r
                  if (k.startswith('_gate_') or k.startswith('_gp_')
                      or k.startswith('_score_')) and k not in live]:
            r.pop(k, None)


# ---------------------------------------------------------------------------
# Report-facing metadata
# ---------------------------------------------------------------------------

# Prose explanation per gate, surfaced as a hover tooltip in the report.
#
# These say what the measure is and *why it is built that way* — the reasoning
# is otherwise buried in this file's comments, where a reader of the report
# will never see it. A tooltip that only restates its own column label is
# worse than none, so none of these do.
#
# tests/test_gates.py enforces this map against GATES in both directions: no
# gate without a tooltip, no tooltip without a gate.
_GATE_TIPS = {
    'reer_z20':
        'Where the real effective exchange rate sits in its own 20-year '
        'distribution, in standard deviations. Negative is cheap. Normalising '
        'by each currency’s own volatility is what makes −1.5 mean the '
        'same thing for CHF as for TRY, which a raw percentage gap would not. '
        'Double-weighted — this is the model’s primary thesis.',
    'reer_dev10':
        'The current real exchange rate against its own 10-year average. '
        '−20% means it buys a fifth less abroad than its decade norm. '
        'Answers “how far”, where the z-score answers “how unusual”.',
    'reer_dev5':
        'The same measure over five years. Not redundant with the 10-year '
        'reading: the two disagree exactly when a currency’s regime has '
        'shifted within the decade, and that disagreement is itself a signal.',
    'real_carry':
        'This currency’s real policy rate minus the US dollar’s. '
        'Double-weighted over nominal carry on purpose: ranking on nominal '
        'alone puts every high-inflation currency at the top of the book, '
        'which is the classic way carry trades lose money.',
    'nominal_carry':
        'The policy-rate differential against USD, before inflation. What the '
        'position pays in nominal terms — informative, but on its own a trap.',
    'real_policy_rate':
        'Policy rate minus annual CPI inflation. A 37% rate against 35% '
        'inflation is roughly break-even, not a 37% yield.',
    'carry_to_vol':
        'Carry per unit of realised volatility. Four points of carry on a '
        '6%-volatility currency is a different proposition from four points on '
        'a 25% one. Scored by rank within the universe rather than a fixed '
        'scale, because “good” depends on the volatility regime of the moment.',
    'current_account_pct_gdp':
        'The current account balance as a share of GDP; positive is a surplus. '
        'The best single summary of whether a country can fund itself without '
        'selling its own currency.',
    'reserves_months_imports':
        'FX reserves expressed as months of import cover. Three months is the '
        'conventional adequacy floor.',
    'inflation_gap':
        'Distance outside the 1–3% target band, in percentage points. '
        'Scored as a gap rather than a level so that deflation is penalised '
        'like inflation — on a raw reading, a deflating economy would screen '
        'as maximally price-stable.',
    'gdp_growth_pct':
        'Annual real GDP growth. Slow-moving and one to two years stale by '
        'construction, so this pillar acts as a solvency filter rather than a '
        'timing signal.',
    'ret_12m_1m':
        'The twelve-month return against USD, excluding the most recent month. '
        'The skip is deliberate: one-month FX moves tend to reverse while '
        'three-to-twelve-month moves tend to continue, and blending the two '
        'blunts both.',
    'ret_3m':
        'Spot return against the US dollar over roughly three months. Positive '
        'means the currency appreciated.',
    'spot_vs_ma200':
        'Spot against its own 200-day moving average. Above zero is trading '
        'above trend.',
    'reer_trend_6m':
        'Six-month change in the real effective exchange rate. Nominal spot '
        'can drift on an inflation differential alone; a rising REER means '
        'real purchasing power is genuinely improving.',
    'vol_1y':
        'Annualised standard deviation of daily log returns over one year. '
        'The scoring range is calibrated to this universe’s observed '
        '4–12% span rather than textbook extremes — a wider range '
        'compressed every currency into the same score and made the pillar a '
        'constant offset.',
    'max_dd_3y':
        'The largest peak-to-trough fall against USD over three years. Catches '
        'what volatility misses: a managed crawl posts very low volatility '
        'while still losing a great deal of ground.',
    'vol_ratio':
        'Three-month volatility divided by one-year. Above 1 means the '
        'currency is more turbulent than its own baseline — an early '
        'warning that fires months before annual macro data would show '
        'anything.',
    'inflation_vol':
        'Standard deviation of the annual inflation series. Persistent price '
        'instability, which is a different fact from the current inflation '
        'level scored in the External pillar.',
}

# Pillar-level explanations, including why each carries the weight it does.
_PILLAR_TIPS = {
    'Valuation':
        'Is the real exchange rate cheap against its own history? The heaviest '
        'pillar, because mean reversion in the real exchange rate is the '
        'best-evidenced medium-horizon effect in FX — and because it rests '
        'entirely on BIS REER, the most reliable source in the project.',
    'Carry':
        'What does holding this currency pay, net of inflation? Reliable '
        'positive expectancy with a fat left tail, which is why real carry '
        'outweighs nominal carry inside the pillar.',
    'External':
        'Can the country fund itself without depreciating? Annual data, one to '
        'two years stale by construction — a solvency filter rather than a '
        'timing signal.',
    'Momentum':
        'Has the repricing started? Deliberately light: momentum is a '
        'confirmation overlay on valuation, not an independent thesis.',
    'Stability':
        'What the position costs in risk. Light because it is partly '
        'double-counted inside carry-to-vol, and because pegs — which would '
        'otherwise sweep it — are masked out of its market-risk gates.',
}

# Explanations for the report’s non-gate columns and UI concepts.
_UI_TIPS = {
    'composite':
        'The weighted average of the five pillar scores, over the pillars that '
        'apply to this currency. The denominator renormalises, so a pillar '
        'dropping out rescales the rest instead of dragging the score to zero.',
    'rating':
        'LONG ≥ 57, LEAN LONG ≥ 39, NEUTRAL ≥ 25, otherwise AVOID — '
        'then capped. Thresholds are provisional: they are set on the '
        'composite’s natural scale, not yet calibrated against forward returns.',
    'gates':
        'Gates passed out of gates that apply. The denominator varies by '
        'currency because structurally inapplicable gates are excluded from '
        'it — a peg is not failed for having no volatility signal.',
    'coverage':
        'The share of applicable gates that actually had data. Below 25% the '
        'composite is an artifact of what happened to be missing, and the '
        'rating is capped at NEUTRAL.',
    'regime':
        'float — freely floating, every gate applies. managed — heavily '
        'leaned against, gates still apply. peg — hard peg or currency board: '
        'valuation and market-risk gates describe the anchor, not this '
        'currency, so they are switched off and the rating is capped.',
    'na':
        'Grey means not applicable, and there are two kinds. Structurally '
        'inapplicable gates leave both the numerator and the denominator, so '
        'the pillar renormalises. Missing data scores zero and stays in the '
        'denominator, so sparse coverage remains penalised.',
    'spark':
        'US dollars per unit of this currency, sampled weekly over ten years. '
        'Rising means the currency strengthened against the dollar.',
    # Displayed columns that are inputs to gates rather than gates themselves.
    'policy_rate':
        'The central bank’s policy rate, from BIS. Shown as a level; what the '
        'model actually scores is the differential against USD, and the same '
        'rate net of inflation.',
    'inflation_pct':
        'Latest annual CPI inflation, from the World Bank. Annual data, so it '
        'lags the monthly policy rate it is subtracted from — a real-rate '
        'reading here mixes vintages.',
    'name':
        'Click any row for the full metric breakdown: every gate with its '
        'value and score, the ten-year price history, and any rating caps.',
}


# Display formatting per gate field. 'pct' = fraction rendered as a percent,
# 'pp' = already in percentage points, 'num' = plain number, 'x' = ratio.
_GATE_DISPLAY = {
    'reer_z20':                ('REER Z-Score', 'z < -0.5', 'num'),
    'reer_dev10':              ('REER vs 10y',  '< -5%',    'pct'),
    'reer_dev5':               ('REER vs 5y',   '< -3%',    'pct'),
    'real_carry':              ('Real Carry',   '> +1pp',   'pp'),
    'nominal_carry':           ('Nom. Carry',   '> 0',      'pp'),
    'real_policy_rate':        ('Real Rate',    '> 0',      'pp'),
    'carry_to_vol':            ('Carry/Vol',    '> 0.15',   'x'),
    'current_account_pct_gdp': ('Curr. Acct',   '> 0',      'pp'),
    'reserves_months_imports': ('Reserves',     '> 3mo',    'num'),
    'inflation_gap':           ('Infl. Gap',    '< 2pp',    'pp'),
    'gdp_growth_pct':          ('GDP Growth',   '> 1.5%',   'pp'),
    'ret_12m_1m':              ('12m-1m',       '> 0',      'pct'),
    'ret_3m':                  ('3m Return',    '> 0',      'pct'),
    'spot_vs_ma200':           ('vs 200d MA',   '> 0',      'pct'),
    'reer_trend_6m':           ('REER Trend',   '> 0',      'pct'),
    'vol_1y':                  ('Volatility',   '< 12%',    'pct'),
    'max_dd_3y':               ('Max DD',       '< 20%',    'pct'),
    'vol_ratio':               ('Vol Regime',   '< 1.2',    'x'),
    'inflation_vol':           ('Infl. Vol',    '< 3pp',    'pp'),
}


def gate_metadata(params=None):
    """Gate and pillar metadata as plain JSON-able data.

    The report renders its Gate Matrix columns straight from this, so adding
    a gate to GATES adds a column to the UI with no template edit.
    """
    p = params or {}
    cat_weights = {cat: p.get('score_weight_' + cat.lower(), w)
                   for cat, w in PILLAR_WEIGHTS.items()}

    gates = []
    for g in GATES:
        label, threshold, fmt = _GATE_DISPLAY.get(
            g.field, (g.name.split(': ')[1], '', 'num'))
        gates.append({
            'name': g.name,
            'category': g.category,
            'field': g.field,
            'label': label,
            'threshold': threshold,
            'fmt': fmt,
            'weight': g.weight,
            'relative': bool(g.relative_mode),
            'tip': _GATE_TIPS.get(g.field, ''),
            'key': _gate_key(g.name),
            'gpKey': _gp_key(g.name),
            'scoreKey': _score_key(g.name),
        })

    categories = [{
        'name': c,
        'weight': cat_weights[c],
        'dark': PILLAR_COLORS[c]['dark'],
        'light': PILLAR_COLORS[c]['light'],
        'scoreKey': '_score_pillar_' + c.lower(),
        'count': sum(1 for g in GATES if g.category == c),
        'tip': _PILLAR_TIPS.get(c, ''),
    } for c in PILLAR_ORDER]

    return {
        'gates': gates,
        'categories': categories,
        'ui': dict(_UI_TIPS),
        'ratings': ['LONG', 'LEAN LONG', 'NEUTRAL', 'AVOID'],
        'thresholds': {
            'long': p.get('rating_threshold_long', RATING_THRESHOLD_LONG),
            'lean': p.get('rating_threshold_lean', RATING_THRESHOLD_LEAN),
            'neutral': p.get('rating_threshold_neutral', RATING_THRESHOLD_NEUTRAL),
        },
    }
