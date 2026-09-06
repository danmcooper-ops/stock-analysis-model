# scripts/scoring.py
"""Scoring, screening, and validation functions for the stock analysis pipeline."""

import statistics
from collections import namedtuple

from scripts.config import (SCORE_WEIGHT_VALUATION, SCORE_WEIGHT_QUALITY,
                             SCORE_WEIGHT_MOAT, SCORE_WEIGHT_GROWTH,
                             SCORE_WEIGHT_OWNERSHIP, MIN_SECTOR_STOCKS,
                             MIN_ADV_FOR_BUY,
                            MC_CLIP_RATE_DOWNGRADE, MC_INVALID_RATE_DOWNGRADE)

# Unified gate spec — ONE entry per metric drives both the pass/fail Gate
# Matrix cell and the continuous 0-100 score, so a gate's threshold and its
# scoring range live side by side and cannot drift apart.
#   test_fn:  (value, row) -> bool | None (None = missing data, renders N/A)
#   score_fn: (value, row, percentile_or_None) -> 0-100
#   relative_mode: False=absolute, 'global'/'sector'=percentile-ranked score
#   higher_better: direction for percentile ranking (relative gates only)
#   weight: scales the gate's contribution within its category (default 1.0)
#   applicable: optional row-predicate; when it returns False the gate is
#     STRUCTURALLY INAPPLICABLE for that ticker (a bank has no meaningful FCF
#     yield; negative tangible book makes P/TBV undefined) and is excluded
#     from numerator AND denominator — unlike missing data, which still
#     scores 0 (worst). `applicable=None` means always applicable.
# The category is the gate name's prefix ("Valuation: MoS" → "Valuation").
class Gate(namedtuple('Gate',
                      ['name', 'field', 'test_fn', 'score_fn',
                       'relative_mode', 'higher_better', 'weight',
                       'applicable'],
                      defaults=(False, True, 1.0, None))):
    __slots__ = ()

    @property
    def category(self):
        return self.name.split(': ')[0]


def _gate_applicable(gate, row):
    """True unless the gate declares an applicability predicate that fails."""
    return gate.applicable is None or gate.applicable(row)


# ---------------------------------------------------------------------------
# Applicability predicates (shared across GATES)
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
    a capital-light compounder returning cash must not score 0. Masked for
    Financial Services alongside the other ROIC gates (see Moat: Spread)."""
    return _appl_non_financial(r) and not r.get('_incr_roic_undefined')


def _appl_mult_history(r):
    """Multiple-vs-own-history needs a real baseline: >=5 positive-EBIT years
    with matchable year-end prices and share counts."""
    return (r.get('mult_hist_years') or 0) >= 5


def _appl_pool_share(r):
    """Pool-share trajectory is undefined (not bad) when structurally
    uncomputable: <3y of operating-income history, fewer than
    MIN_SECTOR_STOCKS consistent-panel peers, missing sector, or a
    non-positive endpoint share. NOTE: deviates from the house
    "sparse data scores 0" rule — missing history is N/A here because
    old snapshots lack operating_income_history entirely (added 2026-07)."""
    return not r.get('_pool_share_undefined')

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


def _mc_confidence_label(cv, clip_rate=None, invalid_rate=None):
    """Convert coefficient of variation to a confidence label with CV%.

    When the simulation's constraint diagnostics are supplied, the label is
    downgraded one notch (HIGH → MEDIUM → LOW) and tagged "constrained" if
    the most binding wall forced more than MC_CLIP_RATE_DOWNGRADE of the
    draws or more than MC_INVALID_RATE_DOWNGRADE of them wiped out equity.
    A tight CV in that state says more about the walls than the inputs.
    """
    if cv is None:
        return None
    pct = round(cv * 100)
    if cv < 0.20:
        level = 'HIGH'
    elif cv < 0.40:
        level = 'MEDIUM'
    else:
        level = 'LOW'
    constrained = ((clip_rate is not None and clip_rate > MC_CLIP_RATE_DOWNGRADE) or
                   (invalid_rate is not None and invalid_rate > MC_INVALID_RATE_DOWNGRADE))
    if constrained:
        level = {'HIGH': 'MEDIUM', 'MEDIUM': 'LOW', 'LOW': 'LOW'}[level]
        return f'{level} ({pct}%, constrained)'
    return f'{level} ({pct}%)'


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


# Unified gate definitions — ONE list drives both the pass/fail Gate Matrix
# (test_fn → _gate_*/_gp_* fields, the "gates passed" diagnostic) and the
# continuous composite scoring (score_fn → _score_* fields, category
# averages, composite, rating). Tune a gate in one place.
#
# Order matters — the Gate Matrix renders columns within each category in the
# order they appear here. Grouping is intentional:
#   Valuation:  DCF-derived → multiples → alternative models
#   Quality:    leverage → earnings quality → returns → composite
#   Moat:       ROIC family (spread, consistency, trend) → margin family
#   Growth:     top-line → bottom-line → margin trend → composite
#   Ownership:  share-count direction (yield, shrink) → dilution → alignment
GATES = [
    # ---- Valuation ----
    # MoS is the headline price-vs-value signal; double weight within
    # Valuation so it isn't diluted to 1/6 of a 0.30-weight category.
    Gate('Valuation: MoS', 'mos',
         lambda v, r: v > 0.10 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.10, 0.40),  # tightened: worst raised -20%→-10%
         weight=2.0),
    # MAD/median statistic (see prepare_scoring_fields); pass at 0.15
    # preserves the ~29% pass rate the old (max−min)/median @0.50 produced.
    # Provisional. Lower is better; score range rescaled for the MAD/median
    # statistic.
    Gate('Valuation: FV Dispersion', 'fv_dispersion',
         lambda v, r: v <= 0.15 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.70, 0.0)),
    # EBIT/EV earnings yield: capital-structure-neutral market multiple
    # (replaced P/FCF, which was the exact reciprocal of FCF Yield — the
    # same signal counted twice). Absolute pass threshold ≈ ≤12.5x EV/EBIT;
    # score range absolute, not rf-relative: FCF Yield already carries the
    # rate-regime beta for the category. 6% ≈ market-average EV/EBIT.
    Gate('Valuation: EBIT/EV', 'ebit_ev',
         lambda v, r: v > 0.08 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.0, 0.12),
         applicable=_appl_non_financial),
    Gate('Valuation: P/TBV', 'p_tbv',
         lambda v, r: 0 < v <= 2.5 if v is not None else None,
         lambda v, r, pct: (_score_linear(v, 5.0, 1.0)
                            if v is not None and v > 0 else None),
         applicable=_appl_positive_tbv),
    # Time-series cheapness: current mcap/EBIT vs the firm's own ~10y median
    # (negative = below own history). Replaced EPV Floor, a third price-vs-
    # intrinsic ratio that correlated 0.68 with MoS; this is the one valuation
    # axis the cross-sectional and model-based gates don't cover.
    # Score range from the 2026-07-02 snapshot distribution: p25 ≈ −0.28 → 85,
    # median ≈ +0.13 → 58, p75 ≈ +0.70 → 20; long expensive tail clamps to 0.
    Gate('Valuation: Mult vs Hist', 'mult_vs_hist',
         lambda v, r: v < -0.10 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 1.0, -0.50),
         applicable=_appl_mult_history),
    # FCF yield vs the risk-free rate: an absolute, model-independent value
    # floor (replaced RIM MoS, a third price-vs-intrinsic-value gate that
    # triangulated the same signal as MoS).
    Gate('Valuation: FCF Yield', 'fcf_yield',
         lambda v, r: v > (r.get('_risk_free_rate') or 0.045) if v is not None else None,
         lambda v, r, pct: (_score_linear(v - (r.get('_risk_free_rate') or 0.045),
                                          -0.03, 0.08)
                            if v is not None else None),
         applicable=_appl_non_financial),

    # ---- Quality ----
    # Int Coverage and Net Debt/EBITDA are masked for Financial Services:
    # interest is a bank's cost of goods (not a fixed charge to cover) and
    # deposits/funding aren't corporate debt, so both metrics were actively
    # wrong there — nd_ebitda resolved for ~87% of FS rows and scored
    # garbage. FS leverage belongs to the FDIC KPIs (CET1/NPL), which are
    # collected but not yet scored.
    Gate('Quality: Int Coverage', 'int_cov',
         lambda v, r: v > 3.0 if v is not None else None,
         lambda v, r, pct: _score_linear(
             min(v, 40) if v is not None else None, 1.0, 20.0),
         applicable=_appl_non_financial),
    Gate('Quality: Net Debt/EBITDA', 'nd_ebitda',
         lambda v, r: v <= 1.5 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 4.0, -0.5),  # tightened: worst 5→4, best 0→-0.5 (net cash rewarded)
         applicable=_appl_non_financial),
    # Accruals = (NI - CFO) / Assets. Sloan (1996): high *positive* accruals
    # predict negative future returns — earnings exceed cash, often via
    # aggressive recognition. Negative accruals (CFO > NI) indicate
    # conservative accounting and strong cash generation; do not penalize.
    # Scored as a sector percentile (lower accruals → higher score).
    Gate('Quality: Accruals', 'accruals',
         lambda v, r: v < 0.08 if v is not None else None,
         lambda v, r, pct: pct,
         relative_mode='sector', higher_better=False),
    # (Cash Conv removed: CFO-vs-NI is the accruals signal inverted, and
    # Piotroski's internals check it again — three gates, one signal.)
    # Revenue-growth volatility (lower = steadier: a steady top line scores
    # high). Replaced the ROE gate, which double-counted returns-on-capital
    # with the Moat/ROIC block and is distortable by leverage and buybacks.
    Gate('Quality: Rev Volatility', 'rev_growth_vol',
         lambda v, r: v < 0.12 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.40, 0.05)),
    # Over-earning guard: current operating margin vs the company's own
    # ~10y average. All four FV models eat the same point-in-time earnings,
    # so at a cyclical margin peak FV Dispersion is tight and MoS inflated
    # exactly when they shouldn't be — this is the orthogonal check.
    # Score is one-sided: 8pp+ above own history scores 0; at/below history
    # clamps to 100. Margin DETERIORATION is already penalized by
    # Growth: Margins.
    Gate('Quality: Margin vs Hist', 'margin_vs_hist',
         lambda v, r: v < 0.05 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.08, 0.0),
         applicable=_appl_margin_history),
    Gate('Quality: Piotroski', 'piotroski',
         lambda v, r: v >= 7 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0, 9)),

    # ---- Moat ----
    # The ROIC family (Spread, Consistency, Incr ROIC) is masked for
    # Financial Services: NOPAT / (equity + debt - cash) is meaningless for
    # capital intermediaries (deposits are not "debt", cash is inventory),
    # which is exactly why the Phase-1 screen already bypasses the spread
    # filter for the sector. Scoring the same number here contradicted that.
    # Bank quality is carried by the FDIC metrics (NIM / efficiency / CET1).
    Gate('Moat: Spread', 'spread',
         lambda v, r: v > 0.07 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.0, 0.20),  # tightened: best 25%→20%
         applicable=_appl_non_financial),
    Gate('Moat: ROIC Consistency', 'roic_cv',
         lambda v, r: v < 0.30 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.60, 0.0),
         applicable=_appl_non_financial),
    # Incremental ROIC (ΔNOPAT/ΔIC over the statement window): the moat-
    # TRAJECTORY test — is each new dollar of capital still earning above
    # the cost of capital? The better replacement for the retired ROIC
    # Trend gate.
    Gate('Moat: Incr ROIC', 'incremental_roic',
         lambda v, r: v > 0.10 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.05, 0.25),
         applicable=_appl_incr_roic),
    # Operating margin vs the sector median: a structural competitive-position
    # signal orthogonal to ROIC. Also carries the margin-vs-sector signal of
    # the retired Gross Margin percentile gate (same axis, operating level).
    # Score: sector-median → 33, +20pp → 100.
    # Masked for Financial Services on the same grounds as Int Coverage and
    # the ROIC family: a bank's operating income is derived as pretax +
    # interest expense, and interest is its cost of goods, so the "margin"
    # is not one. In the 2026-09-03 snapshot the 80 banks that had a revenue
    # read a median operating margin of 174% (max 15.7x) and 78 of them
    # passed this gate; the other 144 read N/A. Bank quality is carried by
    # the FDIC metrics (NIM / efficiency / CET1).
    Gate('Moat: Margin Advantage', 'margin_advantage',
         lambda v, r: v > 0.05 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.10, 0.20),
         applicable=_appl_non_financial),
    # Pool-share trajectory: 5-yr CAGR of the company's share of its
    # sector's operating-profit pool (consistent panel) — is it WINNING
    # share of sector profit over time? Share-of-pool complement to
    # Margin Advantage's level-vs-sector signal.
    # (FCF Margin retired: one signal counted twice — margin LEVEL is
    # Margin Advantage's axis, and FCF already votes through FCF Yield,
    # FCF Durability, and Accruals.)
    Gate('Moat: Pool Share', 'pool_share_cagr',
         lambda v, r: v > 0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.10, 0.15),
         applicable=_appl_pool_share),

    # ---- Growth ----
    Gate('Growth: Rev Durability', 'rev_cagr_10y',
         lambda v, r: v > 0.02 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.05, 0.15)),
    Gate('Growth: FCF Durability', 'fcf_cagr_5y',
         lambda v, r: v > 0.05 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.05, 0.15),
         applicable=_appl_non_financial),
    # Masked for Financial Services on the same grounds as Int Coverage and
    # FCF Durability: banks and insurers report no cost of revenue, so gross
    # margin — and therefore its trend — does not exist for them. 94% of FS
    # rows resolved to None here, and because a missing-data N/A stays in the
    # applicable-gate denominator, every one was carrying a guaranteed failed
    # gate on a metric its filings structurally cannot produce (TRV read
    # 8/19 with this gate unpassable). The 6% that did resolve were fintech
    # and exchange names tagging a GrossProfit line — too thin a base to
    # score the sector on, and inconsistent with masking the other two.
    Gate('Growth: Margins', 'gross_margin_trend',
         lambda v, r: v >= 0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.05, 0.05),
         applicable=_appl_non_financial),
    Gate('Growth: Fund Growth', 'fundamental_growth',
         lambda v, r: v > 0.03 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.0, 0.10)),

    # ---- Ownership ----
    Gate('Ownership: Shrhldr Yield', 'shareholder_yield',
         lambda v, r: v > 0.02 if v is not None else None,
         lambda v, r, pct: _score_linear(v, -0.01, 0.08)),
    # (Buyback Rate removed: already inside Shareholder Yield, and Share
    # Shrink is its 5y integral — three gates on share-count direction.)
    Gate('Ownership: Share Shrink', 'shares_cagr_5y',
         lambda v, r: v < 0 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.04, -0.04)),
    Gate('Ownership: SBC Dilution', 'sbc_pct_rev',
         lambda v, r: v <= 0.02 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.06, 0.0)),  # tightened: worst 10%→6%
    Gate('Ownership: Insider Own', 'insider_pct',
         lambda v, r: v >= 0.05 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.0, 0.15)),
    # Open-market insider net buying (Form 4 P vs S transactions): the
    # classic conviction signal. Neutral 0.5 buy-ratio scores 50.
    # Inapplicable (not zero) below 4 open-market transactions/365d.
    Gate('Ownership: Insider Buying', 'insider_buy_ratio',
         lambda v, r: v >= 0.5 if v is not None else None,
         lambda v, r, pct: _score_linear(v, 0.0, 1.0),
         applicable=_appl_insider_activity),
]


def _compute_pool_share_trajectory(results):
    """Cross-sectional pre-pass: 5-yr CAGR of each ticker's share of its
    sector's operating-profit pool (Moat: Pool Share gate).

    CONSISTENT PANEL: the two endpoint-year pools are summed over only the
    tickers with operating income in BOTH endpoint years, so a company's
    share change is measured against the same peer set in both years —
    composition drift (IPOs entering, thin histories dropping out) doesn't
    masquerade as share gain or loss. The pool is this run's analyzed
    universe, not the true market: the panel fixes within-window drift,
    not cross-run universe drift.

    Window: prefer exactly (latest−5, latest); else fall back to the oldest
    available year (actual span, annualized), minimum 3 years — modeled on
    cfo_cagr in scripts/enrich_reit.py with a 3y floor. Negative operating
    income clamps to 0 in numerator and pool, matching the single-year
    pp_profit_share pass in analyze_stock.py.
    """
    sector_rows = {}                      # sector -> [(row, {int_year: oi})]
    for r in results:
        # Defaults-first so stale values from a prior rescore are always
        # overwritten (these persist in snapshot JSONs like
        # _incr_roic_undefined — not scrubbed by _purge_stale_gate_fields).
        r['pool_share_cagr'] = None
        r['_pool_share_undefined'] = True
        oi = ((r.get('edgar_history') or {})
              .get('operating_income_history')) or {}
        h = {}
        for k, v in oi.items():           # int keys live, str after JSON round-trip
            if v is None:
                continue
            try:
                h[int(str(k)[:4])] = v
            except (TypeError, ValueError):
                continue
        s = r.get('sector')
        if s and h:
            sector_rows.setdefault(s, []).append((r, h))

    for _sector, rows in sector_rows.items():
        pools = {}                        # (y0, y1) -> (pool0, pool1, n_panel)
        for r, h in rows:
            ys = sorted(h)
            if len(ys) < 2:
                continue
            y1 = ys[-1]
            y0 = y1 - 5 if (y1 - 5) in h else ys[0]
            span = y1 - y0
            if span < 3:
                continue
            key = (y0, y1)
            if key not in pools:          # one sector pass per endpoint pair
                p0 = p1 = 0.0
                n = 0
                for _, h2 in rows:
                    if y0 in h2 and y1 in h2:
                        p0 += max(h2[y0], 0.0)
                        p1 += max(h2[y1], 0.0)
                        n += 1
                pools[key] = (p0, p1, n)
            p0, p1, n = pools[key]
            if n < MIN_SECTOR_STOCKS or p0 <= 0 or p1 <= 0:
                continue
            sh0 = max(h[y0], 0.0) / p0
            sh1 = max(h[y1], 0.0) / p1
            if sh0 <= 0 or sh1 <= 0:
                continue                  # CAGR undefined crossing zero
            # Clamp for sane display (cf. incremental_roic): a near-zero
            # starting share can produce absurd CAGRs; the score range is
            # far narrower.
            r['pool_share_cagr'] = max(-1.0, min(1.0,
                                       (sh1 / sh0) ** (1.0 / span) - 1))
            r['_pool_share_undefined'] = False


# --- Value-trap overlay -----------------------------------------------------
# A parallel risk diagnostic, deliberately NOT a 27th gate: its inputs reuse
# fields the composite already scores (a gate would double-count them and
# force a re-quantile of the 57/39/25 thresholds), trap-ness only matters
# within the cheap cohort, and — decisively — it follows the rating-cap
# fail-open convention (missing data skips, never scores as bad), which is
# the opposite of the gates' missing-scores-0 convention. Display-only until
# the forward-return validation in scripts/backtest.py clears it for a cap.
#
# Momentum note: momentum stays UNSCORED as attractiveness everywhere in this
# model (a momentum gate would invert the deliberately-negative composite-vs-
# trailing-return check). The F axis below uses deep 12-1 drawdown only as a
# minority-weight corroborator of an already-fundamental trap profile — and
# the validation runbook includes a with/without-F ablation so its marginal
# contribution is measured, not assumed.

# Calibrated 2026-08-16 against 30d forward excess returns on the 74-snapshot
# corpus (2026-04-20→08-15), within the cheap cohort (top-quartile MoS):
# T=65 → flagged−unflagged = −3.3pp, Cohen's d −0.25, direction 93% of 42
# snapshots; survives exclusion of beneish/altman-capped rows (−3.5pp,
# d −0.32, 100% of 23) and the ex-momentum ablation. Prevalence ~2% of the
# cheap cohort — the signal is a TAIL FLAG, not a ranking: quartile mean
# spreads are ~0 (Q4 mixes mostly-benign scores; medians/hit-rates do fall
# monotonically). 90d re-check pending corpus maturation before any cap.
TRAP_FLAG_THRESHOLD = 65
_TRAP_MIN_AXES = 4        # coverage floor: >=4 of 6 axes resolved...
_TRAP_MIN_WEIGHT = 0.60   # ...covering >=60% of total axis weight


def _t01(x, x0, x1):
    """Piecewise-linear ramp: 0 at x0, 1 at x1, clamped (x1 > x0)."""
    if x is None:
        return None
    return max(0.0, min(1.0, (x - x0) / (x1 - x0)))


def _trap_axes(r):
    """Per-axis sub-scores for one row. None sub-inputs are skipped; an axis
    with zero resolvable sub-inputs is omitted entirely (fail-open)."""
    def num(v):
        return v if isinstance(v, (int, float)) else None

    axes = {}

    # A — Structural decline: the business is shrinking, not just cheap.
    a = []
    rdy = num(r.get('rev_down_years'))
    if rdy is not None:
        a.append(_t01(rdy, 1.0, 3.0))
    rc5 = num(r.get('rev_cagr_5y'))
    if rc5 is not None:
        a.append(_t01(-rc5, 0.0, 0.05))
    rc10 = num(r.get('rev_cagr_10y'))
    if rc10 is not None:
        a.append(1.0 if rc10 < 0 else 0.0)   # decade-scale decline is binary
    gmt = num(r.get('gross_margin_trend'))
    if gmt is not None:
        a.append(_t01(-gmt, 0.0, 0.01))      # 1pp/yr erosion saturates
    fneg = num(r.get('fcf_neg_years_5y'))
    if fneg is not None:
        a.append(_t01(fneg, 1.0, 3.0))
    if a:
        axes['decline'] = (sum(a) / len(a), 0.25)

    # B — Balance-sheet pressure: leverage that forecloses the turnaround.
    b = []
    nde = num(r.get('nd_ebitda'))
    if nde is not None:
        b.append(_t01(nde, 2.0, 5.0))
    ic = num(r.get('int_cov'))
    if ic is not None:
        b.append(1.0 - _t01(ic, 2.0, 6.0))
    zone = r.get('altman_z_zone')
    if zone in ('distress', 'grey', 'safe'):
        b.append({'distress': 1.0, 'grey': 0.5, 'safe': 0.0}[zone])
    nds = num(r.get('net_debt_slope_3y'))
    if nds is not None and nde is not None and nde >= 2.0:
        # Rising debt is trap fuel only when leverage is already elevated —
        # a net-cash company drawing a revolver once is not a trap signal.
        b.append(_t01(nds, 0.0, 0.05))
    if b:
        axes['balance_sheet'] = (sum(b) / len(b), 0.20)

    # C — Value destruction: capital compounding below its cost.
    c = []
    spread = num(r.get('spread'))
    if spread is None:
        roic, wacc = num(r.get('roic')), num(r.get('wacc'))
        if roic is not None and wacc is not None:
            spread = roic - wacc
    if spread is not None:
        c.append(1.0 - _t01(spread, -0.05, 0.02))
    rts = num(r.get('roic_trend_slope'))
    if rts is not None:
        c.append(_t01(-rts, 0.0, 0.10))
    if not r.get('_incr_roic_undefined'):
        iroic, wacc = num(r.get('incremental_roic')), num(r.get('wacc'))
        if iroic is not None and wacc is not None:
            c.append(1.0 - _t01(iroic - wacc, -0.05, 0.05))
    psc = num(r.get('pool_share_cagr'))
    if psc is not None:
        c.append(_t01(-psc, 0.0, 0.10))
    if c:
        axes['value_destruction'] = (sum(c) / len(c), 0.20)

    # D — Structural derating: the market has durably marked the multiple
    # down, and the FV models can't agree the MoS is real (the scoring
    # docstring's own "a large MoS may be a value trap" case).
    d = []
    mvh = num(r.get('mult_vs_hist'))
    if mvh is not None:
        d.append(_t01(-mvh, 0.20, 0.50))
    fvd = num(r.get('fv_dispersion'))
    if fvd is not None:
        d.append(_t01(fvd, 0.15, 0.50))
    if d:
        axes['derating'] = (sum(d) / len(d), 0.15)

    # E — Payout risk: the yield that lures value buyers is uncovered.
    dfr = num(r.get('div_fcf_ratio_3y'))
    if dfr is not None:
        axes['payout'] = (_t01(dfr, 0.8, 1.5), 0.10)

    # F — Market corroboration (minority weight; see module note).
    f = []
    m12 = num(r.get('momentum_12_1'))
    if m12 is not None:
        f.append(_t01(-m12, 0.20, 0.50))
    spf = num(r.get('short_pct_float'))
    if spf is not None:
        f.append(_t01(spf, 0.10, 0.25))
    if f:
        axes['market'] = (sum(f) / len(f), 0.10)

    return axes


_TRAP_REASON = {
    'decline':           'Structural revenue/margin decline',
    'balance_sheet':     'Leveraged balance sheet under pressure',
    'value_destruction': 'ROIC below cost of capital',
    'derating':          'Durably derated by the market',
    'payout':            'Dividend not covered by free cash flow',
    'market':            'Heavy short interest / falling knife',
}


def compute_trap_signals(results):
    """Attach trap_score / trap_flag / trap_reasons / _trap_components.

    trap_score: 0-100, higher = more value-trap-like. None (and trap_flag
    None, never True) unless >=_TRAP_MIN_AXES axes resolve covering
    >=_TRAP_MIN_WEIGHT of axis weight — thin data is unknown, not safe and
    not dangerous. Field names carry no _gate_/_gp_/_score_ prefix, so
    _purge_stale_gate_fields leaves them alone on snapshot round-trips.
    """
    for r in results:
        axes = _trap_axes(r)
        wsum = sum(w for _, w in axes.values())
        if len(axes) < _TRAP_MIN_AXES or wsum < _TRAP_MIN_WEIGHT:
            r['trap_score'] = None
            r['trap_flag'] = None
            r['trap_reasons'] = []
            r['_trap_components'] = {k: {'score': round(s, 3), 'weight': w}
                                     for k, (s, w) in axes.items()}
            continue
        score = 100.0 * sum(s * w for s, w in axes.values()) / wsum
        r['trap_score'] = round(score, 1)
        r['trap_flag'] = score >= TRAP_FLAG_THRESHOLD
        r['trap_reasons'] = [
            _TRAP_REASON[k] for k, (s, w) in
            sorted(axes.items(), key=lambda kv: -kv[1][0] * kv[1][1])
            if s >= 0.5]
        r['_trap_components'] = {k: {'score': round(s, 3), 'weight': w}
                                 for k, (s, w) in axes.items()}


def prepare_scoring_fields(results):
    """Populate derived fields shared by gates and continuous scoring."""
    _compute_pool_share_trajectory(results)
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
        # so OCF − capex is not a valid FCF proxy — the same reason the FCF
        # Yield/Durability gates are legitimately N/A for financials (they're
        # scored on FDIC KPIs / combined ratios instead).
        if fcf is None and r.get('sector') != 'Financial Services':
            fcf_edgar = r.get('fcf_edgar')
            if fcf_edgar is not None:
                fcf = fcf_edgar
                r['fcf'] = fcf_edgar
                r['_fcf_source'] = 'edgar'
        # Same fallback shape for Interest Coverage: yfinance surfaced no
        # income statement for ~37% of non-financial rows (AAPL and MSFT
        # included), so Quality: Int Coverage read N/A — and a missing-data
        # N/A stays in the applicable-gate denominator, scoring those rows as
        # a FAILED leverage test on absent data. EBIT/interest from EDGAR is
        # the same ratio off the filing itself; where both sources resolve
        # they agree (HON: 6.05 either way). Financial Services stays excluded
        # — interest is a bank's cost of goods, not a fixed charge to cover,
        # which is why the gate masks the sector outright.
        if r.get('int_cov') is None and r.get('sector') != 'Financial Services':
            int_cov_edgar = r.get('int_cov_edgar')
            if int_cov_edgar is not None:
                r['int_cov'] = int_cov_edgar
                r['_int_cov_source'] = 'edgar'
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
        # SBC/Revenue: prefer the XBRL-derived value (enrich_xbrl; ~1,100
        # rows) over the yfinance cash-flow one (~240 rows) — under the
        # missing-data-scores-0 rule the sparse yfinance field left most of
        # the universe scoring 0 on data the pipeline already had. On live
        # runs the XBRL field lands after Phase 2, so the swap takes effect
        # at the rescore_and_render step (1f), same as other enrichments.
        sbc_xbrl = r.get('sbc_pct_rev_xbrl')
        if sbc_xbrl is not None:
            r['sbc_pct_rev'] = sbc_xbrl
        else:
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

    # Trap overlay LAST: it consumes fields derived in the loop above
    # (fv_dispersion, incremental_roic, margin_vs_hist, roic_trend_slope,
    # the int_cov EDGAR swap), so it cannot ride the pool-share pre-pass at
    # the top of this function. Running inside prepare_scoring_fields keeps
    # live, rescore, replay and calibrate identical via score_and_rate.
    compute_trap_signals(results)


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
        passed = 0
        applicable_total = 0
        inapplicable = 0
        for gate in GATES:
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
        print("\n--- Separation Test ---")
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
    for gate in GATES:
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

            for _sector, group in sector_groups.items():
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
    for gate in GATES:
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
        for gate in GATES:
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
    Re-confirmed after the 2026-07 Pool Share swap (FCF Margin retired):
    rescoring the 2026-07-03 universe (n=2210) moved the distribution
    only 17/328/600/1265 → 13/340/597/1260 (BUY −0.18pp, LEAN +0.5pp) —
    inside tolerance, thresholds unchanged.
    Re-confirmed again after the data-integrity pass (SBC/Rev now prefers
    the XBRL field, coverage 243→1375 rows; Int Coverage + Net Debt/EBITDA
    masked for Financial Services): 13/340/597/1260 → 20/361/589/1240
    (BUY +0.32pp, LEAN +1.0pp) — inside tolerance, thresholds unchanged.
    PROVISIONAL: old snapshots carry no per-year NOPAT/invested-capital,
    so Incr ROIC scores 0 on rescores; the first live run will lift
    composites slightly — revisit then. `backtest.py calibrate
    --include-thresholds` can search these thresholds against forward
    returns once the corpus holds enough independent periods (it refuses
    below MIN_EFFECTIVE_N; see `backtest.py readiness`).

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
    'pool_share': {'label': 'Pool Share Δ', 'threshold': '5y profit-pool share CAGR > 0', 'fmt': 'pct1'},
    'fcf_durability': {'label': '5Y FCF CAGR', 'threshold': '5Y FCF CAGR > 5%', 'fmt': 'pct1'},
    'share_shrink': {'label': 'Share Shrink', 'threshold': '5Y Shares CAGR < 0', 'fmt': 'pct1'},
    'piotroski': {'label': 'Piotroski', 'threshold': 'F-Score >= 7', 'fmt': 'int'},
    'margin_advantage': {'label': 'Margin Adv', 'threshold': 'OpM vs sector > 5pp', 'fmt': 'pct1'},
    'mult_vs_hist': {'label': 'Mult vs Hist', 'threshold': '>=10% below own 10y median', 'fmt': 'pct1'},
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
    gates = []
    for gate in GATES:
        short = _gate_short(gate.name)
        display = _GATE_DISPLAY.get(short, {})
        gates.append({
            'key': _gate_key(gate.name),
            'label': display.get('label', gate.name.split(': ')[1]),
            'gpKey': _gp_key(gate.name),
            'scoreKey': _score_key(gate.name),
            'threshold': display.get('threshold', ''),
            'category': gate.category,
            'fmt': display.get('fmt', 'ratio'),
            'weight': gate.weight,
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

    # Tradeability. A BUY that can't be filled or exited at a sane price isn't
    # a recommendation. Deliberately fails OPEN: only caps when the metric is
    # present, so the ~2% of price files without a Volume column (and any name
    # whose prices are too stale to measure) are never demoted on absent data.
    adv = row.get('avg_dollar_volume_3m')
    if adv is not None and adv < MIN_ADV_FOR_BUY:
        add('HOLD', f'insufficient liquidity (${adv / 1e6:.2f}M avg daily volume)')

    return cap, reasons


def apply_rating_caps(results, params=None):
    """Apply critical-only rating caps and expose raw/final rating diagnostics.

    rating_raw is the uncapped composite-derived rating; rating is the final
    (possibly capped) one. rating != rating_raw ⇔ a cap fired — the reports
    key their ⚠ badge off that comparison plus _rating_cap_reasons.
    """
    for r in results:
        raw = rating_from_composite(r.get('_composite_score'), params)
        r['rating_raw'] = raw
        cap, reasons = _rating_cap_for_row(r, params=params)
        r['_rating_cap'] = cap
        r['_rating_cap_reasons'] = reasons
        r['rating'] = _cap_rating(raw, cap) if cap and raw else raw
        # Retired duplicate of rating_raw; scrub from round-tripped snapshots.
        r.pop('_rating_from_score', None)


def _purge_stale_gate_fields(results):
    """Drop _gate_*/_gp_*/_score_* keys for gates that no longer exist.

    Snapshot rows round-trip through re-scoring: without this, fields from
    retired gates persist forever and leak into the report payloads.
    """
    valid = set()
    for g in GATES:
        valid.add(_gate_key(g.name))
        valid.add(_gp_key(g.name))
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
