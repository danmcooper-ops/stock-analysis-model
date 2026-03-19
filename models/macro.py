"""Macro-economic regime assessment and parameter adjustment functions.

Scores five market indicators on a [-1, +1] scale (bearish → bullish),
computes a weighted composite, classifies the economic regime, and returns
conservative parameter adjustments for the DCF pipeline.

All functions are pure (no I/O, no side-effects) and fully testable.
"""

# ---------------------------------------------------------------------------
# Indicator scoring functions  [-1.0 bearish … 0 neutral … +1.0 bullish]
# ---------------------------------------------------------------------------

def _score_vix(vix):
    """VIX level → sentiment signal."""
    if vix is None:
        return 0.0
    if vix < 15:
        return 1.0
    if vix < 20:
        return 0.5
    if vix < 25:
        return 0.0
    if vix < 30:
        return -0.5
    return -1.0


def _score_yield_curve(slope):
    """Yield-curve slope (10yr − 3mo, as decimal) → recession signal."""
    if slope is None:
        return 0.0
    if slope > 0.015:
        return 1.0
    if slope > 0.005:
        return 0.5
    if slope > -0.005:
        return 0.0
    if slope > -0.015:
        return -0.5
    return -1.0


def _score_credit_spread(spread_signal):
    """LQD − HYG 3-month return. Positive = HYG under-performs = stress."""
    if spread_signal is None:
        return 0.0
    if spread_signal < -0.02:
        return 1.0
    if spread_signal < -0.005:
        return 0.5
    if spread_signal < 0.005:
        return 0.0
    if spread_signal < 0.02:
        return -0.5
    return -1.0


def _score_spy_momentum(ratio):
    """SPY price / 200-day SMA ratio."""
    if ratio is None:
        return 0.0
    if ratio > 1.05:
        return 1.0
    if ratio > 1.02:
        return 0.5
    if ratio > 0.98:
        return 0.0
    if ratio > 0.95:
        return -0.5
    return -1.0


def _score_industrial_rs(rs):
    """XLI − SPY 3-month relative return.  Positive = cyclical strength."""
    if rs is None:
        return 0.0
    if rs > 0.03:
        return 1.0
    if rs > 0.01:
        return 0.5
    if rs > -0.01:
        return 0.0
    if rs > -0.03:
        return -0.5
    return -1.0


# ---------------------------------------------------------------------------
# Regime assessment
# ---------------------------------------------------------------------------

# Weights for combining indicator scores
MACRO_WEIGHTS = {
    'vix': 0.25,
    'yield_curve': 0.25,
    'credit_spread': 0.20,
    'spy_momentum': 0.15,
    'industrial_rs': 0.15,
}


def assess_macro_regime(indicators):
    """Combine raw indicators into a regime classification.

    Args:
        indicators: dict with keys 'vix', 'yield_curve_slope',
            'credit_spread_3m', 'spy_sma200_ratio', 'xli_rel_strength_3m'.
            Values may be float or None.

    Returns:
        dict with 'regime', 'composite_score', 'indicator_scores',
        'raw_indicators'.
    """
    scores = {
        'vix':            _score_vix(indicators.get('vix')),
        'yield_curve':    _score_yield_curve(indicators.get('yield_curve_slope')),
        'credit_spread':  _score_credit_spread(indicators.get('credit_spread_3m')),
        'spy_momentum':   _score_spy_momentum(indicators.get('spy_sma200_ratio')),
        'industrial_rs':  _score_industrial_rs(indicators.get('xli_rel_strength_3m')),
    }

    composite = sum(scores[k] * MACRO_WEIGHTS[k] for k in MACRO_WEIGHTS)

    if composite > 0.40:
        regime = 'expansion'
    elif composite > 0.10:
        regime = 'mid_cycle'
    elif composite > -0.10:
        regime = 'neutral'
    elif composite > -0.40:
        regime = 'late_cycle'
    else:
        regime = 'contraction'

    return {
        'regime': regime,
        'composite_score': round(composite, 4),
        'indicator_scores': scores,
        'raw_indicators': dict(indicators),
    }


# ---------------------------------------------------------------------------
# Parameter adjustments  (all scale linearly with composite score)
# ---------------------------------------------------------------------------

def compute_macro_adjustments(regime_result):
    """Compute parameter adjustments from macro regime.

    All adjustments are ADDITIVE to base parameters.
    At composite_score == 0, every adjustment is zero (identity).

    ASYMMETRIC design: full adjustments in stress (s < 0), but expansion
    (s > 0) adjustments are heavily dampened to avoid inflating valuations.
    The model's base parameters already assume a normal/healthy economy,
    so expansion should not loosen them further.

    Returns:
        dict with erp_adjustment, terminal_growth_adjustment,
        wacc_sigma_adjustment, growth_sigma_multiplier,
        exit_mult_adjustment, growth_weight_shift.
    """
    s = regime_result['composite_score']

    # Asymmetric scaling: full effect in stress, 20% effect in expansion
    EXPANSION_DAMPING = 0.2
    s_erp = s if s <= 0 else s * EXPANSION_DAMPING
    s_tg = s if s <= 0 else s * EXPANSION_DAMPING
    s_exit = s if s <= 0 else s * EXPANSION_DAMPING
    s_weight = s if s <= 0 else s * EXPANSION_DAMPING

    return {
        # In contraction (s < 0): ERP increases → higher discount rate
        # In expansion: minimal ERP reduction (dampened)
        'erp_adjustment': round(-s_erp * 0.015, 6),
        # In contraction: terminal growth decreases
        # In expansion: minimal terminal growth increase (dampened)
        'terminal_growth_adjustment': round(s_tg * 0.005, 6),
        # One-sided: only widen WACC uncertainty in stress (never narrow)
        'wacc_sigma_adjustment': round(max(0.0, -s * 0.005), 6),
        # One-sided: only widen growth sigma in stress
        'growth_sigma_multiplier': round(1.0 + max(0.0, -s * 0.3), 4),
        # Exit multiples contract in stress; minimal expansion in boom
        'exit_mult_adjustment': round(s_exit * 2.0, 4),
        # In stress: shift weight from analyst LT to fundamental growth
        'growth_weight_shift': round(s_weight * 0.05, 6),
    }


# ---------------------------------------------------------------------------
# Sector headwind / tailwind generator
# ---------------------------------------------------------------------------

# Sectors sensitive to interest-rate direction (rising rates = headwind)
_RATE_SENSITIVE_GROWTH = {'Technology', 'Consumer Cyclical', 'Communication Services'}
_RATE_SENSITIVE_YIELD = {'Utilities', 'Real Estate'}
# Defensive sectors that hold up in downturns
_DEFENSIVE = {'Healthcare', 'Consumer Defensive', 'Utilities'}
# Cyclical sectors that benefit from expansion
_CYCLICAL = {'Energy', 'Basic Materials', 'Industrials', 'Consumer Cyclical'}


def generate_sector_signals(sector_data, macro_regime_result):
    """Generate headwinds and tailwinds per sector from ETF data + macro regime.

    Args:
        sector_data: dict[sector → metrics] from MacroClient.fetch_sector_data()
        macro_regime_result: dict from assess_macro_regime() (may be None)

    Returns:
        dict[sector → {'headwinds': [str], 'tailwinds': [str]}]
    """
    if not sector_data:
        return {}

    regime = macro_regime_result.get('regime', 'neutral') if macro_regime_result else 'neutral'
    raw = macro_regime_result.get('raw_indicators', {}) if macro_regime_result else {}
    vix = raw.get('vix')
    yc_slope = raw.get('yield_curve_slope')

    result = {}
    for sector, metrics in sector_data.items():
        headwinds = []
        tailwinds = []

        # --- Macro-level signals (apply to all sectors) ---
        if vix is not None:
            if vix > 25:
                headwinds.append(f'Elevated market volatility (VIX {vix:.0f})')
            elif vix < 15:
                tailwinds.append(f'Low volatility environment (VIX {vix:.0f})')

        if yc_slope is not None:
            if yc_slope < 0:
                headwinds.append('Inverted yield curve signals recession risk')
            elif yc_slope > 0.015:
                tailwinds.append('Positive yield curve supports economic growth')

        if regime == 'expansion':
            tailwinds.append('Economy in expansion phase')
        elif regime == 'contraction':
            headwinds.append('Economy in contraction phase')
        elif regime == 'late_cycle':
            headwinds.append('Late-cycle environment — rising recession probability')

        # --- Sector ETF momentum signals ---
        rel_3m = metrics.get('rel_strength_3m')
        if rel_3m is not None:
            if rel_3m > 0.03:
                tailwinds.append(f'Sector outperforming S&P 500 by {rel_3m:.1%} over 3 months')
            elif rel_3m < -0.03:
                headwinds.append(f'Sector underperforming S&P 500 by {abs(rel_3m):.1%} over 3 months')

        sma_ratio = metrics.get('sma200_ratio')
        if sma_ratio is not None:
            if sma_ratio > 1.03:
                tailwinds.append('Sector ETF trading above 200-day moving average')
            elif sma_ratio < 0.97:
                headwinds.append('Sector ETF trading below 200-day moving average')

        vol = metrics.get('volatility_30d')
        if vol is not None and vol > 0.25:
            headwinds.append(f'Elevated sector volatility ({vol:.0%} annualized)')

        ret_6m = metrics.get('return_6m')
        if ret_6m is not None:
            if ret_6m > 0.15:
                tailwinds.append(f'Strong 6-month sector return ({ret_6m:.1%})')
            elif ret_6m < -0.10:
                headwinds.append(f'Weak 6-month sector return ({ret_6m:.1%})')

        # --- Sector-sensitivity rules ---
        if sector in _RATE_SENSITIVE_GROWTH and yc_slope is not None:
            if yc_slope > 0.02:
                headwinds.append('Higher rates pressure growth valuations')
            elif yc_slope < 0.005:
                tailwinds.append('Low rate environment supports growth valuations')

        if sector in _RATE_SENSITIVE_YIELD and yc_slope is not None:
            if yc_slope > 0.02:
                headwinds.append('Rising rates reduce relative yield attractiveness')
            elif yc_slope < 0.005:
                tailwinds.append('Low rates make yield sectors more attractive')

        if sector in _DEFENSIVE:
            if regime in ('contraction', 'late_cycle'):
                tailwinds.append('Defensive sector benefits from risk-off rotation')
            elif regime == 'expansion':
                headwinds.append('Expansion favors cyclicals over defensives')

        if sector in _CYCLICAL:
            if regime == 'expansion':
                tailwinds.append('Cyclical sector benefits from economic expansion')
            elif regime in ('contraction', 'late_cycle'):
                headwinds.append('Cyclical sector vulnerable in economic slowdown')

        result[sector] = {'headwinds': headwinds, 'tailwinds': tailwinds}

    return result


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _fmt_pct(v):
    """Format a decimal as percentage string, or 'N/A'."""
    if v is None:
        return 'N/A'
    return f'{v:+.2%}'


def _fmt_ratio(v):
    """Format a ratio, or 'N/A'."""
    if v is None:
        return 'N/A'
    return f'{v:.3f}'


def print_macro_summary(regime_result, adjustments):
    """Print formatted macro regime summary to console."""
    r = regime_result
    a = adjustments
    raw = r['raw_indicators']
    sig = r['indicator_scores']

    print()
    print('=' * 70)
    print('MACRO-ECONOMIC OVERLAY')
    print('=' * 70)
    print(f"  Regime:            {r['regime'].upper().replace('_', ' ')}"
          f"  (composite: {r['composite_score']:+.2f})")
    print()

    vix_val = raw.get('vix')
    vix_str = f'{vix_val:.1f}' if vix_val is not None else 'N/A'
    print(f"  VIX:               {vix_str:>8s}"
          f"   signal: {sig['vix']:+.1f}")
    print(f"  Yield Curve:       {_fmt_pct(raw.get('yield_curve_slope')):>8s}"
          f"   signal: {sig['yield_curve']:+.1f}")
    print(f"  Credit Spread:     {_fmt_pct(raw.get('credit_spread_3m')):>8s}"
          f"   signal: {sig['credit_spread']:+.1f}")
    print(f"  SPY / SMA200:      {_fmt_ratio(raw.get('spy_sma200_ratio')):>8s}"
          f"   signal: {sig['spy_momentum']:+.1f}")
    print(f"  Industrial RS:     {_fmt_pct(raw.get('xli_rel_strength_3m')):>8s}"
          f"   signal: {sig['industrial_rs']:+.1f}")

    print()
    print('  Adjustments:')
    erp_new = 0.055 + a['erp_adjustment']
    print(f"    ERP:             {a['erp_adjustment']:+.4f}"
          f"   (5.50% -> {erp_new:.2%})")
    print(f"    Terminal Growth: {a['terminal_growth_adjustment']:+.4f}")
    wacc_s_new = 0.01 + a['wacc_sigma_adjustment']
    print(f"    WACC Sigma:      {a['wacc_sigma_adjustment']:+.4f}"
          f"   (0.0100 -> {wacc_s_new:.4f})")
    print(f"    Growth Sigma:    x{a['growth_sigma_multiplier']:.2f}")
    em_new = 12.0 + a['exit_mult_adjustment']
    print(f"    Exit Multiple:   {a['exit_mult_adjustment']:+.1f}"
          f"   (12.0 -> {em_new:.1f})")
    print(f"    Analyst LT wt:   shift {a['growth_weight_shift']:+.4f}")
    print('=' * 70)
    print()
