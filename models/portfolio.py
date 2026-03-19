# models/portfolio.py
"""Portfolio construction: position sizing and concentration analysis."""


def position_sizes(stocks_data, max_weight=0.08, min_weight=0.01):
    """Compute allocation weights based on MoS and confidence.

    Higher MoS + higher confidence (lower Monte Carlo CV) = bigger position.

    Parameters
    ----------
    stocks_data : list of dict
        Each dict must have: 'ticker', 'mos' (float or None),
        'rating' (str), 'mc_cv' (float or None).
        Optionally: '_composite_score' (float).
    max_weight : float
        Maximum weight for any single position.
    min_weight : float
        Minimum weight (positions below this are excluded).

    Returns
    -------
    dict
        {ticker: weight} for included stocks. Weights sum to 1.0.
        Empty dict if no valid candidates.
    """
    if not stocks_data:
        return {}

    # Filter to buy-rated stocks with positive margin of safety
    candidates = []
    for s in stocks_data:
        rating = s.get('rating', '')
        mos = s.get('mos')
        if rating not in ('BUY', 'LEAN BUY'):
            continue
        if mos is None or mos <= 0:
            continue
        candidates.append(s)

    if not candidates:
        return {}

    # Compute raw scores
    raw_scores = {}
    for s in candidates:
        ticker = s['ticker']
        mos = s['mos']

        # Confidence factor from Monte Carlo CV (lower CV = higher confidence)
        mc_cv = s.get('mc_cv')
        if mc_cv is not None:
            confidence = (100 - min(mc_cv * 100, 60)) / 100
        else:
            confidence = 0.5  # default mid-confidence

        raw = mos * confidence

        # Scale by composite score if available
        composite = s.get('_composite_score')
        if composite is not None and composite > 0:
            raw *= composite / 100.0

        raw_scores[ticker] = max(raw, 0.001)  # floor to avoid zero

    if not raw_scores:
        return {}

    # Normalize to sum to 1.0
    total = sum(raw_scores.values())
    weights = {t: v / total for t, v in raw_scores.items()}

    # Clamp to [min_weight, max_weight] and re-normalize (iterate until stable)
    for _ in range(10):
        clamped = {t: max(min_weight, min(w, max_weight))
                   for t, w in weights.items()}
        total = sum(clamped.values())
        if total <= 0:
            break
        weights = {t: w / total for t, w in clamped.items()}

    # Remove positions below min_weight and re-normalize
    weights = {t: w for t, w in weights.items() if w >= min_weight}
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    return weights


def concentration_analysis(stocks_data):
    """Flag sector/factor concentration in top picks.

    Parameters
    ----------
    stocks_data : list of dict
        Each dict must have: 'ticker', 'sector'.
        Optionally: 'position_weight' for weighted analysis.

    Returns
    -------
    dict
        {
            'sector_weights': {sector: float},
            'top_sector': str or None,
            'top_sector_weight': float,
            'concentration_flag': bool (True if any sector > 40% or HHI > 0.25),
            'hhi': float (Herfindahl-Hirschman Index of sector weights),
            'n_sectors': int,
        }
    """
    if not stocks_data:
        return {
            'sector_weights': {},
            'top_sector': None,
            'top_sector_weight': 0.0,
            'concentration_flag': False,
            'hhi': 0.0,
            'n_sectors': 0,
        }

    # Use position weights if available, otherwise equal weight
    sector_totals = {}
    total_weight = 0.0
    for s in stocks_data:
        sector = s.get('sector') or 'Unknown'
        weight = s.get('position_weight') or (1.0 / len(stocks_data))
        sector_totals[sector] = sector_totals.get(sector, 0) + weight
        total_weight += weight

    # Normalize
    if total_weight > 0:
        sector_weights = {s: w / total_weight for s, w in sector_totals.items()}
    else:
        sector_weights = sector_totals

    # Find top sector
    top_sector = max(sector_weights, key=sector_weights.get) if sector_weights else None
    top_weight = sector_weights.get(top_sector, 0) if top_sector else 0

    # HHI (Herfindahl-Hirschman Index)
    hhi = sum(w ** 2 for w in sector_weights.values())

    concentration_flag = top_weight > 0.40 or hhi > 0.25

    return {
        'sector_weights': sector_weights,
        'top_sector': top_sector,
        'top_sector_weight': top_weight,
        'concentration_flag': concentration_flag,
        'hhi': hhi,
        'n_sectors': len(sector_weights),
    }
