"""REIT-specific metric proxies derived from edgar_history.

Real Estate Investment Trusts report on FFO (Funds From Operations) and
AFFO (Adjusted FFO) per the NAREIT-defined framework, not GAAP net
income. The exact figures live in REIT quarterly supplemental
presentations which aren't standardized in XBRL — there's no
FDIC-equivalent for REITs.

This module computes proxies from the operating-cash-flow and capex
histories already captured in each stock's `edgar_history`:

  FFO Growth   ≈ 5-yr CAGR of operating cash flow
                 (CFO is the closest standardized analog to FFO since
                 both add back non-cash D&A; CFO additionally absorbs
                 working-capital swings, but for a 5-yr CAGR those
                 wash out in most years)

  AFFO Margin  ≈ (CFO − capex) / revenue, latest fully-populated year
                 (the NAREIT AFFO definition subtracts maintenance
                 capex; we subtract total capex which is slightly more
                 conservative — yields a sector-relative ranking that's
                 directionally correct even if the absolute number is
                 a bit understated)

Coverage in today's universe: ~85% of REITs have enough history to
compute both proxies. The remainder show "—" in the banner.

The two metrics genuinely missing — Occupancy % and Same-Store NOI —
require per-REIT quarterly supplemental parsing and aren't covered by
these proxies.
"""


def cfo_cagr(edgar_history, n_years=5):
    """Compound annual growth in operating cash flow over n_years.

    Falls back to actual span if the full window isn't available;
    requires at least 2 years and positive endpoints (CAGR is undefined
    when crossing zero).
    """
    if not edgar_history:
        return None
    h = edgar_history.get("operating_cf_history")
    if not h:
        return None
    ys = sorted(h.keys())
    if len(ys) < 2:
        return None
    latest_y = ys[-1]
    target_y = str(int(latest_y) - n_years)
    if target_y in h:
        start_y = target_y
        span = n_years
    else:
        start_y = ys[0]
        span = int(latest_y) - int(start_y)
    if span < 2:
        return None
    start = h[start_y]
    end = h[latest_y]
    if not start or not end or start <= 0 or end <= 0:
        return None
    return (end / start) ** (1.0 / span) - 1


def affo_margin(edgar_history):
    """Latest-year (CFO − capex) / Revenue.

    Picks the most recent year that has populated operating CF and
    revenue; treats missing capex as 0 (a slightly optimistic floor).
    Returns None if revenue is missing or non-positive.
    """
    if not edgar_history:
        return None
    h_cfo = edgar_history.get("operating_cf_history") or {}
    h_capex = edgar_history.get("capex_history") or {}
    h_rev = edgar_history.get("revenue_history") or {}
    if not h_cfo or not h_rev:
        return None
    for y in sorted(h_cfo.keys(), reverse=True):
        cfo = h_cfo.get(y)
        cap = h_capex.get(y) or 0
        rev = h_rev.get(y)
        if cfo is not None and rev is not None and rev > 0:
            return (cfo - cap) / rev
    return None
