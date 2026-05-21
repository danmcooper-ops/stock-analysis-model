"""Net Asset Value valuation — asset-floor sanity check.

Tangible Book Value (TBV) per share strips goodwill and other intangibles
out of shareholders' equity, yielding a more conservative book floor than
plain P/B. Useful for financials, REITs, distressed names, and holding
companies whose cash-flow models give garbage results.

This is a universal computation. Sector-specific NAV recipes (REIT
property cap-rate uplift, financials mark-to-market investment portfolio,
holding-company NAV from listed investee market values) are deferred and
will layer on top of this baseline in a future revision.
"""
from models.field_keys import (
    _get,
    EQUITY_KEYS,
    GOODWILL_KEYS,
    INTANGIBLES_KEYS,
)


def tangible_book_value_per_share(financials):
    """Tangible book value per share = (Equity − Goodwill − Intangibles) / Shares.

    *financials* is the same dict shape used by the other model functions —
    ``balance_sheet`` (a pandas DataFrame, latest period in column 0) and
    ``info`` (a dict with ``sharesOutstanding`` or similar).

    Returns
    -------
    float
        TBV per share when equity and shares are both present and positive.
    None
        When equity, shares, or the balance sheet itself are missing.
        Goodwill / intangibles absent are treated as 0 (some firms legitimately
        carry none); the function only refuses to compute when the required
        inputs (equity, shares) are unavailable.
    """
    bs = financials.get('balance_sheet')
    info = financials.get('info') or {}
    if bs is None or bs.empty:
        return None
    latest_bs = bs.iloc[:, 0]

    equity = _get(latest_bs, EQUITY_KEYS)
    if not equity or equity <= 0:
        return None

    goodwill = _get(latest_bs, GOODWILL_KEYS) or 0
    intangibles = _get(latest_bs, INTANGIBLES_KEYS) or 0

    # 'Goodwill And Other Intangible Assets' (the second GOODWILL_KEYS entry) is
    # a combined field some balance sheets use instead of two separate lines.
    # When that's what _get matched, don't ALSO add an intangibles row — that
    # would double-count. Heuristic: if the matched goodwill key was the
    # combined one AND a separate intangibles line was matched, prefer the
    # combined value alone.
    if 'Goodwill And Other Intangible Assets' in latest_bs.index:
        gw_combined = latest_bs['Goodwill And Other Intangible Assets']
        try:
            if gw_combined == goodwill:
                intangibles = 0
        except Exception:
            pass

    tangible_equity = float(equity) - float(goodwill) - float(intangibles)
    if tangible_equity <= 0:
        return None

    shares = info.get('sharesOutstanding')
    if not shares or shares <= 0:
        return None

    return tangible_equity / float(shares)
