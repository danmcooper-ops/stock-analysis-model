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
import warnings as _py_warnings

from models.field_keys import (
    _get,
    EQUITY_KEYS,
    GOODWILL_KEYS,
    INTANGIBLES_KEYS,
    SHARES_OUTSTANDING_KEYS,
)

# A balance-sheet share count is only trusted when it sits within this band
# of the live count (when one exists). Outside it the row is almost always
# a single class of a multi-class filer, not the whole capital base.
_SHARE_COUNT_BAND = (2.0 / 3.0, 1.5)


def _resolve_share_count(latest_bs, info):
    """Pick the denominator for per-share book metrics.

    Prefers the period-end count from the balance sheet — dated like the
    equity it divides — and falls back to ``info['sharesOutstanding']``
    (today's count) when the row is absent, non-positive, or implausible
    against the live count. Returns None when neither is usable.
    """
    bs_shares = _get(latest_bs, SHARES_OUTSTANDING_KEYS)
    live = info.get('sharesOutstanding')
    live_ok = isinstance(live, (int, float)) and live > 0
    try:
        bs_ok = bs_shares is not None and float(bs_shares) > 0
    except (TypeError, ValueError):
        bs_ok = False
    if bs_ok:
        if not live_ok:
            return float(bs_shares)
        ratio = float(bs_shares) / float(live)
        if _SHARE_COUNT_BAND[0] <= ratio <= _SHARE_COUNT_BAND[1]:
            return float(bs_shares)
    if live_ok:
        return float(live)
    return None


def tangible_equity_per_share(financials):
    """Signed tangible equity per share = (Equity − Goodwill − Intangibles) / Shares.

    *financials* is the same dict shape used by the other model functions —
    ``balance_sheet`` (a pandas DataFrame, latest period in column 0) and
    ``info`` (a dict with ``sharesOutstanding`` or similar). The share
    count comes from the balance sheet's period-end 'Ordinary Shares
    Number' when present and plausible, so numerator and denominator carry
    the same date; ``info['sharesOutstanding']`` is the fallback.

    Unlike :func:`tangible_book_value_per_share`, this keeps the SIGN: a
    buyback-rich compounder whose goodwill exceeds its equity returns a
    negative number rather than None. Scoring relies on that distinction —
    "negative tangible book" makes P/TBV structurally inapplicable, whereas
    "balance sheet missing" (None) is a data gap that should score worst.

    Returns
    -------
    float
        Signed tangible equity per share when equity and shares are present
        and shares are positive. Goodwill / intangibles absent are treated
        as 0 (some firms legitimately carry none).
    None
        When equity, shares, or the balance sheet itself are missing.

    Emits a warning when goodwill + intangibles exceed 50% of (positive)
    equity — in that regime TBV strips most of book value and the
    asset-floor reading is materially below BV, so callers should weight
    it as a floor only, not a fair-value estimate.
    """
    bs = financials.get('balance_sheet')
    info = financials.get('info') or {}
    if bs is None or bs.empty:
        return None
    latest_bs = bs.iloc[:, 0]

    equity = _get(latest_bs, EQUITY_KEYS)
    if equity is None:
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
            # Non-scalar comparison (duplicate index rows) — keep both lines;
            # worst case is a conservative (lower) tangible book value.
            pass

    shares = _resolve_share_count(latest_bs, info)
    if shares is None:
        return None

    tangible_equity = float(equity) - float(goodwill) - float(intangibles)

    # Heads-up when intangibles dominate equity — caller may want to treat
    # TBV strictly as a floor and lean on other valuation methods.
    if equity > 0:
        intangible_pct = (float(goodwill) + float(intangibles)) / float(equity)
        if intangible_pct > 0.5:
            _py_warnings.warn(
                f'Goodwill + intangibles are {intangible_pct:.0%} of equity — '
                'TBV strips most of book; treat as floor only, not fair value',
                RuntimeWarning,
                stacklevel=2,
            )

    return tangible_equity / float(shares)


def tangible_book_value_per_share(financials):
    """Tangible book value per share, as a positive asset floor.

    Thin wrapper over :func:`tangible_equity_per_share` that returns None
    when tangible equity is non-positive (insolvent on a tangible basis, or
    equity itself non-positive) — a negative "floor" is not a usable fair
    value. Callers that need to tell negative tangible book apart from
    missing data should use the signed function directly.
    """
    tbv = tangible_equity_per_share(financials)
    if tbv is None or tbv <= 0:
        return None
    return tbv
