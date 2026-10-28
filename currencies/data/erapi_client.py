"""open.er-api.com — an independent spot-rate source, used for verification.

    https://open.er-api.com/v6/latest/USD     (free, no key, ~166 currencies)

This is deliberately **not** wired into the pipeline. Its job is to
cross-check the one calculation in this project that fails silently and
catastrophically: the ECB cross.

ECB publishes units-per-EUR, so a USD-based rate is
`(USD per EUR) / (CCY per EUR)`. Invert that by accident and nothing raises —
every momentum, drawdown and volatility number simply flips sign, and the
screener confidently ranks the universe backwards. A second provider with a
completely different quote convention is the cheapest way to catch it.

Rates are latest-only (updated daily), which is why this cannot serve as a
history source and why the universe is built from ECB instead. Consumed by
`tests/test_live_endpoints.py`.
"""

from data.cache import fetch_json

_URL = 'https://open.er-api.com/v6/latest/USD'


def fetch_usd_rates(ttl_days=0.4, meta=None):
    """Latest rates as {currency: units_per_USD}, or None on failure.

    Note the convention is the *inverse* of this project's internal one: the
    API quotes units per USD (JPY -> ~160), while the model works in USD per
    unit (JPY -> ~0.0062). `usd_per_unit` does the flip.
    """
    payload = fetch_json(_URL, 'erapi', ttl_days=ttl_days, meta=meta)
    if not isinstance(payload, dict) or payload.get('result') != 'success':
        return None
    rates = payload.get('rates')
    if not isinstance(rates, dict):
        return None
    out = {}
    for code, value in rates.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out[code] = v
    return out or None


def usd_per_unit(rates, code):
    """Convert the API's units-per-USD to this project's USD-per-unit."""
    v = (rates or {}).get(code)
    return (1.0 / v) if v else None
