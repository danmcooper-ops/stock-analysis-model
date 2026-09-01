# models/rim.py
"""Residual Income Model (RIM) — book value plus present value of excess earnings.

Value = Book Value + PV of Residual Income stream + Terminal Value.
Residual Income = Book Value * (ROE - cost_of_equity).
Useful complement to DCF: anchored to book value rather than free cash flow.
"""
import warnings as _py_warnings


from models.valuation_types import Valuation, _validate_numeric


DEFAULT_SPREAD_PERSISTENCE = 0.5
DEFAULT_MAX_BOOK_GROWTH = 0.25


def retention_from_shareholder_returns(net_income, total_shareholder_return):
    """Earnings retention implied by TOTAL shareholder returns.

    retention = 1 − (dividends + net buybacks) / net income, clamped to
    [0, 1]. The dividend-only ``payoutRatio`` overstates retention for
    buyback-heavy firms (which return most earnings without paying a
    dividend), and under clean surplus that overstated retention compounds
    book value at ROE × retention every year. Returns None when net income
    is missing or non-positive (retention is undefined for a loss year) or
    when the return figure is missing, so the caller can fall back.
    """
    if net_income is None or total_shareholder_return is None:
        return None
    try:
        ni = float(net_income)
        ret = float(total_shareholder_return)
    except (TypeError, ValueError):
        return None
    if not (ni > 0) or ni != ni or ret != ret:
        return None
    return max(0.0, min(1.0, 1.0 - ret / ni))


def residual_income_model_valuation(book_value_per_share, roe, cost_of_equity,
                                    g=0.03, years=10, retention_ratio=None,
                                    spread_persistence=DEFAULT_SPREAD_PERSISTENCE,
                                    max_book_growth=DEFAULT_MAX_BOOK_GROWTH):
    """Residual Income Model as a Valuation envelope: BV + PV of excess earnings.

    Value = BV + sum(RI_t / (1+Re)^t) + TV
    where RI_t = BV_t-1 * (ROE_t - Re), BV_t = BV_t-1 * (1 + ROE_t × retention)
    and ROE_t fades linearly from the input ROE toward Re over the explicit
    horizon (see ``spread_persistence``). Terminal value uses Gordon Growth
    on the year-N+1 residual income.

    Parameters
    ----------
    book_value_per_share, roe, cost_of_equity, g, years
        Standard RIM inputs.
    retention_ratio : float or None
        Fraction of earnings retained (i.e. not paid out as dividend or
        buyback). Drives how fast book value compounds. When None
        (default), it is inferred from g/ROE — the Gordon-consistent
        retention rate that produces the stated long-term growth — and
        a warning is emitted. Pass an explicit value to override.

        The earlier implementation silently assumed all earnings were
        retained (book grew at fixed `g` regardless of payout), which
        inflated terminal-period book value for any firm that paid a
        dividend or bought back stock. The new default is closer to
        the truth for most firms; pass `retention_ratio=1.0` only if
        the company genuinely reinvests every dollar.
    spread_persistence : float
        Fraction of the initial excess spread (ROE − Re) that survives at
        the end of the explicit horizon and into the terminal period. The
        spread fades linearly: ROE_t = Re + spread × (1 − (1 − p) · t / N).
        1.0 reproduces the old no-fade behaviour (a single trailing-year
        ROE capitalised in perpetuity); 0.0 fades fully to the cost of
        equity so the terminal value is zero. Negative spreads mean-revert
        the same way. Default 0.5.
    max_book_growth : float
        Ceiling on the per-year clean-surplus book growth ROE_t × retention.
        Without it a high-ROE, high-retention input compounds book value
        past any plausible level (150% ROE × 85% retention = +127%/yr for
        ten years). Hitting the cap records a warning. Default 25%/yr.
    """
    method = 'residual_income'
    warns = []

    def _warn(msg):
        warns.append(msg)
        _py_warnings.warn(msg, RuntimeWarning, stacklevel=3)

    if book_value_per_share is None or book_value_per_share <= 0:
        return Valuation.invalid(method, 'book value per share missing or non-positive')
    try:
        roe = _validate_numeric('roe', roe, low=-1.0, high=2.0)
        cost_of_equity = _validate_numeric('cost_of_equity', cost_of_equity,
                                           positive=True, low=0.01, high=0.40)
        g = _validate_numeric('g', g, low=-0.05, high=0.10)
    except ValueError as e:
        _py_warnings.warn(f"residual_income_model input invalid: {e}", RuntimeWarning, stacklevel=3)
        return Valuation.invalid(method, f'input invalid: {e}')
    inputs = {'book_value_per_share': book_value_per_share, 'roe': roe,
              'cost_of_equity': cost_of_equity, 'g': g, 'years': years}
    if cost_of_equity <= g:
        return Valuation.invalid(
            method, 'cost_of_equity <= g — terminal value undefined', inputs)

    # Resolve retention ratio
    if retention_ratio is None:
        if roe > 0:
            retention_ratio = max(min(g / roe, 1.0), 0.0)
            _warn(f'retention_ratio not provided — inferred {retention_ratio:.2f} '
                  'from g/ROE (Gordon-consistent). Pass an explicit value if you '
                  'have payout data.')
        else:
            retention_ratio = 1.0
            _warn('retention_ratio not provided and ROE <= 0 — falling back to '
                  '100% retention. Book-value evolution is suspect.')
    retention_ratio = max(min(retention_ratio, 1.0), 0.0)
    spread_persistence = max(min(float(spread_persistence), 1.0), 0.0)
    max_book_growth = max(float(max_book_growth), 0.0)
    inputs['retention_ratio'] = retention_ratio
    inputs['spread_persistence'] = spread_persistence
    inputs['max_book_growth'] = max_book_growth

    bv = book_value_per_share
    pv_ri = 0.0
    spread = roe - cost_of_equity
    roe_t = roe
    capped_years = 0

    for t in range(1, years + 1):
        # Excess returns mean-revert: competition erodes an above-cost
        # spread and restructuring/repricing lifts a below-cost one. Fade
        # linearly toward Re, keeping `spread_persistence` of the initial
        # spread by year N. A single trailing-year ROE held flat for a
        # decade and then in perpetuity capitalised peak-cycle years.
        roe_t = cost_of_equity + spread * (1 - (1 - spread_persistence) * t / years)
        ri = bv * (roe_t - cost_of_equity)
        pv_ri += ri / (1 + cost_of_equity) ** t
        # Book value grows by retained earnings (clean surplus), capped so
        # an extreme ROE × retention cannot compound book past plausibility.
        growth = roe_t * retention_ratio
        if growth > max_book_growth:
            growth = max_book_growth
            capped_years += 1
        bv = bv * (1 + growth)

    if capped_years:
        _warn(f'book-value growth (ROE × retention) capped at '
              f'{max_book_growth:.0%}/yr in {capped_years} of {years} years — '
              'input ROE/retention imply implausible compounding')

    # Terminal value: residual income continues growing at g in perpetuity.
    # bv is already BV_N after the loop and roe_t is the faded year-N ROE,
    # so ri_terminal = BV_N·(ROE_N−Re) is RI_{N+1}; the Gordon value at year
    # N is RI_{N+1}/(Re−g) with NO extra (1+g) — the old factor double-grew
    # the stream and overstated every TV by ~(1+g). Negative terminal RI
    # (ROE < Re) is kept, not zeroed: a firm earning below its cost of
    # equity in perpetuity destroys value, and truncating only the downside
    # overstated chronic under-earners.
    ri_terminal = bv * (roe_t - cost_of_equity)
    if (cost_of_equity - g) > 0.005:
        tv = ri_terminal / (cost_of_equity - g)
        pv_tv = tv / (1 + cost_of_equity) ** years
    else:
        pv_tv = 0.0

    if abs(roe - cost_of_equity) < 0.005:
        _warn('ROE ≈ cost_of_equity — RIM degenerates toward book value '
              '(residual income near zero)')

    intrinsic = book_value_per_share + pv_ri + pv_tv
    if intrinsic <= 0:
        return Valuation.invalid(method, 'non-positive intrinsic value', inputs)
    return Valuation(value=intrinsic, method=method,
                     confidence=max(0.4, 1.0 - 0.15 * len(warns)),
                     warnings=tuple(warns), inputs_used=inputs)


def residual_income_model(book_value_per_share, roe, cost_of_equity,
                          g=0.03, years=10, retention_ratio=None,
                          spread_persistence=DEFAULT_SPREAD_PERSISTENCE,
                          max_book_growth=DEFAULT_MAX_BOOK_GROWTH):
    """Legacy float|None wrapper around residual_income_model_valuation()."""
    return residual_income_model_valuation(
        book_value_per_share, roe, cost_of_equity, g=g, years=years,
        retention_ratio=retention_ratio, spread_persistence=spread_persistence,
        max_book_growth=max_book_growth).value
