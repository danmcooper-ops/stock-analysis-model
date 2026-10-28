"""Shared series helpers. Pure functions, no I/O.

Two series shapes flow through this project, both sorted ascending as
(period, value) pairs:

    daily spot   [('1999-01-04', 0.008816), ...]   USD per unit of currency
    monthly REER [('1994-01', 164.28), ...]        index, 2020 = 100

Every function here returns None rather than raising or substituting a
default when there is not enough data. A currency with two years of history
must score as "insufficient history", never as "zero volatility".
"""

import math
import statistics

# Trading days per year in the ECB reference series (weekdays minus TARGET
# holidays). Used to convert lookbacks in years to observation counts and to
# annualize daily volatility.
TRADING_DAYS_YEAR = 258


def tail(series, n):
    """Last n observations, or None if the series is shorter than n."""
    if not series or n <= 0 or len(series) < n:
        return None
    return series[-n:]


def values(series):
    return [v for _p, v in series] if series else []


def last_value(series):
    return series[-1][1] if series else None


def last_period(series):
    return series[-1][0] if series else None


def mean(xs):
    return statistics.fmean(xs) if xs else None


def stdev(xs):
    """Sample standard deviation; None for fewer than two points."""
    if not xs or len(xs) < 2:
        return None
    try:
        return statistics.stdev(xs)
    except statistics.StatisticsError:
        return None


def z_score(series, lookback):
    """Standard score of the latest value against the trailing `lookback` window.

    The window *includes* the current observation, which is conventional for
    a valuation z-score: the question is where today sits within the recent
    distribution, not how far it is from a window that excludes it.

    Returns None when the window is unavailable or degenerate (a currency
    genuinely pegged flat has zero dispersion, and dividing by it would
    manufacture an infinite signal).
    """
    window = tail(series, lookback)
    if window is None:
        return None
    xs = values(window)
    mu, sd = mean(xs), stdev(xs)
    if mu is None or not sd:
        return None
    return (xs[-1] - mu) / sd


def pct_deviation(series, lookback):
    """Latest value versus the mean of the trailing window, as a fraction.

    -0.20 means the current level is 20% below its own trailing average.
    """
    window = tail(series, lookback)
    if window is None:
        return None
    xs = values(window)
    mu = mean(xs)
    if not mu:
        return None
    return xs[-1] / mu - 1.0


def pct_change(series, periods_back):
    """Fractional change over `periods_back` observations."""
    if not series or len(series) <= periods_back or periods_back < 0:
        return None
    prev = series[-1 - periods_back][1]
    cur = series[-1][1]
    if not prev:
        return None
    return cur / prev - 1.0


def pct_change_between(series, older_back, newer_back):
    """Change from `older_back` ago to `newer_back` ago.

    This is what makes a 12-minus-1 momentum measure possible: the most
    recent month is skipped because short-horizon FX reversal is well
    documented and would otherwise fight the medium-horizon trend signal.
    """
    if not series or older_back <= newer_back:
        return None
    if len(series) <= older_back:
        return None
    prev = series[-1 - older_back][1]
    cur = series[-1 - newer_back][1]
    if not prev:
        return None
    return cur / prev - 1.0


def moving_average(series, window):
    w = tail(series, window)
    if w is None:
        return None
    return mean(values(w))


def log_returns(series):
    """Consecutive log returns. Non-positive prices are skipped, not zeroed."""
    out = []
    prev = None
    for _p, v in series or []:
        if v is not None and v > 0:
            if prev is not None:
                out.append(math.log(v / prev))
            prev = v
    return out


def realized_vol(series, lookback, periods_per_year=TRADING_DAYS_YEAR):
    """Annualized standard deviation of log returns over the trailing window.

    Returned as a fraction (0.09 = 9% annualized).
    """
    window = tail(series, lookback + 1)
    if window is None:
        return None
    rets = log_returns(window)
    sd = stdev(rets)
    if sd is None:
        return None
    return sd * math.sqrt(periods_per_year)


def max_drawdown(series, lookback):
    """Largest peak-to-trough decline over the window, as a positive fraction.

    0.35 means the currency fell 35% from a running high at some point in the
    window. Returned positive so "bigger is worse" reads naturally.
    """
    window = tail(series, lookback)
    if window is None:
        return None
    peak = None
    worst = 0.0
    for _p, v in window:
        if v is None or v <= 0:
            continue
        if peak is None or v > peak:
            peak = v
        elif peak:
            worst = max(worst, 1.0 - v / peak)
    return worst


def annual_series_stdev(pairs):
    """Standard deviation of an annual (year, value) series."""
    return stdev([v for _y, v in pairs]) if pairs else None
