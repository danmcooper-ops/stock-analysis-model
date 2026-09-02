# models/capm.py
import numpy as np
import pandas as pd

from models.valuation_types import _validate_returns

# Prior for precision-weighted beta shrinkage (see shrink_beta). Mirrored in
# scripts/config.py as BETA_PRIOR_MEAN / BETA_PRIOR_SD; the pipeline passes
# those explicitly, these are library defaults.
DEFAULT_BETA_PRIOR_MEAN = 1.0
DEFAULT_BETA_PRIOR_SD = 0.20

# Trailing windows, in weekly observations, for the rolling-beta diagnostic.
ROLLING_BETA_WINDOWS = {'1y': 52, '3y': 156, '5y': 260}


def shrink_beta(raw_beta, se_beta, prior_mean=DEFAULT_BETA_PRIOR_MEAN,
                prior_sd=DEFAULT_BETA_PRIOR_SD):
    """Precision-weighted (Vasicek) shrinkage of a regression beta toward a prior.

        weight = prior_sd² / (prior_sd² + se_beta²)
        beta   = weight * raw_beta + (1 - weight) * prior_mean

    A precisely estimated beta (small standard error) keeps most of its own
    value; a noisy one is pulled toward the prior. With the default prior
    sd of 0.20: weight ≈ 0.80 at the ~0.10 standard error typical of a
    5-year weekly regression, 0.64 at 0.15 (about the Blume 2/3 rule), and
    ≈ 0.30 at the ~0.24 typical of a 1-year window.

    Returns (shrunk_beta, weight). Raises ValueError on a non-finite
    standard error so NaN cannot leak into discount-rate math.
    """
    if se_beta is None or not np.isfinite(se_beta) or se_beta < 0:
        raise ValueError(f"se_beta must be a finite non-negative number, got {se_beta!r}")
    prior_var = float(prior_sd) ** 2
    if prior_var <= 0:
        return float(prior_mean), 0.0
    weight = prior_var / (prior_var + float(se_beta) ** 2)
    return float(weight * raw_beta + (1.0 - weight) * prior_mean), float(weight)


def calculate_beta(stock_returns, market_returns, adjust=True, min_obs=24,
                   prior_mean=DEFAULT_BETA_PRIOR_MEAN,
                   prior_sd=DEFAULT_BETA_PRIOR_SD):
    """Regression beta with Blume and precision-weighted adjustments, R² and SE.

    Returns a dict with:
      raw_beta       OLS slope of stock on market returns
      adjusted_beta  Blume (2/3 raw + 1/3) when adjust=True, else raw_beta
      shrunk_beta    precision-weighted shrinkage toward prior_mean (see
                     shrink_beta) — the value the pipeline feeds to CAPM
      shrink_weight  share of raw_beta retained by the shrinkage
      r_squared, se_beta (standard error of raw_beta), n_observations,
      warnings (list of str)

    Hardened: requires equal-length series, min_obs observations, finite
    returns, and nonzero market variance. Raises ValueError on any of
    these so callers can't silently propagate inf / NaN into downstream
    discount-rate math. Wrap the call in try/except if upstream data
    quality is uncertain.

    Sample statistics use ddof=1 (the ratio that defines beta is invariant
    to ddof; it matters for the standard error). R² is clipped into
    [0, 1] to absorb floating-point noise.
    """
    s = _validate_returns('stock_returns', stock_returns, min_obs=min_obs)
    m = _validate_returns('market_returns', market_returns, min_obs=min_obs)
    if s.size != m.size:
        raise ValueError(
            f"stock_returns ({s.size}) and market_returns ({m.size}) "
            "must be equal length"
        )

    market_var = float(np.var(m, ddof=1))
    if market_var <= 0:
        raise ValueError("market_returns variance is zero — beta is undefined")
    if float(np.var(s, ddof=1)) <= 0:
        # A flat stock series makes correlation/R² NaN; fail loudly rather
        # than leak NaN into beta_r2 and the JSON payload.
        raise ValueError("stock_returns variance is zero — beta is undefined")

    cov_matrix = np.cov(s, m, ddof=1)
    raw_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    adjusted_beta = (2 / 3) * raw_beta + (1 / 3) * 1.0 if adjust else raw_beta

    r_squared = float(np.clip(np.corrcoef(s, m)[0, 1] ** 2, 0.0, 1.0))

    n = s.size
    se_beta = (np.sqrt((1 - r_squared) / (n - 2))
               * (np.std(s, ddof=1) / np.std(m, ddof=1))) if n > 2 else None

    if se_beta is not None:
        shrunk_beta, shrink_weight = shrink_beta(
            float(raw_beta), float(se_beta), prior_mean=prior_mean, prior_sd=prior_sd)
    else:  # unreachable with min_obs >= 3; keep the payload finite regardless
        shrunk_beta, shrink_weight = float(raw_beta), 1.0

    warnings = []
    if n < 60:
        warnings.append(f'Only {n} observations — beta estimate is noisy')
    if r_squared < 0.20:
        warnings.append(
            f'Low R² ({r_squared:.2f}) — the market explains little of the '
            'return variance'
        )
    if shrink_weight < 0.5:
        warnings.append(
            f'Beta shrunk {1 - shrink_weight:.0%} toward {prior_mean:.1f} '
            f'(standard error {se_beta:.2f})'
        )

    return {
        'raw_beta': float(raw_beta),
        'adjusted_beta': float(adjusted_beta),
        'shrunk_beta': shrunk_beta,
        'shrink_weight': shrink_weight,
        'r_squared': r_squared,
        'se_beta': float(se_beta) if se_beta is not None else None,
        'n_observations': n,
        'warnings': warnings,
    }


def _dedupe_index(series):
    idx = series.index
    if idx.has_duplicates:
        return series[~idx.duplicated(keep='last')]
    return series


def weekly_returns(stock_close, market_close, min_days_edge_bin=4):
    """Aligned weekly simple returns for a stock and its market proxy.

    Joins the two close series on their shared trading dates, bins them
    into Friday-ended weeks, takes the last shared close of each week and
    differences. A first or last bin holding fewer than *min_days_edge_bin*
    trading days is dropped: a mid-week run would otherwise contribute a
    one- or two-day "week", while a holiday-shortened 4-day week is kept.

    Both series must share tz-awareness (strip tz upstream when mixing
    sources). Duplicate index labels keep the last value.

    Returns (stock_returns, market_returns, index) — two equal-length 1-D
    float arrays and the DatetimeIndex (week-end dates) of the returns.
    Empty arrays when there is no overlap.
    """
    combined = pd.DataFrame({
        'stock': _dedupe_index(pd.Series(stock_close, dtype=float)),
        'market': _dedupe_index(pd.Series(market_close, dtype=float)),
    }).dropna().sort_index()
    if combined.empty:
        empty = np.array([], dtype=float)
        return empty, empty, pd.DatetimeIndex([])
    if not isinstance(combined.index, pd.DatetimeIndex):
        raise ValueError("weekly_returns needs a DatetimeIndex")

    groups = combined.resample('W-FRI')
    weekly = groups.last().dropna()
    counts = groups.size().reindex(weekly.index)
    if len(weekly) and counts.iloc[-1] < min_days_edge_bin:
        weekly = weekly.iloc[:-1]
        counts = counts.iloc[:-1]
    if len(weekly) and counts.iloc[0] < min_days_edge_bin:
        weekly = weekly.iloc[1:]

    rets = weekly.pct_change().dropna()
    return (rets['stock'].to_numpy(dtype=float),
            rets['market'].to_numpy(dtype=float),
            rets.index)


def rolling_betas(stock_returns, market_returns, windows=None, min_obs=24,
                  min_coverage=0.9, **beta_kwargs):
    """Beta over trailing windows of one return series (weekly by convention).

    *windows* maps a label to a trailing observation count (default
    ROLLING_BETA_WINDOWS: 52 / 156 / 260 weeks). A window is reported only
    when the series covers at least *min_coverage* of it (so a young stock
    gets a 1y beta but no 5y one, rather than three copies of the same
    short regression; the 10% slack absorbs holiday weeks and edge-bin
    drops in a nominal 5-year history). Windows below *min_obs* or that
    fail calculate_beta are omitted.
    Extra keyword arguments (prior_mean, prior_sd, adjust) are forwarded
    to calculate_beta.

    Returns {label: {'beta': raw, 'shrunk': shrunk, 'r2', 'se', 'n'}, ...,
    'stability': std of the RAW betas across windows (None with fewer than
    two windows)}. Raw betas are reported because short windows shrink
    hard toward the prior and would make every stock look stable. Returns
    {} when no window can be computed.
    """
    windows = ROLLING_BETA_WINDOWS if windows is None else windows
    s_all = np.asarray(stock_returns, dtype=float)
    m_all = np.asarray(market_returns, dtype=float)
    n_all = min(s_all.size, m_all.size)
    s_all, m_all = s_all[:n_all], m_all[:n_all]

    out = {}
    raws = []
    for label, n in windows.items():
        if n_all < min_coverage * n or n < min_obs:
            continue
        s, m = s_all[-n:], m_all[-n:]
        try:
            r = calculate_beta(s, m, min_obs=min_obs, **beta_kwargs)
        except ValueError:
            continue
        out[label] = {
            'beta': round(r['raw_beta'], 4),
            'shrunk': round(r['shrunk_beta'], 4),
            'r2': round(r['r_squared'], 4),
            'se': round(r['se_beta'], 4) if r['se_beta'] is not None else None,
            'n': r['n_observations'],
        }
        raws.append(r['raw_beta'])
    if not out:
        return {}
    out['stability'] = round(float(np.std(raws)), 4) if len(raws) > 1 else None
    return out


def r2_diagnostic(r2):
    """Descriptive R² tier for the beta regression (diagnostic label only).

    Returns (classification, method):
      >= 60%: 'reliable'     -> 'capm'
      40-59%: 'directional' -> 'capm_plus_alternative'
      < 40%:  'unreliable'  -> 'fundamental_only'

    The pipeline no longer selects the cost-of-equity method from this
    tier — R² measures how much of the return variance the market
    explains, not how precisely beta is estimated. Beta precision is
    handled by shrink_beta; this label is kept for the report.
    """
    if r2 >= 0.60:
        return 'reliable', 'capm'
    elif r2 >= 0.40:
        return 'directional', 'capm_plus_alternative'
    else:
        return 'unreliable', 'fundamental_only'


def expected_return(risk_free_rate, beta, market_return):
    """CAPM: Re = Rf + β(Rm - Rf)."""
    return risk_free_rate + beta * (market_return - risk_free_rate)


def ggm_implied_re(dividend_yield, growth_rate, forward=False):
    """Gordon-growth-implied cost of equity: Re = D1 / P + g.

    *dividend_yield* is D/P. With forward=False it is treated as a trailing
    yield (D0 / P) and grown one year: Re = yield * (1 + g) + g. With
    forward=True it is already D1 / P (e.g. yfinance's ``dividendRate``,
    the indicated forward annual rate) and Re = yield + g.
    """
    if dividend_yield is None or dividend_yield <= 0:
        return None
    if forward:
        return dividend_yield + growth_rate
    return dividend_yield * (1 + growth_rate) + growth_rate


def buildup_re(risk_free_rate, erp=0.045, size_premium=0.02, industry_premium=0.01):
    """Build-Up Re = Rf + ERP + size premium + industry premium.

    The *erp* default mirrors scripts.config.ERP; keep the two in sync.
    """
    return risk_free_rate + erp + size_premium + industry_premium
