"""Shared sampling machinery for the Monte Carlo valuation simulators.

monte_carlo_dcf and monte_carlo_ddm draw their parameter scenarios here so
both describe the same uncertainty regime:

- Scrambled Sobol quasi-random points (scipy.stats.qmc) mapped through the
  normal inverse CDF. At the production iteration count the sampling error
  on the median is a fraction of what pseudo-random draws give for the same
  n, and it shrinks faster as n grows.
- Per-ticker seeds via seed_from_ticker(): reproducible for a company, but
  independent across the universe. A single global seed gave every stock
  with similar inputs the same z-scores, so the residual sampling error
  carried the same sign across the whole screen.
- Correlated discount-rate / terminal-growth draws (correlate()): both share
  the inflation and real-rate component, and sampling them independently
  overstated how often the terminal spread collapses against its wall.
- Exact truncated-normal draws (truncated_normal_from_uniform) for the
  discount-rate floor: conditioning on "the rate is not below the floor"
  instead of piling probability mass at the floor. The terminal-spread wall
  is deliberately NOT truncated: the point estimates substitute the minimum
  spread rather than reject tight spreads, so the simulators clip to the
  same rule and agree with the point estimate there.
- Log-normal exit multiples (lognormal_from_uniform): never negative, so the
  floor only binds for genuinely low multiples.
- Growth sigma with a floor (growth_sigma): proportional to the point growth,
  but never below an absolute spread. Proportional alone gave a company at
  +0.1% growth a 0.03pp sigma, near-certain growth, while one at exactly 0%
  got the 2pp fallback, so the simulation read stagnant and declining
  businesses as the most predictable ones.
"""
import warnings
import zlib

import numpy as np
from scipy.stats import norm, qmc

DEFAULT_SEED = 42
_EPS = 1e-10
GROWTH_SIGMA_RATIO = 0.30
GROWTH_SIGMA_FLOOR = 0.02


def seed_from_ticker(ticker):
    """Stable 32-bit seed for a ticker.

    CRC32 rather than Python's hash(), which is salted per process. None
    (no ticker context) falls back to the historical global seed.
    """
    if ticker is None:
        return DEFAULT_SEED
    return zlib.crc32(str(ticker).strip().upper().encode('utf-8')) & 0xFFFFFFFF


def growth_sigma(g, ratio=GROWTH_SIGMA_RATIO, floor=GROWTH_SIGMA_FLOOR):
    """Growth-rate sigma: `ratio` of |g|, never below `floor` (absolute)."""
    return max(abs(g) * ratio, floor)


def sobol_uniforms(n, dims, seed=None):
    """(n, dims) scrambled Sobol points, clipped into the open interval (0, 1)."""
    sampler = qmc.Sobol(d=dims, scramble=True,
                        seed=DEFAULT_SEED if seed is None else int(seed))
    with warnings.catch_warnings():
        # A non-power-of-two n forfeits Sobol's balance property but is still
        # a valid low-discrepancy sample; config.MC_ITERATIONS is 2**10.
        warnings.simplefilter('ignore', UserWarning)
        u = sampler.random(n)
    return np.clip(u, _EPS, 1 - _EPS)


def normal_from_uniform(u, mean, sigma):
    """N(mean, sigma) via the inverse CDF."""
    return mean + sigma * norm.ppf(u)


def truncated_normal_from_uniform(u, mean, sigma, lower=None, upper=None):
    """Exact inverse-CDF draw from N(mean, sigma) restricted to [lower, upper].

    Returns (values, excluded_mass): excluded_mass is the probability the
    untruncated distribution put outside the bounds — the smooth analogue of
    a clip rate. If the bounds exclude (numerically) everything the draw
    collapses onto the nearest bound with excluded_mass = 1.0.
    """
    a = 0.0 if lower is None else float(norm.cdf((lower - mean) / sigma))
    b = 1.0 if upper is None else float(norm.cdf((upper - mean) / sigma))
    width = b - a
    if width < 1e-12:
        fill = lower if lower is not None else upper
        return np.full(np.shape(u), float(fill)), 1.0
    z = norm.ppf(np.clip(a + u * width, _EPS, 1 - _EPS))
    return mean + sigma * z, float(1.0 - width)


def correlate(z_base, z_indep, rho):
    """Standard-normal z correlated with z_base at coefficient rho."""
    return rho * z_base + np.sqrt(1.0 - rho * rho) * z_indep


def lognormal_from_uniform(u, median, sigma):
    """Log-normal draw with the given MEDIAN and a spread of roughly `sigma`
    in the median's units (log-sd = sigma / median). Median-preserving so
    a collapsed sigma reproduces the point estimate exactly."""
    s = sigma / median
    return median * np.exp(s * norm.ppf(u))


# ---------------------------------------------------------------------------
# Growth uncertainty from evidence
# ---------------------------------------------------------------------------

GROWTH_SIGMA_CAP = 0.10
HISTORY_YEARS = 5


def evidence_growth_sigma(g, signals=None, weights=None, rev_growth_vol=None,
                          years=HISTORY_YEARS, floor=GROWTH_SIGMA_FLOOR,
                          cap=GROWTH_SIGMA_CAP, ratio=GROWTH_SIGMA_RATIO):
    """Growth-rate sigma from the company's own evidence, not its growth level.

    The simulation holds one growth draw for the explicit stage, so the sigma
    is the uncertainty of a ~5-year average growth rate. Two independent
    parts, combined in quadrature:

    - disagreement: the standard error of the weighted mean of the growth
      signals the point estimate blended (trailing FCF and revenue CAGRs,
      analyst short and long term, earnings growth, fundamental growth): their
      weighted standard deviation x sqrt(sum w^2) / sum w. The blend is an
      average of the signals, and an average is less uncertain than the
      signals' own spread (about 0.4x for six equal weights).
    - history: the revenue-growth volatility (std of year-on-year growth)
      divided by sqrt(years), the noise of a `years`-year average around its
      mean. Capped at `cap` before combining: revenue jumps from acquisitions
      or a first product sale put a few volatilities above 1,000%.

    The result is floored at `floor` and capped at `cap`. Without either part
    (fewer than two signals and no revenue history) it falls back to the old
    rule, `ratio` x |g| floored at `floor`.

    Returns (sigma, basis, parts): basis names the parts used, for the row
    ('disagreement+history', 'disagreement', 'history' or 'ratio'), and parts
    holds each part's value before combining.
    """
    parts = {}
    pairs = [(float(s), float(w)) for s, w in zip(signals or [], weights or [], strict=False)
             if s is not None and w is not None and np.isfinite(s) and np.isfinite(w) and w > 0]
    if len(pairs) >= 2:
        s = np.array([p[0] for p in pairs])
        w = np.array([p[1] for p in pairs])
        mean = float(np.sum(s * w) / np.sum(w))
        spread = float(np.sqrt(np.sum(w * (s - mean) ** 2) / np.sum(w)))
        parts['disagreement'] = spread * float(np.sqrt(np.sum(w ** 2)) / np.sum(w))
    if rev_growth_vol is not None and np.isfinite(rev_growth_vol) and rev_growth_vol >= 0:
        parts['history'] = min(float(rev_growth_vol) / np.sqrt(years), cap)
    if not parts:
        return growth_sigma(g, ratio, floor), 'ratio', {}
    sigma = float(np.hypot(parts.get('disagreement', 0.0), parts.get('history', 0.0)))
    return min(max(sigma, floor), max(cap, floor)), '+'.join(parts), parts


# ---------------------------------------------------------------------------
# Quantiles: where a price sits in the simulated fair values
# ---------------------------------------------------------------------------

QUANTILE_LEVELS = (0.01,) + tuple(round(0.05 * k, 2) for k in range(1, 20)) + (0.99,)


def fv_quantiles(values):
    """Fair values at QUANTILE_LEVELS (21 points, 1% to 99%)."""
    return [float(q) for q in np.quantile(np.asarray(values, dtype=float), QUANTILE_LEVELS)]


def quantile_rank(quantiles, x, levels=QUANTILE_LEVELS):
    """Percentile of `x` within a distribution given by its `quantiles`.

    Linear between stored quantiles. Beyond the outermost stored quantile
    the answer is that quantile's level (0.01 or 0.99): the run cannot say
    how far into the tail `x` sits. Flat stretches (a block of draws clamped
    to zero) return the middle of the flat run. None if inputs are missing.
    """
    if x is None or not quantiles or len(quantiles) != len(levels):
        return None
    q = np.asarray(quantiles, dtype=float)
    if not np.all(np.isfinite(q)) or not np.isfinite(x):
        return None
    lv = np.asarray(levels, dtype=float)
    if x < q[0]:
        return float(lv[0])
    if x > q[-1]:
        return float(lv[-1])
    hit = q == x
    if hit.any():
        return float(lv[hit].mean())
    hi = int(np.searchsorted(q, x, side='right'))   # q[hi - 1] < x < q[hi]
    lo = hi - 1
    return float(lv[lo] + (lv[hi] - lv[lo]) * (x - q[lo]) / (q[hi] - q[lo]))


# ---------------------------------------------------------------------------
# Attribution: which input drives the spread
# ---------------------------------------------------------------------------

def sobol_indices(f, n, dims, seed=None):
    """First-order and total-effect Sobol indices of `f` over `dims`
    independent uniform inputs.

    `f` maps an (m, dims) array of uniforms in (0, 1) to m outputs. Uses the
    pick-freeze design (two base matrices A and B and one A_B^(i) per input,
    n x (dims + 2) evaluations) with Saltelli's first-order and Jansen's
    total-effect estimators. Points come from one 2*dims Sobol sequence so
    A and B are independent and balanced.

    Returns (first, total) arrays, or (None, None) when the output does not
    vary. Estimates can fall slightly outside [0, 1] from sampling error;
    they are clipped.
    """
    u = sobol_uniforms(n, 2 * dims, seed)
    a, b = u[:, :dims], u[:, dims:]
    fa, fb = np.asarray(f(a), dtype=float), np.asarray(f(b), dtype=float)
    var = float(np.var(np.concatenate([fa, fb])))
    if not np.isfinite(var) or var <= 1e-18:
        return None, None
    first, total = np.empty(dims), np.empty(dims)
    for i in range(dims):
        abi = a.copy()
        abi[:, i] = b[:, i]
        fabi = np.asarray(f(abi), dtype=float)
        first[i] = np.mean(fb * (fabi - fa)) / var
        total[i] = 0.5 * np.mean((fa - fabi) ** 2) / var
    return np.clip(first, 0.0, 1.0), np.clip(total, 0.0, 1.0)
