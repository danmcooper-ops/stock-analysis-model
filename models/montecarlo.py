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
"""
import warnings
import zlib

import numpy as np
from scipy.stats import norm, qmc

DEFAULT_SEED = 42
_EPS = 1e-10


def seed_from_ticker(ticker):
    """Stable 32-bit seed for a ticker.

    CRC32 rather than Python's hash(), which is salted per process. None
    (no ticker context) falls back to the historical global seed.
    """
    if ticker is None:
        return DEFAULT_SEED
    return zlib.crc32(str(ticker).strip().upper().encode('utf-8')) & 0xFFFFFFFF


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
