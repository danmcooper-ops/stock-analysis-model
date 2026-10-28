"""Resampling inference for the backtest: how much of a measured edge is noise.

The backtest's headline numbers are means over snapshots (one rank IC, or one
bucket's mean excess return, per snapshot date), and daily snapshots over a
30-day horizon share most of their return window, so the snapshot count says
nothing about how much independent evidence there is. Two tools answer that:

- **Stationary block bootstrap** (Politis & Romano 1994) of a per-snapshot
  series: resample runs of consecutive snapshots, with geometric run lengths
  whose mean is the number of snapshots in one horizon, so each run is about
  one independent period. Gives a confidence interval and a two-sided
  p-value for the mean. With few independent periods (the corpus has had
  about six at 30 days) the bootstrap understates the spread, as any
  resampling of six observations must; the interval is a floor on the
  uncertainty, never a guarantee.
- **Ticker permutation test** of the mean rank IC: re-pair every ticker's
  forward returns with another ticker's scores, using ONE permutation across
  every snapshot. That keeps each ticker's return path intact, so the overlap
  between snapshots is preserved and the null distribution is as correlated
  across dates as the real IC series. Shuffling each date independently would
  treat overlapping days as independent and make the null far too narrow.
  Unlike the bootstrap it stays valid on a single period, but then it only
  says the scores ranked returns in THAT episode, not that they will again.

Spearman here is the backtest's own formula (``1 - 6Σd²/(n(n²-1))`` on
tie-averaged ranks, ``scripts.backtest._spearman``), so an observed IC from
this module equals the one ``composite_ic_summary`` reports.
"""
import math

import numpy as np
from scipy.stats import rankdata

DEFAULT_N_BOOT = 2000
DEFAULT_N_PERM = 200
DEFAULT_CI = 0.90
MIN_ROWS = 10
# Below this many independent periods the bootstrap has nothing to resample
# across: with one block it only reshuffles days of the same market episode
# and returns a tight interval around it (the corpus' 90-day horizon on
# 2026-10-28: one period, "CI" +0.088..+0.122, p 0.000).
MIN_BLOCKS = 3


def block_length(dates, horizon_days):
    """Mean bootstrap block length in snapshots: snapshots per independent period.

    ``dates`` are the snapshot dates (``datetime.date``) of one series. The
    independent-period count is the backtest's effective n,
    ``span // horizon + 1``.
    """
    n = len(dates)
    if n < 2:
        return 1
    span = (max(dates) - min(dates)).days
    eff = span // max(int(horizon_days), 1) + 1
    return max(1, int(round(n / eff)))


def stationary_bootstrap_indices(n, mean_block, n_boot, rng):
    """``(n_boot, n)`` index array from the circular stationary bootstrap.

    Each resample starts at a uniform index; every next position continues the
    current run with probability ``1 - 1/mean_block`` (wrapping at the end) or
    jumps to a fresh uniform index.
    """
    p = 1.0 / max(float(mean_block), 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    if n > 1:
        jump = rng.random((n_boot, n - 1)) < p
        fresh = rng.integers(0, n, (n_boot, n - 1))
        for j in range(1, n):
            cont = (idx[:, j - 1] + 1) % n
            idx[:, j] = np.where(jump[:, j - 1], fresh[:, j - 1], cont)
    return idx


def bootstrap_mean(values, mean_block, n_boot=DEFAULT_N_BOOT, ci=DEFAULT_CI,
                   seed=0):
    """Block-bootstrap CI and p-value for the mean of a per-snapshot series.

    ``values`` must be in date order. Returns ``{mean, ci_low, ci_high, se,
    p_value, ci, block, n_blocks, n_boot}``, or None for fewer than two
    values. ``p_value`` is two-sided against a zero mean: the share of
    bootstrap means at least as far from the sample mean as the sample mean
    is from zero (the centred bootstrap null). ``n_blocks`` is the number of
    independent periods the series holds; below ~10 read the interval as a
    lower bound on the uncertainty, and below MIN_BLOCKS ``ci_low``,
    ``ci_high``, ``se`` and ``p_value`` are None (``reason`` says why) since
    there is nothing independent to resample.
    """
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)],
                   dtype=float)
    n = len(x)
    if n < 2:
        return None
    n_blocks = max(1, int(round(n / max(mean_block, 1))))
    if n_blocks < MIN_BLOCKS:
        return {'mean': float(x.mean()), 'ci_low': None, 'ci_high': None,
                'se': None, 'p_value': None, 'ci': ci, 'block': int(mean_block),
                'n_blocks': n_blocks, 'n_boot': 0,
                'reason': f'{n_blocks} independent period(s), need {MIN_BLOCKS}'}
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    means = x[idx].mean(axis=1)
    mean = float(x.mean())
    lo, hi = np.quantile(means, [(1 - ci) / 2, (1 + ci) / 2])
    p = float(np.mean(np.abs(means - mean) >= abs(mean))) if mean != 0 else 1.0
    return {
        'mean': mean,
        'ci_low': float(lo),
        'ci_high': float(hi),
        'se': float(means.std(ddof=1)),
        'p_value': p,
        'ci': ci,
        'block': int(mean_block),
        'n_blocks': n_blocks,
        'n_boot': int(n_boot),
    }


def _spearman_vec(a, b):
    """The backtest's Spearman on two aligned float arrays (no NaNs)."""
    n = len(a)
    ra = rankdata(a, method='average')
    rb = rankdata(b, method='average')
    return 1.0 - 6.0 * float(np.sum((ra - rb) ** 2)) / (n * (n * n - 1))


def _panel_matrices(panels):
    """Stack ``[(tickers, scores, returns), ...]`` into date × ticker arrays."""
    universe = sorted({t for tickers, _, _ in panels for t in tickers})
    col = {t: i for i, t in enumerate(universe)}
    S = np.full((len(panels), len(universe)), np.nan)
    R = np.full_like(S, np.nan)
    for d, (tickers, scores, rets) in enumerate(panels):
        for t, s, r in zip(tickers, scores, rets, strict=True):
            j = col[t]
            S[d, j] = s
            R[d, j] = r
    return S, R


def _mean_ic(S, R, min_rows=MIN_ROWS):
    ics = []
    for d in range(S.shape[0]):
        ok = np.isfinite(S[d]) & np.isfinite(R[d])
        if ok.sum() >= min_rows:
            ics.append(_spearman_vec(S[d, ok], R[d, ok]))
    return float(np.mean(ics)) if ics else None


def ic_permutation_test(panels, n_perm=DEFAULT_N_PERM, seed=0,
                        min_rows=MIN_ROWS):
    """Ticker-permutation null for the mean per-snapshot rank IC.

    ``panels`` is one ``(tickers, scores, returns)`` triple per snapshot of one
    horizon (rows with a non-finite score or return already dropped). Each
    permutation re-pairs ticker ``t``'s scores with ticker ``π(t)``'s returns
    on every snapshot alike; a pair whose partner has no return that day drops
    out, as an unpriced row does. Returns ``{observed, null_mean, null_sd,
    null_p95, p_value, n_perm}`` (one-sided: how often noise matches or beats
    the observed mean IC, with the +1 correction so it is never 0), or None
    when no snapshot has ``min_rows`` rows.
    """
    if not panels:
        return None
    S, R = _panel_matrices(panels)
    observed = _mean_ic(S, R, min_rows)
    if observed is None:
        return None
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_perm)):
        v = _mean_ic(S, R[:, rng.permutation(R.shape[1])], min_rows)
        if v is not None:
            null.append(v)
    if not null:
        return None
    null = np.asarray(null)
    return {
        'observed': observed,
        'null_mean': float(null.mean()),
        'null_sd': float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        'null_p95': float(np.quantile(null, 0.95)),
        'p_value': float((np.sum(null >= observed) + 1) / (len(null) + 1)),
        'n_perm': len(null),
    }


def permutation_p(observed, null):
    """One-sided ``(#null ≥ observed + 1) / (n + 1)``; None without a null."""
    null = [v for v in null if v is not None and math.isfinite(v)]
    if observed is None or not null:
        return None
    return (sum(1 for v in null if v >= observed) + 1) / (len(null) + 1)
