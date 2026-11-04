"""Where the composite's rank IC comes from: ranking within groups, or between them.

The headline IC (``scripts.backtest.composite_ic_summary``) ranks every row
against every other, so it blends two different skills: ordering companies
inside a sector, and favouring one sector over another. The scoring favours
some sectors by construction (fixed gate scales suit some balance sheets;
Financial Services has 11 gates scored 0), and that tilt is only worth keeping
if it predicts returns. Per snapshot this module measures:

- ``total``: the headline Spearman IC, score against excess return;
- ``within_sector`` / ``within_industry``: the pooled correlation of
  within-group centred ranks of score and return, i.e. how well the score
  orders names inside their own group. An industry with fewer than
  ``MIN_GROUP_ROWS`` rows falls back to its sector; a sector that small is
  left out;
- ``between_sector``: Spearman across sectors (at least ``MIN_GROUP_ROWS``
  rows each, at least ``MIN_GROUPS`` sectors) of the sector's mean score
  against its mean excess return, the sector tilt on its own;
- ``sector_neutral``: Spearman of the score less its sector's mean, against
  the RAW excess return. It is the IC a sector-neutralised composite would
  have earned, measured on the same target as ``total``; comparing
  ``within_sector`` with ``total`` would not be, since within-sector returns
  are an easier target. ``neutral_minus_total`` is the paired difference.

Each component's per-snapshot series is block-bootstrapped as the headline IC
is (``models.inference.bootstrap_mean``), and also averaged per independent
period (``by_period``: consecutive horizon-long windows from the first
snapshot), because on 2026-07..10 the components flipped sign between
periods (the sector tilt read -0.06 then +0.09) and a mean alone hides that.
``within_permutation_test`` re-pairs each ticker's whole return path with
another ticker's from the same sector, one permutation across every snapshot,
so overlap between snapshots stays in the null as in
``models.inference.ic_permutation_test``.

Display only: nothing here feeds a rating.
"""
from collections import Counter, defaultdict
from datetime import date

import numpy as np
from scipy.stats import rankdata

from models.inference import (DEFAULT_N_PERM, _spearman_vec, block_length,
                              bootstrap_mean)

MIN_GROUP_ROWS = 15
MIN_GROUPS = 5
MIN_ROWS = 10
COMPONENTS = ('total', 'within_sector', 'within_industry', 'between_sector',
              'sector_neutral', 'neutral_minus_total')


def _centred_ranks(values, groups, keep):
    """Ranks within each kept group, centred on 0 and scaled to (-0.5, 0.5)."""
    out = np.full(len(values), np.nan)
    for g in keep:
        idx = np.flatnonzero(groups == g)
        n = len(idx)
        out[idx] = (rankdata(values[idx], method='average') - (n + 1) / 2) / n
    return out


def _pooled_within(scores, rets, groups):
    """Pooled correlation of within-group centred ranks; None if too few rows."""
    counts = Counter(groups.tolist())
    keep = [g for g, n in counts.items() if g is not None and n >= MIN_GROUP_ROWS]
    if not keep:
        return None
    a = _centred_ranks(scores, groups, keep)
    b = _centred_ranks(rets, groups, keep)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_ROWS or a[ok].std() == 0 or b[ok].std() == 0:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _industry_groups(sectors, industries):
    """Industry label, or the sector's when the industry is under MIN_GROUP_ROWS."""
    counts = Counter(industries.tolist())
    return np.array([i if i is not None and counts[i] >= MIN_GROUP_ROWS else s
                     for s, i in zip(sectors, industries, strict=True)], dtype=object)


def snapshot_components(sectors, industries, scores, rets):
    """Every component for one snapshot, from aligned per-row lists.

    Rows need a finite score and return; a missing sector or industry is
    allowed and keeps the row out of the groups that need it. Returns
    ``{component: float | None, 'per_sector': {sector: ic}}``, or None under
    ``MIN_ROWS`` rows.
    """
    s = np.asarray(scores, dtype=float)
    r = np.asarray(rets, dtype=float)
    if len(s) < MIN_ROWS:
        return None
    sec = np.array([x or None for x in sectors], dtype=object)
    ind = _industry_groups(sec, np.array([x or None for x in industries], dtype=object))
    out = {'total': _spearman_vec(s, r)}
    out['within_sector'] = _pooled_within(s, r, sec)
    out['within_industry'] = _pooled_within(s, r, ind)

    counts = Counter(sec.tolist())
    big = [g for g, n in counts.items() if g is not None and n >= MIN_GROUP_ROWS]
    if len(big) >= MIN_GROUPS:
        ms = [s[sec == g].mean() for g in big]
        mr = [r[sec == g].mean() for g in big]
        out['between_sector'] = _spearman_vec(np.array(ms), np.array(mr))
    else:
        out['between_sector'] = None

    # Sector means only for groups large enough to estimate one; any other
    # row keeps the universe mean, i.e. is left as it is.
    centred = s - s.mean()
    for g in big:
        m = sec == g
        centred[m] = s[m] - s[m].mean()
    out['sector_neutral'] = _spearman_vec(centred, r)
    out['neutral_minus_total'] = out['sector_neutral'] - out['total']

    out['per_sector'] = {}
    for g in big:
        m = sec == g
        if np.ptp(s[m]) > 0 and np.ptp(r[m]) > 0:
            out['per_sector'][g] = _spearman_vec(s[m], r[m])
    return out


def _as_date(d):
    return d if isinstance(d, date) else date.fromisoformat(str(d)[:10])


def decompose(snapshots, horizon_days):
    """Bootstrap and per-period summary of every component for one horizon.

    ``snapshots`` is ``[(run_date, {component: value, 'per_sector': {...}}), ...]``
    (``snapshot_components`` results; None entries are skipped). Returns
    ``{'components': {name: bootstrap + n_snapshots}, 'by_period': [...],
    'per_sector': {sector: bootstrap + n_snapshots}, 'min_group_rows': n}``,
    or None when no snapshot qualifies.
    """
    snaps = sorted(((_as_date(d), c) for d, c in snapshots if c), key=lambda t: t[0])
    if not snaps:
        return None
    seed = int(horizon_days)

    def boot(pairs):
        pairs = [(d, v) for d, v in pairs if v is not None]
        if len(pairs) < 2:
            return None
        res = bootstrap_mean([v for _, v in pairs],
                             block_length([d for d, _ in pairs], horizon_days), seed=seed)
        if res:
            res['n_snapshots'] = len(pairs)
        return res

    components = {}
    for name in COMPONENTS:
        b = boot([(d, c.get(name)) for d, c in snaps])
        if b:
            components[name] = b

    d0 = snaps[0][0]
    periods = defaultdict(list)
    for d, c in snaps:
        periods[(d - d0).days // int(horizon_days)].append((d, c))
    by_period = []
    for k in sorted(periods):
        rows = periods[k]
        entry = {'period': k, 'first': rows[0][0].isoformat(),
                 'last': rows[-1][0].isoformat(), 'n_snapshots': len(rows)}
        for name in COMPONENTS:
            vals = [c[name] for _, c in rows if c.get(name) is not None]
            entry[name] = float(np.mean(vals)) if vals else None
        by_period.append(entry)

    per_sector = {}
    for g in sorted({g for _, c in snaps for g in c.get('per_sector', {})}):
        b = boot([(d, c['per_sector'].get(g)) for d, c in snaps])
        if b:
            per_sector[g] = b
    return {'components': components, 'by_period': by_period,
            'per_sector': per_sector, 'min_group_rows': MIN_GROUP_ROWS}


def within_permutation_test(panels, n_perm=DEFAULT_N_PERM, seed=0):
    """Same-sector ticker-permutation null for the mean within-sector IC.

    ``panels`` is one ``(tickers, sectors, scores, returns)`` tuple per
    snapshot of one horizon, in date order. Each ticker is grouped under the
    sector it carries most often, so a permutation maps whole return paths
    inside one sector on every snapshot alike. Returns ``{observed,
    null_mean, null_p95, p_value, n_perm}`` (one-sided, +1 corrected), or None
    when no snapshot yields a within-sector IC.
    """
    if not panels or not n_perm:
        return None
    seen = defaultdict(Counter)
    for tickers, sectors, _, _ in panels:
        for t, g in zip(tickers, sectors, strict=True):
            if g:
                seen[t][g] += 1
    universe = sorted(seen)
    col = {t: j for j, t in enumerate(universe)}
    home = np.array([seen[t].most_common(1)[0][0] for t in universe], dtype=object)
    S = np.full((len(panels), len(universe)), np.nan)
    R = np.full_like(S, np.nan)
    for d, (tickers, _, scores, rets) in enumerate(panels):
        for t, s, r in zip(tickers, scores, rets, strict=True):
            if t in col:
                S[d, col[t]] = s
                R[d, col[t]] = r

    def mean_within(Rm):
        ics = []
        for d in range(S.shape[0]):
            ok = np.isfinite(S[d]) & np.isfinite(Rm[d])
            v = _pooled_within(S[d, ok], Rm[d, ok], home[ok])
            if v is not None:
                ics.append(v)
        return float(np.mean(ics)) if ics else None

    observed = mean_within(R)
    if observed is None:
        return None
    groups = [np.flatnonzero(home == g) for g in sorted(set(home.tolist()))]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_perm)):
        perm = np.arange(len(universe))
        for idx in groups:
            perm[idx] = idx[rng.permutation(len(idx))]
        v = mean_within(R[:, perm])
        if v is not None:
            null.append(v)
    if not null:
        return None
    null = np.asarray(null)
    return {
        'observed': observed,
        'null_mean': float(null.mean()),
        'null_p95': float(np.quantile(null, 0.95)),
        'p_value': float((np.sum(null >= observed) + 1) / (len(null) + 1)),
        'n_perm': len(null),
    }
