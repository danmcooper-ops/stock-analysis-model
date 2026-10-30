# data/splits.py
"""Restate EDGAR share counts and per-share amounts for stock splits.

SEC companyfacts hold each value as filed. A count or EPS for a period that
was last reported *before* a split stays on the old basis, while periods
re-reported as comparatives after it are restated, so one series mixes both
bases. Alphabet's period-end shares read 691M for FY16-FY20 and 13.24B from
FY21 (its 20:1 split was 2022-07-18): the Summary sheet's ten-year record
showed FCF per share falling 18%/yr, and ``shares_cagr_5y`` — the Ownership:
Share Shrink gate — read CMG +114%/yr, AVGO +63%, NVDA +58% and WMT +23%.

The split events come from the price parquets (yfinance's ``Stock Splits``
column: exact date and ratio), not from the share series itself. Guessing a
ratio from a jump is unsafe: buybacks drift a ratio (AAPL's 7:1 reads 6.57x
in weighted shares) and a loss-to-profit year moves the diluted count (TSLA's
5:1 reads 6.12x, which a "nearest round ratio" rule would call 6:1).

Knowing the ratios, what is left is deciding which points are still on an
old basis, and that is decided per point, not by one boundary per series:
a value's basis follows the filing that last reported it, so period-end
shares, weighted-average shares and EPS were restated over different spans
(GOOGL's period-end shares jump between FY20 and FY21, its diluted EPS
between FY21 and FY22) and quarterly points can interleave the two bases.
``_choose_factors`` picks the restatement that makes each series move least
from point to point; EPS is judged through its implied count (net income /
EPS). A series already on one basis is left alone, so adjusting an adjusted
series is a no-op.
"""

import functools
import logging
import math
import os
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Count-like series (multiplied by a split's ratio while on the old basis)
# and per-share series (divided by it). Keys are the edgar_history names.
COUNT_KEYS = ('shares_history', 'wavg_basic_history', 'wavg_diluted_history')
PER_SHARE_KEYS = ('eps_basic_history', 'eps_diluted_history')

# Stock dividends of a few percent (ABCB's 1.0077) are not worth restating
# and would only add states.
_MIN_LOG_RATIO = math.log(1.2)
# Transition cost cap (natural log; x12). SEC data carries unit-scale errors
# (CSX's FY2009 weighted count is 0.4M against 381M), and uncapped, one such
# point outweighs a real split's jump and pulls the restatement onto the
# wrong years. Capped, a garbage point costs the same however it is scaled.
_MAX_STEP_COST = 2.5
# A point dated up to this long after a split may still be on the old basis:
# fiscal-year keys normalised to YYYY-12-31 can sit up to a year after the
# period they name (NVDA's FY ending January 2025 is keyed 2025-12-31).
_LOOKAHEAD = timedelta(days=400)
# Filers restate comparatives at most ~5 years back (selected financial
# data), so a point older than this before a split is on the old basis
# unless something strong says otherwise. Leaving it unrestated costs
# _OLD_BASIS_COST per point: without it a real jump of similar size anywhere
# earlier (PANW's x3.4 pre-IPO count in 2011-12) can absorb the split's
# jump, and the series is restated from the wrong point.
_RESTATE_REACH = timedelta(days=5 * 366)
_OLD_BASIS_COST = 0.1


def _to_date(key):
    try:
        if isinstance(key, int):
            return date(key, 12, 31)
        s = str(key)
        if len(s) == 4 and s.isdigit():
            return date(int(s), 12, 31)
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def is_split_ratio(r):
    """True for a share split's ratio: a small fraction p/q (q in 1-5 or 10),
    or its inverse for a reverse split, exactly as yfinance records one.

    yfinance's ``Stock Splits`` column also carries the price-adjustment
    factor of a spin-off (GE 1.281 and 1.253, DHR 1.319, FDX 1.241, A 1.398),
    which changes no share count and must not restate one.
    """
    try:
        x = float(r)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(x) or x <= 0:
        return False
    x = x if x >= 1 else 1 / x
    for q in (1, 2, 3, 4, 5, 10):
        p = round(x * q)
        if p > q and abs(x * q - p) <= 5e-4 * x * q:
            return True
    return False


@functools.lru_cache(maxsize=8192)
def _corporate_actions(path, mtime):
    """``(splits, dividends)`` from one price parquet, each ``[(date, value),
    ...]`` oldest first (raw: split ratios unfiltered, dividends > 0). Cached
    by path and mtime, because the render reads both for every ticker."""
    try:
        import pandas as pd
        df = pd.read_parquet(path)
    except Exception as e:
        logger.debug("splits: unreadable parquet %s (%s)", path, e)
        return (), ()

    def _events(col, keep):
        if col not in df.columns:
            return ()
        s = df[col]
        out = []
        for idx, v in s[keep(s)].items():
            try:
                d = idx.date() if hasattr(idx, 'date') else _to_date(idx)
                r = float(v)
            except (TypeError, ValueError):
                continue
            if d:
                out.append((d, r))
        return tuple(sorted(out))
    return (_events('Stock Splits', lambda s: (s > 0) & (s != 1)),
            _events('Dividends', lambda s: s > 0))


def _actions_for(prices_dir, ticker):
    if not prices_dir or not ticker:
        return (), ()
    path = os.path.join(prices_dir, f"{ticker}.parquet")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return (), ()
    return _corporate_actions(path, mtime)


def split_events_from_prices(prices_dir, ticker):
    """``[(date, ratio), ...]`` oldest first from ``<prices_dir>/<ticker>.parquet``.

    Ratio is new shares per old share (2.0 for 2:1, 0.1 for a 1:10 reverse).
    Empty when there is no file, no ``Stock Splits`` column (a close-only
    write-through) or no split.
    """
    return [(d, r) for d, r in _actions_for(prices_dir, ticker)[0] if is_split_ratio(r)]


def dividend_events_from_prices(prices_dir, ticker, since=None):
    """``[(date, amount), ...]`` oldest first: the ex-dividend dates and
    per-share amounts (split-adjusted, as Yahoo reports them beside its
    adjusted closes), on or after ``since`` when given. Empty without a
    ``Dividends`` column."""
    return [(d, a) for d, a in _actions_for(prices_dir, ticker)[1]
            if since is None or d >= since]


def _choose_factors(points, events):
    """Per-point restatement factor for *points* (``[(date, count)]`` sorted).

    Each point may still need any split whose date (plus _LOOKAHEAD, for
    fiscal-year keys that sit after their period) is after it; a point past
    that is on the post-split basis by construction. A Viterbi pass picks,
    for every point, which of its open splits to apply so that the restated
    series moves least from point to point (sum of |log ratio|, each step
    capped at _MAX_STEP_COST). That
    handles a series where as-filed and restated values interleave, which
    happens because a value's basis follows the date of the filing that last
    reported it, not the period it covers. A tiny per-split penalty breaks
    the tie toward "as filed" when nothing anchors the series after a split;
    beyond _RESTATE_REACH the prior leans the other way (_OLD_BASIS_COST).
    """
    n, k = len(points), len(events)
    if n < 2 or k == 0:
        return [1.0] * n
    # An isolated spike further from BOTH neighbours than the splits still
    # open at it could explain (x5) is a unit-scale error (CMG's 2022-04-25
    # period-end count is 0; AAPL's 2014-03-29 reads 1M), not a basis. Left
    # in, capped steps make it a free place to hide a split's jump, and the
    # "as filed" penalty then settles the tie wrongly; it is set aside and
    # kept as filed.
    lv = [math.log(c) for _, c in points]
    span = [math.log(5) + sum(abs(math.log(r)) for sd, r in events
                              if d <= sd + _LOOKAHEAD)
            for d, _ in points]
    keep = [i for i in range(n)
            if not (0 < i < n - 1
                    and (lv[i] - lv[i - 1]) * (lv[i] - lv[i + 1]) > 0
                    and min(abs(lv[i] - lv[i - 1]),
                            abs(lv[i] - lv[i + 1])) > span[i])]
    if len(keep) < n:
        sub = _choose_factors([points[i] for i in keep], events)
        out = [1.0] * n
        for i, f in zip(keep, sub, strict=True):
            out[i] = f
        return out
    logs = [math.log(r) for _, r in events]
    open_mask = []
    for d, _ in points:
        m = 0
        for j, (sd, _r) in enumerate(events):
            if d <= sd + _LOOKAHEAD:
                m |= 1 << j
        open_mask.append(m)
    if not any(open_mask):
        return [1.0] * n

    def states(m):
        sub = m
        while True:
            yield sub
            if sub == 0:
                return
            sub = (sub - 1) & m

    def shift(s):
        return sum(logs[j] for j in range(k) if s >> j & 1)

    eps = 1e-6
    old_mask = []
    for d, _ in points:
        m = 0
        for j, (sd, _r) in enumerate(events):
            if d < sd - _RESTATE_REACH:
                m |= 1 << j
        old_mask.append(m)

    def prior(i, s):
        return (eps * bin(s).count('1')
                + _OLD_BASIS_COST * bin(old_mask[i] & ~s).count('1'))

    cost = {s: prior(0, s) for s in states(open_mask[0])}
    back = []
    for i in range(1, n):
        nxt, bk = {}, {}
        for s in states(open_mask[i]):
            here = lv[i] + shift(s)
            best_c, best_p = None, None
            for p, c in cost.items():
                cc = c + min(abs(here - (lv[i - 1] + shift(p))), _MAX_STEP_COST)
                if best_c is None or cc < best_c:
                    best_c, best_p = cc, p
            nxt[s] = best_c + prior(i, s)
            bk[s] = best_p
        cost = nxt
        back.append(bk)
    s = min(cost, key=cost.get)
    chosen = [s]
    for bk in reversed(back):
        s = bk[s]
        chosen.append(s)
    chosen.reverse()
    return [math.exp(shift(s)) for s in chosen]


def _count_points(series):
    pts = []
    for k, v in series.items():
        d = _to_date(k)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if d is not None and math.isfinite(fv) and fv > 0:
            pts.append((d, fv, k))
    return sorted(pts, key=lambda p: p[0])


def _implied_count_points(eps, ni):
    """``ni / eps`` at the keys both carry (same sign, non-zero)."""
    by_date = {}
    for k, v in (ni or {}).items():
        d = _to_date(k)
        if d is not None:
            by_date[d.year] = v
    pts = []
    for k, e in eps.items():
        d = _to_date(k)
        try:
            fe, fn = float(e), float(by_date.get(d.year) if d else None)
        except (TypeError, ValueError):
            continue
        if d is None or not (math.isfinite(fe) and math.isfinite(fn)) or fe == 0:
            continue
        c = fn / fe
        if c > 0:
            pts.append((d, c, k))
    return sorted(pts, key=lambda p: p[0])


# Key recording the splits a history has been restated for, as
# ``[["YYYY-MM-DD", ratio], ...]``. The restatement is a best fit, not a
# projection: a second pass over its own output can move a point again (the
# old-basis prior keeps leaning on long pre-split stretches), so a split is
# applied once and remembered. The live run restates; the render and the
# re-score then find the marker and leave the history alone, while a
# snapshot from before this change has no marker and is restated there.
APPLIED_KEY = 'split_adjusted'


def adjust_history(history, events):
    """Return a copy of *history* (an edgar_history-shaped dict) restated
    for *events*, plus ``[(key, n_points_restated)]`` for what changed.

    Only COUNT_KEYS and PER_SHARE_KEYS change (and APPLIED_KEY is set);
    every other series is the same object. Per-share values are judged
    through their implied count (net income / EPS); a point with no usable
    net income stays as filed. Splits already listed under APPLIED_KEY are
    skipped, so a restated history is never restated twice.
    """
    if not history:
        return history, []
    done = {(str(d)[:10], round(float(r), 6))
            for d, r in (history.get(APPLIED_KEY) or [])}
    events = sorted((d, r) for d, r in (events or [])
                    if is_split_ratio(r) and abs(math.log(r)) >= _MIN_LOG_RATIO
                    and (d.isoformat(), round(float(r), 6)) not in done)
    if not events:
        return history, []
    out = dict(history)
    applied = []
    ni = history.get('earnings_history') or {}
    for key in COUNT_KEYS + PER_SHARE_KEYS:
        series = out.get(key)
        if not isinstance(series, dict) or len(series) < 2:
            continue
        per_share = key in PER_SHARE_KEYS
        pts = (_implied_count_points(series, ni) if per_share
               else _count_points(series))
        factors = _choose_factors([(d, c) for d, c, _k in pts], events)
        by_key = {k: f for (_d, _c, k), f in zip(pts, factors, strict=True)
                  if abs(f - 1.0) > 1e-9}
        if not by_key:
            continue
        new = dict(series)
        for k, f in by_key.items():
            new[k] = float(series[k]) / f if per_share else float(series[k]) * f
        out[key] = new
        applied.append((key, len(by_key)))
    out[APPLIED_KEY] = sorted(
        [list(x) for x in done] + [[d.isoformat(), float(r)] for d, r in events])
    return out, applied


def adjust_for_ticker(history, ticker, prices_dir):
    """``adjust_history`` with the ticker's split events from *prices_dir*.

    Returns *history* unchanged when there is no parquet, no split or no
    history; never raises, since a failed restatement must not cost a row.
    """
    if not history or not prices_dir:
        return history
    try:
        adj, applied = adjust_history(
            history, split_events_from_prices(prices_dir, ticker))
    except Exception as e:
        logger.warning("splits: restatement failed for %s (%s); kept as filed",
                       ticker, e)
        return history
    if applied:
        logger.debug("splits: %s restated %s", ticker, applied)
    return adj
