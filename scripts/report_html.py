# scripts/report_html.py
"""HTML report builder — renders the interactive Jinja2 report."""
import os
import json
import hashlib
import logging
import math
import re
import shutil
import jinja2
from datetime import date, datetime, timedelta, timezone
import numpy as np

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.dirname(_HERE))
try:
    from models.narrative import generate_sector_profit_pool_narrative
except Exception:
    generate_sector_profit_pool_narrative = None
from models.data_tab_narrative import generate_data_tab_summaries
from models.profile_verdict import profile_verdict
from models.sector_forces import evaluate_forces
from models.sector_pool import (economic_pool, industry_pools, pool_structure,
                                 sector_pool_history, universe_totals)
from scripts.scoring import gate_metadata
from scripts.config import (CARRY_FORWARD_MAX_PRICE_LAG_BARS,
                            CARRY_FORWARD_STOPPED_GUARD_FLOOR,
                            CARRY_FORWARD_STOPPED_MAX_SHARE,
                            ETF_FV_CARRY_SNAPSHOTS,
                            PHASE1_LOCAL_PRICE_MAX_AGE_DAYS)
from data import etf_holdings
from models.etf_lookthrough import lookthrough as etf_lookthrough, relative_to
from data.price_store import mass_stop, stopped_trading
from data.splits import adjust_for_ticker, dividend_events_from_prices
from data.news_tags import TAG_LABELS
from scripts.safe_json import dumps_for_script

logger = logging.getLogger('report_html')


def _json_default(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import math
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (complex, np.complexfloating)):
        # Last line of defence, deliberately narrow. A complex value here is
        # an upstream arithmetic bug (a negative base raised to a fractional
        # power), and it used to abort the whole render at the END of a
        # 13-hour pipeline: on 2026-09-08 a single HBNC rpe_cagr killed
        # build_html after every expensive phase had already completed.
        # Losing one cell to a null beats losing the report, but only
        # because this WARNING makes the bad value findable — the encoder
        # still raises for every other unhandled type, so genuinely novel
        # breakage is not silently swallowed.
        logger.warning("report payload: complex value %r coerced to null "
                       "(upstream growth-rate guard missing)", obj)
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# The row payload is inlined into index.html, which Cloudflare Pages caps at
# 25 MiB. Written row by row it was 23.3 MiB on 2026-09-25 (2,531 rows x ~320
# keys): each key name repeated per row, padded separators and 17-digit float
# reprs. Packed column-wise it is ~7.8 MiB, with no change to what the page
# sees once templates/report.html's _unpackRows rebuilds the rows.
_PACK_SIG_DIGITS = 10   # far beyond any displayed precision


def _trim_floats(v):
    """Round finite floats to _PACK_SIG_DIGITS significant digits, recursively.
    Ten digits leave every displayed figure (at most ~4 significant decimals)
    unchanged; NaN/inf pass through untouched, as json.dumps writes them."""
    if isinstance(v, float):
        return float(f'{v:.{_PACK_SIG_DIGITS}g}') if math.isfinite(v) else v
    if isinstance(v, dict):
        return {k: _trim_floats(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_trim_floats(x) for x in v]
    return v



def pack_rows(records):
    """Column-wise encoding of a list of dicts, decoded by _unpackRows.

    ``k`` lists every key once (first-appearance order); ``c[i]`` holds key
    i's values for the rows that have it, in row order. Which rows those are
    is ``p[i]`` (present row indices) or ``a[i]`` (absent ones), whichever
    is shorter, or all rows when neither is given — so an absent key stays
    absent and an explicit null stays null.
    """
    keys = list(dict.fromkeys(k for r in records for k in r))
    n = len(records)
    cols, present, absent = [], {}, {}
    for i, k in enumerate(keys):
        have = [j for j, r in enumerate(records) if k in r]
        cols.append([_trim_floats(records[j][k]) for j in have])
        if len(have) < n:
            if len(have) <= n - len(have):
                present[str(i)] = have
            else:
                hs = set(have)
                absent[str(i)] = [j for j in range(n) if j not in hs]
    return {'n': n, 'k': keys, 'c': cols, 'p': present, 'a': absent}


def unpack_rows(packed):
    """Python mirror of the template's _unpackRows (tests, tooling)."""
    n, rows = packed['n'], [dict() for _ in range(packed['n'])]
    for i, k in enumerate(packed['k']):
        col = packed['c'][i]
        if str(i) in packed['p']:
            idx = packed['p'][str(i)]
        else:
            miss = set(packed['a'].get(str(i), ()))
            idx = [j for j in range(n) if j not in miss]
        for j, v in zip(idx, col, strict=True):
            rows[j][k] = v
    return rows


_COMPLEX_REPR = re.compile(r'^\(-?\d[\d.eE+-]*[+-]\d[\d.eE+-]*j\)$')


def _label(path):
    return '.'.join(str(p) for p in path) if path else '<root>'


def _sanitize(obj, _path=()):
    """Recursively replace inf/-inf/NaN (numeric or stringified) with None.

    Some upstream snapshots round-trip numeric infinity through ``str()`` and
    arrive here as the literal string ``"Infinity"``, which silently breaks
    the browser-side popup formatters. Normalise those to ``None`` so the JS
    sees a clean ``null``.

    Complex values get the same treatment, but noisily: they are always an
    upstream arithmetic bug rather than a data quirk, so each one is logged
    at WARNING with the ticker and field that carried it. ``_path`` is the
    breadcrumb used to build that label — a record's own ``ticker`` value
    roots the path when present, so the operator gets ``HBNC.rpe_cagr``
    rather than an anonymous list index. It is carried as a tuple and only
    joined in the warning branch: this walks every scalar in a ~60 MB
    payload, so the common path must not format a string per key.
    """
    import math
    if obj is None:
        return None
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (complex, np.complexfloating)):
        logger.warning("%s: complex value %r coerced to null "
                       "(upstream growth-rate guard missing)",
                       _label(_path), obj)
        return None
    if isinstance(obj, str):
        if obj in ('Infinity', '-Infinity', 'inf', '-inf', 'NaN', 'nan'):
            return None
        # A snapshot written before the JSON writer learned about complex
        # numbers can carry one as its repr, e.g. "(-0.0877+0.2082j)".
        # Cheap prefix/suffix test first: the regex must not run against
        # every string in a 60 MB payload.
        if obj[:1] == '(' and obj[-2:] == 'j)' and _COMPLEX_REPR.match(obj):
            logger.warning("%s: stringified complex value %r coerced to null "
                           "(stale snapshot)", _label(_path), obj)
            return None
        return obj
    if isinstance(obj, dict):
        tk = obj.get('ticker')
        base = (tk,) if isinstance(tk, str) and tk else _path
        return {k: _sanitize(v, base + (k,)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x, _path + (i,)) for i, x in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(_sanitize(x, _path + (i,)) for i, x in enumerate(obj))
    return obj


def fmt_pct(val):
    return f"{val:.1%}" if val is not None else "N/A"

def fmt_num(val, decimals=2):
    return f"{val:,.{decimals}f}" if val is not None else "N/A"

def fmt_dollar(val):
    return f"${val:,.0f}" if val is not None else "N/A"

def fmt_dollar_short(val):
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"${val/1e12:.1f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"

def _data_val(val):
    """Format a value as a data-value attribute for HTML sorting.

    Returns the value as a rounded string for JS-based table sorting,
    or empty string if None.
    """
    return "" if val is None else str(round(float(val), 6))


def _rating_style(rating):
    colors = {'BUY': '#1a9850', 'LEAN BUY': '#74c476', 'HOLD': '#fd8d3c', 'PASS': '#de2d26'}
    c = colors.get(rating, '#888')
    return f' style="color:{c};font-weight:700;text-align:center"'

_RATING_VAL = {'BUY': 3, 'LEAN BUY': 2, 'HOLD': 1, 'PASS': 0}

def _rating_num(rating):
    return _RATING_VAL.get(rating, -1)


def _rating_delta(prev, cur):
    """Signed step change between two ratings (+ = upgrade, - = downgrade).

    Returns None when either side is missing or unrecognized (e.g. a ticker
    that is new since the prior run, or a snapshot without ratings), so the
    column can render 'NEW'/'—' and sort those rows apart from real moves.
    """
    if prev in _RATING_VAL and cur in _RATING_VAL:
        return _RATING_VAL[cur] - _RATING_VAL[prev]
    return None


_PREV_DRIVER_KEYS = (
    'rating', 'rating_raw', '_rating_cap', '_rating_cap_reasons',
    '_composite_score', '_data_coverage_score', 'mos',
    '_score_valuation', '_score_quality', '_score_moat',
    '_score_growth', '_score_ownership',
)


def _load_prev_ratings(out_dir, run_date, max_lookback=7, extra_keys=()):
    """Load each ticker's last-known rating (plus its score drivers) for the
    rate-change column and the popup's "why the rating changed" explanation.

    Scans ``out_dir`` for ``results_YYYY-MM-DD.json`` snapshots dated strictly
    before ``run_date`` (ISO date strings sort lexicographically, so a plain
    string compare is correct) and walks them newest-first (from the DuckDB
    snapshot store when it holds that window, else by parsing the files). For
    every ticker it
    records the rating from the *most recent* snapshot that contains it — i.e.
    "last known rating", not "rating in exactly the prior file". This makes the
    rating-change baseline (the ticker page's "Rating upgraded: X → Y" line
    and the Upgrades/Downgrades filters) resilient to a single degraded run: a
    ticker that transiently dropped out of the immediately-prior snapshot (e.g.
    a yfinance/SEC fetch timeout) still compares against its last real rating.

    The look-back is bounded to the ``max_lookback`` most recent prior snapshots
    (~1-2 weeks) so a genuinely delisted or long-absent ticker isn't resurrected
    from stale history.

    Returns ``(prev_date_str, {ticker: drivers})`` where ``drivers`` is a dict
    of ``_PREV_DRIVER_KEYS`` plus any ``extra_keys`` (per-gate raw values and
    scores) taken from the same snapshot record that supplied the rating, and
    ``prev_date_str`` is the date of the most recent prior snapshot (the
    primary baseline, still the reference for the large majority of tickers).
    Returns ``(None, {})`` when there is no earlier snapshot at all — the
    column then degrades to 'NEW'/N/A for every row rather than raising.
    """
    from data.snapshot_store import list_snapshot_files, read_snapshot
    try:
        cur = run_date.isoformat() if run_date is not None else None
    except Exception:
        cur = None
    # Collect all prior snapshots, newest date first.  list_snapshot_files
    # owns the canonical-name rule and also finds the .json.gz archive form.
    dated = [(d, p) for d, p in list_snapshot_files(out_dir)
             if cur is None or d < cur]
    if not dated:
        return None, {}
    dated.sort(key=lambda t: t[0], reverse=True)
    primary_date = dated[0][0]
    from_store = _prev_ratings_from_store(
        out_dir, cur, [d for d, _ in dated[:max_lookback]], extra_keys)
    if from_store is not None:
        return from_store
    out = {}
    filled_via_fallback = 0
    for d, p in dated[:max_lookback]:
        try:
            snap = read_snapshot(p)
        except Exception as e:
            print(f"[report_html] prior-rating load failed ({p}): {e}")
            continue
        recs = snap.get('results') if (isinstance(snap, dict) and 'results' in snap) else snap
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            tk, rt = rec.get('ticker'), rec.get('rating')
            # Newest-first walk: only fill a ticker the first time we see it, so
            # the most recent snapshot that rated it wins.
            if tk and rt and tk not in out:
                drv = {k: rec.get(k) for k in _PREV_DRIVER_KEYS}
                for k in extra_keys:
                    drv[k] = rec.get(k)
                out[tk] = drv
                if d != primary_date:
                    filled_via_fallback += 1
    if filled_via_fallback:
        print(f"[report_html] rate-change: {filled_via_fallback} ticker(s) "
              f"baselined via fallback look-back (missing from {primary_date})")
    return primary_date, out

def _open_snapshot_store(out_dir):
    """Read-only handle on out_dir/snapshots.duckdb, or None when absent."""
    try:
        from data.snapshot_store import SnapshotStore
        return SnapshotStore.for_results_dir(out_dir)
    except Exception as e:  # missing duckdb, locked/corrupt file: use JSON
        print(f"[report_html] snapshot store unavailable, parsing JSON: {e}")
        return None


def _prev_ratings_from_store(out_dir, cur, window_dates, extra_keys):
    """_load_prev_ratings via the DuckDB snapshot store.

    *window_dates* are the (newest-first) snapshot files the JSON path would
    parse; the store is used only when it holds exactly that window, so a
    store that lags the files (a snapshot written but not yet ingested) can
    never change the baseline.  Returns ``(primary_date, drivers)`` or None
    to fall back to parsing the JSON files.
    """
    if cur is None or not window_dates:
        return None
    store = _open_snapshot_store(out_dir)
    if store is None:
        return None
    try:
        with store:
            have = store.dates(before=cur)[-len(window_dates):]
            if have != sorted(window_dates):
                return None
            cols = list(_PREV_DRIVER_KEYS) + [k for k in extra_keys
                                              if k not in _PREV_DRIVER_KEYS]
            primary, out, n_fallback = store.last_known_rows(
                cur, cols, max_lookback=len(window_dates))
    except Exception as e:
        print(f"[report_html] snapshot store read failed, parsing JSON: {e}")
        return None
    if n_fallback:
        print(f"[report_html] rate-change: {n_fallback} ticker(s) "
              f"baselined via fallback look-back (missing from {primary})")
    print("[report_html] rate-change baseline read from snapshot store")
    return primary, out


def _rating_history_from_store(out_dir, cur, file_dates):
    """_load_rating_history via the snapshot store, or None to fall back.
    Used only when the store holds exactly the snapshot dates the JSON path
    would scan (every file dated before *cur*)."""
    store = _open_snapshot_store(out_dir)
    if store is None:
        return None
    try:
        with store:
            have = store.dates(before=cur)
            if getattr(store, 'authoritative', False):
                # The database holds the whole published history, while the
                # cloud stages only the newest few files: it must cover them,
                # not equal them (plan item R3).
                if not set(file_dates) <= set(have):
                    return None
            elif have != sorted(file_dates):
                return None
            hist = store.rating_history(before=cur)
    except Exception as e:
        print(f"[report_html] snapshot store read failed, parsing JSON: {e}")
        return None
    print(f"[report_html] rating history read from snapshot store ({len(hist)} tickers)")
    return hist


# Composite-score floors for each rating band (see rating_from_composite in
# scripts/scoring.py — 57/39/25 are the calibrated defaults). Keyed by the
# _RATING_VAL rank of the band the threshold admits into.
_RATING_THRESHOLD_BY_RANK = {3: 57, 2: 39, 1: 25}
_RATING_BY_RANK = {v: k for k, v in _RATING_VAL.items()}


def _fmt_gate_value(v, fmt):
    """Format a raw gate value with the gate's display format (mirrors the
    template's _fmtGate so the explanation reads like the gate table)."""
    if v is None:
        return 'N/A'
    try:
        if fmt == 'pct1':
            return f"{v * 100:.1f}%"
        if fmt == 'ratio':
            return f"{v:.2f}×"
        if fmt == 'int':
            return f"{round(v)}"
        return f"{v:.2f}"
    except (TypeError, ValueError):
        return str(v)


def _explain_rating_change(prev, cur, gate_meta, score_terms=False):
    """Build the popup's "why the rating changed" bullet list.

    With *score_terms* the bullets name score lines and bands instead of
    ratings ("crossing above the 57 line"): what the report and the app show
    (``rating_chg_why``). The default wording is the portfolio digest's,
    which prints these bullets verbatim and stays as it was.

    ``prev`` is the driver dict captured by _load_prev_ratings for the same
    ticker (None for new tickers); ``cur`` is the full current record. Returns
    a list of plain-English sentences, or None when the rating didn't change.

    The explanation mirrors how the rating is actually produced (see
    scripts/scoring.py): a rating cap firing/releasing, the composite score
    crossing a band threshold, then the category and gate-level score moves
    that drove the composite.
    """
    if not prev:
        return None
    p_rt, c_rt = prev.get('rating'), cur.get('rating')
    delta = _rating_delta(p_rt, c_rt)
    if not delta:
        return None
    up = delta > 0
    why = []
    st = score_terms

    def band(r):
        """A rating as its score band, for the score-terms wording."""
        th = {3: _RATING_THRESHOLD_BY_RANK.get(3), 2: _RATING_THRESHOLD_BY_RANK.get(2),
              1: _RATING_THRESHOLD_BY_RANK.get(1)}
        rk = _RATING_VAL.get(r)
        if rk is None:
            return 'no'
        if rk == 3:
            return f'the {th[3]}+'
        if rk == 0:
            return f'the below-{th[1]}'
        return f'the {th[rk]}–{th[rk + 1]}'

    # --- Rating caps: a cap firing (downgrade) or releasing (upgrade) can
    #     change the rating with no composite move at all, so it leads. ---
    p_raw, c_raw = prev.get('rating_raw'), cur.get('rating_raw')
    p_reasons = [str(x) for x in (prev.get('_rating_cap_reasons') or [])]
    c_reasons = [str(x) for x in (cur.get('_rating_cap_reasons') or [])]
    p_capped = p_raw is not None and p_rt is not None and p_rt != p_raw
    c_capped = c_raw is not None and c_rt is not None and c_rt != c_raw
    if not up and c_capped:
        new_r = [x for x in c_reasons if x not in p_reasons]
        if new_r:
            why.append(('A flag lowered its band this run: ' if st else
                        'A rating cap fired this run: ')
                       + '; '.join(new_r)
                       + (f'. On the score alone it would sit in {band(c_raw)} band.' if st
                          else f'. On the composite alone it would rate {c_raw}.'))
        elif not p_capped:
            why.append(('A flag is holding its band down: ' if st else
                        'The rating is being held down by a cap: ')
                       + ('; '.join(c_reasons) or ('see the flags' if st else 'see rating cap'))
                       + (f'. On the score alone it would sit in {band(c_raw)} band.' if st
                          else f'. On the composite alone it would rate {c_raw}.'))
    if up and p_capped and not c_capped:
        gone = [x for x in p_reasons if x not in c_reasons]
        why.append(("The prior run's band-lowering flag no longer applies" if st
                    else "The prior run's rating cap no longer applies")
                   + (f" ({'; '.join(gone)})" if gone else '')
                   + '.')

    # --- Composite score move, with the band threshold(s) it crossed. ---
    p_cs, c_cs = prev.get('_composite_score'), cur.get('_composite_score')
    if p_cs is not None and c_cs is not None:
        # One decimal when integer rounding would collapse a real move into
        # "57 → 57, crossing the BUY (57) threshold".
        dp = 1 if round(p_cs) == round(c_cs) else 0
        s = f'Composite score moved {p_cs:.{dp}f} → {c_cs:.{dp}f}'
        if (p_raw in _RATING_VAL and c_raw in _RATING_VAL and p_raw != c_raw):
            lo = min(_RATING_VAL[p_raw], _RATING_VAL[c_raw])
            hi = max(_RATING_VAL[p_raw], _RATING_VAL[c_raw])
            if st:
                crossed = [f'{_RATING_THRESHOLD_BY_RANK[rk]}' for rk in range(hi, lo, -1)]
                s += (', crossing ' + ('above' if up else 'below') + ' the '
                      + ' and '.join(crossed)
                      + (' lines' if len(crossed) > 1 else ' line'))
            else:
                crossed = [f'{_RATING_BY_RANK[rk]} ({_RATING_THRESHOLD_BY_RANK[rk]})'
                           for rk in range(hi, lo, -1)]
                s += (', crossing ' + ('above' if up else 'below') + ' the '
                      + ' and '.join(crossed)
                      + (' thresholds' if len(crossed) > 1 else ' threshold'))
        elif p_raw == c_raw and p_raw is not None:
            s += (' (its band on the score alone did not change)' if st
                  else ' (the composite-only rating band did not change)')
        why.append(s + '.')
    elif p_cs is None and c_cs is not None:
        why.append('The prior run had no composite score (insufficient scoring '
                   'data); this run scored it '
                   + (f'{c_cs:.0f}.' if st else f'{c_cs:.0f}, rating {c_rt}.'))
    elif p_cs is not None and c_cs is None:
        why.append('This run produced no composite score (insufficient scoring '
                   + ('data), so it has no band this run.' if st else
                      'data), so the rating fell back to the cap/default path.'))

    # --- Category-level attribution: which score groups moved the composite. ---
    cat_moves = []
    for c in gate_meta.get('categories', []):
        pv, cv = prev.get(c['scoreKey']), cur.get(c['scoreKey'])
        if pv is None or cv is None:
            continue
        d = cv - pv
        if abs(d) < 3:
            continue
        cat_moves.append((abs(d) * (c.get('weight') or 0), c['name'], pv, cv, d))
    cat_moves.sort(key=lambda t: -t[0])
    if cat_moves:
        why.append('Biggest category moves: ' + ', '.join(
            f'{name} {pv:.0f} → {cv:.0f} ({d:+.0f})'
            for _, name, pv, cv, d in cat_moves[:3]) + '.')

    # --- Gate-level attribution: the individual metrics behind those moves. ---
    gate_moves = []
    for g in gate_meta.get('gates', []):
        ps, cs = prev.get(g['scoreKey']), cur.get(g['scoreKey'])
        if ps is None or cs is None:
            continue
        d = cs - ps
        if abs(d) < 15:
            continue
        gate_moves.append((abs(d) * (g.get('weight') or 0), g, ps, cs,
                           prev.get(g['key']), cur.get(g['key'])))
    gate_moves.sort(key=lambda t: -t[0])
    if gate_moves:
        why.append('Largest metric swings: ' + '; '.join(
            f"{g['label']} {_fmt_gate_value(pv, g['fmt'])} → "
            f"{_fmt_gate_value(cv, g['fmt'])} (score {ps:.0f} → {cs:.0f})"
            for _, g, ps, cs, pv, cv in gate_moves[:3]) + '.')

    if not why:
        why.append('Underlying inputs shifted between runs, but no single '
                   'score driver moved enough to attribute the change '
                   + ('(likely small moves straddling a band line).' if st else
                      '(likely small moves straddling a rating threshold).'))
    return why


def _load_rating_history(out_dir, run_date, cache_name='rating_history.json'):
    """Per-ticker rating change-points across prior snapshots, for the popup's
    "BUY since ..." context line.

    Returns ``{ticker: [[date, rating], ...]}`` — change-points only (the first
    observation plus every transition), ascending by date, filtered to dates
    strictly before ``run_date`` so re-rendering an old snapshot never sees
    "future" history.

    When ``out_dir/snapshots.duckdb`` (data/snapshot_store.py) holds every
    prior snapshot, the change-points come from one window-function query.
    Otherwise: parsing every historical results_*.json (~66MB each) on every
    render would be minutes of work, so an incremental cache
    (``out_dir/rating_history.json``)
    stores the accumulated change-points plus the last snapshot date scanned;
    each render only parses snapshots newer than that. Only strictly-newer dates
    are appended, so a late-backfilled older snapshot can't corrupt ordering;
    folding one in takes a full rebuild. A missing or corrupt cache rebuilds
    over the snapshots in *out_dir* only, which in the cloud routine is just
    the staged days: rebuild over the whole archive with
    scripts/rebuild_rating_history.py (what run.sh does when the archive has
    no cache) — see _advance_rating_history_cache for why scanning stops at an
    unreadable day.

    The cache advances even when a store answers: with the database backend
    it is the independent record the nightly parity check compares the
    database against (scripts/db_night_check.py), and the fallback when the
    database misses a day. Left frozen, that check would compare the same
    day forever, and a fallback would lose every change point in the gap.
    """
    from data.snapshot_store import list_snapshot_files
    try:
        cur = run_date.isoformat() if run_date is not None else None
    except Exception:
        cur = None
    dated = list_snapshot_files(out_dir)
    if not dated:
        return {}
    hist = _advance_rating_history_cache(out_dir, dated, cur, cache_name)
    from_store = _rating_history_from_store(
        out_dir, cur, [d for d, _ in dated if cur is None or d < cur])
    if from_store is not None:
        return from_store
    if cur is None:
        return hist
    # Exclude entries on/after the render date (rescoring an older snapshot).
    return {tk: [cp for cp in seq if cp[0] < cur]
            for tk, seq in hist.items()}


def _advance_rating_history_cache(out_dir, dated, cur, cache_name='rating_history.json'):
    """Fold every snapshot in *dated* newer than the cache's ``last_scanned``
    (and before *cur*) into ``out_dir/cache_name``; returns the history.

    Scanning is strictly in date order and **stops at the first snapshot that
    cannot be read**. Only strictly-newer dates are ever folded in, so
    skipping an unreadable day and scanning a later one would move
    ``last_scanned`` past it for good: the day would be missing from the
    history forever, with nothing to say so. Stopping leaves it first in line
    for the next render instead, and a day that never becomes readable holds
    the cache still, which the nightly parity check (07e) reports as stale.
    """
    from data.snapshot_store import read_snapshot
    cache_path = os.path.join(out_dir, cache_name)
    hist, last_scanned = {}, None
    try:
        with open(cache_path, encoding='utf-8') as f:
            c = json.load(f)
        if isinstance(c, dict) and isinstance(c.get('hist'), dict):
            hist = c['hist']
            last_scanned = c.get('last_scanned')
    except Exception:
        hist, last_scanned = {}, None
    # Never scan the render date's own snapshot: during a rescore it is
    # rewritten *after* this runs, so caching it here would freeze pre-rescore
    # ratings. The next render picks it up once it is final.
    todo = [(d, p) for d, p in dated
            if (last_scanned is None or d > last_scanned)
            and (cur is None or d < cur)]
    scanned = 0
    for d, p in todo:
        try:
            snap = read_snapshot(p)
        except Exception as e:
            print(f"[report_html] rating-history load failed ({p}): {e}; "
                  f"cache stops at {last_scanned} until it can be read")
            break
        recs = snap.get('results') if (isinstance(snap, dict) and 'results' in snap) else snap
        if not isinstance(recs, list):
            print(f"[report_html] rating-history load failed ({p}): no results list; "
                  f"cache stops at {last_scanned} until it can be read")
            break
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            tk, rt = rec.get('ticker'), rec.get('rating')
            if not tk or not rt:
                continue
            seq = hist.setdefault(tk, [])
            if not seq or seq[-1][1] != rt:
                seq.append([d, rt])
        last_scanned = d
        scanned += 1
    if scanned:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'last_scanned': last_scanned, 'hist': hist}, f)
            print(f"[report_html] rating-history cache: +{scanned} snapshot(s), "
                  f"{len(hist)} tickers, through {last_scanned}")
        except Exception as e:
            print(f"[warn] rating-history cache write failed: {e}")
    return hist


def _load_russell2000():
    """Ticker set for the Russell 2000 quick filter.

    Read from data/russell2000_tickers.txt (dash-style tickers, one per
    line; refresh via scripts/update_russell2000.py). Loaded at render
    time so a rescore_and_render picks up a refreshed list without a
    live run. Missing file degrades to an empty set — the filter then
    simply matches nothing.
    """
    path = os.path.join(_HERE, '..', 'data', 'russell2000_tickers.txt')
    try:
        with open(path, encoding='utf-8') as f:
            return {ln.strip().upper() for ln in f
                    if ln.strip() and not ln.startswith('#')}
    except OSError:
        print("[report_html] russell2000_tickers.txt not found — "
              "Russell 2000 quick filter will match nothing")
        return set()


def _load_portfolio_payload(rows, out_dir=None, run_date=None, prices_dir=None,
                            rating_hist=None):
    """Portfolio groupings for the report: ``(payload, {ticker: [ids]})``.

    The payload (inline ``PORTFOLIOS``) carries the definitions with colors
    filled in and their revision hash, which the report's local-edit sync
    compares against. Membership is resolved here against the raw rows so
    the unedited published set never depends on the browser's evaluator.
    ``PORTFOLIOS_FILE`` overrides the path (tests, alternate books). A
    missing or invalid file degrades to no portfolios; it never fails a
    render.
    """
    from models import portfolio_groups as pg
    path = os.environ.get('PORTFOLIOS_FILE') or pg.DEFAULT_PORTFOLIOS_PATH
    try:
        doc = pg.load_portfolios(path)
    except (OSError, ValueError) as e:
        logger.warning("portfolios: %s unusable (%s); rendering without portfolios",
                       path, e)
        return {'rev': None, 'portfolios': [], 'error': str(e)}, {}
    pfs = pg.with_colors(doc['portfolios'])
    resolved = pg.resolve_all(pfs, rows)
    index = {}
    for p in pfs:
        for t in resolved[p['id']]['members']:
            index.setdefault(t, []).append(p['id'])
        p['missing'] = resolved[p['id']]['missing']
    payload = {'rev': pg.revision(doc), 'portfolios': pfs,
               'prev_date': None, 'changes': [], 'events': [], 'systemic': None}
    # Change alerts vs the prior run, classified Action / Watch / FYI by
    # models.portfolio_groups.classify_changes. Per-ticker entries cover the
    # whole universe so portfolios edited in the browser get them too (the
    # client applies each portfolio's alert mode and adds earnings dates);
    # joined/left events need the prior rows' rule columns and so exist only
    # for the published definitions. "Why" bullets are not shipped: the
    # client already has them as DATA.rating_chg_why. Best-effort: a failure
    # here costs the alerts, never the render.
    prev_day, prev = None, []
    if out_dir and pfs:
        try:
            from scripts.portfolios import drop_stopped, prior_rows
            day = (run_date or date.today()).isoformat()
            prev_day, prev = prior_rows(out_dir, day, pg.rule_columns(pfs))
            if prev_day:
                # Same rows the nightly alerts classify: neither day's
                # stopped names, so a frozen quote never reads as a signal.
                cur, prev_live, stopped = drop_stopped(
                    rows, prev, day if run_date else None, prev_day, prices_dir)
                by_tk, prev_by_tk = pg.rows_by_ticker(cur), pg.rows_by_ticker(prev_live)
                payload['prev_date'] = prev_day
                # The run before the prior one confirms buy-below crossings;
                # the most sensitive portfolio thresholds set the floors and
                # each portfolio keeps its own in the browser (_pfPasses).
                _, prev2 = prior_rows(out_dir, prev_day, ['ticker', 'mos', 'price',
                                                          'company_name', 'sector'])
                sd, fj = pg.alert_floors(pfs)
                entries, stats = pg.classify_changes(by_tk, prev_by_tk, day,
                                                     history=rating_hist or {},
                                                     score_drop=sd, fv_jump=fj,
                                                     prev2_by_tk=pg.rows_by_ticker(prev2))
                payload['changes'] = entries
                payload['thresholds'] = {'score_drop': pg.SCORE_DROP_PTS, 'fv_jump': pg.FV_JUMP}
                payload['systemic'] = dict(stats, message=pg.systemic_message(stats),
                                             display=pg.systemic_display(stats))
                payload['events'] = pg.membership_events(pfs, by_tk, prev_by_tk, day,
                                                         stopped, systemic=stats)
        except Exception as e:
            logger.warning("portfolios: change alerts unavailable (%s)", e)
    if out_dir and pfs:
        payload['nav'] = _portfolio_nav(pfs, rows, prev, out_dir, run_date, prices_dir)
    if pfs:
        payload['analytics'] = _portfolio_analytics(pfs, rows, out_dir, run_date, prices_dir)
    if pfs:
        print(f"[report_html] portfolios: {len(pfs)} "
              f"({sum(len(v) for v in index.values())} memberships, "
              f"{len(payload['changes'])} change entries, {len(payload['events'])} membership "
              f"events vs {payload['prev_date']}"
              f"{'; SYSTEMIC ' + payload['systemic']['cause'] if (payload['systemic'] or {}).get('flood') else ''})")
    return payload, index


def _portfolio_nav(pfs, rows, prev, out_dir, run_date, prices_dir):
    """Advance the NAV ledger (output/portfolio_nav.json) to this run and
    return the report's compact series.

    Like the rating-history cache, the render keeps the ledger current. It
    only writes when *run_date* is explicit, prices are available, and
    ``results_<run_date>`` sits in *out_dir* — so re-rendering someone
    else's rows (backfill_edgar_hist, a test) can never plant a point, and
    re-rendering the same day replaces that day's point.
    A ``held`` portfolio's book-weighted series goes to the private ledger
    beside the private book, never into this payload (the report is public).
    """
    from data import portfolio_nav as pn
    from data.snapshot_store import list_snapshot_files
    from models import portfolio_positions as pp
    path = pn.ledger_path(out_dir)
    try:
        led = pn.load_ledger(path)
        day = run_date.isoformat() if run_date else None
        if day and prices_dir and os.path.isdir(prices_dir) and \
                any(d == day for d, _ in list_snapshot_files(out_dir)):
            books, private, ppath = {}, None, None
            if any(p.get('weighting') == 'held' for p in pfs):
                try:
                    books = pp.load_books()
                    ppath = pn.private_ledger_path()
                    private = pn.load_ledger(ppath)
                except (OSError, ValueError) as e:
                    logger.warning("portfolios: private book unusable (%s); "
                                   "held portfolios stay equal-weighted", e)
            if pn.update(led, pfs, day, rows, pn.prices_closes_fn(prices_dir), prev_rows=prev,
                         books=books, private=private):
                pn.save_ledger(path, led)
                if private is not None and private['portfolios']:
                    pn.save_ledger(ppath, private, private=True)
        return pn.payload(led, [p['id'] for p in pfs])
    except Exception as e:
        logger.warning("portfolios: NAV ledger not updated (%s)", e)
        return None


def _portfolio_analytics(pfs, rows, out_dir, run_date, prices_dir):
    """Look-through valuation, ex-ante risk and NAV attribution per portfolio
    (``models.portfolio_analytics``) for the published definitions. Uses only
    public weights (a ``held`` portfolio is equal-weighted here). Best-effort:
    a failure costs these panels, never the render."""
    from data import portfolio_nav as pn
    from models import portfolio_analytics as pa
    try:
        led = pn.load_ledger(pn.ledger_path(out_dir)) if out_dir else None
        have_px = bool(prices_dir and os.path.isdir(prices_dir))
        return pa.analyze(pfs, rows, led, prices_dir if have_px else None,
                          pn.prices_closes_fn(prices_dir) if have_px else None,
                          asof=run_date.isoformat() if run_date else None)
    except Exception as e:
        logger.warning("portfolios: analytics unavailable (%s)", e)
        return None


# Every sidecar is written compact. json.dump's default separators put a
# space after each comma, which on prices.json alone (11.6M array elements)
# wasted 11.6 MB of the 100 MB Pages budget.
_COMPACT = (',', ':')


def _summary_stats(rows):
    """Header summary counts for the report banner."""
    spread_vals = [r['spread'] for r in rows if r.get('spread') is not None]
    avg_spread = sum(spread_vals) / len(spread_vals) if spread_vals else 0
    qualifying_with_mos = sum(1 for r in rows if r.get('mos') is not None and r['mos'] > 0)
    buy_count = sum(1 for r in rows if r.get('rating') == 'BUY')
    lean_buy_count = sum(1 for r in rows if r.get('rating') == 'LEAN BUY')
    return avg_spread, qualifying_with_mos, buy_count, lean_buy_count


def _load_score_history(out_dir, run_date):
    """The re-scored composite history (``scripts/score_history.py``) for the
    render, or an empty series when its cache is missing.

    ``stale`` is True when the cache was built by another scoring model than
    the one this checkout scores with; ``score_prev`` then falls back to the
    prior snapshot's stored composite (``_score_fields``).
    """
    from scripts import score_history as sh
    empty = {'dates': [], 'hist': {}, 'prev_date': None, 'prev': {}, 'stale': True}
    if run_date is None:
        return empty
    try:
        from scripts.param_set import scoring_params_hash
        cache = sh.load_cache(os.path.join(out_dir, sh.CACHE_NAME))
        if not cache.get('scores'):
            return empty
        return sh.series(cache, run_date, scoring_params_hash())
    except Exception as e:
        logger.warning("score history unavailable (%s)", e)
        return empty


def _score_fields(r, prev_drivers, score_series):
    """``score_prev`` / ``score_chg`` / ``score_prev_basis`` / ``score_hist``
    for one row.

    The prior score comes from the re-scored history (same model as tonight,
    so a model change never reads as a move) unless that history is stale
    or lacks the ticker; then from the prior snapshot's stored composite,
    marked ``score_prev_basis: 'stored'`` so the page can say a move may be
    the model's. ``score_hist`` is aligned to ``SCORE_HIST.dates``.
    """
    tk = r.get('ticker')
    cur = r.get('_composite_score')
    ss = score_series or {}
    prev, basis = None, None
    if not ss.get('stale') and tk in (ss.get('prev') or {}):
        prev, basis = ss['prev'][tk], 'rescored'
    else:
        _p = (prev_drivers or {}).get('_composite_score')
        if isinstance(_p, (int, float)) and _p == _p:
            prev, basis = round(float(_p), 1), 'stored'
    chg = None
    if prev is not None and isinstance(cur, (int, float)) and cur == cur:
        chg = round(float(cur) - prev, 1)
    hist = None if ss.get('stale') else (ss.get('hist') or {}).get(tk)
    return {'score_prev': prev, 'score_chg': chg,
            'score_prev_basis': basis if prev is not None else None,
            'score_hist': hist}


def _row_context(r, gate_meta_obj, _r2000, _prev_ratings, _rating_hist,
                 _score_series=None):
    """Per-row payload dict consumed by the template's inline DATA blob."""
    return {
        **_score_fields(r, _prev_ratings.get(r.get('ticker')), _score_series),
        'ticker': r['ticker'],
        'roic': r.get('roic'), 'wacc': r.get('wacc'), 'spread': r.get('spread'),
        'dcf_fv': r.get('dcf_fv'), 'price': r.get('price'), 'mos': r.get('mos'),
        # Pre-blend DCF: the Summary PDF's model-convergence check compares it
        # with EPV/RIM/DDM, as scoring's fv_dispersion does.
        '_dcf_fv_preblend': r.get('_dcf_fv_preblend'),
        # The FV the MoS was actually computed against (may be a blend of
        # models) — the popup banner shows this so FV and MoS never disagree.
        '_fv_effective': r.get('_fv_effective'),
        '_fv_source': r.get('_fv_source'),
        'piotroski': r.get('piotroski'),
        'pe': r.get('pe'), 'ev_ebitda': r.get('ev_ebitda'),
        'rating': r.get('rating'), 'rating_raw': r.get('rating_raw'),
        # Rate change vs the prior run: prior rating string + signed step
        # delta (+ upgrade / - downgrade / 0 unchanged / None if new-or-missing),
        # plus the "why it changed" bullets for the popup (None unless changed).
        'rating_prev': (_prev_ratings.get(r.get('ticker')) or {}).get('rating'),
        'rating_chg': _rating_delta(
            (_prev_ratings.get(r.get('ticker')) or {}).get('rating'),
            r.get('rating')),
        'rating_chg_why': _explain_rating_change(
            _prev_ratings.get(r.get('ticker')), r, gate_meta_obj, score_terms=True),
        # Rating change-points (last 12) for "BUY since ..." in the popup.
        'rating_hist': (_rating_hist.get(r.get('ticker')) or [])[-12:],
        '_rating_cap': r.get('_rating_cap'),
        '_rating_cap_reasons': r.get('_rating_cap_reasons', []),
        'analyst_rec': r.get('analyst_rec'),
        'company_name': r.get('company_name', ''),
        # Other tickers of this issuer folded into this row (data/issuers.py):
        # the portfolio evaluator resolves a definition naming one of them.
        'listing_aliases': r.get('listing_aliases'),
        'description': (r.get('description') or '')[:200],
        'sector': r.get('sector'),
        'industry': r.get('industry'),
        'country': r.get('country'),
        'ceo': r.get('ceo'),
        'description_full': r.get('description') or '',
        'ceo_bio': r.get('ceo_bio') or '',
        'founder_led': r.get('founder_led', False),
        # Russell 2000 membership (see _load_russell2000). Tickers in the
        # constituent file are dash-style, matching the model's own tickers.
        'r2000': (r.get('ticker') or '').upper() in _r2000,
        'fcf': r.get('fcf'),
        'fcf_margin': r.get('fcf_margin'),
        'sbc_pct_rev': r.get('sbc_pct_rev'),
        'mcap': r.get('mcap'),
        'roic_by_year': r.get('roic_by_year'),
        'roic_cv': r.get('roic_cv'),
        'gross_margin': r.get('gross_margin'),
        'er': r.get('er'),
        're_method': r.get('re_method'),
        'beta_raw': r.get('beta_raw'),
        'beta_adjusted': r.get('beta_adjusted'),
        'beta_r2': r.get('beta_r2'),
        'beta_se': r.get('beta_se'),
        'beta_n_obs': r.get('beta_n_obs'),
        'beta_r2_class': r.get('beta_r2_class'),
        'beta_shrink_weight': r.get('beta_shrink_weight'),
        'beta_source': r.get('beta_source'),
        'beta_warnings': r.get('beta_warnings'),
        'dcf_sens_range': list(r['dcf_sens_range']) if r.get('dcf_sens_range') else None,
        'fcf_growth': r.get('fcf_growth'),
        'pfcf': r.get('pfcf'),
        'pb': r.get('pb'),
        'tangible_book_per_share': r.get('tangible_book_per_share'),
        'nav_fv': r.get('nav_fv'),
        'nav_mos': r.get('nav_mos'),
        'p_tbv': r.get('p_tbv'),
        'num_analysts': r.get('num_analysts'),
        'target_mean': r.get('target_mean'),
        'target_high': r.get('target_high'),
        'target_low': r.get('target_low'),
        'analyst_ltg': r.get('analyst_ltg'),
        # Forward-looking (live-run only; absent on older snapshots and the
        # popup hides the affected rows/chips)
        'earnings_next_date': r.get('earnings_next_date'),
        'earnings_date_est': r.get('earnings_date_est'),
        'cash_conv': r.get('cash_conv'),
        'accruals': r.get('accruals'),
        'rev_cagr': r.get('rev_cagr'),
        'rev_cagr_5y': r.get('rev_cagr_5y'),
        'rev_cagr_10y': r.get('rev_cagr_10y'),
        'int_cov': r.get('int_cov'),
        'nd_ebitda': r.get('nd_ebitda'),
        'de': r.get('de'),
        'total_debt': r.get('total_debt'),
        'net_debt': r.get('net_debt'),
        'cash': r.get('cash'),
        'total_liabilities': r.get('total_liabilities'),
        'cr': r.get('cr'),
        'roe': r.get('roe'),
        'roa': r.get('roa'),
        # FDIC call-report enrichment (Financial Services). See
        # data/fdic_client.py + scripts/enrich_fdic.py for the source.
        'fdic_cert': r.get('fdic_cert'),
        'fdic_repdte': r.get('fdic_repdte'),
        'nim': r.get('nim'),
        'efficiency_ratio': r.get('efficiency_ratio'),
        'cet1_ratio': r.get('cet1_ratio'),
        'npl_ratio': r.get('npl_ratio'),
        # REIT proxies (Real Estate). See data/reit_metrics.py +
        # scripts/enrich_reit.py. ffo_growth_5y is the 5-yr CAGR of
        # operating cash flow; affo_margin is (CFO - capex) / revenue
        # latest year — both approximate the NAREIT FFO/AFFO concepts.
        'ffo_growth_5y': r.get('ffo_growth_5y'),
        'affo_margin': r.get('affo_margin'),
        # Phase 3 — SEC XBRL line items for Tech / Healthcare / Comm.
        # See scripts/enrich_xbrl.py. The *_xbrl variants are the
        # authoritative XBRL-derived versions of fields that yfinance
        # leaves sparse.
        'rd_intensity_xbrl': r.get('rd_intensity_xbrl'),
        'sbc_pct_rev_xbrl': r.get('sbc_pct_rev_xbrl'),
        'fcf_margin_ex_sbc': r.get('fcf_margin_ex_sbc'),
        'net_cash_to_mcap': r.get('net_cash_to_mcap'),
        'deferred_rev_growth': r.get('deferred_rev_growth'),
        # Phase 4 — Industrials KPIs. capex_intensity is universal
        # (derived from edgar_history); backlog_to_revenue uses XBRL
        # RemainingPerformanceObligation.
        'capex_intensity': r.get('capex_intensity'),
        'backlog_to_revenue': r.get('backlog_to_revenue'),
        # Phase 5 — Consumer Cyclical / Defensive working capital +
        # ad spend. Inventory Days and Working Capital Days from XBRL
        # InventoryNet + COGS + receivables/payables. brand_spend_pct_rev
        # from AdvertisingExpense.
        'inventory_days': r.get('inventory_days'),
        'working_capital_days': r.get('working_capital_days'),
        'brand_spend_pct_rev': r.get('brand_spend_pct_rev'),
        # Phase 6 — Energy / Utilities / Materials capital reinvestment.
        'capex_to_dd_ratio': r.get('capex_to_dd_ratio'),
        # Phase 7 — cross-sector quick wins + new XBRL/free-source KPIs.
        # rule_of_40 / book_to_bill_proxy / brand_spend_trend are pure-local
        # (enrich_xbrl); debt_maturity_wall_yrs from XBRL maturity buckets;
        # combined_ratio / float_cost from insurer XBRL; deposit_beta from
        # multi-period FDIC (enrich_fdic); fda_pipeline_count from
        # ClinicalTrials.gov (enrich_pipeline).
        'rule_of_40': r.get('rule_of_40'),
        'book_to_bill_proxy': r.get('book_to_bill_proxy'),
        'brand_spend_trend': r.get('brand_spend_trend'),
        'debt_maturity_wall_yrs': r.get('debt_maturity_wall_yrs'),
        'combined_ratio': r.get('combined_ratio'),
        'float_cost': r.get('float_cost'),
        'deposit_beta': r.get('deposit_beta'),
        'fda_pipeline_count': r.get('fda_pipeline_count'),
        # Ownership
        'shares_out': r.get('shares_out'),
        'float_shares': r.get('float_shares'),
        'insider_pct': r.get('insider_pct'),
        'inst_pct': r.get('inst_pct'),
        'shares_short': r.get('shares_short'),
        'short_ratio': r.get('short_ratio'),
        'short_pct_float': r.get('short_pct_float'),
        'share_turnover_rate': r.get('share_turnover_rate'),
        'share_buyback_rate': r.get('share_buyback_rate'),
        'shareholder_yield': r.get('shareholder_yield'),
        'div_yield': r.get('div_yield'),
        'payout_ratio': r.get('payout_ratio'),
        # Insider activity (Form 4)
        'insider_buy_ratio': r.get('insider_buy_ratio'),
        'insider_buy_count_90d': r.get('insider_buy_count_90d'),
        'insider_sell_count_90d': r.get('insider_sell_count_90d'),
        'insider_buy_count_365d': r.get('insider_buy_count_365d'),
        'insider_sell_count_365d': r.get('insider_sell_count_365d'),
        'insider_net_shares': r.get('insider_net_shares'),
        'insider_net_value': r.get('insider_net_value'),
        'insider_transactions': r.get('insider_transactions', []),
        # EDGAR validation
        'edgar_quality_score': r.get('edgar_quality_score'),
        'edgar_fields_flagged': r.get('edgar_fields_flagged', 0),
        'edgar_discrepancies': r.get('edgar_discrepancies', []),
        # Balance sheet risk flags
        'goodwill_pct': r.get('goodwill_pct'),
        'rd_intensity': r.get('rd_intensity'),
        'sga_pct_rev': r.get('sga_pct_rev'),
        'sga_yoy_change': r.get('sga_yoy_change'),
        # Profit pool
        'revenue': r.get('revenue'),
        'operating_income': r.get('operating_income'),
        'operating_margin': r.get('operating_margin'),
        'pp_revenue_share': r.get('pp_revenue_share'),
        'pp_profit_share': r.get('pp_profit_share'),
        'pp_multiple': r.get('pp_multiple'),
        'pp_margin_advantage': r.get('pp_margin_advantage'),
        'pp_sector_hhi': r.get('pp_sector_hhi'),
        'pp_sector_cr4': r.get('pp_sector_cr4'),
        'pp_sector_count': r.get('pp_sector_count'),
        '_sector_median_opm': r.get('_sector_median_opm'),
        # Per-gate _gate_*/_gp_*/_score_* keys are emitted dynamically from
        # the active gate definitions further down (see the gate_meta_obj
        # spread) so the payload can't drift out of sync with scoring.py —
        # a hand-maintained list carried dead keys for retired gates and
        # missed newly-added ones.
        # Category totals + composite
        '_score_valuation': r.get('_score_valuation'),
        '_score_quality': r.get('_score_quality'),
        '_score_moat': r.get('_score_moat'),
        '_score_growth': r.get('_score_growth'),
        '_score_ownership': r.get('_score_ownership'),
        '_composite_score': r.get('_composite_score'),
        '_composite_score_raw': r.get('_composite_score_raw'),
        '_gates_passed': r.get('_gates_passed'),
        # MC confidence
        'mc_confidence': r.get('mc_confidence'),
        'mc_cv': r.get('mc_cv'),
        'mc_p10_fv': r.get('mc_p10_fv'),
        'mc_p90_fv': r.get('mc_p90_fv'),
        'mc_clip_rate': r.get('mc_clip_rate'),
        'mc_invalid_rate': r.get('mc_invalid_rate'),
        # MC diagnostics (engine #597): where the price sits in the simulated
        # fair values and what drives their spread. mc_quantiles and
        # mc_drivers are detail-only (_extract_details_payload lifts them).
        'mc_price_pctile': r.get('mc_price_pctile'),
        'mc_prob_above_price': r.get('mc_prob_above_price'),
        'implied_growth_z': r.get('implied_growth_z'),
        'mc_driver_top': r.get('mc_driver_top'),
        'mc_drivers': _round_map(r.get('mc_drivers'), 3),
        'mc_quantiles': _round_list(r.get('mc_quantiles'), 2),
        # Macro
        'macro_regime': r.get('macro_regime'),
        'macro_composite': r.get('macro_composite'),
        'macro_erp': r.get('macro_erp'),
        'financial_summary': r.get('financial_summary', []),
        'sector_headwinds': r.get('sector_headwinds', []),
        'sector_tailwinds': r.get('sector_tailwinds', []),
        'news_headlines': r.get('news_headlines', []),
        'news_sentiment': r.get('news_sentiment'),
        'legal_filings': r.get('legal_filings', []),
        'legal_count': r.get('legal_count', 0),
        'legal_latest': r.get('legal_latest'),
        'suppliers': r.get('suppliers', []),
        'customers': r.get('customers', []),
        'supply_chain_available': r.get('supply_chain_available', False),
        'finnhub_peers': r.get('finnhub_peers', []),
        # Growth diagnostics
        'margin_trend': r.get('margin_trend'),
        'surprise_avg': r.get('surprise_avg'),
        'fundamental_growth': r.get('fundamental_growth'),
        'reinvestment_rate': r.get('reinvestment_rate'),
        'terminal_growth': r.get('terminal_growth'),
        # Peers + source
        '_peers': r.get('_peers'),
        'source_group': r.get('source_group'),
        # DDM (Dividend Discount Model)
        'ddm_eligible': r.get('ddm_eligible', False),
        'ddm_fv': r.get('ddm_fv'),
        'ddm_h_fv': r.get('ddm_h_fv'),
        'ddm_growth': r.get('ddm_growth'),
        'ddm_div_cagr': r.get('ddm_div_cagr'),
        'ddm_sustainable_growth': r.get('ddm_sustainable_growth'),
        'ddm_payout_flag': r.get('ddm_payout_flag', False),
        'ddm_consecutive_years': r.get('ddm_consecutive_years'),
        'ddm_reason': r.get('ddm_reason'),
        'dividend_cagr_5y': r.get('dividend_cagr_5y'),
        'ddm_mc_median': r.get('ddm_mc_median'),
        'ddm_mc_p10': r.get('ddm_mc_p10'),
        'ddm_mc_p90': r.get('ddm_mc_p90'),
        'ddm_mc_cv': r.get('ddm_mc_cv'),
        'ddm_mc_clip_rate': r.get('ddm_mc_clip_rate'),
        'ddm_mc_invalid_rate': r.get('ddm_mc_invalid_rate'),
        '_blended_method': r.get('_blended_method', 'DCF'),
        '_ddm_low_confidence': r.get('_ddm_low_confidence', False),
        # Reverse DCF
        'implied_growth': r.get('implied_growth'),
        'implied_vs_estimated': r.get('implied_vs_estimated'),
        # EPV
        'epv_fv': r.get('epv_fv'),
        'epv_pfv': r.get('epv_pfv'),
        'epv_mos': r.get('epv_mos'),
        'epv_growth_fv': r.get('epv_growth_fv'),
        # RIM
        'rim_fv': r.get('rim_fv'),
        'rim_mos': r.get('rim_mos'),
        # Altman Z
        'altman_z': r.get('altman_z'),
        'altman_z_zone': r.get('altman_z_zone'),
        # Beneish
        'beneish_m': r.get('beneish_m'),
        'beneish_flag': r.get('beneish_flag'),
        # Value-trap overlay (display-only; experimental until the forward-
        # return calibration lands — see scoring.compute_trap_signals)
        'trap_score': r.get('trap_score'),
        'trap_flag': r.get('trap_flag'),
        'trap_reasons': r.get('trap_reasons'),
        '_trap_components': r.get('_trap_components'),
        # DuPont
        'dupont_margin': r.get('dupont_margin'),
        'dupont_turnover': r.get('dupont_turnover'),
        'dupont_leverage': r.get('dupont_leverage'),
        # 52-week
        'high_52w': r.get('high_52w'),
        'low_52w': r.get('low_52w'),
        'pct_from_52w_high': r.get('pct_from_52w_high'),
        'pct_from_52w_low': r.get('pct_from_52w_low'),
        'range_52w_position': r.get('range_52w_position'),
        # Market & Risk tab + the detail panel's Price Signals group. These were
        # computed and persisted to the results JSON since the rolling-beta work
        # but never projected into the payload, so every consumer of them
        # rendered N/A. Momentum is display-only by design (see TC.mkt in the
        # template); only avg_dollar_volume_3m feeds scoring, as a HOLD cap.
        'momentum_12_1': r.get('momentum_12_1'),
        'momentum_6_1': r.get('momentum_6_1'),
        'momentum_3m': r.get('momentum_3m'),
        'vol_adj_momentum': r.get('vol_adj_momentum'),
        'realized_vol': r.get('realized_vol'),
        'avg_dollar_volume_3m': r.get('avg_dollar_volume_3m'),
        'amihud_illiquidity': r.get('amihud_illiquidity'),
        'volume_trend': r.get('volume_trend'),
        'price_data_stale': r.get('price_data_stale'),
        'drawdown_2008': r.get('drawdown_2008'),
        'drawdown_2020': r.get('drawdown_2020'),
        'drawdown_2022': r.get('drawdown_2022'),
        'rolling_betas': r.get('rolling_betas'),
        # Portfolio
        'position_weight': r.get('position_weight'),
        # Culture narrative (descriptive, woven into company description)
        'culture_narrative': r.get('culture_narrative'),
        # Culture raw inputs
        'employees': r.get('employees'),
        'ceo_total_pay': r.get('ceo_total_pay'),
        'compensation_risk': r.get('compensation_risk'),
        'sbc': r.get('sbc'),
        # Glassdoor (best-effort)
        'glassdoor_rating': r.get('glassdoor_rating'),
        'glassdoor_ceo_pct': r.get('glassdoor_ceo_pct'),
        'glassdoor_rec_pct': r.get('glassdoor_rec_pct'),
        # Derived culture metrics
        'revenue_per_emp': r.get('revenue_per_emp'),
        'fcf_per_emp': r.get('fcf_per_emp'),
        'ceo_pay_ratio': r.get('ceo_pay_ratio'),
        'sbc_per_emp': r.get('sbc_per_emp'),
        'rpe_cagr': r.get('rpe_cagr'),
        # Culture signals
        'employment_legal_flag': r.get('employment_legal_flag', False),
        'layoff_news_signal': r.get('layoff_news_signal', False),
        'culture_award_signal': r.get('culture_award_signal', False),
        **{k: r.get(k) for g in gate_meta_obj['gates']
           for k in (g['key'], g['gpKey'], g['scoreKey'])},
    }


# Fields the Data-tab summaries compare against a sector median. A median
# needs at least _SECTOR_STAT_MIN_N finite, positive values to be quoted.
_SECTOR_STAT_FIELDS = ('pe', 'ev_ebitda', 'pfcf', 'revenue_per_emp')
# The Profile tab adds P/B to the multiples, and quotes returns, margins and
# yield signed: a median of only the positive FCF margins would flatter the
# sector, and a zero dividend is a real observation.
_PROFILE_MULTIPLE_FIELDS = ('pe', 'ev_ebitda', 'pfcf', 'pb')
_PROFILE_SIGNED_FIELDS = ('roic', 'gross_margin', 'fcf_margin', 'div_yield')
_SECTOR_STAT_MIN_N = 5


def _sector_stats(chart_records, fields=_SECTOR_STAT_FIELDS, positive_only=True):
    """Return {sector: {field: median}} over the rendered rows.

    ``positive_only`` drops values <= 0 (a negative P/E is not a multiple).
    """
    import math
    import statistics
    pools = {}
    for rec in chart_records:
        sector = rec.get('sector')
        if not sector:
            continue
        for f in fields:
            v = rec.get(f)
            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and (v > 0 or not positive_only)):
                pools.setdefault(sector, {}).setdefault(f, []).append(float(v))
    return {sector: {f: statistics.median(vs) for f, vs in by_field.items()
                     if len(vs) >= _SECTOR_STAT_MIN_N}
            for sector, by_field in pools.items()}


def _attach_data_summaries(chart_records):
    """Add the popup Data-tab summaries (``data_summaries``) to each record.

    Built at render time from the payload itself, so a rescore or re-render
    picks up generator changes without a live run. A failure on one row
    drops that row's summaries rather than the report.
    """
    stats = _sector_stats(chart_records)
    for rec in chart_records:
        try:
            summ = generate_data_tab_summaries(rec, stats.get(rec.get('sector')))
            rec['data_summaries'] = {k: v for k, v in summ.items() if v}
        except Exception:
            logger.warning('data-tab summaries failed for %s', rec.get('ticker'),
                           exc_info=True)
            rec['data_summaries'] = {}


def _attach_profiles(chart_records):
    """Add the Profile tab's verdict (``profile``) to each record.

    Like the Data-tab summaries it is built at render time from the payload,
    so a rescore or re-render picks up a rule change without a live run, and
    a failure on one row drops only that row's verdict.
    """
    stats = _sector_stats(chart_records, _PROFILE_MULTIPLE_FIELDS)
    signed = _sector_stats(chart_records, _PROFILE_SIGNED_FIELDS, positive_only=False)
    for sector, meds in signed.items():
        stats.setdefault(sector, {}).update(meds)
    for rec in chart_records:
        try:
            rec['profile'] = profile_verdict(rec, stats.get(rec.get('sector')))
        except Exception:
            logger.warning('profile verdict failed for %s', rec.get('ticker'),
                           exc_info=True)


def _round_list(values, ndigits):
    """A list of numbers rounded for the page, or None unless every entry
    is a finite number (a partial list would misplace the rest)."""
    if not isinstance(values, (list, tuple)) or not values:
        return None
    out = []
    for v in values:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            return None
        out.append(round(float(v), ndigits))
    return out


def _round_map(values, ndigits):
    """A {name: number} map rounded for the page; non-numbers dropped."""
    if not isinstance(values, dict):
        return None
    out = {str(k): round(float(v), ndigits) for k, v in values.items()
           if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
    return out or None


def _extract_details_payload(chart_records):
    # Heavy text fields only consumed inside the detail panel. Strip them
    # from the inline DATA blob into the details/ parts the template
    # lazy-loads after first paint. Cuts the HTML by ~25 MB at 2k tickers.
    _DETAIL_HEAVY_KEYS = (
        'description_full', 'ceo_bio', 'culture_narrative',
        'financial_summary', 'sector_headwinds', 'sector_tailwinds',
        'news_headlines', 'news_sentiment',
        'legal_filings', 'insider_transactions',
        '_trap_components',  # per-axis trap sub-scores; popup-only detail
        'data_summaries',    # Data sub-tab narratives (_attach_data_summaries)
        'profile',           # Profile tab verdict (_attach_profiles)
        'mc_quantiles',      # DCF Monte Carlo distribution strip (21 values)
        'mc_drivers',        # DCF Monte Carlo spread drivers
    )
    details_payload = {}
    for _rec in chart_records:
        _tk = _rec.get('ticker')
        if not _tk:
            continue
        _heavy = {}
        for _k in _DETAIL_HEAVY_KEYS:
            if _k in _rec:
                _v = _rec.pop(_k)
                if _v not in (None, [], '', {}):
                    _heavy[_k] = _v
        if _heavy:
            details_payload[_tk] = _heavy
    return details_payload


def _build_sector_pool_data(rows, macro_sidecar=None):
    # Per-sector profit pool narratives (top-level Profit Pool tab)
    sector_pool_data = {}
    if generate_sector_profit_pool_narrative is not None:
        _by_sector = {}
        for r in rows:
            s = r.get('sector')
            if not s or r.get('pp_revenue_share') is None:
                continue
            _by_sector.setdefault(s, []).append(r)
        # Guarded like every block below: one malformed row must cost the
        # universe shares (economic pool, price of the pool), not the render.
        try:
            universe = universe_totals([r for v in _by_sector.values() for r in v])
        except Exception as e:
            logger.warning('sector pool universe totals failed: %s', e)
            universe = None
        for s, srows in _by_sector.items():
            try:
                narr = generate_sector_profit_pool_narrative(s, srows)
                if narr:
                    sector_pool_data[s] = narr
            except Exception as e:
                print(f"[warn] sector pool narrative failed for {s}: {e}")
                continue
            if not narr:
                continue
            # The pool over time (pp-history / pp-shifts), by industry
            # (pp-chart), against its cost of capital (pp-econ) and its
            # concentration and price (pp-structure).
            try:
                narr['history'] = sector_pool_history(srows)
            except Exception as e:
                logger.warning('sector pool history failed for %s: %s', s, e)
            # Industries grow over the sector's window where they can, so
            # the table and the industry-evidenced forces compare like years.
            window = (narr.get('history') or {}).get('window')
            for key, build, args in (
                    ('industries', industry_pools, (srows, window)),
                    ('economic', economic_pool, (srows, universe)),
                    ('structure_stats', pool_structure, (srows, universe))):
                try:
                    narr[key] = build(*args)
                except Exception as e:
                    logger.warning('sector pool %s failed for %s: %s', key, s, e)
            # The headwinds and tailwinds with their evidence (pp-signals);
            # reads the history and industries just built.
            try:
                narr['forces'] = evaluate_forces(s, macro_sidecar, narr, srows)
            except Exception as e:
                logger.warning('sector forces failed for %s: %s', s, e)
    return sector_pool_data


_DETAILS_PART_BYTES = 8 * 1024 * 1024


def _write_details_parts(details_dir, index_path, details_payload, legacy_path=None,
                         part_bytes=_DETAILS_PART_BYTES):
    """Write the details payload as ``details/<n>.json`` parts of about
    *part_bytes* each, plus ``details_index.json`` (``{"parts": [...]}``);
    return the part names.

    details.json reached ~30 MiB, over Cloudflare Pages' 25 MiB per-file
    limit (design/supabase-migration.md, P4c). The page still loads every
    part after first paint and merges it into DATA exactly as it merged the
    single file, so nothing downstream changes; the parts only keep each
    file small. Tickers are split in sorted order, so a part only ever holds
    whole tickers. Rebuilt wholesale each run; the legacy details.json is
    removed so a stale copy cannot be published.
    """
    parts = []
    try:
        if legacy_path and os.path.exists(legacy_path):
            os.remove(legacy_path)
        if os.path.isdir(details_dir):
            shutil.rmtree(details_dir)
        chunks, cur, size = [], {}, 0
        for _tk in sorted(details_payload or {}):
            _n = len(json.dumps(details_payload[_tk], default=_json_default, separators=_COMPACT))
            if cur and size + _n > part_bytes:
                chunks.append(cur)
                cur, size = {}, 0
            cur[_tk] = details_payload[_tk]
            size += _n + len(_tk) + 4
        if cur:
            chunks.append(cur)
        if chunks:
            os.makedirs(details_dir, exist_ok=True)
        for _i, _chunk in enumerate(chunks):
            with open(os.path.join(details_dir, f'{_i}.json'), 'w', encoding='utf-8') as _df:
                json.dump(_chunk, _df, default=_json_default, separators=_COMPACT)
            parts.append(str(_i))
        if parts:
            with open(index_path, 'w', encoding='utf-8') as _xf:
                json.dump({'parts': parts}, _xf, separators=_COMPACT)
        elif os.path.exists(index_path):
            os.remove(index_path)
    except Exception as _e:
        print(f"[warn] details/ part write failed: {_e}")
        parts = []
    return parts


def _write_macro_sidecar(macro_path, macro_sidecar):
    # Write macro.json for the Macro Outlook tab (or remove a stale one so an
    # offline rebuild doesn't serve last week's macro data).
    try:
        if macro_sidecar:
            with open(macro_path, 'w', encoding='utf-8') as _mf:
                json.dump(_sanitize(macro_sidecar), _mf,
                          default=_json_default, separators=_COMPACT)
        elif os.path.exists(macro_path):
            os.remove(macro_path)
    except Exception as _e:
        print(f"[warn] macro.json write failed: {_e}")


def _build_hist_payload(rows, prices_dir=None):
    # Historical fundamentals (EDGAR annual) per ticker. Share counts and EPS
    # are restated for stock splits here too (data/splits.py), for snapshots
    # written before the live run did it; a restated history carries a marker
    # and passes through unchanged.
    hist_payload = None
    try:
        _hist_out = {}
        _hist_fields = [
            ('revenue_history', 'rev'),
            ('earnings_history', 'ni'),
            ('operating_cf_history', 'ocf'),
            ('capex_history', 'capex'),
            ('gross_profit_history', 'gp'),
            ('shares_history', 'shares'),
            ('dividends_paid_history', 'div'),
            ('total_debt_history', 'debt'),
            ('cash_history', 'cash'),
            # Track Record rows: Op Margin + Interest Coverage by year
            ('operating_income_history', 'op'),
            ('interest_expense_history', 'intexp'),
            # Popup statement tabs (Balance Sheet / Income Statement / Cash
            # Flow). Absent until a run refreshes edgar_history with the
            # 2026-08 extractor; the tabs hide rows whose series is missing.
            ('pretax_income_history', 'pretax'),
            ('tax_provision_history', 'tax'),
            ('d_and_a_history', 'dna'),
            ('total_assets_history', 'assets'),
            ('current_assets_history', 'ca'),
            ('current_liabilities_history', 'cl'),
            ('total_liabilities_history', 'liabs'),
            ('total_equity_history', 'equity'),
            ('retained_earnings_history', 'retearn'),
            # GAAP presentation lines, so the three statement tabs can follow
            # Reg S-X / ASC 220-210-230 ordering instead of listing whatever
            # happened to be extracted. Short keys keep the sidecar small —
            # it already ships ~2,000 tickers.
            ('cost_of_revenue_history', 'cogs'),
            ('rd_expense_history', 'rd'),
            ('sga_expense_history', 'sga'),
            ('selling_marketing_history', 'selmkt'),
            ('total_opex_history', 'opex'),
            ('other_nonop_history', 'othnonop'),
            ('eps_basic_history', 'epsb'),
            ('eps_diluted_history', 'epsd'),
            ('wavg_basic_history', 'wavgb'),
            ('wavg_diluted_history', 'wavgd'),
            ('st_investments_history', 'stinv'),
            ('receivables_history', 'recv'),
            ('inventory_history', 'invty'),
            ('ppe_net_history', 'ppe'),
            ('goodwill_history', 'gw'),
            ('intangibles_history', 'intang'),
            ('accounts_payable_history', 'ap'),
            ('liab_and_equity_history', 'lae'),
            ('cs_apic_history', 'csapic'),
            ('aoci_history', 'aoci'),
            ('treasury_stock_history', 'treas'),
            ('minority_interest_history', 'minint'),
            ('debt_current_history', 'debtcur'),
            ('debt_noncurrent_history', 'debtnc'),
            ('investing_cf_history', 'icf'),
            ('financing_cf_history', 'fincf'),
            ('sbc_cf_history', 'sbccf'),
            ('deferred_tax_history', 'deftax'),
            ('buybacks_history', 'buyback'),
            ('debt_issued_history', 'dissued'),
            ('debt_repaid_history', 'drepaid'),
            ('acquisitions_history', 'acq'),
            ('net_change_cash_history', 'chgcash'),
            ('fx_effect_cash_history', 'fxeff'),
        ]
        for r in rows:
            tk = r.get('ticker')
            if not tk:
                continue
            eh = adjust_for_ticker(r.get('edgar_history') or {}, tk, prices_dir)
            tick_hist = {}
            has_any = False
            for src_key, out_key in _hist_fields:
                series = eh.get(src_key) or r.get(src_key) or {}
                if isinstance(series, dict) and series:
                    # Accept either int year keys (legacy snapshots) or
                    # date-string keys ("YYYY-MM-DD") from the quarterly
                    # extractor. Normalize legacy ints to year-end dates so
                    # the chart sees a uniform date axis.
                    out = {}
                    for k, v in series.items():
                        if v is None:
                            continue
                        try:
                            fv = float(v)
                        except (ValueError, TypeError):
                            continue
                        if isinstance(k, int):
                            key = f"{k}-12-31"
                        else:
                            ks = str(k)
                            if len(ks) == 4 and ks.isdigit():
                                key = f"{ks}-12-31"
                            else:
                                key = ks
                        out[key] = fv
                    if out:
                        tick_hist[out_key] = out
                        has_any = True
            if has_any:
                _hist_avail_and_dividends(tick_hist, eh, tk, prices_dir)
                _hist_out[tk] = tick_hist
        if _hist_out:
            hist_payload = _hist_out
            print(f"[report_html] edgar history: {len(_hist_out)} tickers")
    except Exception as _e:
        print(f"[warn] edgar history load failed: {_e}")
    return hist_payload


# Days after a fiscal year's end before its figures count as known on the
# Charts view's daily multiples: the 10-K deadline (90 days), and 120 for a
# filer reporting in another currency (a 20-F has four months).
_FILING_LAG_DAYS = 90
_FILING_LAG_DAYS_FOREIGN = 120
_DIVIDEND_YEARS = 25


def _hist_avail_and_dividends(tick_hist, eh, ticker, prices_dir):
    """Add what the Charts view needs to rebuild daily multiples from a
    shard: ``fya``, the date each fiscal year's figures became public (its
    real end from ``fy_ends`` when the history has it, else Dec 31 of the
    label year, plus the filing window), and ``dv``, ex-dividend dates and
    amounts, which undo the dividend adjustment in the ``px/`` closes and
    give a trailing yield. Never raises: a shard without them charts the
    annual figures as before."""
    try:
        years = set()
        for series in tick_hist.values():
            if isinstance(series, dict):
                years.update(k[:4] for k in series if isinstance(k, str))
        ends = {str(k): str(v)[:10] for k, v in (eh.get('fy_ends') or {}).items() if v}
        foreign = str(eh.get('reporting_currency') or 'USD').upper() != 'USD'
        lag = timedelta(days=_FILING_LAG_DAYS_FOREIGN if foreign else _FILING_LAG_DAYS)
        fya = {}
        for y in sorted(years):
            try:
                end = date.fromisoformat(ends.get(y) or f'{y}-12-31')
            except ValueError:
                continue
            fya[y] = (end + lag).isoformat()
        if fya:
            tick_hist['fya'] = fya
        since = date(date.today().year - _DIVIDEND_YEARS, 1, 1)
        dv = [[d.isoformat(), round(a, 6)]
              for d, a in dividend_events_from_prices(prices_dir, ticker, since)]
        if dv:
            tick_hist['dv'] = dv
    except Exception as e:
        logger.debug("hist availability/dividends failed for %s: %s", ticker, e)


_SHARD_TICKER = re.compile(r'[A-Za-z0-9._-]{1,15}')


def _write_hist_shards(hist_dir, index_path, hist_payload, legacy_path=None):
    """Write the hist/ shards (one ``<TICKER>.json`` per ticker) and
    ``hist_index.json`` (``{"tickers": [...]}``); return the tickers written.

    hist.json was one ~29 MB file, over Cloudflare Pages' 25 MiB per-file
    limit (design/supabase-migration.md, P4c), and every consumer read a
    single ticker out of it. Rebuilt wholesale each run, like vol/, so a
    ticker that leaves the universe cannot leave a shard behind; the legacy
    hist.json is removed so a stale copy cannot be published.
    """
    written = []
    try:
        if legacy_path and os.path.exists(legacy_path):
            os.remove(legacy_path)
        if os.path.isdir(hist_dir):
            shutil.rmtree(hist_dir)
        if hist_payload:
            os.makedirs(hist_dir, exist_ok=True)
            for _tk in sorted(hist_payload):
                if not _SHARD_TICKER.fullmatch(str(_tk)):
                    continue
                with open(os.path.join(hist_dir, f'{_tk}.json'), 'w', encoding='utf-8') as _hf:
                    json.dump(hist_payload[_tk], _hf, default=_json_default, separators=_COMPACT)
                written.append(_tk)
        if written:
            with open(index_path, 'w', encoding='utf-8') as _xf:
                json.dump({'tickers': written}, _xf, separators=_COMPACT)
        elif os.path.exists(index_path):
            os.remove(index_path)
    except Exception as _e:
        print(f"[warn] hist/ shard write failed: {_e}")
        written = []
    return written


def _load_price_payloads(rows, prices_dir):
    # Load price history from local Parquet files
    prices_payload = None
    px_payload = None
    vol_payload = None
    if prices_dir and os.path.isdir(prices_dir):
        try:
            import pandas as _pd

            def _clean_close(s):
                # The yfinance cache carries garbage closes on ~57 OTC lines:
                # negative back-adjusted prices, 1e28-scale blowups, and
                # sub-penny quotes that round to 0.00 in the shard format.
                # Any of these poisons every chart downstream (a zero or
                # negative rebase base turns a whole sector composite line
                # into inf/None), so they are dropped here at ingestion, not
                # patched around per-chart. Bounds: positive after rounding,
                # and within 100,000x of the series median — the full
                # adjusted history of a mega-compounder sits ~1000x from its
                # own median (AAPL back to 1980 is ~800x), while the
                # adjustment glitches sit at 1e6x and beyond.
                s = s[s > 0.005]
                if len(s):
                    _med = float(s.median())
                    if _med > 0:
                        s = s[s <= _med * 100000]
                return s
            _series = {}
            _vol_series = {}
            _buyfrac_series = {}
            for r in rows:
                tk = r.get('ticker')
                if not tk:
                    continue
                pf = os.path.join(prices_dir, f"{tk}.parquet")
                if not os.path.exists(pf):
                    continue
                try:
                    # Full history, no cutoff: the parquet cache holds
                    # whatever period="max" returned (decades for mature
                    # listings), and the chart's MAX range should show all
                    # of it — a 20-year clip here silently truncated over
                    # half the universe at the same start date.
                    df = _pd.read_parquet(pf).sort_index()
                    df.index = _pd.to_datetime(df.index).tz_localize(None).normalize()
                    s = _clean_close(df['Close'].dropna())
                    if len(s) >= 20:
                        _series[tk] = s
                        # Volume rides along for the popup chart's sub-panel.
                        # Never name it in a columns= projection: ~0.6% of the
                        # files were written Close-only by the yfinance_client
                        # fallback and the projection would raise on those.
                        if 'Volume' in df.columns:
                            _v = df['Volume'].dropna()
                            if len(_v) >= 20:
                                _vol_series[tk] = _v
                                # Estimated buying share of the day's volume,
                                # from where the close sits in the day's range:
                                # the money-flow multiplier behind Chaikin
                                # Accumulation/Distribution. Daily bars can't
                                # tell us who initiated a trade — that needs
                                # tick data — so this is explicitly an estimate
                                # and the report labels it as one. A zero-range
                                # day (limit move, halt) splits evenly.
                                if 'High' in df.columns and 'Low' in df.columns:
                                    _hi = df['High']
                                    _lo = df['Low']
                                    _rng = _hi - _lo
                                    _f = ((df['Close'] - _lo) / _rng)
                                    _f = _f.where(_rng > 0, 0.5).clip(0.0, 1.0)
                                    _buyfrac_series[tk] = _f
                except Exception:
                    pass
            # Also load any available index files
            _index_labels = {'SPY': 'S&P 500', 'QQQ': 'NASDAQ', 'IWM': 'Russell 2000', 'DIA': 'Dow Jones'}
            _indices_found = []
            for _itk, _ilabel in _index_labels.items():
                _ipf = os.path.join(prices_dir, f"{_itk}.parquet")
                if not os.path.exists(_ipf):
                    continue
                try:
                    _idf = _pd.read_parquet(_ipf)[['Close']].sort_index()
                    _idf.index = _pd.to_datetime(_idf.index).tz_localize(None).normalize()
                    _is = _clean_close(_idf['Close'].dropna())
                    if len(_is) >= 20:
                        _series[_itk] = _is
                        _indices_found.append({'ticker': _itk, 'label': _ilabel})
                except Exception:
                    pass
            # The look-through's funds ship as ordinary px/ shards (not as
            # indices): each sector page charts its own SPDR against SPY, and
            # the ETFs tab's Detail any fund. They are never rows, so nothing
            # else picks them up.
            for _ftk in etf_holdings.fund_tickers():
                if _ftk in _series:
                    continue
                _fpf = os.path.join(prices_dir, f"{_ftk}.parquet")
                if not os.path.exists(_fpf):
                    continue
                try:
                    _fdf = _pd.read_parquet(_fpf)[['Close']].sort_index()
                    _fdf.index = _pd.to_datetime(_fdf.index).tz_localize(None).normalize()
                    _fs = _clean_close(_fdf['Close'].dropna())
                    if len(_fs) >= 20:
                        _series[_ftk] = _fs
                except Exception as _e:
                    logger.debug('sector ETF %s price load failed: %s', _ftk, _e)
            if _series:
                _all_dates = sorted(set().union(*[set(s.index.tolist()) for s in _series.values()]))
                _date_strs = [d.strftime('%Y-%m-%d') for d in _all_dates]
                _date_idx = {d: i for i, d in enumerate(_all_dates)}
                _prices_out = {}
                for tk, s in _series.items():
                    arr = [None] * len(_all_dates)
                    for dt, v in s.items():
                        i = _date_idx.get(dt)
                        if i is not None:
                            # Sub-$1 closes get 4 decimals: decades-old
                            # split/dividend-adjusted prices routinely sit
                            # at a few cents, where 2-decimal rounding
                            # quantizes a year of moves into flat steps.
                            v = float(v)
                            arr[i] = round(v, 2) if v >= 1 else round(v, 4)
                    _prices_out[tk] = arr
                # Daily volume ships as one small file per ticker rather than a
                # second dense matrix: the popup opens one company at a time, so
                # it fetches ~30 KB instead of the ~53 MB the matrix would cost —
                # and prices.json stays clear of the 100 MB Pages cap.
                _vol_out = {}
                for tk, v in _vol_series.items():
                    if tk not in _prices_out:
                        continue
                    # Ticker names become filenames below; keep the plainly
                    # safe ones rather than trusting the universe file.
                    if not re.fullmatch(r'[A-Za-z0-9._-]{1,15}', tk):
                        continue
                    arr = [None] * len(_all_dates)
                    for dt, x in v.items():
                        i = _date_idx.get(dt)
                        if i is not None:
                            arr[i] = int(x)
                    # Run-trim the leading/trailing null padding — a quarter of
                    # the dense layout's cells are nulls a short-history ticker
                    # should not have to pay for.
                    lo, hi = 0, len(arr) - 1
                    while lo <= hi and arr[lo] is None:
                        lo += 1
                    while hi >= lo and arr[hi] is None:
                        hi -= 1
                    if hi < lo:
                        continue
                    shard = {'i0': lo, 'v': arr[lo:hi + 1]}
                    # Buying share as an int 0-100, on the SAME run as v so the
                    # client can index both with one offset. 1% of a day's
                    # volume is finer than a 40px-tall stacked bar can show.
                    fs = _buyfrac_series.get(tk)
                    if fs is not None:
                        farr = [None] * len(_all_dates)
                        for dt, x in fs.dropna().items():
                            i = _date_idx.get(dt)
                            if i is not None and arr[i] is not None:
                                farr[i] = int(round(float(x) * 100))
                        shard['f'] = farr[lo:hi + 1]
                    _vol_out[tk] = shard
                # Close prices ship one small shard per ticker (px/<TICKER>.json,
                # ~30 KB), exactly like vol/ — the dense prices.json matrix had
                # grown to ~62 MB, a 30s+ blocking download for every chart and
                # on course for the GitHub Pages 100 MB cap. Shard shape mirrors
                # vol: {i0, p} run-trimmed of null padding, indexed into dates.
                _px_out = {}
                for tk, arr in _prices_out.items():
                    if not re.fullmatch(r'[A-Za-z0-9._-]{1,15}', tk):
                        continue
                    lo, hi = 0, len(arr) - 1
                    while lo <= hi and arr[lo] is None:
                        lo += 1
                    while hi >= lo and arr[hi] is None:
                        hi -= 1
                    if hi < lo:
                        continue
                    _px_out[tk] = {'i0': lo, 'p': arr[lo:hi + 1]}
                # The landing sector-performance chart used to mcap-weight every
                # constituent's series client-side, which is what forced the full
                # matrix download. Precompute the composite here instead: per
                # sector, a weighted index level (each component rebased to its
                # own first quote, mcap-weighted over whatever weight has data
                # that day) plus a per-day data-coverage fraction so the client
                # can keep its "≥60% of weight present" guard. The client
                # re-indexes the level to the selected range start — composite
                # re-indexing rather than per-component re-basing, the same way
                # any published index handles constituents entering the window.
                _sector_px = {}
                try:
                    import numpy as _np
                    _sec_members = {}
                    for r in rows:
                        _tk, _sec, _mc = r.get('ticker'), r.get('sector'), r.get('mcap')
                        # Upper mcap sanity bound: yfinance hands FNMFO (a
                        # Fannie Mae preferred line) a $180T market cap, which
                        # made it 93% of the Financial Services composite —
                        # every junk print on that one OTC line moved the
                        # sector index by +/-80%. No real company is within
                        # 4x of $20T; anything past it is a data error, not a
                        # constituent.
                        if (_tk in _px_out and _sec
                                and isinstance(_mc, (int, float)) and 0 < _mc < 2e13):
                            _sec_members.setdefault(_sec, []).append((_tk, float(_mc)))
                    for _sec, _members in _sec_members.items():
                        if len(_members) < 3:
                            continue
                        M = _np.array([[_np.nan if v is None else v
                                        for v in _prices_out[_t]] for _t, _ in _members],
                                      dtype=float)
                        w = _np.array([_m for _, _m in _members], dtype=float)
                        w = w / w.sum()
                        valid = ~_np.isnan(M) & (M > 0)
                        # Chain-linked index of weighted daily returns, not a
                        # weighted average of rebase-to-first-quote levels: a
                        # component only moves the index on days where it has
                        # quotes on BOTH t-1 and t, so a sparse OTC line that
                        # trades twice a year contributes two clamped daily
                        # moves instead of an R=price/tiny-base level that
                        # steps the whole sector by hundreds of points. This
                        # is what turned the 20Y sector chart into a
                        # Dec'11-Mar'13 sliver: penny-base components made lv
                        # inf/None on most days, and the client's
                        # all-sectors-valid gate kept only the 313 days where
                        # every sector dodged its own junk.
                        vboth = valid[:, :-1] & valid[:, 1:]
                        with _np.errstate(invalid='ignore', divide='ignore'):
                            r = _np.where(vboth, M[:, 1:] / M[:, :-1] - 1.0, 0.0)
                        # Junk-print guard: no index constituent legitimately
                        # quadruples or loses 75% between closes; adjustment
                        # glitches routinely do.
                        r = _np.clip(r, -0.75, 3.0)
                        wm = _np.where(vboth, w[:, None], 0.0)
                        cov_t = wm.sum(axis=0)
                        with _np.errstate(invalid='ignore'):
                            cr = _np.where(cov_t > 0,
                                           (r * wm).sum(axis=0) / _np.where(cov_t > 0, cov_t, 1.0),
                                           0.0)
                        lv = 100.0 * _np.cumprod(1.0 + cr)
                        lv = _np.concatenate(([100.0], lv))
                        # cov[0] is quote coverage (no return exists on day
                        # one); after that it is the weight fraction with a
                        # valid return, which is what the client's >=60% gate
                        # actually wants to know.
                        cov0 = float(_np.where(valid[:, 0], w, 0.0).sum()) if M.shape[1] else 0.0
                        cov = _np.concatenate(([cov0], cov_t))
                        _sector_px[_sec] = {
                            'lv': [round(float(x), 2) if _np.isfinite(x) else None for x in lv],
                            'cov': [round(float(x), 3) for x in cov],
                            'nc': len(_members),
                        }
                except Exception as _e:
                    print(f"[warn] sector price composite failed: {_e}")
                prices_payload = {
                    'dates': _date_strs,
                    # Shard indices are positions into `dates`; the client bails
                    # rather than misalign if it ever sees mismatched artifacts.
                    'n': len(_date_strs),
                    'indices': _indices_found,
                    # Index series embedded whole: the popup's SPY overlay and
                    # the Charts-tab index chips shouldn't cost a fetch each.
                    'index_px': {x['ticker']: _px_out[x['ticker']]
                                 for x in _indices_found if x['ticker'] in _px_out},
                    # Tickers with a px/<TICKER>.json shard — membership checks
                    # (search, seeding, popup section visibility) read this
                    # instead of eating a 404 per miss.
                    'manifest': sorted(_px_out.keys()),
                    # Tickers with a vol/<TICKER>.json shard. The client checks
                    # this instead of eating a 404 on every popup open.
                    'vol': sorted(_vol_out.keys()),
                    'sectors': _sector_px,
                }
                px_payload = _px_out
                vol_payload = _vol_out
                print(f"[report_html] price history: {len(_px_out)} px shards, {len(_date_strs)} dates, "
                      f"indices: {[x['ticker'] for x in _indices_found]}, "
                      f"sector composites: {len(_sector_px)}")
                print(f"[report_html] volume shards: {len(_vol_out)} tickers")
        except Exception as _e:
            print(f"[warn] price history load failed: {_e}")
    return prices_payload, px_payload, vol_payload


def _write_prices_meta_sidecar(prices_meta_path, prices_path, prices_payload):
    # Write prices_meta.json (dates + manifests + index series + sector
    # composites) or remove a stale one. The legacy dense prices.json is
    # always removed — per-ticker px/ shards replaced it.
    prices_size_mb = 0
    try:
        if prices_payload is not None:
            with open(prices_meta_path, 'w', encoding='utf-8') as _pf2:
                json.dump(prices_payload, _pf2, default=_json_default,
                          separators=_COMPACT)
            # Uncompressed size, stamped into the client so download progress
            # can be honest — the transfer is gzipped, so Content-Length alone
            # can't say how much of the decompressed body has arrived.
            prices_size_mb = round(os.path.getsize(prices_meta_path) / 1048576)
        elif os.path.exists(prices_meta_path):
            os.remove(prices_meta_path)
        if os.path.exists(prices_path):
            os.remove(prices_path)
    except Exception as _e:
        print(f"[warn] prices_meta.json write failed: {_e}")
    return prices_size_mb


def _write_px_shards(px_dir, px_payload):
    # Write the px/ close-price shard directory — same lifecycle as vol/
    # below: rebuilt wholesale, then swept of anything the manifest doesn't
    # claim (iCloud conflict copies regenerate continuously in this folder).
    try:
        if px_payload is not None:
            import shutil as _sh
            if os.path.isdir(px_dir):
                _sh.rmtree(px_dir, ignore_errors=True)
            os.makedirs(px_dir, exist_ok=True)
            for _tk, _shd in px_payload.items():
                with open(os.path.join(px_dir, f'{_tk}.json'), 'w', encoding='utf-8') as _xf:
                    json.dump(_shd, _xf, separators=_COMPACT)
            _stale_px = 0
            for _fn in os.listdir(px_dir):
                if _fn.endswith('.json') and _fn[:-5] not in px_payload:
                    os.remove(os.path.join(px_dir, _fn))
                    _stale_px += 1
            if _stale_px:
                print(f"[report_html] px/: swept {_stale_px} unclaimed shard file(s)")
    except Exception as _e:
        print(f"[warn] px/ shard write failed: {_e}")


def _write_vol_shards(vol_dir, vol_payload):
    # Write the vol/ shard directory. Rebuilt wholesale each run so a ticker
    # that drops out of the universe can't leave a shard behind whose indices
    # no longer line up with the current dates array.
    try:
        if os.path.isdir(vol_dir):
            shutil.rmtree(vol_dir)
        if vol_payload:
            os.makedirs(vol_dir, exist_ok=True)
            for _tk, _sh in vol_payload.items():
                with open(os.path.join(vol_dir, f'{_tk}.json'), 'w', encoding='utf-8') as _vf:
                    json.dump(_sh, _vf, separators=_COMPACT)
            # Belt and braces on top of the rmtree: sweep anything the manifest
            # doesn't claim. A 2026-08-08 run found 2,139 macOS conflict copies
            # ("AAPL 3.json") that outlived the rmtree — harmless to the client,
            # which only ever fetches manifest entries, but ~55 MB of junk that
            # a wholesale copy would publish.
            _stale = 0
            for _fn in os.listdir(vol_dir):
                if _fn.endswith('.json') and _fn[:-5] not in vol_payload:
                    os.remove(os.path.join(vol_dir, _fn))
                    _stale += 1
            if _stale:
                print(f"[report_html] vol/: swept {_stale} unclaimed shard file(s)")
    except Exception as _e:
        print(f"[warn] vol/ shard write failed: {_e}")


# ------------------------------------------------------------------ app feed
# The iOS app (Projects/StockModel-iOS-Handoff, its README section 4) reads
# three files of its own rather than the inline DATA blob in index.html. The
# desktop sidecars are the wrong shape for a phone: details/ is ~8 MiB of
# parts to open one company, and DATA is only reachable by parsing a 10 MB
# HTML page.
_APP_SCHEMA_VERSION = 1

# What the ratings list shows, searches, filters and sorts on (handoff
# section 3) and nothing else: the app downloads this whole file each night.
_APP_LIST_FIELDS = (
    'ticker', 'company_name', 'sector', 'industry', 'rating', 'price',
    '_fv_effective', '_fv_source', 'mos', '_composite_score',
    '_score_valuation', '_score_quality', '_score_moat', '_score_growth',
    '_score_ownership', 'trap_score', 'price_data_stale',
    # The prior run's composite (re-scored with tonight's model where the
    # history allows) and the change, for "crossed above 57" and the
    # Score moves screen; the app shows the score, never the rating.
    'score_prev', 'score_chg', 'score_prev_basis',
)


def _app_risk_free_rate(rows):
    # The results JSON's top-level risk_free_rate never reaches build_html,
    # but every row carries the run's rate.
    for _r in rows or ():
        _v = _r.get('_risk_free_rate')
        if isinstance(_v, (int, float)) and not isinstance(_v, bool) \
                and math.isfinite(_v):
            return float(_v)
    return None


def _build_app_list(chart_records, run_date, risk_free_rate, score_hist_dates=None):
    """Build the slim list feed's payload.

    A field the row does not carry is left out rather than nulled: every
    numeric in the app's Codable types is already optional, and omitting
    them is what keeps the file inside the handoff's 1.5 MB target. Values
    are sanitised here rather than at write time so that a NaN price is
    absent too, instead of becoming the one explicit null in the file.
    """
    _rows = []
    for _rec in chart_records or ():
        _tk = _rec.get('ticker')
        if not _tk:
            continue
        _row = {}
        for _k in _APP_LIST_FIELDS:
            _v = _sanitize(_rec.get(_k), (_tk, _k))
            if _v is not None:
                _row[_k] = _v
        _row['ticker'] = _tk
        _rows.append(_row)
    _rows.sort(key=lambda r: r['ticker'])
    return {
        'schema_version': _APP_SCHEMA_VERSION,
        'date': run_date,
        'risk_free_rate': risk_free_rate,
        # Dates the detail shards' score_hist arrays are aligned to.
        'score_hist_dates': list(score_hist_dates or []),
        'count': len(_rows),
        'results': _rows,
    }


def _sha256_file(path, _chunk=1 << 20):
    _h = hashlib.sha256()
    with open(path, 'rb') as _fh:
        for _blk in iter(lambda: _fh.read(_chunk), b''):
            _h.update(_blk)
    return _h.hexdigest()


def _app_json_default(obj):
    """``_json_default``, plus dates as ISO strings.

    It raises for ``datetime.date``, which is right for the inline DATA blob
    (a date there is a bug) but wrong here: the nightly render passes
    ``run_date`` as a ``date`` (rescore_and_render.py), and a detail shard
    carries whatever ~370 fields the row has. One of either cost the whole
    feed on 2026-11-06 — the write is wrapped, so the only trace was a
    ``[warn] app/ feed write failed`` line and no manifest.
    """
    if isinstance(obj, date):        # datetime is a subclass of date
        return obj.isoformat()
    return _json_default(obj)


def _write_app_json(path, payload):
    """Write one app feed file.

    ``_sanitize`` nulls NaN/inf (numeric or stringified) because the raw
    snapshots carry both and ``json.dump`` writes them as the JavaScript
    literals ``NaN``/``Infinity``, which are not JSON. The browser's parser
    tolerates them in practice; Swift's ``JSONDecoder`` does not, so one
    such value would read as a corrupt night rather than a missing field.
    ``allow_nan=False`` makes that a guarantee instead of a convention:
    whatever the sanitiser missed raises here, while there is still no
    manifest pointing the app at the file.
    """
    with open(path, 'w', encoding='utf-8') as _fh:
        json.dump(_sanitize(payload), _fh, default=_app_json_default,
                  separators=_COMPACT, allow_nan=False)


# The Sector Analysis numbers the report computes in the browser from the
# inline DATA blob, which the app never downloads: the Sector Heatmap
# (renderCrossSectorDashboard's _sumStats) and the Cross-Sector Profit Pool
# (renderCrossSectorProfitPool). Ported line for line so the app shows the
# report's figures; tests/test_app_feed.py pins them against the same rows.
_APP_LENDER_INDUSTRY = re.compile(r'Bank|Mortgage|Credit Services|Capital Markets')


def _app_num(v):
    """A finite number, or None (the JS ``v!=null&&!isNaN(v)``)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _app_or0(v):
    """The JS ``(v||0)``: a missing or non-finite value counts as zero."""
    n = _app_num(v)
    return n if n is not None else 0.0


def _app_median(vals):
    if not vals:
        return None
    s = sorted(vals)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def _app_is_lender(r):
    # A lender's FCF is deposit and loan flow and Altman Z was built for
    # industrial balance sheets, so the heatmap leaves lenders out of both.
    return (r.get('sector') == 'Financial Services'
            and bool(_APP_LENDER_INDUSTRY.search(r.get('industry') or '')))


def _app_sector_heatmap(chart_records):
    """One row per sector: the report's Sector Heatmap."""
    by_sector = {}
    for r in chart_records or ():
        if r.get('sector') and r.get('pp_revenue_share') is not None:
            by_sector.setdefault(r['sector'], []).append(r)
    out = []
    for sec in sorted(by_sector):
        cos = by_sector[sec]
        srt = sorted(cos, key=lambda d: _app_or0(d.get('operating_margin')), reverse=True)
        total_rev = sum(_app_or0(d.get('pp_revenue_share')) for d in srt)
        if not total_rev:
            continue
        wtd_m = sum(_app_or0(d.get('operating_margin')) * _app_or0(d.get('pp_revenue_share'))
                    for d in srt) / total_rev
        best = srt[0]
        for d in srt[1:]:
            # reduce((a,b) => a > b ? a : b): a tie goes to the later row.
            if not _app_or0(best.get('pp_multiple')) > _app_or0(d.get('pp_multiple')):
                best = d

        def vals(key, rows=cos, keep=lambda v: True):
            return [v for v in (_app_num(d.get(key)) for d in rows) if v is not None and keep(v)]

        growth = vals('rev_cagr_5y') or vals('rev_cagr')
        shr = []
        for d in cos:
            sy = _app_num(d.get('shareholder_yield'))
            if sy is None or sy == 0:
                # yfinance's cash-flow statement is sparse, leaving sy at 0.0
                # where a real dividend exists: fall back to buyback + yield.
                bb, dy = _app_num(d.get('share_buyback_rate')), _app_num(d.get('div_yield'))
                if (bb is not None and bb != 0) or (dy is not None and dy != 0):
                    sy = (bb or 0.0) + (dy or 0.0)
            if sy is not None:
                shr.append(sy)
        non_lend = [d for d in cos if not _app_is_lender(d)]
        n_buy = sum(1 for d in cos if d.get('rating') in ('BUY', 'LEAN BUY'))
        out.append({
            'sector': sec, 'n': len(srt), 'op_margin': wtd_m,
            'hhi': _app_num(srt[0].get('pp_sector_hhi')), 'cr4': _app_num(srt[0].get('pp_sector_cr4')),
            'med_mos': _app_median(vals('mos')),
            'med_p_tbv': _app_median(vals('p_tbv', keep=lambda v: 0 < v < 50)),
            'med_ev_ebitda': _app_median(vals('ev_ebitda', keep=lambda v: 0 < v < 100)),
            'med_spread': _app_median(vals('spread')),
            'med_growth': _app_median(growth),
            'med_roic': _app_median(vals('roic')),
            'med_nd_ebitda': _app_median(vals('nd_ebitda')),
            'med_shr_yld': _app_median(shr),
            'med_fcf_margin': _app_median(vals('fcf_margin', rows=non_lend)),
            'med_altman_z': _app_median(vals('altman_z', rows=non_lend)),
            'n_buy': n_buy, 'pct_buy': n_buy / len(cos),
            'best': {'ticker': best.get('ticker'), 'pp_multiple': _app_num(best.get('pp_multiple'))},
        })
    return out


def _app_profit_pool(chart_records):
    """Each sector's operating revenue, operating income and margin, one
    contribution per issuer (share classes and preferreds repeat the parent's
    statements), ordered by margin: the Cross-Sector Profit Pool."""
    agg, seen = {}, set()
    for r in chart_records or ():
        sec = r.get('sector')
        rev, oi = _app_num(r.get('revenue')), _app_num(r.get('operating_income'))
        if not sec or rev is None or oi is None or rev <= 0:
            continue
        key = (sec, rev, oi)
        if key in seen:
            continue
        seen.add(key)
        a = agg.setdefault(sec, {'sector': sec, 'revenue': 0.0, 'operating_income': 0.0, 'companies': 0})
        a['revenue'] += rev
        a['operating_income'] += oi
        a['companies'] += 1
    if len(agg) < 2:
        return []
    for a in agg.values():
        a['margin'] = a['operating_income'] / a['revenue']
    return sorted(agg.values(), key=lambda a: a['margin'], reverse=True)


_APP_POOL_TOP_N = 25   # the report's _ppTopN


def _app_pool_companies(rows):
    """A sector's "Who Earns It": the PP_TOP_N largest companies by revenue
    share with what the report's pool chart and table show for each, and
    the rest pooled as Others (renderPool / renderPoolCompanyTable)."""
    by_rev = sorted(rows, key=lambda d: _app_or0(d.get('pp_revenue_share')), reverse=True)
    shown, tail = by_rev[:_APP_POOL_TOP_N], by_rev[_APP_POOL_TOP_N:]
    out = {
        'n': len(by_rev),
        # The sector's market value over the charted set and the tail (mcap
        # > 0): the denominator of each company's value / pool.
        'mcap_total': sum(_app_num(d.get('mcap')) or 0.0 for d in by_rev
                          if (_app_num(d.get('mcap')) or 0) > 0),
        'shown': [{
            'ticker': d.get('ticker'), 'company_name': d.get('company_name'),
            'rating': d.get('rating'), 'industry': d.get('industry'),
            'revenue_share': _app_num(d.get('pp_revenue_share')),
            'pool_share': _app_num(d.get('pp_profit_share')),
            'margin': _app_num(d.get('operating_margin')),
            'spread': _app_num(d.get('spread')),
            # The Pool Share gate: the 5-year CAGR of the company's share.
            'pool_share_trend': _app_num(d.get('_gate_pool_share')),
            'mcap': _app_num(d.get('mcap')),
        } for d in shown],
    }
    if tail:
        rev = sum(_app_or0(d.get('pp_revenue_share')) for d in tail)
        out['others'] = {
            'n': len(tail),
            'revenue_share': rev,
            'pool_share': sum(_app_or0(d.get('pp_profit_share')) for d in tail),
            'margin': (sum(_app_or0(d.get('operating_margin')) * _app_or0(d.get('pp_revenue_share'))
                           for d in tail) / rev) if rev > 0 else None,
            'mcap': sum(_app_num(d.get('mcap')) or 0.0 for d in tail if (_app_num(d.get('mcap')) or 0) > 0),
        }
    return out


def _build_app_sectors(chart_records, sector_pool, run_date):
    """``app/sectors.json``: the heatmap, the profit pool, and each sector's
    pool (the sector pages' narrative, history, industries, economics and
    forces, as built for SECTOR_POOL), plus its largest companies."""
    pools = {}
    by_sector = {}
    for r in chart_records or ():
        if r.get('sector') and r.get('pp_revenue_share') is not None:
            by_sector.setdefault(r['sector'], []).append(r)
    for sec, pool in (sector_pool or {}).items():
        # A shallow copy: SECTOR_POOL itself is also inlined in the report.
        pools[sec] = dict(pool, companies=_app_pool_companies(by_sector.get(sec, [])))
    return {
        'schema_version': _APP_SCHEMA_VERSION,
        'date': run_date,
        'heatmap': _app_sector_heatmap(chart_records),
        'profit_pool': _app_profit_pool(chart_records),
        'pools': pools,
    }


# The Investing screener: every column of the report's Financial Data table
# tabs (templates/report.html TC), which the app filters and sorts the way
# the report's applyFilters does. Too wide for latest.json (the app downloads
# that every night), so it is its own file, fetched when the screener opens.
# tests/test_app_feed.py fails if this drifts from the template's columns.
_APP_SCREENER_FIELDS = (
    'rating', 'price', 'dcf_fv', 'mos', 'wacc', 'implied_growth',
    'pct_from_52w_high', 'range_52w_position', 'pe', 'ev_ebitda', 'pfcf', 'pb',
    'p_tbv', 'epv_fv', 'rim_fv', 'nav_fv', 'nav_mos', 'roic', 'roic_cv', 'spread',
    'roe', 'dupont_margin', 'dupont_turnover', 'dupont_leverage', 'gross_margin',
    'fcf_margin', 'fcf_margin_ex_sbc', 'cash_conv', 'accruals',
    'rd_intensity_xbrl', 'sbc_pct_rev_xbrl', 'brand_spend_pct_rev', 'nim',
    'efficiency_ratio', 'affo_margin', 'int_cov', 'nd_ebitda', 'altman_z',
    'net_cash_to_mcap', 'cr', 'inventory_days', 'working_capital_days',
    'beneish_m', 'trap_score', 'cet1_ratio', 'npl_ratio', 'reinvestment_rate',
    'fundamental_growth', 'margin_trend', 'rev_cagr', 'rev_cagr_5y',
    'rev_cagr_10y', 'fcf_growth', 'capex_intensity', 'capex_to_dd_ratio',
    'ffo_growth_5y', 'deferred_rev_growth', 'backlog_to_revenue', 'er',
    '_sens_low', '_sens_high', 'mc_p10_fv', 'mc_p90_fv', 'mc_confidence',
    'terminal_growth', 'shares_out', 'float_shares', 'share_buyback_rate',
    'insider_pct', 'inst_pct', 'short_ratio', 'short_pct_float', 'div_yield',
    'payout_ratio', 'shareholder_yield', 'ddm_consecutive_years',
    'dividend_cagr_5y', 'ddm_div_cagr', 'ddm_sustainable_growth', 'ddm_growth',
    'ddm_fv', 'ddm_h_fv', 'ddm_mc_p10', 'ddm_mc_median', 'ddm_mc_p90', 'ddm_mc_cv',
    'ddm_eligible', 'ddm_reason', 'momentum_12_1', 'momentum_6_1', 'momentum_3m',
    'vol_adj_momentum', 'realized_vol', 'avg_dollar_volume_3m', 'volume_trend',
    'amihud_illiquidity', 'share_turnover_rate', 'drawdown_2020', 'drawdown_2022',
    'drawdown_2008', 'beta_adjusted',
)
# Beyond the table columns: what the quick filters read, and the Filters
# panel's "Debt" column filters (_CF_NUM). Its "Scores" filters ride the
# list feed already.
# earnings_next_date: a portfolio Summary's "Earnings ≤ 7 days" (_pfRowStats).
_APP_SCREENER_EXTRA = ('country', 'founder_led', 'r2000', 'rating_chg',
                       'mcap', 'total_debt', 'net_debt', 'de', 'debt_maturity_wall_yrs',
                       'earnings_next_date',
                       # A sector's Trading tab (renderPoolLiquidityInsights).
                       'revenue', 'operating_income', 'pp_revenue_share', 'pp_sector_cr4')
_APP_SCREENER_SIG_DIGITS = 6   # the screener shows at most four


def _app_trim(v):
    if isinstance(v, bool) or not isinstance(v, float):
        return v
    return float(f'{v:.{_APP_SCREENER_SIG_DIGITS}g}') if math.isfinite(v) else None


def _app_is_blue_chip(r):
    """The report's _isBlueChip: $50B+ market cap, profitable, ROE of 10%+,
    and a sound balance sheet by any one of three tests."""
    mc, oi, roe = _app_num(r.get('mcap')), _app_num(r.get('operating_income')), _app_num(r.get('roe'))
    if mc is None or mc < 50e9 or oi is None or oi <= 0 or roe is None or roe < 0.10:
        return False
    nde, ic, az = _app_num(r.get('nd_ebitda')), _app_num(r.get('int_cov')), _app_num(r.get('altman_z'))
    return (nde is not None and nde <= 4) or (ic is not None and ic >= 6) or (az is not None and az >= 3)


def _build_app_screener(chart_records, run_date):
    """``app/screener.json``: the table columns for every ticker, packed
    column-wise (pack_rows), in the list feed's ticker order."""
    recs = sorted((r for r in chart_records or () if r.get('ticker')), key=lambda r: str(r['ticker']))
    rows, gate_null = [], {}
    for i, r in enumerate(recs):
        row = {}
        for k in _APP_SCREENER_FIELDS + _APP_SCREENER_EXTRA:
            v = _sanitize(r.get(k), (r.get('ticker'), k))
            if v is not None:
                row[k] = _app_trim(v)
            # A column whose scoring gate is null is filtered out even when
            # its raw value is present (applyFilters' _gate_ check), so a
            # column filter agrees with the Gate Matrix.
            g = '_gate_' + k
            if g in r and _app_num(r.get(g)) is None:
                gate_null.setdefault(k, []).append(i)
        if _app_is_blue_chip(r):
            row['blue_chip'] = True
        rows.append(row)
    return {
        'schema_version': _APP_SCHEMA_VERSION,
        'date': run_date,
        'tickers': [r['ticker'] for r in recs],
        'rows': pack_rows(rows),
        'gate_null': gate_null,
    }


def _build_app_gates(chart_records, run_date, gate_meta=None):
    """``app/gates.json``: the Gate Matrix (Investing › Stocks › Model) and
    the Stocks Dashboard's pass rates, for the app.

    The gate definitions as the template's GM gets them (gate_metadata),
    and per gate two columns in the list feed's ticker order: the gate's
    value (``values[key]``) and its pass flag (``pass[gpKey]``, null when
    the gate does not apply). Column-wise because 26 gates × ~3,900 rows
    as objects would repeat every key on every row. The category totals
    and the composite are already in latest.json.
    """
    gm = gate_meta or gate_metadata()
    recs = sorted((r for r in chart_records or () if r.get('ticker')), key=lambda r: str(r['ticker']))
    values, passes = {}, {}
    for g in gm.get('gates', []):
        col, flags = [], []
        for r in recs:
            v = _sanitize(r.get(g['key']), (r.get('ticker'), g['key']))
            col.append(_app_trim(v) if v is not None else None)
            p = _sanitize(r.get(g['gpKey']), (r.get('ticker'), g['gpKey']))
            flags.append(None if p is None else bool(p))
        values[g['key']] = col
        passes[g['gpKey']] = flags
    return {
        'schema_version': _APP_SCHEMA_VERSION,
        'date': run_date,
        'gate_meta': gm,
        'tickers': [r['ticker'] for r in recs],
        'values': values,
        'pass': passes,
    }


def _build_app_etfs(payload, run_date):
    """``app/etfs.json``: the SPDR look-through (Investing › ETFs) exactly
    as the template's ETF_LT gets it, dated; None when the night has none
    (no holdings archived, or a coverage-degraded run)."""
    if not payload or not payload.get('funds'):
        return None
    return {'schema_version': _APP_SCHEMA_VERSION, 'date': run_date, **payload}


# The report's Portfolios view, for the app: what the inline PORTFOLIOS
# payload already publishes, and no more. Fields are taken by allow-list, so
# anything a definition grows later (strategy targets, say) stays out until
# someone decides the app should carry it; a held portfolio's book-weighted
# NAV and the lots, cash and sales in positions.json never reach this
# payload in the first place (_portfolio_nav keeps them private).
_APP_PORTFOLIO_FIELDS = ('id', 'name', 'color', 'description', 'created', 'alerts', 'rule')


# A portfolio's News tab (the template's _pfNewsItems), built here because
# the app does not download every member's detail shard. The word lists are
# the template's, and tests/test_app_feed.py fails if the two drift apart.
_PFN_MAX = 60
_PFN_LEGAL = frozenset((
    'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited', 'plc', 'llc',
    'lp', 'l.p.', 'holdings', 'holding', 'group', 'sa', 'nv', 'ag', 'se', 'ab', 'the', 'class',
    'common', 'stock', 'adr', 'ads', 's.a.', 'n.v.', 'n.v', 'inc.', 'corp.', 'co.', 'ltd.'))
_PFN_GENERIC = frozenset((
    'american', 'general', 'united', 'first', 'national', 'international', 'global', 'advanced',
    'taiwan', 'china', 'new', 'southern', 'northern', 'western', 'eastern', 'energy', 'capital',
    'financial', 'bank', 'royal', 'the', 'applied', 'digital', 'us', 'u', 's', 'east', 'west',
    'north', 'south', 'pacific', 'atlantic', 'texas', 'canadian', 'japan', 'british', 'data',
    'service', 'services', 'health', 'healthcare', 'real', 'public', 'home', 'power', 'world',
    'best', 'city', 'century', 'great', 'old', 'standard', 'premier', 'universal', 'continental',
    'federal', 'state', 'ally', 'target', 'block', 'match', 'visa', 'progressive', 'ford'))
_PFN_STOPTK = frozenset((
    'A', 'AN', 'AI', 'ALL', 'ARE', 'BE', 'BIG', 'CAN', 'CEO', 'FOR', 'GO', 'HAS', 'IT', 'KEY',
    'NOW', 'ON', 'ONE', 'OR', 'OUT', 'SO', 'TV', 'UP', 'US', 'USA', 'WELL', 'ET', 'EV', 'IPO',
    'ETF', 'FAST', 'LOW', 'MAIN', 'RUN', 'SEE', 'TRUE', 'TWO', 'VS'))
# A headline's fields the app shows.
_PFN_ITEM_FIELDS = ('title', 'link', 'date', 'source', 'sentiment', 'tags')


def _pfn_word(w):
    """The template's _pfnRe: the phrase between non-alphanumerics, any case."""
    return re.compile(r'(?<![A-Za-z0-9])' + re.escape(w) + r'(?![A-Za-z0-9])', re.IGNORECASE)


def _pfn_matchers(ticker, row):
    """_pfnMatchers: what in a headline names this member. Its ticker
    (case-sensitive, an optional $), its full name, and a one-word name or
    a longer name's first two words."""
    out = []
    nm = re.sub(r'\(the\)', '', str((row or {}).get('company_name') or ''), flags=re.IGNORECASE)
    words = [w for w in nm.replace(',', ' ').split() if w.lower() not in _PFN_LEGAL]
    while words and re.fullmatch(r'and|&|of', words[-1], re.IGNORECASE):
        words.pop()
    words = [w for w in words if re.search(r'[A-Za-z0-9]', w)]
    base = str(ticker).split('-', 1)[0]
    if len(base) >= 2 and base not in _PFN_STOPTK:
        out.append(re.compile(r'(?<![A-Za-z0-9$])\$?' + re.escape(base) + r'(?![A-Za-z0-9])'))
    if len(words) > 1:
        out.append(_pfn_word(' '.join(words)))
    w0 = re.sub(r"[^A-Za-z0-9&'-]", '', words[0] if words else '')
    if len(words) == 1 and len(w0) >= 4 and w0.lower() not in _PFN_GENERIC:
        out.append(_pfn_word(w0))
    elif len(words) >= 3:
        out.append(_pfn_word(' '.join(words[:2])))
    return out


def _pfn_norm(title):
    return re.sub(r'[^A-Za-z0-9_]+', ' ', str(title).lower()).strip()


def _app_portfolio_news(members, by_tk, details):
    """_pfNewsItems: the members' company headlines that name a member,
    duplicates folded and each credited to every member it names, newest
    first (several members first within a day). ``{'items', 'total'}`` with
    at most _PFN_MAX items, the most the tab shows."""
    matchers = {t: _pfn_matchers(t, by_tk.get(t)) for t in members}
    seen, out = {}, []
    for t in members:
        feed = ((details or {}).get(t) or {}).get('news_headlines')
        if feed is None:
            feed = (by_tk.get(t) or {}).get('news_headlines')
        for n in feed or ():
            if not isinstance(n, dict) or not n.get('title'):
                continue
            if (n.get('scope') == 'sector') if n.get('scope') else (n.get('origin') == 'google_news'):
                continue
            norm = _pfn_norm(n['title'])
            key = n.get('link') or norm
            hit = seen.get(key, seen.get('t:' + norm))
            if hit is not None:
                if n.get('tier') in (0, 1) and t not in out[hit]['tks']:
                    out[hit]['tks'].append(t)
                continue
            tks = [u for u in members if any(r.search(n['title']) for r in matchers[u])]
            if n.get('tier') in (0, 1) and t not in tks:
                tks.append(t)
            if not tks:
                continue
            seen[key] = seen['t:' + norm] = len(out)
            out.append({'n': n, 'tks': tks})
    # The template's comparator, as three stable sorts, least significant
    # first: timestamp, then members named, then date.
    out.sort(key=lambda x: x['n'].get('timestamp') or 0, reverse=True)
    out.sort(key=lambda x: len(x['tks']), reverse=True)
    out.sort(key=lambda x: str(x['n'].get('date') or ''), reverse=True)
    items = [dict({k: x['n'][k] for k in _PFN_ITEM_FIELDS if x['n'].get(k) is not None}, tickers=x['tks'])
             for x in out[:_PFN_MAX]]
    return {'items': items, 'total': len(out)}


def _build_app_portfolios(payload, index, run_date, chart_records=None, details_payload=None):
    """``app/portfolios.json``: each portfolio's definition and members as
    resolved for the report, its News tab's headlines, the public NAV
    series, the published analytics (Value & risk; PF_PUB.analytics) and
    the day the changes compare against; None without any portfolios."""
    pfs = (payload or {}).get('portfolios') or []
    if not pfs:
        return None
    members = {}
    for tk, ids in (index or {}).items():
        for pid in ids:
            members.setdefault(pid, []).append(tk)
    by_tk = {r['ticker']: r for r in chart_records or () if isinstance(r, dict) and r.get('ticker')}
    out = []
    for pf in pfs:
        row = {k: pf.get(k) for k in _APP_PORTFOLIO_FIELDS if pf.get(k) is not None}
        row['members'] = sorted(members.get(pf.get('id'), []))
        row['missing'] = sorted(pf.get('missing') or [])
        if by_tk or details_payload:
            row['news'] = _app_portfolio_news(row['members'], by_tk, details_payload)
        out.append(row)
    return {
        'schema_version': _APP_SCHEMA_VERSION,
        'date': run_date,
        'portfolios': out,
        'nav': (payload or {}).get('nav'),
        'analytics': (payload or {}).get('analytics'),
        'prev_date': (payload or {}).get('prev_date'),
        'news_tag_labels': TAG_LABELS,
    }


def _write_app_feed(app_dir, chart_records, details_payload, rows, run_date,
                    sector_pool=None, portfolios=None, pf_index=None, etfs=None,
                    score_hist_dates=None):
    """Write ``app/latest.json``, ``app/detail/<TICKER>.json``,
    ``app/sectors.json``, ``app/screener.json``, ``app/portfolios.json`` and
    ``app/manifest.json``;
    return the tickers written.

    The manifest goes last, so the app can tell a complete night from a
    half-written one: with no manifest, the app keeps the night it already
    has instead of mixing two runs. Rebuilt wholesale each run, like vol/
    and hist/, so a ticker that leaves the universe leaves no shard behind.

    ``files`` hashes only ``latest.json``, not the per-ticker shards, which
    narrows the handoff's section 4 deliberately. Hashing 3,859 shards made
    the manifest 363 KiB, and the app fetches it on every launch and
    foreground to answer one question — is there a new night. The shards
    carry their own ``date`` instead, so a detail can still be matched to
    the list it was fetched for, and ``tickers`` still says which shards
    that night has. Bump _APP_SCHEMA_VERSION if they are ever hashed again:
    an app that checks a hash it no longer finds would reject every night.
    """
    _written = []
    try:
        if not run_date:
            # Without a run date the app can neither date a number nor judge
            # staleness, so no feed beats a feed it must not trust.
            print("[report_html] app/: skipped (no run date)")
            return []
        # analyze_stock passes a string, the nightly re-render a date.
        # The feed always stores YYYY-MM-DD, which is what the app parses.
        _date = run_date.isoformat() if isinstance(run_date, date) else str(run_date)
        _date = _date[:10]
        _detail_dir = os.path.join(app_dir, 'detail')
        _latest_path = os.path.join(app_dir, 'latest.json')
        _manifest_path = os.path.join(app_dir, 'manifest.json')
        if os.path.isdir(app_dir):
            shutil.rmtree(app_dir)
        os.makedirs(_detail_dir, exist_ok=True)

        _write_app_json(_latest_path,
                        _build_app_list(chart_records, _date,
                                        _app_risk_free_rate(rows),
                                        score_hist_dates))
        # The one file the app fetches before it trusts the night.
        _files = {'app/latest.json': _sha256_file(_latest_path)}

        for _rec in sorted(chart_records or (),
                           key=lambda r: str(r.get('ticker') or '')):
            _tk = _rec.get('ticker')
            if not _tk or not _SHARD_TICKER.fullmatch(str(_tk)):
                continue
            # The row's remaining fields only: the app already holds the
            # list feed's copy of the rest before it opens a detail.
            _detail = {_k: _v for _k, _v in _rec.items()
                       if _k not in _APP_LIST_FIELDS}
            # The heavy text _extract_details_payload lifted out of the row,
            # merged flat exactly as the template merges details/ into DATA.
            for _k, _v in ((details_payload or {}).get(_tk) or {}).items():
                _detail.setdefault(_k, _v)
            _detail['ticker'] = _tk
            # Each shard names its own night, so a detail served from the
            # app's cache can be matched to the list without the manifest.
            _detail['date'] = _date
            _write_app_json(os.path.join(_detail_dir, f'{_tk}.json'), _detail)
            _written.append(_tk)

        # Sector Analysis. Dated like the detail shards rather than hashed,
        # and best-effort: a failure here costs the app its sector pages for
        # the night, not the night.
        try:
            _write_app_json(os.path.join(app_dir, 'sectors.json'),
                            _build_app_sectors(chart_records, sector_pool, _date))
        except Exception as _se:
            print(f"[warn] app/sectors.json write failed: {_se}")
        # The screener: best-effort the same way.
        try:
            _write_app_json(os.path.join(app_dir, 'screener.json'),
                            _build_app_screener(chart_records, _date))
        except Exception as _ce:
            print(f"[warn] app/screener.json write failed: {_ce}")
        # The Gate Matrix and the Dashboard's pass rates: best-effort the
        # same way.
        try:
            _write_app_json(os.path.join(app_dir, 'gates.json'),
                            _build_app_gates(chart_records, _date))
        except Exception as _ge:
            print(f"[warn] app/gates.json write failed: {_ge}")
        # The ETF look-through: best-effort the same way, and only when the
        # night has one.
        try:
            _etfs = _build_app_etfs(etfs, _date)
            if _etfs:
                _write_app_json(os.path.join(app_dir, 'etfs.json'), _etfs)
        except Exception as _ee:
            print(f"[warn] app/etfs.json write failed: {_ee}")
        # Portfolios: best-effort the same way, and only when there are any.
        try:
            _pfs = _build_app_portfolios(portfolios, pf_index, _date,
                                         chart_records=chart_records, details_payload=details_payload)
            if _pfs:
                _write_app_json(os.path.join(app_dir, 'portfolios.json'), _pfs)
        except Exception as _pe:
            print(f"[warn] app/portfolios.json write failed: {_pe}")

        with open(_manifest_path, 'w', encoding='utf-8') as _mf:
            json.dump({
                'schema_version': _APP_SCHEMA_VERSION,
                'date': _date,
                'generated_at': datetime.now(timezone.utc)
                                        .replace(microsecond=0).isoformat(),
                'count': len(_written),
                # publish_vol_shards.py syncs app/detail by this list rather
                # than by directory listing, as it does the other families.
                'tickers': _written,
                'files': dict(sorted(_files.items())),
            }, _mf, separators=_COMPACT, allow_nan=False)
        print(f"[report_html] app/: {len(_written)} detail shard(s), "
              f"latest.json {os.path.getsize(_latest_path) / 1048576:.2f} MiB, "
              f"manifest {os.path.getsize(_manifest_path) / 1024:.0f} KiB")
    except Exception as _e:
        # Never fail a render over the app feed. The absent manifest is
        # itself the signal, and the web report is unaffected.
        print(f"[warn] app/ feed write failed: {_e}")
        _written = []
    return _written


def stopped_map(tickers, prices_dir, as_of, what='row'):
    """``{ticker: (last_bar, lag_bars)}`` for *tickers* whose price data shows
    they stopped trading as of *as_of* (``data.price_store.stopped_trading``
    with the run's thresholds), or {} when the rule cannot judge.

    {} without *prices_dir* or an explicit *as_of* — judging an old snapshot
    against today's parquets would flag names that were live on its date —
    when the check fails, or when so many read as stopped that it is a failed
    price refresh (``mass_stop``), which is logged. Shared by the render and
    the portfolio alerts so both leave out the same names.
    """
    tickers = {str(t) for t in tickers if t}
    if not prices_dir or as_of is None or not tickers:
        return {}
    try:
        stopped = stopped_trading(prices_dir, tickers, as_of,
                                  max_lag_bars=CARRY_FORWARD_MAX_PRICE_LAG_BARS,
                                  spy_max_age_days=PHASE1_LOCAL_PRICE_MAX_AGE_DAYS)
    except Exception as e:
        logger.warning("stopped-trading check failed (%s); keeping every %s", e, what)
        return {}
    if stopped and mass_stop(len(stopped), len(tickers), CARRY_FORWARD_STOPPED_MAX_SHARE,
                             CARRY_FORWARD_STOPPED_GUARD_FLOOR):
        logger.warning("%d of %d %s(s) read as stopped trading as of %s — treating it "
                       "as a failed price refresh and keeping every %s",
                       len(stopped), len(tickers), what, as_of, what)
        return {}
    return stopped


def _drop_stopped_rows(rows, prices_dir, run_date):
    """*rows* without the tickers whose price data shows they stopped trading.

    Phase 1 now drops such carry-forwards before they are analysed, but the
    snapshots written before that rule — and any row that reached Phase 2
    another way — still hold acquired companies at their frozen last quote
    (JHG at $51.95 for weeks after its last bar). The render applies the same
    rule (``stopped_map``) so the table, portfolio stats and sidecars never
    show them. run.sh step 05e refreshes every row's parquet just before the
    re-render, which is what makes a lagging file evidence.

    Needs an explicit *run_date* (see ``stopped_map``). The snapshot JSON is
    left as it is — it stays the canonical record of what the run saw.
    """
    if not rows:
        return rows
    stopped = stopped_map((r.get('ticker') for r in rows if isinstance(r, dict)),
                          prices_dir, run_date)
    if not stopped:
        return rows
    for tk, (last_bar, lag) in sorted(stopped.items()):
        logger.warning("%s: left out of the report — last price bar %s is %d SPY "
                       "trading days old (stopped trading)", tk, last_bar, lag)
    print(f"[report_html] left out {len(stopped)} row(s) whose price data "
          f"stopped: {', '.join(sorted(stopped))}")
    return [r for r in rows
            if not (isinstance(r, dict) and str(r.get('ticker')) in stopped)]


def _load_prev_fv(out_dir, run_date, tickers, max_lookback=ETF_FV_CARRY_SNAPSHOTS):
    """``{ticker: _fv_effective}`` — each of *tickers*' last non-null fair
    value over the *max_lookback* snapshots before *run_date*, from the
    snapshot store when it holds that window, else the JSON files."""
    from data.snapshot_store import list_snapshot_files, read_snapshot
    want = set(tickers)
    if not want or run_date is None:
        return {}
    cur = run_date.isoformat()
    dated = sorted(((d, p) for d, p in list_snapshot_files(out_dir) if d < cur),
                   reverse=True)[:max_lookback]
    if not dated:
        return {}
    store = _open_snapshot_store(out_dir)
    if store is not None:
        try:
            with store:
                have = store.dates(before=cur)[-len(dated):]
                if have == sorted(d for d, _ in dated):
                    _, rows, _ = store.last_known_rows(
                        cur, ['_fv_effective'], max_lookback=len(dated),
                        require='_fv_effective')
                    return {t: r['_fv_effective'] for t, r in rows.items() if t in want}
        except Exception as e:
            print(f"[report_html] ETF prior fair values: store read failed, parsing JSON: {e}")
    out = {}
    for _, p in dated:
        try:
            recs = read_snapshot(p).get('results') or []
        except Exception as e:
            print(f"[report_html] ETF prior fair values: {p} unreadable: {e}")
            continue
        for rec in recs:
            tk = rec.get('ticker') if isinstance(rec, dict) else None
            if tk in want and tk not in out and rec.get('_fv_effective'):
                out[tk] = rec['_fv_effective']
        if len(out) == len(want):
            break
    return out


def _holding_drift(prices_dir, as_of, run_date, tickers):
    """``{ticker: Padj(run) / Padj(as_of)}`` for a holdings file older than
    the run, so its weights move with prices as the fund's did."""
    if not prices_dir or run_date is None or not as_of or as_of >= run_date.isoformat():
        return {}
    from data.price_store import asof_closes
    closes = asof_closes(prices_dir, [as_of, run_date.isoformat()], tickers) or {}
    a, b = closes.get(as_of) or {}, closes.get(run_date.isoformat()) or {}
    return {t: b[t][1] / a[t][1] for t in tickers
            if t in a and t in b and a[t][1] > 0}


def _build_etf_payload(rows, out_dir, run_date, prices_dir=None, run_provenance=None,
                       holdings_root=None):
    """The SPDR look-through (models/etf_lookthrough.py) over the rendered
    rows: ``{base, funds: {T: result}, sector_fund: {sector: T}}``, or None
    when no holdings are archived or the run is coverage-degraded (missing
    rows would move every fund's value). Also written to
    ``etfs_<date>.json`` beside the report, the dated record of the read."""
    cov = (run_provenance or {}).get('coverage') or {}
    if cov.get('degraded'):
        print("[report_html] ETF look-through skipped: run is coverage-degraded")
        return None
    root = holdings_root or os.environ.get('ETF_HOLDINGS_DIR') or os.path.join(out_dir, 'etf_holdings')
    held = {}
    for fund in etf_holdings.fund_tickers():
        h = etf_holdings.load_holdings(root, fund, run_date)
        if h:
            held[fund] = h
    if not held:
        return None
    by_tk = {r['ticker']: r for r in rows if isinstance(r, dict) and r.get('ticker')}
    aliases = {a: r['ticker'] for r in by_tk.values() for a in (r.get('listing_aliases') or [])}
    need = set()
    for h in held.values():
        for x in h['holdings']:
            r = by_tk.get(aliases.get(x['ticker'], x['ticker']))
            if r is not None and not r.get('_fv_effective'):
                need.add(r['ticker'])
    prev_fv = _load_prev_fv(out_dir, run_date, need)
    funds = {}
    for fund, h in held.items():
        tks = sorted({aliases.get(x['ticker'], x['ticker']) for x in h['holdings']
                      if x['ticker'] in by_tk or x['ticker'] in aliases})
        funds[fund] = etf_lookthrough(h, by_tk, aliases, prev_fv,
                                      _holding_drift(prices_dir, h['as_of'], run_date, tks))
        reg = etf_holdings.ETF_FUNDS.get(fund) or {}
        funds[fund]['sector'] = reg.get('sector')
        funds[fund]['category'] = reg.get('category') or 'broad'
    relative_to(funds, base='SPY')
    stale = sorted(f for f, h in held.items()
                   if run_date is not None and h['as_of'] < run_date.isoformat())
    ok = [f for f, r in funds.items() if r.get('status') == 'ok']
    print(f"[report_html] ETF look-through: {len(ok)}/{len(funds)} funds read"
          + (f", {len(prev_fv)} fair value(s) carried from prior snapshots" if prev_fv else '')
          + (f"; holdings older than the run for {', '.join(stale)}" if stale else ''))
    payload = {'base': 'SPY', 'funds': funds,
               'sector_fund': {s: f for f, s in etf_holdings.SPDR_FUNDS.items() if s and f in funds}}
    if run_date is not None:
        path = os.path.join(out_dir, f'etfs_{run_date.isoformat()}.json')
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'date': run_date.isoformat(), **payload}, f,
                      default=_json_default, separators=(',', ':'), sort_keys=True)
        os.replace(tmp, path)
    return payload


BOND_REPORT_VERSION = 1      # bonds/: REPORT_DATA_VERSION we draw
CURRENCY_REPORT_VERSION = 1  # currencies/: REPORT_DATA_VERSION we draw

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bond_output_dir():
    """The bond model's output/ (``BOND_MODEL_OUTPUT`` overrides)."""
    return os.environ.get('BOND_MODEL_OUTPUT') or os.path.join(_REPO_ROOT, 'bonds', 'output')


def _currency_output_dir():
    """The currency screener's output/ (``CURRENCY_MODEL_OUTPUT`` overrides)."""
    return os.environ.get('CURRENCY_MODEL_OUTPUT') or os.path.join(
        _REPO_ROOT, 'currencies', 'output')


def _stage_model_report(out_dir, run_date, src_dir, prefix, dest_name, version, label,
                        index=None):
    """Copy a subproject's newest ``<prefix>_<date>.json`` on or before
    *run_date* beside the report as *dest_name*, the sidecar its Financial
    Model tab loads when opened. Returns ``{date, total}`` for the page, or
    None (and removes a stale *dest_name*) when there is no report or its
    shape is not *version*, the one the template draws. Display only.
    *index*, given the report, returns a small list inlined as the meta's
    ``index`` so the top-bar search can find rows before the sidecar loads."""
    dest = os.path.join(out_dir, dest_name)
    cutoff = run_date.isoformat() if run_date is not None else '9999-12-31'
    pat = re.escape(prefix) + r'_(\d{4}-\d{2}-\d{2})\.json'
    picks = sorted(n for n in (os.listdir(src_dir) if os.path.isdir(src_dir) else [])
                   if (m := re.fullmatch(pat, n)) and m.group(1) <= cutoff)
    data = None
    if picks:
        src = os.path.join(src_dir, picks[-1])
        try:
            with open(src, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("%s report %s unreadable: %s", label, src, e)
        if data is not None and data.get('version') != version:
            logger.warning("%s report %s is version %s, the template draws %s",
                           label, src, data.get('version'), version)
            data = None
    if data is None:
        if os.path.exists(dest):
            os.remove(dest)
        print(f"[report_html] {label}: no report on or before {cutoff} in {src_dir}")
        return None
    tmp = dest + '.tmp'
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    print(f"[report_html] {label}: {picks[-1]} ({data.get('total', 0):,} rows)")
    meta = {'date': data.get('run_date'), 'total': data.get('total')}
    if index is not None:
        meta['index'] = index(data)
    return meta


def _stage_paper_dashboard(out_dir, run_date, pfs):
    """The Alpaca paper account against the model, as ``paper.json`` beside
    the report (``models/paper_dashboard``), for the Paper account sub-tab
    of ``PAPER_PORTFOLIO``'s page. Public by choice: a paper account's
    equity and positions. Runs after the portfolio payload, which advances
    the NAV ledger to this run, so model and account cover the same dates.
    Returns ``{portfolio, as_of}`` for the page, or None (and removes a stale
    ``paper.json``) when there is no such portfolio or no account record.
    Only the render on the machine that trades has the records; anywhere
    else the tab is simply absent."""
    from data import paper_account as pa
    from data import portfolio_nav as pn
    from models import paper_dashboard as pd
    from scripts import config
    dest = os.path.join(out_dir, 'paper.json')
    pid = config.PAPER_PORTFOLIO
    data = None
    if any(p.get('id') == pid for p in pfs or ()):
        led = pn.load_ledger(pn.ledger_path(out_dir))
        data = pd.build(pid, pa.read_ledger(), pa.load_nav(),
                        led['portfolios'].get(pid, []), led['bench'].get('spy', []),
                        run_date.isoformat() if run_date else None)
    if data is None:
        if os.path.exists(dest):
            os.remove(dest)
        return None
    tmp = dest + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_sanitize(data), f, separators=(',', ':'), default=_json_default)
    os.replace(tmp, dest)
    print(f"[report_html] paper account: {len(data['holdings'])} positions/targets "
          f"as of {data['as_of']}")
    return {'portfolio': pid, 'as_of': data['as_of'], 'start': data['start']}


def _stage_bond_report(out_dir, run_date, src_dir=None):
    """bonds/'s ``bond_report_<date>.json`` as ``bonds.json``, for Financial
    Model › Bonds (``_stage_model_report``)."""
    return _stage_model_report(out_dir, run_date, src_dir or _bond_output_dir(),
                               'bond_report', 'bonds.json', BOND_REPORT_VERSION,
                               'Bond valuations')


def _stage_currency_report(out_dir, run_date, src_dir=None):
    """currencies/'s ``currency_report_<date>.json`` as ``currencies.json``,
    for Investing › Currencies (``_stage_model_report``)."""
    return _stage_model_report(out_dir, run_date, src_dir or _currency_output_dir(),
                               'currency_report', 'currencies.json',
                               CURRENCY_REPORT_VERSION, 'Currencies',
                               index=_currency_search_index)


def _currency_search_index(data):
    """``[code, name]`` per currency (~30), for the top-bar search."""
    return [[r.get('code'), r.get('name') or ''] for r in data.get('rows') or []
            if r.get('code')]


def build_html(rows, filename, prices_dir=None, run_date=None, run_provenance=None,
               macro_payload=None):
    """Render the interactive HTML report via Jinja2 template."""
    rows = _drop_stopped_rows(_sanitize(rows), prices_dir, run_date)
    _r2000 = _load_russell2000()
    # Prior-run ratings for the "Δ vs prior" column. Sourced from the most
    # recent earlier results_*.json sitting next to this HTML output.
    try:
        _out_dir_early = os.path.dirname(os.path.abspath(filename)) or '.'
    except OSError:
        # abspath needs getcwd(), which can raise EPERM if the working
        # directory became unreadable mid-run (e.g. a macOS TCC revocation).
        _out_dir_early = os.path.dirname(filename) or '.'
    gate_meta_obj = gate_metadata()
    # Per-gate raw values + scores from the prior snapshot feed the popup's
    # "why the rating changed" explanation.
    _prev_gate_keys = [k for g in gate_meta_obj['gates']
                       for k in (g['key'], g['scoreKey'])]
    prev_run_date, _prev_ratings = _load_prev_ratings(
        _out_dir_early, run_date, extra_keys=_prev_gate_keys)
    if _prev_ratings:
        print(f"[report_html] rate-change baseline: {prev_run_date} "
              f"({len(_prev_ratings)} tickers)")
    else:
        print("[report_html] rate-change baseline: none found (no rating changes shown)")
    # Long-horizon rating change-points for the popup's "BUY since ..." line.
    _rating_hist = _load_rating_history(_out_dir_early, run_date)
    # Re-scored composite history: the score trend and the prior-run score.
    _score_series = _load_score_history(_out_dir_early, run_date)
    if _score_series.get('dates'):
        print(f"[report_html] score history: {len(_score_series['dates'])} weekly "
              f"points, prior {_score_series.get('prev_date')}"
              + (" (stale: other model)" if _score_series.get('stale') else ""))
    else:
        print("[report_html] score history: none (prior scores from the stored snapshot)")
    avg_spread, qualifying_with_mos, buy_count, lean_buy_count = \
        _summary_stats(rows)

    chart_records = [_row_context(r, gate_meta_obj, _r2000, _prev_ratings,
                                  _rating_hist, _score_series) for r in rows]
    portfolios_payload, _pf_index = _load_portfolio_payload(rows, _out_dir_early, run_date,
                                                            prices_dir, _rating_hist)
    for _rec in chart_records:
        # Portfolio ids this ticker belongs to (resolved server-side; the
        # client re-resolves only portfolios edited in the browser).
        _rec['pf'] = _pf_index.get(str(_rec['ticker']).upper(), [])
    _attach_data_summaries(chart_records)
    _attach_profiles(chart_records)

    details_payload = _extract_details_payload(chart_records)

    chart_data = dumps_for_script(pack_rows(chart_records), default=_json_default,
                                  separators=(',', ':'))

    # Gate metadata for Matrix view rendering in JavaScript
    gate_meta = dumps_for_script(gate_meta_obj, default=_json_default)

    sector_pool_data = _build_sector_pool_data(
        rows, (macro_payload or {}).get('sidecar'))
    sector_pool_json = dumps_for_script(sector_pool_data, default=_json_default)

    try:
        etf_payload = _build_etf_payload(rows, _out_dir_early, run_date, prices_dir,
                                         run_provenance)
    except Exception as e:  # display only: never fail a render over it
        logger.warning("ETF look-through failed: %s", e, exc_info=True)
        etf_payload = None

    try:
        bond_meta = _stage_bond_report(_out_dir_early, run_date)
    except Exception as e:  # display only: never fail a render over it
        logger.warning("Bond valuations failed: %s", e, exc_info=True)
        bond_meta = None

    try:
        paper_meta = _stage_paper_dashboard(_out_dir_early, run_date,
                                            (portfolios_payload or {}).get('portfolios'))
    except Exception as e:  # display only: never fail a render over it
        logger.warning("Paper account dashboard failed: %s", e, exc_info=True)
        paper_meta = None

    try:
        ccy_meta = _stage_currency_report(_out_dir_early, run_date)
    except Exception as e:  # display only: never fail a render over it
        logger.warning("Currencies failed: %s", e, exc_info=True)
        ccy_meta = None

    # Sidecar JSON files (PRICES, HIST) live next to the HTML output. The
    # template lazy-fetches them on first chart open, so the embedded HTML
    # stays small enough to publish via GitHub Pages (<100 MB hard cap) and
    # every file under Cloudflare Pages' 25 MiB per-file limit.
    try:
        out_dir = os.path.dirname(os.path.abspath(filename)) or '.'
    except OSError:
        # Same getcwd()/EPERM guard as _out_dir_early above.
        out_dir = os.path.dirname(filename) or '.'
    hist_path = os.path.join(out_dir, 'hist.json')   # legacy monolith; now only removed
    hist_dir = os.path.join(out_dir, 'hist')
    hist_index_path = os.path.join(out_dir, 'hist_index.json')
    prices_path = os.path.join(out_dir, 'prices.json')  # legacy monolith; now only removed
    prices_meta_path = os.path.join(out_dir, 'prices_meta.json')
    details_path = os.path.join(out_dir, 'details.json')   # legacy monolith; now only removed
    details_dir = os.path.join(out_dir, 'details')
    details_index_path = os.path.join(out_dir, 'details_index.json')
    macro_path = os.path.join(out_dir, 'macro.json')
    vol_dir = os.path.join(out_dir, 'vol')
    px_dir = os.path.join(out_dir, 'px')
    app_dir = os.path.join(out_dir, 'app')

    details_parts = _write_details_parts(details_dir, details_index_path, details_payload,
                                         legacy_path=details_path)

    macro_sidecar = (macro_payload or {}).get('sidecar')
    _write_macro_sidecar(macro_path, macro_sidecar)

    hist_payload = _build_hist_payload(rows, prices_dir)
    hist_tickers = _write_hist_shards(hist_dir, hist_index_path, hist_payload, legacy_path=hist_path)

    prices_payload, px_payload, vol_payload = \
        _load_price_payloads(rows, prices_dir)
    prices_size_mb = _write_prices_meta_sidecar(
        prices_meta_path, prices_path, prices_payload)
    _write_px_shards(px_dir, px_payload)
    _write_vol_shards(vol_dir, vol_payload)
    # The iOS app's feed, written after every sidecar it mirrors: its
    # manifest is what tells the app the night is complete.
    _write_app_feed(app_dir, chart_records, details_payload, rows, run_date,
                    sector_pool=sector_pool_data, portfolios=portfolios_payload,
                    pf_index=_pf_index, etfs=etf_payload,
                    score_hist_dates=([] if _score_series.get('stale')
                                      else _score_series.get('dates')))

    # Render Jinja2 template
    template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=False,
    )
    template = env.get_template('report.html')
    html = template.render(
        avg_spread_fmt=f"{avg_spread:.1%}",
        qualifying_with_mos=qualifying_with_mos,
        buy_count=buy_count,
        lean_buy_count=lean_buy_count,
        chart_data=chart_data,
        gate_meta=gate_meta,
        score_hist_meta=dumps_for_script({
            'dates': [] if _score_series.get('stale') else _score_series.get('dates', []),
            'prev_date': _score_series.get('prev_date'),
            'stale': bool(_score_series.get('stale'))}),
        portfolios_json=dumps_for_script(portfolios_payload, default=_json_default),
        sector_pool_json=sector_pool_json,
        etf_json=dumps_for_script(_sanitize(etf_payload), default=_json_default),
        bond_meta=dumps_for_script(bond_meta),
        ccy_meta=dumps_for_script(ccy_meta),
        paper_meta=dumps_for_script(paper_meta),
        prices_available=('true' if prices_payload is not None else 'false'),
        prices_size_mb=prices_size_mb,
        hist_available=('true' if hist_tickers else 'false'),
        hist_tickers=dumps_for_script(hist_tickers),
        news_tag_labels=dumps_for_script(TAG_LABELS),
        details_available=('true' if details_parts else 'false'),
        details_parts=dumps_for_script(details_parts),
        macro_available=('true' if macro_sidecar else 'false'),
        macro_summary=dumps_for_script(
            _sanitize((macro_payload or {}).get('summary')) or None,
            default=_json_default),
        generated_at=(run_date or date.today()).strftime('%B %-d, %Y'),
        run_date_iso=(run_date or date.today()).isoformat(),
        prev_run_date=(prev_run_date or ''),
    )

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
