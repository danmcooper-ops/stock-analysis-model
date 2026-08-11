# scripts/report_html.py
"""HTML report builder — renders the interactive Jinja2 report."""
import os
import json
import re
import shutil
import jinja2
from datetime import date
import numpy as np

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, '..') not in _sys.path:
    _sys.path.append(os.path.join(_HERE, '..'))
try:
    from models.narrative import generate_sector_profit_pool_narrative
except Exception:
    generate_sector_profit_pool_narrative = None
from scripts.scoring import gate_metadata
from scripts.safe_json import dumps_for_script


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
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sanitize(obj):
    """Recursively replace inf/-inf/NaN (numeric or stringified) with None.

    Some upstream snapshots round-trip numeric infinity through ``str()`` and
    arrive here as the literal string ``"Infinity"``, which silently breaks
    the browser-side popup formatters. Normalise those to ``None`` so the JS
    sees a clean ``null``.
    """
    import math
    if obj is None:
        return None
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, str):
        if obj in ('Infinity', '-Infinity', 'inf', '-inf', 'NaN', 'nan'):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize(x) for x in obj)
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
    string compare is correct) and walks them newest-first. For every ticker it
    records the rating from the *most recent* snapshot that contains it — i.e.
    "last known rating", not "rating in exactly the prior file". This makes the
    "Yesterday's Rating" column resilient to a single degraded run: a ticker
    that transiently dropped out of the immediately-prior snapshot (e.g. a
    yfinance/SEC fetch timeout) still shows its last real rating instead of N/A.

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
    import glob
    import re
    try:
        cur = run_date.isoformat() if run_date is not None else None
    except Exception:
        cur = None
    # Collect all prior snapshots, newest date first.
    dated = []
    for p in glob.glob(os.path.join(out_dir, 'results_*.json')):
        m = re.search(r'results_(\d{4}-\d{2}-\d{2})\.json$', os.path.basename(p))
        if not m:
            continue
        d = m.group(1)
        if cur is not None and d >= cur:
            continue
        dated.append((d, p))
    if not dated:
        return None, {}
    dated.sort(key=lambda t: t[0], reverse=True)
    primary_date = dated[0][0]
    out = {}
    filled_via_fallback = 0
    for d, p in dated[:max_lookback]:
        try:
            with open(p) as f:
                snap = json.load(f)
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


def _explain_rating_change(prev, cur, gate_meta):
    """Build the popup's "why the rating changed" bullet list.

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
            why.append('A rating cap fired this run: '
                       + '; '.join(new_r)
                       + f'. On the composite alone it would rate {c_raw}.')
        elif not p_capped:
            why.append('The rating is being held down by a cap: '
                       + ('; '.join(c_reasons) or 'see rating cap')
                       + f'. On the composite alone it would rate {c_raw}.')
    if up and p_capped and not c_capped:
        gone = [x for x in p_reasons if x not in c_reasons]
        why.append("The prior run's rating cap no longer applies"
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
            crossed = [f'{_RATING_BY_RANK[rk]} ({_RATING_THRESHOLD_BY_RANK[rk]})'
                       for rk in range(hi, lo, -1)]
            s += (', crossing ' + ('above' if up else 'below') + ' the '
                  + ' and '.join(crossed)
                  + (' thresholds' if len(crossed) > 1 else ' threshold'))
        elif p_raw == c_raw and p_raw is not None:
            s += ' (the composite-only rating band did not change)'
        why.append(s + '.')
    elif p_cs is None and c_cs is not None:
        why.append('The prior run had no composite score (insufficient scoring '
                   'data); this run scored it '
                   f'{c_cs:.0f}, rating {c_rt}.')
    elif p_cs is not None and c_cs is None:
        why.append('This run produced no composite score (insufficient scoring '
                   'data), so the rating fell back to the cap/default path.')

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
                   '(likely small moves straddling a rating threshold).')
    return why


def _load_rating_history(out_dir, run_date, cache_name='rating_history.json'):
    """Per-ticker rating change-points across prior snapshots, for the popup's
    "BUY since ..." context line.

    Returns ``{ticker: [[date, rating], ...]}`` — change-points only (the first
    observation plus every transition), ascending by date, filtered to dates
    strictly before ``run_date`` so re-rendering an old snapshot never sees
    "future" history.

    Parsing every historical results_*.json (~66MB each) on every render would
    be minutes of work, so an incremental cache (``out_dir/rating_history.json``)
    stores the accumulated change-points plus the last snapshot date scanned;
    each render only parses snapshots newer than that. Only strictly-newer dates
    are appended, so a late-backfilled older snapshot can't corrupt ordering
    (delete the cache to force a full rebuild that includes it). A missing or
    corrupt cache also triggers a full rebuild.
    """
    import glob
    import re
    try:
        cur = run_date.isoformat() if run_date is not None else None
    except Exception:
        cur = None
    dated = []
    for p in glob.glob(os.path.join(out_dir, 'results_*.json')):
        m = re.search(r'results_(\d{4}-\d{2}-\d{2})\.json$', os.path.basename(p))
        if m:
            dated.append((m.group(1), p))
    dated.sort()
    if not dated:
        return {}
    cache_path = os.path.join(out_dir, cache_name)
    hist, last_scanned = {}, None
    try:
        with open(cache_path) as f:
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
    for d, p in todo:
        try:
            with open(p) as f:
                snap = json.load(f)
        except Exception as e:
            print(f"[report_html] rating-history load failed ({p}): {e}")
            continue
        recs = snap.get('results') if (isinstance(snap, dict) and 'results' in snap) else snap
        if not isinstance(recs, list):
            continue
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
    if todo:
        try:
            with open(cache_path, 'w') as f:
                json.dump({'last_scanned': last_scanned, 'hist': hist}, f)
            print(f"[report_html] rating-history cache: +{len(todo)} snapshot(s), "
                  f"{len(hist)} tickers, through {last_scanned}")
        except Exception as e:
            print(f"[warn] rating-history cache write failed: {e}")
    if cur is None:
        return hist
    # Exclude entries on/after the render date (rescoring an older snapshot).
    return {tk: [cp for cp in seq if cp[0] < cur]
            for tk, seq in hist.items()}


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
        with open(path) as f:
            return {ln.strip().upper() for ln in f
                    if ln.strip() and not ln.startswith('#')}
    except OSError:
        print("[report_html] russell2000_tickers.txt not found — "
              "Russell 2000 quick filter will match nothing")
        return set()


def build_html(rows, filename, prices_dir=None, run_date=None, run_provenance=None):
    """Render the interactive HTML report via Jinja2 template."""
    rows = _sanitize(rows)
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
        print("[report_html] rate-change baseline: none found (Yesterday's Rating shows N/A)")
    # Long-horizon rating change-points for the popup's "BUY since ..." line.
    _rating_hist = _load_rating_history(_out_dir_early, run_date)
    spread_vals = [r['spread'] for r in rows if r.get('spread') is not None]
    avg_spread = sum(spread_vals) / len(spread_vals) if spread_vals else 0
    qualifying_with_mos = sum(1 for r in rows if r.get('mos') is not None and r['mos'] > 0)
    buy_count = sum(1 for r in rows if r.get('rating') == 'BUY')
    lean_buy_count = sum(1 for r in rows if r.get('rating') == 'LEAN BUY')

    chart_records = [{
        'ticker': r['ticker'],
        'roic': r.get('roic'), 'wacc': r.get('wacc'), 'spread': r.get('spread'),
        'dcf_fv': r.get('dcf_fv'), 'price': r.get('price'), 'mos': r.get('mos'),
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
            _prev_ratings.get(r.get('ticker')), r, gate_meta_obj),
        # Rating change-points (last 12) for "BUY since ..." in the popup.
        'rating_hist': (_rating_hist.get(r.get('ticker')) or [])[-12:],
        '_rating_cap': r.get('_rating_cap'),
        '_rating_cap_reasons': r.get('_rating_cap_reasons', []),
        'analyst_rec': r.get('analyst_rec'),
        'company_name': r.get('company_name', ''),
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
        'analyst_ltg': r.get('analyst_ltg'),
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
    } for r in rows]

    # Heavy text fields only consumed inside the detail panel. Strip them
    # from the inline DATA blob into a details.json sidecar the template
    # lazy-loads after first paint. Cuts the HTML by ~25 MB at 2k tickers.
    _DETAIL_HEAVY_KEYS = (
        'description_full', 'ceo_bio', 'culture_narrative',
        'financial_summary', 'sector_headwinds', 'sector_tailwinds',
        'news_headlines', 'news_sentiment',
        'legal_filings', 'insider_transactions',
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

    chart_data = dumps_for_script(chart_records, default=_json_default)

    # Gate metadata for Matrix view rendering in JavaScript
    gate_meta = dumps_for_script(gate_meta_obj, default=_json_default)

    # Per-sector profit pool narratives (top-level Profit Pool tab)
    sector_pool_data = {}
    if generate_sector_profit_pool_narrative is not None:
        _by_sector = {}
        for r in rows:
            s = r.get('sector')
            if not s or r.get('pp_revenue_share') is None:
                continue
            _by_sector.setdefault(s, []).append(r)
        for s, srows in _by_sector.items():
            try:
                narr = generate_sector_profit_pool_narrative(s, srows)
                if narr:
                    sector_pool_data[s] = narr
            except Exception as e:
                print(f"[warn] sector pool narrative failed for {s}: {e}")
    sector_pool_json = dumps_for_script(sector_pool_data, default=_json_default)

    # Sidecar JSON files (PRICES, HIST) live next to the HTML output. The
    # template lazy-fetches them on first chart open, so the embedded HTML
    # stays small enough to publish via GitHub Pages (<100 MB hard cap).
    try:
        out_dir = os.path.dirname(os.path.abspath(filename)) or '.'
    except OSError:
        # Same getcwd()/EPERM guard as _out_dir_early above.
        out_dir = os.path.dirname(filename) or '.'
    hist_path = os.path.join(out_dir, 'hist.json')
    prices_path = os.path.join(out_dir, 'prices.json')
    details_path = os.path.join(out_dir, 'details.json')
    vol_dir = os.path.join(out_dir, 'vol')

    # Every sidecar is written compact. json.dump's default separators put a
    # space after each comma, which on prices.json alone (11.6M array elements)
    # wasted 11.6 MB of the 100 MB Pages budget.
    _COMPACT = (',', ':')

    # Write details.json sidecar (or remove a stale one)
    try:
        if details_payload:
            with open(details_path, 'w') as _df:
                json.dump(details_payload, _df, default=_json_default,
                          separators=_COMPACT)
        elif os.path.exists(details_path):
            os.remove(details_path)
    except Exception as _e:
        print(f"[warn] details.json write failed: {_e}")

    # Historical fundamentals (EDGAR annual) per ticker
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
            eh = r.get('edgar_history') or {}
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
                _hist_out[tk] = tick_hist
        if _hist_out:
            hist_payload = _hist_out
            print(f"[report_html] edgar history: {len(_hist_out)} tickers")
    except Exception as _e:
        print(f"[warn] edgar history load failed: {_e}")

    # Write hist.json sidecar (or remove a stale one so old data doesn't linger)
    try:
        if hist_payload is not None:
            with open(hist_path, 'w') as _hf:
                json.dump(hist_payload, _hf, default=_json_default,
                          separators=_COMPACT)
        elif os.path.exists(hist_path):
            os.remove(hist_path)
    except Exception as _e:
        print(f"[warn] hist.json write failed: {_e}")

    # Load price history from local Parquet files
    prices_payload = None
    vol_payload = None
    if prices_dir and os.path.isdir(prices_dir):
        try:
            import pandas as _pd
            _cutoff = _pd.Timestamp.now().normalize() - _pd.DateOffset(years=20)
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
                    df = _pd.read_parquet(pf).sort_index()
                    df.index = _pd.to_datetime(df.index).tz_localize(None).normalize()
                    _keep = df.index >= _cutoff
                    s = df['Close'][_keep].dropna()
                    if len(s) >= 20:
                        _series[tk] = s
                        # Volume rides along for the popup chart's sub-panel.
                        # Never name it in a columns= projection: ~0.6% of the
                        # files were written Close-only by the yfinance_client
                        # fallback and the projection would raise on those.
                        if 'Volume' in df.columns:
                            _v = df['Volume'][_keep].dropna()
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
                                    _hi = df['High'][_keep]
                                    _lo = df['Low'][_keep]
                                    _rng = _hi - _lo
                                    _f = ((df['Close'][_keep] - _lo) / _rng)
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
                    _is = _idf['Close'][_idf.index >= _cutoff].dropna()
                    if len(_is) >= 20:
                        _series[_itk] = _is
                        _indices_found.append({'ticker': _itk, 'label': _ilabel})
                except Exception:
                    pass
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
                            arr[i] = round(float(v), 2)
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
                prices_payload = {
                    'dates': _date_strs,
                    'prices': _prices_out,
                    'indices': _indices_found,
                    # Tickers with a vol/<TICKER>.json shard. The client checks
                    # this instead of eating a 404 on every popup open.
                    'vol': sorted(_vol_out.keys()),
                    # Shard indices are positions into `dates`; the client bails
                    # rather than misalign if it ever sees mismatched artifacts.
                    'n': len(_date_strs),
                }
                vol_payload = _vol_out
                print(f"[report_html] price history: {len(_prices_out)} tickers, {len(_date_strs)} dates, indices: {[x['ticker'] for x in _indices_found]}")
                print(f"[report_html] volume shards: {len(_vol_out)} tickers")
        except Exception as _e:
            print(f"[warn] price history load failed: {_e}")

    # Write prices.json sidecar (or remove a stale one)
    prices_size_mb = 0
    try:
        if prices_payload is not None:
            with open(prices_path, 'w') as _pf2:
                json.dump(prices_payload, _pf2, default=_json_default,
                          separators=_COMPACT)
            # Uncompressed size, stamped into the client so its download
            # progress ("Loading price history… 12 / 62 MB") can be honest —
            # the transfer is gzipped, so Content-Length alone can't say how
            # much of the decompressed body has arrived.
            prices_size_mb = round(os.path.getsize(prices_path) / 1048576)
        elif os.path.exists(prices_path):
            os.remove(prices_path)
    except Exception as _e:
        print(f"[warn] prices.json write failed: {_e}")

    # Write the vol/ shard directory. Rebuilt wholesale each run so a ticker
    # that drops out of the universe can't leave a shard behind whose indices
    # no longer line up with the current dates array.
    try:
        if os.path.isdir(vol_dir):
            shutil.rmtree(vol_dir)
        if vol_payload:
            os.makedirs(vol_dir, exist_ok=True)
            for _tk, _sh in vol_payload.items():
                with open(os.path.join(vol_dir, f'{_tk}.json'), 'w') as _vf:
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
        sector_pool_json=sector_pool_json,
        prices_available=('true' if prices_payload is not None else 'false'),
        prices_size_mb=prices_size_mb,
        hist_available=('true' if hist_payload is not None else 'false'),
        details_available=('true' if details_payload else 'false'),
        generated_at=(run_date or date.today()).strftime('%B %-d, %Y'),
        run_date_iso=(run_date or date.today()).isoformat(),
        prev_run_date=(prev_run_date or ''),
    )

    with open(filename, 'w') as f:
        f.write(html)
