"""Re-score a saved snapshot JSON and regenerate the HTML report.

Use this after editing scoring.py / report_html.py to refresh today's HTML
without re-running the 3-6h analysis. Loads the JSON, re-applies the canonical
scoring pipeline, and writes a new HTML next to the JSON.

Usage:
    python scripts/rescore_and_render.py output/results_YYYY-MM-DD.json
"""
import argparse
import json
import os
import sys
from datetime import date

import pandas as pd

# Ensure repo root is on sys.path so `scripts.*` imports resolve when invoked
# directly (i.e. `python scripts/rescore_and_render.py ...`).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.snapshot_store import (read_snapshot, sync_snapshot_file,
                                 write_snapshot_file)
from scripts.scoring import score_and_rate
from scripts.report_html import build_html
from scripts.analyze_stock import (derive_edgar_metrics, _load_local_prices,
                                   compute_multiple_vs_history,
                                   compute_price_metrics, _round_price_metrics)


def _refresh_edgar_derived_metrics(results):
    """Re-derive multi-year CAGRs from edgar_history on each row.

    Necessary after backfill_edgar_hist populates new history into a
    snapshot — without this, the scoring pipeline would still see the
    stale None values that were computed during the original live run
    when edgar_history was empty.

    FX safety: this rescore path never re-fetches yfinance data, so the
    USD-normalized financials persisted by analyze_stock.py (rows with
    ``fx_converted=True``) are not at risk of double-conversion. CAGRs
    derived here come from edgar_history values that are growth ratios —
    currency-neutral by construction.
    """
    n = 0
    for r in results:
        eh = r.get('edgar_history')
        if not eh:
            continue
        m = derive_edgar_metrics(eh)
        for k, v in m.items():
            r[k] = v
        n += 1
    print(f'Refreshed EDGAR-derived metrics for {n} rows')


def _refresh_multiple_vs_history(results, prices_dir):
    """Compute the time-series cheapness metric (Valuation: Mult vs Hist)
    from local parquet prices + edgar_history for snapshots that predate the
    field (or after a backfill). Rows whose parquet can't be loaded are
    left UNTOUCHED — a live-run value must survive a rescore on a machine
    without the prices dir, so we only overwrite when we could actually
    recompute the inputs."""
    n = skipped = 0
    for r in results:
        if not r.get('edgar_history'):
            continue
        close = _load_local_prices(r.get('ticker'), prices_dir)
        if close is None or len(close) == 0:
            skipped += 1
            continue
        mvh, yrs = compute_multiple_vs_history(
            close, r.get('edgar_history'), r.get('operating_income'))
        r['mult_vs_hist'] = mvh
        r['mult_hist_years'] = yrs
        if mvh is not None:
            n += 1
    print(f'Refreshed Mult-vs-Hist for {n} rows'
          + (f' ({skipped} without local prices left as-is)' if skipped else ''))


def _refresh_price_metrics(results, prices_dir, as_of=None):
    """Compute the market/risk metrics (momentum, liquidity) from local parquet.

    Same contract as _refresh_multiple_vs_history: rows whose parquet can't be
    read are left UNTOUCHED so a live-run value survives a rescore on a machine
    without the prices dir.

    *as_of* is the snapshot date, not today — otherwise re-rendering an older
    snapshot would measure every trailing window to the wrong endpoint and
    declare the whole universe stale.
    """
    n = skipped = stale = 0
    for r in results:
        metrics = compute_price_metrics(r.get('ticker'), prices_dir, as_of=as_of)
        if metrics['price_data_stale'] is None:
            # No usable local price history — leave whatever the row already had.
            skipped += 1
            continue
        if metrics['price_data_stale']:
            stale += 1
        r.update(_round_price_metrics(metrics))
        if metrics['momentum_12_1'] is not None:
            n += 1
    print(f'Refreshed price metrics for {n} rows'
          + (f' ({stale} stale-priced)' if stale else '')
          + (f' ({skipped} without local prices left as-is)' if skipped else ''))


def _macro_payload_for_render(html_path, run_date, snap):
    """Macro Outlook payload for the re-render path.

    Step 1f of the daily routine re-renders the HTML after the enrichment
    passes. Rendering with no payload would drop the Macro Outlook tab AND
    delete the macro.json the main run wrote, so the re-render has to supply
    one of its own. Two sources, in order:

      1. A fresh FRED build, as_of the snapshot's own date so the macro
         numbers match the run the HTML is pinned to.
      2. Failing that, the sidecar already sitting next to the HTML — a FRED
         outage during a re-render must not remove a tab that worked minutes
         ago.

    Returns None only when neither is available, which correctly renders the
    report without the tab.
    """
    regime = snap.get('macro_regime') if isinstance(snap, dict) else None
    adj = snap.get('macro_adjustments') if isinstance(snap, dict) else None
    local_rs = snap.get('sector_local_rs') if isinstance(snap, dict) else None
    try:
        from data.fred_client import FREDClient
        from scripts.macro_dashboard import build_macro_payload, make_narrative_client
        # No sector_etf_data here — the snapshot doesn't store ETF metrics;
        # local_rs still gives the narrative per-sector relative strength.
        # The narrative's per-as_of cache makes the daily re-render free.
        payload = build_macro_payload(FREDClient(), regime, adj, as_of=run_date,
                                      local_rs=local_rs,
                                      narrative_client=make_narrative_client())
        if payload:
            print(f"Macro dashboard: {len(payload['sidecar']['series'])} FRED series")
            return payload
    except Exception as e:
        print(f'  Macro dashboard rebuild failed ({e}).')

    macro_path = os.path.join(os.path.dirname(html_path) or '.', 'macro.json')
    try:
        from scripts.macro_dashboard import summary_from_sidecar
        with open(macro_path, encoding='utf-8') as fh:
            sidecar = json.load(fh)
        summary = summary_from_sidecar(sidecar)
        if summary:
            print('Macro dashboard: reused existing macro.json')
            return {'summary': summary, 'sidecar': sidecar}
    except (OSError, ValueError, ImportError):
        pass
    print('Macro dashboard: unavailable, rendering without the tab.')
    return None


def rescore_and_render(json_path, prices_dir='output/prices'):
    snap = read_snapshot(json_path)

    # Snapshot may be a bare list or wrapped in a dict with 'results' key
    if isinstance(snap, dict) and 'results' in snap:
        results = snap['results']
        snap_date = snap.get('date') or date.today().isoformat()
    else:
        results = snap
        snap_date = date.today().isoformat()

    _refresh_edgar_derived_metrics(results)
    _refresh_multiple_vs_history(results, prices_dir)
    try:
        _price_as_of = pd.Timestamp(snap_date).normalize()
    except (TypeError, ValueError):
        _price_as_of = None
    _refresh_price_metrics(results, prices_dir, as_of=_price_as_of)
    # Propagate the snapshot's risk-free rate onto each row so the
    # Valuation: FCF Yield gate has its hurdle available on re-render (the
    # live pipeline stamps this per row; here it comes from the snapshot).
    _rf = snap.get('risk_free_rate') if isinstance(snap, dict) else None
    if _rf is not None:
        for _r in results:
            _r['_risk_free_rate'] = _rf
    score_and_rate(results)

    # Write new HTML alongside the JSON.
    html_path = os.path.join(os.path.dirname(json_path) or '.',
                             f'stock_analysis_results_{snap_date}.html')
    run_provenance = snap.get('provenance') if isinstance(snap, dict) else None
    # Pin the banner's "Last updated" to the snapshot's own date, not today's.
    # Without this, re-rendering a snapshot on a later day (e.g. a delayed
    # enrichment pass) stamps the banner with date.today() instead of the run
    # date, contradicting the filename and the snapshot's `date` field.
    try:
        run_date = date.fromisoformat(snap_date)
    except (TypeError, ValueError):
        run_date = None
    build_html(results, html_path, prices_dir=prices_dir,
               run_date=run_date, run_provenance=run_provenance,
               macro_payload=_macro_payload_for_render(html_path, run_date, snap))
    print(f'Wrote {html_path}')

    # Persist the rescored JSON back so the snapshot is consistent with the HTML
    if isinstance(snap, dict) and 'results' in snap:
        snap['results'] = results
        out = snap
    else:
        out = results
    write_snapshot_file(json_path, out)
    print(f'Updated {json_path}')
    # Keep the DuckDB snapshot store in step with the rewritten file (the
    # enrichment scripts also rewrite it, so this re-sync covers steps 1b-1f).
    if sync_snapshot_file(json_path, data=out):
        print(f'Synced {json_path} into the snapshot store')

    # Quick sanity summary
    n = len(results)
    cs = [r['_composite_score'] for r in results if r.get('_composite_score') is not None]
    # Denominators are per-ticker now (applicability mask excludes gates
    # that can't describe a business), so report the observed range.
    denoms = sorted({int(r['_gates_passed'].split('/')[1])
                     for r in results if r.get('_gates_passed')})
    denoms = (f"{denoms[0]}-{denoms[-1]} (applicable gates)"
              if denoms else 'n/a')
    rating_dist = {}
    for r in results:
        rating_dist[r.get('rating', '?')] = rating_dist.get(r.get('rating', '?'), 0) + 1
    print('\nSummary:')
    print(f'  Tickers: {n}')
    print(f'  Gate denominators: {denoms}')
    if cs:
        print(f'  Composite score: min={min(cs):.1f}, '
              f'median={sorted(cs)[len(cs)//2]:.1f}, max={max(cs):.1f}')
    print(f'  Rating distribution: {rating_dist}')
    _print_cohort_parity(results)


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _print_cohort_parity(results):
    """Compare the DCF cohort vs the consensus-fallback cohort so the parity
    the consensus aims for is measurable (it isn't, if the two diverge)."""
    def pfv(sub):
        vals = [r['price'] / r['_fv_effective'] for r in sub
                if isinstance(r.get('_fv_effective'), (int, float)) and r['_fv_effective'] > 0
                and isinstance(r.get('price'), (int, float)) and r['price'] > 0]
        return _median(vals), len(vals)
    def ms_err(sub):
        # median |our FV / Morningstar FV − 1|, now that ms_fv is un-gated
        errs = [abs(r['_fv_effective'] / r['ms_fv'] - 1) for r in sub
                if isinstance(r.get('_fv_effective'), (int, float)) and r['_fv_effective'] > 0
                and isinstance(r.get('ms_fv'), (int, float)) and r['ms_fv'] > 0]
        return _median(errs), len(errs)

    dcf = [r for r in results if r.get('_fv_source') == 'dcf']
    blend = [r for r in results if r.get('_fv_source') == 'blend']
    for label, sub in (('DCF cohort', dcf), ('consensus cohort', blend)):
        m, k = pfv(sub)
        me, mk = ms_err(sub)
        m_s = f'{m:.2f}' if m is not None else '—'
        me_s = f'{me:.1%} (n={mk})' if me is not None else '—'
        print(f'  {label:16} n={len(sub):5d}  median P/FV_eff={m_s:>5}  '
              f'MS median |err|={me_s}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('json_path', help='Path to results_YYYY-MM-DD.json')
    ap.add_argument('--prices-dir', default='output/prices')
    args = ap.parse_args()
    rescore_and_render(args.json_path, prices_dir=args.prices_dir)
