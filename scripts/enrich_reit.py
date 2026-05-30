"""Compute REIT-specific metric proxies and join them into the daily
results JSON.

For every Real Estate stock with sufficient edgar_history, derives:

  ffo_growth_5y      5-yr CAGR of operating cash flow (FFO Growth proxy)
  affo_margin        (CFO − capex) / revenue, latest year (AFFO Margin proxy)

Per data/reit_metrics.py for the rationale behind the proxies.

Usage:
    python scripts/enrich_reit.py output/results_YYYY-MM-DD.json [out]

Idempotent — strips any prior REIT enrichment before running. Pure
local computation, no network calls; the data comes from each stock's
existing edgar_history already in the JSON.
"""
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data.reit_metrics import cfo_cagr, affo_margin

_REIT_FIELDS = ("ffo_growth_5y", "affo_margin")


def _records(d):
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return None


def enrich(records, verbose=True):
    reit_total = sum(1 for r in records if r.get("sector") == "Real Estate")
    # Idempotency: strip any prior enrichment.
    for r in records:
        if r.get("sector") == "Real Estate":
            for k in _REIT_FIELDS:
                r.pop(k, None)
    n_ffo = 0
    n_affo = 0
    for r in records:
        if r.get("sector") != "Real Estate":
            continue
        eh = r.get("edgar_history")
        g = cfo_cagr(eh)
        m = affo_margin(eh)
        if g is not None:
            r["ffo_growth_5y"] = g
            n_ffo += 1
        if m is not None:
            r["affo_margin"] = m
            n_affo += 1
    if verbose:
        print(f"  Real Estate total:                 {reit_total}")
        print(f"  Enriched with FFO Growth proxy:    {n_ffo}")
        print(f"  Enriched with AFFO Margin proxy:   {n_affo}")
    return n_ffo, n_affo


def main():
    if len(sys.argv) < 2:
        print("usage: enrich_reit.py <input.json> [output.json]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path
    with open(in_path) as f:
        d = json.load(f)
    recs = _records(d)
    if recs is None:
        print("Could not locate records list in JSON")
        sys.exit(1)
    enrich(recs)
    # Print sample enriched names so the user can sanity-check
    sample = sorted(
        (r for r in recs if r.get("sector") == "Real Estate" and r.get("ffo_growth_5y") is not None),
        key=lambda r: -(r.get("_composite_score") or 0),
    )[:10]
    if sample:
        print("\n  Sample enriched REITs (top by composite):")
        for r in sample:
            g = r.get("ffo_growth_5y")
            m = r.get("affo_margin")
            gs = f"{g * 100:>+5.1f}%" if g is not None else "    —"
            ms = f"{m * 100:>5.1f}%" if m is not None else "    —"
            print(f"    {r['ticker']:6}  FFO Growth: {gs}   AFFO Margin: {ms}")
    with open(out_path, "w") as f:
        json.dump(d, f)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
