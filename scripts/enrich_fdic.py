"""Join FDIC call-report data into the daily results JSON.

For every Financial Services stock with a known FDIC CERT (mapped via
data/ticker_fdic_map.py), fetches the latest financials and writes
four fields into the record:

  nim                net interest margin (decimal, e.g. 0.0296 = 2.96%)
  efficiency_ratio   non-interest expense / revenue (decimal)
  cet1_ratio         Common Equity Tier 1 capital ratio (decimal)
  npl_ratio          non-current loans / total loans (decimal)

These convert FDIC's percent-units to decimals so downstream filters
and the sector banner treat them consistently with other framework
metrics. The FDIC report date is stored alongside as `fdic_repdte`.

Usage:
    python scripts/enrich_fdic.py output/results_YYYY-MM-DD.json [out]

Re-runs are idempotent; FDIC responses are cached for 30 days under
data/cache/fdic/.
"""
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data import fdic_client
from data.ticker_fdic_map import TICKER_TO_CERT


def _decimal_pct(v):
    return v / 100.0 if v is not None else None


def _records(d):
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return None


_FDIC_FIELDS = ("fdic_cert", "fdic_repdte", "nim", "efficiency_ratio", "cet1_ratio", "npl_ratio")


def enrich(records, verbose=True):
    fs_total = sum(1 for r in records if r.get("sector") == "Financial Services")
    # Idempotency: strip any prior FDIC enrichment so removed mappings or
    # stale records from earlier runs don't linger in the JSON.
    for r in records:
        if r.get("sector") == "Financial Services":
            for k in _FDIC_FIELDS:
                r.pop(k, None)
    n_attempted = 0
    n_enriched = 0
    n_failed = 0
    for r in records:
        if r.get("sector") != "Financial Services":
            continue
        tk = r.get("ticker")
        cert = TICKER_TO_CERT.get(tk)
        if cert is None:
            continue
        n_attempted += 1
        try:
            fin = fdic_client.financials(cert)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  [{tk:6} CERT={cert}] FDIC fetch failed: {e}")
            continue
        if fin is None:
            n_failed += 1
            if verbose:
                print(f"  [{tk:6} CERT={cert}] no financials returned")
            continue
        # Staleness guard — REPDTE format is YYYYMMDD. Reject any record
        # older than ~2 years; FDIC files quarterly so the latest should
        # always be within the last few months.
        repdte = fin.get("REPDTE")
        if repdte is None or int(repdte) < 20240101:
            n_failed += 1
            if verbose:
                print(f"  [{tk:6} CERT={cert}] stale record (repdte={repdte}) — skipping")
            # Clear any prior enrichment that may be on the record
            for k in ("fdic_cert", "fdic_repdte", "nim", "efficiency_ratio", "cet1_ratio", "npl_ratio"):
                r.pop(k, None)
            continue
        r["fdic_cert"] = cert
        r["fdic_repdte"] = repdte
        r["nim"] = _decimal_pct(fin.get("NIMY"))
        r["efficiency_ratio"] = _decimal_pct(fin.get("EEFFR"))
        r["cet1_ratio"] = _decimal_pct(fin.get("RBCT1CER"))
        nclnls = fin.get("NCLNLS")
        lnlsgr = fin.get("LNLSGR")
        r["npl_ratio"] = (nclnls / lnlsgr) if (nclnls and lnlsgr and lnlsgr > 0) else None
        n_enriched += 1
        # Small pause to be a polite API citizen on cold-cache runs
        time.sleep(0.05)
    if verbose:
        print(f"\n  Financial Services total: {fs_total}")
        print(f"  Mapped to FDIC CERT:      {n_attempted}")
        print(f"  Successfully enriched:    {n_enriched}")
        print(f"  Failed:                   {n_failed}")
    return n_enriched


def main():
    if len(sys.argv) < 2:
        print("usage: enrich_fdic.py <input.json> [output.json]")
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
    # Print sample of enriched records for sanity-check
    def _pct(v, decs=2):
        return f"{v * 100:.{decs}f}%" if v is not None else "    —"

    sample = [r for r in recs if r.get("fdic_cert") is not None][:10]
    if sample:
        print("\n  Sample enriched records:")
        for r in sample:
            print(
                f"    {r['ticker']:6}  "
                f"NIM={_pct(r.get('nim'), 2):>7}  "
                f"Eff={_pct(r.get('efficiency_ratio'), 1):>7}  "
                f"CET1={_pct(r.get('cet1_ratio'), 1):>7}  "
                f"NPL={_pct(r.get('npl_ratio'), 2):>7}  "
                f"(repdte={r.get('fdic_repdte')})"
            )
    with open(out_path, "w") as f:
        json.dump(d, f)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
