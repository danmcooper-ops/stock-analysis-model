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
from datetime import date, datetime

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data import fdic_client
from data.provenance import (append_events, attach_enrichment,
                             enrichment_block, make_event, strip_enrichment)
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


_FDIC_FIELDS = ("fdic_cert", "fdic_repdte", "nim", "efficiency_ratio", "cet1_ratio", "npl_ratio", "deposit_beta")

# Deposit beta = Δ(cost of deposits) / Δ(fed funds) across the 2021→2023
# hiking cycle. Cost of deposits ≈ EDEP (full-year deposit interest
# expense, YTD at year-end) / DEP (total deposits). The Fed lifted the
# effective funds rate from ~0.08% (Dec 2021) to ~5.33% (Dec 2023) — a
# 525 bps move — so we divide the cost change by 0.0525. Low beta (<0.4)
# = sticky, low-cost deposit franchise.
_DB_TROUGH_REPDTE = "20211231"
_DB_PEAK_REPDTE = "20231231"
_DB_FED_FUNDS_DELTA = 0.0525

# FDIC data is quarterly, so cache age is the wrong staleness signal (a
# 26-day-old cache of the latest quarter is fine — flagging it made the
# alert fire on most days by construction). What matters is whether the
# report date itself has fallen behind: call reports publish ~2 months
# after quarter end, so with one full quarter of slack the freshest
# available repdte should never be older than ~180 days.
_STALE_REPDTE_DAYS = 180


def _repdte_age_days(repdte):
    """Days since an FDIC YYYYMMDD report date, or None if unparseable."""
    try:
        return (date.today() - datetime.strptime(str(repdte), "%Y%m%d").date()).days
    except (ValueError, TypeError):
        return None


def _deposit_cost(cert, repdte):
    """Cost of deposits = full-year deposit interest expense / deposits."""
    fin = fdic_client.financials(cert, repdte=repdte)
    if not fin:
        return None
    edep = fin.get("EDEP")
    dep = fin.get("DEP")
    if edep is None or not dep or dep <= 0:
        return None
    return edep / dep


def _deposit_beta(cert):
    """Δ cost of deposits / Δ fed funds over the 2021→2023 hike cycle."""
    try:
        c0 = _deposit_cost(cert, _DB_TROUGH_REPDTE)
        c1 = _deposit_cost(cert, _DB_PEAK_REPDTE)
    except Exception:
        return None
    if c0 is None or c1 is None:
        return None
    beta = (c1 - c0) / _DB_FED_FUNDS_DELTA
    # Clamp out nonsense from data gaps / restatements.
    if beta < 0 or beta > 1.5:
        return None
    return beta


def enrich(records, verbose=True, events=None):
    if events is None:
        events = []
    fs_total = sum(1 for r in records if r.get("sector") == "Financial Services")
    # Idempotency: strip any prior FDIC enrichment so removed mappings or
    # stale records from earlier runs don't linger in the JSON.
    for r in records:
        if r.get("sector") == "Financial Services":
            for k in _FDIC_FIELDS:
                r.pop(k, None)
            strip_enrichment(r, "fdic")
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
        # Provenance is only metered on this latest-quarter call — the
        # _deposit_beta lookups below hit fixed 2021/2023 repdtes whose
        # cache is legitimately old, so flagging them would be noise.
        m = {}
        try:
            fin = fdic_client.financials(cert, meta=m)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  [{tk:6} CERT={cert}] FDIC fetch failed: {e}")
            attach_enrichment(r, "fdic", enrichment_block(applied=False, reason="fetch_failed"))
            events.append(make_event("enrichment_skipped", tk, "fdic", {"reason": "fetch_failed"}))
            continue
        if fin is None:
            n_failed += 1
            if verbose:
                print(f"  [{tk:6} CERT={cert}] no financials returned")
            attach_enrichment(r, "fdic", enrichment_block(applied=False, reason="no_financials"))
            events.append(make_event("enrichment_skipped", tk, "fdic", {"reason": "no_financials"}))
            continue
        # Staleness guard — REPDTE format is YYYYMMDD. Reject any record
        # older than ~2 years; FDIC files quarterly so the latest should
        # always be within the last few months. Cutoff is COMPUTED, not a
        # hardcoded date that silently stops guarding as time passes.
        from datetime import date as _date, timedelta as _td
        _stale_cutoff = int((_date.today() - _td(days=730)).strftime("%Y%m%d"))
        repdte = fin.get("REPDTE")
        if repdte is None or int(repdte) < _stale_cutoff:
            n_failed += 1
            if verbose:
                print(f"  [{tk:6} CERT={cert}] stale record (repdte={repdte}) — skipping")
            # Clear any prior enrichment that may be on the record
            for k in _FDIC_FIELDS:
                r.pop(k, None)
            attach_enrichment(r, "fdic", enrichment_block(applied=False, reason="stale_repdte",
                                                          repdte=repdte))
            events.append(make_event("enrichment_skipped", tk, "fdic",
                                     {"reason": "stale_repdte", "repdte": repdte}))
            continue
        attach_enrichment(r, "fdic", enrichment_block(
            applied=True, cache_hit=m.get("cache_hit"),
            cache_age_days=m.get("cache_age_days"), repdte=repdte))
        repdte_age = _repdte_age_days(repdte)
        if repdte_age is not None and repdte_age > _STALE_REPDTE_DAYS:
            events.append(make_event("stale_cache", tk, "fdic",
                                     {"repdte": repdte, "repdte_age_days": repdte_age,
                                      "threshold_days": _STALE_REPDTE_DAYS}))
        r["fdic_cert"] = cert
        r["fdic_repdte"] = repdte
        r["nim"] = _decimal_pct(fin.get("NIMY"))
        r["efficiency_ratio"] = _decimal_pct(fin.get("EEFFR"))
        r["cet1_ratio"] = _decimal_pct(fin.get("RBCT1CER"))
        nclnls = fin.get("NCLNLS")
        lnlsgr = fin.get("LNLSGR")
        # `is not None`, not truthiness: NCLNLS == 0 is a pristine credit
        # book (best-in-class 0.00%), not missing data.
        r["npl_ratio"] = ((nclnls / lnlsgr)
                          if (nclnls is not None and lnlsgr and lnlsgr > 0)
                          else None)
        r["deposit_beta"] = _deposit_beta(cert)
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
    with open(in_path, encoding="utf-8") as f:
        d = json.load(f)
    recs = _records(d)
    if recs is None:
        print("Could not locate records list in JSON")
        sys.exit(1)
    events = []
    enrich(recs, events=events)
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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    print(f"\n  Wrote {out_path}")
    run_date = (d.get("date") if isinstance(d, dict) else None) or \
        os.path.basename(in_path).replace("results_", "").replace(".json", "")
    append_events(os.path.dirname(out_path) or ".", run_date, "enrich_fdic", events)


if __name__ == "__main__":
    main()
