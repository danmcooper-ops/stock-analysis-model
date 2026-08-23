"""Enrich Healthcare (drug / biotech) records with an FDA-pipeline proxy.

For every drug-or-biotech stock, counts the active interventional studies
the company sponsors on ClinicalTrials.gov and writes it as:

  fda_pipeline_count   int — active sponsored interventional trials

This is a depth-of-pipeline proxy. It's scoped to drug/biotech industries
(pharma, biotech) because pipeline count is only meaningful there;
devices, payers, and services would just contribute zeros and distort the
sector median. Sponsor name comes from data/ticker_sponsor_map.py when
mapped, else the cleaned company name.

Usage:
    python scripts/enrich_pipeline.py output/results_YYYY-MM-DD.json [out]

Re-runs are idempotent; ClinicalTrials.gov responses are cached for 7
days under data/cache/clinicaltrials/.
"""
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data import clinicaltrials_client
from data.provenance import (STALE_CACHE_DAYS, append_events, attach_enrichment,
                             enrichment_block, make_event, strip_enrichment)
from data.ticker_sponsor_map import TICKER_TO_SPONSOR

_NEW_FIELDS = ("fda_pipeline_count",)

# Only these Healthcare industries get a pipeline lookup. Substring match
# against the record's `industry`, lower-cased.
_PIPELINE_INDUSTRIES = ("drug", "biotech", "pharmac")

# Corporate suffixes stripped from company_name to improve the lenient
# ClinicalTrials.gov sponsor text match.
_SUFFIXES = (
    " incorporated", " inc.", " inc", " corporation", " corp.", " corp",
    " company", " co.", " plc", " ltd.", " ltd", " limited", " holdings",
    " group", " sa", " ag", " nv", " a/s", ",",
)


def _clean_sponsor(name):
    if not name:
        return None
    s = name.strip()
    low = s.lower()
    for suf in _SUFFIXES:
        if low.endswith(suf):
            s = s[: len(s) - len(suf)].strip()
            low = s.lower()
    return s or None


def _records(d):
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return None


def _is_pipeline_co(r):
    if r.get("sector") != "Healthcare":
        return False
    industry = (r.get("industry") or "").lower()
    return any(h in industry for h in _PIPELINE_INDUSTRIES)


def enrich(records, verbose=True, events=None):
    if events is None:
        events = []
    targets = [r for r in records if _is_pipeline_co(r)]
    # Idempotency: strip prior enrichment.
    for r in targets:
        for k in _NEW_FIELDS:
            r.pop(k, None)
        strip_enrichment(r, "clinicaltrials")

    n_attempted = 0
    n_enriched = 0
    n_failed = 0
    for r in targets:
        tk = r.get("ticker")
        sponsor = TICKER_TO_SPONSOR.get(tk) or _clean_sponsor(r.get("company_name"))
        if not sponsor:
            continue
        n_attempted += 1
        m = {}
        count = clinicaltrials_client.active_pipeline_count(sponsor, meta=m)
        if count is None:
            n_failed += 1
            if verbose:
                print(f"  [{tk:6}] pipeline lookup failed (sponsor={sponsor!r})")
            attach_enrichment(r, "clinicaltrials", enrichment_block(
                applied=False, reason="lookup_failed", sponsor=sponsor))
            events.append(make_event("enrichment_skipped", tk, "clinicaltrials",
                                     {"reason": "lookup_failed", "sponsor": sponsor}))
            continue
        attach_enrichment(r, "clinicaltrials", enrichment_block(
            applied=True, cache_hit=m.get("cache_hit"),
            cache_age_days=m.get("cache_age_days"), sponsor=sponsor))
        if (m.get("cache_age_days") or 0) > STALE_CACHE_DAYS:
            events.append(make_event("stale_cache", tk, "clinicaltrials",
                                     {"cache_age_days": round(m["cache_age_days"], 2),
                                      "threshold_days": STALE_CACHE_DAYS}))
        r["fda_pipeline_count"] = count
        n_enriched += 1
        # Polite pause on cold-cache runs.
        time.sleep(0.05)

    if verbose:
        print(f"\n  Healthcare drug/biotech targets: {len(targets)}")
        print(f"  Sponsor resolved:                {n_attempted}")
        print(f"  Successfully enriched:           {n_enriched}")
        print(f"  Failed:                          {n_failed}")
    return n_enriched


def main():
    if len(sys.argv) < 2:
        print("usage: enrich_pipeline.py <input.json> [output.json]")
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
    sample = [r for r in recs if r.get("fda_pipeline_count") is not None]
    sample.sort(key=lambda r: r.get("fda_pipeline_count", 0), reverse=True)
    if sample:
        print("\n  Top pipelines:")
        for r in sample[:10]:
            print(f"    {r['ticker']:6}  {r.get('fda_pipeline_count'):>5}  {r.get('company_name')}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    print(f"\n  Wrote {out_path}")
    run_date = (d.get("date") if isinstance(d, dict) else None) or \
        os.path.basename(in_path).replace("results_", "").replace(".json", "")
    append_events(os.path.dirname(out_path) or ".", run_date, "enrich_pipeline", events)


if __name__ == "__main__":
    main()
