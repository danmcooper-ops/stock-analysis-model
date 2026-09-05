"""Phase 3: enrich Technology / Healthcare / Communication Services
records with SEC XBRL line items that aren't currently extracted by
analyze_stock.py.

For every stock in those three sectors with a known CIK, fetches the
companyfacts blob from SEC EDGAR once and pulls out:

  R&D expense           -> rd_intensity_xbrl   = R&D / Revenue (latest yr)
  Stock-based comp      -> sbc_pct_rev_xbrl    = SBC / Revenue
  FCF − SBC margin      -> fcf_margin_ex_sbc   = (CFO − capex − SBC) / Revenue
  Net cash position     -> net_cash_to_mcap    = (Cash + STI − Debt) / Mcap
  Deferred revenue grw  -> deferred_rev_growth = YoY change in
                                                 ContractWithCustomerLiability

These are point-in-time / annual flow extractions from US-GAAP XBRL
tags. The framework's existing rd_intensity / sbc_pct_rev fields are
populated from yfinance income statements, which cover only ~5-10% of
the universe; XBRL extends that to ~70-80% for US filers.

Usage:
    python scripts/enrich_xbrl.py output/results_YYYY-MM-DD.json [out]

Reuses data/sec_xbrl_client.SECXBRLClient. Bounded by SEC's 10 req/sec
rate limit; for ~500 mapped tickers expect ~60-90 seconds wall clock
on a cold cache. Idempotent — strips prior enrichment before running.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data.snapshot_store import (read_snapshot, snapshot_date_from_path,
                                 sync_snapshot_file, write_snapshot_file)
from data.provenance import (append_events, attach_enrichment,
                             enrichment_block, make_event, strip_enrichment)
from data.sec_legal_client import SECLegalClient
from data.sec_xbrl_client import SECXBRLClient

# Sectors we enrich. Communication Services included so Meta / Google /
# Netflix get clean SBC and Tech-style derived KPIs. Industrials added
# in Phase 4 for backlog / capex-intensity. Consumer Cyclical and
# Consumer Defensive added in Phase 5 for working-capital + ad-spend
# metrics.
_TARGET_SECTORS = {
    "Technology", "Healthcare", "Communication Services",
    "Industrials",
    "Consumer Cyclical", "Consumer Defensive",
    # Phase 6 — heavy-asset sectors. The canonical KPIs (Reserves,
    # Authorized ROE, AISC) need specialty data sources, but Capex/D&A
    # is the universal capital-reinvestment discipline ratio that
    # applies across all three.
    "Energy", "Utilities", "Basic Materials",
    # Phase 7 — Real Estate (debt-maturity wall from XBRL) and Financial
    # Services (insurer combined ratio). Banks in Financial Services get
    # nothing from XBRL here — they're enriched via FDIC — so the enrich
    # loop skips non-insurer financials to avoid wasted companyfacts
    # fetches (see the industry guard in enrich()).
    "Real Estate", "Financial Services",
}

# Financial Services records are only worth a companyfacts fetch when
# they're insurers (combined ratio). Banks / asset managers / exchanges
# carry no insurer XBRL concepts, so we skip them.
_INSURER_HINTS = ("insurance", "insurer")

# XBRL US-GAAP tags. First match wins per concept. Multiple aliases so
# we tolerate filer-specific taxonomy choices across the 8000+ filers.
_TAGS = {
    "rd": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "sbc": [
        "ShareBasedCompensation",
        "StockBasedCompensation",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "st_investments": [
        "ShortTermInvestments",
        "AvailableForSaleSecuritiesCurrent",
        "MarketableSecuritiesCurrent",
    ],
    # Debt components live in SECXBRLClient._XBRL_TAG_MAP now — total debt
    # is resolved via xbrl_client._resolve_total_debt_annual (the same
    # non-alias-merged priority ladder that used to live here).
    "def_rev_current": [
        "ContractWithCustomerLiabilityCurrent",
        "DeferredRevenueCurrent",
    ],
    "def_rev_noncurrent": [
        "ContractWithCustomerLiabilityNoncurrent",
        "DeferredRevenueNoncurrent",
    ],
    # Phase 4 (Industrials) — total contract backlog. RPO is the ASC-606
    # successor to legacy Backlog disclosures; we try both.
    "backlog": [
        "RevenueRemainingPerformanceObligation",
        "Backlog",
    ],
    # Phase 5 (Consumer Cyclical / Defensive) — working capital +
    # advertising spend. Inventory and COGS combine into Inventory Days
    # (a primary late-cycle stress signal); the receivables/payables
    # pair completes the cash conversion cycle. AdvertisingExpense is
    # the closest XBRL-standardized brand-spend proxy.
    "inventory": [
        "InventoryNet",
        "InventoryGross",
    ],
    "cogs": [
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfRevenue",
    ],
    "receivables": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "payables": [
        "AccountsPayableCurrent",
        "AccountsPayableTradeCurrent",
    ],
    "advertising": [
        "AdvertisingExpense",
        "MarketingExpense",
    ],
    # Phase 6 (Energy / Utilities / Materials) — D&A. For oil & gas
    # filers, the DepreciationDepletionAndAmortization tag captures
    # the depletion of reserves alongside ordinary depreciation, which
    # is the right cash-vs-accounting basis for Capex/D&A in those
    # sectors.
    "dd_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    # Phase 7 (Real Estate) — long-term debt maturity schedule. Each
    # bucket is the principal due in that future window as of the
    # balance-sheet date. Weighted by bucket midpoint these give a
    # weighted-average years-to-maturity (the "debt maturity wall").
    "debt_mat_y1": [
        "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
        "LongTermDebtMaturitiesRepaymentsOfPrincipalRemainderOfFiscalYear",
    ],
    "debt_mat_y2": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo"],
    "debt_mat_y3": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree"],
    "debt_mat_y4": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour"],
    "debt_mat_y5": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive"],
    "debt_mat_after5": ["LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive"],
    # Phase 7 (Financial Services — insurers only). Combined ratio is
    # cleanly 1 − UnderwritingIncomeLoss / PremiumsEarnedNet; we keep
    # the losses + DAC-amortization fallback for filers that don't tag
    # a single underwriting-income line.
    "premiums_earned": ["PremiumsEarnedNet", "PremiumsEarnedNetPropertyAndCasualty"],
    "underwriting_income": ["UnderwritingIncomeLoss"],
    "claims_incurred": ["PolicyholderBenefitsAndClaimsIncurredNet", "IncurredClaimsPropertyCasualtyAndLiability"],
    "dac_amortization": ["DeferredPolicyAcquisitionCostAmortizationExpense"],
}

_NEW_FIELDS = (
    "rd_intensity_xbrl",
    "sbc_pct_rev_xbrl",
    "fcf_margin_ex_sbc",
    "net_cash_to_mcap",
    "deferred_rev_growth",
    # Phase 4
    "capex_intensity",
    "backlog_to_revenue",
    # Phase 5
    "inventory_days",
    "working_capital_days",
    "brand_spend_pct_rev",
    # Phase 6
    "capex_to_dd_ratio",
    # Phase 7 — cross-sector quick wins (local, no extra fetch)
    "rule_of_40",
    "book_to_bill_proxy",
    "brand_spend_trend",
    # Phase 7 — new XBRL concepts
    "debt_maturity_wall_yrs",
    "combined_ratio",
    "float_cost",
)


def _records(d):
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return None


def _latest(d):
    """Return (year, value) for the most recent year in a {year: val} dict."""
    if not d:
        return None, None
    y = max(d.keys())
    return y, d[y]


def _extract(xbrl_client, facts, concept):
    """Pull the annual time series for a concept (tries each tag alias)."""
    tags = _TAGS.get(concept, [])
    if not tags:
        return {}
    vals = xbrl_client._extract_annual_values(facts, tags)
    return vals or {}


def _compute_one(rec, facts, xbrl_client):
    """Derive the five Phase-3 KPIs for one stock record. Mutates rec."""
    # Pull every concept we need from the cached facts blob.
    rd = _extract(xbrl_client, facts, "rd")
    sbc = _extract(xbrl_client, facts, "sbc")
    cash = _extract(xbrl_client, facts, "cash")
    sti = _extract(xbrl_client, facts, "st_investments")
    dr_cur = _extract(xbrl_client, facts, "def_rev_current")
    dr_nc = _extract(xbrl_client, facts, "def_rev_noncurrent")

    # Revenue and CFO histories already in the record (populated by
    # analyze_stock.py from edgar_history).
    eh = rec.get("edgar_history") or {}
    revs = eh.get("revenue_history") or {}
    cfo_h = eh.get("operating_cf_history") or {}
    capex_h = eh.get("capex_history") or {}

    # edgar_history dicts are keyed by string years ('2024'); XBRL
    # extraction returns int year keys (2024). Convert when looking up.
    def _rev(y):
        return revs.get(str(y))

    def _cfo(y):
        return cfo_h.get(str(y))

    def _capex(y):
        return capex_h.get(str(y))

    # 1. R&D / Revenue — latest year with both populated
    rd_year, rd_val = _latest(rd)
    if rd_year and rd_val and _rev(rd_year):
        rec["rd_intensity_xbrl"] = rd_val / _rev(rd_year)

    # 2. SBC / Revenue — latest year with both
    sbc_year, sbc_val = _latest(sbc)
    if sbc_year and sbc_val and _rev(sbc_year):
        rec["sbc_pct_rev_xbrl"] = sbc_val / _rev(sbc_year)

    # 3. FCF Margin ex-SBC — needs CFO + capex + SBC + revenue, latest year
    #    where all four are populated. Capex is reported negative or
    #    positive depending on filer convention; we use abs() so the
    #    subtraction is always "less reinvestment, less compensation".
    if sbc_year and sbc_val and _rev(sbc_year):
        cfo = _cfo(sbc_year)
        capex = _capex(sbc_year)
        if cfo is not None and capex is not None:
            fcf_ex_sbc = cfo - abs(capex) - sbc_val
            rec["fcf_margin_ex_sbc"] = fcf_ex_sbc / _rev(sbc_year)

    # 4. Net Cash / Mcap — point-in-time. Sum the latest cash + ST
    #    investments, subtract latest long-term + current debt, divide
    #    by current mcap. Use the same fiscal-year point for all four
    #    so we're comparing balance-sheet-instants.
    mcap = rec.get("mcap")
    cash_year, cash_val = _latest(cash)
    if mcap and mcap > 0 and cash_year:
        sti_val = sti.get(cash_year) or 0
        # Resolve total debt for the cash-anchor year WITHOUT fabricating
        # zero: a missing debt value made heavily-levered foreign filers
        # (JKS, SID) display 3-4x their market cap in "net cash". The
        # priority ladder (noncurrent+current split first, tier-4 zero only
        # when NO debt concept is ever tagged, otherwise UNKNOWN -> skip)
        # lives in SECXBRLClient._resolve_total_debt_annual.
        debt_series, debt_tagged = xbrl_client._resolve_total_debt_annual(facts)
        debt_val = debt_series.get(cash_year)
        if debt_val is None and not debt_tagged:
            debt_val = 0.0
        if debt_val is not None:
            net_cash = (cash_val or 0) + sti_val - debt_val
            rec["net_cash_to_mcap"] = net_cash / mcap

    # 5. Deferred Revenue Growth — sum current + noncurrent contract
    #    liabilities, compute YoY change. Falls through cleanly if the
    #    filer doesn't carry deferred revenue (e.g. one-time hardware
    #    vendors with no recurring billings).
    dr_total = {}
    for y in set(list(dr_cur.keys()) + list(dr_nc.keys())):
        dr_total[y] = (dr_cur.get(y) or 0) + (dr_nc.get(y) or 0)
    if len(dr_total) >= 2:
        years = sorted(dr_total.keys())
        latest, prior = years[-1], years[-2]
        if dr_total[prior] and dr_total[prior] > 0:
            rec["deferred_rev_growth"] = (
                dr_total[latest] / dr_total[prior] - 1
            )

    # --- Phase 4 ---
    # 6. Capex Intensity (Capex / Revenue) — purely local from
    #    edgar_history; useful sector-relative quality signal. For
    #    Industrials, low capex intensity = capital-light scale; high =
    #    heavy reinvestment. No XBRL fetch needed.
    if capex_h and revs:
        years_both = sorted(set(capex_h.keys()) & set(revs.keys()))
        if years_both:
            latest_y = years_both[-1]
            cap = capex_h.get(latest_y)
            r_v = revs.get(latest_y)
            if cap is not None and r_v and r_v > 0:
                rec["capex_intensity"] = abs(cap) / r_v

    # 7. Backlog / Revenue — XBRL RemainingPerformanceObligation
    #    (or legacy Backlog). The number reported is the total
    #    contracted-but-not-yet-recognized revenue at year-end;
    #    dividing by trailing revenue gives "years of revenue locked in"
    #    in expressed-as-a-ratio form. >1.0× = strong forward visibility.
    bk = _extract(xbrl_client, facts, "backlog")
    bk_year, bk_val = _latest(bk)
    if bk_year and bk_val and _rev(bk_year) and _rev(bk_year) > 0:
        rec["backlog_to_revenue"] = bk_val / _rev(bk_year)

    # --- Phase 5 ---
    # 8. Inventory Days = Inventory / (COGS / 365). Diagnostic for
    #    cyclical stress: rising inventory days into a downturn predicts
    #    future markdowns. <60 = lean ops (apparel / electronics);
    #    >120 = heavy stockholders (industrial parts, specialty pharma).
    inv = _extract(xbrl_client, facts, "inventory")
    cogs = _extract(xbrl_client, facts, "cogs")
    inv_year, inv_val = _latest(inv)
    if inv_year and inv_val:
        cogs_val = cogs.get(inv_year)
        if cogs_val and cogs_val > 0:
            rec["inventory_days"] = (inv_val / cogs_val) * 365

    # 9. Working Capital Days = Inventory Days + Receivables Days −
    #    Payables Days (the Cash Conversion Cycle). Negative = working
    #    capital is a SOURCE of cash (Costco / Amazon model); high
    #    positive = capital tied up funding the operation.
    ar = _extract(xbrl_client, facts, "receivables")
    ap = _extract(xbrl_client, facts, "payables")
    wc_year = inv_year
    if (wc_year and inv_val and cogs.get(wc_year) and _rev(wc_year) and _rev(wc_year) > 0):
        ar_val = ar.get(wc_year)
        ap_val = ap.get(wc_year)
        if ar_val is not None and ap_val is not None:
            cogs_val = cogs.get(wc_year)
            inv_days = (inv_val / cogs_val) * 365
            ar_days = (ar_val / _rev(wc_year)) * 365
            ap_days = (ap_val / cogs_val) * 365
            rec["working_capital_days"] = inv_days + ar_days - ap_days

    # 10. Brand Spend / Revenue — Advertising Expense / Revenue.
    #     Consumer Defensive moat signal: falling ad spend with rising
    #     market share = moat tightening. Standardized XBRL coverage
    #     is uneven (many filers bury advertising inside SG&A), so
    #     coverage is moderate.
    adv = _extract(xbrl_client, facts, "advertising")
    adv_year, adv_val = _latest(adv)
    if adv_year and adv_val and _rev(adv_year) and _rev(adv_year) > 0:
        rec["brand_spend_pct_rev"] = adv_val / _rev(adv_year)

    # --- Phase 6 ---
    # 11. Capex / D&A — capital reinvestment discipline. >1.5× sustained
    #     = building ahead of depreciation (future writedown risk in
    #     commodity sectors, but rate-base growth in utilities); <1× =
    #     liquidating the asset base. For oil & gas filers, D&A here
    #     correctly includes depletion of reserves.
    dd = _extract(xbrl_client, facts, "dd_amortization")
    # Anchor on the latest year where BOTH D&A (companyfacts) and capex
    # (edgar_history) exist. companyfacts often carries a newer fiscal
    # year than edgar_history's capex series; keying solely on the latest
    # D&A year would then null the ratio even when an aligned prior year
    # is available.
    for y in sorted(dd.keys(), reverse=True):
        dd_val = dd.get(y)
        cap = _capex(y)
        if dd_val and dd_val > 0 and cap is not None:
            rec["capex_to_dd_ratio"] = abs(cap) / dd_val
            break

    # --- Phase 7 ---
    # 12. Rule of 40 (Technology): 5yr revenue CAGR + FCF-ex-SBC margin,
    #     in points. >40 = healthy compounder. Both inputs already on the
    #     record (rev_cagr_5y from analyze_stock; fcf_margin_ex_sbc above).
    rev_g = rec.get("rev_cagr_5y")
    fcf_m = rec.get("fcf_margin_ex_sbc")
    if rev_g is not None and fcf_m is not None:
        rec["rule_of_40"] = 100.0 * (rev_g + fcf_m)

    # 13. Book-to-Bill proxy (Industrials): (ΔBacklog + Revenue) / Revenue
    #     from the RPO series. >1.0 = orders replacing and exceeding
    #     billings. Reuses the `bk` backlog series fetched in Phase 4.
    if bk:
        bk_years = sorted(bk.keys())
        if len(bk_years) >= 2:
            b_latest, b_prior = bk_years[-1], bk_years[-2]
            rev_l = _rev(b_latest)
            if (rev_l and rev_l > 0 and bk.get(b_latest) is not None
                    and bk.get(b_prior) is not None):
                rec["book_to_bill_proxy"] = (bk[b_latest] - bk[b_prior] + rev_l) / rev_l

    # 14. Brand Spend Trend (Consumer Cyclical/Defensive): annualized
    #     change in advertising/revenue. Positive = ramping brand
    #     investment. Reuses the `adv` series fetched in Phase 5.
    if adv:
        ratios = []
        for y in sorted(adv.keys()):
            r_v = _rev(y)
            if adv.get(y) is not None and r_v and r_v > 0:
                ratios.append((int(y), adv[y] / r_v))
        if len(ratios) >= 2:
            span = ratios[-1][0] - ratios[0][0]
            if span > 0:
                rec["brand_spend_trend"] = (ratios[-1][1] - ratios[0][1]) / span

    # 15. Debt Maturity Wall (Real Estate): weighted-average years to
    #     maturity across the disclosed repayment buckets, weighted by
    #     principal due in each. Bucket midpoints in years; the long tail
    #     (>5yr) is assumed to average ~8 years.
    _MAT_MID = (
        ("debt_mat_y1", 0.5), ("debt_mat_y2", 1.5), ("debt_mat_y3", 2.5),
        ("debt_mat_y4", 3.5), ("debt_mat_y5", 4.5), ("debt_mat_after5", 8.0),
    )
    wsum = 0.0
    psum = 0.0
    for concept, mid in _MAT_MID:
        _, val = _latest(_extract(xbrl_client, facts, concept))
        if val and val > 0:
            wsum += mid * val
            psum += val
    if psum > 0:
        rec["debt_maturity_wall_yrs"] = wsum / psum

    # 16. Combined Ratio + Float Cost (Financial Services — insurers).
    #     Combined ratio = 1 − UnderwritingIncomeLoss / PremiumsEarnedNet.
    #     NO partial fallback: (claims + DAC amortization) / premiums omits
    #     commissions and other underwriting expenses — typically 20-30
    #     points of combined ratio — so a fallback-scored insurer at ~0.70
    #     would sit beside a properly-scored peer at ~0.98 in the same
    #     column as if comparable. Better honestly absent than fabricated
    #     low. Float Cost = Combined − 1 (negative = profitable float).
    prem_year, prem_val = _latest(_extract(xbrl_client, facts, "premiums_earned"))
    if prem_year and prem_val and prem_val > 0:
        uw = _extract(xbrl_client, facts, "underwriting_income").get(prem_year)
        if uw is not None:
            cr = 1.0 - (uw / prem_val)
            rec["combined_ratio"] = cr
            rec["float_cost"] = cr - 1.0


def enrich(records, verbose=True, events=None):
    if events is None:
        events = []
    # Idempotency: strip any prior Phase-3 enrichment.
    target_recs = [r for r in records if r.get("sector") in _TARGET_SECTORS]
    for r in target_recs:
        for k in _NEW_FIELDS:
            r.pop(k, None)
        strip_enrichment(r, "xbrl")

    # Initialize SEC clients. SECLegalClient owns the CIK map; SECXBRLClient
    # uses it. Both are throttled internally to stay under SEC's 10 req/sec
    # limit.
    sec = SECLegalClient(email="stockanalysis@example.com", request_delay=0.1)
    sec._load_cik_map()
    xbrl = SECXBRLClient(
        cik_map=sec._cik_map,
        name_map={},
        email="stockanalysis@example.com",
        request_delay=0.15,
        facts_cache=True,
    )
    _sweep = xbrl.refresh_stale_facts()
    if _sweep.get("invalidated"):
        print(f"  SEC facts cache: {_sweep['invalidated']} blob(s) evicted "
              f"({_sweep.get('filers', 0)} filer(s) filed since the last sweep)")

    n_mapped = 0
    counters = {k: 0 for k in _NEW_FIELDS}
    n_failed = 0
    for i, r in enumerate(target_recs):
        tk = r.get("ticker")
        # Financial Services: only insurers carry useful XBRL concepts
        # (combined ratio). Skip banks / asset managers / exchanges so we
        # don't burn a companyfacts fetch that yields nothing.
        if r.get("sector") == "Financial Services":
            industry = (r.get("industry") or "").lower()
            if not any(h in industry for h in _INSURER_HINTS):
                continue
        if tk not in sec._cik_map:
            attach_enrichment(r, "xbrl", enrichment_block(applied=False, reason="no_cik"))
            continue
        n_mapped += 1
        try:
            facts = xbrl.fetch_company_facts(tk)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  [{tk}] facts fetch failed: {e}")
            attach_enrichment(r, "xbrl", enrichment_block(applied=False, reason="fetch_failed"))
            events.append(make_event("enrichment_skipped", tk, "sec_xbrl",
                                     {"reason": "fetch_failed"}))
            continue
        if not facts:
            n_failed += 1
            attach_enrichment(r, "xbrl", enrichment_block(applied=False, reason="fetch_failed"))
            events.append(make_event("enrichment_skipped", tk, "sec_xbrl",
                                     {"reason": "empty_facts"}))
            continue
        attach_enrichment(r, "xbrl", enrichment_block(applied=True))
        _compute_one(r, facts, xbrl)
        for k in _NEW_FIELDS:
            if r.get(k) is not None:
                counters[k] += 1
        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(target_recs)} processed...")

    if verbose:
        print(f"\n  Target-sector total:               {len(target_recs)}")
        print(f"  With CIK in SEC map:               {n_mapped}")
        print(f"  Fetch failures:                    {n_failed}")
        for k in _NEW_FIELDS:
            print(f"  Enriched with {k:22} {counters[k]}")
    return counters


def main():
    if len(sys.argv) < 2:
        print("usage: enrich_xbrl.py <input.json> [output.json]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path
    d = read_snapshot(in_path)
    recs = _records(d)
    if recs is None:
        print("Could not locate records list in JSON")
        sys.exit(1)
    events = []
    enrich(recs, events=events)
    write_snapshot_file(out_path, d)
    sync_snapshot_file(out_path, data=d)  # keep the DuckDB snapshot store in step
    print(f"\n  Wrote {out_path}")
    run_date = (d.get("date") if isinstance(d, dict) else None) or \
        snapshot_date_from_path(in_path)
    append_events(os.path.dirname(out_path) or ".", run_date, "enrich_xbrl", events)


if __name__ == "__main__":
    main()
