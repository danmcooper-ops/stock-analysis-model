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
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

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
}

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
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
    ],
    "current_debt": [
        "LongTermDebtCurrent",
        "DebtCurrent",
    ],
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
    ltd = _extract(xbrl_client, facts, "long_term_debt")
    cur_debt = _extract(xbrl_client, facts, "current_debt")
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
        ltd_val = ltd.get(cash_year) or 0
        cur_debt_val = cur_debt.get(cash_year) or 0
        net_cash = (cash_val or 0) + sti_val - ltd_val - cur_debt_val
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


def enrich(records, verbose=True):
    # Idempotency: strip any prior Phase-3 enrichment.
    target_recs = [r for r in records if r.get("sector") in _TARGET_SECTORS]
    for r in target_recs:
        for k in _NEW_FIELDS:
            r.pop(k, None)

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
    )

    n_mapped = 0
    counters = {k: 0 for k in _NEW_FIELDS}
    n_failed = 0
    for i, r in enumerate(target_recs):
        tk = r.get("ticker")
        if tk not in sec._cik_map:
            continue
        n_mapped += 1
        try:
            facts = xbrl.fetch_company_facts(tk)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  [{tk}] facts fetch failed: {e}")
            continue
        if not facts:
            n_failed += 1
            continue
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
    with open(in_path) as f:
        d = json.load(f)
    recs = _records(d)
    if recs is None:
        print("Could not locate records list in JSON")
        sys.exit(1)
    enrich(recs)
    with open(out_path, "w") as f:
        json.dump(d, f)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
