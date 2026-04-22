# api/main.py
"""FastAPI backend — exposes the stock analysis engine over HTTP.

Endpoints:
    GET  /health                          — liveness check
    POST /analyze/{ticker}                — queue a single-ticker analysis, returns job_id
    GET  /report/{id}                     — poll status + download links
    GET  /report/{id}/download/html       — download HTML report
    GET  /report/{id}/download/xlsx       — download Excel report
"""
import gc
import os
import re
import sys
import uuid
from datetime import datetime
from typing import Dict

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_env_path = os.path.join(_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from data.culture_client import CultureClient
from data.finnhub_supply_client import FinnhubSupplyClient
from data.news_client import NewsClient
from data.sec_insider_client import SECInsiderClient
from data.sec_legal_client import SECLegalClient
from data.sec_supply_client import SECSupplyClient
from data.sec_xbrl_client import SECXBRLClient
from data.tiingo_client import TiingoClient
from data.treasury_rate import fetch_risk_free_rate
from data.yfinance_client import YFinanceClient
from models.dcf import reverse_dcf
from models.epv import earnings_power_value, epv_with_growth_premium
from models.market import compute_analyst_consensus, compute_rating, compute_relative_multiples
from models.quality import (
    calculate_altman_z,
    calculate_beneish_m,
    calculate_earnings_quality,
    calculate_interest_coverage,
    calculate_net_debt_ebitda,
    calculate_piotroski_f,
    calculate_revenue_cagr,
    get_net_debt,
)
from models.ratios import calculate_roic, calculate_wacc, compute_dupont, compute_ratios
from models.rim import residual_income_model
from scripts.analyze_stock import (
    _compute_gross_margin,
    _compute_shareholder_yield,
    _extract_latest_financials,
    run_ddm_valuation,
    run_forward_dcf,
    select_cost_of_equity,
)
from scripts.config import (
    ERP,
    EXIT_MULT_DEFAULT_EV_EBITDA,
    MC_TERMINAL_GROWTH_SIGMA,
    MC_WACC_SIGMA,
    _get_sector_config,
)
from scripts.report_excel import build_excel
from scripts.report_html import build_html
from scripts.scoring import (
    _mc_confidence_label,
    apply_composite_rating_override,
    apply_screening_matrix,
    compute_continuous_scores,
)

app = FastAPI(title="Stock Analysis API", version="1.0.0")

_OUTPUT_DIR = os.path.join(_ROOT, "output", "api")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")

# job_id -> {job_id, ticker, status, created_at, html_path?, xlsx_path?, summary?, error?}
_jobs: Dict[str, dict] = {}


def _run_analysis(job_id: str, ticker: str) -> None:
    """Run full single-ticker analysis; updates _jobs[job_id] on completion."""
    try:
        _jobs[job_id]["status"] = "running"

        risk_free_rate = fetch_risk_free_rate()
        yf_client = YFinanceClient()
        tiingo_client = TiingoClient(request_delay=0.5)

        # --- Phase 1: fetch data, screen ROIC / WACC / cost-of-equity ---
        yf_data = yf_client.fetch_financials(ticker)
        info = yf_data.get("info") or {}
        if not info:
            raise ValueError(f"No data found for ticker '{ticker}'")

        sector = info.get("sector", "")
        roic_data = calculate_roic(yf_data)
        if not roic_data:
            raise ValueError(f"Insufficient financial data for '{ticker}'")

        cost_of_equity, re_method, beta_diag = select_cost_of_equity(
            yf_data, risk_free_rate, yf_client, ticker, erp=ERP,
            tiingo_client=tiingo_client,
        )
        wacc = calculate_wacc(yf_data, cost_of_equity)
        if wacc is not None:
            s_cfg = _get_sector_config(sector)
            wacc = max(s_cfg["wacc_floor"], min(s_cfg["wacc_cap"], wacc))

        # --- Phase 2: full analysis ---
        news_client = NewsClient(request_delay=1.0, max_age_days=30)
        sec_client = SECLegalClient(email="stockanalysis@example.com", request_delay=1.0)
        supply_client = FinnhubSupplyClient(request_delay=1.0)
        sec_client._load_cik_map()
        sec_supply_client = SECSupplyClient(
            cik_map=sec_client._cik_map,
            name_map=sec_client._name_map,
            email="stockanalysis@example.com",
            request_delay=1.0,
        )
        sec_xbrl_client = SECXBRLClient(
            cik_map=sec_client._cik_map,
            name_map=sec_client._name_map,
            email="stockanalysis@example.com",
            request_delay=1.0,
        )
        sec_insider_client = SECInsiderClient(
            cik_map=sec_client._cik_map,
            name_map=sec_client._name_map,
            email="stockanalysis@example.com",
            request_delay=1.0,
            max_form4_files=15,
        )
        culture_client = CultureClient()

        gross_margin = _compute_gross_margin(yf_data)
        latest_fins = _extract_latest_financials(yf_data)
        roic_cv = None
        roic_years = roic_data.get("roic_by_year", {})
        if len(roic_years) >= 2:
            vals = list(roic_years.values())
            mean_r = sum(vals) / len(vals)
            if mean_r > 0:
                var_r = sum((x - mean_r) ** 2 for x in vals) / (len(vals) - 1)
                roic_cv = (var_r ** 0.5) / mean_r

        mcap = info.get("marketCap") or 0
        sy_result = _compute_shareholder_yield(yf_data, mcap)
        shareholder_yield = sy_result["shareholder_yield"] if sy_result else None
        share_buyback_rate = sy_result["buyback_rate"] if sy_result else None
        description = info.get("longBusinessSummary") or ""
        company_name = info.get("shortName") or info.get("longName") or ""
        industry = info.get("industry") or ""

        if tiingo_client.available:
            tiingo_news = tiingo_client.fetch_ticker_news(ticker, max_age_days=30, max_items=12)
        else:
            tiingo_news = []
        if tiingo_news:
            ticker_news = tiingo_news
            news_sentiment = tiingo_client.fetch_ticker_sentiment(ticker, max_age_days=30, max_items=12)
        else:
            news_client.prefetch_all_sectors({sector})
            ticker_news = news_client.get_combined_news(ticker, sector, max_total=12)
            news_sentiment = None

        legal_data = sec_client.fetch_legal_filings(ticker, days_back=730)
        supply_data = supply_client.fetch_supply_chain(ticker)
        if not supply_data.get("available"):
            supply_data = sec_supply_client.fetch_supply_chain(ticker)
        finnhub_peers = supply_client.fetch_peers(ticker)

        xbrl_validation = sec_xbrl_client.validate_against_yfinance(ticker, yf_data)
        edgar_history = sec_xbrl_client.fetch_historical_financials(ticker, min_years=10)
        sec_xbrl_client._cache.pop(ticker, None)
        insider_data = sec_insider_client.fetch_insider_activity(ticker, days_back=365)

        officers = info.get("companyOfficers") or []
        ceo_officer = next(
            (o for o in officers
             if "ceo" in (o.get("title") or "").lower()
             or "chief executive" in (o.get("title") or "").lower()),
            officers[0] if officers else None,
        )
        ceo = ceo_officer.get("name") if ceo_officer else None
        ceo_bio = None
        if ceo_officer:
            bio_parts = [ceo_officer.get("name", "N/A"), ceo_officer.get("title", "")]
            age = ceo_officer.get("age")
            if age:
                bio_parts.append(f"Age {age}")
            year_born = ceo_officer.get("yearBorn")
            if year_born and not age:
                bio_parts.append(f"Born {year_born}")
            total_pay = ceo_officer.get("totalPay")
            if total_pay:
                pay_str = (f"${total_pay/1e6:.1f}M" if total_pay >= 1e6
                           else f"${total_pay:,.0f}")
                fy = ceo_officer.get("fiscalYear", "")
                bio_parts.append("Compensation: " + pay_str + (f" (FY{fy})" if fy else ""))
            ceo_bio = " | ".join(p for p in bio_parts if p)

        founder_led = ("founder" in (ceo_officer.get("title") or "").lower()
                       if ceo_officer else False)
        _culture_raw = culture_client.extract(info, yf_data)
        _culture_gd = culture_client.fetch_glassdoor(company_name, ticker)

        multiples = compute_relative_multiples(yf_data)
        current_price = multiples.get("price")
        shares = multiples.get("shares")
        analyst = compute_analyst_consensus(yf_data)

        dcf_fv, dcf_sens_range, fcf_growth, growth_diag, mc_result = run_forward_dcf(
            yf_data, wacc, sector=sector,
            exit_multiple=EXIT_MULT_DEFAULT_EV_EBITDA,
            roic_data=roic_data,
            terminal_growth_adj=0.0,
            wacc_sigma=MC_WACC_SIGMA,
            tg_sigma=MC_TERMINAL_GROWTH_SIGMA,
            growth_sigma_mult=1.0,
            growth_weight_shift=0.0,
        )
        mos = ((dcf_fv - current_price) / dcf_fv
               if (dcf_fv and current_price and dcf_fv > 0) else None)

        try:
            div_series = yf_client.fetch_dividends(ticker)
        except Exception:
            div_series = pd.Series(dtype=float)
        ddm_result = run_ddm_valuation(
            yf_data, div_series, cost_of_equity,
            analyst_ltg=growth_diag.get("analyst_ltg"),
        )

        eq = calculate_earnings_quality(yf_data)
        piotroski = calculate_piotroski_f(yf_data)
        rev_cagr = calculate_revenue_cagr(yf_data)

        rev_cagr_5y = None
        rev_cagr_10y = None
        if edgar_history and edgar_history.get("revenue_history"):
            rev_hist = edgar_history["revenue_history"]
            sorted_years = sorted(rev_hist.keys())
            newest_rev = rev_hist[sorted_years[-1]] if sorted_years else None
            if newest_rev and newest_rev > 0:
                if len(sorted_years) >= 6:
                    yr5_rev = rev_hist.get(sorted_years[-6])
                    if yr5_rev and yr5_rev > 0:
                        rev_cagr_5y = (newest_rev / yr5_rev) ** (1 / 5) - 1
                if len(sorted_years) >= 11:
                    yr10_rev = rev_hist.get(sorted_years[-11])
                    if yr10_rev and yr10_rev > 0:
                        rev_cagr_10y = (newest_rev / yr10_rev) ** (1 / 10) - 1

        int_cov = calculate_interest_coverage(yf_data)
        nd_ebitda = calculate_net_debt_ebitda(yf_data)
        ratios = compute_ratios(yf_data)

        cf = yf_data.get("cash_flow")
        fcf = None
        if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
            fcf_vals = cf.loc["Free Cash Flow"].dropna().sort_index()
            if len(fcf_vals) > 0:
                fcf = fcf_vals.iloc[-1]

        altman_z = calculate_altman_z(yf_data)
        altman_z_zone = None
        if altman_z is not None:
            if altman_z > 2.99:
                altman_z_zone = "safe"
            elif altman_z >= 1.81:
                altman_z_zone = "grey"
            else:
                altman_z_zone = "distress"

        beneish = calculate_beneish_m(yf_data)
        dupont = compute_dupont(yf_data)

        inc_stmt = yf_data.get("income_statement")
        _epv_ebit = None
        _epv_eff_tax = 0.21
        if inc_stmt is not None and not inc_stmt.empty:
            _latest_inc = inc_stmt.iloc[:, 0]
            _epv_ebit = _latest_inc.get("Operating Income")
            if pd.notna(_epv_ebit) and _epv_ebit is not None:
                _tax_prov = _latest_inc.get("Tax Provision")
                _pretax = _latest_inc.get("Pretax Income")
                if (pd.notna(_tax_prov) and pd.notna(_pretax)
                        and _pretax and _pretax != 0):
                    _epv_eff_tax = max(0, min(float(_tax_prov) / float(_pretax), 0.50))
            else:
                _epv_ebit = None

        _epv_excess_cash = 0
        bs = yf_data.get("balance_sheet")
        if bs is not None and not bs.empty:
            _cash_val = bs.iloc[:, 0].get("Cash And Cash Equivalents")
            if pd.notna(_cash_val) and _cash_val is not None:
                _epv_excess_cash = float(_cash_val)

        epv_fv = earnings_power_value(
            float(_epv_ebit) if _epv_ebit is not None else None,
            _epv_eff_tax, wacc, shares, _epv_excess_cash,
        )
        epv_growth_fv = epv_with_growth_premium(epv_fv, ratios.get("ROE"), cost_of_equity)

        _book_value = info.get("bookValue")
        if _book_value is None and shares and shares > 0:
            if bs is not None and not bs.empty:
                _eq_val = bs.iloc[:, 0].get("Stockholders Equity")
                if pd.notna(_eq_val) and _eq_val:
                    _book_value = float(_eq_val) / shares
        rim_fv = residual_income_model(_book_value, ratios.get("ROE"), cost_of_equity)

        rev_dcf = None
        if dcf_fv and current_price and current_price > 0 and fcf and shares:
            net_debt_val = get_net_debt(yf_data)
            rev_dcf = reverse_dcf(current_price, fcf, wacc, shares, net_debt_val)

        high_52w = info.get("fiftyTwoWeekHigh")
        low_52w = info.get("fiftyTwoWeekLow")
        pct_from_52w_high = ((current_price - high_52w) / high_52w
                             if (current_price and high_52w and high_52w > 0) else None)
        pct_from_52w_low = ((current_price - low_52w) / low_52w
                            if (current_price and low_52w and low_52w > 0) else None)
        range_52w_position = ((current_price - low_52w) / (high_52w - low_52w) * 100
                              if (current_price and high_52w and low_52w
                                  and high_52w > low_52w) else None)

        shares_out = info.get("sharesOutstanding")
        float_shares = info.get("floatShares")
        insider_pct = info.get("heldPercentInsiders")
        inst_pct = info.get("heldPercentInstitutions")
        shares_short = info.get("sharesShort")
        short_ratio = info.get("shortRatio")
        short_pct_float = info.get("shortPercentOfFloat")
        avg_vol = info.get("averageVolume")
        share_turnover_rate = (avg_vol / shares_out
                               if (avg_vol and shares_out and shares_out > 0) else None)

        _div_rate = info.get("dividendRate")
        _div_price = info.get("currentPrice") or info.get("regularMarketPrice")
        div_yield = (_div_rate / _div_price
                     if (_div_rate and _div_price and _div_price > 0) else None)
        payout_ratio = info.get("payoutRatio")

        goodwill_pct = None
        rd_intensity = None
        sga_pct_rev = None
        sga_yoy_change = None
        if bs is not None and not bs.empty:
            total_assets = bs.iloc[:, 0].get("Total Assets")
            gw = bs.iloc[:, 0].get("Goodwill")
            if (pd.notna(gw) and gw and pd.notna(total_assets)
                    and total_assets and total_assets > 0):
                goodwill_pct = float(gw) / float(total_assets)
        inc_stmt2 = yf_data.get("income_statement")
        if inc_stmt2 is not None and not inc_stmt2.empty:
            _rev_latest = inc_stmt2.iloc[:, 0].get("Total Revenue")
            _rd_latest = inc_stmt2.iloc[:, 0].get("Research And Development")
            _sga_latest = inc_stmt2.iloc[:, 0].get("Selling General And Administration")
            if (pd.notna(_rd_latest) and _rd_latest
                    and pd.notna(_rev_latest) and _rev_latest and _rev_latest > 0):
                rd_intensity = float(_rd_latest) / float(_rev_latest)
            if (pd.notna(_sga_latest) and _sga_latest
                    and pd.notna(_rev_latest) and _rev_latest and _rev_latest > 0):
                sga_pct_rev = float(_sga_latest) / float(_rev_latest)
            if inc_stmt2.shape[1] >= 2:
                _sga_prior = inc_stmt2.iloc[:, 1].get("Selling General And Administration")
                if (pd.notna(_sga_latest) and _sga_latest
                        and pd.notna(_sga_prior) and _sga_prior and _sga_prior > 0):
                    sga_yoy_change = (float(_sga_latest) / float(_sga_prior)) - 1

        row = {
            "ticker": ticker,
            "source_group": "quality",
            "description": description,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "ceo": ceo,
            "ceo_bio": ceo_bio,
            "founder_led": founder_led,
            "employees": _culture_raw.get("employees"),
            "ceo_total_pay": _culture_raw.get("ceo_total_pay"),
            "compensation_risk": _culture_raw.get("compensation_risk"),
            "sbc": _culture_raw.get("sbc"),
            "glassdoor_rating": _culture_gd.get("glassdoor_rating"),
            "glassdoor_ceo_pct": _culture_gd.get("glassdoor_ceo_pct"),
            "glassdoor_rec_pct": _culture_gd.get("glassdoor_rec_pct"),
            "fcf": fcf,
            "shares_out": shares_out,
            "float_shares": float_shares,
            "insider_pct": insider_pct,
            "inst_pct": inst_pct,
            "shares_short": shares_short,
            "short_ratio": short_ratio,
            "short_pct_float": short_pct_float,
            "share_turnover_rate": share_turnover_rate,
            "share_buyback_rate": share_buyback_rate,
            "insider_buy_ratio": (insider_data.get("insider_buy_ratio")
                                  if insider_data and insider_data.get("available") else None),
            "insider_buy_count_90d": (insider_data.get("buy_count_90d")
                                      if insider_data and insider_data.get("available") else None),
            "insider_sell_count_90d": (insider_data.get("sell_count_90d")
                                       if insider_data and insider_data.get("available") else None),
            "insider_buy_count_365d": (insider_data.get("buy_count_365d")
                                       if insider_data and insider_data.get("available") else None),
            "insider_sell_count_365d": (insider_data.get("sell_count_365d")
                                        if insider_data and insider_data.get("available") else None),
            "insider_net_shares": (insider_data.get("net_shares_365d")
                                   if insider_data and insider_data.get("available") else None),
            "insider_net_value": (insider_data.get("net_value_365d")
                                  if insider_data and insider_data.get("available") else None),
            "insider_transactions": (insider_data.get("transactions", [])[:10]
                                     if insider_data and insider_data.get("available") else []),
            "roic_by_year": roic_data.get("roic_by_year"),
            "roic_cv": roic_cv,
            "gross_margin": gross_margin,
            "shareholder_yield": shareholder_yield,
            "div_yield": div_yield,
            "payout_ratio": payout_ratio,
            "roic": roic_data["avg_roic"],
            "wacc": wacc,
            "spread": ((roic_data["avg_roic"] - wacc) if wacc is not None else None),
            "mcap": multiples.get("market_cap"),
            "revenue": latest_fins.get("revenue"),
            "operating_income": latest_fins.get("operating_income"),
            "net_income": latest_fins.get("net_income"),
            "operating_margin": (
                latest_fins["operating_income"] / latest_fins["revenue"]
                if (latest_fins.get("operating_income") is not None
                    and latest_fins.get("revenue") and latest_fins["revenue"] > 0)
                else None
            ),
            "er": cost_of_equity,
            "re_method": re_method,
            "beta_raw": beta_diag.get("raw_beta") if beta_diag else None,
            "beta_adjusted": beta_diag.get("adjusted_beta") if beta_diag else None,
            "beta_r2": beta_diag.get("r_squared") if beta_diag else None,
            "beta_se": beta_diag.get("se_beta") if beta_diag else None,
            "beta_n_obs": beta_diag.get("n_observations") if beta_diag else None,
            "beta_r2_class": beta_diag.get("r2_classification") if beta_diag else None,
            "dcf_fv": dcf_fv,
            "price": current_price,
            "mos": mos,
            "dcf_sens_range": dcf_sens_range,
            "fcf_growth": fcf_growth,
            "analyst_ltg": growth_diag.get("analyst_ltg"),
            "margin_trend": growth_diag.get("margin_trend"),
            "surprise_avg": growth_diag.get("surprise_avg"),
            "fundamental_growth": growth_diag.get("fundamental_growth"),
            "reinvestment_rate": growth_diag.get("reinvestment_rate"),
            "terminal_growth": _get_sector_config(sector)["terminal_growth"],
            "exit_mult_fv": growth_diag.get("exit_mult_fv"),
            "tv_method_spread": growth_diag.get("tv_method_spread"),
            "mc_p10_fv": mc_result["p10_fv"] if mc_result else None,
            "mc_p90_fv": mc_result["p90_fv"] if mc_result else None,
            "mc_cv": mc_result["cv"] if mc_result else None,
            "mc_confidence": (_mc_confidence_label(mc_result["cv"])
                              if mc_result and mc_result.get("cv") is not None else None),
            "ms_diff": None,
            "ms_fv": None,
            "ms_pfv": None,
            "pe": multiples.get("pe"),
            "ev_ebitda": multiples.get("ev_ebitda"),
            "enterprise_value": multiples.get("enterprise_value"),
            "pfcf": multiples.get("pfcf"),
            "pb": multiples.get("pb"),
            "analyst_rec": (analyst.get("rec_key", "").upper()
                            if analyst.get("rec_key") else None),
            "num_analysts": analyst.get("num_analysts"),
            "target_mean": analyst.get("target_mean"),
            "target_high": analyst.get("target_high"),
            "target_low": analyst.get("target_low"),
            "piotroski": piotroski,
            "cash_conv": eq.get("cash_conversion"),
            "accruals": eq.get("accruals_ratio"),
            "rev_cagr": rev_cagr,
            "rev_cagr_5y": rev_cagr_5y,
            "rev_cagr_10y": rev_cagr_10y,
            "edgar_quality_score": (xbrl_validation.get("edgar_quality_score")
                                    if xbrl_validation else None),
            "edgar_fields_flagged": (xbrl_validation.get("fields_flagged", 0)
                                     if xbrl_validation else 0),
            "edgar_discrepancies": (xbrl_validation.get("discrepancies", [])
                                    if xbrl_validation else []),
            "edgar_history": edgar_history,
            "int_cov": int_cov,
            "nd_ebitda": nd_ebitda,
            "roe": ratios.get("ROE"),
            "de": ratios.get("Debt-to-Equity"),
            "cr": ratios.get("Current Ratio"),
            "roa": ratios.get("ROA"),
            "macro_regime": None,
            "macro_composite": None,
            "macro_erp": ERP,
            "sector_headwinds": [],
            "sector_tailwinds": [],
            "news_headlines": ticker_news,
            "news_sentiment": news_sentiment,
            "legal_filings": legal_data.get("filings", []),
            "legal_count": legal_data.get("count", 0),
            "legal_latest": legal_data.get("latest_date"),
            "suppliers": supply_data.get("suppliers", []),
            "customers": supply_data.get("customers", []),
            "supply_chain_available": supply_data.get("available", False),
            "finnhub_peers": finnhub_peers,
            "ddm_eligible": ddm_result.get("ddm_eligible", False),
            "ddm_reason": ddm_result.get("ddm_reason"),
            "ddm_fv": ddm_result.get("ddm_fv"),
            "ddm_h_fv": ddm_result.get("ddm_h_fv"),
            "ddm_growth": ddm_result.get("ddm_growth"),
            "ddm_div_cagr": ddm_result.get("ddm_div_cagr"),
            "ddm_sustainable_growth": ddm_result.get("ddm_sustainable_growth"),
            "ddm_payout_flag": ddm_result.get("ddm_payout_flag", False),
            "ddm_consecutive_years": ddm_result.get("ddm_consecutive_years"),
            "ddm_mc_median": ddm_result.get("ddm_mc_median"),
            "ddm_mc_p10": ddm_result.get("ddm_mc_p10"),
            "ddm_mc_p90": ddm_result.get("ddm_mc_p90"),
            "ddm_mc_cv": ddm_result.get("ddm_mc_cv"),
            "implied_growth": (rev_dcf["implied_growth"]
                               if rev_dcf and rev_dcf.get("converged") else None),
            "implied_vs_estimated": (
                (rev_dcf["implied_growth"] - fcf_growth)
                if (rev_dcf and rev_dcf.get("converged") and fcf_growth) else None
            ),
            "epv_fv": epv_fv,
            "epv_pfv": (current_price / epv_fv
                        if (epv_fv and current_price and epv_fv > 0) else None),
            "epv_mos": ((epv_fv - current_price) / epv_fv
                        if (epv_fv and current_price and epv_fv > 0) else None),
            "epv_growth_fv": epv_growth_fv,
            "rim_fv": rim_fv,
            "rim_mos": ((rim_fv - current_price) / rim_fv
                        if (rim_fv and current_price and rim_fv > 0) else None),
            "altman_z": altman_z,
            "altman_z_zone": altman_z_zone,
            "beneish_m": beneish["m_score"] if beneish else None,
            "beneish_flag": beneish["manipulation_flag"] if beneish else None,
            "dupont_margin": dupont["margin"] if dupont else None,
            "dupont_turnover": dupont["turnover"] if dupont else None,
            "dupont_leverage": dupont["leverage"] if dupont else None,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pct_from_52w_high": pct_from_52w_high,
            "pct_from_52w_low": pct_from_52w_low,
            "range_52w_position": range_52w_position,
            "goodwill_pct": goodwill_pct,
            "rd_intensity": rd_intensity,
            "sga_pct_rev": sga_pct_rev,
            "sga_yoy_change": sga_yoy_change,
            # No local price files available in API mode
            "momentum_12_1": None,
            "realized_vol": None,
            "drawdown_2008": None,
            "drawdown_2020": None,
            "drawdown_2022": None,
            "rolling_betas": None,
        }

        row["rating"] = compute_rating(row)
        results = [row]
        apply_screening_matrix(results)
        compute_continuous_scores(results)
        apply_composite_rating_override(results)

        job_dir = os.path.join(_OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        html_path = os.path.join(job_dir, f"{ticker}.html")
        xlsx_path = os.path.join(job_dir, f"{ticker}.xlsx")
        build_html(results, html_path)
        build_excel(results, xlsx_path)

        summary = {k: v for k, v in row.items()
                   if isinstance(v, (int, float, str, bool, type(None)))}

        _jobs[job_id].update({
            "status": "complete",
            "html_path": html_path,
            "xlsx_path": xlsx_path,
            "summary": summary,
        })

    except Exception as exc:
        _jobs[job_id].update({"status": "failed", "error": str(exc)})
    finally:
        gc.collect()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze/{ticker}", status_code=202)
def analyze(ticker: str, background_tasks: BackgroundTasks):
    """Queue a single-ticker analysis. Returns a job_id to poll with GET /report/{id}."""
    ticker = ticker.upper().strip()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "ticker": ticker,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    background_tasks.add_task(_run_analysis, job_id, ticker)
    return {"job_id": job_id, "ticker": ticker, "status": "queued"}


@app.get("/report/{job_id}")
def report(job_id: str):
    """Return job status. When complete, includes download links and a valuation summary."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    resp = {k: v for k, v in job.items() if k not in ("html_path", "xlsx_path")}
    if job["status"] == "complete":
        resp["links"] = {
            "html": f"/report/{job_id}/download/html",
            "xlsx": f"/report/{job_id}/download/xlsx",
        }
    return resp


@app.get("/report/{job_id}/download/html")
def download_html(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Report not ready")
    return FileResponse(
        job["html_path"],
        media_type="text/html",
        filename=f"{job['ticker']}_analysis.html",
    )


@app.get("/report/{job_id}/download/xlsx")
def download_xlsx(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Report not ready")
    return FileResponse(
        job["xlsx_path"],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"{job['ticker']}_analysis.xlsx",
    )
