# scripts/analyze_stock.py
import gc
import sys
import os
import io
import json
from datetime import date
from statistics import median as _median
import numpy as np
import pandas as pd
from urllib.request import urlopen, Request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env from project root (simple key=value parser, no dependency needed)
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from data.yfinance_client import YFinanceClient
from data.treasury_rate import fetch_risk_free_rate
from data.validation import validate_financials
from models.capm import (calculate_beta, r2_diagnostic, ggm_implied_re, buildup_re)
from models.dcf import (two_stage_ev, fair_value_per_share, dcf_sensitivity,
                        two_stage_ev_exit_multiple, monte_carlo_dcf,
                        reverse_dcf)
from models.ratios import (compute_ratios, calculate_roic, calculate_wacc,
                           calculate_fundamental_growth, compute_dupont)
from models.quality import (calculate_earnings_quality, calculate_piotroski_f,
                            calculate_revenue_cagr, calculate_interest_coverage,
                            calculate_net_debt_ebitda, get_net_debt,
                            calculate_altman_z, calculate_beneish_m)
from models.market import (compute_relative_multiples, compute_analyst_consensus,
                           compute_rating)
from models.ddm import (ddm_eligibility, estimate_ddm_growth, two_stage_ddm,
                         ddm_h_model, monte_carlo_ddm)
from models.epv import earnings_power_value, epv_with_growth_premium
from models.rim import residual_income_model
from models.portfolio import position_sizes, concentration_analysis
from models.utils import rank
from scripts.report_excel import build_excel
from scripts.report_html import build_html
from data.macro_client import MacroClient
from models.macro import (assess_macro_regime, compute_macro_adjustments,
                          print_macro_summary, generate_sector_signals,
                          compute_sector_rs_from_local)
from models.narrative import generate_stock_narrative, generate_financial_summary
from data.news_client import NewsClient
from data.tiingo_client import TiingoClient
from data.sec_legal_client import SECLegalClient
from data.finnhub_supply_client import FinnhubSupplyClient
from data.sec_supply_client import SECSupplyClient
from data.sec_xbrl_client import SECXBRLClient
from data.sec_insider_client import SECInsiderClient
from data.culture_client import CultureClient

from scripts.config import (DEFAULT_RISK_FREE_RATE, ERP, TERMINAL_GROWTH_RATE,
                            MIN_MARKET_CAP, WACC_FLOOR, WACC_CAP,
                            GROWTH_WEIGHT_FCF, GROWTH_WEIGHT_REV,
                            GROWTH_WEIGHT_ANALYST_ST, GROWTH_WEIGHT_ANALYST_LT,
                            GROWTH_WEIGHT_EARNINGS_G, GROWTH_WEIGHT_FUNDAMENTAL,
                            SURPRISE_THRESHOLD, SURPRISE_UPLIFT,
                            MARGIN_TREND_SENSITIVITY,
                            BETA_MIN, BETA_MAX, RE_MIN, RE_MAX,
                            CAPEX_DA_THRESHOLD, EXCESS_CAPEX_ADDBACK,
                            YIELD_CEILING_MULT, HYPER_GROWTH_YIELD,
                            HYPER_GROWTH_CAP, ANALYST_HAIRCUT, FALLBACK_GROWTH,
                            DCF_YEARS, DCF_STAGE1,
                            EXIT_MULT_DIVERGENCE_THRESHOLD,
                            EXIT_MULT_DEFAULT_EV_EBITDA,
                            EXIT_MULT_MIN, EXIT_MULT_MAX,
                            MC_ITERATIONS, MC_GROWTH_SIGMA_RATIO, MC_WACC_SIGMA,
                            MC_TERMINAL_GROWTH_SIGMA, MC_EXIT_MULT_SIGMA_RATIO,
                            MC_HIGH_DIVERGENCE_SIGMA_MULT,
                            DDM_HIGH_GROWTH_YEARS, DDM_BLEND_WEIGHT,
                            DCF_BLEND_WEIGHT_WITH_DDM, DDM_DIVERGENCE_THRESHOLD,
                            BLEND_TRIGGER, BLEND_DCF_WEIGHT, BLEND_MULT_WEIGHT,
                            EV_EBITDA_OUTLIER_MAX, MIN_SECTOR_STOCKS,
                            DATA_QUALITY_MIN, MIN_MORNINGSTAR_SAMPLE,
                            _get_sector_config)
from scripts.scoring import (_mc_confidence_label, apply_screening_matrix,
                             compute_continuous_scores,
                             apply_composite_rating_override,
                             _print_validation_stats)


# ---------------------------------------------------------------------------
# Local price file helpers
# ---------------------------------------------------------------------------

def _load_local_prices(ticker, prices_dir):
    """Load Close price series from local Parquet file. Returns pd.Series or None."""
    if not prices_dir:
        return None
    path = os.path.join(prices_dir, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)[['Close']].sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df['Close']
    except Exception:
        return None


def _compute_rolling_beta(stock_close, market_close, window_years):
    """Compute beta and R² over a trailing window of *window_years* years."""
    window_days = int(window_years * 252)
    s = stock_close.tail(window_days)
    m = market_close.reindex(s.index, method='nearest').reindex(s.index)
    combined = pd.DataFrame({'s': s, 'm': m}).dropna()
    if len(combined) < 60:
        return None
    stock_ret  = combined['s'].pct_change().dropna().values
    market_ret = combined['m'].pct_change().dropna().values
    n = min(len(stock_ret), len(market_ret))
    return calculate_beta(stock_ret[:n], market_ret[:n])


def _realized_vol(close_series, window_days=252):
    """Annualized realized volatility over the trailing *window_days* trading days."""
    ret = close_series.pct_change().dropna()
    tail = ret.tail(window_days)
    if len(tail) < 60:
        return None
    return float(tail.std() * np.sqrt(252))


def _momentum_12_1(close_series, as_of=None):
    """12-minus-1 month price momentum (skips most recent month to avoid reversal)."""
    as_of = as_of or pd.Timestamp.today().normalize()
    mo12 = as_of - pd.DateOffset(months=12)
    mo1  = as_of - pd.DateOffset(months=1)
    s    = close_series.loc[close_series.index <= as_of]
    if s.empty:
        return None
    after_mo12 = s.loc[s.index >= mo12]
    before_mo1 = s.loc[s.index <= mo1]
    if after_mo12.empty or before_mo1.empty:
        return None
    p_start = float(after_mo12.iloc[0])
    p_end   = float(before_mo1.iloc[-1])
    if p_start <= 0:
        return None
    return (p_end - p_start) / p_start


def _max_drawdown_period(close_series, start, end):
    """Max drawdown (as negative fraction) within [start, end]. Returns None if no data."""
    s = close_series.loc[(close_series.index >= pd.Timestamp(start)) &
                         (close_series.index <= pd.Timestamp(end))]
    if len(s) < 5:
        return None
    roll_max = s.cummax()
    dd = (s - roll_max) / roll_max
    return float(dd.min())


# ---------------------------------------------------------------------------
# Ticker universe helpers
# ---------------------------------------------------------------------------

def _read_wiki_tables(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    return pd.read_html(io.StringIO(html))

def get_sp500_tickers():
    tables = _read_wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return tables[0]['Symbol'].tolist()

def get_nyse_tickers():
    try:
        return pd.read_csv("nyse_tickers.csv")['Symbol'].tolist()
    except FileNotFoundError:
        print("nyse_tickers.csv not found, skipping NYSE tickers.")
        return []

def get_dow_tickers():
    tables = _read_wiki_tables("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
    for table in tables:
        if 'Symbol' in table.columns:
            return table['Symbol'].tolist()
    return []


# ---------------------------------------------------------------------------
# CAPM / cost-of-equity helpers (Worksheet Steps 4A-4B)
# ---------------------------------------------------------------------------


def select_cost_of_equity(financials, risk_free_rate, yf_client=None, ticker=None,
                          erp=None, tiingo_client=None):
    """Select cost of equity using a four-level hierarchy.

    Tries each method in order, returning the first that passes validation:
      1. CAPM with computed beta + R² quality gate (preferred)
      2. CAPM with yfinance-reported beta (fallback)
      3. GGM-implied Re (for dividend payers)
      4. Build-Up (last resort)

    Args:
        financials: Dict of financial data from YFinanceClient.
        risk_free_rate: Current risk-free rate (10-yr Treasury yield).
        yf_client: Optional YFinanceClient for computing beta from prices.
        ticker: Optional ticker symbol for price history lookup.
        erp: Equity risk premium override (defaults to module-level ERP).

    Returns:
        Tuple of (cost_of_equity, method_label, beta_diagnostics_or_None).
    """
    if erp is None:
        erp = ERP
    info = (financials.get('info') or {}) if financials else {}
    beta_diag = None

    # 1. CAPM with computed beta and R² quality gate
    if yf_client and ticker:
        try:
            stock_prices = yf_client.fetch_history(ticker, period="5y")
            market_prices = yf_client.fetch_history('SPY', period="5y")
            # Fall back to Tiingo if yfinance returns insufficient price data
            if tiingo_client and tiingo_client.available:
                if stock_prices is None or len(stock_prices) <= 60:
                    stock_prices = tiingo_client.fetch_history(ticker, period="5y")
                if market_prices is None or len(market_prices) <= 60:
                    market_prices = tiingo_client.fetch_history('SPY', period="5y")
            if (stock_prices is not None and market_prices is not None
                    and len(stock_prices) > 60):
                combined = pd.DataFrame({
                    'stock': stock_prices,
                    'market': market_prices,
                }).dropna()
                if len(combined) > 60:
                    stock_ret = combined['stock'].pct_change().dropna().values
                    market_ret = combined['market'].pct_change().dropna().values
                    min_len = min(len(stock_ret), len(market_ret))
                    stock_ret = stock_ret[:min_len]
                    market_ret = market_ret[:min_len]

                    beta_result = calculate_beta(stock_ret, market_ret)
                    r2_class, r2_method = r2_diagnostic(beta_result['r_squared'])
                    beta_diag = {
                        **beta_result,
                        'r2_classification': r2_class,
                        'r2_method': r2_method,
                    }

                    # Use computed beta if R² is at least directional (≥ 0.40)
                    if r2_class in ('reliable', 'directional'):
                        adj_beta = beta_result['adjusted_beta']
                        if BETA_MIN < adj_beta < BETA_MAX:
                            re = risk_free_rate + adj_beta * erp
                            if RE_MIN < re < RE_MAX:
                                return re, f'capm ({r2_class})', beta_diag
        except Exception:
            pass  # Fall through to yfinance beta

    # 2. CAPM with yfinance-reported beta (no R² quality check)
    beta = info.get('beta')
    if beta is not None and BETA_MIN < beta < BETA_MAX:
        re = risk_free_rate + beta * erp
        if RE_MIN < re < RE_MAX:
            return re, 'capm', beta_diag

    # 3. GGM-implied: Re = D1/P + g  (works for dividend payers)
    div_rate = info.get('dividendRate')
    price = info.get('currentPrice') or info.get('regularMarketPrice')
    div_yield = (div_rate / price) if (div_rate and price and price > 0) else None
    if div_yield and div_yield > 0:
        re = ggm_implied_re(div_yield, TERMINAL_GROWTH_RATE)
        if re is not None and RE_MIN < re < RE_MAX:
            return re, 'ggm', beta_diag

    # 4. Build-Up fallback (no size/industry premiums — too imprecise)
    re = buildup_re(risk_free_rate, erp, size_premium=0, industry_premium=0)
    return re, 'buildup', beta_diag


# ---------------------------------------------------------------------------
# Growth estimation helpers
# ---------------------------------------------------------------------------

def _get_analyst_lt_growth(yf_data):
    """Extract analyst forward growth estimate from yfinance growth_estimates.

    Priority: LTG (long-term ~5yr) → +1y (next year) → 0y (current year).
    LTG is often NaN in yfinance, so +1y is the practical primary signal.
    Returns decimal (e.g. 0.12 for 12%) or None.
    """
    ge = yf_data.get('growth_estimates')
    if ge is not None and hasattr(ge, 'empty') and not ge.empty:
        try:
            # yfinance column name varies across versions: 'stockTrend', 'Stock', etc.
            col = next((c for c in ('stockTrend', 'Stock') if c in ge.columns), None)
            if col is None:
                return None
            # Try LTG first, then +1y, then 0y
            for period in ['LTG', '+1y', '0y']:
                if period in ge.index:
                    val = ge.loc[period, col]
                    if pd.notna(val) and isinstance(val, (int, float)):
                        return float(val)
        except Exception:
            pass
    return None


def _get_earnings_growth(yf_data):
    """Extract analyst 1-year forward earnings growth from info dict.

    Returns decimal (e.g. 0.18 for 18%) or None.
    """
    info = yf_data.get('info') or {}
    eg = info.get('earningsGrowth')
    if eg is not None and isinstance(eg, (int, float)):
        return float(eg)
    return None


def _compute_surprise_adjustment(yf_data):
    """Compute growth adjustment from earnings surprise history.

    If the company consistently beats estimates by > SURPRISE_THRESHOLD,
    analyst estimates are systematically low → uplift growth.
    If consistently misses by > threshold → penalise growth.

    Returns adjustment in decimal (e.g. +0.015 or -0.015 or 0.0).
    Also returns the average surprise % for display.
    """
    eh = yf_data.get('earnings_history')
    if eh is None or not hasattr(eh, 'empty') or eh.empty:
        return 0.0, None

    try:
        if 'surprisePercent' not in eh.columns:
            return 0.0, None
        surprises = eh['surprisePercent'].dropna()
        if len(surprises) < 2:
            return 0.0, None
        avg_surprise = float(surprises.mean())
        if avg_surprise > SURPRISE_THRESHOLD:
            return SURPRISE_UPLIFT, avg_surprise
        elif avg_surprise < -SURPRISE_THRESHOLD:
            return -SURPRISE_UPLIFT, avg_surprise
        else:
            return 0.0, avg_surprise
    except Exception:
        return 0.0, None


def _compute_margin_trend(yf_data):
    """Compute operating margin trend from income statement.

    Calculates operating margin (Operating Income / Total Revenue) for each
    year in the income statement and returns the average annual change.

    If margins are expanding, FCF should grow faster than revenue.
    Returns annual margin change in decimal (e.g. +0.02 = expanding 2pp/yr)
    or None if insufficient data.
    """
    inc = yf_data.get('income_statement')
    if inc is None or (hasattr(inc, 'empty') and inc.empty):
        inc = yf_data.get('income_stmt')
    if inc is None or (hasattr(inc, 'empty') and inc.empty):
        return None

    op_inc_keys = ['Operating Income', 'Total Operating Income As Reported']
    rev_keys = ['Total Revenue']

    margins = []
    # yfinance columns are fiscal years (most recent first)
    for col in inc.columns:
        year_data = inc[col]
        op_inc = None
        for k in op_inc_keys:
            if k in year_data.index and pd.notna(year_data[k]):
                op_inc = float(year_data[k])
                break
        rev = None
        for k in rev_keys:
            if k in year_data.index and pd.notna(year_data[k]):
                rev = float(year_data[k])
                break
        if op_inc is not None and rev and rev > 0:
            margins.append(op_inc / rev)

    if len(margins) < 2:
        return None

    # margins[0] is most recent, margins[-1] is oldest
    # Annual change = (newest - oldest) / number_of_gaps
    n_gaps = len(margins) - 1
    annual_change = (margins[0] - margins[-1]) / n_gaps
    return annual_change


def _compute_gross_margin(yf_data):
    """Compute median gross margin from income statement.

    Gross Margin = Gross Profit / Total Revenue for each fiscal year.
    Returns the median across available years (more robust to one-off swings
    than the mean). A high, stable gross margin (>40%) signals pricing power
    and is a classic moat indicator.

    Returns median gross margin in decimal (e.g. 0.45 = 45%) or None.
    """
    inc = yf_data.get('income_statement')
    if inc is None or (hasattr(inc, 'empty') and inc.empty):
        inc = yf_data.get('income_stmt')
    if inc is None or (hasattr(inc, 'empty') and inc.empty):
        return None

    gp_keys = ['Gross Profit']
    rev_keys = ['Total Revenue']

    margins = []
    for col in inc.columns:
        year_data = inc[col]
        gp = None
        for k in gp_keys:
            if k in year_data.index and pd.notna(year_data[k]):
                gp = float(year_data[k])
                break
        rev = None
        for k in rev_keys:
            if k in year_data.index and pd.notna(year_data[k]):
                rev = float(year_data[k])
                break
        if gp is not None and rev and rev > 0:
            margins.append(gp / rev)

    if not margins:
        return None

    margins_sorted = sorted(margins)
    n = len(margins_sorted)
    if n % 2 == 1:
        return margins_sorted[n // 2]
    return (margins_sorted[n // 2 - 1] + margins_sorted[n // 2]) / 2.0


def _extract_latest_financials(yf_data):
    """Extract most recent year revenue, operating income, net income.

    Returns dict with absolute dollar values (or None for missing fields).
    Used for profit pool analysis: sector-level revenue/profit concentration.
    """
    inc = yf_data.get('income_statement')
    if inc is None or (hasattr(inc, 'empty') and inc.empty):
        inc = yf_data.get('income_stmt')
    if inc is None or (hasattr(inc, 'empty') and inc.empty):
        return {}
    latest = inc.iloc[:, 0]  # most recent fiscal year

    rev = None
    for k in ['Total Revenue']:
        if k in latest.index and pd.notna(latest[k]):
            rev = float(latest[k])
            break

    op_inc = None
    for k in ['Operating Income', 'Total Operating Income As Reported']:
        if k in latest.index and pd.notna(latest[k]):
            op_inc = float(latest[k])
            break

    net_inc = None
    for k in ['Net Income', 'Net Income Common Stockholders']:
        if k in latest.index and pd.notna(latest[k]):
            net_inc = float(latest[k])
            break

    return {
        'revenue': rev,
        'operating_income': op_inc,
        'net_income': net_inc,
    }


def _compute_shareholder_yield(yf_data, mcap):
    """Compute total shareholder yield = (dividends + buybacks) / market cap.

    Uses the most recent fiscal year from cash flow statement.
    Returns dict with 'shareholder_yield' and 'buyback_rate' (both decimal), or None.
    """
    if not mcap or mcap <= 0:
        return None
    cf = yf_data.get('cash_flow')
    if cf is None or (hasattr(cf, 'empty') and cf.empty):
        return None

    latest = cf.iloc[:, 0]

    # Dividends paid (negative in cash flow = cash out)
    div_paid = 0
    for k in ['Common Stock Dividend Paid', 'Cash Dividends Paid']:
        if k in latest.index and pd.notna(latest[k]):
            div_paid = abs(float(latest[k]))
            break

    # Share buybacks (negative in cash flow = cash out)
    buyback = 0
    for k in ['Repurchase Of Capital Stock', 'Common Stock Payments']:
        if k in latest.index and pd.notna(latest[k]):
            buyback = abs(float(latest[k]))
            break

    # Subtract any new issuance to get net buyback
    issuance = 0
    for k in ['Issuance Of Capital Stock', 'Common Stock Issuance']:
        if k in latest.index and pd.notna(latest[k]):
            issuance = abs(float(latest[k]))
            break
    net_buyback = max(0, buyback - issuance)

    total_return = div_paid + net_buyback
    shareholder_yield = 0.0 if (total_return == 0 and div_paid == 0) else total_return / mcap
    buyback_rate = net_buyback / mcap

    # Sanity cap — yield > 50% is almost certainly a data error (e.g. stale
    # yfinance cash-flow snapshot with mismatched period/mcap units).
    if shareholder_yield > 0.50:
        shareholder_yield = None
        buyback_rate = None

    return {'shareholder_yield': shareholder_yield, 'buyback_rate': buyback_rate}


# ---------------------------------------------------------------------------
# Forward DCF (Worksheet Step 5A)
# ---------------------------------------------------------------------------

def run_forward_dcf(yf_data, wacc, sector=None, exit_multiple=None, roic_data=None,
                    terminal_growth_adj=0.0, wacc_sigma=None, tg_sigma=None,
                    growth_sigma_mult=1.0, growth_weight_shift=0.0):
    """Run a two-stage 10-year DCF with sector-specific parameters.

    Includes: sector growth caps, averaged FCF for cyclicals,
    owner-earnings adjustment, FCF yield mean-reversion ceiling,
    fundamental growth signal, exit multiple cross-check, and
    Monte Carlo uncertainty quantification.

    Args:
        terminal_growth_adj: Additive adjustment to terminal growth (macro overlay).
        wacc_sigma: Override for MC WACC sigma (defaults to MC_WACC_SIGMA).
        tg_sigma: Override for MC terminal growth sigma (defaults to MC_TERMINAL_GROWTH_SIGMA).
        growth_sigma_mult: Multiplicative adjustment to growth sigma (macro overlay).
        growth_weight_shift: Shift from analyst LT weight to fundamental weight (macro overlay).

    Returns:
        Tuple of (fair_value, sensitivity_range, fcf_growth, growth_diag, mc_result),
        or (None, None, None, {}, None) on insufficient data.
    """
    if wacc_sigma is None:
        wacc_sigma = MC_WACC_SIGMA
    if tg_sigma is None:
        tg_sigma = MC_TERMINAL_GROWTH_SIGMA
    cf = yf_data.get('cash_flow')
    info = yf_data.get('info') or {}

    if cf is None or cf.empty:
        return None, None, None, {}, None
    if 'Free Cash Flow' not in cf.index:
        return None, None, None, {}, None
    cfg = _get_sector_config(sector)
    term_g = cfg['terminal_growth'] + terminal_growth_adj

    if wacc is None or wacc <= term_g:
        return None, None, None, {}, None

    fcf_series = cf.loc['Free Cash Flow'].dropna().sort_index()
    fcf_values = fcf_series.values.tolist()
    if not fcf_values:
        return None, None, None, {}, None

    # --- Fix E: For cyclical sectors, average multiple years of FCF ---
    avg_years = cfg['avg_fcf_years']
    if avg_years > 1 and len(fcf_values) >= 2:
        recent = [v for v in fcf_values[-avg_years:] if v > 0]
        base_fcf = sum(recent) / len(recent) if recent else fcf_values[-1]
    else:
        base_fcf = fcf_values[-1]

    # --- Fix F: Growth capex add-back for capex-heavy companies ---
    # If Capex > 2× D&A, significant growth capex is depressing accounting FCF.
    # Add back 50% of excess capex (above maintenance proxy = D&A).
    # This is more conservative than full owner earnings (OCF - D&A) which
    # over-inflates companies like GOOG where capex is partially maintenance.
    if cfg['check_owner_earnings']:
        try:
            sorted_cols = sorted(cf.columns)
            latest_cf = cf[sorted_cols[-1]]
        except Exception:
            latest_cf = cf.iloc[:, -1] if cf.columns.is_monotonic_increasing else cf.iloc[:, 0]

        ocf_labels = ['Operating Cash Flow', 'Total Cash From Operating Activities']
        da_labels = ['Depreciation And Amortization', 'Depreciation Amortization Depletion']
        capex_labels = ['Capital Expenditure', 'Capital Expenditures']

        ocf = None
        for lbl in ocf_labels:
            if lbl in latest_cf.index and pd.notna(latest_cf[lbl]):
                ocf = latest_cf[lbl]
                break
        da = None
        for lbl in da_labels:
            if lbl in latest_cf.index and pd.notna(latest_cf[lbl]):
                da = latest_cf[lbl]
                break
        capex = None
        for lbl in capex_labels:
            if lbl in latest_cf.index and pd.notna(latest_cf[lbl]):
                capex = abs(latest_cf[lbl])
                break

        if ocf and da and capex and da > 0 and capex / da > CAPEX_DA_THRESHOLD:
            # Excess capex above maintenance (D&A) is growth capex.
            # Add back portion of it — assume half is truly discretionary growth.
            excess_capex = capex - da
            growth_add_back = excess_capex * EXCESS_CAPEX_ADDBACK
            adjusted = base_fcf + growth_add_back
            if adjusted > 0:
                base_fcf = adjusted

    if base_fcf <= 0:
        return None, None, None, {}, None

    # --- Mean-reversion: cap FCF at sector-normal yield ceiling ---
    # If FCF/Mcap >> sector normal, FCF is likely at peak cycle.
    # Cap at 1.25× sector normal yield to prevent extrapolating peak earnings.
    mcap = info.get('marketCap')
    if mcap and mcap > 0 and base_fcf > 0:
        actual_yield = base_fcf / mcap
        norm_yield = cfg.get('norm_fcf_yield', 0.035)
        yield_ceiling = norm_yield * YIELD_CEILING_MULT
        if actual_yield > yield_ceiling:
            base_fcf = mcap * yield_ceiling

    # --- FCF growth estimation ---
    fcf_cagr = None
    if len(fcf_values) >= 2 and fcf_values[0] > 0 and fcf_values[-1] > 0:
        n = len(fcf_values) - 1
        fcf_cagr = (fcf_values[-1] / fcf_values[0]) ** (1 / n) - 1

    # Revenue CAGR as secondary signal
    rev_cagr = None
    inc = yf_data.get('income_statement')
    if inc is None or (hasattr(inc, 'empty') and inc.empty):
        inc = yf_data.get('income_stmt')
    if inc is not None and not (hasattr(inc, 'empty') and inc.empty):
        if 'Total Revenue' in inc.index:
            rev_series = inc.loc['Total Revenue'].dropna().sort_index()
            revs = [v for v in rev_series.values if v and v > 0]
            if len(revs) >= 2:
                # revs sorted ascending: [oldest, ..., newest]
                rev_cagr = (revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1

    analyst_st = info.get('revenueGrowth')          # short-term (1yr revenue)
    analyst_lt = _get_analyst_lt_growth(yf_data)      # long-term (~5yr)
    earnings_g = _get_earnings_growth(yf_data)        # 1yr earnings growth

    # --- Sector-specific growth cap with hyper-growth override ---
    best_analyst = analyst_lt or analyst_st
    growth_cap = cfg['growth_cap']
    if (best_analyst is not None and best_analyst > growth_cap
            and mcap and mcap > 0 and base_fcf > 0):
        actual_yield = base_fcf / mcap
        if actual_yield < HYPER_GROWTH_YIELD:
            growth_cap = max(growth_cap, min(HYPER_GROWTH_CAP, best_analyst * ANALYST_HAIRCUT))

    # --- 6-signal weighted average (auto-normalise when signals missing) ---
    growth_signals = []
    growth_weights = []
    if fcf_cagr is not None:
        growth_signals.append(fcf_cagr)
        growth_weights.append(GROWTH_WEIGHT_FCF)
    if rev_cagr is not None:
        growth_signals.append(rev_cagr)
        growth_weights.append(GROWTH_WEIGHT_REV)
    if analyst_st is not None:
        growth_signals.append(analyst_st)
        growth_weights.append(GROWTH_WEIGHT_ANALYST_ST)
    # Apply macro growth weight shift: move weight from analyst LT to fundamental
    _w_analyst_lt = max(0.05, GROWTH_WEIGHT_ANALYST_LT + growth_weight_shift)
    _w_fundamental = max(0.05, GROWTH_WEIGHT_FUNDAMENTAL - growth_weight_shift)
    if analyst_lt is not None:
        growth_signals.append(analyst_lt)
        growth_weights.append(_w_analyst_lt)
    if earnings_g is not None:
        growth_signals.append(earnings_g)
        growth_weights.append(GROWTH_WEIGHT_EARNINGS_G)

    # Signal 6: Fundamental growth (Reinvestment Rate × ROIC)
    fund_result = calculate_fundamental_growth(yf_data,
                    roic_override=roic_data.get('avg_roic') if roic_data else None)
    fundamental_g = fund_result.get('fundamental_growth')
    if fundamental_g is not None:
        growth_signals.append(fundamental_g)
        growth_weights.append(_w_fundamental)

    if growth_signals:
        total_weight = sum(growth_weights)
        weighted_avg = sum(s * w for s, w in zip(growth_signals, growth_weights)) / total_weight
    else:
        weighted_avg = FALLBACK_GROWTH

    # --- Margin trend adjustment ---
    margin_trend = _compute_margin_trend(yf_data)
    if margin_trend is not None:
        weighted_avg += margin_trend * MARGIN_TREND_SENSITIVITY

    # --- Earnings surprise adjustment ---
    surprise_adj, surprise_avg = _compute_surprise_adjustment(yf_data)
    weighted_avg += surprise_adj

    fcf_growth = min(growth_cap, max(term_g, weighted_avg))

    # --- Two-stage DCF via shared model function (GGM terminal value) ---
    ev_ggm = two_stage_ev(base_fcf, fcf_growth, wacc, term_g,
                          total_years=DCF_YEARS, stage1_years=DCF_STAGE1)
    if ev_ggm is None or ev_ggm <= 0:
        return None, None, None, {}, None

    net_debt = get_net_debt(yf_data)
    shares = info.get('sharesOutstanding')
    fv_ggm = fair_value_per_share(ev_ggm, net_debt, shares)

    # --- Exit multiple cross-check ---
    exit_mult_fv = None
    tv_method_spread = None
    base_ebitda = None

    # Extract base EBITDA = Operating Income + D&A
    inc_stmt = yf_data.get('income_statement')
    if inc_stmt is not None and not inc_stmt.empty:
        try:
            sorted_inc_cols = sorted(inc_stmt.columns)
            latest_inc = inc_stmt[sorted_inc_cols[-1]]
        except Exception:
            latest_inc = inc_stmt.iloc[:, 0]
        op_inc_val = None
        for k in ['Operating Income', 'Total Operating Income As Reported']:
            if k in latest_inc.index and pd.notna(latest_inc[k]):
                op_inc_val = float(latest_inc[k])
                break
        da_val = None
        cf_for_da = yf_data.get('cash_flow')
        if cf_for_da is not None and not cf_for_da.empty:
            try:
                sorted_cf_cols = sorted(cf_for_da.columns)
                latest_cf_da = cf_for_da[sorted_cf_cols[-1]]
            except Exception:
                latest_cf_da = cf_for_da.iloc[:, 0]
            for k in ['Depreciation And Amortization', 'Depreciation Amortization Depletion']:
                if k in latest_cf_da.index and pd.notna(latest_cf_da[k]):
                    da_val = abs(float(latest_cf_da[k]))
                    break
        if op_inc_val and da_val:
            base_ebitda = op_inc_val + da_val

    if base_ebitda and base_ebitda > 0 and exit_multiple and shares and shares > 0:
        ev_exit = two_stage_ev_exit_multiple(
            base_fcf, fcf_growth, wacc, term_g,
            base_ebitda, exit_multiple,
            total_years=DCF_YEARS, stage1_years=DCF_STAGE1)
        if ev_exit and ev_exit > 0:
            exit_mult_fv = fair_value_per_share(ev_exit, net_debt, shares)

    # Average GGM and exit multiple FVs
    fv = fv_ggm
    if fv_ggm and exit_mult_fv:
        fv = (fv_ggm + exit_mult_fv) / 2.0
        avg_fv = (fv_ggm + exit_mult_fv) / 2.0
        tv_method_spread = abs(fv_ggm - exit_mult_fv) / avg_fv if avg_fv > 0 else None
    elif exit_mult_fv:
        fv = exit_mult_fv

    # Sensitivity range (supplementary)
    sens_range = None
    if shares and shares > 0:
        sens = dcf_sensitivity(base_fcf, fcf_growth, wacc, term_g,
                               net_debt, shares, years=DCF_YEARS, stage1=DCF_STAGE1)
        vals = [v for v in sens.values() if v is not None]
        if vals:
            sens_range = (min(vals), max(vals))

    # --- Monte Carlo uncertainty quantification ---
    mc_result = None
    if base_fcf > 0 and shares and shares > 0:
        g_sigma = abs(fcf_growth) * MC_GROWTH_SIGMA_RATIO if fcf_growth != 0 else 0.02
        g_sigma *= growth_sigma_mult  # macro overlay: widen in stress
        em_sigma = exit_multiple * MC_EXIT_MULT_SIGMA_RATIO if exit_multiple else None
        # Widen sigma if TV methods diverge significantly
        if tv_method_spread and tv_method_spread > EXIT_MULT_DIVERGENCE_THRESHOLD:
            g_sigma *= MC_HIGH_DIVERGENCE_SIGMA_MULT
            if em_sigma:
                em_sigma *= MC_HIGH_DIVERGENCE_SIGMA_MULT

        mc_result = monte_carlo_dcf(
            base_fcf, fcf_growth, wacc, term_g,
            net_debt, shares,
            base_ebitda=base_ebitda, exit_multiple=exit_multiple,
            n_iterations=MC_ITERATIONS,
            growth_sigma=g_sigma, wacc_sigma=wacc_sigma,
            tg_sigma=tg_sigma, exit_mult_sigma=em_sigma,
            total_years=DCF_YEARS, stage1_years=DCF_STAGE1)

    # Use MC percentiles for bear/bull instead of sensitivity grid
    if mc_result:
        sens_range = (mc_result['p10_fv'], mc_result['p90_fv'])

    growth_diag = {
        'analyst_ltg': analyst_lt,
        'earnings_growth': earnings_g,
        'margin_trend': margin_trend,
        'surprise_avg': surprise_avg,
        'fundamental_growth': fundamental_g,
        'reinvestment_rate': fund_result.get('reinvestment_rate'),
        'exit_mult_fv': exit_mult_fv,
        'tv_method_spread': tv_method_spread,
        'mc_result': mc_result,
    }
    return fv, sens_range, fcf_growth, growth_diag, mc_result


# ---------------------------------------------------------------------------
# Dividend Discount Model helper
# ---------------------------------------------------------------------------

def _annualise_dividends(div_series):
    """Convert a per-payment dividend Series to annual DPS (oldest first).

    Groups by calendar year, sums payments per year, and returns a list
    of annual totals for years that have at least one payment.
    """
    if div_series is None or len(div_series) == 0:
        return []
    # Normalise DataFrame → Series (yfinance >=1.2 may return a single-column
    # DataFrame from stock.dividends instead of the expected Series).
    if isinstance(div_series, pd.DataFrame):
        div_series = div_series.iloc[:, 0] if not div_series.empty else pd.Series(dtype=float)
    annual = div_series.groupby(div_series.index.year).sum()
    return annual.sort_index().tolist()


def run_ddm_valuation(yf_data, div_series, cost_of_equity, analyst_ltg=None):
    """Run Dividend Discount Model valuation for a single stock.

    Args:
        yf_data: Dict of financial data from YFinanceClient.
        div_series: Pandas Series of historical dividends (from fetch_dividends).
        cost_of_equity: Required return (from CAPM / select_cost_of_equity).
        analyst_ltg: Analyst long-term growth estimate (optional).

    Returns:
        Dict with DDM results or dict with eligible=False for non-payers.
    """
    info = (yf_data.get('info') or {}) if yf_data else {}
    eps = info.get('trailingEps')
    dps = info.get('dividendRate')
    payout = info.get('payoutRatio')
    roe = None
    # Compute ROE from balance sheet if available
    bs = yf_data.get('balance_sheet') if yf_data else None
    inc = yf_data.get('income_statement') if yf_data else None
    if bs is not None and not bs.empty and inc is not None and not inc.empty:
        try:
            sorted_bs = sorted(bs.columns)
            equity = bs[sorted_bs[-1]].get('Stockholders Equity')
            sorted_inc = sorted(inc.columns)
            ni = inc[sorted_inc[-1]].get('Net Income')
            if equity and ni and equity > 0:
                roe = ni / equity
        except Exception:
            pass

    annual_divs = _annualise_dividends(div_series)

    # 1. Eligibility check
    elig = ddm_eligibility(annual_divs, payout, eps, dps)
    result = {
        'ddm_eligible': elig['eligible'],
        'ddm_reason': elig['reason'],
        'ddm_consecutive_years': elig['consecutive_years'],
        'ddm_payout_flag': elig['payout_flag'],
        'ddm_fv': None,
        'ddm_h_fv': None,
        'ddm_growth': None,
        'ddm_div_cagr': None,
        'ddm_sustainable_growth': None,
        'ddm_mc_median': None,
        'ddm_mc_p10': None,
        'ddm_mc_p90': None,
        'ddm_mc_cv': None,
    }

    if not elig['eligible']:
        return result

    # 2. Growth estimation
    growth_est = estimate_ddm_growth(annual_divs, payout, roe, analyst_ltg)
    g = growth_est['growth']
    if g is None:
        g = TERMINAL_GROWTH_RATE  # fallback to terminal growth
    result['ddm_growth'] = g
    result['ddm_div_cagr'] = growth_est['div_cagr']
    result['ddm_sustainable_growth'] = growth_est['sustainable_growth']

    re = cost_of_equity
    tg = TERMINAL_GROWTH_RATE

    # 3. Two-stage DDM
    ddm_fv = two_stage_ddm(dps, g, tg, re, years=DDM_HIGH_GROWTH_YEARS)
    result['ddm_fv'] = ddm_fv

    # 4. H-Model cross-check
    h_fv = ddm_h_model(dps, g, tg, re, half_life=DDM_HIGH_GROWTH_YEARS)
    result['ddm_h_fv'] = h_fv

    # Average the two methods when both available
    if ddm_fv and h_fv:
        result['ddm_fv'] = (ddm_fv + h_fv) / 2.0
    elif h_fv:
        result['ddm_fv'] = h_fv

    # 5. Monte Carlo DDM
    mc = monte_carlo_ddm(dps, g, re, tg, n=MC_ITERATIONS, years=DDM_HIGH_GROWTH_YEARS)
    if mc:
        result['ddm_mc_median'] = mc['median_fv']
        result['ddm_mc_p10'] = mc['p10_fv']
        result['ddm_mc_p90'] = mc['p90_fv']
        result['ddm_mc_cv'] = mc['cv']

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _main():
    """Entry point: screen tickers, run DCF analysis, generate reports."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Excel file with tickers in first column')
    parser.add_argument('--validation', '-v',
                        help='Validation Excel (poor performers, same column structure as --input)')
    parser.add_argument('--macro', action='store_true',
                        help='Enable macro-economic overlay (adjusts ERP, growth, sigma based on market regime)')
    parser.add_argument('--prices-dir', default='output/prices',
                        help='Directory of per-ticker Parquet price files for rolling beta, '
                             'realized vol, momentum, and drawdown (default: output/prices)')
    args = parser.parse_args()
    prices_dir = args.prices_dir if os.path.isdir(args.prices_dir) else None

    # Fetch live risk-free rate (10-yr Treasury yield)
    risk_free_rate = fetch_risk_free_rate()
    print(f"Risk-free rate: {risk_free_rate:.2%} (10-yr Treasury)")

    # --- Macro-economic overlay (opt-in via --macro) ---
    macro_regime_result = None
    macro_adj = None
    effective_erp = ERP
    effective_tg_adj = 0.0
    effective_wacc_sigma = MC_WACC_SIGMA
    effective_growth_sigma_mult = 1.0
    effective_exit_mult_adj = 0.0
    effective_growth_weight_shift = 0.0
    sector_signals = {}
    commodity_data = {}
    sector_etf_data = {}
    local_rs = None

    if args.macro:
        try:
            mc_client = MacroClient()
            macro_indicators = mc_client.fetch_macro_indicators()
            macro_regime_result = assess_macro_regime(macro_indicators)
            macro_adj = compute_macro_adjustments(macro_regime_result)
            print_macro_summary(macro_regime_result, macro_adj)

            effective_erp = ERP + macro_adj['erp_adjustment']
            effective_tg_adj = macro_adj['terminal_growth_adjustment']
            effective_wacc_sigma = MC_WACC_SIGMA + macro_adj['wacc_sigma_adjustment']
            effective_growth_sigma_mult = macro_adj['growth_sigma_multiplier']
            effective_exit_mult_adj = macro_adj['exit_mult_adjustment']
            effective_growth_weight_shift = macro_adj['growth_weight_shift']

            # Sector headwind/tailwind analysis
            sector_etf_data = mc_client.fetch_sector_data()
            local_rs = None
            if prices_dir:
                try:
                    local_rs = compute_sector_rs_from_local(prices_dir)
                except Exception as _rs_exc:
                    print(f"  Sector RS from local prices failed ({_rs_exc}), skipping.")
            sector_signals = generate_sector_signals(sector_etf_data, macro_regime_result, local_rs=local_rs)
            # Commodity & cross-sector data for stock-level narrative
            commodity_data = mc_client.fetch_commodity_data()
        except Exception as e:
            print(f"  Macro overlay failed ({e}), proceeding with defaults.")

    ms_pfv_data = {}  # Morningstar Price/Fair Value ratios (if input file has them)
    ticker_source = {}  # ticker -> 'quality' | 'poor'

    def _load_pfv_from_xlsx(path, pfv_dict):
        """Extract tickers and P/FV ratios from a Morningstar-format xlsx."""
        import openpyxl as _ox
        wb = _ox.load_workbook(path)
        ws = wb[wb.sheetnames[0]]
        tickers = sorted(set(
            str(ws.cell(r, 1).value).strip()
            for r in range(2, ws.max_row + 1)
            if ws.cell(r, 1).value
        ))
        headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
        pfv_col = None
        for ci, h in enumerate(headers):
            if h and 'Price/Fair Value' in str(h):
                pfv_col = ci + 1
                break
        n_pfv = 0
        if pfv_col:
            for r in range(2, ws.max_row + 1):
                tk = ws.cell(r, 1).value
                pfv = ws.cell(r, pfv_col).value
                if tk and pfv and isinstance(pfv, (int, float)) and pfv > 0:
                    pfv_dict[str(tk).strip()] = pfv
                    n_pfv += 1
        return tickers, n_pfv

    # Always start with the full SP500/NYSE/DOW universe
    sp500 = set(get_sp500_tickers())
    nyse = set(get_nyse_tickers())
    dow = set(get_dow_tickers())
    all_tickers = sorted(sp500 | nyse | dow)
    for t in all_tickers:
        ticker_source[t] = 'quality'

    if args.input:
        # Merge input file tickers + P/FV data on top of the universe
        input_tickers, n_pfv = _load_pfv_from_xlsx(args.input, ms_pfv_data)
        existing = set(all_tickers)
        extra = [t for t in input_tickers if t not in existing]
        all_tickers = all_tickers + extra
        for t in extra:
            ticker_source[t] = 'quality'
        if n_pfv:
            print(f"Universe: {len(existing)} tickers + {len(extra)} extra from input "
                  f"+ {n_pfv} MS Price/FV ratios from {args.input}")
        else:
            print(f"Universe: {len(existing)} tickers + {len(extra)} extra from input "
                  f"(no Price/Fair Value column found in {args.input})")

    if args.validation:
        val_tickers, val_n_pfv = _load_pfv_from_xlsx(args.validation, ms_pfv_data)
        for t in val_tickers:
            ticker_source[t] = 'poor'
        existing = set(all_tickers)
        all_tickers = all_tickers + [t for t in val_tickers if t not in existing]
        print(f"Loaded {len(val_tickers)} validation tickers + {val_n_pfv} P/FV from {args.validation} "
              f"(combined universe: {len(all_tickers)} tickers, {len(ms_pfv_data)} total P/FV)")

    yf_client = YFinanceClient()

    # Tiingo client initialized here so it's available for Phase 1 beta calculation
    tiingo_client = TiingoClient(request_delay=0.5)
    if tiingo_client.available:
        print('Tiingo API configured — using as primary news source.')
    else:
        print('Tiingo API not configured (set TIINGO_API_KEY) — falling back to yfinance/Google RSS.')

    # -----------------------------------------------------------------------
    # Phase 1: Collect data for full universe (no ROIC > WACC pre-filter)
    # -----------------------------------------------------------------------
    print(f"Processing {len(all_tickers)} tickers (full universe)...")
    qualifying = []
    screen_cache = {}
    screen_outcomes = {'quality': {'total': 0, 'passed': 0},
                       'poor': {'total': 0, 'passed': 0}}

    for i, ticker in enumerate(all_tickers, 1):
        _grp = ticker_source.get(ticker, 'quality')
        screen_outcomes[_grp]['total'] += 1
        try:
            yf_data = yf_client.fetch_financials(ticker)
            info = yf_data.get('info') or {}
            sector = info.get('sector', '')

            roic_data = calculate_roic(yf_data)
            cost_of_equity, re_method, beta_diag = select_cost_of_equity(
                yf_data, risk_free_rate, yf_client, ticker, erp=effective_erp,
                tiingo_client=tiingo_client)
            wacc = calculate_wacc(yf_data, cost_of_equity)
            if wacc is not None:
                s_cfg = _get_sector_config(sector)
                wacc = max(s_cfg['wacc_floor'], min(s_cfg['wacc_cap'], wacc))

            spread = (roic_data['avg_roic'] - wacc
                      if (roic_data and wacc is not None) else None)
            qualifying.append(ticker)
            screen_outcomes[_grp]['passed'] += 1
            screen_cache[ticker] = {
                'roic_data': roic_data, 'wacc': wacc,
                'cost_of_equity': cost_of_equity,
                're_method': re_method, 'yf_data': yf_data,
                'beta_diag': beta_diag,
            }
            roic_str = f"ROIC {roic_data['avg_roic']:.1%} " if roic_data else "ROIC N/A "
            wacc_str = f"WACC {wacc:.1%} " if wacc is not None else "WACC N/A "
            spread_str = f"spread {spread:.1%}" if spread is not None else "spread N/A"
            print(f"  [{i}/{len(all_tickers)}] {ticker} - {roic_str}{wacc_str}{spread_str} [{re_method}]")

        except Exception as e:
            print(f"  [{i}/{len(all_tickers)}] {ticker} - error: {e}")
        # Flush after every ticker so the log reflects progress if OOM-killed
        sys.stdout.flush()

    # Free memory: drop ALL cached financials and price histories.
    # Qualifying tickers' data survives via screen_cache references.
    yf_client.evict_financials()
    yf_client.clear_history_cache()
    gc.collect()

    print(f"\n{len(qualifying)} tickers collected out of {len(all_tickers)} total.")
    if args.validation:
        for grp in ('quality', 'poor'):
            o = screen_outcomes[grp]
            rate = o['passed'] / o['total'] if o['total'] > 0 else 0
            print(f"  {grp:>8}: {o['passed']}/{o['total']} passed ({rate:.0%})")
    print()

    # Pre-compute sector median EV/EBITDA for exit multiple cross-check
    _pre_sector_ee = {}
    for ticker in qualifying:
        cached_pre = screen_cache[ticker]
        info_pre = (cached_pre['yf_data'].get('info') or {})
        ee_pre = info_pre.get('enterpriseToEbitda')
        sector_pre = info_pre.get('sector', '')
        if ee_pre and 0 < ee_pre < EV_EBITDA_OUTLIER_MAX:
            _pre_sector_ee.setdefault(sector_pre, []).append(ee_pre)
    sector_exit_multiples = {}
    for s, v in _pre_sector_ee.items():
        if len(v) >= MIN_SECTOR_STOCKS:
            med = sorted(v)[len(v) // 2]
            sector_exit_multiples[s] = max(EXIT_MULT_MIN, min(EXIT_MULT_MAX, med))

    # Apply macro exit multiple adjustment
    effective_exit_mult_default = max(EXIT_MULT_MIN,
        min(EXIT_MULT_MAX, EXIT_MULT_DEFAULT_EV_EBITDA + effective_exit_mult_adj))
    if effective_exit_mult_adj != 0.0:
        for s in sector_exit_multiples:
            sector_exit_multiples[s] = max(EXIT_MULT_MIN,
                min(EXIT_MULT_MAX, sector_exit_multiples[s] + effective_exit_mult_adj))

    # -----------------------------------------------------------------------
    # News pipeline: Tiingo (primary) + yfinance/Google RSS (fallback)
    # -----------------------------------------------------------------------
    news_client = NewsClient(request_delay=1.0, max_age_days=30)
    _sectors_for_news = set(
        (screen_cache[t]['yf_data'].get('info') or {}).get('sector', '')
        for t in qualifying
    )
    news_client.prefetch_all_sectors(_sectors_for_news)

    # SEC EDGAR: legal proceedings search
    sec_client = SECLegalClient(email='stockanalysis@example.com', request_delay=1.0)

    # Finnhub: supply chain relationships
    supply_client = FinnhubSupplyClient(request_delay=1.0)
    if supply_client.available:
        print('Finnhub supply chain API configured.')
    else:
        print('Finnhub supply chain API not configured (set FINNHUB_API_KEY).')

    # SEC EDGAR: supply chain extraction from 10-K filings (free fallback)
    sec_client._load_cik_map()
    sec_supply_client = SECSupplyClient(
        cik_map=sec_client._cik_map,
        name_map=sec_client._name_map,
        email='stockanalysis@example.com',
        request_delay=1.0,
    )

    # SEC EDGAR: XBRL cross-validation and historical depth
    sec_xbrl_client = SECXBRLClient(
        cik_map=sec_client._cik_map,
        name_map=sec_client._name_map,
        email='stockanalysis@example.com',
        request_delay=1.0,
    )

    # SEC EDGAR: insider transaction tracking from Form 4 filings
    sec_insider_client = SECInsiderClient(
        cik_map=sec_client._cik_map,
        name_map=sec_client._name_map,
        email='stockanalysis@example.com',
        request_delay=1.0,
        max_form4_files=15,
    )

    # Culture metrics client (no external API — derives signals from yfinance)
    culture_client = CultureClient()

    # -----------------------------------------------------------------------
    # Phase 2: Full analysis on qualifying tickers (Worksheet Steps 2-5)
    # -----------------------------------------------------------------------

    # Pre-load SPY local prices once for rolling-beta comparisons
    _spy_local = _load_local_prices('SPY', prices_dir)

    results = []
    for ticker in qualifying:
        print(f"Analyzing {ticker}...")
        try:
            cached = screen_cache[ticker]
            yf_data = cached['yf_data']
            wacc = cached['wacc']
            roic_data = cached['roic_data']
            cost_of_equity = cached['cost_of_equity']
            beta_diag = cached.get('beta_diag')

            # --- Price-history enrichments (local Parquet) ---
            _local_close = _load_local_prices(ticker, prices_dir)
            _ticker_realized_vol = None
            _ticker_momentum     = None
            _ticker_dd_2008      = None
            _ticker_dd_2020      = None
            _ticker_dd_2022      = None
            _rolling_beta_diag   = {}

            if _local_close is not None and len(_local_close) > 60:
                # 1. Realized volatility (252-day) → replaces fixed MC_WACC_SIGMA
                _ticker_realized_vol = _realized_vol(_local_close)

                # 2. 12-minus-1 month momentum
                _ticker_momentum = _momentum_12_1(_local_close)

                # 3. Max drawdown in key stress periods
                _ticker_dd_2008 = _max_drawdown_period(_local_close, '2008-01-01', '2009-03-31')
                _ticker_dd_2020 = _max_drawdown_period(_local_close, '2020-01-01', '2020-09-30')
                _ticker_dd_2022 = _max_drawdown_period(_local_close, '2022-01-01', '2022-12-31')

                # 4. Rolling beta across 1yr / 3yr / 5yr windows
                if _spy_local is not None:
                    for _yrs, _label in [(1, '1y'), (3, '3y'), (5, '5y')]:
                        _rb = _compute_rolling_beta(_local_close, _spy_local, _yrs)
                        if _rb:
                            _rolling_beta_diag[_label] = {
                                'beta': round(_rb['adjusted_beta'], 3),
                                'r2':   round(_rb['r_squared'], 3),
                            }
                    # Promote the best-R² window's beta into beta_diag
                    if _rolling_beta_diag:
                        _best = max(_rolling_beta_diag.items(),
                                    key=lambda kv: kv[1]['r2'])
                        if beta_diag is None:
                            beta_diag = {}
                        beta_diag['rolling_betas'] = _rolling_beta_diag
                        beta_diag['best_window']   = _best[0]
                        # Compute beta stability (std of betas across windows)
                        _betas = [v['beta'] for v in _rolling_beta_diag.values()]
                        beta_diag['beta_stability'] = round(float(np.std(_betas)), 3) if len(_betas) > 1 else 0.0

            # Use ticker realized vol for MC WACC sigma (floor at macro-adjusted base)
            _effective_wacc_sigma_ticker = effective_wacc_sigma
            if _ticker_realized_vol is not None:
                # WACC sigma ≈ 30% of realized equity vol (equity → WACC dampening)
                _rv_wacc = _ticker_realized_vol * 0.30
                _effective_wacc_sigma_ticker = max(effective_wacc_sigma, min(_rv_wacc, 0.04))

            # Gross margin (moat gate) and ROIC consistency
            gross_margin = _compute_gross_margin(yf_data)
            # Latest financials for profit pool analysis
            latest_fins = _extract_latest_financials(yf_data)
            roic_cv = None
            roic_years = roic_data.get('roic_by_year', {})
            if len(roic_years) >= 2:
                vals = list(roic_years.values())
                mean_r = sum(vals) / len(vals)
                if mean_r > 0:
                    var_r = sum((x - mean_r) ** 2 for x in vals) / (len(vals) - 1)
                    roic_cv = (var_r ** 0.5) / mean_r

            # Company description and CEO (Worksheet Step 3)
            info = yf_data.get('info') or {}
            # mcap must be read from THIS ticker's info, not the stale Phase 1 variable
            mcap = info.get('marketCap') or 0

            # Shareholder yield (dividends + buybacks) / market cap
            sy_result = _compute_shareholder_yield(yf_data, mcap)
            shareholder_yield = sy_result['shareholder_yield'] if sy_result else None
            share_buyback_rate = sy_result['buyback_rate'] if sy_result else None
            description = info.get('longBusinessSummary') or ''
            company_name = info.get('shortName') or info.get('longName') or ''
            sector = info.get('sector') or ''
            industry = info.get('industry') or ''
            # Tiingo is primary news source; fall back to yfinance/Google RSS
            if tiingo_client.available:
                tiingo_news = tiingo_client.fetch_ticker_news(ticker, max_age_days=30, max_items=12)
            else:
                tiingo_news = []
            if tiingo_news:
                ticker_news = tiingo_news
                news_sentiment = tiingo_client.fetch_ticker_sentiment(ticker, max_age_days=30, max_items=12)
            else:
                ticker_news = news_client.get_combined_news(ticker, sector, max_total=12)
                news_sentiment = None
            legal_data = sec_client.fetch_legal_filings(ticker, days_back=730)
            supply_data = supply_client.fetch_supply_chain(ticker)
            if not supply_data.get('available'):
                supply_data = sec_supply_client.fetch_supply_chain(ticker)
            finnhub_peers = supply_client.fetch_peers(ticker)

            # SEC EDGAR: XBRL cross-validation
            xbrl_validation = sec_xbrl_client.validate_against_yfinance(ticker, yf_data)
            # SEC EDGAR: long-duration revenue/earnings history
            edgar_history = sec_xbrl_client.fetch_historical_financials(ticker, min_years=10)
            # Evict the raw XBRL JSON blob (~1-10 MB) now that both
            # validate_against_yfinance and fetch_historical_financials are done.
            sec_xbrl_client._cache.pop(ticker, None)
            # SEC EDGAR: insider transactions from Form 4
            insider_data = sec_insider_client.fetch_insider_activity(ticker, days_back=365)

            officers = info.get('companyOfficers') or []
            ceo_officer = next(
                (o for o in officers
                 if 'ceo' in (o.get('title') or '').lower() or
                    'chief executive' in (o.get('title') or '').lower()),
                officers[0] if officers else None
            )
            ceo = ceo_officer.get('name') if ceo_officer else None
            # Build CEO biography from available yfinance officer data
            ceo_bio = None
            if ceo_officer:
                bio_parts = [ceo_officer.get('name', 'N/A'), ceo_officer.get('title', '')]
                age = ceo_officer.get('age')
                if age:
                    bio_parts.append(f"Age {age}")
                year_born = ceo_officer.get('yearBorn')
                if year_born and not age:
                    bio_parts.append(f"Born {year_born}")
                total_pay = ceo_officer.get('totalPay')
                if total_pay:
                    if total_pay >= 1e6:
                        pay_str = f"${total_pay/1e6:.1f}M"
                    else:
                        pay_str = f"${total_pay:,.0f}"
                    fy = ceo_officer.get('fiscalYear', '')
                    bio_parts.append(f"Compensation: {pay_str}" + (f" (FY{fy})" if fy else ""))
                ceo_bio = " | ".join(p for p in bio_parts if p)

            # Culture raw metrics: employees, CEO pay, comp risk, SBC
            _culture_raw = culture_client.extract(info, yf_data)
            _culture_gd = culture_client.fetch_glassdoor(company_name, ticker)

            # Step 2: Relative multiples
            multiples = compute_relative_multiples(yf_data)
            current_price = multiples.get('price')
            shares = multiples.get('shares')

            # Analyst consensus (Worksheet Step 8)
            analyst = compute_analyst_consensus(yf_data)

            # Step 5A: Forward DCF (sector-aware: Fixes C/D/E/F)
            dcf_fv, dcf_sens_range, fcf_growth, growth_diag, mc_result = run_forward_dcf(
                yf_data, wacc, sector=sector,
                exit_multiple=sector_exit_multiples.get(sector, effective_exit_mult_default),
                roic_data=roic_data,
                terminal_growth_adj=effective_tg_adj,
                wacc_sigma=_effective_wacc_sigma_ticker,
                tg_sigma=MC_TERMINAL_GROWTH_SIGMA,
                growth_sigma_mult=effective_growth_sigma_mult,
                growth_weight_shift=effective_growth_weight_shift)
            mos = (dcf_fv - current_price) / dcf_fv if (dcf_fv and current_price and dcf_fv > 0) else None

            # Step 5B: Dividend Discount Model (for dividend payers)
            try:
                div_series = yf_client.fetch_dividends(ticker)
            except Exception:
                div_series = pd.Series(dtype=float)
            ddm_result = run_ddm_valuation(
                yf_data, div_series, cost_of_equity,
                analyst_ltg=growth_diag.get('analyst_ltg'))

            # Step 3A/3B: Earnings quality
            eq = calculate_earnings_quality(yf_data)

            # Step 3B: Piotroski F
            piotroski = calculate_piotroski_f(yf_data)

            # Revenue CAGR (3Y from yfinance)
            rev_cagr = calculate_revenue_cagr(yf_data)

            # Extended CAGRs and derived metrics from EDGAR history
            rev_cagr_5y = None
            rev_cagr_10y = None
            fcf_cagr_5y = None
            fcf_cagr_10y = None
            gross_margin_avg_5y = None
            gross_margin_trend = None   # slope of gross margin over 5Y (positive = improving)
            dividend_cagr_5y = None
            shares_cagr_5y = None       # negative = buybacks shrinking share count

            if edgar_history:
                # Revenue CAGRs
                rev_hist = edgar_history.get('revenue_history', {})
                if rev_hist:
                    sy = sorted(rev_hist.keys())
                    newest_rev = rev_hist[sy[-1]] if sy else None
                    if newest_rev and newest_rev > 0:
                        if len(sy) >= 6:
                            yr5 = rev_hist.get(sy[-6])
                            if yr5 and yr5 > 0:
                                rev_cagr_5y = (newest_rev / yr5) ** (1 / 5) - 1
                        if len(sy) >= 11:
                            yr10 = rev_hist.get(sy[-11])
                            if yr10 and yr10 > 0:
                                rev_cagr_10y = (newest_rev / yr10) ** (1 / 10) - 1

                # FCF history: operating cash flow minus capex
                ocf_hist = edgar_history.get('operating_cf_history', {})
                cap_hist = edgar_history.get('capex_history', {})
                if ocf_hist:
                    common_years = sorted(set(ocf_hist) & set(cap_hist)) if cap_hist else sorted(ocf_hist)
                    fcf_hist = {yr: ocf_hist[yr] - abs(cap_hist.get(yr, 0)) for yr in common_years}
                    # EDGAR CapEx tags are reported as positive outflows — take abs() to be safe
                    fcf_vals = [fcf_hist[yr] for yr in sorted(fcf_hist) if fcf_hist[yr] is not None]
                    if len(fcf_vals) >= 6 and fcf_vals[-6] > 0 and fcf_vals[-1] > 0:
                        fcf_cagr_5y = (fcf_vals[-1] / fcf_vals[-6]) ** (1 / 5) - 1
                    if len(fcf_vals) >= 11 and fcf_vals[-11] > 0 and fcf_vals[-1] > 0:
                        fcf_cagr_10y = (fcf_vals[-1] / fcf_vals[-11]) ** (1 / 10) - 1

                # Gross margin history (gross profit / revenue)
                gp_hist = edgar_history.get('gross_profit_history', {})
                if gp_hist and rev_hist:
                    common_gy = sorted(set(gp_hist) & set(rev_hist))[-5:]  # last 5 years
                    margins = [gp_hist[yr] / rev_hist[yr] for yr in common_gy
                               if rev_hist.get(yr) and rev_hist[yr] > 0]
                    if len(margins) >= 3:
                        gross_margin_avg_5y = sum(margins) / len(margins)
                        # Simple linear trend: slope of last 5 margin observations
                        n = len(margins)
                        xs = list(range(n))
                        x_mean = sum(xs) / n
                        y_mean = sum(margins) / n
                        denom = sum((x - x_mean) ** 2 for x in xs)
                        if denom:
                            gross_margin_trend = sum(
                                (xs[i] - x_mean) * (margins[i] - y_mean) for i in range(n)
                            ) / denom

                # Dividend growth (5Y CAGR of dividends paid)
                div_hist = edgar_history.get('dividends_paid_history', {})
                if div_hist:
                    dy = sorted(div_hist.keys())
                    if len(dy) >= 6:
                        d0, d1 = abs(div_hist.get(dy[-6], 0) or 0), abs(div_hist.get(dy[-1], 0) or 0)
                        if d0 > 0 and d1 > 0:
                            dividend_cagr_5y = (d1 / d0) ** (1 / 5) - 1

                # Share count trend (5Y CAGR — negative means shrinking = buybacks)
                sh_hist = edgar_history.get('shares_history', {})
                if sh_hist:
                    shy = sorted(sh_hist.keys())
                    if len(shy) >= 6:
                        s0, s1 = sh_hist.get(shy[-6]), sh_hist.get(shy[-1])
                        if s0 and s0 > 0 and s1 and s1 > 0:
                            shares_cagr_5y = (s1 / s0) ** (1 / 5) - 1

            # Step 3C: Balance sheet health
            int_cov = calculate_interest_coverage(yf_data)
            nd_ebitda = calculate_net_debt_ebitda(yf_data)

            # Traditional ratios
            ratios = compute_ratios(yf_data)

            # Free cash flow — use annual cash flow statement (same source as DCF)
            cf = yf_data.get('cash_flow')
            fcf = None
            if cf is not None and not cf.empty and 'Free Cash Flow' in cf.index:
                fcf_vals = cf.loc['Free Cash Flow'].dropna().sort_index()
                if len(fcf_vals) > 0:
                    fcf = fcf_vals.iloc[-1]  # most recent annual

            # --- NEW MODELS ---

            # Altman Z-Score (exists in comparisons, now wired)
            altman_z = calculate_altman_z(yf_data)
            altman_z_zone = None
            if altman_z is not None:
                if altman_z > 2.99:
                    altman_z_zone = 'safe'
                elif altman_z >= 1.81:
                    altman_z_zone = 'grey'
                else:
                    altman_z_zone = 'distress'

            # Beneish M-Score
            beneish = calculate_beneish_m(yf_data)

            # DuPont Decomposition
            dupont = compute_dupont(yf_data)

            # EPV (Earnings Power Value)
            inc_stmt = yf_data.get('income_statement')
            _epv_ebit = None
            _epv_eff_tax = 0.21
            if inc_stmt is not None and not inc_stmt.empty:
                _latest_inc = inc_stmt.iloc[:, 0]
                _epv_ebit = _latest_inc.get('Operating Income')
                if pd.notna(_epv_ebit) and _epv_ebit is not None:
                    _tax_prov = _latest_inc.get('Tax Provision')
                    _pretax = _latest_inc.get('Pretax Income')
                    if (pd.notna(_tax_prov) and pd.notna(_pretax) and
                            _pretax and _pretax != 0):
                        _epv_eff_tax = max(0, min(float(_tax_prov) / float(_pretax), 0.50))
                else:
                    _epv_ebit = None

            _epv_excess_cash = 0
            bs = yf_data.get('balance_sheet')
            if bs is not None and not bs.empty:
                _cash_val = bs.iloc[:, 0].get('Cash And Cash Equivalents')
                if pd.notna(_cash_val) and _cash_val is not None:
                    _epv_excess_cash = float(_cash_val)

            epv_fv = earnings_power_value(
                float(_epv_ebit) if _epv_ebit is not None else None,
                _epv_eff_tax, wacc, shares,
                _epv_excess_cash)
            epv_growth_fv = epv_with_growth_premium(
                epv_fv, ratios.get('ROE'), cost_of_equity)

            # RIM (Residual Income Model)
            _book_value = info.get('bookValue')
            if _book_value is None and shares and shares > 0:
                if bs is not None and not bs.empty:
                    _eq_val = bs.iloc[:, 0].get('Stockholders Equity')
                    if pd.notna(_eq_val) and _eq_val:
                        _book_value = float(_eq_val) / shares
            rim_fv = residual_income_model(
                _book_value, ratios.get('ROE'), cost_of_equity)

            # Reverse DCF (solve for implied growth)
            rev_dcf = None
            if dcf_fv and current_price and current_price > 0 and fcf and shares:
                net_debt_val = get_net_debt(yf_data)
                rev_dcf = reverse_dcf(current_price, fcf, wacc, shares, net_debt_val)

            # 52-Week Range
            high_52w = info.get('fiftyTwoWeekHigh')
            low_52w = info.get('fiftyTwoWeekLow')
            pct_from_52w_high = ((current_price - high_52w) / high_52w
                                 if (current_price and high_52w and high_52w > 0) else None)
            pct_from_52w_low = ((current_price - low_52w) / low_52w
                                if (current_price and low_52w and low_52w > 0) else None)
            range_52w_position = ((current_price - low_52w) / (high_52w - low_52w) * 100
                                  if (current_price and high_52w and low_52w
                                      and high_52w > low_52w) else None)

            # Founder-led detection: check if CEO title contains 'founder'
            founder_led = False
            if ceo_officer:
                title = (ceo_officer.get('title') or '').lower()
                founder_led = 'founder' in title

            # Ownership data from yfinance info
            shares_out = info.get('sharesOutstanding')
            float_shares = info.get('floatShares')
            insider_pct = info.get('heldPercentInsiders')
            inst_pct = info.get('heldPercentInstitutions')
            shares_short = info.get('sharesShort')
            short_ratio = info.get('shortRatio')
            short_pct_float = info.get('shortPercentOfFloat')

            # Share turnover rate = avg daily volume / shares outstanding
            avg_vol = info.get('averageVolume')
            share_turnover_rate = None
            if avg_vol and shares_out and shares_out > 0:
                share_turnover_rate = avg_vol / shares_out

            # Dividend yield and payout ratio (for narrative & template)
            _div_rate = info.get('dividendRate')
            _div_price = info.get('currentPrice') or info.get('regularMarketPrice')
            div_yield = (_div_rate / _div_price
                         if (_div_rate and _div_price and _div_price > 0) else None)
            payout_ratio = info.get('payoutRatio')

            # Balance sheet risk flags (goodwill, R&D, SGA)
            goodwill_pct = None
            rd_intensity = None
            sga_pct_rev = None
            sga_yoy_change = None
            inc_stmt = yf_data.get('income_statement')
            if bs is not None and not bs.empty:
                total_assets = bs.iloc[:, 0].get('Total Assets')
                gw = bs.iloc[:, 0].get('Goodwill')
                if (pd.notna(gw) and gw and pd.notna(total_assets)
                        and total_assets and total_assets > 0):
                    goodwill_pct = float(gw) / float(total_assets)
            if inc_stmt is not None and not inc_stmt.empty:
                _rev_latest = inc_stmt.iloc[:, 0].get('Total Revenue')
                _rd_latest = inc_stmt.iloc[:, 0].get('Research And Development')
                _sga_latest = inc_stmt.iloc[:, 0].get('Selling General And Administration')
                if (pd.notna(_rd_latest) and _rd_latest
                        and pd.notna(_rev_latest) and _rev_latest and _rev_latest > 0):
                    rd_intensity = float(_rd_latest) / float(_rev_latest)
                if (pd.notna(_sga_latest) and _sga_latest
                        and pd.notna(_rev_latest) and _rev_latest and _rev_latest > 0):
                    sga_pct_rev = float(_sga_latest) / float(_rev_latest)
                # SGA YoY change (compare most recent two years)
                if inc_stmt.shape[1] >= 2:
                    _sga_prior = inc_stmt.iloc[:, 1].get('Selling General And Administration')
                    if (pd.notna(_sga_latest) and _sga_latest
                            and pd.notna(_sga_prior) and _sga_prior and _sga_prior > 0):
                        sga_yoy_change = (float(_sga_latest) / float(_sga_prior)) - 1

            # Morningstar: fair value and difference vs model
            ms_diff = None
            ms_fv = None
            ms_pfv = ms_pfv_data.get(ticker)
            if ms_pfv and current_price and dcf_fv:
                ms_fv = current_price / ms_pfv
                if ms_fv > 0:
                    ms_diff = (dcf_fv / ms_fv) - 1

            row = {
                'ticker': ticker,
                'source_group': ticker_source.get(ticker, 'quality'),
                # Company info (Step 3)
                'description': description,
                'company_name': company_name,
                'sector': sector,
                'industry': industry,
                'ceo': ceo,
                'ceo_bio': ceo_bio,
                'founder_led': founder_led,
                # Culture raw inputs (narrative built in post-processing)
                'employees': _culture_raw.get('employees'),
                'ceo_total_pay': _culture_raw.get('ceo_total_pay'),
                'compensation_risk': _culture_raw.get('compensation_risk'),
                'sbc': _culture_raw.get('sbc'),
                'glassdoor_rating': _culture_gd.get('glassdoor_rating'),
                'glassdoor_ceo_pct': _culture_gd.get('glassdoor_ceo_pct'),
                'glassdoor_rec_pct': _culture_gd.get('glassdoor_rec_pct'),
                'fcf': fcf,
                # Ownership
                'shares_out': shares_out,
                'float_shares': float_shares,
                'insider_pct': insider_pct,
                'inst_pct': inst_pct,
                'shares_short': shares_short,
                'short_ratio': short_ratio,
                'short_pct_float': short_pct_float,
                'share_turnover_rate': share_turnover_rate,
                'share_buyback_rate': share_buyback_rate,
                # Insider activity (Form 4)
                'insider_buy_ratio': insider_data.get('insider_buy_ratio') if insider_data and insider_data.get('available') else None,
                'insider_buy_count_90d': insider_data.get('buy_count_90d') if insider_data and insider_data.get('available') else None,
                'insider_sell_count_90d': insider_data.get('sell_count_90d') if insider_data and insider_data.get('available') else None,
                'insider_buy_count_365d': insider_data.get('buy_count_365d') if insider_data and insider_data.get('available') else None,
                'insider_sell_count_365d': insider_data.get('sell_count_365d') if insider_data and insider_data.get('available') else None,
                'insider_net_shares': insider_data.get('net_shares_365d') if insider_data and insider_data.get('available') else None,
                'insider_net_value': insider_data.get('net_value_365d') if insider_data and insider_data.get('available') else None,
                'insider_transactions': (insider_data.get('transactions', [])[:10] if insider_data and insider_data.get('available') else []),
                'roic_by_year': roic_data.get('roic_by_year'),
                'roic_cv': roic_cv,
                'gross_margin': gross_margin,
                'shareholder_yield': shareholder_yield,
                'div_yield': div_yield,
                'payout_ratio': payout_ratio,
                # Core screen
                'roic': roic_data['avg_roic'],
                'wacc': wacc,
                'spread': roic_data['avg_roic'] - wacc,
                'mcap': multiples.get('market_cap'),
                # Latest financials (for profit pool analysis)
                'revenue': latest_fins.get('revenue'),
                'operating_income': latest_fins.get('operating_income'),
                'net_income': latest_fins.get('net_income'),
                'operating_margin': (latest_fins['operating_income'] / latest_fins['revenue']
                    if latest_fins.get('operating_income') is not None and latest_fins.get('revenue') and latest_fins['revenue'] > 0
                    else None),
                'er': cost_of_equity,
                're_method': cached['re_method'],
                # Beta diagnostics (Step 4A)
                'beta_raw': beta_diag.get('raw_beta') if beta_diag else None,
                'beta_adjusted': beta_diag.get('adjusted_beta') if beta_diag else None,
                'beta_r2': beta_diag.get('r_squared') if beta_diag else None,
                'beta_se': beta_diag.get('se_beta') if beta_diag else None,
                'beta_n_obs': beta_diag.get('n_observations') if beta_diag else None,
                'beta_r2_class': beta_diag.get('r2_classification') if beta_diag else None,
                # Valuation (Step 5)
                'dcf_fv': dcf_fv,
                'price': current_price,
                'mos': mos,
                'dcf_sens_range': dcf_sens_range,
                'fcf_growth': fcf_growth,
                'analyst_ltg': growth_diag.get('analyst_ltg'),
                'margin_trend': growth_diag.get('margin_trend'),
                'surprise_avg': growth_diag.get('surprise_avg'),
                'fundamental_growth': growth_diag.get('fundamental_growth'),
                'reinvestment_rate': growth_diag.get('reinvestment_rate'),
                'terminal_growth': _get_sector_config(sector)['terminal_growth'],
                'exit_mult_fv': growth_diag.get('exit_mult_fv'),
                'tv_method_spread': growth_diag.get('tv_method_spread'),
                'mc_p10_fv': mc_result['p10_fv'] if mc_result else None,
                'mc_p90_fv': mc_result['p90_fv'] if mc_result else None,
                'mc_cv': mc_result['cv'] if mc_result else None,
                'mc_confidence': _mc_confidence_label(mc_result['cv']) if mc_result and mc_result.get('cv') is not None else None,
                'ms_diff': ms_diff,
                'ms_fv': ms_fv,
                'ms_pfv': ms_pfv,
                # Multiples (Step 2)
                'pe': multiples.get('pe'),
                'ev_ebitda': multiples.get('ev_ebitda'),
                'enterprise_value': multiples.get('enterprise_value'),
                'pfcf': multiples.get('pfcf'),
                'pb': multiples.get('pb'),
                # Analyst consensus (Step 8)
                'analyst_rec': analyst.get('rec_key', '').upper() if analyst.get('rec_key') else None,
                'num_analysts': analyst.get('num_analysts'),
                'target_mean': analyst.get('target_mean'),
                'target_high': analyst.get('target_high'),
                'target_low': analyst.get('target_low'),
                # Quality (Step 3B)
                'piotroski': piotroski,
                'cash_conv': eq.get('cash_conversion'),
                'accruals': eq.get('accruals_ratio'),
                'rev_cagr': rev_cagr,
                'rev_cagr_5y': rev_cagr_5y,
                'rev_cagr_10y': rev_cagr_10y,
                'fcf_cagr_5y': fcf_cagr_5y,
                'fcf_cagr_10y': fcf_cagr_10y,
                'gross_margin_avg_5y': gross_margin_avg_5y,
                'gross_margin_trend': gross_margin_trend,
                'dividend_cagr_5y': dividend_cagr_5y,
                'shares_cagr_5y': shares_cagr_5y,
                # EDGAR XBRL validation
                'edgar_quality_score': xbrl_validation.get('edgar_quality_score') if xbrl_validation else None,
                'edgar_fields_flagged': xbrl_validation.get('fields_flagged', 0) if xbrl_validation else 0,
                'edgar_discrepancies': xbrl_validation.get('discrepancies', []) if xbrl_validation else [],
                'edgar_history': edgar_history,
                # Balance sheet (Step 3C)
                'int_cov': int_cov,
                'nd_ebitda': nd_ebitda,
                # Traditional ratios
                'roe': ratios.get('ROE'),
                'de': ratios.get('Debt-to-Equity'),
                'cr': ratios.get('Current Ratio'),
                'roa': ratios.get('ROA'),
                # Macro overlay
                'macro_regime': macro_regime_result['regime'] if macro_regime_result else None,
                'macro_composite': macro_regime_result['composite_score'] if macro_regime_result else None,
                'macro_erp': effective_erp,
                'sector_headwinds': sector_signals.get(sector, {}).get('headwinds', []),
                'sector_tailwinds': sector_signals.get(sector, {}).get('tailwinds', []),
                'news_headlines': ticker_news,
                'news_sentiment': news_sentiment,
                'legal_filings': legal_data.get('filings', []),
                'legal_count': legal_data.get('count', 0),
                'legal_latest': legal_data.get('latest_date'),
                'suppliers': supply_data.get('suppliers', []),
                'customers': supply_data.get('customers', []),
                'supply_chain_available': supply_data.get('available', False),
                'finnhub_peers': finnhub_peers,
                # DDM (Dividend Discount Model)
                'ddm_eligible': ddm_result.get('ddm_eligible', False),
                'ddm_reason': ddm_result.get('ddm_reason'),
                'ddm_fv': ddm_result.get('ddm_fv'),
                'ddm_h_fv': ddm_result.get('ddm_h_fv'),
                'ddm_growth': ddm_result.get('ddm_growth'),
                'ddm_div_cagr': ddm_result.get('ddm_div_cagr'),
                'ddm_sustainable_growth': ddm_result.get('ddm_sustainable_growth'),
                'ddm_payout_flag': ddm_result.get('ddm_payout_flag', False),
                'ddm_consecutive_years': ddm_result.get('ddm_consecutive_years'),
                'ddm_mc_median': ddm_result.get('ddm_mc_median'),
                'ddm_mc_p10': ddm_result.get('ddm_mc_p10'),
                'ddm_mc_p90': ddm_result.get('ddm_mc_p90'),
                'ddm_mc_cv': ddm_result.get('ddm_mc_cv'),
                # Reverse DCF
                'implied_growth': rev_dcf['implied_growth'] if rev_dcf and rev_dcf.get('converged') else None,
                'implied_vs_estimated': ((rev_dcf['implied_growth'] - fcf_growth)
                    if rev_dcf and rev_dcf.get('converged') and fcf_growth else None),
                # EPV (Earnings Power Value)
                'epv_fv': epv_fv,
                'epv_pfv': (current_price / epv_fv
                    if (epv_fv and current_price and epv_fv > 0) else None),
                'epv_mos': ((epv_fv - current_price) / epv_fv
                    if (epv_fv and current_price and epv_fv > 0) else None),
                'epv_growth_fv': epv_growth_fv,
                # RIM (Residual Income Model)
                'rim_fv': rim_fv,
                'rim_mos': ((rim_fv - current_price) / rim_fv
                    if (rim_fv and current_price and rim_fv > 0) else None),
                # Altman Z-Score
                'altman_z': altman_z,
                'altman_z_zone': altman_z_zone,
                # Beneish M-Score
                'beneish_m': beneish['m_score'] if beneish else None,
                'beneish_flag': beneish['manipulation_flag'] if beneish else None,
                # DuPont Decomposition
                'dupont_margin': dupont['margin'] if dupont else None,
                'dupont_turnover': dupont['turnover'] if dupont else None,
                'dupont_leverage': dupont['leverage'] if dupont else None,
                # 52-Week Range
                'high_52w': high_52w,
                'low_52w': low_52w,
                'pct_from_52w_high': pct_from_52w_high,
                'pct_from_52w_low': pct_from_52w_low,
                'range_52w_position': range_52w_position,
                # Balance sheet risk flags
                'goodwill_pct': goodwill_pct,
                'rd_intensity': rd_intensity,
                'sga_pct_rev': sga_pct_rev,
                'sga_yoy_change': sga_yoy_change,
                # --- Price-history signals ---
                'momentum_12_1':   round(_ticker_momentum, 4) if _ticker_momentum is not None else None,
                'realized_vol':    round(_ticker_realized_vol, 4) if _ticker_realized_vol is not None else None,
                'drawdown_2008':   round(_ticker_dd_2008, 4) if _ticker_dd_2008 is not None else None,
                'drawdown_2020':   round(_ticker_dd_2020, 4) if _ticker_dd_2020 is not None else None,
                'drawdown_2022':   round(_ticker_dd_2022, 4) if _ticker_dd_2022 is not None else None,
                'rolling_betas':   _rolling_beta_diag if _rolling_beta_diag else None,
            }
            # Composite rating (Worksheet Decision Matrix)
            row['rating'] = compute_rating(row)
            results.append(row)
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")
        finally:
            # Drop the heavy yf_data reference now that this ticker is done
            if ticker in screen_cache:
                screen_cache[ticker].pop('yf_data', None)
            sys.stdout.flush()

    # All per-ticker analysis complete — release remaining caches
    screen_cache.clear()
    yf_client.evict_financials()
    gc.collect()

    # -----------------------------------------------------------------------
    # Post-processing: sector-median EV/EBITDA comparison + DCF cross-check
    # -----------------------------------------------------------------------

    # 1. Compute sector median EV/EBITDA
    _sector_ee = {}
    for r in results:
        s, ee = r.get('sector'), r.get('ev_ebitda')
        if s and ee and 0 < ee < EV_EBITDA_OUTLIER_MAX:  # filter outliers
            _sector_ee.setdefault(s, []).append(ee)
    sector_median_ee = {s: _median(v) for s, v in _sector_ee.items() if len(v) >= MIN_SECTOR_STOCKS}

    for r in results:
        s = r.get('sector')
        ee = r.get('ev_ebitda')
        med = sector_median_ee.get(s)
        r['_sector_median_ee'] = med
        r['_ee_vs_sector'] = (ee / med - 1) if ee and med and med > 0 else None

        # 2. DCF cross-check: compute multiples-implied fair value from sector median
        # If DCF FV > 1.5× multiples FV, blend toward multiples (40% weight)
        # This reins in over-estimates where DCF extrapolates peak FCF
        ev_raw = r.get('enterprise_value')
        ev_eb = r.get('ev_ebitda')
        dcf_fv = r.get('dcf_fv')
        price = r.get('price')
        shares = r.get('shares_out')

        if ev_raw and ev_eb and ev_eb > 0 and med and shares and shares > 0 and dcf_fv:
            ebitda = ev_raw / ev_eb
            multiples_ev = med * ebitda
            net_debt = ev_raw - (price * shares if price else 0)
            multiples_fv = (multiples_ev - net_debt) / shares
            if multiples_fv > 0:
                r['_multiples_fv'] = multiples_fv
                # Blend when DCF is significantly above multiples-implied value
                # BUT only when DCF > price (model sees upside → possible over-estimate).
                # If DCF < price the model already under-values; pulling down worsens it.
                if dcf_fv > multiples_fv * BLEND_TRIGGER and price and dcf_fv > price:
                    blended = BLEND_DCF_WEIGHT * dcf_fv + BLEND_MULT_WEIGHT * multiples_fv
                    blend_ratio = blended / dcf_fv if dcf_fv > 0 else 1.0
                    r['dcf_fv'] = blended
                    # Recompute dependent fields
                    if blended > 0:
                        r['mos'] = (blended - price) / blended
                    # Scale sensitivity range proportionally to keep Bear/Bull consistent
                    sens = r.get('dcf_sens_range')
                    if sens and len(sens) == 2:
                        r['dcf_sens_range'] = (sens[0] * blend_ratio, sens[1] * blend_ratio)
                    r['_blended'] = True
                else:
                    r['_blended'] = False
            else:
                r['_multiples_fv'] = None
                r['_blended'] = False
        else:
            r['_multiples_fv'] = None
            r['_blended'] = False

        # Recompute Morningstar fields after potential blending
        ms_pfv_val = ms_pfv_data.get(r['ticker'])
        if ms_pfv_val and price and r.get('dcf_fv'):
            ms_fv = price / ms_pfv_val
            if ms_fv > 0:
                r['ms_fv'] = ms_fv
                r['ms_pfv'] = ms_pfv_val
                r['ms_diff'] = (r['dcf_fv'] / ms_fv) - 1

    # DDM blending: for eligible stocks, blend 70% DCF + 30% DDM
    for r in results:
        r['_blended_method'] = 'DCF'
        if r.get('ddm_eligible') and r.get('ddm_fv') and r.get('dcf_fv'):
            ddm_fv = r['ddm_fv']
            dcf_fv = r['dcf_fv']
            if ddm_fv > 0 and dcf_fv > 0:
                blended = DCF_BLEND_WEIGHT_WITH_DDM * dcf_fv + DDM_BLEND_WEIGHT * ddm_fv
                r['dcf_fv'] = blended
                r['_blended_method'] = 'DCF+DDM'
                # Recompute MoS
                price = r.get('price')
                if price and blended > 0:
                    r['mos'] = (blended - price) / blended
                # Flag divergence
                avg_fv = (dcf_fv + ddm_fv) / 2.0
                divergence = abs(dcf_fv - ddm_fv) / avg_fv if avg_fv > 0 else 0
                if divergence > DDM_DIVERGENCE_THRESHOLD:
                    r['_ddm_low_confidence'] = True
                else:
                    r['_ddm_low_confidence'] = False
            else:
                r['_ddm_low_confidence'] = False
        else:
            r['_ddm_low_confidence'] = False

    # Recompute ratings after blending
    for r in results:
        r['rating'] = compute_rating(r)

    # -----------------------------------------------------------------------
    # Profit pool analysis (sector-level revenue/profit concentration)
    # Must run BEFORE screening matrix so pp_multiple is available for gates
    # -----------------------------------------------------------------------
    # 1. Aggregate sector totals
    _sector_rev = {}     # sector → total revenue
    _sector_opinc = {}   # sector → total operating income (clamped ≥0)
    _sector_tickers = {} # sector → [(ticker, revenue, operating_income)]
    for r in results:
        s = r.get('sector')
        rev = r.get('revenue')
        opinc = r.get('operating_income')
        if s and rev and rev > 0:
            _sector_rev[s] = _sector_rev.get(s, 0) + rev
            if opinc is not None:
                _sector_opinc[s] = _sector_opinc.get(s, 0) + max(opinc, 0)
            _sector_tickers.setdefault(s, []).append((r['ticker'], rev, opinc or 0))

    # 2. Sector-level operating margin median
    _sector_opm = {}
    for r in results:
        s = r.get('sector')
        opm = r.get('operating_margin')
        if s and opm is not None:
            _sector_opm.setdefault(s, []).append(opm)
    sector_median_opm = {s: _median(v) for s, v in _sector_opm.items()
                         if len(v) >= MIN_SECTOR_STOCKS}

    # 3. Per-ticker profit pool metrics
    for r in results:
        s = r.get('sector')
        rev = r.get('revenue')
        opinc = r.get('operating_income')

        # Revenue share (fraction of sector total revenue in analysis universe)
        sec_rev = _sector_rev.get(s, 0)
        r['pp_revenue_share'] = (rev / sec_rev) if (rev and sec_rev > 0) else None

        # Profit share (fraction of sector total operating income)
        sec_opinc = _sector_opinc.get(s, 0)
        r['pp_profit_share'] = (max(opinc, 0) / sec_opinc
                                if (opinc is not None and sec_opinc > 0) else None)

        # Profit pool multiple = profit_share / revenue_share
        # > 1 means disproportionate profit capture; < 1 means under-earning
        rs = r.get('pp_revenue_share')
        ps = r.get('pp_profit_share')
        r['pp_multiple'] = (ps / rs) if (ps and rs and rs > 0) else None

        # Margin advantage vs sector median operating margin
        opm = r.get('operating_margin')
        med_opm = sector_median_opm.get(s)
        r['pp_margin_advantage'] = ((opm - med_opm)
                                    if (opm is not None and med_opm is not None) else None)
        r['_sector_median_opm'] = med_opm

        # Sector-level concentration metrics (same for all tickers in sector)
        tickers_in_sector = _sector_tickers.get(s, [])
        if len(tickers_in_sector) >= 3 and sec_rev > 0:
            shares = [(t_rev / sec_rev) for _, t_rev, _ in tickers_in_sector]
            r['pp_sector_hhi'] = round(sum(sh ** 2 for sh in shares), 4)
            top4 = sorted(shares, reverse=True)[:4]
            r['pp_sector_cr4'] = round(sum(top4), 4)
            r['pp_sector_count'] = len(tickers_in_sector)
        else:
            r['pp_sector_hhi'] = None
            r['pp_sector_cr4'] = None
            r['pp_sector_count'] = len(tickers_in_sector) if tickers_in_sector else 0

    # Apply screening matrix (override ratings that fail critical gates)
    apply_screening_matrix(results)

    # Pre-compute _price_fv so continuous scoring can use it
    for r in results:
        p, fv = r.get('price'), r.get('dcf_fv')
        r['_price_fv'] = p / fv if p and fv else None

    compute_continuous_scores(results)
    apply_composite_rating_override(results)

    # Position sizing and concentration analysis
    weights = position_sizes(results)
    for r in results:
        r['position_weight'] = weights.get(r['ticker'])
    concentration = concentration_analysis(
        [r for r in results if r.get('rating') in ('BUY', 'LEAN BUY')])
    if concentration.get('concentration_flag'):
        print(f"\n  Portfolio concentration warning: {concentration['top_sector']} "
              f"= {concentration['top_sector_weight']:.0%} "
              f"(HHI={concentration['hhi']:.2f})")

    # -----------------------------------------------------------------------
    # Peer percentile ranking (sector-relative position for key metrics)
    # -----------------------------------------------------------------------
    _peer_metrics = ['roic', 'gross_margin', 'rev_cagr', 'nd_ebitda', 'piotroski',
                     'rd_intensity', 'goodwill_pct', 'operating_margin', 'pp_multiple']
    _peer_buckets = {}  # metric → sector → sorted list of values
    for metric in _peer_metrics:
        _peer_buckets[metric] = {}
        for r in results:
            s = r.get('sector')
            v = r.get(metric)
            if s and v is not None:
                _peer_buckets[metric].setdefault(s, []).append(v)
        for s in _peer_buckets[metric]:
            _peer_buckets[metric][s].sort()

    for r in results:
        s = r.get('sector')
        for metric in _peer_metrics:
            vals = _peer_buckets[metric].get(s, [])
            v = r.get(metric)
            if v is not None and len(vals) >= 3:
                # Percentile: fraction of peers this value exceeds
                below = sum(1 for x in vals if x < v)
                pctile = below / len(vals)
                r[f'_peer_pctile_{metric}'] = round(pctile, 2)
            else:
                r[f'_peer_pctile_{metric}'] = None

    # -----------------------------------------------------------------------
    # Culture narrative: workforce productivity, pay, ownership culture
    # -----------------------------------------------------------------------
    # Step 1 — derive per-employee metrics
    for r in results:
        emp     = r.get('employees')
        rev     = r.get('revenue')
        fcf_val = r.get('fcf')
        ceo_pay = r.get('ceo_total_pay')
        sbc     = r.get('sbc')

        rpe = (rev / emp) if (emp and rev and rev > 0) else None
        r['revenue_per_emp'] = rpe
        r['fcf_per_emp']  = (fcf_val / emp) if (emp and fcf_val is not None) else None
        r['ceo_pay_ratio'] = (ceo_pay / rpe) if (ceo_pay and rpe and rpe > 0) else None
        r['sbc_per_emp']   = (sbc / emp) if (sbc and emp) else None

    # Step 2 — sector-percentile buckets for revenue per employee
    _cult_sector_rpe: dict = {}
    for r in results:
        s, rpe = r.get('sector'), r.get('revenue_per_emp')
        if s and rpe and rpe > 0:
            _cult_sector_rpe.setdefault(s, []).append(rpe)
    for s in _cult_sector_rpe:
        _cult_sector_rpe[s].sort()

    # Step 3 — multi-year RPE trend from EDGAR revenue history
    for r in results:
        emp = r.get('employees')
        if not emp:
            r['rpe_cagr'] = None
            continue
        rev_hist = (r.get('edgar_history') or {}).get('revenue_history') or {}
        years = sorted(rev_hist.keys())
        if len(years) >= 3:
            earliest = rev_hist[years[0]]
            latest   = rev_hist[years[-1]]
            n_years  = years[-1] - years[0]
            if earliest and latest and earliest > 0 and n_years > 0:
                rpe_earliest = earliest / emp
                rpe_latest   = latest   / emp
                r['rpe_cagr'] = (rpe_latest / rpe_earliest) ** (1 / n_years) - 1
            else:
                r['rpe_cagr'] = None
        else:
            r['rpe_cagr'] = None

    # Step 4 — employment-related legal flag
    _EMPLOYMENT_KEYWORDS = {
        'labor', 'labour', 'employee', 'employment', 'wage', 'salary',
        'discrimination', 'wrongful termination', 'class action', 'nlrb',
        'union', 'strike', 'layoff', 'hostile work',
    }
    for r in results:
        filings = r.get('legal_filings') or []
        flag = False
        for f in filings:
            text = ' '.join([
                (f.get('description') or ''),
                (f.get('summary') or ''),
            ]).lower()
            if any(kw in text for kw in _EMPLOYMENT_KEYWORDS):
                flag = True
                break
        r['employment_legal_flag'] = flag

    # Step 5 — layoff / culture news signal
    _LAYOFF_KEYWORDS = {
        'layoff', 'lay off', 'laid off', 'job cut', 'workforce reduction',
        'redundan', 'downsiz', 'restructur', 'reorg',
    }
    _CULTURE_POS_KEYWORDS = {
        'best place', 'top employer', 'great place to work',
        'best company', 'culture award',
    }
    for r in results:
        headlines = r.get('news_headlines') or []
        layoff_signal = False
        culture_award = False
        for h in headlines:
            text = (h.get('title') or '').lower()
            if any(kw in text for kw in _LAYOFF_KEYWORDS):
                layoff_signal = True
            if any(kw in text for kw in _CULTURE_POS_KEYWORDS):
                culture_award = True
        r['layoff_news_signal'] = layoff_signal
        r['culture_award_signal'] = culture_award

    # Step 6 — plain-English narrative
    def _fmt_emp(n):
        if n >= 1_000_000: return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:     return f"{n // 1_000:,}K"
        return str(n)

    def _fmt_money(v):
        if abs(v) >= 1_000_000: return f"${v / 1_000_000:.1f}M"
        if abs(v) >= 1_000:     return f"${v / 1_000:.0f}K"
        return f"${v:.0f}"

    for r in results:
        s         = r.get('sector') or 'its sector'
        emp       = r.get('employees')
        rpe       = r.get('revenue_per_emp')
        rpe_cagr  = r.get('rpe_cagr')
        fcf_pe    = r.get('fcf_per_emp')
        ceo_ratio = r.get('ceo_pay_ratio')
        sbc_pe    = r.get('sbc_per_emp')
        rd        = r.get('rd_intensity')
        crisk     = r.get('compensation_risk')
        gd_rating = r.get('glassdoor_rating')
        gd_ceo    = r.get('glassdoor_ceo_pct')
        gd_rec    = r.get('glassdoor_rec_pct')
        emp_legal = r.get('employment_legal_flag', False)
        layoff    = r.get('layoff_news_signal', False)
        cult_award = r.get('culture_award_signal', False)

        sector_rpes = _cult_sector_rpe.get(r.get('sector'), [])
        rpe_pct = None
        if rpe and len(sector_rpes) >= 3:
            rpe_pct = sum(1 for x in sector_rpes if x < rpe) / len(sector_rpes)

        sentences = []

        # --- Glassdoor (highest credibility — leads if available) ---------
        if gd_rating is not None:
            stars = f"{gd_rating:.1f}/5"
            rec_str = f", with {gd_rec}% of employees recommending it to a friend" if gd_rec else ""
            ceo_str = f" and {gd_ceo}% CEO approval" if gd_ceo else ""
            sentences.append(
                f"Glassdoor-rated {stars}{rec_str}{ceo_str}."
            )

        # --- Workforce size -----------------------------------------------
        if emp:
            sentences.append(f"Employs approximately {_fmt_emp(emp)} people.")

        # --- Revenue per employee with trend -----------------------------
        if rpe and rpe_pct is not None:
            if rpe_pct >= 0.75:   pct_desc = "top quartile"
            elif rpe_pct >= 0.50: pct_desc = "above the sector median"
            elif rpe_pct >= 0.25: pct_desc = "below the sector median"
            else:                 pct_desc = "bottom quartile"
            trend_str = ""
            if rpe_cagr is not None:
                if rpe_cagr > 0.05:
                    trend_str = f", improving at {rpe_cagr:.0%}/yr — growing workforce leverage"
                elif rpe_cagr < -0.05:
                    trend_str = f", declining at {abs(rpe_cagr):.0%}/yr — weakening productivity"
            sentences.append(
                f"Revenue per employee of {_fmt_money(rpe)} ranks in the "
                f"{pct_desc} among {s} peers{trend_str}."
            )
        elif rpe:
            sentences.append(f"Revenue per employee is {_fmt_money(rpe)}.")

        # --- FCF per employee --------------------------------------------
        if fcf_pe and fcf_pe > 0 and emp:
            sentences.append(
                f"Each employee generates {_fmt_money(fcf_pe)} of free cash flow annually."
            )

        # --- SBC per employee (ownership culture) ------------------------
        if sbc_pe and sbc_pe > 0:
            sentences.append(
                f"Stock-based compensation of {_fmt_money(sbc_pe)} per employee "
                f"reflects an ownership culture."
            )

        # --- CEO pay alignment -------------------------------------------
        if ceo_ratio is not None:
            if ceo_ratio <= 10:   ratio_desc = "modest"
            elif ceo_ratio <= 30: ratio_desc = "reasonable"
            elif ceo_ratio <= 75: ratio_desc = "elevated"
            elif ceo_ratio <= 150: ratio_desc = "high"
            else:                  ratio_desc = "very high"
            sentences.append(
                f"CEO compensation is {ceo_ratio:.0f}\u00d7 revenue per employee "
                f"({ratio_desc} relative to workforce productivity)."
            )

        # --- yfinance compensation risk ----------------------------------
        if crisk is not None:
            if crisk <= 3:
                crisk_desc = "low compensation governance risk"
            elif crisk <= 6:
                crisk_desc = "moderate compensation governance risk"
            else:
                crisk_desc = "elevated compensation governance risk"
            sentences.append(
                f"Governance score flags {crisk_desc} ({crisk}/10)."
            )

        # --- R&D intensity -----------------------------------------------
        if rd and rd > 0.01:
            if rd >= 0.20:   rd_desc = "heavy"
            elif rd >= 0.10: rd_desc = "significant"
            elif rd >= 0.05: rd_desc = "moderate"
            else:            rd_desc = "limited"
            sentences.append(
                f"R&D investment of {rd:.0%} of revenue ({rd_desc}) signals "
                f"commitment to product development and talent."
            )

        # --- Contradiction detection -------------------------------------
        if rpe_pct is not None and rd and rd >= 0.10 and rpe_pct < 0.25:
            sentences.append(
                "Note: heavy R&D investment has not yet translated to "
                "above-average workforce productivity — watch for commercialisation lag."
            )

        # --- External signals --------------------------------------------
        if cult_award:
            sentences.append(
                "Recent news includes recognition as a top employer or culture award."
            )
        if layoff:
            sentences.append(
                "Recent headlines include layoff or workforce-reduction announcements."
            )
        if emp_legal:
            sentences.append(
                "Active legal proceedings include employment or labour-related filings."
            )

        r['culture_narrative'] = " ".join(sentences) if sentences else None

    # -----------------------------------------------------------------------
    # Stock-level narrative (replaces sector-only headwinds/tailwinds)
    # -----------------------------------------------------------------------
    for r in results:
        hw, tw = generate_stock_narrative(
            r,
            sector_data=sector_etf_data if args.macro else None,
            macro_regime_result=macro_regime_result,
            commodity_data=commodity_data,
            sector_medians={'sector_median_ee': sector_median_ee,
                            'sector_median_opm': sector_median_opm},
        )
        r['sector_headwinds'] = hw
        r['sector_tailwinds'] = tw
        r['financial_summary'] = generate_financial_summary(r)

    results.sort(key=lambda r: (r.get('_composite_score') or 0, r.get('spread') or 0), reverse=True)

    # -----------------------------------------------------------------------
    # Morningstar comparison statistics
    # -----------------------------------------------------------------------
    if ms_pfv_data:
        ms_pairs = []
        for r in results:
            pfv_val = ms_pfv_data.get(r['ticker'])
            if pfv_val and r.get('price') and r.get('dcf_fv') and r['dcf_fv'] > 0:
                ms_fv = r['price'] / pfv_val
                if ms_fv > 0:
                    ms_pairs.append((r['dcf_fv'], ms_fv))
        if len(ms_pairs) >= MIN_MORNINGSTAR_SAMPLE:
            model_fvs = [p[0] for p in ms_pairs]
            ms_fvs = [p[1] for p in ms_pairs]
            rel_errors = [(m - ms) / ms for m, ms in ms_pairs]
            mae = sum(abs(e) for e in rel_errors) / len(rel_errors)
            mse = sum(rel_errors) / len(rel_errors)
            within_20 = sum(1 for e in rel_errors if abs(e) <= 0.20) / len(rel_errors)
            ratios_sorted = sorted(m / ms for m, ms in ms_pairs)
            median_ratio = ratios_sorted[len(ratios_sorted) // 2]
            # Spearman rank correlation
            n = len(model_fvs)
            rank_m = rank(model_fvs)
            rank_ms = rank(ms_fvs)
            d_sq = sum((rm - rms) ** 2 for rm, rms in zip(rank_m, rank_ms))
            spearman_rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1)) if n > 1 else 0.0

            print(f"\nMorningstar comparison ({len(ms_pairs)} stocks):")
            print(f"  Mean Absolute Error: {mae:.1%}")
            print(f"  Mean Signed Error:   {mse:+.1%} "
                  f"({'overestimates' if mse > 0 else 'underestimates'})")
            print(f"  Within ±20%:         {within_20:.0%}")
            print(f"  Median FV Ratio:     {median_ratio:.2f}")
            print(f"  Spearman ρ:          {spearman_rho:.3f}")

            # Per-group MS comparison (when validation data present)
            if args.validation:
                for grp_name in ('quality', 'poor'):
                    grp_pairs = []
                    for r in results:
                        if r.get('source_group') != grp_name:
                            continue
                        pfv_val = ms_pfv_data.get(r['ticker'])
                        if pfv_val and r.get('price') and r.get('dcf_fv') and r['dcf_fv'] > 0:
                            gms_fv = r['price'] / pfv_val
                            if gms_fv > 0:
                                grp_pairs.append((r['dcf_fv'], gms_fv))
                    if len(grp_pairs) >= 3:
                        grp_rel_errors = [(m - ms) / ms for m, ms in grp_pairs]
                        grp_mae = sum(abs(e) for e in grp_rel_errors) / len(grp_rel_errors)
                        grp_mse = sum(grp_rel_errors) / len(grp_rel_errors)
                        grp_w20 = sum(1 for e in grp_rel_errors if abs(e) <= 0.20) / len(grp_rel_errors)
                        print(f"  {grp_name:>8} ({len(grp_pairs)} stocks): "
                              f"MAE={grp_mae:.1%}  MSE={grp_mse:+.1%}  within20={grp_w20:.0%}")

    if args.validation:
        _print_validation_stats(results, screen_outcomes)

    os.makedirs("output", exist_ok=True)
    today_str = date.today().isoformat()
    html_filename = os.path.join("output", f"stock_analysis_results_{today_str}.html")
    build_html(results, html_filename, prices_dir=prices_dir)
    xlsx_filename = os.path.join("output", f"stock_analysis_results_{today_str}.xlsx")
    build_excel(results, xlsx_filename)

    # Save results as JSON for backtesting pipeline
    json_filename = os.path.join("output", f"results_{date.today().isoformat()}.json")
    json_rows = []
    for r in results:
        jr = {}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                jr[k] = v
            elif isinstance(v, list):
                jr[k] = v
            elif isinstance(v, dict):
                jr[k] = {str(dk): dv for dk, dv in v.items()
                         if isinstance(dv, (int, float, str, bool, type(None)))}
            elif isinstance(v, tuple):
                jr[k] = list(v)
            else:
                continue  # skip non-serializable
        json_rows.append(jr)
    json_meta = {
        'date': date.today().isoformat(),
        'risk_free_rate': risk_free_rate,
        'count': len(results),
    }
    if macro_regime_result:
        json_meta['macro_regime'] = macro_regime_result
        json_meta['macro_adjustments'] = macro_adj
        if local_rs:
            json_meta['sector_local_rs'] = local_rs
    json_meta['results'] = json_rows
    with open(json_filename, 'w') as f:
        json.dump(json_meta, f, indent=2, default=str)

    print(f"\nAnalysis complete. {len(results)} stocks.")
    print(f"  HTML: {html_filename}")
    print(f"  Excel: {xlsx_filename}")
    print(f"  JSON: {json_filename}")


if __name__ == "__main__":
    _main()
