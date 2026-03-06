# scripts/analyze_stock.py
import sys
import os
import io
import json
import pandas as pd
import numpy as np
from urllib.request import urlopen, Request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.sec_edgar import SECEdgarClient
from data.yfinance_client import YFinanceClient
from models.capm import (calculate_beta, expected_return, calculate_r2,
                          calculate_alpha, calculate_residual_sigma,
                          r2_diagnostic, ggm_implied_re, buildup_re,
                          geometric_annualized_return)
from models.dcf import (calculate_dcf, project_forward_dcf,
                         fair_value_per_share, dcf_sensitivity, ggm_value)
from data.sentiment import fetch_sentiment
from data.social_sentiment import fetch_social_sentiment
from models.comparisons import (compute_ratios, calculate_roic, calculate_wacc,
                                  calculate_earnings_quality, calculate_altman_z,
                                  calculate_piotroski_f, calculate_revenue_cagr,
                                  compute_relative_multiples, calculate_interest_coverage,
                                  calculate_net_debt_ebitda, get_net_debt,
                                  compute_analyst_consensus, compute_rating)

# --- Constants ---
RISK_FREE_RATE = 0.04          # 10-yr Treasury proxy
ERP = 0.055                    # Equity Risk Premium (Damodaran)
MARKET_TICKER = "^GSPC"
TERMINAL_GROWTH_RATE = 0.03
MIN_MARKET_CAP = 10e9          # Worksheet Step 1: Market Cap > $10B


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

def run_capm(yf_client, ticker, market_history):
    """Returns a dict with beta, r2, alpha, residual_sigma, capm_er — or None."""
    stock_history = yf_client.fetch_history(ticker, period="5y")
    aligned = pd.DataFrame({'stock': stock_history, 'market': market_history}).dropna()
    if len(aligned) < 60:
        return None
    stock_returns = aligned['stock'].pct_change().dropna().values
    market_returns = aligned['market'].pct_change().dropna().values
    min_len = min(len(stock_returns), len(market_returns))
    if min_len < 30:
        return None
    stock_returns = stock_returns[-min_len:]
    market_returns = market_returns[-min_len:]

    beta_result = calculate_beta(stock_returns, market_returns, adjust=True)
    r2 = calculate_r2(stock_returns, market_returns)
    alpha = calculate_alpha(stock_returns, market_returns)
    residual_sigma = calculate_residual_sigma(stock_returns, market_returns)
    market_annual_return = geometric_annualized_return(market_returns)
    if market_annual_return is None:
        market_annual_return = (1 + np.mean(market_returns)) ** 252 - 1
    capm_er = expected_return(RISK_FREE_RATE, beta_result['adjusted_beta'], market_annual_return)
    r2_class, re_method = r2_diagnostic(r2)

    return {
        'beta': beta_result['adjusted_beta'],
        'raw_beta': beta_result['raw_beta'],
        'r2': r2, 'alpha': alpha,
        'residual_sigma': residual_sigma,
        'se_beta': beta_result['se_beta'],
        'n_observations': beta_result['n_observations'],
        'r2_class': r2_class, 're_method': re_method,
        'capm_er': capm_er,
    }


def select_cost_of_equity(capm_data, financials):
    """
    Worksheet Step 4B: gate cost-of-equity method by R².
    Returns (cost_of_equity, method_label).
    """
    info = (financials.get('info') or {}) if financials else {}
    div_yield = info.get('dividendYield')

    # Fallback alternative: GGM-implied or Build-Up
    alt_re = None
    if div_yield and div_yield > 0:
        alt_re = ggm_implied_re(div_yield, TERMINAL_GROWTH_RATE)
    if alt_re is None or not (0.04 < alt_re < 0.30):
        alt_re = buildup_re(RISK_FREE_RATE, ERP)

    if capm_data is None:
        return alt_re, 'buildup_fallback'

    re_method = capm_data['re_method']
    capm_er = capm_data['capm_er']

    if re_method == 'capm':
        return capm_er, 'capm'
    elif re_method == 'capm_plus_alternative':
        return (capm_er + alt_re) / 2, 'capm+alt_midpoint'
    else:
        return alt_re, 'ggm_or_buildup'


# ---------------------------------------------------------------------------
# Forward DCF (Worksheet Step 5A)
# ---------------------------------------------------------------------------

def run_forward_dcf(yf_data, wacc):
    """
    Forward-looking 5-year DCF with Gordon Growth terminal value.
    Returns (fair_value_per_share, sensitivity_range, fcf_growth_rate) or (None, None, None).
    """
    cf = yf_data.get('cash_flow')
    info = yf_data.get('info') or {}
    bs = yf_data.get('balance_sheet')

    if cf is None or cf.empty:
        return None, None, None
    if 'Free Cash Flow' not in cf.index:
        return None, None, None
    if wacc is None or wacc <= TERMINAL_GROWTH_RATE:
        return None, None, None

    fcf_values = cf.loc['Free Cash Flow'].dropna().sort_index().values.tolist()
    if not fcf_values:
        return None, None, None

    base_fcf = fcf_values[-1]  # most recent after sort
    if base_fcf <= 0:
        return None, None, None

    # Historical FCF CAGR, capped 0–20%
    if len(fcf_values) >= 2 and fcf_values[0] > 0:
        n = len(fcf_values) - 1
        cagr = (fcf_values[-1] / fcf_values[0]) ** (1 / n) - 1
        fcf_growth = max(0.0, min(0.20, cagr))
    else:
        fcf_growth = 0.05

    ev = project_forward_dcf(base_fcf, fcf_growth, wacc, TERMINAL_GROWTH_RATE)
    if ev is None:
        return None, None, None

    net_debt = get_net_debt(yf_data)
    shares = info.get('sharesOutstanding')
    fv = fair_value_per_share(ev, net_debt, shares)

    # Sensitivity range (min/max of 5×5 table)
    sens_range = None
    if shares and shares > 0:
        sens = dcf_sensitivity(base_fcf, fcf_growth, wacc, TERMINAL_GROWTH_RATE, net_debt, shares)
        vals = [v for v in sens.values() if v is not None]
        if vals:
            sens_range = (min(vals), max(vals))

    return fv, sens_range, fcf_growth


# ---------------------------------------------------------------------------
# HTML report builder (tabbed table, Plotly charts)
# ---------------------------------------------------------------------------

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

def _dv(val):
    """data-value attribute — raw float for sorting, empty string if None."""
    return "" if val is None else str(round(float(val), 6))


def _rating_style(rating):
    colors = {'BUY': '#1a9850', 'LEAN BUY': '#74c476', 'HOLD': '#fd8d3c', 'PASS': '#de2d26'}
    c = colors.get(rating, '#888')
    return f' style="color:{c};font-weight:700;text-align:center"'

def _fmt_rec(rec):
    if not rec:
        return 'N/A'
    labels = {'strong_buy': 'Strong Buy', 'buy': 'Buy', 'hold': 'Hold',
              'sell': 'Sell', 'strong_sell': 'Strong Sell',
              'strongbuy': 'Strong Buy', 'strongsell': 'Strong Sell'}
    return labels.get(rec.lower().replace('_',''), rec.title())

def _rec_style(rec):
    if not rec:
        return ''
    r = rec.lower().replace('_','').replace(' ','')
    if r in ('strongbuy', 'buy'):   return ' style="color:#1a9850;font-weight:600"'
    if r in ('sell', 'strongsell'): return ' style="color:#de2d26;font-weight:600"'
    return ''

def _sentiment_style(label):
    if label == 'Positive': return ' style="color:#1a9850;font-weight:600"'
    if label == 'Negative': return ' style="color:#de2d26;font-weight:600"'
    return ' style="color:#888"'

_RATING_VAL = {'BUY': 3, 'LEAN BUY': 2, 'HOLD': 1, 'PASS': 0}
_REC_VAL = {'strongbuy': 4, 'buy': 3, 'hold': 2, 'sell': 1, 'strongsell': 0}

def _rating_num(rating):
    return _RATING_VAL.get(rating, -1)

def _rec_num(rec):
    if not rec:
        return -1
    return _REC_VAL.get(rec.lower().replace('_', '').replace(' ', ''), -1)

def build_html(rows, filename):
    total = len(rows)
    avg_spread = sum(r['spread'] for r in rows if r.get('spread')) / total if total else 0
    qualifying_with_mos = sum(1 for r in rows if r.get('mos') is not None and r['mos'] > 0)
    buy_count = sum(1 for r in rows if r.get('rating') == 'BUY')
    lean_buy_count = sum(1 for r in rows if r.get('rating') == 'LEAN BUY')

    chart_data = json.dumps([{
        'ticker': r['ticker'],
        'roic': r.get('roic'), 'wacc': r.get('wacc'), 'spread': r.get('spread'),
        'beta': r.get('beta'), 'r2': r.get('r2'), 'er': r.get('er'),
        'dcf_fv': r.get('dcf_fv'), 'price': r.get('price'), 'mos': r.get('mos'),
        'piotroski': r.get('piotroski'), 'altman_z': r.get('altman_z'),
        'pe': r.get('pe'), 'ev_ebitda': r.get('ev_ebitda'),
        'rating': r.get('rating'), 'analyst_rec': r.get('analyst_rec'),
        'description': (r.get('description') or '')[:200],
        'sector': r.get('sector'),
        'industry': r.get('industry'),
        'ceo': r.get('ceo'),
        'sentiment_score': r.get('sentiment_score'),
        'sentiment_label': r.get('sentiment_label'),
        'social_score': r.get('social_score'),
        'social_label': r.get('social_label'),
        # Detail modal fields
        'description_full': r.get('description') or '',
        'ceo_bio': r.get('ceo_bio') or '',
        'mcap': r.get('mcap'),
        'roic_by_year': r.get('roic_by_year'),
        'raw_beta': r.get('raw_beta'),
        'alpha': r.get('alpha'),
        'residual_sigma': r.get('residual_sigma'),
        'se_beta': r.get('se_beta'),
        'n_observations': r.get('n_observations'),
        're_method': r.get('re_method'),
        'dcf_sens_range': list(r['dcf_sens_range']) if r.get('dcf_sens_range') else None,
        'fcf_growth': r.get('fcf_growth'),
        'pfcf': r.get('pfcf'),
        'pb': r.get('pb'),
        'peg': r.get('peg'),
        'num_analysts': r.get('num_analysts'),
        'target_mean': r.get('target_mean'),
        'target_high': r.get('target_high'),
        'target_low': r.get('target_low'),
        'sentiment_articles': r.get('sentiment_articles'),
        'sentiment_bull': r.get('sentiment_bull'),
        'sentiment_bear': r.get('sentiment_bear'),
        'st_bull_pct': r.get('st_bull_pct'),
        'st_bear_pct': r.get('st_bear_pct'),
        'st_labeled': r.get('st_labeled'),
        'reddit_score': r.get('reddit_score'),
        'reddit_posts': r.get('reddit_posts'),
        'cash_conv': r.get('cash_conv'),
        'accruals': r.get('accruals'),
        'rev_cagr': r.get('rev_cagr'),
        'int_cov': r.get('int_cov'),
        'nd_ebitda': r.get('nd_ebitda'),
        'de': r.get('de'),
        'cr': r.get('cr'),
        'roe': r.get('roe'),
        'roa': r.get('roa'),
    } for r in rows])

    # --- Table rows ---
    table_rows = []
    for r in rows:
        mos = r.get('mos')
        mos_style = ''
        if mos is not None:
            mos_style = ' style="color:#1a9850;font-weight:600"' if mos > 0.15 else (' style="color:#de2d26"' if mos < 0 else '')
        pf = r.get('piotroski')
        pf_style = ''
        if pf is not None:
            pf_style = ' style="color:#1a9850;font-weight:600"' if pf >= 7 else (' style="color:#de2d26"' if pf <= 3 else '')
        az = r.get('altman_z')
        az_style = ''
        if az is not None:
            az_style = ' style="color:#1a9850;font-weight:600"' if az > 2.99 else (' style="color:#de2d26"' if az < 1.81 else ' style="color:#fd8d3c"')
        cc = r.get('cash_conv')
        cc_style = ' style="color:#1a9850;font-weight:600"' if (cc is not None and cc > 0.8) else ''

        dcf_fv_v  = r.get('dcf_fv')
        er_v      = r.get('er')
        price_v   = r.get('price')
        dcf_bear  = dcf_fv_v * 0.70 if dcf_fv_v is not None else None
        dcf_bull  = dcf_fv_v * 1.35 if dcf_fv_v is not None else None
        p2yr      = price_v * (1 + er_v) ** 2  if price_v is not None and er_v is not None else None
        p5yr      = price_v * (1 + er_v) ** 5  if price_v is not None and er_v is not None else None
        p10yr     = price_v * (1 + er_v) ** 10 if price_v is not None and er_v is not None else None

        rating = r.get('rating') or 'N/A'
        rec = r.get('analyst_rec')
        sent_label = r.get('sentiment_label')
        sent_score = r.get('sentiment_score')
        sent_articles = r.get('sentiment_articles') or 0
        sent_display = f"{sent_label} ({sent_score:+.2f}, {sent_articles}n)" if sent_label and sent_score is not None else 'N/A'
        # Social media sentiment display
        social_label = r.get('social_label')
        social_score = r.get('social_score')
        st_bull = r.get('st_bull_pct')
        st_bear = r.get('st_bear_pct')
        st_labeled = r.get('st_labeled') or 0
        rd_score = r.get('reddit_score')
        rd_posts = r.get('reddit_posts') or 0
        social_parts = []
        if st_labeled > 0 and st_bull is not None:
            social_parts.append(f"ST:{st_bull:.0%}↑/{st_bear:.0%}↓ ({st_labeled})")
        if rd_posts > 0 and rd_score is not None:
            social_parts.append(f"Reddit:{rd_score:+.2f} ({rd_posts}p)")
        social_display = " | ".join(social_parts) if social_parts else 'N/A'
        desc = (r.get('description') or '').replace('"', '&quot;').replace("'", '&#39;')
        ceo = r.get('ceo') or 'N/A'
        ceo_bio = (r.get('ceo_bio') or ceo).replace('"', '&quot;').replace("'", '&#39;')
        sector = r.get('sector') or 'N/A'
        industry = r.get('industry') or 'N/A'
        target_mean = r.get('target_mean')
        num_analysts = r.get('num_analysts')
        analyst_label = f"{_fmt_rec(rec)}" + (f" ({num_analysts})" if num_analysts else "")

        table_rows.append(
            f"<tr>"
            # Core — ticker has description tooltip; Rating, CEO, Sector, Industry visible here
            f"<td class='ticker' title='{desc}'>{r['ticker']}</td>"
            f"<td class='grp-ovr' style='font-size:0.75em;color:#555;max-width:120px;overflow:hidden;text-overflow:ellipsis' title='{ceo_bio}'>{ceo}</td>"
            f"<td class='grp-ovr grp-con' style='font-size:0.75em;color:#555;max-width:100px;overflow:hidden;text-overflow:ellipsis'>{sector}</td>"
            f"<td class='grp-ovr grp-con' style='font-size:0.75em;color:#555;max-width:120px;overflow:hidden;text-overflow:ellipsis'>{industry}</td>"
            f"<td class='grp-con' data-value='{_rating_num(rating)}'{_rating_style(rating)}>{rating}</td>"
            f"<td class='grp-ovr' data-value='{_dv(r.get('mcap'))}'>{fmt_dollar_short(r.get('mcap'))}</td>"
            f"<td class='grp-ovr' data-value='{_dv(r.get('roic'))}'>{fmt_pct(r.get('roic'))}</td>"
            f"<td class='grp-ovr' data-value='{_dv(r.get('wacc'))}'>{fmt_pct(r.get('wacc'))}</td>"
            f"<td class='grp-ovr' data-value='{_dv(r.get('spread'))}'>{fmt_pct(r.get('spread'))}</td>"
            # Valuation
            f"<td class='grp-val' data-value='{_dv(r.get('dcf_fv'))}'>{fmt_dollar(r.get('dcf_fv'))}</td>"
            f"<td class='grp-val' data-value='{_dv(r.get('price'))}'>{fmt_dollar(r.get('price'))}</td>"
            f"<td class='grp-val' data-value='{_dv(mos)}'{mos_style}>{fmt_pct(mos)}</td>"
            f"<td class='grp-con' data-value='{_rec_num(rec)}'{_rec_style(rec)}>{analyst_label}</td>"
            f"<td class='grp-con' data-value='{_dv(sent_score)}'{_sentiment_style(sent_label)}>{sent_display}</td>"
            f"<td class='grp-con' data-value='{_dv(social_score)}'{_sentiment_style(social_label)}>{social_display}</td>"
            f"<td class='grp-val' data-value='{_dv(target_mean)}'>{fmt_dollar(target_mean)}</td>"
            f"<td class='grp-val' data-value='{_dv(r.get('pe'))}'>{fmt_num(r.get('pe'), 1)}</td>"
            f"<td class='grp-val' data-value='{_dv(r.get('ev_ebitda'))}'>{fmt_num(r.get('ev_ebitda'), 1)}</td>"
            f"<td class='grp-val' data-value='{_dv(r.get('pfcf'))}'>{fmt_num(r.get('pfcf'), 1)}</td>"
            f"<td class='grp-val' data-value='{_dv(r.get('pb'))}'>{fmt_num(r.get('pb'), 2)}</td>"
            f"<td class='grp-val' data-value='{_dv(r.get('peg'))}'>{fmt_num(r.get('peg'), 2)}</td>"
            # Risk
            f"<td class='grp-risk' data-value='{_dv(r.get('beta'))}'>{fmt_num(r.get('beta'), 2)}</td>"
            f"<td class='grp-risk' data-value='{_dv(r.get('r2'))}'>{fmt_pct(r.get('r2'))}</td>"
            f"<td class='grp-risk' data-value='{_dv(r.get('alpha'))}'>{fmt_pct(r.get('alpha'))}</td>"
            f"<td class='grp-risk' data-value='{_dv(r.get('er'))}'>{fmt_pct(r.get('er'))}</td>"
            # Profitability
            f"<td class='grp-prof' data-value='{_dv(r.get('roe'))}'>{fmt_pct(r.get('roe'))}</td>"
            f"<td class='grp-prof' data-value='{_dv(r.get('roa'))}'>{fmt_pct(r.get('roa'))}</td>"
            f"<td class='grp-prof' data-value='{_dv(cc)}'{cc_style}>{fmt_num(cc, 2) if cc is not None else 'N/A'}</td>"
            f"<td class='grp-prof' data-value='{_dv(r.get('accruals'))}'>{fmt_num(r.get('accruals'), 3) if r.get('accruals') is not None else 'N/A'}</td>"
            f"<td class='grp-prof' data-value='{_dv(r.get('rev_cagr'))}'>{fmt_pct(r.get('rev_cagr'))}</td>"
            # Financial Health
            f"<td class='grp-hlth' data-value='{_dv(pf)}'{pf_style}>{fmt_num(pf, 0) if pf is not None else 'N/A'}</td>"
            f"<td class='grp-hlth' data-value='{_dv(az)}'{az_style}>{fmt_num(az, 2) if az is not None else 'N/A'}</td>"
            f"<td class='grp-hlth' data-value='{_dv(r.get('int_cov'))}'>{fmt_num(r.get('int_cov'), 1) if r.get('int_cov') is not None else 'N/A'}</td>"
            f"<td class='grp-hlth' data-value='{_dv(r.get('nd_ebitda'))}'>{fmt_num(r.get('nd_ebitda'), 2) if r.get('nd_ebitda') is not None else 'N/A'}</td>"
            f"<td class='grp-hlth' data-value='{_dv(r.get('de'))}'>{fmt_num(r.get('de'), 2)}</td>"
            f"<td class='grp-hlth' data-value='{_dv(r.get('cr'))}'>{fmt_num(r.get('cr'), 2)}</td>"
            # Projections
            f"<td class='grp-proj' data-value='{_dv(dcf_bear)}'>{fmt_dollar(dcf_bear)}</td>"
            f"<td class='grp-proj' data-value='{_dv(dcf_bull)}'>{fmt_dollar(dcf_bull)}</td>"
            f"<td class='grp-proj' data-value='{_dv(p2yr)}'>{fmt_dollar(p2yr)}</td>"
            f"<td class='grp-proj' data-value='{_dv(p5yr)}'>{fmt_dollar(p5yr)}</td>"
            f"<td class='grp-proj' data-value='{_dv(p10yr)}'>{fmt_dollar(p10yr)}</td>"
            f"</tr>"
        )
    table_body = "\n".join(table_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Analysis — Securities Analyst Worksheet Screen</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; color: #333; }}
  .header {{ background: linear-gradient(135deg, #1a252f, #2980b9); color: white; padding: 36px 20px; text-align: center; }}
  .header h1 {{ font-size: 2em; margin-bottom: 6px; }}
  .header p {{ font-size: 1em; opacity: 0.85; }}
  .stats {{ display: flex; gap: 24px; justify-content: center; margin-top: 16px; flex-wrap: wrap; }}
  .stat {{ background: rgba(255,255,255,0.15); padding: 10px 20px; border-radius: 8px; }}
  .stat .val {{ font-size: 1.4em; font-weight: bold; }}
  .stat .label {{ font-size: 0.8em; opacity: 0.8; }}
  .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
  .chart-box {{ background: white; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
  .toolbar {{ display: flex; align-items: center; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
  .tab-btn {{ padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85em; font-weight: 500;
              background: #dde; color: #555; transition: all 0.15s; }}
  .tab-btn.active {{ background: #2980b9; color: white; }}
  .tab-btn:hover:not(.active) {{ background: #c8d0e0; }}
  .search-bar input {{ padding: 8px 14px; font-size: 0.9em; border: 2px solid #ddd; border-radius: 6px; outline: none; width: 220px; }}
  .search-bar input:focus {{ border-color: #2980b9; }}
  .table-wrap {{ overflow-x: auto; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
  table {{ width: 100%; border-collapse: collapse; background: white; white-space: nowrap; }}
  thead {{ position: sticky; top: 0; z-index: 10; }}
  th {{ background: #1a252f; color: white; padding: 10px 8px; text-align: right; cursor: pointer;
        user-select: none; font-size: 0.78em; border-right: 1px solid #2c3e50; }}
  th:first-child {{ text-align: left; }}
  th:hover {{ background: #2c3e50; }}
  th .arrow {{ margin-left: 3px; font-size: 0.7em; }}
  th.grp-val  {{ background: #1a3a5c; }}
  th.grp-risk {{ background: #2c1a5c; }}
  th.grp-prof {{ background: #1a4a2c; }}
  th.grp-hlth {{ background: #3a2a10; }}
  th.grp-con  {{ background: #4a1a2a; }}
  th.grp-proj {{ background: #164a4a; }}
  td {{ padding: 8px; text-align: right; font-size: 0.82em; border-bottom: 1px solid #eee; border-right: 1px solid #f0f0f0; }}
  td.ticker {{ text-align: left; font-weight: 600; color: #1a252f; min-width: 60px; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  tr:hover {{ background: #eaf4fd; }}
  .legend {{ font-size: 0.78em; color: #888; margin: 8px 0 16px; }}
  .col-filter {{ position: relative; }}
  .col-filter-btn {{ padding: 8px 14px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85em; font-weight: 500; background: #dde; color: #555; transition: all 0.15s; }}
  .col-filter-btn:hover {{ background: #c8d0e0; }}
  .col-panel {{ position: absolute; top: 110%; left: 0; background: white; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); padding: 14px; z-index: 200; min-width: 300px; display: none; max-height: 440px; overflow-y: auto; }}
  .col-panel.open {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2px 14px; }}
  .col-grp-hdr {{ grid-column: 1 / -1; font-weight: 700; font-size: 0.73em; color: #888; margin-top: 10px; text-transform: uppercase; letter-spacing: 0.05em; display: flex; align-items: center; justify-content: space-between; border-top: 1px solid #eee; padding-top: 6px; }}
  .col-grp-hdr:first-child {{ margin-top: 0; border-top: none; padding-top: 0; }}
  .col-grp-hdr .grp-btns button {{ font-size: 0.78em; padding: 1px 7px; border: 1px solid #ccc; border-radius: 3px; cursor: pointer; background: #f5f5f5; margin-left: 3px; }}
  .col-grp-hdr .grp-btns button:hover {{ background: #e0e0e0; }}
  .col-panel label {{ font-size: 0.82em; cursor: pointer; display: flex; align-items: center; gap: 5px; padding: 2px 0; white-space: nowrap; }}
  .col-panel label:hover {{ color: #2980b9; }}
  /* Tab bar — own row above toolbar */
  .tabs-row {{ display: flex; align-items: center; gap: 6px; padding: 16px 0 10px; flex-wrap: wrap; border-bottom: 2px solid #e0e4ea; }}
  .filter-row {{ display: flex; align-items: center; gap: 12px; padding: 10px 0 6px; flex-wrap: wrap; }}
  /* Per-tab active color matches column group header */
  .tab-btn[data-tab="ovr"].active  {{ background: #1a252f; }}
  .tab-btn[data-tab="val"].active  {{ background: #1a3a5c; }}
  .tab-btn[data-tab="risk"].active {{ background: #2c1a5c; }}
  .tab-btn[data-tab="prof"].active {{ background: #1a4a2c; }}
  .tab-btn[data-tab="hlth"].active {{ background: #3a2a10; }}
  .tab-btn[data-tab="con"].active  {{ background: #4a1a2a; }}
  .tab-btn[data-tab="proj"].active {{ background: #164a4a; }}
  .tab-btn[data-tab="all"].active  {{ background: #444;    }}
  /* Chart section header */
  .chart-section-hdr {{ display: flex; align-items: baseline; gap: 12px; margin: 14px 0 8px; padding: 10px 16px; background: white; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.06); border-left: 4px solid #1a252f; transition: border-color 0.25s; }}
  #chart-tab-label {{ font-size: 1em; font-weight: 700; color: #1a252f; min-width: 110px; }}
  #chart-tab-desc  {{ font-size: 0.82em; color: #666; line-height: 1.4; }}
  /* --- Detail Modal --- */
  .detail-modal {{
    display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.55); z-index: 1000; overflow-y: auto;
    animation: fadeIn 0.15s ease;
  }}
  .detail-modal.open {{ display: flex; justify-content: center; padding: 40px 20px; }}
  @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
  .detail-content {{
    background: #fff; border-radius: 12px; max-width: 960px; width: 100%;
    box-shadow: 0 8px 40px rgba(0,0,0,0.25); position: relative;
    padding: 32px; max-height: calc(100vh - 80px); overflow-y: auto;
  }}
  .detail-close {{
    position: absolute; top: 12px; right: 16px; border: none; background: none;
    font-size: 1.8em; color: #888; cursor: pointer; line-height: 1;
  }}
  .detail-close:hover {{ color: #333; }}
  .detail-header {{
    display: flex; align-items: center; gap: 16px; margin-bottom: 8px; flex-wrap: wrap;
  }}
  .detail-ticker {{ font-size: 1.8em; font-weight: 800; color: #1a252f; }}
  .detail-rating {{
    font-size: 1em; font-weight: 700; padding: 4px 14px; border-radius: 6px; color: white;
  }}
  .detail-price-block {{ margin-left: auto; text-align: right; }}
  .detail-price {{ font-size: 1.4em; font-weight: 700; color: #1a252f; }}
  .detail-mcap {{ display: block; font-size: 0.8em; color: #888; }}
  .detail-meta {{ font-size: 0.85em; color: #666; margin-bottom: 12px; }}
  .detail-description {{
    font-size: 0.82em; color: #555; line-height: 1.6; margin-bottom: 20px;
    max-height: 120px; overflow-y: auto; padding: 10px 14px;
    background: #f8f9fa; border-radius: 8px; border-left: 3px solid #2980b9;
  }}
  .detail-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;
  }}
  @media (max-width: 700px) {{ .detail-grid {{ grid-template-columns: 1fr; }} }}
  .detail-section {{
    background: #f8f9fb; border-radius: 8px; padding: 14px 16px;
  }}
  .detail-section h3 {{
    font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.05em;
    color: #1a252f; margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 2px solid #e0e4ea;
  }}
  .detail-kv-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px;
  }}
  .detail-kv {{
    display: flex; justify-content: space-between; padding: 3px 0;
    font-size: 0.82em; border-bottom: 1px solid #eee;
  }}
  .detail-kv .kv-label {{ color: #888; }}
  .detail-kv .kv-value {{ font-weight: 600; color: #333; }}
  .detail-charts {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }}
  @media (max-width: 700px) {{ .detail-charts {{ grid-template-columns: 1fr; }} }}
  .detail-chart-box {{
    background: #f8f9fb; border-radius: 8px; padding: 8px;
  }}
  td.ticker {{ cursor: pointer; text-decoration: underline; text-decoration-color: rgba(41,128,185,0.3); text-underline-offset: 2px; }}
  td.ticker:hover {{ color: #2980b9; text-decoration-color: #2980b9; }}
</style>
</head>
<body>
<div class="header">
  <h1>Stock Analysis</h1>
  <p>Securities Analyst Worksheet — ROIC&gt;WACC Screen with Multi-Model Valuation</p>
  <div class="stats">
    <div class="stat"><div class="val">{total}</div><div class="label">Qualifying Companies</div></div>
    <div class="stat"><div class="val">{avg_spread:.1%}</div><div class="label">Avg ROIC-WACC Spread</div></div>
    <div class="stat"><div class="val">{qualifying_with_mos}</div><div class="label">DCF Undervalued (MoS&gt;0)</div></div>
    <div class="stat"><div class="val">{buy_count}</div><div class="label">BUY Rated</div></div>
    <div class="stat"><div class="val">{lean_buy_count}</div><div class="label">LEAN BUY Rated</div></div>
  </div>
</div>
<div class="container">
  <div class="tabs-row">
    <button class="tab-btn active" data-tab="ovr" onclick="showTab('ovr')">Overview</button>
    <button class="tab-btn" data-tab="val" onclick="showTab('val')">Valuation</button>
    <button class="tab-btn" data-tab="risk" onclick="showTab('risk')">Risk</button>
    <button class="tab-btn" data-tab="prof" onclick="showTab('prof')">Profitability</button>
    <button class="tab-btn" data-tab="hlth" onclick="showTab('hlth')">Financial Health</button>
    <button class="tab-btn" data-tab="con" onclick="showTab('con')">Consensus</button>
    <button class="tab-btn" data-tab="proj" onclick="showTab('proj')">Projections</button>
    <button class="tab-btn" data-tab="all" onclick="showTab('all')">All</button>
  </div>

  <div class="filter-row">
    <div class="search-bar">
      <input type="text" id="search" placeholder="Search ticker or company..." oninput="filterTable()">
    </div>
    <div class="col-filter">
      <button class="col-filter-btn" onclick="toggleColPanel()">Columns &#9660;</button>
      <div id="col-panel" class="col-panel">
        <div class="col-grp-hdr">Overview <span class="grp-btns"><button onclick="setGroup(['ceo','sector','industry','mcap','roic','wacc','spread'],true)">All</button><button onclick="setGroup(['ceo','sector','industry','mcap','roic','wacc','spread'],false)">None</button></span></div>
        <label><input type="checkbox" data-colkey="ceo" checked onchange="toggleColumn('ceo',this.checked)"> CEO</label>
        <label><input type="checkbox" data-colkey="sector" checked onchange="toggleColumn('sector',this.checked)"> Sector</label>
        <label><input type="checkbox" data-colkey="industry" checked onchange="toggleColumn('industry',this.checked)"> Industry</label>
        <label><input type="checkbox" data-colkey="mcap" checked onchange="toggleColumn('mcap',this.checked)"> Mkt Cap</label>
        <label><input type="checkbox" data-colkey="roic" checked onchange="toggleColumn('roic',this.checked)"> ROIC</label>
        <label><input type="checkbox" data-colkey="wacc" checked onchange="toggleColumn('wacc',this.checked)"> WACC</label>
        <label><input type="checkbox" data-colkey="spread" checked onchange="toggleColumn('spread',this.checked)"> Spread</label>
        <div class="col-grp-hdr">Valuation <span class="grp-btns"><button onclick="setGroup(['dcf_fv','price','mos','target_mean','pe','ev_ebitda','pfcf','pb','peg'],true)">All</button><button onclick="setGroup(['dcf_fv','price','mos','target_mean','pe','ev_ebitda','pfcf','pb','peg'],false)">None</button></span></div>
        <label><input type="checkbox" data-colkey="dcf_fv" checked onchange="toggleColumn('dcf_fv',this.checked)"> DCF/Shr</label>
        <label><input type="checkbox" data-colkey="price" checked onchange="toggleColumn('price',this.checked)"> Price</label>
        <label><input type="checkbox" data-colkey="mos" checked onchange="toggleColumn('mos',this.checked)"> MoS%</label>
        <label><input type="checkbox" data-colkey="target_mean" checked onchange="toggleColumn('target_mean',this.checked)"> Target $</label>
        <label><input type="checkbox" data-colkey="pe" checked onchange="toggleColumn('pe',this.checked)"> P/E</label>
        <label><input type="checkbox" data-colkey="ev_ebitda" checked onchange="toggleColumn('ev_ebitda',this.checked)"> EV/EBITDA</label>
        <label><input type="checkbox" data-colkey="pfcf" checked onchange="toggleColumn('pfcf',this.checked)"> P/FCF</label>
        <label><input type="checkbox" data-colkey="pb" checked onchange="toggleColumn('pb',this.checked)"> P/B</label>
        <label><input type="checkbox" data-colkey="peg" checked onchange="toggleColumn('peg',this.checked)"> PEG</label>
        <div class="col-grp-hdr">Risk <span class="grp-btns"><button onclick="setGroup(['beta','r2','alpha','er'],true)">All</button><button onclick="setGroup(['beta','r2','alpha','er'],false)">None</button></span></div>
        <label><input type="checkbox" data-colkey="beta" checked onchange="toggleColumn('beta',this.checked)"> Beta</label>
        <label><input type="checkbox" data-colkey="r2" checked onchange="toggleColumn('r2',this.checked)"> R&#178;</label>
        <label><input type="checkbox" data-colkey="alpha" checked onchange="toggleColumn('alpha',this.checked)"> Alpha</label>
        <label><input type="checkbox" data-colkey="er" checked onchange="toggleColumn('er',this.checked)"> Exp Ret</label>
        <div class="col-grp-hdr">Profitability <span class="grp-btns"><button onclick="setGroup(['roe','roa','cash_conv','accruals','rev_cagr'],true)">All</button><button onclick="setGroup(['roe','roa','cash_conv','accruals','rev_cagr'],false)">None</button></span></div>
        <label><input type="checkbox" data-colkey="roe" checked onchange="toggleColumn('roe',this.checked)"> ROE</label>
        <label><input type="checkbox" data-colkey="roa" checked onchange="toggleColumn('roa',this.checked)"> ROA</label>
        <label><input type="checkbox" data-colkey="cash_conv" checked onchange="toggleColumn('cash_conv',this.checked)"> CashConv</label>
        <label><input type="checkbox" data-colkey="accruals" checked onchange="toggleColumn('accruals',this.checked)"> Accruals</label>
        <label><input type="checkbox" data-colkey="rev_cagr" checked onchange="toggleColumn('rev_cagr',this.checked)"> Rev CAGR</label>
        <div class="col-grp-hdr">Financial Health <span class="grp-btns"><button onclick="setGroup(['piotroski','altman_z','int_cov','nd_ebitda','de','cr'],true)">All</button><button onclick="setGroup(['piotroski','altman_z','int_cov','nd_ebitda','de','cr'],false)">None</button></span></div>
        <label><input type="checkbox" data-colkey="piotroski" checked onchange="toggleColumn('piotroski',this.checked)"> F-Score</label>
        <label><input type="checkbox" data-colkey="altman_z" checked onchange="toggleColumn('altman_z',this.checked)"> Altman Z</label>
        <label><input type="checkbox" data-colkey="int_cov" checked onchange="toggleColumn('int_cov',this.checked)"> Int Cov</label>
        <label><input type="checkbox" data-colkey="nd_ebitda" checked onchange="toggleColumn('nd_ebitda',this.checked)"> ND/EBITDA</label>
        <label><input type="checkbox" data-colkey="de" checked onchange="toggleColumn('de',this.checked)"> D/E</label>
        <label><input type="checkbox" data-colkey="cr" checked onchange="toggleColumn('cr',this.checked)"> Curr R</label>
        <div class="col-grp-hdr">Consensus <span class="grp-btns"><button onclick="setGroup(['rating','analyst_rec','sentiment_score','social_score'],true)">All</button><button onclick="setGroup(['rating','analyst_rec','sentiment_score','social_score'],false)">None</button></span></div>
        <label><input type="checkbox" data-colkey="rating" checked onchange="toggleColumn('rating',this.checked)"> Rating</label>
        <label><input type="checkbox" data-colkey="analyst_rec" checked onchange="toggleColumn('analyst_rec',this.checked)"> Analyst Rec</label>
        <label><input type="checkbox" data-colkey="sentiment_score" checked onchange="toggleColumn('sentiment_score',this.checked)"> News Sentiment</label>
        <label><input type="checkbox" data-colkey="social_score" checked onchange="toggleColumn('social_score',this.checked)"> Social Sentiment</label>
        <div class="col-grp-hdr">Projections <span class="grp-btns"><button onclick="setGroup(['dcf_bear','dcf_bull','p2yr','p5yr','p10yr'],true)">All</button><button onclick="setGroup(['dcf_bear','dcf_bull','p2yr','p5yr','p10yr'],false)">None</button></span></div>
        <label><input type="checkbox" data-colkey="dcf_bear" checked onchange="toggleColumn('dcf_bear',this.checked)"> DCF Bear</label>
        <label><input type="checkbox" data-colkey="dcf_bull" checked onchange="toggleColumn('dcf_bull',this.checked)"> DCF Bull</label>
        <label><input type="checkbox" data-colkey="p2yr" checked onchange="toggleColumn('p2yr',this.checked)"> Price 2yr</label>
        <label><input type="checkbox" data-colkey="p5yr" checked onchange="toggleColumn('p5yr',this.checked)"> Price 5yr</label>
        <label><input type="checkbox" data-colkey="p10yr" checked onchange="toggleColumn('p10yr',this.checked)"> Price 10yr</label>
      </div>
    </div>
  </div>

  <div class="chart-section-hdr" id="chart-section-hdr">
    <div id="chart-tab-label">Overview</div>
    <div id="chart-tab-desc">Capital efficiency screening — ROIC vs WACC spread, DCF vs price, top-spread ranking, and model vs analyst rating breakdown.</div>
  </div>
  <div class="charts">
    <div class="chart-box"><div id="scatter-roic-wacc"></div></div>
    <div class="chart-box"><div id="scatter-dcf-price"></div></div>
    <div class="chart-box"><div id="top-spread"></div></div>
    <div class="chart-box"><div id="rating-chart"></div></div>
  </div>

  <div class="legend">
    Rating: <span style="color:#1a9850">&#9632;</span> BUY &nbsp;
    <span style="color:#74c476">&#9632;</span> LEAN BUY &nbsp;
    <span style="color:#fd8d3c">&#9632;</span> HOLD &nbsp;
    <span style="color:#de2d26">&#9632;</span> PASS &nbsp;&nbsp;
    Piotroski F: <span style="color:#1a9850">&#9632;</span> 7-9 strong &nbsp;
    <span style="color:#de2d26">&#9632;</span> 0-3 weak &nbsp;&nbsp;
    Altman Z: <span style="color:#1a9850">&#9632;</span> &gt;2.99 safe &nbsp;
    <span style="color:#fd8d3c">&#9632;</span> grey &nbsp;
    <span style="color:#de2d26">&#9632;</span> &lt;1.81 distress &nbsp;&nbsp;
    MoS%: <span style="color:#1a9850">&#9632;</span> &gt;15% undervalued &nbsp;&nbsp;
    <i>Hover ticker for company description. Hover CEO for biography (name, age, compensation). Consumer sentiment requires external API.</i>
  </div>
  <div class="table-wrap">
  <table id="data-table">
    <thead>
      <tr>
        <th data-col="ticker" onclick="sortCol('ticker')" title="Stock ticker symbol identifying this security on the exchange. Hover a row to see the full company description and business overview.">Ticker <span class="arrow"></span></th>
        <th class="grp-ovr" data-col="ceo" onclick="sortCol('ceo')" title="Chief Executive Officer. Look for tenure &gt;3 years and recent insider purchases as signals of confidence. Hover a cell for name, title, age, and total annual compensation.">CEO <span class="arrow"></span></th>
        <th class="grp-ovr grp-con" data-col="sector" onclick="sortCol('sector')" title="GICS market sector (e.g. Technology, Healthcare, Financials). Always compare companies within the same sector — valuation multiples and margin profiles vary widely across sectors.">Sector <span class="arrow"></span></th>
        <th class="grp-ovr grp-con" data-col="industry" onclick="sortCol('industry')" title="GICS sub-industry within a sector (e.g. Semiconductors within Technology). Use for peer-level comparison of multiples, margins, and capital intensity.">Industry <span class="arrow"></span></th>
        <th class="grp-con" data-col="rating" onclick="sortCol('rating')" style="text-align:center" title="Composite model rating: BUY / LEAN BUY / HOLD / PASS. Combines 3 signals: ROIC-WACC Spread &gt;0, DCF Margin of Safety &gt;15%, Analyst consensus &gt;=Buy. BUY = all 3; LEAN BUY = 2 of 3; HOLD = 1 of 3; PASS = 0 of 3.">Rating <span class="arrow"></span></th>
        <th class="grp-ovr" data-col="mcap" onclick="sortCol('mcap')" title="Market Capitalization = Share Price x Shares Outstanding. Screen requires &gt;$10B to filter for liquid, institutional-grade companies. Larger cap generally implies lower volatility and tighter bid-ask spreads.">Mkt Cap <span class="arrow"></span></th>
        <th class="grp-ovr" data-col="roic" onclick="sortCol('roic')" title="Return on Invested Capital = NOPAT / Invested Capital. NOPAT = EBIT x (1 - Tax Rate). Invested Capital = Total Equity + Net Debt. Displayed as 5-year average. Target &gt;15% for a high-quality moat business. ROIC must exceed WACC to create economic value.">ROIC <span class="arrow"></span></th>
        <th class="grp-ovr" data-col="wacc" onclick="sortCol('wacc')" title="Weighted Average Cost of Capital = (E/V) x Re + (D/V) x Rd x (1-t). Re = cost of equity (CAPM or GGM), Rd = pre-tax cost of debt, t = tax rate, E/V and D/V = equity and debt weights by market value. This is the hurdle rate — every dollar invested must earn above WACC to create shareholder value.">WACC <span class="arrow"></span></th>
        <th class="grp-ovr" data-col="spread" onclick="sortCol('spread')" title="Economic Value Spread = ROIC - WACC. A sustained positive spread signals a durable competitive moat. Target &gt;3% for high-conviction holdings. All companies shown here have Spread &gt;0 (value creators only).">Spread <span class="arrow"></span></th>
        <th class="grp-val" data-col="dcf_fv" onclick="sortCol('dcf_fv')" title="DCF Fair Value Per Share. Formula: FV = [Sum of FCFt/(1+WACC)^t for t=1..5] + [FCF5x(1+g)/(WACC-g)]/(1+WACC)^5, then divided by shares. FCF growth estimated from 5-yr historical CAGR (capped 0-20%). Terminal growth g=3%. Compare to Price for margin of safety.">DCF/Shr <span class="arrow"></span></th>
        <th class="grp-val" data-col="price" onclick="sortCol('price')" title="Current market price per share at time of analysis, sourced from yfinance. Compare to DCF/Shr to determine whether the stock trades at a discount or premium to intrinsic value.">Price <span class="arrow"></span></th>
        <th class="grp-val" data-col="mos" onclick="sortCol('mos')" title="Margin of Safety = (DCF Fair Value - Price) / DCF Fair Value. &gt;15%: potential undervaluation — consider as a buy signal (green). &lt;0%: price exceeds DCF estimate — avoid initiating. 0-15%: fairly valued. Only initiate a position when MoS &gt;15% to protect against model error.">MoS% <span class="arrow"></span></th>
        <th class="grp-con" data-col="analyst_rec" onclick="sortCol('analyst_rec')" title="Wall Street consensus recommendation from all sell-side analysts covering the stock: Strong Buy, Buy, Hold, Sell, Strong Sell. Higher analyst count improves reliability. Use as a confirming signal alongside the model rating, not as a standalone trigger.">Analyst Rec <span class="arrow"></span></th>
        <th class="grp-con" data-col="sentiment_score" onclick="sortCol('sentiment_score')" title="News sentiment score from VADER analysis of recent headlines via yfinance. Score = Sum(VADER compound scores) / n. Range: -1 (very negative) to +1 (very positive). |score| &gt;0.05 labels as Positive/Negative. Used as a real-time market mood indicator.">News Sentiment <span class="arrow"></span></th>
        <th class="grp-con" data-col="social_score" onclick="sortCol('social_score')" title="Composite social media sentiment. Formula: Score = 0.6 x StockTwits (Bullish%-Bearish%) + 0.4 x Reddit VADER score. StockTwits uses crowdsourced Bullish/Bearish labels; Reddit aggregates r/stocks, r/wallstreetbets, r/investing. High retail interest can amplify short-term volatility.">Social Sentiment <span class="arrow"></span></th>
        <th class="grp-val" data-col="target_mean" onclick="sortCol('target_mean')" title="Analyst consensus mean 12-month price target from all covering analysts. Implied upside = (Target - Price) / Price. Higher analyst count improves reliability. Use in conjunction with DCF MoS% as a cross-check on fair value.">Target $ <span class="arrow"></span></th>
        <th class="grp-val" data-col="pe" onclick="sortCol('pe')" title="Price-to-Earnings = Market Price / Trailing 12-month EPS. Lower P/E relative to sector median may indicate cheapness. Compare to the company's own 5-year average P/E for context. Negative P/E means earnings are negative — exclude from peer comparisons.">P/E <span class="arrow"></span></th>
        <th class="grp-val" data-col="ev_ebitda" onclick="sortCol('ev_ebitda')" title="EV/EBITDA = Enterprise Value / EBITDA. EV = Market Cap + Total Debt - Cash. EBITDA = Earnings before Interest, Tax, Depreciation &amp; Amortization. Capital-structure-neutral — useful for comparing companies with different leverage. Benchmark: &lt;10x attractive for mature businesses; tech often trades &gt;20x.">EV/EBITDA <span class="arrow"></span></th>
        <th class="grp-val" data-col="pfcf" onclick="sortCol('pfcf')" title="Price-to-Free Cash Flow = Market Cap / FCF. FCF = Operating Cash Flow - Capital Expenditures. Often more reliable than P/E since FCF is harder to manipulate with accounting choices. Lower P/FCF means you pay less per dollar of real cash generated by the business.">P/FCF <span class="arrow"></span></th>
        <th class="grp-val" data-col="pb" onclick="sortCol('pb')" title="Price-to-Book = Market Price / Book Value Per Share. Book Value = Total Equity / Shares Outstanding. P/B &lt;1 may signal undervaluation or impaired assets. High-ROIC companies typically command a premium to book. Adjust interpretation using P/B / ROE ratio.">P/B <span class="arrow"></span></th>
        <th class="grp-val" data-col="peg" onclick="sortCol('peg')" title="PEG Ratio = P/E / EPS Annual Growth Rate (%). Adjusts P/E for growth to avoid penalizing fast-growers. PEG &lt;1 often suggests undervaluation relative to growth; PEG &gt;2 may be expensive. Invalid (shown as N/A) when P/E or growth rate is negative.">PEG <span class="arrow"></span></th>
        <th class="grp-risk" data-col="beta" onclick="sortCol('beta')" title="CAPM Beta = Cov(Rs, Rm) / Var(Rm). Estimated from 5-year monthly regression vs S&amp;P 500. Beta=1: moves with market. Beta&gt;1: amplifies market swings. Beta&lt;1: more defensive. Treat as unreliable if R-squared &lt;40% — use alternative cost-of-equity in that case.">Beta <span class="arrow"></span></th>
        <th class="grp-risk" data-col="r2" onclick="sortCol('r2')" title="R-squared from 5-year monthly CAPM regression. Formula: R2 = [Corr(Rs, Rm)]^2. Fraction of return variance explained by market movements. Interpretation: &gt;=60% = beta is reliable; 40-60% = directional only; &lt;40% = beta unreliable, use GGM or Build-Up cost of equity instead.">R&sup2; <span class="arrow"></span></th>
        <th class="grp-risk" data-col="alpha" onclick="sortCol('alpha')" title="Jensen's Alpha = Annualized actual return - CAPM expected return. Formula: Alpha = Rs - [Rf + Beta x (Rm - Rf)]. Positive alpha = stock outperformed its level of systematic risk. Rf=4.0% (10-yr Treasury), ERP=5.5% (Damodaran). Only meaningful when R-squared &gt;=40%.">Alpha <span class="arrow"></span></th>
        <th class="grp-risk" data-col="er" onclick="sortCol('er')" title="CAPM Expected Return = Rf + Beta x (Rm - Rf). With Rf=4.0% (10-yr Treasury proxy) and ERP=5.5% (Damodaran equity risk premium). This is the theoretically required return to compensate for systematic market risk. Compare to ROIC: if ROIC &gt; Exp Ret, the company earns above its cost of capital.">Exp Ret <span class="arrow"></span></th>
        <th class="grp-prof" data-col="roe" onclick="sortCol('roe')" title="Return on Equity = Net Income / Average Shareholders Equity. Measures return earned for equity holders. High ROE with low D/E indicates genuine profitability; high ROE driven by high leverage may be misleading. Target &gt;15% for a quality business. Compare alongside ROA to assess leverage impact.">ROE <span class="arrow"></span></th>
        <th class="grp-prof" data-col="roa" onclick="sortCol('roa')" title="Return on Assets = Net Income / Average Total Assets. Asset efficiency unaffected by capital structure — comparable across companies with different leverage. Target &gt;10% for a capital-light, high-quality business. Always compare within the same industry.">ROA <span class="arrow"></span></th>
        <th class="grp-prof" data-col="cash_conv" onclick="sortCol('cash_conv')" title="Cash Conversion = Operating Cash Flow / Net Income. &gt;1.0 means every reported dollar of earnings is backed by real cash — high earnings quality. &lt;0.8 suggests heavy accruals or non-recurring items inflating reported net income. Look for sustained CashConv &gt;0.8 as a quality signal.">CashConv <span class="arrow"></span></th>
        <th class="grp-prof" data-col="accruals" onclick="sortCol('accruals')" title="Accruals Ratio = (Net Income - Operating Cash Flow) / Average Total Assets. Based on Sloan (1996): high accruals predict subsequent earnings reversals. Lower (more negative) = higher quality earnings. Target &lt;0.05; values &gt;0.10 warrant investigation into revenue recognition or expense deferrals.">Accruals <span class="arrow"></span></th>
        <th class="grp-prof" data-col="rev_cagr" onclick="sortCol('rev_cagr')" title="5-Year Revenue CAGR = (Revenue_now / Revenue_5yr_ago)^(1/5) - 1. Measures organic top-line growth trajectory. Target &gt;7-10% for sustained demand growth. Combine with margin trends: growing revenue with shrinking margins signals price competition or cost pressure.">Rev CAGR <span class="arrow"></span></th>
        <th class="grp-hlth" data-col="piotroski" onclick="sortCol('piotroski')" title="Piotroski F-Score (0-9). Sum of 9 binary signals (0 or 1 each): Profitability — ROA&gt;0, delta-ROA&gt;0, OCF&gt;0, Accruals&lt;0; Leverage — delta-Leverage&lt;0, delta-Liquidity&gt;0, No new shares; Efficiency — delta-Gross Margin&gt;0, delta-Asset Turnover&gt;0. Score 7-9: strong (green). Score 0-3: weak (red). Score 4-6: neutral.">F-Score <span class="arrow"></span></th>
        <th class="grp-hlth" data-col="altman_z" onclick="sortCol('altman_z')" title="Altman Z-Score = 1.2xX1 + 1.4xX2 + 3.3xX3 + 0.6xX4 + 1.0xX5. X1=Working Capital/Assets, X2=Retained Earnings/Assets, X3=EBIT/Assets, X4=Mkt Cap/Total Liabilities, X5=Revenue/Assets. &gt;2.99: Safe zone. 1.81-2.99: Grey zone. &lt;1.81: Distress zone. Developed for manufacturing; adjust interpretation for asset-light companies.">Altman Z <span class="arrow"></span></th>
        <th class="grp-hlth" data-col="int_cov" onclick="sortCol('int_cov')" title="Interest Coverage = EBIT / Interest Expense. Measures how many times operating earnings cover debt servicing obligations. &gt;3x: comfortable. 1.5-3x: watch closely. &lt;1.5x: potential financial distress. Negative EBIT makes the ratio meaningless — flag separately.">Int Cov <span class="arrow"></span></th>
        <th class="grp-hlth" data-col="nd_ebitda" onclick="sortCol('nd_ebitda')" title="Net Debt / EBITDA = (Total Debt - Cash) / EBITDA. How many years of earnings needed to repay all net debt. &lt;2x: conservative leverage. 2-3x: moderate. &gt;4x: high leverage, watch for covenant risk. Negative value = net cash position (fortress balance sheet).">ND/EBITDA <span class="arrow"></span></th>
        <th class="grp-hlth" data-col="de" onclick="sortCol('de')" title="Debt-to-Equity = Total Debt / Total Shareholders Equity. Measures financial leverage. Higher D/E = more debt in the capital structure = higher financial risk. Capital-intensive industries (utilities, airlines, banks) naturally carry higher D/E. &lt;1x is generally conservative for most sectors.">D/E <span class="arrow"></span></th>
        <th class="grp-hlth" data-col="cr" onclick="sortCol('cr')" title="Current Ratio = Current Assets / Current Liabilities. Measures ability to cover short-term obligations with short-term assets. &gt;1: can meet near-term obligations. &gt;2: very liquid (but may signal idle assets). &lt;1: potential short-term liquidity stress — investigate cash burn and working capital cycle.">Curr R <span class="arrow"></span></th>
        <th class="grp-proj" data-col="dcf_bear" onclick="sortCol('dcf_bear')" title="DCF Bear = DCF Fair Value × 0.70. Pessimistic scenario applying a −30% sensitivity to the base-case DCF, reflecting higher discount rates or lower terminal growth.">DCF Bear <span class="arrow"></span></th>
        <th class="grp-proj" data-col="dcf_bull" onclick="sortCol('dcf_bull')" title="DCF Bull = DCF Fair Value × 1.35. Optimistic scenario applying a +35% sensitivity to the base-case DCF, reflecting lower discount rates or higher terminal growth.">DCF Bull <span class="arrow"></span></th>
        <th class="grp-proj" data-col="p2yr" onclick="sortCol('p2yr')" title="Price in 2 Years (base) = Current Price × (1 + CAPM Expected Return)². Projects price forward 2 years assuming the market converges to CAPM fair value.">Price 2yr <span class="arrow"></span></th>
        <th class="grp-proj" data-col="p5yr" onclick="sortCol('p5yr')" title="Price in 5 Years (base) = Current Price × (1 + CAPM Expected Return)⁵. Projects price forward 5 years at the CAPM expected rate of return.">Price 5yr <span class="arrow"></span></th>
        <th class="grp-proj" data-col="p10yr" onclick="sortCol('p10yr')" title="Price in 10 Years (base) = Current Price × (1 + CAPM Expected Return)¹⁰. Long-horizon projection at the CAPM expected rate of return; most sensitive to Er accuracy.">Price 10yr <span class="arrow"></span></th>
      </tr>
    </thead>
    <tbody>
      {table_body}
    </tbody>
  </table>
  </div>
</div>

<!-- Ticker Detail Modal -->
<div id="detail-modal" class="detail-modal" onclick="if(event.target===this)closeDetail()">
  <div class="detail-content">
    <button class="detail-close" onclick="closeDetail()">&times;</button>
    <div class="detail-header">
      <div class="detail-ticker" id="detail-ticker"></div>
      <div class="detail-rating" id="detail-rating"></div>
      <div class="detail-price-block">
        <span class="detail-price" id="detail-price"></span>
        <span class="detail-mcap" id="detail-mcap"></span>
      </div>
    </div>
    <div class="detail-meta" id="detail-meta"></div>
    <div class="detail-description" id="detail-description"></div>
    <div class="detail-grid">
      <div class="detail-section"><h3>Valuation</h3><div class="detail-kv-grid" id="detail-val-kvs"></div></div>
      <div class="detail-section"><h3>Risk (CAPM)</h3><div class="detail-kv-grid" id="detail-risk-kvs"></div></div>
      <div class="detail-section"><h3>Quality &amp; Profitability</h3><div class="detail-kv-grid" id="detail-quality-kvs"></div></div>
      <div class="detail-section"><h3>Financial Health</h3><div class="detail-kv-grid" id="detail-health-kvs"></div></div>
      <div class="detail-section" style="grid-column:1/-1"><h3>Analyst &amp; Sentiment</h3><div class="detail-kv-grid" id="detail-sentiment-kvs"></div></div>
    </div>
    <div class="detail-charts">
      <div class="detail-chart-box"><div id="detail-chart-roic"></div></div>
      <div class="detail-chart-box"><div id="detail-chart-valuation"></div></div>
    </div>
  </div>
</div>

<script>
const DATA = {chart_data};

// --- Charts ---
const _cfg = {{responsive: true, displayModeBar: 'hover'}};
const _base = {{
  height: 415,
  margin: {{t: 60, b: 55, l: 65, r: 22}},
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: '#f8f9fb',
  font: {{family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', size: 12}}
}};

function getColVals(colKey, minV, maxV) {{
  const hdrs = Array.from(document.querySelectorAll('th'));
  const idx = hdrs.findIndex(h => h.dataset.col === colKey);
  if (idx < 0) return [];
  return Array.from(document.querySelectorAll('#data-table tbody tr')).map(r => {{
    const dv = r.querySelectorAll('td')[idx]?.dataset.value;
    return dv !== undefined && dv !== '' ? parseFloat(dv) : NaN;
  }}).filter(v => !isNaN(v) && isFinite(v) &&
    (minV === undefined || v >= minV) && (maxV === undefined || v <= maxV));
}}

function renderCharts(tab) {{
  if (tab === 'ovr' || tab === 'all') {{
    const valid = DATA.filter(d => d.roic != null && d.wacc != null);
    const tkrs = valid.map(d => d.ticker);
    const maxVal = Math.max(...valid.map(d => d.roic * 100), ...valid.map(d => d.wacc * 100), 50);
    Plotly.react('scatter-roic-wacc', [
      {{ x: valid.map(d => d.wacc * 100), y: valid.map(d => d.roic * 100),
         text: tkrs, mode: 'markers', type: 'scatter',
         marker: {{ size: 7, color: valid.map(d => (d.spread||0)*100),
                    colorscale: 'RdYlGn', colorbar: {{ title: 'Spread%' }}, opacity: 0.8 }},
         hovertemplate: '<b>%{{text}}</b><br>WACC: %{{x:.1f}}%<br>ROIC: %{{y:.1f}}%<extra></extra>' }},
      {{ x: [0, maxVal], y: [0, maxVal], mode: 'lines',
         line: {{ dash: 'dash', color: '#aaa' }}, showlegend: false }}
    ], Object.assign({{}}, _base, {{ title: 'ROIC vs WACC',
       xaxis: {{ title: 'WACC (%)' }}, yaxis: {{ title: 'ROIC (%)' }} }}), _cfg);

    const vd = DATA.filter(d => d.dcf_fv != null && d.price != null);
    if (vd.length > 0) {{
      const maxP = Math.max(...vd.map(d => Math.max(d.dcf_fv, d.price)));
      Plotly.react('scatter-dcf-price', [
        {{ x: vd.map(d => d.price), y: vd.map(d => d.dcf_fv),
           text: vd.map(d => d.ticker), mode: 'markers', type: 'scatter',
           marker: {{ size: 7, color: vd.map(d => (d.mos||0)*100),
                      colorscale: 'RdYlGn', cmin: -30, cmax: 30,
                      colorbar: {{ title: 'MoS%' }}, opacity: 0.8 }},
           hovertemplate: '<b>%{{text}}</b><br>Price: $%{{x:,.2f}}<br>DCF: $%{{y:,.2f}}<extra></extra>' }},
        {{ x: [0, maxP], y: [0, maxP], mode: 'lines',
           line: {{ dash: 'dash', color: '#aaa' }}, showlegend: false }}
      ], Object.assign({{}}, _base, {{ title: 'DCF Fair Value vs Current Price',
         xaxis: {{ title: 'Price ($)' }}, yaxis: {{ title: 'DCF FV ($)' }} }}), _cfg);
    }}

    const sorted = DATA.filter(d => d.spread != null).sort((a,b) => b.spread - a.spread).slice(0,20).reverse();
    Plotly.react('top-spread', [{{
      y: sorted.map(d => d.ticker), x: sorted.map(d => d.spread * 100),
      type: 'bar', orientation: 'h',
      marker: {{ color: '#2c6fad', opacity: 0.82 }},
      hovertemplate: '<b>%{{y}}</b><br>Spread: %{{x:.1f}}%<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Top 20 by ROIC-WACC Spread',
       xaxis: {{ title: 'Spread (%)' }}, margin: {{ t: 44, l: 60 }} }}), _cfg);

    const ratingOrder = ['BUY', 'LEAN BUY', 'HOLD', 'PASS'];
    const ratingColors = {{ 'BUY': '#1a9850', 'LEAN BUY': '#74c476', 'HOLD': '#fd8d3c', 'PASS': '#de2d26' }};
    const ratingCounts = ratingOrder.map(r => DATA.filter(d => d.rating === r).length);
    const recMapC = {{}};
    DATA.forEach(d => {{ if (d.analyst_rec) recMapC[d.analyst_rec] = (recMapC[d.analyst_rec]||0)+1; }});
    // Map analyst 5-point scale → model 4-point labels: strong_buy→BUY, buy→LEAN BUY, hold→HOLD, sell+strong_sell→PASS
    const analystMappedC = [
      recMapC['strong_buy'] || 0,
      recMapC['buy']        || 0,
      recMapC['hold']       || 0,
      (recMapC['sell'] || 0) + (recMapC['strong_sell'] || 0)
    ];
    Plotly.react('rating-chart', [
      {{ name: 'Model Rating', x: ratingOrder, y: ratingCounts,
         type: 'bar', marker: {{ color: '#1a252f' }},
         hovertemplate: '%{{x}}: %{{y}}<extra></extra>' }},
      {{ name: 'Analyst Consensus', x: ratingOrder, y: analystMappedC, type: 'bar',
         marker: {{ color: '#3182bd' }},
         hovertemplate: '%{{x}}: %{{y}}<extra></extra>' }}
    ], Object.assign({{}}, _base, {{ title: 'Model Rating vs Analyst Consensus',
       barmode: 'group', xaxis: {{ title: '' }}, yaxis: {{ title: 'Count' }},
       legend: {{ x: 0.7, y: 0.95 }} }}), _cfg);

  }} else if (tab === 'val') {{
    const vd2 = DATA.filter(d => d.dcf_fv != null && d.price != null);
    if (vd2.length > 0) {{
      const maxP2 = Math.max(...vd2.map(d => Math.max(d.dcf_fv, d.price)));
      Plotly.react('scatter-roic-wacc', [
        {{ x: vd2.map(d => d.price), y: vd2.map(d => d.dcf_fv),
           text: vd2.map(d => d.ticker), mode: 'markers', type: 'scatter',
           marker: {{ size: 7, color: vd2.map(d => (d.mos||0)*100),
                      colorscale: 'RdYlGn', cmin: -30, cmax: 30,
                      colorbar: {{ title: 'MoS%' }}, opacity: 0.8 }},
           hovertemplate: '<b>%{{text}}</b><br>Price: $%{{x:,.2f}}<br>DCF: $%{{y:,.2f}}<extra></extra>' }},
        {{ x: [0, maxP2], y: [0, maxP2], mode: 'lines',
           line: {{ dash: 'dash', color: '#aaa' }}, showlegend: false }}
      ], Object.assign({{}}, _base, {{ title: 'DCF Fair Value vs Current Price',
         xaxis: {{ title: 'Price ($)' }}, yaxis: {{ title: 'DCF FV ($)' }} }}), _cfg);
    }}

    const mosVals = DATA.filter(d => d.mos != null).map(d => d.mos * 100);
    Plotly.react('scatter-dcf-price', [{{
      x: mosVals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#2c6fad' }},
      hovertemplate: 'MoS: %{{x:.1f}}%<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Margin of Safety Distribution',
       xaxis: {{ title: 'MoS (%)' }}, yaxis: {{ title: 'Count' }},
       shapes: [{{ type: 'line', x0: 15, x1: 15, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dash', color: '#1a9850' }} }},
                {{ type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dash', color: '#de2d26' }} }}] }}), _cfg);

    const peVals = DATA.filter(d => d.pe != null && d.pe > 0 && d.pe < 150).map(d => d.pe);
    Plotly.react('top-spread', [{{
      x: peVals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#6b4c9a' }},
      hovertemplate: 'P/E: %{{x:.1f}}<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'P/E Ratio Distribution',
       xaxis: {{ title: 'P/E' }}, yaxis: {{ title: 'Count' }} }}), _cfg);

    const recMapV = {{}};
    DATA.forEach(d => {{ if (d.analyst_rec) recMapV[d.analyst_rec] = (recMapV[d.analyst_rec]||0)+1; }});
    const recOrderV = ['strong_buy','buy','hold','sell','strong_sell'];
    const recLabelsV = {{ strong_buy:'Strong Buy', buy:'Buy', hold:'Hold', sell:'Sell', strong_sell:'Strong Sell' }};
    const recColorsV = {{ strong_buy:'#1a9850', buy:'#74c476', hold:'#fd8d3c', sell:'#e6750a', strong_sell:'#de2d26' }};
    Plotly.react('rating-chart', [{{
      x: recOrderV.map(r => recLabelsV[r]||r),
      y: recOrderV.map(r => recMapV[r]||0),
      type: 'bar', marker: {{ color: recOrderV.map(r => recColorsV[r]||'#aaa') }},
      hovertemplate: '%{{x}}: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Analyst Consensus Distribution',
       xaxis: {{ title: '' }}, yaxis: {{ title: 'Count' }} }}), _cfg);

  }} else if (tab === 'risk') {{
    const rd = DATA.filter(d => d.beta != null && d.r2 != null);
    Plotly.react('scatter-roic-wacc', [{{
      x: rd.map(d => d.beta), y: rd.map(d => d.r2 * 100),
      text: rd.map(d => d.ticker), mode: 'markers', type: 'scatter',
      marker: {{ size: 7, color: rd.map(d => d.r2 * 100),
                 colorscale: 'Blues', cmin: 0, cmax: 100,
                 colorbar: {{ title: 'R²%' }}, opacity: 0.8 }},
      hovertemplate: '<b>%{{text}}</b><br>Beta: %{{x:.2f}}<br>R\u00b2: %{{y:.1f}}%<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Beta vs R\u00b2',
       xaxis: {{ title: 'Beta' }}, yaxis: {{ title: 'R\u00b2 (%)' }},
       shapes: [{{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 60, y1: 60,
                   line: {{ dash: 'dash', color: '#1a9850' }} }},
                {{ type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 40, y1: 40,
                   line: {{ dash: 'dot', color: '#fd8d3c' }} }}] }}), _cfg);

    const betaVals = DATA.filter(d => d.beta != null && Math.abs(d.beta) < 5).map(d => d.beta);
    Plotly.react('scatter-dcf-price', [{{
      x: betaVals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#6b4c9a' }},
      hovertemplate: 'Beta: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Beta Distribution',
       xaxis: {{ title: 'Beta' }}, yaxis: {{ title: 'Count' }},
       shapes: [{{ type: 'line', x0: 1, x1: 1, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dash', color: '#de2d26' }} }}] }}), _cfg);

    const r2Vals = DATA.filter(d => d.r2 != null).map(d => d.r2 * 100);
    Plotly.react('top-spread', [{{
      x: r2Vals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#3182bd' }},
      hovertemplate: 'R\u00b2: %{{x:.1f}}%<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'R\u00b2 Distribution',
       xaxis: {{ title: 'R\u00b2 (%)' }}, yaxis: {{ title: 'Count' }},
       shapes: [{{ type: 'line', x0: 60, x1: 60, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dash', color: '#1a9850' }} }},
                {{ type: 'line', x0: 40, x1: 40, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dot', color: '#fd8d3c' }} }}] }}), _cfg);

    const reliable = DATA.filter(d => d.r2 != null && d.r2 >= 0.60).length;
    const directional = DATA.filter(d => d.r2 != null && d.r2 >= 0.40 && d.r2 < 0.60).length;
    const unreliable = DATA.filter(d => d.r2 != null && d.r2 < 0.40).length;
    Plotly.react('rating-chart', [{{
      x: ['Reliable (R\u00b2\u226560%)', 'Directional (40\u201360%)', 'Unreliable (<40%)'],
      y: [reliable, directional, unreliable],
      type: 'bar', marker: {{ color: ['#1a9850', '#fd8d3c', '#de2d26'] }},
      hovertemplate: '%{{x}}: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'CAPM Beta Reliability',
       xaxis: {{ title: '' }}, yaxis: {{ title: 'Count' }} }}), _cfg);

  }} else if (tab === 'prof') {{
    const hdrs = Array.from(document.querySelectorAll('th'));
    const roeIdx = hdrs.findIndex(h => h.dataset.col === 'roe');
    const roaIdx = hdrs.findIndex(h => h.dataset.col === 'roa');
    const tkrIdx = hdrs.findIndex(h => h.dataset.col === 'ticker');
    const fRows = Array.from(document.querySelectorAll('#data-table tbody tr')).map(r => {{
      const tds = r.querySelectorAll('td');
      const roe = roeIdx >= 0 ? parseFloat(tds[roeIdx]?.dataset.value) : NaN;
      const roa = roaIdx >= 0 ? parseFloat(tds[roaIdx]?.dataset.value) : NaN;
      const t = tkrIdx >= 0 ? (tds[tkrIdx]?.textContent||'').trim() : '';
      return {{t, roe: roe * 100, roa: roa * 100}};
    }}).filter(d => !isNaN(d.roe) && !isNaN(d.roa) && isFinite(d.roe) && isFinite(d.roa));
    Plotly.react('scatter-roic-wacc', [{{
      x: fRows.map(d => d.roa), y: fRows.map(d => d.roe),
      text: fRows.map(d => d.t), mode: 'markers', type: 'scatter',
      marker: {{ size: 7, color: fRows.map(d => d.roe),
                 colorscale: 'RdYlGn', colorbar: {{ title: 'ROE%' }}, opacity: 0.8 }},
      hovertemplate: '<b>%{{text}}</b><br>ROA: %{{x:.1f}}%<br>ROE: %{{y:.1f}}%<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'ROE vs ROA',
       xaxis: {{ title: 'ROA (%)' }}, yaxis: {{ title: 'ROE (%)' }} }}), _cfg);

    const ccVals = getColVals('cash_conv', -5, 20);
    Plotly.react('scatter-dcf-price', [{{
      x: ccVals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#2a7a4b' }},
      hovertemplate: 'CashConv: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Cash Conversion (OCF / Net Income)',
       xaxis: {{ title: 'OCF / Net Income' }}, yaxis: {{ title: 'Count' }},
       shapes: [{{ type: 'line', x0: 1, x1: 1, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dash', color: '#1a9850' }} }}] }}), _cfg);

    const accrVals = getColVals('accruals', -1, 1);
    Plotly.react('top-spread', [{{
      x: accrVals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#8b5018' }},
      hovertemplate: 'Accruals: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Accruals Ratio Distribution (Sloan)',
       xaxis: {{ title: '(Net Income - OCF) / Avg Total Assets' }}, yaxis: {{ title: 'Count' }},
       shapes: [{{ type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dash', color: '#555' }} }}] }}), _cfg);

    const cagrVals = getColVals('rev_cagr', -0.5, 1.0).map(v => v * 100);
    Plotly.react('rating-chart', [{{
      x: cagrVals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#2a7a4b' }},
      hovertemplate: 'Rev CAGR: %{{x:.1f}}%<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: '5-Year Revenue CAGR Distribution',
       xaxis: {{ title: 'CAGR (%)' }}, yaxis: {{ title: 'Count' }} }}), _cfg);

  }} else if (tab === 'hlth') {{
    const pfCounts = Array.from({{length: 10}}, (_, i) => DATA.filter(d => d.piotroski === i).length);
    const pfColors = Array.from({{length: 10}}, (_, i) => i >= 7 ? '#1a9850' : i <= 3 ? '#de2d26' : '#fd8d3c');
    Plotly.react('scatter-roic-wacc', [{{
      x: Array.from({{length: 10}}, (_, i) => i),
      y: pfCounts, type: 'bar',
      marker: {{ color: pfColors }},
      hovertemplate: 'F-Score %{{x}}: %{{y}} companies<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Piotroski F-Score Distribution',
       xaxis: {{ title: 'F-Score (0\u20139)', dtick: 1 }}, yaxis: {{ title: 'Count' }} }}), _cfg);

    const azSafe = DATA.filter(d => d.altman_z != null && d.altman_z > 2.99).length;
    const azGrey = DATA.filter(d => d.altman_z != null && d.altman_z >= 1.81 && d.altman_z <= 2.99).length;
    const azDistress = DATA.filter(d => d.altman_z != null && d.altman_z < 1.81).length;
    Plotly.react('scatter-dcf-price', [{{
      x: ['Safe (>2.99)', 'Grey (1.81\u20132.99)', 'Distress (<1.81)'],
      y: [azSafe, azGrey, azDistress], type: 'bar',
      marker: {{ color: ['#1a9850', '#fd8d3c', '#de2d26'] }},
      hovertemplate: '%{{x}}: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Altman Z-Score Zones',
       xaxis: {{ title: '' }}, yaxis: {{ title: 'Count' }} }}), _cfg);

    const intCovVals = getColVals('int_cov', -10, 100);
    Plotly.react('top-spread', [{{
      x: intCovVals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#3182bd' }},
      hovertemplate: 'Int Cov: %{{x:.1f}}x<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Interest Coverage Distribution',
       xaxis: {{ title: 'EBIT / Interest Expense (x)' }}, yaxis: {{ title: 'Count' }},
       shapes: [{{ type: 'line', x0: 3, x1: 3, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dash', color: '#1a9850' }} }}] }}), _cfg);

    const ndVals = getColVals('nd_ebitda', -10, 20);
    Plotly.react('rating-chart', [{{
      x: ndVals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#7b4a1e' }},
      hovertemplate: 'ND/EBITDA: %{{x:.1f}}x<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Net Debt / EBITDA Distribution',
       xaxis: {{ title: 'Net Debt / EBITDA (x)' }}, yaxis: {{ title: 'Count' }},
       shapes: [{{ type: 'line', x0: 3, x1: 3, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dash', color: '#de2d26' }} }}] }}), _cfg);

  }} else if (tab === 'con') {{
    const ratingOrder2 = ['BUY', 'LEAN BUY', 'HOLD', 'PASS'];
    const ratingColors2 = {{ 'BUY': '#1a9850', 'LEAN BUY': '#74c476', 'HOLD': '#fd8d3c', 'PASS': '#de2d26' }};
    const ratingCounts2 = ratingOrder2.map(r => DATA.filter(d => d.rating === r).length);
    Plotly.react('scatter-roic-wacc', [{{
      x: ratingOrder2, y: ratingCounts2, type: 'bar',
      marker: {{ color: ratingOrder2.map(r => ratingColors2[r]) }},
      hovertemplate: '%{{x}}: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'Model Rating Distribution',
       xaxis: {{ title: '' }}, yaxis: {{ title: 'Count' }} }}), _cfg);

    const recMapR = {{}};
    DATA.forEach(d => {{ if (d.analyst_rec) recMapR[d.analyst_rec] = (recMapR[d.analyst_rec]||0)+1; }});
    // Map analyst 5-point scale → model 4-point labels: strong_buy→BUY, buy→LEAN BUY, hold→HOLD, sell+strong_sell→PASS
    const analystMappedR = [
      recMapR['strong_buy'] || 0,
      recMapR['buy']        || 0,
      recMapR['hold']       || 0,
      (recMapR['sell'] || 0) + (recMapR['strong_sell'] || 0)
    ];
    Plotly.react('scatter-dcf-price', [
      {{ name: 'Model Rating', x: ratingOrder2, y: ratingCounts2, type: 'bar',
         marker: {{ color: '#1a252f' }},
         hovertemplate: '%{{x}}: %{{y}}<extra></extra>' }},
      {{ name: 'Analyst Consensus', x: ratingOrder2, y: analystMappedR, type: 'bar',
         marker: {{ color: '#3182bd' }},
         hovertemplate: '%{{x}}: %{{y}}<extra></extra>' }}
    ], Object.assign({{}}, _base, {{ title: 'Model Rating vs Analyst Consensus',
       barmode: 'group', xaxis: {{ title: '' }}, yaxis: {{ title: 'Count' }},
       legend: {{ x: 0.7, y: 0.95 }} }}), _cfg);

    const sentVals = DATA.filter(d => d.sentiment_score != null).map(d => d.sentiment_score);
    Plotly.react('top-spread', [{{
      x: sentVals, type: 'histogram', nbinsx: 20,
      marker: {{ color: '#3d6f9e' }},
      hovertemplate: 'Score: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{ title: 'News Sentiment Distribution',
       xaxis: {{ title: 'VADER Score (-1 to +1)' }}, yaxis: {{ title: 'Count' }},
       shapes: [{{ type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
                   line: {{ dash: 'dash', color: '#555' }} }}] }}), _cfg);

    const socialVals = DATA.filter(d => d.social_score != null).map(d => d.social_score);
    if (socialVals.length > 0) {{
      Plotly.react('rating-chart', [{{
        x: socialVals, type: 'histogram', nbinsx: 20,
        marker: {{ color: '#7a3d6b' }},
        hovertemplate: 'Score: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>'
      }}], Object.assign({{}}, _base, {{ title: 'Social Sentiment Distribution',
         xaxis: {{ title: 'Composite Score (-1 to +1)' }}, yaxis: {{ title: 'Count' }},
         shapes: [{{ type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
                     line: {{ dash: 'dash', color: '#555' }} }}] }}), _cfg);
    }} else {{
      Plotly.react('rating-chart', [{{
        x: ['No Data'], y: [0], type: 'bar', marker: {{ color: ['#ccc'] }}
      }}], Object.assign({{}}, _base, {{ title: 'Social Sentiment (No Data)',
         xaxis: {{ title: '' }}, yaxis: {{ title: 'Count' }} }}), _cfg);
    }}

  }} else if (tab === 'proj') {{
    // --- Chart 1: DCF Fair Value Range (Bear / Base / Bull) vs Current Price ---
    const pTop = DATA.filter(d => d.dcf_fv != null && d.price != null && d.spread != null)
                     .sort((a,b) => b.spread - a.spread).slice(0, 15);
    const tkPr = pTop.map(d => d.ticker);
    const bDCF = pTop.map(d => +(d.dcf_fv * 0.70).toFixed(2));
    const mDCF = pTop.map(d => +(d.dcf_fv       ).toFixed(2));
    const uDCF = pTop.map(d => +(d.dcf_fv * 1.35).toFixed(2));
    const xFill1 = [...tkPr, ...tkPr.slice().reverse()];
    const yFill1 = [...uDCF, ...bDCF.slice().reverse()];
    Plotly.react('scatter-roic-wacc', [
      {{ x: xFill1, y: yFill1, type: 'scatter', fill: 'toself',
         fillcolor: 'rgba(44,111,173,0.13)', line: {{color:'transparent'}},
         showlegend: false, hoverinfo: 'skip' }},
      {{ name: 'Bear (−30%)', x: tkPr, y: bDCF, type: 'scatter', mode: 'lines+markers',
         line: {{color:'#de2d26',width:2,dash:'dash'}}, marker:{{size:5}},
         hovertemplate: '<b>%{{x}}</b><br>Bear DCF: $%{{y:,.2f}}<extra></extra>' }},
      {{ name: 'Base DCF',    x: tkPr, y: mDCF, type: 'scatter', mode: 'lines+markers',
         line: {{color:'#2980b9',width:3}}, marker:{{size:6}},
         hovertemplate: '<b>%{{x}}</b><br>Base DCF: $%{{y:,.2f}}<extra></extra>' }},
      {{ name: 'Bull (+35%)', x: tkPr, y: uDCF, type: 'scatter', mode: 'lines+markers',
         line: {{color:'#1a9850',width:2,dash:'dot'}}, marker:{{size:5}},
         hovertemplate: '<b>%{{x}}</b><br>Bull DCF: $%{{y:,.2f}}<extra></extra>' }},
      {{ name: 'Current Price', x: tkPr, y: pTop.map(d=>+d.price.toFixed(2)),
         type: 'scatter', mode: 'markers',
         marker: {{color:'#e6750a',size:11,symbol:'diamond'}},
         hovertemplate: '<b>%{{x}}</b><br>Price: $%{{y:,.2f}}<extra></extra>' }},
    ], Object.assign({{}}, _base, {{ title: 'DCF Fair Value — Bear / Base / Bull vs Current Price (Top 15 by ROIC−WACC Spread)',
       xaxis: {{title:'',tickangle:-30}}, yaxis: {{title:'$ Per Share'}},
       legend: {{x:0.01,y:0.99}} }}), _cfg);

    // --- Chart 2: Price Projection at 2yr / 5yr / 10yr with Bear–Bull Error Bars ---
    const ppTop = DATA.filter(d => d.er != null && d.er > 0 && d.price != null && d.spread != null)
                      .sort((a,b) => b.spread - a.spread).slice(0, 15);
    const tkPP = ppTop.map(d => d.ticker);
    const mk2  = ppTop.map(d => +(d.price * Math.pow(1 + d.er,       2)).toFixed(2));
    const mk5  = ppTop.map(d => +(d.price * Math.pow(1 + d.er,       5)).toFixed(2));
    const mk10 = ppTop.map(d => +(d.price * Math.pow(1 + d.er,      10)).toFixed(2));
    const bk2  = ppTop.map(d => +(d.price * Math.pow(1 + d.er * 0.6, 2)).toFixed(2));
    const bk5  = ppTop.map(d => +(d.price * Math.pow(1 + d.er * 0.6, 5)).toFixed(2));
    const bk10 = ppTop.map(d => +(d.price * Math.pow(1 + d.er * 0.6,10)).toFixed(2));
    const uk2  = ppTop.map(d => +(d.price * Math.pow(1 + d.er * 1.4, 2)).toFixed(2));
    const uk5  = ppTop.map(d => +(d.price * Math.pow(1 + d.er * 1.4, 5)).toFixed(2));
    const uk10 = ppTop.map(d => +(d.price * Math.pow(1 + d.er * 1.4,10)).toFixed(2));
    Plotly.react('scatter-dcf-price', [
      {{ name: '2-Year', x: tkPP, y: mk2, type: 'bar',
         marker: {{color:'rgba(44,111,173,0.70)'}},
         error_y: {{type:'data', array: mk2.map((v,i)=>uk2[i]-v), arrayminus: mk2.map((v,i)=>v-bk2[i]),
                    visible:true, color:'#2980b9', thickness:2, width:4}},
         hovertemplate: '<b>%{{x}}</b><br>2yr base: $%{{y:,.0f}}<extra></extra>' }},
      {{ name: '5-Year', x: tkPP, y: mk5, type: 'bar',
         marker: {{color:'rgba(26,152,80,0.70)'}},
         error_y: {{type:'data', array: mk5.map((v,i)=>uk5[i]-v), arrayminus: mk5.map((v,i)=>v-bk5[i]),
                    visible:true, color:'#1a9850', thickness:2, width:4}},
         hovertemplate: '<b>%{{x}}</b><br>5yr base: $%{{y:,.0f}}<extra></extra>' }},
      {{ name: '10-Year', x: tkPP, y: mk10, type: 'bar',
         marker: {{color:'rgba(107,76,154,0.70)'}},
         error_y: {{type:'data', array: mk10.map((v,i)=>uk10[i]-v), arrayminus: mk10.map((v,i)=>v-bk10[i]),
                    visible:true, color:'#8e44ad', thickness:2, width:4}},
         hovertemplate: '<b>%{{x}}</b><br>10yr base: $%{{y:,.0f}}<extra></extra>' }},
      {{ name: 'Current Price', x: tkPP, y: ppTop.map(d=>d.price),
         type:'scatter', mode:'markers', marker:{{color:'#e6750a',size:10,symbol:'diamond'}},
         hovertemplate: '<b>%{{x}}</b><br>Now: $%{{y:,.2f}}<extra></extra>' }},
    ], Object.assign({{}}, _base, {{ title: 'Price Projection at 2 / 5 / 10 Years — Bear–Bull Ranges (CAPM Expected Return)',
       barmode: 'group', xaxis: {{title:'',tickangle:-30}}, yaxis: {{title:'Projected Price ($)'}},
       legend: {{x:0.01,y:0.99}} }}), _cfg);

    // --- Chart 3: Compounded Return Fan — Portfolio Average, 0–10 Years ---
    const erVals = DATA.filter(d => d.er != null && d.er > 0).map(d => d.er);
    const avgEr  = erVals.length ? erVals.reduce((s,v)=>s+v,0)/erVals.length : 0.10;
    const yrs    = [0,1,2,3,4,5,6,7,8,9,10];
    const bearY  = yrs.map(t => +Math.pow(1 + avgEr * 0.6, t).toFixed(4));
    const baseY  = yrs.map(t => +Math.pow(1 + avgEr,       t).toFixed(4));
    const bullY  = yrs.map(t => +Math.pow(1 + avgEr * 1.4, t).toFixed(4));
    const xFan   = [...yrs, ...yrs.slice().reverse()];
    const yFan   = [...bullY, ...bearY.slice().reverse()];
    Plotly.react('top-spread', [
      {{ x: xFan, y: yFan, type: 'scatter', fill: 'toself',
         fillcolor: 'rgba(100,150,200,0.13)', line: {{color:'transparent'}},
         showlegend: false, hoverinfo: 'skip' }},
      {{ name: 'Bear (0.6× Er)', x: yrs, y: bearY, type: 'scatter', mode: 'lines+markers',
         line: {{color:'#de2d26',width:2,dash:'dash'}}, marker:{{size:5}},
         hovertemplate: 'Year %{{x}}: %{{y:.2f}}× <extra>Bear</extra>' }},
      {{ name: 'Base (Er)',      x: yrs, y: baseY, type: 'scatter', mode: 'lines+markers',
         line: {{color:'#2980b9',width:3}}, marker:{{size:6}},
         hovertemplate: 'Year %{{x}}: %{{y:.2f}}× <extra>Base</extra>' }},
      {{ name: 'Bull (1.4× Er)',x: yrs, y: bullY, type: 'scatter', mode: 'lines+markers',
         line: {{color:'#1a9850',width:2,dash:'dot'}}, marker:{{size:5}},
         hovertemplate: 'Year %{{x}}: %{{y:.2f}}× <extra>Bull</extra>' }},
    ], Object.assign({{}}, _base, {{
       title: `Portfolio Compounded Return Fan — Avg CAPM Er: ${{(avgEr*100).toFixed(1)}}% — Shaded Region = Bear–Bull Range`,
       xaxis: {{title:'Year (0 = Today)', dtick:1}},
       yaxis: {{title:'Return Multiple (1.0 = Break Even)'}},
       legend: {{x:0.05,y:0.95}},
       shapes: [
         {{type:'line',x0:2, x1:2, y0:0,y1:1,yref:'paper',line:{{dash:'dot',color:'#aaa',width:1}}}},
         {{type:'line',x0:5, x1:5, y0:0,y1:1,yref:'paper',line:{{dash:'dot',color:'#aaa',width:1}}}},
         {{type:'line',x0:10,x1:10,y0:0,y1:1,yref:'paper',line:{{dash:'dot',color:'#aaa',width:1}}}}
       ],
       annotations: [
         {{x:2, y:0.02,yref:'paper',text:'2yr',showarrow:false,font:{{size:10,color:'#aaa'}}}},
         {{x:5, y:0.02,yref:'paper',text:'5yr',showarrow:false,font:{{size:10,color:'#aaa'}}}},
         {{x:10,y:0.02,yref:'paper',text:'10yr',showarrow:false,font:{{size:10,color:'#aaa'}}}}
       ]
    }}), _cfg);

    // --- Chart 4: Revenue Growth Projection at 2 / 5 / 10 Years with Bear–Bull Error Bars ---
    const hdrsJ = Array.from(document.querySelectorAll('th'));
    const cagrI = hdrsJ.findIndex(h => h.dataset.col === 'rev_cagr');
    const tkrI2 = hdrsJ.findIndex(h => h.dataset.col === 'ticker');
    const cagrRows = Array.from(document.querySelectorAll('#data-table tbody tr')).map(r => {{
      const tds  = r.querySelectorAll('td');
      const cagr = cagrI  >= 0 ? parseFloat(tds[cagrI ]?.dataset.value) : NaN;
      const t    = tkrI2 >= 0 ? (tds[tkrI2]?.textContent||'').trim() : '';
      return {{t, cagr}};
    }}).filter(d => !isNaN(d.cagr) && isFinite(d.cagr) && d.cagr > 0)
       .sort((a,b) => b.cagr - a.cagr).slice(0, 12);
    const tkRev = cagrRows.map(d => d.t);
    function revGain(cagr, scale, n) {{ return +((Math.pow(1 + Math.min(0.50, Math.max(0, cagr * scale)), n) - 1) * 100).toFixed(1); }}
    const r2m  = cagrRows.map(d => revGain(d.cagr, 1.0, 2));
    const r5m  = cagrRows.map(d => revGain(d.cagr, 1.0, 5));
    const r10m = cagrRows.map(d => revGain(d.cagr, 1.0,10));
    const r2b  = cagrRows.map(d => revGain(d.cagr, 0.5, 2));
    const r5b  = cagrRows.map(d => revGain(d.cagr, 0.5, 5));
    const r10b = cagrRows.map(d => revGain(d.cagr, 0.5,10));
    const r2u  = cagrRows.map(d => revGain(d.cagr, 1.5, 2));
    const r5u  = cagrRows.map(d => revGain(d.cagr, 1.5, 5));
    const r10u = cagrRows.map(d => revGain(d.cagr, 1.5,10));
    Plotly.react('rating-chart', [
      {{ name: '2-Year Rev Growth', x: tkRev, y: r2m, type: 'bar',
         marker: {{color:'rgba(44,111,173,0.70)'}},
         error_y: {{type:'data', array: r2m.map((v,i)=>r2u[i]-v), arrayminus: r2m.map((v,i)=>v-r2b[i]),
                    visible:true, color:'#2980b9', thickness:2, width:4}},
         hovertemplate: '<b>%{{x}}</b><br>2yr base: %{{y:.1f}}%<extra></extra>' }},
      {{ name: '5-Year Rev Growth', x: tkRev, y: r5m, type: 'bar',
         marker: {{color:'rgba(26,152,80,0.70)'}},
         error_y: {{type:'data', array: r5m.map((v,i)=>r5u[i]-v), arrayminus: r5m.map((v,i)=>v-r5b[i]),
                    visible:true, color:'#1a9850', thickness:2, width:4}},
         hovertemplate: '<b>%{{x}}</b><br>5yr base: %{{y:.1f}}%<extra></extra>' }},
      {{ name: '10-Year Rev Growth', x: tkRev, y: r10m, type: 'bar',
         marker: {{color:'rgba(107,76,154,0.70)'}},
         error_y: {{type:'data', array: r10m.map((v,i)=>r10u[i]-v), arrayminus: r10m.map((v,i)=>v-r10b[i]),
                    visible:true, color:'#8e44ad', thickness:2, width:4}},
         hovertemplate: '<b>%{{x}}</b><br>10yr base: %{{y:.1f}}%<extra></extra>' }},
    ], Object.assign({{}}, _base, {{ title: 'Revenue Cumulative Growth at 2 / 5 / 10 Years — Bear–Bull Ranges (Top 12 by Rev CAGR)',
       barmode: 'group', xaxis: {{title:'',tickangle:-30}}, yaxis: {{title:'Cumulative Revenue Growth (%)'}},
       legend: {{x:0.01,y:0.99}} }}), _cfg);
  }}

// --- Tab switcher ---
const GROUPS = ['ovr','val','risk','prof','hlth','con','proj'];
function showTab(tab) {{
  // Reset all grouped elements first
  GROUPS.forEach(g => {{
    document.querySelectorAll('.grp-' + g).forEach(el => {{ el.style.display = 'none'; }});
  }});
  // Show elements belonging to the active tab (handles multi-class elements correctly)
  const activeGroups = tab === 'all' ? GROUPS : [tab];
  activeGroups.forEach(g => {{
    document.querySelectorAll('.grp-' + g).forEach(el => {{ el.style.display = ''; }});
  }});
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelector('[data-tab="' + tab + '"]').classList.add('active');
  renderCharts(tab);
  const _tabMeta = {{
    ovr:  ['Overview',          'Capital efficiency screening — ROIC vs WACC spread, DCF vs price, top-spread ranking, and model vs analyst rating breakdown.',  '#1a252f'],
    val:  ['Valuation',         'Valuation multiples — DCF fair value vs price, margin of safety distribution, P/E, and analyst consensus.',                       '#1a3a5c'],
    risk: ['Risk',              'CAPM risk metrics — Beta vs R², beta and R² distributions, and regression reliability tiers.',                                    '#2c1a5c'],
    prof: ['Profitability',     'Earnings quality — ROE vs ROA scatter, cash conversion ratio, Sloan accruals, and 5-year revenue CAGR.',                          '#1a4a2c'],
    hlth: ['Financial Health',  'Balance-sheet strength — Piotroski F-score, Altman Z-score zones, interest coverage, and net debt/EBITDA.',                       '#3a2a10'],
    con:  ['Consensus',         'Market sentiment — model rating distribution, model vs analyst comparison, and news/social VADER sentiment scores.',               '#4a1a2a'],
    proj: ['Projections',       'Forward projections — DCF bear/base/bull range, CAPM price targets at 2/5/10 yr, compound return fan, and revenue growth.',       '#164a4a'],
    all:  ['All Columns',       'Full table view showing all available columns across every category.',                                                             '#444'],
  }};
  const _m = _tabMeta[tab] || [tab, '', '#2980b9'];
  document.getElementById('chart-tab-label').textContent = _m[0];
  document.getElementById('chart-tab-desc').textContent  = _m[1];
  document.getElementById('chart-section-hdr').style.borderLeftColor = _m[2];
}}
// Default: show overview
showTab('ovr');

// --- Sortable table ---
let _sortCol = null, _sortAsc = true;
function sortCol(colKey) {{
  if (_sortCol === colKey) _sortAsc = !_sortAsc;
  else {{ _sortCol = colKey; _sortAsc = true; }}
  const headers = Array.from(document.querySelectorAll('th'));
  const colIdx = headers.findIndex(h => h.dataset.col === colKey);
  if (colIdx < 0) return;
  const tbody = document.querySelector('#data-table tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {{
    const tdA = a.querySelectorAll('td')[colIdx];
    const tdB = b.querySelectorAll('td')[colIdx];
    if (!tdA || !tdB) return 0;
    const dvA = tdA.dataset.value, dvB = tdB.dataset.value;
    // Numeric sort when both cells carry a data-value
    if (dvA !== undefined && dvA !== '' && dvB !== undefined && dvB !== '') {{
      const va = parseFloat(dvA), vb = parseFloat(dvB);
      if (!isNaN(va) && !isNaN(vb)) return _sortAsc ? va - vb : vb - va;
    }}
    // Text sort — push N/A to bottom regardless of direction
    const ta = (tdA.textContent || '').trim();
    const tb = (tdB.textContent || '').trim();
    const naA = ta === 'N/A' || ta === '', naB = tb === 'N/A' || tb === '';
    if (naA && naB) return 0;
    if (naA) return 1;
    if (naB) return -1;
    return _sortAsc ? ta.localeCompare(tb) : tb.localeCompare(ta);
  }});
  rows.forEach(r => tbody.appendChild(r));
  document.querySelectorAll('th .arrow').forEach(a => a.textContent = '');
  const activeHeader = document.querySelector('th[data-col="' + colKey + '"]');
  if (activeHeader?.querySelector('.arrow'))
    activeHeader.querySelector('.arrow').textContent = _sortAsc ? '\\u25B2' : '\\u25BC';
}}

// --- Column visibility ---
function toggleColPanel() {{
  document.getElementById('col-panel').classList.toggle('open');
}}
document.addEventListener('click', function(e) {{
  if (!e.target.closest('.col-filter')) {{
    document.getElementById('col-panel').classList.remove('open');
  }}
}});
function toggleColumn(colKey, visible) {{
  const headers = Array.from(document.querySelectorAll('th'));
  const colIdx = headers.findIndex(h => h.dataset.col === colKey);
  if (colIdx < 0) return;
  headers[colIdx].style.display = visible ? '' : 'none';
  document.querySelectorAll('#data-table tbody tr').forEach(function(row) {{
    const td = row.querySelectorAll('td')[colIdx];
    if (td) td.style.display = visible ? '' : 'none';
  }});
}}
function setGroup(colKeys, visible) {{
  colKeys.forEach(function(k) {{
    const cb = document.querySelector('input[data-colkey="' + k + '"]');
    if (cb) {{ cb.checked = visible; toggleColumn(k, visible); }}
  }});
}}

// --- Search ---
function filterTable() {{
  const q = document.getElementById('search').value.toUpperCase();
  document.querySelectorAll('#data-table tbody tr').forEach(r => {{
    r.style.display = r.querySelector('.ticker')?.textContent.toUpperCase().includes(q) ? '' : 'none';
  }});
}}

// --- Detail Modal ---
function _fmtPct(v) {{ return v != null ? (v * 100).toFixed(1) + '%' : 'N/A'; }}
function _fmtNum(v, dec) {{ return v != null ? v.toFixed(dec || 2) : 'N/A'; }}
function _fmtDollar(v) {{ return v != null ? '$' + Math.round(v).toLocaleString() : 'N/A'; }}
function _fmtDollarShort(v) {{
  if (v == null) return 'N/A';
  if (Math.abs(v) >= 1e12) return '$' + (v/1e12).toFixed(1) + 'T';
  if (Math.abs(v) >= 1e9)  return '$' + (v/1e9).toFixed(1) + 'B';
  if (Math.abs(v) >= 1e6)  return '$' + (v/1e6).toFixed(1) + 'M';
  return '$' + Math.round(v).toLocaleString();
}}
function _buildKV(containerId, pairs) {{
  const el = document.getElementById(containerId);
  el.innerHTML = pairs.map(function(p) {{
    var label = p[0], value = p[1], style = p[2] || '';
    return '<div class="detail-kv"><span class="kv-label">' + label +
      '</span><span class="kv-value"' + (style ? ' style="'+style+'"' : '') + '>' +
      value + '</span></div>';
  }}).join('');
}}

document.querySelector('#data-table').addEventListener('click', function(e) {{
  var tickerCell = e.target.closest('td.ticker');
  if (!tickerCell) return;
  var ticker = tickerCell.textContent.trim();
  var item = DATA.find(function(d) {{ return d.ticker === ticker; }});
  if (item) openDetail(item);
}});

function openDetail(d) {{
  document.getElementById('detail-ticker').textContent = d.ticker;
  var rEl = document.getElementById('detail-rating');
  rEl.textContent = d.rating || 'N/A';
  var rc = {{'BUY':'#1a9850','LEAN BUY':'#74c476','HOLD':'#fd8d3c','PASS':'#de2d26'}};
  rEl.style.background = rc[d.rating] || '#888';

  document.getElementById('detail-price').textContent = d.price != null ? '$' + d.price.toLocaleString(undefined,{{maximumFractionDigits:2}}) : 'N/A';
  document.getElementById('detail-mcap').textContent = _fmtDollarShort(d.mcap);

  var meta = [];
  if (d.ceo_bio || d.ceo) meta.push(d.ceo_bio || d.ceo);
  if (d.sector) meta.push(d.sector);
  if (d.industry) meta.push(d.industry);
  document.getElementById('detail-meta').textContent = meta.join('  \u00b7  ');
  document.getElementById('detail-description').textContent = d.description_full || d.description || '';

  // Valuation
  var mosStyle = d.mos != null ? (d.mos > 0.15 ? 'color:#1a9850;font-weight:700' : (d.mos < 0 ? 'color:#de2d26' : '')) : '';
  var sensRange = d.dcf_sens_range ? _fmtDollar(d.dcf_sens_range[0]) + ' \u2013 ' + _fmtDollar(d.dcf_sens_range[1]) : 'N/A';
  _buildKV('detail-val-kvs', [
    ['DCF Fair Value', _fmtDollar(d.dcf_fv)],
    ['Current Price', _fmtDollar(d.price)],
    ['Margin of Safety', _fmtPct(d.mos), mosStyle],
    ['DCF Sensitivity Range', sensRange],
    ['FCF Growth Rate', _fmtPct(d.fcf_growth)],
    ['P/E', _fmtNum(d.pe, 1)],
    ['EV/EBITDA', _fmtNum(d.ev_ebitda, 1)],
    ['P/FCF', _fmtNum(d.pfcf, 1)],
    ['P/B', _fmtNum(d.pb, 2)],
    ['PEG', _fmtNum(d.peg, 2)],
    ['Analyst Target (Mean)', _fmtDollar(d.target_mean)],
    ['Analyst Target Range', d.target_low != null ? _fmtDollar(d.target_low) + ' \u2013 ' + _fmtDollar(d.target_high) : 'N/A']
  ]);

  // Risk
  _buildKV('detail-risk-kvs', [
    ['Adj. Beta', _fmtNum(d.beta, 2)],
    ['Raw Beta', _fmtNum(d.raw_beta, 2)],
    ['R-squared', _fmtPct(d.r2)],
    ['Jensen Alpha', _fmtPct(d.alpha)],
    ['Residual Sigma', _fmtPct(d.residual_sigma)],
    ['SE(Beta)', _fmtNum(d.se_beta, 3)],
    ['Observations', d.n_observations != null ? d.n_observations.toString() : 'N/A'],
    ['Expected Return', _fmtPct(d.er)],
    ['Cost-of-Equity Method', d.re_method || 'N/A']
  ]);

  // Quality
  var pfStyle = d.piotroski != null ? (d.piotroski >= 7 ? 'color:#1a9850;font-weight:700' : (d.piotroski <= 3 ? 'color:#de2d26;font-weight:700' : '')) : '';
  _buildKV('detail-quality-kvs', [
    ['ROIC (5Y Avg)', _fmtPct(d.roic)],
    ['WACC', _fmtPct(d.wacc)],
    ['Spread (ROIC\u2212WACC)', _fmtPct(d.spread)],
    ['ROE', _fmtPct(d.roe)],
    ['ROA', _fmtPct(d.roa)],
    ['Cash Conversion', _fmtNum(d.cash_conv, 2)],
    ['Accruals Ratio', _fmtNum(d.accruals, 3)],
    ['Rev CAGR (5Y)', _fmtPct(d.rev_cagr)],
    ['Piotroski F-Score', d.piotroski != null ? d.piotroski + '/9' : 'N/A', pfStyle]
  ]);

  // Health
  var azStyle = d.altman_z != null ? (d.altman_z > 2.99 ? 'color:#1a9850;font-weight:700' : (d.altman_z < 1.81 ? 'color:#de2d26;font-weight:700' : 'color:#fd8d3c')) : '';
  _buildKV('detail-health-kvs', [
    ['Altman Z-Score', _fmtNum(d.altman_z, 2), azStyle],
    ['Interest Coverage', _fmtNum(d.int_cov, 1)],
    ['Net Debt / EBITDA', _fmtNum(d.nd_ebitda, 2)],
    ['Debt / Equity', _fmtNum(d.de, 2)],
    ['Current Ratio', _fmtNum(d.cr, 2)]
  ]);

  // Sentiment
  var sentStyle = d.sentiment_label === 'Positive' ? 'color:#1a9850;font-weight:600' : (d.sentiment_label === 'Negative' ? 'color:#de2d26;font-weight:600' : '');
  var recLabels = {{strong_buy:'Strong Buy',buy:'Buy',hold:'Hold',sell:'Sell',strong_sell:'Strong Sell'}};
  var recLabel = recLabels[d.analyst_rec] || d.analyst_rec || 'N/A';
  _buildKV('detail-sentiment-kvs', [
    ['Analyst Consensus', recLabel + (d.num_analysts ? ' (' + d.num_analysts + ' analysts)' : '')],
    ['News Sentiment', (d.sentiment_label || 'N/A') + (d.sentiment_score != null ? ' (' + d.sentiment_score.toFixed(2) + ')' : ''), sentStyle],
    ['News Articles', d.sentiment_articles != null ? d.sentiment_articles.toString() : 'N/A'],
    ['News Bullish %', d.sentiment_bull != null ? (d.sentiment_bull * 100).toFixed(0) + '%' : 'N/A'],
    ['News Bearish %', d.sentiment_bear != null ? (d.sentiment_bear * 100).toFixed(0) + '%' : 'N/A'],
    ['Social Score', d.social_score != null ? d.social_score.toFixed(2) : 'N/A'],
    ['StockTwits Bull/Bear', d.st_bull_pct != null ? (d.st_bull_pct*100).toFixed(0)+'%/'+(d.st_bear_pct*100).toFixed(0)+'% ('+d.st_labeled+')' : 'N/A'],
    ['Reddit Score', d.reddit_score != null ? d.reddit_score.toFixed(2) + ' (' + (d.reddit_posts||0) + ' posts)' : 'N/A']
  ]);

  // Charts
  renderDetailCharts(d);

  document.getElementById('detail-modal').classList.add('open');
  document.body.style.overflow = 'hidden';
}}

function closeDetail() {{
  document.getElementById('detail-modal').classList.remove('open');
  document.body.style.overflow = '';
}}
document.addEventListener('keydown', function(e) {{ if (e.key === 'Escape') closeDetail(); }});

function renderDetailCharts(d) {{
  var roicByYear = d.roic_by_year || {{}};
  var years = Object.keys(roicByYear).sort();
  var roicVals = years.map(function(y) {{ return roicByYear[y] * 100; }});
  var waccLine = years.map(function() {{ return d.wacc != null ? d.wacc * 100 : null; }});

  if (years.length > 0) {{
    Plotly.react('detail-chart-roic', [
      {{ x: years, y: roicVals, type: 'bar', name: 'ROIC',
         marker: {{ color: roicVals.map(function(v) {{ return v > (d.wacc||0)*100 ? '#1a9850' : '#de2d26'; }}), opacity: 0.85 }},
         hovertemplate: '%{{x}}: %{{y:.1f}}%<extra>ROIC</extra>' }},
      {{ x: years, y: waccLine, type: 'scatter', mode: 'lines', name: 'WACC',
         line: {{ color: '#e6750a', width: 2, dash: 'dash' }},
         hovertemplate: 'WACC: %{{y:.1f}}%<extra></extra>' }}
    ], Object.assign({{}}, _base, {{
      title: d.ticker + ' \u2014 ROIC by Year vs WACC', height: 320,
      margin: {{ t: 44, b: 40, l: 50, r: 20 }},
      xaxis: {{ title: '', type: 'category' }}, yaxis: {{ title: 'Return (%)' }},
      legend: {{ x: 0.7, y: 0.95 }}, showlegend: true
    }}), _cfg);
  }} else {{
    document.getElementById('detail-chart-roic').innerHTML = '<p style="text-align:center;color:#999;padding:60px 0">No ROIC data</p>';
  }}

  var cats = [], vals = [], cols = [];
  if (d.dcf_sens_range && d.dcf_sens_range[0] != null) {{ cats.push('DCF Bear'); vals.push(d.dcf_sens_range[0]); cols.push('#de2d26'); }}
  if (d.dcf_fv != null) {{ cats.push('DCF Fair Value'); vals.push(d.dcf_fv); cols.push('#2980b9'); }}
  if (d.dcf_sens_range && d.dcf_sens_range[1] != null) {{ cats.push('DCF Bull'); vals.push(d.dcf_sens_range[1]); cols.push('#1a9850'); }}
  if (d.price != null) {{ cats.push('Current Price'); vals.push(d.price); cols.push('#e6750a'); }}
  if (d.target_low != null) {{ cats.push('Analyst Low'); vals.push(d.target_low); cols.push('#fd8d3c'); }}
  if (d.target_mean != null) {{ cats.push('Analyst Mean'); vals.push(d.target_mean); cols.push('#3182bd'); }}
  if (d.target_high != null) {{ cats.push('Analyst High'); vals.push(d.target_high); cols.push('#74c476'); }}

  if (vals.length > 0) {{
    Plotly.react('detail-chart-valuation', [{{
      x: cats, y: vals, type: 'bar',
      marker: {{ color: cols, opacity: 0.85 }},
      hovertemplate: '%{{x}}: $%{{y:,.2f}}<extra></extra>'
    }}], Object.assign({{}}, _base, {{
      title: d.ticker + ' \u2014 Valuation Reference Points', height: 320,
      margin: {{ t: 44, b: 60, l: 60, r: 20 }},
      xaxis: {{ title: '', tickangle: -20 }}, yaxis: {{ title: '$ Per Share' }}
    }}), _cfg);
  }} else {{
    document.getElementById('detail-chart-valuation').innerHTML = '<p style="text-align:center;color:#999;padding:60px 0">No valuation data</p>';
  }}
}}
</script>
</body>
</html>"""

    with open(filename, 'w') as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sp500 = set(get_sp500_tickers())
    nyse = set(get_nyse_tickers())
    dow = set(get_dow_tickers())
    all_tickers = sorted(sp500 | nyse | dow)

    sec_client = SECEdgarClient("your_email@example.com")
    yf_client = YFinanceClient()

    print("Fetching market history for CAPM benchmark...")
    market_history = yf_client.fetch_history(MARKET_TICKER, period="5y")

    # -----------------------------------------------------------------------
    # Phase 1: Screen — ROIC > WACC + market cap filter (Worksheet Steps 1, 4)
    # -----------------------------------------------------------------------
    print(f"Screening {len(all_tickers)} tickers (ROIC > WACC, mkt cap > $10B)...")
    qualifying = []
    screen_cache = {}

    for i, ticker in enumerate(all_tickers, 1):
        try:
            yf_data = yf_client.fetch_financials(ticker)
            info = yf_data.get('info') or {}

            # Market cap filter (Worksheet Step 1)
            mcap = info.get('marketCap') or 0
            if mcap < MIN_MARKET_CAP:
                print(f"  [{i}/{len(all_tickers)}] {ticker} - mcap ${mcap/1e9:.1f}B < $10B skip")
                continue

            roic_data = calculate_roic(yf_data)
            if not roic_data:
                print(f"  [{i}/{len(all_tickers)}] {ticker} - ROIC N/A skip")
                continue

            capm_data = run_capm(yf_client, ticker, market_history)
            cost_of_equity, re_method = select_cost_of_equity(capm_data, yf_data)
            wacc = calculate_wacc(yf_data, cost_of_equity)

            if wacc is None:
                print(f"  [{i}/{len(all_tickers)}] {ticker} - WACC N/A skip")
                continue

            spread = roic_data['avg_roic'] - wacc
            if spread > 0:
                qualifying.append(ticker)
                screen_cache[ticker] = {
                    'roic_data': roic_data, 'wacc': wacc,
                    'capm_data': capm_data, 'cost_of_equity': cost_of_equity,
                    're_method': re_method, 'yf_data': yf_data,
                }
                print(f"  [{i}/{len(all_tickers)}] {ticker} - ROIC {roic_data['avg_roic']:.1%} "
                      f"WACC {wacc:.1%} spread {spread:.1%} [{re_method}] PASS")
            else:
                print(f"  [{i}/{len(all_tickers)}] {ticker} - spread {spread:.1%} skip")

        except Exception as e:
            print(f"  [{i}/{len(all_tickers)}] {ticker} - error: {e}")

    print(f"\n{len(qualifying)} tickers passed screen out of {len(all_tickers)} total.\n")

    # -----------------------------------------------------------------------
    # Phase 2: Full analysis on qualifying tickers (Worksheet Steps 2-5)
    # -----------------------------------------------------------------------
    results = []
    for ticker in qualifying:
        print(f"Analyzing {ticker}...")
        try:
            cached = screen_cache[ticker]
            yf_data = cached['yf_data']
            capm_data = cached['capm_data']
            wacc = cached['wacc']
            roic_data = cached['roic_data']
            cost_of_equity = cached['cost_of_equity']

            # Company description and CEO (Worksheet Step 3)
            info = yf_data.get('info') or {}
            description = info.get('longBusinessSummary') or ''
            sector = info.get('sector') or ''
            industry = info.get('industry') or ''
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

            # Step 2: Relative multiples
            multiples = compute_relative_multiples(yf_data)
            current_price = multiples.get('price')
            shares = multiples.get('shares')

            # Analyst consensus (Worksheet Step 8)
            analyst = compute_analyst_consensus(yf_data)

            # Consumer sentiment (news headlines via VADER)
            sentiment = fetch_sentiment(ticker)

            # Social media sentiment (StockTwits + Reddit)
            social = fetch_social_sentiment(ticker)

            # Step 4: Risk decomposition
            beta = capm_data['beta'] if capm_data else None
            r2 = capm_data['r2'] if capm_data else None
            alpha = capm_data['alpha'] if capm_data else None

            # Step 5A: Forward DCF
            dcf_fv, dcf_sens_range, fcf_growth = run_forward_dcf(yf_data, wacc)
            mos = (dcf_fv - current_price) / dcf_fv if (dcf_fv and current_price and dcf_fv > 0) else None

            # Step 3A/3B: Earnings quality
            eq = calculate_earnings_quality(yf_data)

            # Step 3B: Altman Z, Piotroski F
            altman_z = calculate_altman_z(yf_data)
            piotroski = calculate_piotroski_f(yf_data)

            # Revenue CAGR
            rev_cagr = calculate_revenue_cagr(yf_data)

            # Step 3C: Balance sheet health
            int_cov = calculate_interest_coverage(yf_data)
            nd_ebitda = calculate_net_debt_ebitda(yf_data)

            # Traditional ratios
            ratios = compute_ratios(yf_data)

            row = {
                'ticker': ticker,
                # Company info (Step 3)
                'description': description,
                'sector': sector,
                'industry': industry,
                'ceo': ceo,
                'ceo_bio': ceo_bio,
                'roic_by_year': roic_data.get('roic_by_year'),
                # Core screen
                'roic': roic_data['avg_roic'],
                'wacc': wacc,
                'spread': roic_data['avg_roic'] - wacc,
                'mcap': multiples.get('market_cap'),
                # Risk (Step 4)
                'beta': beta,
                'raw_beta': capm_data.get('raw_beta') if capm_data else None,
                'r2': r2,
                'alpha': alpha,
                'residual_sigma': capm_data.get('residual_sigma') if capm_data else None,
                'se_beta': capm_data.get('se_beta') if capm_data else None,
                'n_observations': capm_data.get('n_observations') if capm_data else None,
                'er': cost_of_equity,
                're_method': cached['re_method'],
                # Valuation (Step 5)
                'dcf_fv': dcf_fv,
                'price': current_price,
                'mos': mos,
                'dcf_sens_range': dcf_sens_range,
                'fcf_growth': fcf_growth,
                # Multiples (Step 2)
                'pe': multiples.get('pe'),
                'ev_ebitda': multiples.get('ev_ebitda'),
                'pfcf': multiples.get('pfcf'),
                'pb': multiples.get('pb'),
                'peg': multiples.get('peg'),
                # Analyst consensus (Step 8)
                'analyst_rec': analyst.get('rec_key'),
                'num_analysts': analyst.get('num_analysts'),
                'target_mean': analyst.get('target_mean'),
                'target_high': analyst.get('target_high'),
                'target_low': analyst.get('target_low'),
                # Consumer sentiment (VADER on news headlines)
                'sentiment_score': sentiment.get('score'),
                'sentiment_label': sentiment.get('label'),
                'sentiment_articles': sentiment.get('article_count'),
                'sentiment_bull': sentiment.get('bullish_pct'),
                'sentiment_bear': sentiment.get('bearish_pct'),
                # Social media sentiment (StockTwits + Reddit)
                'social_score': social.get('social_score'),
                'social_label': social.get('social_label'),
                'st_bull_pct': social.get('st_bull_pct'),
                'st_bear_pct': social.get('st_bear_pct'),
                'st_labeled': social.get('st_labeled'),
                'reddit_score': social.get('reddit_score'),
                'reddit_posts': social.get('reddit_posts'),
                # Quality (Step 3B)
                'piotroski': piotroski,
                'altman_z': altman_z,
                'cash_conv': eq.get('cash_conversion'),
                'accruals': eq.get('accruals_ratio'),
                'rev_cagr': rev_cagr,
                # Balance sheet (Step 3C)
                'int_cov': int_cov,
                'nd_ebitda': nd_ebitda,
                # Traditional ratios
                'roe': ratios.get('ROE'),
                'de': ratios.get('Debt-to-Equity'),
                'cr': ratios.get('Current Ratio'),
                'roa': ratios.get('ROA'),
            }
            # Composite rating (Worksheet Decision Matrix)
            row['rating'] = compute_rating(row)
            results.append(row)
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

    results.sort(key=lambda r: r.get('spread') or 0, reverse=True)

    os.makedirs("output", exist_ok=True)
    html_filename = os.path.join("output", "stock_analysis_results.html")
    build_html(results, html_filename)
    print(f"\nAnalysis complete. {len(results)} stocks. Results saved to {html_filename}")
