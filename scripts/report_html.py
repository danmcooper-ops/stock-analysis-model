# scripts/report_html.py
"""HTML report builder — renders the interactive Jinja2 report."""
import os
import json
import jinja2
from datetime import date
import numpy as np

import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, '..') not in _sys.path:
    _sys.path.append(os.path.join(_HERE, '..'))
try:
    from models.narrative import generate_sector_profit_pool_narrative
except Exception:
    generate_sector_profit_pool_narrative = None


def _json_default(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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

def _data_val(val):
    """Format a value as a data-value attribute for HTML sorting.

    Returns the value as a rounded string for JS-based table sorting,
    or empty string if None.
    """
    return "" if val is None else str(round(float(val), 6))


def _rating_style(rating):
    colors = {'BUY': '#1a9850', 'LEAN BUY': '#74c476', 'HOLD': '#fd8d3c', 'PASS': '#de2d26'}
    c = colors.get(rating, '#888')
    return f' style="color:{c};font-weight:700;text-align:center"'

_RATING_VAL = {'BUY': 3, 'LEAN BUY': 2, 'HOLD': 1, 'PASS': 0}

def _rating_num(rating):
    return _RATING_VAL.get(rating, -1)

def build_html(rows, filename):
    """Render the interactive HTML report via Jinja2 template."""
    total = len(rows)
    spread_vals = [r['spread'] for r in rows if r.get('spread') is not None]
    avg_spread = sum(spread_vals) / len(spread_vals) if spread_vals else 0
    qualifying_with_mos = sum(1 for r in rows if r.get('mos') is not None and r['mos'] > 0)
    buy_count = sum(1 for r in rows if r.get('rating') == 'BUY')
    lean_buy_count = sum(1 for r in rows if r.get('rating') == 'LEAN BUY')

    chart_data = json.dumps([{
        'ticker': r['ticker'],
        'roic': r.get('roic'), 'wacc': r.get('wacc'), 'spread': r.get('spread'),
        'dcf_fv': r.get('dcf_fv'), 'price': r.get('price'), 'mos': r.get('mos'),
        'piotroski': r.get('piotroski'),
        'pe': r.get('pe'), 'ev_ebitda': r.get('ev_ebitda'),
        'rating': r.get('rating'), 'rating_raw': r.get('rating_raw'),
        'analyst_rec': r.get('analyst_rec'),
        'company_name': r.get('company_name', ''),
        'description': (r.get('description') or '')[:200],
        'sector': r.get('sector'),
        'industry': r.get('industry'),
        'ceo': r.get('ceo'),
        'description_full': r.get('description') or '',
        'ceo_bio': r.get('ceo_bio') or '',
        'founder_led': r.get('founder_led', False),
        'fcf': r.get('fcf'),
        'fcf_margin': r.get('fcf_margin'),
        'sbc_pct_rev': r.get('sbc_pct_rev'),
        'mcap': r.get('mcap'),
        'roic_by_year': r.get('roic_by_year'),
        'roic_cv': r.get('roic_cv'),
        'gross_margin': r.get('gross_margin'),
        'er': r.get('er'),
        're_method': r.get('re_method'),
        'beta_raw': r.get('beta_raw'),
        'beta_adjusted': r.get('beta_adjusted'),
        'beta_r2': r.get('beta_r2'),
        'beta_se': r.get('beta_se'),
        'beta_n_obs': r.get('beta_n_obs'),
        'beta_r2_class': r.get('beta_r2_class'),
        'dcf_sens_range': list(r['dcf_sens_range']) if r.get('dcf_sens_range') else None,
        'fcf_growth': r.get('fcf_growth'),
        'pfcf': r.get('pfcf'),
        'pb': r.get('pb'),
        'num_analysts': r.get('num_analysts'),
        'target_mean': r.get('target_mean'),
        'target_high': r.get('target_high'),
        'target_low': r.get('target_low'),
        'cash_conv': r.get('cash_conv'),
        'accruals': r.get('accruals'),
        'rev_cagr': r.get('rev_cagr'),
        'rev_cagr_5y': r.get('rev_cagr_5y'),
        'rev_cagr_10y': r.get('rev_cagr_10y'),
        'int_cov': r.get('int_cov'),
        'nd_ebitda': r.get('nd_ebitda'),
        'de': r.get('de'),
        'cr': r.get('cr'),
        'roe': r.get('roe'),
        'roa': r.get('roa'),
        # Ownership
        'shares_out': r.get('shares_out'),
        'float_shares': r.get('float_shares'),
        'insider_pct': r.get('insider_pct'),
        'inst_pct': r.get('inst_pct'),
        'shares_short': r.get('shares_short'),
        'short_ratio': r.get('short_ratio'),
        'short_pct_float': r.get('short_pct_float'),
        'share_turnover_rate': r.get('share_turnover_rate'),
        'share_buyback_rate': r.get('share_buyback_rate'),
        'shareholder_yield': r.get('shareholder_yield'),
        'div_yield': r.get('div_yield'),
        'payout_ratio': r.get('payout_ratio'),
        # Insider activity (Form 4)
        'insider_buy_ratio': r.get('insider_buy_ratio'),
        'insider_buy_count_90d': r.get('insider_buy_count_90d'),
        'insider_sell_count_90d': r.get('insider_sell_count_90d'),
        'insider_buy_count_365d': r.get('insider_buy_count_365d'),
        'insider_sell_count_365d': r.get('insider_sell_count_365d'),
        'insider_net_shares': r.get('insider_net_shares'),
        'insider_net_value': r.get('insider_net_value'),
        'insider_transactions': r.get('insider_transactions', []),
        # EDGAR validation
        'edgar_quality_score': r.get('edgar_quality_score'),
        'edgar_fields_flagged': r.get('edgar_fields_flagged', 0),
        'edgar_discrepancies': r.get('edgar_discrepancies', []),
        # Balance sheet risk flags
        'goodwill_pct': r.get('goodwill_pct'),
        'rd_intensity': r.get('rd_intensity'),
        'sga_pct_rev': r.get('sga_pct_rev'),
        'sga_yoy_change': r.get('sga_yoy_change'),
        # Profit pool
        'revenue': r.get('revenue'),
        'operating_income': r.get('operating_income'),
        'operating_margin': r.get('operating_margin'),
        'pp_revenue_share': r.get('pp_revenue_share'),
        'pp_profit_share': r.get('pp_profit_share'),
        'pp_multiple': r.get('pp_multiple'),
        'pp_margin_advantage': r.get('pp_margin_advantage'),
        'pp_sector_hhi': r.get('pp_sector_hhi'),
        'pp_sector_cr4': r.get('pp_sector_cr4'),
        'pp_sector_count': r.get('pp_sector_count'),
        '_sector_median_opm': r.get('_sector_median_opm'),
        # Gate values (actual metric)
        '_gate_mos': r.get('_gate_mos'),
        '_gate_price_fv': r.get('_gate_price_fv'),
        '_gate_p_fcf': r.get('_gate_p_fcf'),
        '_gate_piotroski': r.get('_gate_piotroski'),
        '_gate_int_coverage': r.get('_gate_int_coverage'),
        '_gate_accruals': r.get('_gate_accruals'),
        '_gate_shrhldr_yield': r.get('_gate_shrhldr_yield'),
        '_gate_roic_consistency': r.get('_gate_roic_consistency'),
        '_gate_spread_>_5%': r.get('_gate_spread_>_5%'),
        '_gate_gross_margin': r.get('_gate_gross_margin'),
        '_gate_fund_growth': r.get('_gate_fund_growth'),
        '_gate_margins': r.get('_gate_margins'),
        '_gate_surprise': r.get('_gate_surprise'),
        '_gate_insider_own': r.get('_gate_insider_own'),
        '_gate_turnover': r.get('_gate_turnover'),
        '_gate_buyback_rate': r.get('_gate_buyback_rate'),
        '_gate_insider_buying': r.get('_gate_insider_buying'),
        '_gate_roe': r.get('_gate_roe'),
        '_gate_net_debt_ebitda': r.get('_gate_net_debt_ebitda'),
        '_gate_cash_conv': r.get('_gate_cash_conv'),
        '_gate_rev_durability': r.get('_gate_rev_durability'),
        '_gate_sbc_dilution': r.get('_gate_sbc_dilution'),
        '_gate_price_book': r.get('_gate_price_book'),
        '_gate_fcf_margin': r.get('_gate_fcf_margin'),
        # Gate pass/fail booleans
        '_gp_mos': r.get('_gp_mos'),
        '_gp_price_fv': r.get('_gp_price_fv'),
        '_gp_p_fcf': r.get('_gp_p_fcf'),
        '_gp_piotroski': r.get('_gp_piotroski'),
        '_gp_int_coverage': r.get('_gp_int_coverage'),
        '_gp_accruals': r.get('_gp_accruals'),
        '_gp_shrhldr_yield': r.get('_gp_shrhldr_yield'),
        '_gp_roic_consistency': r.get('_gp_roic_consistency'),
        '_gp_spread_>_5%': r.get('_gp_spread_>_5%'),
        '_gp_gross_margin': r.get('_gp_gross_margin'),
        '_gp_fund_growth': r.get('_gp_fund_growth'),
        '_gp_margins': r.get('_gp_margins'),
        '_gp_surprise': r.get('_gp_surprise'),
        '_gp_insider_own': r.get('_gp_insider_own'),
        '_gp_turnover': r.get('_gp_turnover'),
        '_gp_buyback_rate': r.get('_gp_buyback_rate'),
        '_gp_insider_buying': r.get('_gp_insider_buying'),
        '_gp_roe': r.get('_gp_roe'),
        '_gp_net_debt_ebitda': r.get('_gp_net_debt_ebitda'),
        '_gp_cash_conv': r.get('_gp_cash_conv'),
        '_gp_rev_durability': r.get('_gp_rev_durability'),
        '_gp_sbc_dilution': r.get('_gp_sbc_dilution'),
        '_gp_price_book': r.get('_gp_price_book'),
        '_gp_fcf_margin': r.get('_gp_fcf_margin'),
        # Continuous scores (0-100)
        '_score_mos': r.get('_score_mos'),
        '_score_price_fv': r.get('_score_price_fv'),
        '_score_p_fcf': r.get('_score_p_fcf'),
        '_score_piotroski': r.get('_score_piotroski'),
        '_score_int_coverage': r.get('_score_int_coverage'),
        '_score_accruals': r.get('_score_accruals'),
        '_score_shrhldr_yield': r.get('_score_shrhldr_yield'),
        '_score_roic_consistency': r.get('_score_roic_consistency'),
        '_score_spread': r.get('_score_spread'),
        '_score_gross_margin': r.get('_score_gross_margin'),
        '_score_fund_growth': r.get('_score_fund_growth'),
        '_score_margins': r.get('_score_margins'),
        '_score_surprise': r.get('_score_surprise'),
        '_score_insider_own': r.get('_score_insider_own'),
        '_score_turnover': r.get('_score_turnover'),
        '_score_buyback_rate': r.get('_score_buyback_rate'),
        '_score_insider_buying': r.get('_score_insider_buying'),
        '_score_roe': r.get('_score_roe'),
        '_score_net_debt_ebitda': r.get('_score_net_debt_ebitda'),
        '_score_cash_conv': r.get('_score_cash_conv'),
        '_score_rev_durability': r.get('_score_rev_durability'),
        '_score_sbc_dilution': r.get('_score_sbc_dilution'),
        '_score_price_book': r.get('_score_price_book'),
        '_score_fcf_margin': r.get('_score_fcf_margin'),
        # Category totals + composite
        '_score_valuation': r.get('_score_valuation'),
        '_score_quality': r.get('_score_quality'),
        '_score_moat': r.get('_score_moat'),
        '_score_growth': r.get('_score_growth'),
        '_score_ownership': r.get('_score_ownership'),
        '_composite_score': r.get('_composite_score'),
        '_composite_score_raw': r.get('_composite_score_raw'),
        '_gates_passed': r.get('_gates_passed'),
        # MC confidence
        'mc_confidence': r.get('mc_confidence'),
        'mc_cv': r.get('mc_cv'),
        'mc_p10_fv': r.get('mc_p10_fv'),
        'mc_p90_fv': r.get('mc_p90_fv'),
        # Macro
        'macro_regime': r.get('macro_regime'),
        'macro_composite': r.get('macro_composite'),
        'macro_erp': r.get('macro_erp'),
        'financial_summary': r.get('financial_summary', []),
        'sector_headwinds': r.get('sector_headwinds', []),
        'sector_tailwinds': r.get('sector_tailwinds', []),
        'news_headlines': r.get('news_headlines', []),
        'news_sentiment': r.get('news_sentiment'),
        'legal_filings': r.get('legal_filings', []),
        'legal_count': r.get('legal_count', 0),
        'legal_latest': r.get('legal_latest'),
        'suppliers': r.get('suppliers', []),
        'customers': r.get('customers', []),
        'supply_chain_available': r.get('supply_chain_available', False),
        'finnhub_peers': r.get('finnhub_peers', []),
        # Growth diagnostics
        'analyst_ltg': r.get('analyst_ltg'),
        'margin_trend': r.get('margin_trend'),
        'surprise_avg': r.get('surprise_avg'),
        'fundamental_growth': r.get('fundamental_growth'),
        'reinvestment_rate': r.get('reinvestment_rate'),
        'terminal_growth': r.get('terminal_growth'),
        # Peers + source
        '_peers': r.get('_peers'),
        'source_group': r.get('source_group'),
        # DDM (Dividend Discount Model)
        'ddm_eligible': r.get('ddm_eligible', False),
        'ddm_fv': r.get('ddm_fv'),
        'ddm_h_fv': r.get('ddm_h_fv'),
        'ddm_growth': r.get('ddm_growth'),
        'ddm_div_cagr': r.get('ddm_div_cagr'),
        'ddm_sustainable_growth': r.get('ddm_sustainable_growth'),
        'ddm_payout_flag': r.get('ddm_payout_flag', False),
        'ddm_consecutive_years': r.get('ddm_consecutive_years'),
        'ddm_mc_median': r.get('ddm_mc_median'),
        'ddm_mc_p10': r.get('ddm_mc_p10'),
        'ddm_mc_p90': r.get('ddm_mc_p90'),
        'ddm_mc_cv': r.get('ddm_mc_cv'),
        '_blended_method': r.get('_blended_method', 'DCF'),
        '_ddm_low_confidence': r.get('_ddm_low_confidence', False),
        # Reverse DCF
        'implied_growth': r.get('implied_growth'),
        'implied_vs_estimated': r.get('implied_vs_estimated'),
        # EPV
        'epv_fv': r.get('epv_fv'),
        'epv_pfv': r.get('epv_pfv'),
        'epv_mos': r.get('epv_mos'),
        'epv_growth_fv': r.get('epv_growth_fv'),
        # RIM
        'rim_fv': r.get('rim_fv'),
        'rim_mos': r.get('rim_mos'),
        # Altman Z
        'altman_z': r.get('altman_z'),
        'altman_z_zone': r.get('altman_z_zone'),
        # Beneish
        'beneish_m': r.get('beneish_m'),
        'beneish_flag': r.get('beneish_flag'),
        # DuPont
        'dupont_margin': r.get('dupont_margin'),
        'dupont_turnover': r.get('dupont_turnover'),
        'dupont_leverage': r.get('dupont_leverage'),
        # 52-week
        'high_52w': r.get('high_52w'),
        'low_52w': r.get('low_52w'),
        'pct_from_52w_high': r.get('pct_from_52w_high'),
        'pct_from_52w_low': r.get('pct_from_52w_low'),
        'range_52w_position': r.get('range_52w_position'),
        # Portfolio
        'position_weight': r.get('position_weight'),
        # Culture narrative (descriptive, woven into company description)
        'culture_narrative': r.get('culture_narrative'),
        # Culture raw inputs
        'employees': r.get('employees'),
        'ceo_total_pay': r.get('ceo_total_pay'),
        'compensation_risk': r.get('compensation_risk'),
        'sbc': r.get('sbc'),
        # Glassdoor (best-effort)
        'glassdoor_rating': r.get('glassdoor_rating'),
        'glassdoor_ceo_pct': r.get('glassdoor_ceo_pct'),
        'glassdoor_rec_pct': r.get('glassdoor_rec_pct'),
        # Derived culture metrics
        'revenue_per_emp': r.get('revenue_per_emp'),
        'fcf_per_emp': r.get('fcf_per_emp'),
        'ceo_pay_ratio': r.get('ceo_pay_ratio'),
        'sbc_per_emp': r.get('sbc_per_emp'),
        'rpe_cagr': r.get('rpe_cagr'),
        # Culture signals
        'employment_legal_flag': r.get('employment_legal_flag', False),
        'layoff_news_signal': r.get('layoff_news_signal', False),
        'culture_award_signal': r.get('culture_award_signal', False),
    } for r in rows], default=_json_default)

    # Gate metadata for Matrix view rendering in JavaScript
    gate_meta = json.dumps({
        'gates': [
            {'key': '_gate_mos', 'label': 'MoS', 'gpKey': '_gp_mos',
             'scoreKey': '_score_mos', 'threshold': 'MoS > 10%',
             'category': 'Valuation', 'fmt': 'pct1'},
            {'key': '_gate_price_fv', 'label': 'Price/FV', 'gpKey': '_gp_price_fv',
             'scoreKey': '_score_price_fv', 'threshold': 'P/FV < 1.0',
             'category': 'Valuation', 'fmt': 'ratio'},
            {'key': '_gate_p_fcf', 'label': 'P/FCF', 'gpKey': '_gp_p_fcf',
             'scoreKey': '_score_p_fcf', 'threshold': 'P/FCF \u2264 20\u00d7',
             'category': 'Valuation', 'fmt': 'ratio'},
            {'key': '_gate_int_coverage', 'label': 'Int Cov', 'gpKey': '_gp_int_coverage',
             'scoreKey': '_score_int_coverage', 'threshold': 'IC > 3\u00d7',
             'category': 'Quality', 'fmt': 'ratio'},
            {'key': '_gate_accruals', 'label': 'Accruals', 'gpKey': '_gp_accruals',
             'scoreKey': '_score_accruals', 'threshold': '|Acr| < 8%',
             'category': 'Quality', 'fmt': 'pct1'},
            {'key': '_gate_shrhldr_yield', 'label': 'Shrhldr Yld', 'gpKey': '_gp_shrhldr_yield',
             'scoreKey': '_score_shrhldr_yield', 'threshold': 'Yield > 2%',
             'category': 'Ownership', 'fmt': 'pct1'},
            {'key': '_gate_insider_own', 'label': 'Insider %', 'gpKey': '_gp_insider_own',
             'scoreKey': '_score_insider_own', 'threshold': 'Insider >= 5%',
             'category': 'Ownership', 'fmt': 'pct1'},
            {'key': '_gate_roe', 'label': 'ROE', 'gpKey': '_gp_roe',
             'scoreKey': '_score_roe', 'threshold': 'ROE > 20%',
             'category': 'Growth', 'fmt': 'pct1'},
            {'key': '_gate_buyback_rate', 'label': 'Buyback', 'gpKey': '_gp_buyback_rate',
             'scoreKey': '_score_buyback_rate', 'threshold': 'Buyback > 1%',
             'category': 'Ownership', 'fmt': 'pct1'},
            {'key': '_gate_roic_consistency', 'label': 'ROIC CV', 'gpKey': '_gp_roic_consistency',
             'scoreKey': '_score_roic_consistency', 'threshold': 'CV < 30%',
             'category': 'Moat', 'fmt': 'pct1'},
            {'key': '_gate_spread_>_5%', 'label': 'Spread', 'gpKey': '_gp_spread_>_5%',
             'scoreKey': '_score_spread', 'threshold': 'Spread > 7%',
             'category': 'Moat', 'fmt': 'pct1'},
            {'key': '_gate_gross_margin', 'label': 'Gross Mgn', 'gpKey': '_gp_gross_margin',
             'scoreKey': '_score_gross_margin', 'threshold': 'GM > 35%',
             'category': 'Moat', 'fmt': 'pct1'},
            {'key': '_gate_fund_growth', 'label': 'Fund Growth', 'gpKey': '_gp_fund_growth',
             'scoreKey': '_score_fund_growth', 'threshold': 'FG > 3%',
             'category': 'Growth', 'fmt': 'pct1'},
            {'key': '_gate_margins', 'label': 'Margins', 'gpKey': '_gp_margins',
             'scoreKey': '_score_margins', 'threshold': 'Margin >= 0',
             'category': 'Growth', 'fmt': 'pct1'},
            {'key': '_gate_net_debt_ebitda', 'label': 'ND/EBITDA', 'gpKey': '_gp_net_debt_ebitda',
             'scoreKey': '_score_net_debt_ebitda', 'threshold': 'ND/EBITDA \u2264 1.5\u00d7',
             'category': 'Quality', 'fmt': 'ratio'},
            {'key': '_gate_cash_conv', 'label': 'Cash Conv', 'gpKey': '_gp_cash_conv',
             'scoreKey': '_score_cash_conv', 'threshold': 'CashConv \u2265 0.85\u00d7',
             'category': 'Quality', 'fmt': 'ratio'},
            {'key': '_gate_rev_durability', 'label': '10Y Rev CAGR', 'gpKey': '_gp_rev_durability',
             'scoreKey': '_score_rev_durability', 'threshold': '10Y RevCAGR > 2%',
             'category': 'Growth', 'fmt': 'pct1'},
            {'key': '_gate_sbc_dilution', 'label': 'SBC/Rev', 'gpKey': '_gp_sbc_dilution',
             'scoreKey': '_score_sbc_dilution', 'threshold': 'SBC/Rev \u2264 2%',
             'category': 'Ownership', 'fmt': 'pct1'},
            {'key': '_gate_price_book', 'label': 'P/B', 'gpKey': '_gp_price_book',
             'scoreKey': '_score_price_book', 'threshold': 'P/B \u2264 5\u00d7',
             'category': 'Valuation', 'fmt': 'ratio'},
            {'key': '_gate_fcf_margin', 'label': 'FCF Margin', 'gpKey': '_gp_fcf_margin',
             'scoreKey': '_score_fcf_margin', 'threshold': 'FCF Margin > 12%',
             'category': 'Moat', 'fmt': 'pct1'},
        ],
        'categories': [
            {'name': 'Valuation', 'weight': 0.20, 'dark': '#2F5496', 'light': '#D6E4F0',
             'scoreKey': '_score_valuation'},
            {'name': 'Quality', 'weight': 0.20, 'dark': '#548235', 'light': '#E2EFDA',
             'scoreKey': '_score_quality'},
            {'name': 'Moat', 'weight': 0.40, 'dark': '#C55A11', 'light': '#FCE4CC',
             'scoreKey': '_score_moat'},
            {'name': 'Growth', 'weight': 0.05, 'dark': '#7030A0', 'light': '#E4CCEF',
             'scoreKey': '_score_growth'},
            {'name': 'Ownership', 'weight': 0.15, 'dark': '#BF8F00', 'light': '#FFF2CC',
             'scoreKey': '_score_ownership'},
        ],
    }, default=_json_default)

    # Per-sector profit pool narratives (top-level Profit Pool tab)
    sector_pool_data = {}
    if generate_sector_profit_pool_narrative is not None:
        _by_sector = {}
        for r in rows:
            s = r.get('sector')
            if not s or r.get('pp_revenue_share') is None:
                continue
            _by_sector.setdefault(s, []).append(r)
        for s, srows in _by_sector.items():
            try:
                narr = generate_sector_profit_pool_narrative(s, srows)
                if narr:
                    sector_pool_data[s] = narr
            except Exception as e:
                print(f"[warn] sector pool narrative failed for {s}: {e}")
    sector_pool_json = json.dumps(sector_pool_data, default=_json_default)

    # Render Jinja2 template
    template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=False,
    )
    template = env.get_template('report.html')
    html = template.render(
        total=total,
        avg_spread_fmt=f"{avg_spread:.1%}",
        qualifying_with_mos=qualifying_with_mos,
        buy_count=buy_count,
        lean_buy_count=lean_buy_count,
        chart_data=chart_data,
        gate_meta=gate_meta,
        sector_pool_json=sector_pool_json,
        generated_at=date.today().strftime('%Y-%m-%d'),
    )

    with open(filename, 'w') as f:
        f.write(html)

