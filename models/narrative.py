"""Stock-level headwind / tailwind narrative generator.

Produces contextual, plain-English bullet points for each stock by
combining three layers of analysis:

  1. Macro layer   – economy-wide signals (VIX, yield curve, regime)
  2. Sector layer  – sector ETF momentum + sector-specific macro drivers
  3. Stock layer   – per-company fundamentals, valuation, ownership, analyst

No scoring or matrix impact — purely informational for the detail panel.
All functions are pure (no I/O).
"""

# ---------------------------------------------------------------------------
# Sector-specific macro sensitivity mappings
# ---------------------------------------------------------------------------

# Which commodity / macro proxy matters most per sector
_SECTOR_MACRO_DRIVERS = {
    'Technology': {
        'rate_sensitive': True,
        'driver_label': 'interest rates',
        'cyclical': False,
        'growth_premium': True,
    },
    'Healthcare': {
        'rate_sensitive': False,
        'defensive': True,
        'driver_label': 'defensive rotation',
    },
    'Energy': {
        'rate_sensitive': False,
        'commodity': 'oil',
        'driver_label': 'oil prices',
        'cyclical': True,
    },
    'Financials': {
        'rate_sensitive': True,
        'benefits_from_higher_rates': True,
        'driver_label': 'yield curve & credit spreads',
    },
    'Real Estate': {
        'rate_sensitive': True,
        'yield_sector': True,
        'driver_label': 'interest rates & bond yields',
    },
    'Consumer Cyclical': {
        'rate_sensitive': False,
        'cyclical': True,
        'consumer_sensitive': True,
        'driver_label': 'consumer spending',
    },
    'Consumer Defensive': {
        'rate_sensitive': False,
        'defensive': True,
        'driver_label': 'defensive positioning',
    },
    'Utilities': {
        'rate_sensitive': True,
        'yield_sector': True,
        'defensive': True,
        'driver_label': 'bond yield competition',
    },
    'Industrials': {
        'rate_sensitive': False,
        'cyclical': True,
        'driver_label': 'industrial activity',
    },
    'Basic Materials': {
        'rate_sensitive': False,
        'cyclical': True,
        'commodity': 'metals',
        'driver_label': 'commodity prices',
    },
    'Communication Services': {
        'rate_sensitive': True,
        'growth_premium': True,
        'driver_label': 'growth valuations & ad spend',
    },
}


def _pct(v, digits=0):
    """Format a decimal as a readable percentage string."""
    if v is None:
        return None
    if digits == 0:
        return f'{abs(v) * 100:.0f}%'
    return f'{abs(v) * 100:.{digits}f}%'


def _dollar(v):
    """Format dollar value."""
    if v is None:
        return None
    return f'${v:,.0f}'


# ---------------------------------------------------------------------------
# Layer 1: Macro signals (economy-wide)
# ---------------------------------------------------------------------------

def _macro_signals(macro_regime_result):
    """Generate macro-level headwinds/tailwinds from regime data."""
    hw, tw = [], []
    if not macro_regime_result:
        return hw, tw

    regime = macro_regime_result.get('regime', 'neutral')
    raw = macro_regime_result.get('raw_indicators', {})
    vix = raw.get('vix')
    yc_slope = raw.get('yield_curve_slope')
    credit = raw.get('credit_spread_3m')
    spy_ratio = raw.get('spy_sma200_ratio')

    # VIX
    if vix is not None:
        if vix > 30:
            hw.append(f'High market volatility with VIX at {vix:.0f}, signaling elevated fear')
        elif vix > 25:
            hw.append(f'Above-average market volatility (VIX {vix:.0f})')
        elif vix < 15:
            tw.append(f'Calm market conditions with VIX at {vix:.0f}')

    # Yield curve
    if yc_slope is not None:
        if yc_slope < -0.005:
            hw.append('Inverted yield curve points to rising recession risk')
        elif yc_slope > 0.015:
            tw.append('Healthy yield curve slope supports continued economic growth')

    # Credit spreads
    if credit is not None:
        if credit > 0.015:
            hw.append('Widening credit spreads suggest growing stress in corporate debt markets')
        elif credit < -0.015:
            tw.append('Tight credit spreads reflect confidence in corporate borrowers')

    # Broad market momentum
    if spy_ratio is not None:
        if spy_ratio > 1.05:
            tw.append('Broad market in a strong uptrend, trading well above its 200-day average')
        elif spy_ratio < 0.95:
            hw.append('Broad market in a downtrend, trading below its 200-day average')

    # Regime summary
    if regime == 'expansion':
        tw.append('Macro indicators collectively point to economic expansion')
    elif regime == 'contraction':
        hw.append('Multiple macro indicators signal economic contraction')
    elif regime == 'late_cycle':
        hw.append('Late-cycle dynamics increase the probability of a slowdown')

    return hw, tw


# ---------------------------------------------------------------------------
# Layer 2: Sector signals (sector ETF + sector-specific drivers)
# ---------------------------------------------------------------------------

def _sector_signals(row, sector_data, macro_regime_result, commodity_data):
    """Generate sector-level headwinds/tailwinds."""
    hw, tw = [], []
    sector = row.get('sector')
    if not sector:
        return hw, tw

    drivers = _SECTOR_MACRO_DRIVERS.get(sector, {})
    raw = macro_regime_result.get('raw_indicators', {}) if macro_regime_result else {}
    regime = macro_regime_result.get('regime', 'neutral') if macro_regime_result else 'neutral'
    yc_slope = raw.get('yield_curve_slope')

    # --- Sector ETF momentum ---
    metrics = sector_data.get(sector, {}) if sector_data else {}
    rel_3m = metrics.get('rel_strength_3m')
    sma_ratio = metrics.get('sma200_ratio')
    ret_6m = metrics.get('return_6m')
    vol = metrics.get('volatility_30d')

    if rel_3m is not None:
        if rel_3m > 0.05:
            tw.append(f'{sector} sector showing strong relative momentum, outperforming the market by {_pct(rel_3m)} over 3 months')
        elif rel_3m > 0.03:
            tw.append(f'{sector} sector outperforming the broader market over the past quarter')
        elif rel_3m < -0.05:
            hw.append(f'{sector} sector lagging the market significantly, underperforming by {_pct(rel_3m)} over 3 months')
        elif rel_3m < -0.03:
            hw.append(f'{sector} sector underperforming the broader market this quarter')

    if sma_ratio is not None:
        if sma_ratio < 0.95:
            hw.append(f'{sector} ETF trading well below its long-term trend')
        elif sma_ratio > 1.05:
            tw.append(f'{sector} ETF in a sustained uptrend above its long-term average')

    if vol is not None and vol > 0.30:
        hw.append(f'Elevated sector volatility ({_pct(vol)} annualized) adds uncertainty')

    if ret_6m is not None:
        if ret_6m > 0.20:
            tw.append(f'Strong 6-month sector performance ({_pct(ret_6m, 1)} return)')
        elif ret_6m < -0.15:
            hw.append(f'Sector under pressure with a {_pct(ret_6m, 1)} decline over the past 6 months')

    # --- Sector-specific macro drivers ---

    # Interest rate sensitivity
    if drivers.get('rate_sensitive') and yc_slope is not None:
        if drivers.get('benefits_from_higher_rates'):
            # Financials benefit from steeper curve
            if yc_slope > 0.015:
                tw.append('Steeper yield curve supports net interest margins')
            elif yc_slope < 0:
                hw.append('Flat or inverted yield curve compresses lending margins')
        elif drivers.get('yield_sector'):
            # Utilities, Real Estate compete with bonds
            if yc_slope > 0.02:
                hw.append('Higher bond yields make dividend yields less competitive')
            elif yc_slope < 0.005:
                tw.append('Low bond yields make dividend-paying sectors more attractive')
        elif drivers.get('growth_premium'):
            # Tech, Comm Services — higher rates pressure long-duration growth
            if yc_slope > 0.02:
                hw.append('Rising rates put pressure on growth stock valuations')
            elif yc_slope < 0.005:
                tw.append('Low rate environment supports growth stock valuations')

    # Cyclical vs defensive rotation
    if drivers.get('cyclical'):
        if regime == 'expansion':
            tw.append(f'{sector} sector typically benefits from economic expansion')
        elif regime in ('contraction', 'late_cycle'):
            hw.append(f'Cyclical sectors like {sector} tend to underperform in slowdowns')
    elif drivers.get('defensive'):
        if regime in ('contraction', 'late_cycle'):
            tw.append(f'{sector} tends to hold up well as investors rotate into defensive names')
        elif regime == 'expansion':
            hw.append('Expansion environment favors cyclical sectors over defensives')

    # Commodity-sensitive sectors
    oil_data = commodity_data.get('oil', {}) if commodity_data else {}
    gold_data = commodity_data.get('gold', {}) if commodity_data else {}
    tlt_data = commodity_data.get('bonds', {}) if commodity_data else {}

    if drivers.get('commodity') == 'oil' and oil_data:
        oil_3m = oil_data.get('return_3m')
        if oil_3m is not None:
            if oil_3m > 0.10:
                tw.append(f'Rising oil prices ({_pct(oil_3m)} over 3 months) support energy revenues')
            elif oil_3m < -0.10:
                hw.append(f'Falling oil prices ({_pct(oil_3m)} decline over 3 months) pressure energy earnings')

    if drivers.get('commodity') == 'metals' and gold_data:
        gold_3m = gold_data.get('return_3m')
        if gold_3m is not None:
            if gold_3m > 0.08:
                tw.append('Commodity prices trending higher, supporting materials sector')
            elif gold_3m < -0.08:
                hw.append('Weakness in commodity prices pressures materials companies')

    # Bond yields & yield sectors
    if drivers.get('yield_sector') and tlt_data:
        tlt_3m = tlt_data.get('return_3m')
        if tlt_3m is not None:
            if tlt_3m < -0.05:
                hw.append('Rising long-term yields create headwinds for yield-oriented sectors')
            elif tlt_3m > 0.05:
                tw.append('Falling long-term yields benefit income-oriented investments')

    # Consumer sentiment proxy
    if drivers.get('consumer_sensitive') and commodity_data:
        cons_ratio = commodity_data.get('consumer_sentiment', {}).get('xly_xlp_ratio')
        if cons_ratio is not None:
            if cons_ratio > 1.06:
                tw.append('Consumer discretionary outpacing staples, signaling healthy spending')
            elif cons_ratio < 0.94:
                hw.append('Consumer staples outperforming discretionary, signaling cautious spending')

    return hw, tw


# ---------------------------------------------------------------------------
# Layer 3: Stock-specific signals
# ---------------------------------------------------------------------------

def _stock_signals(row, sector_medians):
    """Generate per-company headwinds/tailwinds from fundamentals."""
    hw, tw = [], []

    sector = row.get('sector', '')

    # === GROWTH signals ===
    rev_cagr = row.get('rev_cagr')
    fund_growth = row.get('fundamental_growth')
    margin_trend = row.get('margin_trend')
    surprise_avg = row.get('surprise_avg')

    if rev_cagr is not None:
        if rev_cagr > 0.12:
            tw.append(f'Strong {rev_cagr * 100:.0f}% revenue CAGR provides a compelling growth runway')
        elif rev_cagr > 0.05:
            tw.append(f'Healthy {rev_cagr * 100:.0f}% revenue growth supports future earnings power')
        elif rev_cagr < -0.03:
            hw.append(f'Revenue declining at {abs(rev_cagr) * 100:.0f}% annually, narrowing the path to earnings growth')
        elif rev_cagr < 0.01:
            hw.append('Stagnant revenue growth limits upside potential')

    if fund_growth is not None:
        if fund_growth > 0.08:
            tw.append(f'High reinvestment rate supports {fund_growth * 100:.0f}% organic growth capacity')
        elif fund_growth < 0.02:
            hw.append('Low reinvestment-driven growth suggests limited organic expansion')

    if margin_trend is not None:
        if margin_trend > 0.02:
            tw.append('Profit margins are expanding, suggesting improving efficiency or pricing power')
        elif margin_trend < -0.02:
            hw.append('Profit margins are contracting, which may indicate cost pressures or competitive headwinds')

    if surprise_avg is not None:
        if surprise_avg > 0.05:
            tw.append(f'Consistently beating earnings estimates by {_pct(surprise_avg)} on average')
        elif surprise_avg > 0.02:
            tw.append('Recent earnings have come in above analyst expectations')
        elif surprise_avg < -0.03:
            hw.append(f'Earnings disappointing vs estimates, missing by {_pct(surprise_avg)} on average')
        elif surprise_avg < 0:
            hw.append('Recent earnings have fallen short of analyst expectations')

    # === PROFIT POOL signals ===
    pp_margin_adv = row.get('pp_margin_advantage')
    pp_rev_share = row.get('pp_revenue_share')
    pp_hhi = row.get('pp_sector_hhi')
    pp_mult = row.get('pp_multiple')

    if pp_mult is not None:
        if pp_mult > 1.5 and pp_rev_share and pp_rev_share > 0.03:
            tw.append(f'Captures a disproportionate {pp_mult:.1f}× share of sector profits — strong pricing power')
        elif pp_mult > 1.0:
            tw.append(f'Profit share exceeds revenue share ({pp_mult:.1f}× profit pool multiple)')
        elif pp_mult < 0.5 and pp_rev_share and pp_rev_share > 0.03:
            hw.append(f'Under-earns vs revenue share ({pp_mult:.1f}× profit pool multiple), competitive pressure evident')
        elif pp_mult < 0.8:
            hw.append(f'Below-average profit capture relative to revenue share ({pp_mult:.1f}×)')

    if pp_margin_adv is not None:
        if pp_margin_adv > 0.08:
            tw.append(f'Operating margins {_pct(pp_margin_adv)} above sector median, indicating pricing power')
        elif pp_margin_adv < -0.08:
            hw.append(f'Operating margins {_pct(pp_margin_adv)} below sector median, suggesting cost disadvantage')

    if pp_hhi is not None:
        if pp_hhi > 0.25 and pp_rev_share and pp_rev_share > 0.10:
            tw.append('Operates in a concentrated industry with significant market position')
        elif pp_hhi < 0.10 and pp_mult is not None and pp_mult < 0.8:
            hw.append('Fragmented industry with below-average profit capture')

    # --- Valuation context ---
    mos = row.get('mos')
    implied_vs_est = row.get('implied_vs_estimated')
    range_pos = row.get('range_52w_position')
    ee_vs_sector = row.get('_ee_vs_sector')

    if mos is not None:
        if mos > 0.30:
            tw.append(f'Trading at a meaningful discount to estimated fair value ({_pct(mos)} margin of safety)')
        elif mos > 0.15:
            tw.append(f'Shares appear undervalued with a {_pct(mos)} margin of safety')
        elif mos < -0.15:
            hw.append(f'Shares trading {_pct(mos)} above estimated fair value')
        elif mos < 0:
            hw.append('Priced near or slightly above estimated fair value')

    if implied_vs_est is not None:
        if implied_vs_est > 0.05:
            hw.append('Market price implies higher growth than fundamentals support')
        elif implied_vs_est < -0.05:
            tw.append('Market is pricing in lower growth than the company has historically delivered')

    if range_pos is not None:
        if range_pos < 20:
            tw.append('Trading near the bottom of its 52-week range')
        elif range_pos > 85:
            hw.append('Trading near 52-week highs, which may limit near-term upside')

    if ee_vs_sector is not None:
        if ee_vs_sector < -0.20:
            tw.append(f'Valued at a {_pct(ee_vs_sector)} discount to sector median EV/EBITDA')
        elif ee_vs_sector > 0.30:
            hw.append(f'Carries a {_pct(ee_vs_sector)} premium to sector median EV/EBITDA')

    # === MOAT signals ===
    spread = row.get('spread')
    roic = row.get('roic')
    roic_cv = row.get('roic_cv')
    gross_margin = row.get('gross_margin')

    if spread is not None and roic is not None:
        if spread > 0.10:
            tw.append(f'Durable moat — earns {roic * 100:.0f}% ROIC, well above its cost of capital')
        elif spread > 0.05:
            tw.append(f'Earns above its cost of capital (ROIC spread {spread * 100:.0f}%), indicating a competitive advantage')
        elif spread < -0.05:
            hw.append('Returns fall short of cost of capital, signaling value destruction risk')
        elif spread < 0:
            hw.append('ROIC barely covers cost of capital, leaving little moat cushion')

    if roic_cv is not None:
        if roic_cv < 0.15:
            tw.append('Highly consistent returns on capital over 5 years — a hallmark of competitive advantage')
        elif roic_cv > 0.40:
            hw.append('Volatile returns on capital suggest an uneven competitive position')

    if gross_margin is not None:
        if gross_margin > 0.60:
            tw.append(f'Premium gross margins ({gross_margin * 100:.0f}%) reflect strong pricing power or asset-light model')
        elif gross_margin < 0.20:
            hw.append(f'Thin gross margins ({gross_margin * 100:.0f}%) leave little buffer against cost inflation')

    # === QUALITY signals ===
    piotroski = row.get('piotroski')
    altman_z = row.get('altman_z')
    altman_zone = row.get('altman_z_zone')
    int_cov = row.get('int_cov')
    beneish_flag = row.get('beneish_flag')
    accruals = row.get('accruals')
    cash_conv = row.get('cash_conv')
    nd_ebitda = row.get('nd_ebitda')

    if piotroski is not None:
        if piotroski >= 7:
            tw.append(f'Strong financial momentum (Piotroski {piotroski}/9) across profitability, leverage, and efficiency')
        elif piotroski <= 3:
            hw.append(f'Weak financial momentum (Piotroski {piotroski}/9) signals deteriorating fundamentals')

    if accruals is not None:
        if abs(accruals) < 0.03:
            tw.append('Very low accruals — earnings are well-backed by actual cash flows')
        elif abs(accruals) > 0.10:
            hw.append('High accrual ratio suggests earnings may not be fully sustainable')

    if cash_conv is not None:
        if cash_conv > 1.2:
            tw.append('Strong cash conversion — free cash flow exceeds reported earnings')
        elif cash_conv < 0.5:
            hw.append('Poor cash conversion raises questions about earnings quality')

    if altman_zone == 'distress':
        hw.append(f'Altman Z-Score in distress zone ({altman_z:.1f}), indicating elevated financial risk')
    elif altman_zone == 'grey':
        hw.append('Altman Z-Score in the grey zone — financial health warrants monitoring')
    elif altman_zone == 'safe' and altman_z and altman_z > 4.0:
        tw.append('Strong financial health with a comfortable Altman Z-Score')

    if int_cov is not None:
        if int_cov < 3.0:
            hw.append(f'Thin interest coverage ({int_cov:.1f}×) leaves little room for earnings volatility')
        elif int_cov > 10:
            tw.append('Ample interest coverage provides financial flexibility')

    if beneish_flag:
        hw.append('Beneish M-Score flags potential earnings manipulation risk')

    if nd_ebitda is not None:
        if nd_ebitda < 0:
            tw.append('Net cash position provides strategic flexibility for acquisitions or buybacks')
        elif nd_ebitda < 1.5:
            tw.append(f'Conservative leverage ({nd_ebitda:.1f}× net debt/EBITDA) gives room to invest through cycles')
        elif nd_ebitda > 4.0:
            hw.append(f'Heavy debt load ({nd_ebitda:.1f}× net debt/EBITDA) limits strategic flexibility and increases downturn risk')
        elif nd_ebitda > 3.0:
            hw.append(f'Elevated leverage ({nd_ebitda:.1f}× net debt/EBITDA) may constrain future capital allocation')

    # --- Ownership & alignment ---
    insider_pct = row.get('insider_pct')
    short_pct = row.get('short_pct_float')
    buyback = row.get('share_buyback_rate')
    founder_led = row.get('founder_led')

    if founder_led:
        tw.append('Founder-led company, often associated with long-term strategic alignment')

    if insider_pct is not None:
        if insider_pct > 0.10:
            tw.append(f'High insider ownership ({_pct(insider_pct)}) aligns management with shareholders')
        elif insider_pct > 0.05:
            tw.append(f'Meaningful insider ownership at {_pct(insider_pct)}')

    if short_pct is not None:
        if short_pct > 0.10:
            hw.append(f'Elevated short interest ({_pct(short_pct, 1)} of float) signals bearish sentiment')
        elif short_pct > 0.05:
            hw.append(f'Above-average short interest at {_pct(short_pct, 1)} of float')

    if buyback is not None and abs(buyback) < 0.50:  # sanity cap
        if buyback > 0.03:
            tw.append(f'Active share repurchase program reducing float by {_pct(buyback)} annually')
        elif buyback < -0.03:
            hw.append('Significant share dilution from new issuance')

    # --- Insider transaction activity (Form 4 filings) ---
    insider_buy_ratio = row.get('insider_buy_ratio')
    insider_net_value = row.get('insider_net_value')
    insider_buy_90d = (row.get('insider_buy_count_90d') or 0)
    insider_sell_90d = (row.get('insider_sell_count_90d') or 0)
    insider_txn_total = insider_buy_90d + insider_sell_90d

    if insider_buy_ratio is not None and insider_txn_total >= 2:
        if insider_buy_ratio > 0.60:
            tw.append(f'Insiders are net buyers ({_pct(insider_buy_ratio)} buy ratio over the past year)')
        elif insider_buy_ratio < 0.30:
            hw.append(f'Insiders are net sellers ({_pct(1 - insider_buy_ratio)} sell ratio over the past year)')

    if insider_net_value is not None:
        if insider_net_value > 500_000:
            tw.append(f'Large open-market insider purchases (net ${insider_net_value / 1e6:.1f}M)')
        elif insider_net_value < -2_000_000:
            hw.append(f'Significant insider selling (net ${abs(insider_net_value) / 1e6:.1f}M)')

    # --- Dividend sustainability ---
    div_yield = row.get('div_yield')
    payout_ratio = row.get('payout_ratio')
    ddm_consecutive = row.get('ddm_consecutive_years')
    ddm_payout_flag = row.get('ddm_payout_flag')

    if ddm_consecutive is not None and ddm_consecutive >= 10:
        tw.append(f'Track record of {ddm_consecutive} consecutive years of dividend payments')
    if ddm_payout_flag:
        hw.append('Dividend payout ratio appears stretched, which may limit sustainability')
    elif payout_ratio is not None:
        if 0 < payout_ratio < 0.50:
            tw.append('Healthy payout ratio leaves room for dividend growth and reinvestment')
        elif payout_ratio > 0.85:
            hw.append(f'High payout ratio ({_pct(payout_ratio)}) leaves thin coverage for the dividend')

    # --- Analyst consensus ---
    analyst_rec = row.get('analyst_rec')
    num_analysts = row.get('num_analysts')
    target_mean = row.get('target_mean')
    price = row.get('price')

    if target_mean and price and price > 0:
        upside = (target_mean - price) / price
        if upside > 0.25 and num_analysts and num_analysts >= 5:
            tw.append(f'Analyst consensus points to {_pct(upside)} upside from current levels')
        elif upside > 0.15 and num_analysts and num_analysts >= 3:
            tw.append(f'Analysts see meaningful upside ({_pct(upside)}) to their mean target')
        elif upside < -0.10:
            hw.append(f'Trading above the mean analyst price target by {_pct(upside)}')

    if analyst_rec:
        rec_lower = analyst_rec.lower().replace('_', ' ')
        if rec_lower in ('strong_buy', 'strong buy') and num_analysts and num_analysts >= 5:
            tw.append('Strong Buy consensus from a broad analyst coverage base')
        elif rec_lower in ('sell', 'strong_sell', 'strong sell'):
            hw.append(f'Analyst consensus is {rec_lower.replace("_", " ").title()}')

    return hw, tw


# ---------------------------------------------------------------------------
# Layer 4: Peer comparison (sector-relative percentile ranking)
# ---------------------------------------------------------------------------

def _peer_signals(row):
    """Generate peer comparison headwinds/tailwinds from sector percentiles."""
    hw, tw = [], []
    sector = row.get('sector', '')

    # ROIC vs peers
    roic_pctile = row.get('_peer_pctile_roic')
    if roic_pctile is not None:
        if roic_pctile >= 0.85:
            tw.append(f'ROIC ranks in the top quartile of {sector} peers')
        elif roic_pctile <= 0.15:
            hw.append(f'ROIC sits in the bottom quartile among {sector} peers')

    # Gross margin vs peers
    gm_pctile = row.get('_peer_pctile_gross_margin')
    if gm_pctile is not None:
        if gm_pctile >= 0.85:
            tw.append(f'Gross margin ranks among the highest in {sector}')
        elif gm_pctile <= 0.15:
            hw.append(f'Gross margin trails most {sector} peers')

    # Revenue growth vs peers
    rev_pctile = row.get('_peer_pctile_rev_cagr')
    if rev_pctile is not None:
        if rev_pctile >= 0.80:
            tw.append(f'Growing revenue faster than most {sector} companies')
        elif rev_pctile <= 0.20:
            hw.append(f'Revenue growth lags the majority of {sector} peers')

    # Leverage vs peers (lower is better, so invert)
    lev_pctile = row.get('_peer_pctile_nd_ebitda')
    if lev_pctile is not None:
        if lev_pctile >= 0.85:
            hw.append(f'More leveraged than most {sector} peers')
        elif lev_pctile <= 0.15:
            tw.append(f'Among the least leveraged companies in {sector}')

    # Piotroski vs peers
    pio_pctile = row.get('_peer_pctile_piotroski')
    if pio_pctile is not None:
        if pio_pctile >= 0.85:
            tw.append(f'Financial quality scores above most {sector} peers')
        elif pio_pctile <= 0.15:
            hw.append(f'Financial quality metrics lag {sector} peers')

    return hw, tw


# ---------------------------------------------------------------------------
# Layer 5: Balance sheet risk flags (goodwill, R&D, SGA)
# ---------------------------------------------------------------------------

def _risk_flag_signals(row):
    """Generate risk-related headwinds/tailwinds from balance sheet data."""
    hw, tw = [], []
    sector = row.get('sector', '')

    # Goodwill concentration
    gw_pct = row.get('goodwill_pct')
    gw_pctile = row.get('_peer_pctile_goodwill_pct')
    if gw_pct is not None:
        if gw_pct > 0.50:
            hw.append(f'Goodwill represents {_pct(gw_pct)} of total assets, creating impairment risk')
        elif gw_pct > 0.35:
            hw.append(f'Significant goodwill ({_pct(gw_pct)} of assets) from past acquisitions')
        elif gw_pct == 0 or gw_pct < 0.05:
            tw.append('Minimal goodwill on the balance sheet — low impairment risk')

    # R&D intensity
    rd = row.get('rd_intensity')
    rd_pctile = row.get('_peer_pctile_rd_intensity')
    if rd is not None and rd > 0:
        if rd_pctile is not None and rd_pctile >= 0.80:
            tw.append(f'R&D investment ({_pct(rd)} of revenue) is above {sector} peers, supporting innovation')
        elif rd_pctile is not None and rd_pctile <= 0.20 and rd > 0.01:
            hw.append(f'R&D spending ({_pct(rd)} of revenue) trails most {sector} peers')

    # SGA spike (potential legal/restructuring costs)
    sga_yoy = row.get('sga_yoy_change')
    if sga_yoy is not None:
        if sga_yoy > 0.20:
            hw.append(f'SGA costs surged {_pct(sga_yoy)} year-over-year, which may reflect legal, restructuring, or operational issues')
        elif sga_yoy > 0.12:
            hw.append(f'Above-average SGA growth ({_pct(sga_yoy)} YoY) warrants monitoring')

    # Beneish M-Score (already in stock_signals, but add context for near-flag)
    beneish_m = row.get('beneish_m')
    if beneish_m is not None and not row.get('beneish_flag'):
        if beneish_m > -1.78 and beneish_m <= -1.50:
            hw.append('Beneish M-Score near the manipulation threshold — earnings quality bears watching')

    return hw, tw


# ---------------------------------------------------------------------------
# Financial position summary (short prose paragraph)
# ---------------------------------------------------------------------------

def generate_financial_summary(row):
    """Build a comprehensive summary of the company's financial position.

    Covers: size & scale, revenue trends, profitability & margins,
    cash flow, balance sheet & debt, and capital returns.
    Returns a plain string.
    """
    parts = []
    name = row.get('company_name') or row.get('ticker', '?')
    sector = row.get('sector', '')

    def _ds(v):
        if v is None:
            return None
        av = abs(v)
        if av >= 1e12:
            return f'${av / 1e12:.1f}T'
        if av >= 1e9:
            return f'${av / 1e9:.1f}B'
        if av >= 1e6:
            return f'${av / 1e6:.0f}M'
        return f'${av:,.0f}'

    mcap = row.get('mcap')
    size_label = ''
    if mcap is not None:
        if mcap >= 200e9:
            size_label = 'mega-cap'
        elif mcap >= 10e9:
            size_label = 'large-cap'
        elif mcap >= 2e9:
            size_label = 'mid-cap'
        elif mcap >= 300e6:
            size_label = 'small-cap'
        else:
            size_label = 'micro-cap'

    # --- Sentence 1: Identity & scale ---
    revenue = row.get('revenue')
    s1 = []
    if size_label and mcap is not None:
        s1.append(f'{name} is a {size_label} {sector.lower()} company (market cap {_ds(mcap)})')
    if revenue is not None:
        s1.append(f'generating {_ds(revenue)} in annual revenue')
    if s1:
        parts.append(', '.join(s1) + '.')

    # --- Sentence 2: Revenue trends ---
    rev_cagr = row.get('rev_cagr')
    rev_cagr_5y = row.get('rev_cagr_5y')
    rev_cagr_10y = row.get('rev_cagr_10y')
    margin_trend = row.get('margin_trend')

    s2 = []
    if rev_cagr is not None:
        if rev_cagr > 0.08:
            s2.append(f'Revenue has grown at {rev_cagr * 100:.0f}% annually over the past 3 years')
        elif rev_cagr > 0.02:
            s2.append(f'Revenue has grown at a modest {rev_cagr * 100:.0f}% over 3 years')
        elif rev_cagr < -0.02:
            s2.append(f'Revenue has declined at {abs(rev_cagr) * 100:.0f}% annually over 3 years')
        else:
            s2.append('Revenue has been largely flat over the past 3 years')

    # Add longer-term context if trajectory differs
    if rev_cagr is not None and rev_cagr_5y is not None:
        diff = rev_cagr - rev_cagr_5y
        if diff > 0.04:
            s2.append('accelerating from its 5-year trend')
        elif diff < -0.04:
            s2.append('decelerating from its 5-year trend')

    if margin_trend is not None:
        if margin_trend > 0.02:
            s2.append('with margins expanding')
        elif margin_trend < -0.02:
            s2.append('with margins compressing')

    if s2:
        parts.append(', '.join(s2) + '.')

    # --- Sentence 3: Profitability ---
    gross_margin = row.get('gross_margin')
    operating_margin = row.get('operating_margin')
    roic = row.get('roic')
    spread = row.get('spread')
    roe = row.get('roe')

    s3 = []
    margin_items = []
    if gross_margin is not None:
        margin_items.append(f'{gross_margin * 100:.0f}% gross')
    if operating_margin is not None:
        margin_items.append(f'{abs(operating_margin) * 100:.0f}% operating')
    if margin_items:
        s3.append(f'Margins are {" / ".join(margin_items)}')

    if roic is not None and spread is not None:
        if spread > 0.05:
            s3.append(f'and the business earns a {roic * 100:.0f}% return on capital, well above its cost of capital')
        elif spread > 0:
            s3.append(f'with a {roic * 100:.0f}% return on capital that modestly exceeds its cost of capital')
        else:
            s3.append(f'though its {roic * 100:.0f}% return on capital falls short of its cost of capital')
    elif roic is not None:
        s3.append(f'with a {roic * 100:.0f}% return on invested capital')

    if s3:
        parts.append(', '.join(s3) + '.')

    # --- Sentence 4: Cash flow ---
    fcf = row.get('fcf')
    cash_conv = row.get('cash_conv')
    accruals = row.get('accruals')

    s4 = []
    if fcf is not None:
        if fcf > 0:
            s4.append(f'Free cash flow is {_ds(fcf)}')
        else:
            s4.append(f'The company is burning {_ds(abs(fcf))} in free cash flow')

    if cash_conv is not None:
        if cash_conv > 1.2:
            s4.append('with strong cash conversion above 1.0×')
        elif cash_conv < 0.5 and cash_conv >= 0:
            s4.append('though cash conversion is weak relative to reported earnings')

    if s4:
        parts.append(', '.join(s4) + '.')

    # --- Sentence 5: Balance sheet & debt ---
    nd_ebitda = row.get('nd_ebitda')
    int_cov = row.get('int_cov')
    de = row.get('de')
    cr = row.get('cr')
    piotroski = row.get('piotroski')

    s5 = []
    if nd_ebitda is not None:
        if nd_ebitda < 0:
            s5.append('The balance sheet carries net cash')
        elif nd_ebitda < 2.0:
            s5.append(f'Leverage is conservative at {nd_ebitda:.1f}× net debt/EBITDA')
        elif nd_ebitda < 4.0:
            s5.append(f'Leverage is moderate at {nd_ebitda:.1f}× net debt/EBITDA')
        else:
            s5.append(f'The balance sheet carries significant debt at {nd_ebitda:.1f}× net debt/EBITDA')

    if int_cov is not None:
        if int_cov < 3.0:
            s5.append(f'with thin {int_cov:.1f}× interest coverage')
        elif int_cov > 10:
            s5.append('with ample interest coverage')

    if piotroski is not None:
        if piotroski >= 7:
            s5.append(f'and financial momentum is strong (Piotroski {piotroski}/9)')
        elif piotroski <= 3:
            s5.append(f'and financial momentum is weak (Piotroski {piotroski}/9)')

    if s5:
        parts.append(', '.join(s5) + '.')

    # --- Sentence 6: Capital returns & profit pool ---
    div_yield = row.get('div_yield')
    buyback = row.get('share_buyback_rate')
    shareholder_yield = row.get('shareholder_yield')
    pp_mult = row.get('pp_multiple')

    s6 = []
    cap_items = []
    if div_yield is not None and div_yield > 0.01:
        cap_items.append(f'a {div_yield * 100:.1f}% dividend')
    if buyback is not None and buyback > 0.01 and buyback < 0.50:
        cap_items.append(f'{buyback * 100:.1f}% annual buybacks')
    if cap_items:
        s6.append('Capital returns include ' + ' and '.join(cap_items))
    elif shareholder_yield is not None and shareholder_yield < -0.02 and shareholder_yield > -0.50:
        s6.append(f'Net shareholder yield is negative at {shareholder_yield * 100:.1f}% due to dilution')

    if pp_mult is not None:
        if pp_mult > 1.5:
            s6.append(f'the company captures a disproportionate {pp_mult:.1f}× share of sector profits relative to its revenue')
        elif pp_mult < 0.5:
            s6.append(f'profit capture is below its revenue share ({pp_mult:.1f}× profit pool multiple)')

    if s6:
        parts.append('; '.join(s6) + '.')

    return ' '.join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_stock_narrative(row, sector_data=None, macro_regime_result=None,
                             commodity_data=None, sector_medians=None,
                             max_per_side=10):
    """Build contextual headwind/tailwind narrative for a single stock.

    Args:
        row: dict — full result row for the stock
        sector_data: dict[sector → ETF metrics] from MacroClient.fetch_sector_data()
        macro_regime_result: dict from assess_macro_regime() (may be None)
        commodity_data: dict from MacroClient.fetch_commodity_data() (may be None)
        sector_medians: dict[sector → median metrics] (may be None)
        max_per_side: int — cap on bullets per side to keep it readable

    Returns:
        (headwinds: list[str], tailwinds: list[str])
    """
    # Layer 1: Macro
    hw_macro, tw_macro = _macro_signals(macro_regime_result)

    # Layer 2: Sector
    hw_sector, tw_sector = _sector_signals(row, sector_data, macro_regime_result,
                                           commodity_data)

    # Layer 3: Stock-specific
    hw_stock, tw_stock = _stock_signals(row, sector_medians)

    # Layer 4: Peer comparison
    hw_peer, tw_peer = _peer_signals(row)

    # Layer 5: Balance sheet risk flags
    hw_risk, tw_risk = _risk_flag_signals(row)

    # Combine: stock signals first (most relevant), then peers, risk, sector, macro
    all_hw = hw_stock + hw_peer + hw_risk + hw_sector + hw_macro
    all_tw = tw_stock + tw_peer + tw_risk + tw_sector + tw_macro

    # Cap at max_per_side to keep the summary readable
    return all_hw[:max_per_side], all_tw[:max_per_side]
