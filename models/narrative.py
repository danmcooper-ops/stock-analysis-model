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
    """Generate per-company headwinds/tailwinds from fundamentals.

    Headwinds are emitted as dicts of shape {'text', 'sev', 'cat'} where:
      sev : 'red' = distress / thesis-breaking; 'amber' = watch item
      cat : 'valuation' | 'insider' | 'quality' | 'moat' | 'trend' | …
            (used downstream by generate_stock_narrative for deduplication)
    Tailwinds remain plain strings — the upside list doesn't need grading.

    Philosophy: the Financial Summary owns *levels* (ROIC, FCF, leverage,
    growth CAGR, margins). These bullets only surface **deltas**,
    **deteriorations**, and **context the summary doesn't cover** — earnings
    surprises, valuation vs. fair value, 52-week position, ownership
    transactions, dividend sustainability, analyst consensus, etc.
    """
    hw, tw = [], []

    def HW(text, sev='amber', cat=None, **extras):
        d = {'text': text, 'sev': sev, 'cat': cat}
        d.update(extras)
        hw.append(d)

    sector = row.get('sector', '')

    # === GROWTH signals (deltas only — summary owns CAGR levels) ===
    fund_growth = row.get('fundamental_growth')
    surprise_avg = row.get('surprise_avg')
    margin_trend = row.get('margin_trend')

    if fund_growth is not None:
        if fund_growth > 0.08:
            tw.append(f'Reinvestment rate supports {fund_growth * 100:.0f}% organic growth capacity — capital is compounding inside the business')
        elif fund_growth < 0.02:
            HW('Low reinvestment-driven growth suggests the compounding engine has stalled', sev='amber', cat='growth')

    if surprise_avg is not None:
        if surprise_avg > 0.05:
            tw.append(f'Consistently beating estimates by {_pct(surprise_avg)} — management is under-promising and over-delivering')
        elif surprise_avg > 0.02:
            tw.append('Recent earnings have come in above analyst expectations')
        elif surprise_avg < -0.05:
            HW(f'Earnings missing estimates by {_pct(surprise_avg)} on average — the first crack Graham would watch for', sev='red', cat='quality')
        elif surprise_avg < -0.03:
            HW(f'Earnings missing estimates by {_pct(surprise_avg)} on average', sev='amber', cat='quality')
        elif surprise_avg < 0:
            HW('Recent earnings have fallen short of analyst expectations', sev='amber', cat='quality')

    # (D) Trend: margin direction — a delta, distinct from the level the summary carries
    if margin_trend is not None:
        if margin_trend < -0.03:
            HW(f'Operating margins contracting {margin_trend * 100:+.1f}pp YoY — competitive pressure is eroding the franchise', sev='red', cat='trend')
        elif margin_trend < -0.01:
            HW(f'Operating margins softening {margin_trend * 100:+.1f}pp YoY — early warning on the moat', sev='amber', cat='trend')

    # === PROFIT POOL signals (peer-relative only — summary owns pp_mult level) ===
    pp_margin_adv = row.get('pp_margin_advantage')
    pp_rev_share = row.get('pp_revenue_share')
    pp_hhi = row.get('pp_sector_hhi')
    pp_mult = row.get('pp_multiple')

    if pp_margin_adv is not None:
        if pp_margin_adv > 0.08:
            tw.append(f'Operating margins {_pct(pp_margin_adv)} above sector median — the quantitative trace of pricing power')
        elif pp_margin_adv < -0.08:
            HW(f'Operating margins {_pct(pp_margin_adv)} below sector median — a cost disadvantage the moat cannot hide', sev='amber', cat='moat')

    if pp_hhi is not None:
        if pp_hhi > 0.25 and pp_rev_share and pp_rev_share > 0.10:
            tw.append('Operates in a concentrated industry where incumbents keep the profit pool — Buffett\u2019s favourite industry structure')
        elif pp_hhi < 0.10 and pp_mult is not None and pp_mult < 0.8:
            HW('Fragmented industry with below-average profit capture — no moat to protect margins', sev='amber', cat='moat')

    # --- Valuation context (A — dedup handled downstream; here we just emit) ---
    mos = row.get('mos')
    implied_vs_est = row.get('implied_vs_estimated')
    range_pos = row.get('range_52w_position')
    ee_vs_sector = row.get('_ee_vs_sector')

    if mos is not None:
        if mos > 0.30:
            tw.append(f'Trading at a {_pct(mos)} margin of safety to estimated fair value — exactly the cushion Graham demanded before buying')
        elif mos > 0.15:
            tw.append(f'Shares offer a {_pct(mos)} margin of safety against estimated fair value')
        elif mos < -0.30:
            HW(f'Price sits {_pct(mos)} above estimated fair value — no margin of safety left, and the cushion Graham demanded has been bid away', sev='red', cat='valuation', magnitude=abs(mos))
        elif mos < -0.15:
            HW(f'Price sits {_pct(mos)} above estimated fair value — no margin of safety left', sev='amber', cat='valuation', magnitude=abs(mos))
        elif mos < 0:
            HW('Priced near or above estimated fair value, so the margin of safety has been bid away', sev='amber', cat='valuation', magnitude=abs(mos))

    if implied_vs_est is not None:
        if implied_vs_est > 0.05:
            HW('Market price implies higher growth than fundamentals support — paying for optimism', sev='amber', cat='valuation', magnitude=implied_vs_est)
        elif implied_vs_est < -0.05:
            tw.append('Market is pricing in lower growth than the company has historically delivered')

    if range_pos is not None:
        if range_pos < 20:
            tw.append('Trading near the bottom of its 52-week range — Mr. Market is offering a discount')
        elif range_pos > 85:
            HW('Trading near 52-week highs — Mr. Market has already recognised the story', sev='amber', cat='valuation', magnitude=(range_pos - 85) / 15.0)

    if ee_vs_sector is not None:
        if ee_vs_sector < -0.20:
            tw.append(f'Valued at a {_pct(ee_vs_sector)} discount to sector median EV/EBITDA')
        elif ee_vs_sector > 0.30:
            HW(f'Carries a {_pct(ee_vs_sector)} premium to sector median EV/EBITDA', sev='amber', cat='valuation', magnitude=ee_vs_sector)

    # === MOAT consistency (summary owns spread/ROIC/gross-margin levels) ===
    roic_cv = row.get('roic_cv')
    if roic_cv is not None:
        if roic_cv < 0.15:
            tw.append('Returns on capital have been remarkably steady for 5 years — the durability Munger calls the real test of a moat')
        elif roic_cv > 0.40:
            HW('Volatile returns on capital suggest the moat is narrower than a snapshot ROIC implies', sev='amber', cat='moat')

    # (D) Trend: accruals building — Sloan's classic quality-of-earnings warning
    accruals = row.get('accruals')
    if accruals is not None and accruals > 0.10:
        HW(f'Accruals elevated at {accruals * 100:+.1f}% of assets — Sloan showed firms building earnings through accruals rather than cash systematically underperform', sev='red', cat='quality')
    elif accruals is not None and accruals > 0.06:
        HW(f'Accruals running hot at {accruals * 100:+.1f}% of assets — earnings quality bears watching', sev='amber', cat='quality')

    # === QUALITY signals (summary owns cash-conv/leverage/int-cov levels) ===
    piotroski = row.get('piotroski')
    altman_z = row.get('altman_z')
    altman_zone = row.get('altman_z_zone')
    beneish_flag = row.get('beneish_flag')

    if piotroski is not None:
        if piotroski >= 7:
            tw.append(f'Piotroski score {piotroski}/9 — financials improving across profitability, leverage and efficiency')
        elif piotroski <= 3:
            HW(f'Piotroski score {piotroski}/9 — the fundamentals are deteriorating on multiple fronts', sev='red', cat='quality')
        elif piotroski == 4:
            HW(f'Piotroski score {piotroski}/9 — below the median for a healthy business', sev='amber', cat='quality')

    if altman_zone == 'distress':
        HW(f'Altman Z in the distress zone ({altman_z:.1f}) — the kind of balance-sheet risk Graham told investors to avoid entirely', sev='red', cat='quality')
    elif altman_zone == 'grey':
        HW('Altman Z in the grey zone — financial health warrants monitoring', sev='amber', cat='quality')
    elif altman_zone == 'safe' and altman_z and altman_z > 4.0:
        tw.append('Altman Z comfortably in the safe zone — a fortress balance sheet by Graham\u2019s test')

    if beneish_flag:
        HW('Beneish M-Score flags potential earnings manipulation — Graham\u2019s question about whether the numbers are facts or opinions', sev='red', cat='quality')

    # --- Ownership & alignment ---
    short_pct = row.get('short_pct_float')
    founder_led = row.get('founder_led')

    if founder_led:
        tw.append('Founder-led — the alignment Buffett and Munger repeatedly say matters more than any incentive plan')

    if short_pct is not None:
        if short_pct > 0.15:
            HW(f'Very elevated short interest ({_pct(short_pct, 1)} of float) — sophisticated money is betting against the business', sev='red', cat='sentiment')
        elif short_pct > 0.10:
            HW(f'Elevated short interest ({_pct(short_pct, 1)} of float) — sophisticated money is betting the other way', sev='amber', cat='sentiment')
        elif short_pct > 0.05:
            HW(f'Above-average short interest at {_pct(short_pct, 1)} of float', sev='amber', cat='sentiment')

    # --- Insider transaction activity (B — dedup handled downstream) ---
    insider_buy_ratio = row.get('insider_buy_ratio')
    insider_net_value = row.get('insider_net_value')
    insider_buy_90d = (row.get('insider_buy_count_90d') or 0)
    insider_sell_90d = (row.get('insider_sell_count_90d') or 0)
    insider_txn_total = insider_buy_90d + insider_sell_90d

    if insider_buy_ratio is not None and insider_txn_total >= 2:
        if insider_buy_ratio > 0.60:
            tw.append(f'Insiders are net buyers ({_pct(insider_buy_ratio)} buy ratio) — the one signal Peter Lynch said there was only one reason for')
        elif insider_buy_ratio < 0.15:
            HW(f'Insiders are overwhelmingly selling ({_pct(1 - insider_buy_ratio)} sell ratio) — the people who know the business want less of it', sev='amber', cat='insider', ratio=insider_buy_ratio)
        elif insider_buy_ratio < 0.30:
            HW(f'Insiders are net sellers ({_pct(1 - insider_buy_ratio)} sell ratio) over the past year', sev='amber', cat='insider', ratio=insider_buy_ratio)

    if insider_net_value is not None:
        if insider_net_value > 500_000:
            tw.append(f'Large open-market insider purchases (net ${insider_net_value / 1e6:.1f}M) — management putting its own capital behind the business')
        elif insider_net_value < -10_000_000:
            HW(f'Heavy insider selling (net ${abs(insider_net_value) / 1e6:.1f}M)', sev='amber', cat='insider', dollars=insider_net_value)
        elif insider_net_value < -2_000_000:
            HW(f'Significant insider selling (net ${abs(insider_net_value) / 1e6:.1f}M)', sev='amber', cat='insider', dollars=insider_net_value)

    # --- Dividend sustainability ---
    payout_ratio = row.get('payout_ratio')
    ddm_consecutive = row.get('ddm_consecutive_years')
    ddm_payout_flag = row.get('ddm_payout_flag')

    if ddm_consecutive is not None and ddm_consecutive >= 10:
        tw.append(f'{ddm_consecutive} consecutive years of dividend payments — the operating record Graham looked for before any dividend claim could be trusted')
    if ddm_payout_flag:
        HW('Payout ratio looks stretched — the dividend\u2019s own margin of safety is thin', sev='red', cat='dividend')
    elif payout_ratio is not None:
        if 0 < payout_ratio < 0.50:
            tw.append('Healthy payout ratio leaves room for both dividend growth and reinvestment')
        elif payout_ratio > 0.95:
            HW(f'Payout ratio of {_pct(payout_ratio)} leaves essentially no coverage if earnings wobble', sev='red', cat='dividend')
        elif payout_ratio > 0.85:
            HW(f'Payout ratio of {_pct(payout_ratio)} leaves thin coverage if earnings wobble', sev='amber', cat='dividend')

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
            HW(f'Trading above the mean analyst price target by {_pct(upside)}', sev='amber', cat='valuation', magnitude=abs(upside))

    if analyst_rec:
        rec_lower = analyst_rec.lower().replace('_', ' ')
        if rec_lower in ('strong_buy', 'strong buy') and num_analysts and num_analysts >= 5:
            tw.append('Strong Buy consensus from a broad analyst coverage base')
        elif rec_lower in ('sell', 'strong_sell', 'strong sell'):
            HW(f'Analyst consensus is {rec_lower.replace("_", " ").title()}', sev='red', cat='sentiment')

    return hw, tw


# ---------------------------------------------------------------------------
# Layer 4: Peer comparison (sector-relative percentile ranking)
# ---------------------------------------------------------------------------

def _peer_signals(row):
    """Generate peer comparison headwinds/tailwinds from sector percentiles.

    Headwinds returned as dicts {text, sev, cat}. Tailwinds as strings.
    """
    hw, tw = [], []
    sector = row.get('sector', '')

    def HW(text, sev='amber', cat='peer'):
        hw.append({'text': text, 'sev': sev, 'cat': cat})

    # ROIC vs peers — frames moat against the competition
    roic_pctile = row.get('_peer_pctile_roic')
    if roic_pctile is not None:
        if roic_pctile >= 0.85:
            tw.append(f'ROIC sits in the top quartile of {sector} peers — the moat is wider than the neighbourhood')
        elif roic_pctile <= 0.10:
            HW(f'ROIC sits in the bottom decile among {sector} peers — no moat advantage to speak of', sev='red', cat='moat')
        elif roic_pctile <= 0.15:
            HW(f'ROIC sits in the bottom quartile among {sector} peers — no moat advantage to speak of', sev='amber', cat='moat')

    # Gross margin vs peers — quantitative trace of pricing power
    gm_pctile = row.get('_peer_pctile_gross_margin')
    if gm_pctile is not None:
        if gm_pctile >= 0.85:
            tw.append(f'Gross margin tops the {sector} peer set — the pricing power Buffett prizes is actually present here')
        elif gm_pctile <= 0.15:
            HW(f'Gross margin trails most {sector} peers — competitors appear to have the pricing hand', sev='amber', cat='moat')

    # Revenue growth vs peers — taking share
    rev_pctile = row.get('_peer_pctile_rev_cagr')
    if rev_pctile is not None:
        if rev_pctile >= 0.80:
            tw.append(f'Growing faster than most {sector} peers — taking share inside the profit pool')
        elif rev_pctile <= 0.10:
            HW(f'Revenue growth in the bottom decile of {sector} peers — losing share inside the profit pool', sev='red', cat='growth')
        elif rev_pctile <= 0.20:
            HW(f'Revenue growth lags most {sector} peers — losing share inside the profit pool', sev='amber', cat='growth')

    # Leverage vs peers (lower is better, so invert)
    lev_pctile = row.get('_peer_pctile_nd_ebitda')
    if lev_pctile is not None:
        if lev_pctile >= 0.85:
            HW(f'Carries more leverage than most {sector} peers — a smaller cushion for the bad year Buffett always plans for', sev='amber', cat='balance_sheet')
        elif lev_pctile <= 0.15:
            tw.append(f'Among the least levered in {sector} — the fortress balance sheet in its peer set')

    # Piotroski vs peers
    pio_pctile = row.get('_peer_pctile_piotroski')
    if pio_pctile is not None:
        if pio_pctile >= 0.85:
            tw.append(f'Financial quality scores above most {sector} peers')
        elif pio_pctile <= 0.15:
            HW(f'Financial quality metrics lag {sector} peers', sev='amber', cat='quality')

    return hw, tw


# ---------------------------------------------------------------------------
# Layer 5: Balance sheet risk flags (goodwill, R&D, SGA)
# ---------------------------------------------------------------------------

def _risk_flag_signals(row):
    """Generate risk-related headwinds/tailwinds from balance sheet data.

    Headwinds returned as {text, sev, cat} dicts; tailwinds as strings.
    """
    hw, tw = [], []
    sector = row.get('sector', '')

    def HW(text, sev='amber', cat='risk'):
        hw.append({'text': text, 'sev': sev, 'cat': cat})

    # Goodwill concentration
    gw_pct = row.get('goodwill_pct')
    if gw_pct is not None:
        if gw_pct > 0.50:
            HW(f'Goodwill represents {_pct(gw_pct)} of total assets — a large impairment risk hanging over book value', sev='red', cat='balance_sheet')
        elif gw_pct > 0.35:
            HW(f'Significant goodwill ({_pct(gw_pct)} of assets) from past acquisitions — the kind of soft asset Graham discounted to zero', sev='amber', cat='balance_sheet')
        elif gw_pct == 0 or gw_pct < 0.05:
            tw.append('Minimal goodwill on the balance sheet — low impairment risk')

    # R&D intensity
    rd = row.get('rd_intensity')
    rd_pctile = row.get('_peer_pctile_rd_intensity')
    if rd is not None and rd > 0:
        if rd_pctile is not None and rd_pctile >= 0.80:
            tw.append(f'R&D investment ({_pct(rd)} of revenue) is above {sector} peers, supporting innovation')
        elif rd_pctile is not None and rd_pctile <= 0.20 and rd > 0.01:
            HW(f'R&D spending ({_pct(rd)} of revenue) trails most {sector} peers — under-investing in the moat', sev='amber', cat='moat')

    # SGA spike (potential legal/restructuring costs) — a trend signal
    sga_yoy = row.get('sga_yoy_change')
    if sga_yoy is not None:
        if sga_yoy > 0.20:
            HW(f'SGA costs surged {_pct(sga_yoy)} YoY — may reflect legal, restructuring, or operational issues', sev='red', cat='trend')
        elif sga_yoy > 0.12:
            HW(f'Above-average SGA growth ({_pct(sga_yoy)} YoY) warrants monitoring', sev='amber', cat='trend')

    # Beneish M-Score near-threshold
    beneish_m = row.get('beneish_m')
    if beneish_m is not None and not row.get('beneish_flag'):
        if beneish_m > -1.78 and beneish_m <= -1.50:
            HW('Beneish M-Score near the manipulation threshold — earnings quality bears watching', sev='amber', cat='quality')

    return hw, tw


# ---------------------------------------------------------------------------
# Layer 6: News & sentiment signals (E)
# ---------------------------------------------------------------------------

def _news_signals(row):
    """Surface narrative risks from news sentiment and layoff flags.

    Returns (hw_dicts, tw_strings). Only emits signals when the data is
    present and meaningful — silent when news sentiment is positive/neutral
    with no layoff flag.
    """
    hw, tw = [], []

    def HW(text, sev='amber', cat='news'):
        hw.append({'text': text, 'sev': sev, 'cat': cat})

    ns = row.get('news_sentiment') or {}
    if isinstance(ns, dict):
        label = ns.get('label')
        score = ns.get('score')
        bearish = ns.get('bearish_pct')
        article_ct = ns.get('article_count') or 0
        # Require enough articles for the signal to be meaningful
        if article_ct >= 5:
            if score is not None and score < -0.20:
                HW(f'News sentiment sharply negative ({score:+.2f}) across {article_ct} recent articles — the narrative has turned', sev='red', cat='news')
            elif score is not None and score < -0.10:
                HW(f'News sentiment running negative ({score:+.2f}) across recent coverage', sev='amber', cat='news')
            elif label == 'Negative':
                HW(f'News tone classified negative across {article_ct} recent articles', sev='amber', cat='news')
            elif bearish is not None and bearish > 0.50:
                HW(f'{_pct(bearish, 0)} of recent news is bearish — worth reading before sizing', sev='amber', cat='news')

    if row.get('layoff_news_signal'):
        HW('Recent layoff news — often the market\u2019s first signal that growth is under pressure or costs are being rebased', sev='amber', cat='news')

    return hw, tw


# ---------------------------------------------------------------------------
# Layer 7: Sector-level thesis-breaker / fat-tail risk (F)
# ---------------------------------------------------------------------------

# Map sector → the single structural risk that most plausibly breaks the
# long-run thesis for companies in that sector. Sourced from Buffett/Graham
# commentary on what actually kills a moat over a decade-plus horizon.
# Educational primer on how each sector actually makes money, plus a
# "business cycle" line describing where in the macro cycle the sector
# tends to outperform and where it struggles. Used to give the reader a
# framework before diving into the company-specific numbers.
_SECTOR_EDUCATION = {
    'Technology': {
        'model': (
            'Technology profit pools are built on intangible assets — software, data, '
            'and network effects — where marginal cost trends toward zero and the '
            'winner often takes most of the pie. Gross margins above 60% and '
            'scalable distribution are the norm, which means scale compounds faster '
            'than in any other sector, but switching costs and platform lock-in are '
            'what sustain the moat after growth slows.'
        ),
        'cycle': (
            'Cyclically, Technology is an early-to-mid-cycle outperformer: it leads '
            'coming out of recessions as enterprises restart capex, then re-rates '
            'again in the late expansion as liquidity chases long-duration growth. '
            'It is highly sensitive to real interest rates — rising rates compress '
            'the present value of distant cash flows, which is why mega-cap tech '
            'can sell off even with unchanged fundamentals.'
        ),
    },
    'Communication Services': {
        'model': (
            'Communication Services covers advertising platforms, telecom carriers, '
            'and media/content franchises. The advertising sub-sector earns a toll '
            'on attention and is highly operating-leveraged, while telecom is a '
            'capital-intensive utility-like business whose returns hinge on spectrum '
            'and tower economics. The profit pool skews sharply toward whoever '
            'controls the attention funnel or the last-mile pipe.'
        ),
        'cycle': (
            'The ad-driven pieces behave cyclically — ad budgets are among the first '
            'line items cut in a downturn and among the first restored in recovery — '
            'while the telecom subscription businesses are defensive and late-cycle. '
            'Investors typically see the sector rally on rate cuts and slump when '
            'corporate marketing budgets tighten.'
        ),
    },
    'Consumer Cyclical': {
        'model': (
            'Consumer Cyclical businesses — autos, retail, restaurants, travel, '
            'apparel — earn most of their profits in the late stages of an expansion '
            'and give them back in recessions. Operating leverage cuts both ways: '
            'high fixed costs amplify good times and wipe out margins when revenue '
            'dips. The long-run winners are those with brand pricing power, '
            'store-level unit economics, or low-cost operations that hold up through '
            'the cycle.'
        ),
        'cycle': (
            'This is the most cyclical sector in the market: it leads coming out of '
            'recessions (consumers unlock pent-up demand and credit loosens), peaks '
            'mid-cycle as confidence and wages run hot, and lags hard into downturns '
            'as discretionary purchases collapse. Watch unemployment and real wage '
            'growth — those are the upstream indicators for every name in the group.'
        ),
    },
    'Consumer Defensive': {
        'model': (
            'Consumer Defensive — packaged food, beverages, household goods, '
            'tobacco — lives off repeat purchases and brand equity built over '
            'decades. Volume growth is low single-digit, so returns come from '
            'pricing power, distribution scale against retailers, and disciplined '
            'SG&A. The profit pool is stable year-to-year, which makes the sector '
            'a bond proxy in valuation terms and a favorite of long-duration investors.'
        ),
        'cycle': (
            'Defensively counter-cyclical: the sector outperforms late-cycle and in '
            'recessions when investors rotate to earnings stability, and lags in '
            'early-cycle rallies when risk assets take the lead. It correlates '
            'inversely with equity risk appetite and sometimes trades like a '
            'duration proxy, rallying when rates fall.'
        ),
    },
    'Energy': {
        'model': (
            'Energy profit pools are dictated by commodity prices the producer '
            'does not set — oil, gas, refined products — and by position on the '
            'cost curve. In an upcycle every barrel is profitable; in a downcycle '
            'only the lowest-cost operators survive. Integrated majors smooth the '
            'cycle across upstream, midstream, and downstream, while pure E&P '
            'names offer leverage to the commodity with more extreme outcomes.'
        ),
        'cycle': (
            'Energy runs on its own commodity cycle that only loosely tracks the '
            'broader economy: it tends to outperform late-cycle when inflation and '
            'demand both run hot, and collapses in recessions as demand destruction '
            'meets fixed supply. The sector is a natural inflation hedge and is '
            'often the last to participate in early-cycle recoveries.'
        ),
    },
    'Financial Services': {
        'model': (
            'Financial Services earns a spread — between what it pays for capital '
            'and what it charges for using it — plus fees on transactions and '
            'advice. Banks make money when rates are rising and the yield curve '
            'is steep; insurers earn underwriting profits plus a float they invest; '
            'asset managers collect a percentage of AUM. Leverage is extreme '
            'across the board, so risk management and credit underwriting are '
            'what separate long-term compounders from cyclical traps.'
        ),
        'cycle': (
            'Banks are mid-to-late-cycle winners while the yield curve is steep and '
            'credit losses are contained; they underperform sharply in recessions '
            'when loan losses spike. Asset managers and exchanges track equity '
            'market levels, so they lead in bull markets. Insurers are the most '
            'defensive corner, protected by diversified float and slow-moving '
            'reserve releases.'
        ),
    },
    'Healthcare': {
        'model': (
            'Healthcare spans pharma (patent-protected monopolies that expire on '
            'a schedule), medical devices (engineering-driven franchises with '
            'sticky hospital relationships), managed care (premium spread businesses '
            'regulated at the state level), and services. Profits cluster around '
            'proprietary innovation and around the entities that sit between the '
            'patient and the payer. The sector is defensive against GDP but exposed '
            'to policy risk.'
        ),
        'cycle': (
            'Healthcare is classically defensive and late-cycle: demand is '
            'relatively inelastic to GDP, so the sector outperforms into slowdowns '
            'and recessions. It lags in early-cycle rallies when risk-on rotation '
            'favors cyclicals, and it can be disrupted by policy headlines more '
            'than by the business cycle itself.'
        ),
    },
    'Industrials': {
        'model': (
            'Industrials includes machinery, aerospace, defense, transportation, '
            'and business services — capital-goods businesses that live on order '
            'backlogs and installed-base service revenue. Margins are thinner than '
            'tech but the installed base acts like an annuity: once a customer '
            'standardizes on a supplier, switching costs are real. The best '
            'operators earn premium returns through aftermarket services, not '
            'equipment sales.'
        ),
        'cycle': (
            'Early-cycle: industrials rip out of recessions as inventory rebuilds, '
            'capex thaws, and order books refill. The sector peaks mid-expansion '
            'and rolls over first when PMIs weaken — the ISM manufacturing index '
            'is the single best leading indicator for the group. Defense is the '
            'one counter-cyclical pocket, tied to government budgets rather than '
            'private capex.'
        ),
    },
    'Basic Materials': {
        'model': (
            'Basic Materials — mining, chemicals, steel, paper — is a pure '
            'commodity game where the cost curve determines everything. Position '
            'on the curve, orebody quality, and balance-sheet strength decide who '
            'survives the trough. Specialty chemicals are the one corner of the '
            'sector where formulation IP and customer-specific engineering can '
            'produce a real moat.'
        ),
        'cycle': (
            'Deeply cyclical and often commodity-led: the sector outperforms '
            'early-to-mid-cycle as industrial demand recovers and commodity prices '
            'rebuild, and collapses fastest in recessions as inventory destocks '
            'cascade through the supply chain. China industrial production and the '
            'dollar are usually better indicators than domestic GDP.'
        ),
    },
    'Utilities': {
        'model': (
            'Utilities are regulated monopolies that earn a pre-set return on '
            'invested capital, approved by state commissions in rate cases. '
            'Growth comes from the rate base (the physical capital deployed), '
            'so capex discipline and regulatory relationships matter more than '
            'unit volume. The sector trades like a long-duration bond — safe '
            'cash flows, but highly sensitive to interest rates.'
        ),
        'cycle': (
            'Defensive and late-cycle: utilities outperform when growth slows and '
            'rates fall, and lag in early-cycle rallies and rising-rate regimes. '
            'Because the dividend yield is the largest part of the total return, '
            'the sector trades inversely to the 10-year Treasury — rising yields '
            'are its biggest enemy regardless of operating results.'
        ),
    },
    'Real Estate': {
        'model': (
            'Real estate, mostly via REITs, earns a spread between property-level '
            'cash yields (NOI) and the cost of debt used to buy the buildings. '
            'Returns depend on property type, tenant quality, lease duration, and '
            'refinancing terms. Because REITs distribute most of their earnings, '
            'growth is funded externally, which makes access to capital a key '
            'competitive weapon and refinancing risk the primary downside.'
        ),
        'cycle': (
            'Rate-sensitive and mid-to-late-cycle: REITs rally when the 10-year '
            'falls and struggle when real rates rise, because cap-rate compression '
            'drives most of the total return. They tend to outperform defensively '
            'in slowdowns as long as tenant credit holds, and crack hardest in '
            'stagflation — high rates, weak growth — which punishes both sides of '
            'the NOI-minus-financing equation.'
        ),
    },
    'Consumer Staples': {
        'model': (
            'Consumer Staples (same playbook as Consumer Defensive) relies on '
            'repeat purchase and brand loyalty for pricing power. Volume growth '
            'is anemic, so returns come from taking price, expanding internationally, '
            'and disciplined cost control.'
        ),
        'cycle': (
            'Defensive and late-cycle. The sector rotates in as the cycle matures '
            'and investors prioritize earnings stability over growth, and lags '
            'whenever risk assets are bid. A dollar-denominated profit base also '
            'makes FX and commodity input costs a meaningful swing factor.'
        ),
    },
}


_SECTOR_THESIS_RISKS = {
    'Technology': (
        'Thesis-breaker: antitrust and platform-regulation risk — the larger the network effect, '
        'the larger the regulatory target on its back, and a single consent decree can reset the moat.'
    ),
    'Communication Services': (
        'Thesis-breaker: advertising-cycle exposure and content/regulatory risk — '
        'a single platform policy shift or ad-spend contraction cascades through the P&L quickly.'
    ),
    'Consumer Cyclical': (
        'Thesis-breaker: consumer demand is fundamentally cyclical — '
        'a recession compresses both unit volume and pricing power, and operating leverage cuts both ways.'
    ),
    'Consumer Defensive': (
        'Thesis-breaker: slow-motion taste and channel shifts — private label, Gen-Z health preferences, '
        'or a DTC disruption can erode decade-old brand moats without a single dramatic quarter.'
    ),
    'Energy': (
        'Thesis-breaker: the commodity cycle — revenue and free cash flow are ultimately hostage to '
        'a price the company does not set, and the long-run energy-transition arc adds a structural overhang.'
    ),
    'Financial Services': (
        'Thesis-breaker: credit cycles and regulatory capital — losses accumulate invisibly in good years '
        'and appear all at once, and Buffett warns the time to worry is when underwriting standards are loosest.'
    ),
    'Healthcare': (
        'Thesis-breaker: patent cliffs, drug-pricing reform, and reimbursement pressure — '
        'a single regulatory change can reset the pricing power that justifies the multiple.'
    ),
    'Industrials': (
        'Thesis-breaker: cyclicality and capital intensity — order books turn on a dime, '
        'and the heavy fixed cost base that creates the moat also magnifies the downturn.'
    ),
    'Basic Materials': (
        'Thesis-breaker: commodity price volatility and cost-curve position — '
        'in a downturn only the lowest-cost producer has a business, and the rest carry the overhead.'
    ),
    'Utilities': (
        'Thesis-breaker: regulatory rate-case risk and rising cost of capital — '
        'the allowed return is set by commissions, and the business lives or dies by that negotiation.'
    ),
    'Real Estate': (
        'Thesis-breaker: interest-rate sensitivity and tenant-credit risk — '
        'cap rates and refinancing costs move faster than the 10-year leases can absorb.'
    ),
}


def _thesis_breaker_signal(row):
    """Always-on forward-looking fat-tail risk, one per company, by sector.

    Returns a dict {text, sev, cat} or None if sector unmapped.
    """
    sector = row.get('sector', '')
    msg = _SECTOR_THESIS_RISKS.get(sector)
    if not msg:
        return None
    return {'text': msg, 'sev': 'amber', 'cat': 'thesis_risk'}


# ---------------------------------------------------------------------------
# Financial position summary (short prose paragraph)
# ---------------------------------------------------------------------------

def generate_financial_summary(row):
    """Build a Graham/Buffett/Munger-style analytical summary of the business.

    Walks through scale, revenue durability, moat & profitability, owner earnings,
    balance sheet conservatism, and capital return discipline — framed as a
    value investor's verdict rather than a recitation of metrics.
    Returns a list of bullet strings (one per dimension), suitable for
    rendering as a <ul> in the report. Empty dimensions are skipped.
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

    revenue = row.get('revenue')
    rev_cagr = row.get('rev_cagr')
    rev_cagr_10y = row.get('rev_cagr_10y')
    margin_trend = row.get('margin_trend')
    gross_margin = row.get('gross_margin')
    roic = row.get('roic')
    spread = row.get('spread')
    wacc = row.get('wacc')
    fcf = row.get('fcf')
    fcf_margin = row.get('fcf_margin')
    cash_conv = row.get('cash_conv')
    accruals = row.get('accruals')
    nd_ebitda = row.get('nd_ebitda')
    int_cov = row.get('int_cov')
    shareholder_yield = row.get('shareholder_yield')
    insider_pct = row.get('insider_pct')
    sbc_pct_rev = row.get('sbc_pct_rev')
    pp_mult = row.get('pp_multiple')

    # --- Sentence 1: What the business is and how big ---
    if size_label and mcap is not None:
        scale_sent = f'{name} is a {size_label} {sector.lower()} business with a {_ds(mcap)} market cap'
        if revenue is not None:
            scale_sent += f' on {_ds(revenue)} of annual revenue'
        parts.append(scale_sent + '.')

    # --- Sentence 2: Growth story ---
    if rev_cagr is not None or rev_cagr_10y is not None:
        growth_pieces = []
        if rev_cagr is not None:
            growth_pieces.append(f'{rev_cagr * 100:+.0f}% over the past three years')
        if rev_cagr_10y is not None:
            growth_pieces.append(f'{rev_cagr_10y * 100:+.0f}% annually across the full decade')
        growth_sent = 'The top line has compounded at ' + ' and '.join(growth_pieces)
        if rev_cagr is not None:
            if rev_cagr > 0.12:
                growth_sent += ', the kind of sustained demand persistence Buffett looks for in a compounder'
            elif rev_cagr > 0.05:
                growth_sent += ', a respectable pace for a business of this size'
            elif rev_cagr > 0:
                growth_sent += ', which puts it in mature, slow-growth territory'
            elif rev_cagr > -0.02:
                growth_sent += ' \u2014 essentially flat, and Buffett would note that a business treading water is rarely a path to compounding'
            else:
                growth_sent += ', and Buffett warns that a shrinking top line is the hardest headwind any business can overcome'
        growth_sent += '.'
        if margin_trend is not None and abs(margin_trend) > 0.02:
            if margin_trend > 0:
                growth_sent += f' Operating margins are expanding by {margin_trend * 100:+.1f}pp year-over-year, which is one of the strongest signals that the moat is deepening.'
            else:
                growth_sent += f' Operating margins are contracting by {margin_trend * 100:+.1f}pp, the first warning that competitive pressure is eroding the franchise.'
        parts.append(growth_sent)

    # --- Sentence 3: The moat — ROIC, WACC, pricing power ---
    if roic is not None and spread is not None and wacc is not None:
        moat_sent = f'Returns on capital run at {roic * 100:.0f}% against a {wacc * 100:.0f}% cost of capital'
        if spread > 0.08 and roic > 0.18:
            moat_sent += f' \u2014 a {spread * 100:.0f}-point spread that is the clearest signature of an economic moat Buffett teaches us to look for'
        elif spread > 0.03:
            moat_sent += f', a {spread * 100:.0f}-point spread that creates value, though narrow enough that competitive pressure could close it'
        elif spread > 0:
            moat_sent += ', which means the business barely earns its keep economically'
        else:
            moat_sent += ', which in Buffett\u2019s language means the business is destroying value with every dollar it reinvests'
        if gross_margin is not None:
            if gross_margin > 0.50:
                moat_sent += f'. Gross margins of {gross_margin * 100:.0f}% confirm the genuine pricing power Buffett prizes above all else'
            elif gross_margin < 0.25:
                moat_sent += f', and {gross_margin * 100:.0f}% gross margins are typical of commodity economics where competitors can replicate the product cheaply'
            else:
                moat_sent += f', with gross margins sitting at {gross_margin * 100:.0f}%'
        parts.append(moat_sent + '.')
    elif gross_margin is not None:
        parts.append(f'Gross margins of {gross_margin * 100:.0f}% are the starting point for assessing pricing power.')

    # --- Sentence 4: Owner earnings ---
    if fcf is not None:
        if fcf > 0:
            fcf_sent = f'The business throws off {_ds(fcf)} in free cash flow'
            if fcf_margin is not None:
                fcf_sent += f' \u2014 roughly {fcf_margin * 100:.0f} cents on every revenue dollar'
            fcf_sent += ', which is the owner-earnings figure that actually belongs to shareholders'
            if cash_conv is not None:
                if cash_conv >= 1.0:
                    fcf_sent += f', and cash conversion sits at {cash_conv:.2f}\u00d7 so those profits are backed by real cash'
                elif cash_conv >= 0.7:
                    fcf_sent += f', with adequate {cash_conv:.2f}\u00d7 cash conversion'
                elif cash_conv >= 0:
                    fcf_sent += f'. Cash conversion is thin at {cash_conv:.2f}\u00d7, however, which raises Graham\u2019s oldest question: are these earnings opinions or facts?'
            parts.append(fcf_sent + '.')
        else:
            parts.append(f'The company is burning through {_ds(abs(fcf))} of cash, and Graham insisted on positive owner earnings before any valuation conversation begins.')

    if accruals is not None and abs(accruals) > 0.08:
        parts.append(f'Accruals are elevated at {accruals * 100:+.1f}% of assets, and Sloan showed firms building earnings through accruals rather than cash systematically underperform.')

    # --- Sentence 5: Balance sheet ---
    if nd_ebitda is not None or int_cov is not None:
        bs_pieces = []
        if nd_ebitda is not None:
            if nd_ebitda < 0:
                bs_pieces.append('the balance sheet carries net cash, which is the gold standard Buffett demands so the business never has to make desperate decisions in a bad year')
            elif nd_ebitda < 2.0:
                bs_pieces.append(f'leverage is comfortably within Buffett\u2019s tolerance at {nd_ebitda:.1f}\u00d7 net debt/EBITDA')
            elif nd_ebitda < 4.0:
                bs_pieces.append(f'debt is moderate at {nd_ebitda:.1f}\u00d7 net debt/EBITDA, serviceable in normal conditions but worth watching through a downturn')
            else:
                bs_pieces.append(f'debt is heavy at {nd_ebitda:.1f}\u00d7 net debt/EBITDA, and Munger reminds us that liquor, ladies, and leverage are the three things that ruin people')
        if int_cov is not None:
            if int_cov >= 100:
                pass  # effectively unlimited; don't bother
            elif int_cov > 12:
                bs_pieces.append(f'interest coverage of {int_cov:.0f}\u00d7 is fortress-like')
            elif int_cov >= 3:
                bs_pieces.append(f'interest coverage runs at {int_cov:.1f}\u00d7')
            else:
                bs_pieces.append(f'thin {int_cov:.1f}\u00d7 interest coverage leaves no room for a bad year')
        if bs_pieces:
            parts.append('On the balance sheet, ' + ', and '.join(bs_pieces) + '.')

    # --- Sentence 6: Capital allocation & ownership ---
    cap_pieces = []
    if shareholder_yield is not None:
        if shareholder_yield > 0.05:
            cap_pieces.append(f'management returns a meaningful {shareholder_yield * 100:.1f}% of market cap to owners through dividends and buybacks, which is Buffett\u2019s favourite signal of disciplined capital allocation')
        elif shareholder_yield > 0.02:
            cap_pieces.append(f'shareholder yield of {shareholder_yield * 100:.1f}% reflects a genuine commitment to returning capital rather than empire-building')
        elif shareholder_yield >= 0:
            cap_pieces.append(f'shareholder yield is only {shareholder_yield * 100:.1f}%, a thin return of capital')
        elif shareholder_yield > -0.50:
            cap_pieces.append(f'net dilution runs at {abs(shareholder_yield) * 100:.1f}% annually, transferring value from owners to stock-based comp in the way Buffett has repeatedly criticised')
    if sbc_pct_rev is not None and sbc_pct_rev > 0.10:
        cap_pieces.append(f'stock-based comp at {sbc_pct_rev * 100:.0f}% of revenue is silent dilution the value investor has to account for')
    if insider_pct is not None:
        if insider_pct > 0.10:
            cap_pieces.append(f'insiders hold {insider_pct * 100:.0f}% of the shares, so management thinks like owners because it is one \u2014 exactly the alignment Munger looks for')
        elif insider_pct >= 0.02:
            cap_pieces.append(f'insiders hold {insider_pct * 100:.1f}% of the shares')
        elif insider_pct >= 0:
            cap_pieces.append('insider ownership is negligible, so management has little personal stake in the long-term outcome')
    if pp_mult is not None:
        if pp_mult > 1.5:
            cap_pieces.append(f'within its sector the business captures {pp_mult:.1f}\u00d7 its fair share of profits, a structural advantage visible in the numbers')
        elif pp_mult < 0.5:
            cap_pieces.append(f'profit capture is only {pp_mult:.1f}\u00d7 the fair share, suggesting no structural edge inside its industry')
    if cap_pieces:
        # Capitalise first word
        line = cap_pieces[0][0].upper() + cap_pieces[0][1:]
        rest = cap_pieces[1:]
        if rest:
            line += ', and ' + ', and '.join(rest)
        parts.append(line + '.')

    return parts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _to_hw_dict(item, default_sev='amber', default_cat=None):
    """Coerce a headwind entry to the {text, sev, cat} dict shape.

    Accepts either a plain string (from legacy layers like sector/macro) or
    an already-structured dict. Unknown extras on dicts are preserved.
    """
    if isinstance(item, dict):
        d = dict(item)
        d.setdefault('sev', default_sev)
        d.setdefault('cat', default_cat)
        return d
    return {'text': str(item), 'sev': default_sev, 'cat': default_cat}


def _dedupe_valuation_cluster(hw_dicts):
    """(A) Collapse multiple 'valuation' headwinds into one primary + confirmation.

    When MOS, implied-vs-est, 52w-high, and EV/EBITDA premium all fire, they
    are saying the same thing in different ways. Keep the strongest by
    magnitude and append a compact "confirmed by N more measures" note.
    """
    val = [h for h in hw_dicts if h.get('cat') == 'valuation']
    if len(val) <= 1:
        return hw_dicts
    # Strongest = largest magnitude, with a tiebreaker preferring red>amber
    def _score(h):
        sev_weight = 1.0 if h.get('sev') == 'red' else 0.0
        return (sev_weight, float(h.get('magnitude') or 0))
    val_sorted = sorted(val, key=_score, reverse=True)
    primary = dict(val_sorted[0])
    extra_count = len(val_sorted) - 1
    if extra_count > 0:
        primary['text'] = (
            primary['text'].rstrip('.') +
            f' (confirmed by {extra_count} other valuation measure' +
            ('s' if extra_count != 1 else '') + ').'
        ).rstrip('.')
    # Preserve order of the non-valuation entries; drop the others
    dropped_ids = {id(h) for h in val_sorted[1:]}
    out = []
    inserted = False
    for h in hw_dicts:
        if h.get('cat') == 'valuation':
            if not inserted:
                out.append(primary)
                inserted = True
            continue
        out.append(h)
    return out


def _dedupe_insider_cluster(hw_dicts):
    """(B) Collapse insider-selling ratio + net-value bullets into one when both fire."""
    insider = [h for h in hw_dicts if h.get('cat') == 'insider']
    if len(insider) <= 1:
        return hw_dicts
    # Find the ratio entry and the dollar entry
    ratio_entry = next((h for h in insider if 'ratio' in h), None)
    dollar_entry = next((h for h in insider if 'dollars' in h), None)
    if ratio_entry and dollar_entry:
        dollars = abs(dollar_entry.get('dollars') or 0)
        ratio = ratio_entry.get('ratio') or 0
        merged_text = (
            f'Insiders are net sellers ({_pct(1 - ratio)} sell ratio, net ${dollars / 1e6:.1f}M) — '
            f'the people closest to the business are reducing their personal stake'
        )
        merged = {
            'text': merged_text,
            'sev': 'amber',
            'cat': 'insider',
        }
        out = []
        inserted = False
        for h in hw_dicts:
            if h.get('cat') == 'insider':
                if not inserted:
                    out.append(merged)
                    inserted = True
                continue
            out.append(h)
        return out
    return hw_dicts


# Severity rank used by the downside sort — red flags first, then amber.
_SEV_RANK = {'red': 0, 'amber': 1, 'normal': 2}


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
        (headwinds: list[dict], tailwinds: list[str])
        Headwinds are dicts of shape {text, sev, cat, ...} with sev in
        {'red','amber'}; the JS renderer reads .text and .sev. Tailwinds
        remain plain strings.
    """
    # Layer 1: Macro
    hw_macro, tw_macro = _macro_signals(macro_regime_result)

    # Layer 2: Sector
    hw_sector, tw_sector = _sector_signals(row, sector_data, macro_regime_result,
                                           commodity_data)

    # Layer 3: Stock-specific (returns dicts for hw)
    hw_stock, tw_stock = _stock_signals(row, sector_medians)

    # Layer 4: Peer comparison (returns dicts for hw)
    hw_peer, tw_peer = _peer_signals(row)

    # Layer 5: Balance sheet risk flags (returns dicts for hw)
    hw_risk, tw_risk = _risk_flag_signals(row)

    # Layer 6: News / sentiment / layoff signals
    hw_news, tw_news = _news_signals(row)

    # Coerce legacy string layers (sector/macro) into dict form so we can
    # sort and render uniformly. They're all amber by default.
    hw_stock = [_to_hw_dict(h) for h in hw_stock]
    hw_peer = [_to_hw_dict(h, default_cat='peer') for h in hw_peer]
    hw_risk = [_to_hw_dict(h, default_cat='risk') for h in hw_risk]
    hw_news = [_to_hw_dict(h, default_cat='news') for h in hw_news]
    hw_sector = [_to_hw_dict(h, default_cat='sector') for h in hw_sector]
    hw_macro = [_to_hw_dict(h, default_cat='macro') for h in hw_macro]

    # Combine: stock signals first (most relevant), then peers, risk, news, sector, macro
    all_hw = hw_stock + hw_peer + hw_risk + hw_news + hw_sector + hw_macro
    all_tw = tw_stock + tw_peer + tw_risk + tw_news + tw_sector + tw_macro

    # (A) Dedupe valuation cluster
    all_hw = _dedupe_valuation_cluster(all_hw)
    # (B) Collapse insider selling duplicates
    all_hw = _dedupe_insider_cluster(all_hw)

    # (F) Sector thesis-breaker — always last in the list so it reads as
    # the forward-looking fat-tail risk after the concrete issues.
    thesis = _thesis_breaker_signal(row)

    # (C) Severity sort: red flags first, then amber, then normal.
    # Stable within each tier so category ordering (stock → peer → …) is preserved.
    all_hw.sort(key=lambda h: _SEV_RANK.get(h.get('sev'), 2))

    # Cap, then append thesis-breaker so it always appears even if we'd otherwise truncate
    capped_hw = all_hw[:max_per_side]
    if thesis is not None:
        # Avoid double-adding if it slipped in somehow
        if not any(h.get('cat') == 'thesis_risk' for h in capped_hw):
            # Ensure thesis-breaker is visible even at the cap — evict the last
            # amber item if we're already at capacity
            if len(capped_hw) >= max_per_side:
                capped_hw = capped_hw[:max_per_side - 1]
            capped_hw.append(thesis)

    return capped_hw, all_tw[:max_per_side]


# ---------------------------------------------------------------------------
# Sector-level Profit Pool Narrative
# ---------------------------------------------------------------------------

def _fmt_dollars_compact(v):
    """Format a dollar amount as $X.XB / $X.XM / $X,XXX."""
    if v is None:
        return 'N/A'
    try:
        a = abs(v)
    except Exception:
        return 'N/A'
    sign = '-' if v < 0 else ''
    if a >= 1e12:
        return f'{sign}${a/1e12:.1f}T'
    if a >= 1e9:
        return f'{sign}${a/1e9:.1f}B'
    if a >= 1e6:
        return f'{sign}${a/1e6:.1f}M'
    return f'{sign}${a:,.0f}'


def generate_sector_profit_pool_narrative(sector, rows_in_sector):
    """Build a rich profit pool narrative for one sector.

    Analyzes the sector's profit pool structure, identifies key players,
    and produces Buffett/Porter-style insights about where the profits
    actually live and why.

    Args:
        sector: str — sector name (e.g., "Technology")
        rows_in_sector: list[dict] — full result rows for all companies in the
            sector that have profit pool data (pp_revenue_share, etc.)

    Returns:
        dict with keys:
            overview:      str — 2–3 sentence framing of the pool size & shape
            concentration: str — HHI/CR4 interpretation sentence
            key_players:   list[dict] — up to 5 notable companies, each with
                           {ticker, company_name, role, role_label, note,
                            rating, profit_share, op_margin, pp_multiple}
            insights:      list[str] — 3–5 strategic bullets about the pool

        or None if the sector has no usable profit pool data.
    """
    # Filter to rows that have revenue & operating income so we can reason
    # about the pool numerically.
    cos = [r for r in (rows_in_sector or [])
           if r.get('pp_revenue_share') is not None
           and r.get('operating_income') is not None
           and r.get('revenue') is not None]
    if not cos:
        return None

    total_rev = sum(r.get('revenue') or 0 for r in cos)
    total_oi = sum(r.get('operating_income') or 0 for r in cos)
    n = len(cos)
    if total_rev <= 0:
        return None

    # Sector-wide weighted margin (profit-pool-weighted)
    wtd_margin = (total_oi / total_rev) if total_rev else 0.0

    # Concentration metrics come pre-computed on rows; any row should have them.
    hhi = next((r.get('pp_sector_hhi') for r in cos
                if r.get('pp_sector_hhi') is not None), None)
    cr4 = next((r.get('pp_sector_cr4') for r in cos
                if r.get('pp_sector_cr4') is not None), None)

    # Sort by profit share desc to find leaders
    by_profit = sorted(
        cos,
        key=lambda r: r.get('pp_profit_share') or 0,
        reverse=True,
    )
    # Sort by operating margin desc
    by_margin = sorted(
        cos,
        key=lambda r: r.get('operating_margin') or -999,
        reverse=True,
    )
    # Sort by PP multiple desc (profit share / revenue share)
    by_multiple = sorted(
        [r for r in cos if r.get('pp_multiple') is not None],
        key=lambda r: r.get('pp_multiple') or 0,
        reverse=True,
    )

    # ----- OVERVIEW -----
    top1 = by_profit[0] if by_profit else None
    top3_profit_share = sum((r.get('pp_profit_share') or 0) for r in by_profit[:3])
    top3_rev_share = sum((r.get('pp_revenue_share') or 0) for r in by_profit[:3])

    overview_parts = []
    overview_parts.append(
        f'The {sector} profit pool in this universe totals roughly '
        f'{_fmt_dollars_compact(total_oi)} of operating income on '
        f'{_fmt_dollars_compact(total_rev)} of revenue across {n} companies, '
        f'a blended operating margin of {wtd_margin*100:.1f}%.'
    )
    if top1 is not None and top1.get('pp_profit_share') is not None:
        overview_parts.append(
            f"{top1['ticker']} alone captures {top1['pp_profit_share']*100:.0f}% "
            f"of the pool on {(top1.get('pp_revenue_share') or 0)*100:.0f}% "
            f"of the revenue, setting the pace for the group."
        )
    if len(by_profit) >= 3 and top3_profit_share > 0:
        overview_parts.append(
            f'The top three names together account for '
            f'{top3_profit_share*100:.0f}% of profits from '
            f'{top3_rev_share*100:.0f}% of the revenue — '
            + ('a lopsided pool where scale and pricing power compound.'
               if top3_profit_share > top3_rev_share + 0.05
               else 'roughly proportional to their revenue footprint.')
        )
    overview = ' '.join(overview_parts)

    # ----- CONCENTRATION -----
    concentration = ''
    if hhi is not None:
        if hhi > 0.25:
            label = 'concentrated'
            gloss = (
                'a small number of players set price and earn most of the profits; '
                'new entrants face a steep climb.'
            )
        elif hhi > 0.15:
            label = 'moderately concentrated'
            gloss = (
                'a recognizable pecking order, but no single firm can dictate terms — '
                'share shifts are possible at the margin.'
            )
        else:
            label = 'competitive'
            gloss = (
                'fragmentation keeps pricing honest and makes the pool hard to defend; '
                'margin discipline matters more than scale.'
            )
        concentration = (
            f'HHI of {hhi:.3f} flags the sector as {label}'
            + (f' with the top 4 capturing {cr4*100:.0f}% of revenue — {gloss}'
               if cr4 is not None else f' — {gloss}')
        )
    elif cr4 is not None:
        concentration = (
            f'The top 4 companies capture {cr4*100:.0f}% of revenue, '
            f'indicating a {"concentrated" if cr4 > 0.7 else "dispersed"} pool.'
        )

    # ----- KEY PLAYERS -----
    key_players = []
    seen = set()

    def _add_player(r, role, role_label, note):
        if r is None or r['ticker'] in seen:
            return
        seen.add(r['ticker'])
        key_players.append({
            'ticker': r['ticker'],
            'company_name': r.get('company_name', ''),
            'role': role,
            'role_label': role_label,
            'note': note,
            'rating': r.get('rating'),
            'profit_share': r.get('pp_profit_share'),
            'revenue_share': r.get('pp_revenue_share'),
            'op_margin': r.get('operating_margin'),
            'pp_multiple': r.get('pp_multiple'),
        })

    # Profit leader
    if by_profit:
        r = by_profit[0]
        ps = (r.get('pp_profit_share') or 0) * 100
        _add_player(
            r, 'leader', 'Profit Leader',
            f"Captures {ps:.0f}% of the sector's operating income — "
            f"the center of gravity for the profit pool."
        )
    # Runner-up
    if len(by_profit) >= 2:
        r = by_profit[1]
        ps = (r.get('pp_profit_share') or 0) * 100
        _add_player(
            r, 'runnerup', 'Runner-Up',
            f"Holds {ps:.0f}% of sector profits, the clear number-two position."
        )
    # Margin leader (if not already in the list)
    if by_margin:
        r = by_margin[0]
        om = (r.get('operating_margin') or 0) * 100
        _add_player(
            r, 'margin_leader', 'Margin Leader',
            f"Runs the sector's fattest operating margin at {om:.1f}%, "
            f"signaling pricing power or cost discipline the rest can't match."
        )
    # Value capture efficiency leader (pp_multiple)
    if by_multiple:
        r = by_multiple[0]
        m = r.get('pp_multiple') or 0
        _add_player(
            r, 'efficiency', 'Efficiency Leader',
            f"Turns each dollar of revenue into {m:.2f}x its share of sector "
            f"profits — the most capital-efficient slice of the pool."
        )
    # Margin laggard (highlight the tail risk)
    if by_margin and len(by_margin) >= 3:
        r = by_margin[-1]
        om = (r.get('operating_margin') or 0) * 100
        # Only flag if it's meaningfully below the weighted average
        if (r.get('operating_margin') or 0) < wtd_margin - 0.05:
            _add_player(
                r, 'laggard', 'Margin Laggard',
                f"Operates at {om:.1f}% — well below the sector's "
                f"{wtd_margin*100:.1f}% blended margin, hinting at a structural "
                f"disadvantage or a turnaround still in progress."
            )

    key_players = key_players[:5]

    # ----- INSIGHTS -----
    insights = []

    # 1) Pool skew
    if top3_profit_share > 0 and top3_rev_share > 0:
        skew = top3_profit_share - top3_rev_share
        if skew > 0.10:
            insights.append(
                f"The top 3 over-earn their revenue share by {skew*100:.0f} "
                f"percentage points, a textbook sign of scale economies or "
                f"brand-driven pricing power — the kind of moat Buffett "
                f"describes as \"a toll bridge.\""
            )
        elif skew < -0.05:
            insights.append(
                f"The top 3 under-earn relative to their revenue share — "
                f"scale has not translated into proportional profits, "
                f"which usually means commoditized output or weak pricing discipline."
            )

    # 2) Margin dispersion
    if len(by_margin) >= 3:
        best_m = by_margin[0].get('operating_margin') or 0
        worst_m = by_margin[-1].get('operating_margin') or 0
        spread = best_m - worst_m
        if spread > 0.15:
            insights.append(
                f"Margin spread between the best ({by_margin[0]['ticker']} at "
                f"{best_m*100:.1f}%) and worst ({by_margin[-1]['ticker']} at "
                f"{worst_m*100:.1f}%) operators exceeds {spread*100:.0f} points — "
                f"the sector rewards operational excellence, and laggards "
                f"have real room to close the gap (or be taken out)."
            )
        elif spread < 0.05:
            insights.append(
                f"Margins are tightly clustered ({spread*100:.1f}pp spread between "
                f"top and bottom operators), suggesting the sector has "
                f"standardized economics — differentiation comes from growth "
                f"and capital allocation, not cost structure."
            )

    # 3) Concentration-linked insight
    if hhi is not None:
        if hhi > 0.25:
            insights.append(
                "With HHI above 0.25, this is an oligopoly in practice: "
                "rational competitors avoid price wars because everyone loses. "
                "Watch for coordinated capacity discipline and share-buyback "
                "intensity as signs the incumbents are playing the long game."
            )
        elif hhi < 0.10:
            insights.append(
                "The fragmented structure (HHI < 0.10) makes this a sector "
                "where consolidation is usually the fastest path to value "
                "creation — an acquirer with the balance sheet can collapse "
                "the cost curve and re-rate the pool."
            )

    # 4) Blended margin quality
    if wtd_margin >= 0.20:
        insights.append(
            f"A blended {wtd_margin*100:.1f}% operating margin puts the sector "
            f"in rarefied air — pricing power is the default, not the exception, "
            f"and the only question is whether customer value grows fast enough "
            f"to justify current multiples."
        )
    elif wtd_margin < 0.08 and wtd_margin > 0:
        insights.append(
            f"Blended margins of just {wtd_margin*100:.1f}% mean the pool is "
            f"thin: a single cycle turn or input-cost shock can wipe out a "
            f"year's earnings, so balance-sheet strength matters more than "
            f"top-line growth."
        )
    elif wtd_margin <= 0:
        insights.append(
            f"The sector's blended margin is negative — this is a pool that "
            f"is collectively destroying capital, and the rational move is "
            f"to own only the producer that can outlast the others."
        )

    # 5) Efficiency spread (pp_multiple)
    if len(by_multiple) >= 2:
        best_mult = by_multiple[0].get('pp_multiple') or 0
        worst_mult = by_multiple[-1].get('pp_multiple') or 0
        if best_mult > 1.5 and worst_mult < 0.7:
            insights.append(
                f"{by_multiple[0]['ticker']} converts revenue into profit "
                f"{best_mult/max(worst_mult, 0.01):.1f}x more efficiently than "
                f"{by_multiple[-1]['ticker']} — the pool is being drained "
                f"toward the efficient operator, and weak hands eventually exit."
            )

    # Cap insights at 5
    insights = insights[:5]

    _edu = _SECTOR_EDUCATION.get(sector)
    if isinstance(_edu, dict):
        education_model = _edu.get('model')
        education_cycle = _edu.get('cycle')
    elif isinstance(_edu, str):
        education_model = _edu
        education_cycle = None
    else:
        education_model = None
        education_cycle = None
    return {
        'education': education_model,
        'education_cycle': education_cycle,
        'overview': overview,
        'concentration': concentration,
        'key_players': key_players,
        'insights': insights,
        'stats': {
            'total_revenue': total_rev,
            'total_op_income': total_oi,
            'weighted_margin': wtd_margin,
            'company_count': n,
            'hhi': hhi,
            'cr4': cr4,
        },
    }
