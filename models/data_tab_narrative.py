"""Plain-English summaries for the stock popup's Data sub-tabs.

Each Data sub-tab (Sector, Market, Valuation, Profitability, Health, Growth,
Ownership, People) is a grid of raw numbers. This module turns the same row
into 0-6 sentences per tab saying what those numbers add up to, so the popup
can lead each tab with a verdict instead of leaving the reader to decode it.

Rule-based and pure (no I/O). The thresholds are the ones the report already
colours by or the scoring gates already test, so a summary never contradicts
the cue next to the number it describes; each constant below names its source.
Gate statements read the row's ``_gp_*`` pass flags rather than re-testing the
value, so a summary always agrees with the scorecard.

Every claim is skipped when its input is missing or non-finite — a summary
says less rather than invent. The report calls this at render time (see
``scripts/report_html.build_html``), so a rescore or re-render picks up a
change here without a live run.
"""

import math

from models.narrative import _fmt_dollars_compact

TAB_KEYS = ('sect', 'mkt', 'val', 'prof', 'hlth', 'growth', 'own', 'people')
# Builders emit sentences most-important first; anything past the cap is cut,
# so the tail of each builder holds the context that is nice to have.
MAX_SENTENCES = 6

# --- Thresholds (source in brackets) ---------------------------------------
MOS_WIDE = 0.15            # popup MoS cue: green above 15%, red below 0
MOS_PASS_CAP = -0.20       # scoring._rating_cap_for_row: MoS <= -20% caps at PASS
FV_DISPERSION_MAX = 0.15   # fv_dispersion gate: model MAD <= 15%
SPREAD_MOAT = 0.07         # spread gate: ROIC - WACC > 7%
ROIC_CV_MAX = 0.30         # roic_consistency gate: CV < 30%
ACCRUALS_MAX = 0.08        # accruals gate: |accruals| < 8%
PIOTROSKI_STRONG = 7       # piotroski gate: F-Score >= 7
PIOTROSKI_WEAK = 3         # narrative.generate_financial_summary weak band
ND_EBITDA_MAX = 1.5        # net_debt_ebitda gate: ND/EBITDA <= 1.5x
ND_EBITDA_HEAVY = 3.0      # narrative._stock_signals leverage red line
INT_COV_MIN = 3.0          # int_coverage gate: IC > 3x
FUND_GROWTH_MIN = 0.03     # fund_growth gate: FG > 3%
SHRHLDR_YIELD_MIN = 0.02   # shrhldr_yield gate: yield > 2%
INSIDER_OWN_MIN = 0.05     # insider_own gate: insider >= 5%
MARGIN_ADV = 0.05          # popup margin-advantage cue / margin_advantage gate: 5pp
PP_MULT_STRONG = 1.5       # popup profit-pool multiple cue: 1.5 / 1.0 / 0.5
PP_MULT_PAR = 1.0
PP_MULT_WEAK = 0.5
HHI_CONCENTRATED = 0.25    # popup HHI cue: > 0.25 concentrated, > 0.15 moderate
HHI_MODERATE = 0.15
BETA_UNSTABLE = 0.2        # popup rolling-beta cue: |1y - 5y| >= 0.2 volatile
TRAP_RED = 70              # popup trap cue: >= 70 red, >= 50 amber
TRAP_AMBER = 50
SBC_DILUTION_MAX = 0.02    # sbc_dilution gate: SBC/Rev <= 2%
MULT_VS_HIST_CHEAP = -0.10  # mult_vs_hist gate: >= 10% below own 10y median
MARGIN_VS_HIST_MAX = 0.05  # margin_vs_hist gate: OpM < hist avg + 5pp
EBIT_EV_MIN = 0.08         # ebit_ev gate: EBIT/EV > 8%
P_TBV_MAX = 2.5            # p_tbv gate: P/TBV <= 2.5x
BENEISH_THRESHOLD = -1.78  # models/quality.py manipulation flag: M > -1.78
BENEISH_NEAR = -2.22       # Beneish's original 5-variable cutoff: the grey zone below the flag
MIN_ADV_FOR_BUY = 1_000_000  # mirrors scripts/config.MIN_ADV_FOR_BUY (HOLD cap)
GOODWILL_HIGH = 0.35       # narrative._risk_flag_signals amber goodwill line
# No existing home — chosen for this module:
MULT_CHEAP = 0.8           # multiple <= 80% of the sector median reads cheap
MULT_RICH = 1.25           # multiple >= 125% of the sector median reads rich
MOMENTUM_STRONG = 0.20     # |12-1 momentum| above 20% is a strong trend
MOMENTUM_MILD = 0.05       # below 5% either way reads flat
RANGE_HIGH = 80            # 52-week range position (0-100 scale)
RANGE_LOW = 20
BETA_HIGH = 1.2
BETA_LOW = 0.8
CAGR_SHIFT = 0.02          # 3y vs 5y revenue CAGR gap that reads as a trend
MARGIN_TREND_MOVE = 0.01   # narrative._stock_signals amber margin-trend line
SURPRISE_MOVE = 0.02
SHORT_HIGH = 0.10
COMP_RISK_HIGH = 8         # yfinance compensation risk, 1-10 (lower = better)
COMP_RISK_LOW = 3
ADV_DEEP = 100_000_000     # $100M+ a day reads as deep liquidity
CAPEX_GROWTH = 1.5         # capex / D&A above 1.5x: building capacity
CAPEX_HARVEST = 0.8        # below 0.8x: running the asset base down
BOOK_TO_BILL_UP = 1.05
BOOK_TO_BILL_DOWN = 0.95
DUPONT_LEVERAGED = 3.0     # equity multiplier above 3x: ROE leans on leverage
DEBT_WALL_NEAR = 2.0       # years
STREET_GAP = 0.30          # popup Street-vs-model cue: |diff| > 30%

_FINANCIALS = ('Financial Services',)
_FV_SOURCE_LABEL = {'dcf': 'DCF', 'blend': 'blended-model', 'consensus': 'multi-model'}


def _num(row, key, lo=None, hi=None):
    """Return row[key] as a finite float, or None (bools and text excluded).

    ``lo``/``hi`` drop values outside a plausible band. The grid still shows
    them, but a summary sentence would repeat a data glitch (a micro-cap's
    1,100% revenue CAGR) as if it were a finding.
    """
    v = row.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    # No company-scale figure comes near 1e15; beyond it, scaling for display
    # (x100 for a percentage) can overflow to inf.
    if not math.isfinite(v) or abs(v) > 1e15:
        return None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return v


def _money(v):
    """Company-scale dollars: $2.8M, $569K, $1.2B."""
    if abs(v) < 1e6 and abs(v) >= 1e3:
        return f'{"-" if v < 0 else ""}${abs(v) / 1e3:.0f}K'
    return _fmt_dollars_compact(v)


def _price(v):
    """Per-share dollars: $204, $31.40, $0.42."""
    return f'${v:,.0f}' if abs(v) >= 100 else f'${v:,.2f}'


def _pct(v, digits=0, signed=False):
    s = f'{v * 100:+.{digits}f}%' if signed else f'{abs(v) * 100:.{digits}f}%'
    return s


def _pp(v):
    return f'{abs(v) * 100:.0f}pp'


def _x(v, digits=1):
    return f'{v:.{digits}f}x'


def _gate(row, short):
    """The row's pass flag for a gate: True / False / None (not evaluated)."""
    v = row.get('_gp_' + short)
    return v if isinstance(v, bool) else None


def _join(parts):
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return ''.join(parts)
    return ', '.join(parts[:-1]) + ' and ' + parts[-1]


def _is_financial(row):
    return row.get('sector') in _FINANCIALS


# --- Sector ----------------------------------------------------------------

def _summ_sector(row, stats):
    out = []
    sector = row.get('sector') or 'its sector'
    opm, med, adv = (_num(row, 'operating_margin'), _num(row, '_sector_median_opm'),
                     _num(row, 'pp_margin_advantage'))
    if opm is not None and adv is not None:
        med_txt = f' ({_pct(med)})' if med is not None else ''
        if adv > MARGIN_ADV:
            out.append(f'Its {_pct(opm)} operating margin runs {_pp(adv)} above the {sector} '
                       f'median{med_txt}, a real cost or pricing edge over peers.')
        elif adv < -MARGIN_ADV:
            out.append(f'Its {_pct(opm)} operating margin trails the {sector} median{med_txt} '
                       f'by {_pp(adv)}, so it is a below-average operator in its field.')
        else:
            out.append(f'Its {_pct(opm)} operating margin is roughly in line with the {sector} '
                       f'median{med_txt}.')
    ppm = _num(row, 'pp_multiple')
    if ppm is not None:
        if ppm >= PP_MULT_STRONG:
            out.append(f'It captures {ppm:.2f}x its revenue share in sector profits, taking a '
                       f'disproportionate slice of the profit pool.')
        elif ppm >= PP_MULT_PAR:
            out.append(f'Its share of sector profits slightly exceeds its share of revenue '
                       f'({ppm:.2f}x).')
        elif ppm >= PP_MULT_WEAK:
            out.append(f'It earns less of the sector\'s profit than its revenue share would '
                       f'suggest ({ppm:.2f}x).')
        else:
            out.append(f'It captures well under its share of sector profits ({ppm:.2f}x its '
                       f'revenue share).')
    rs, ps = _num(row, 'pp_revenue_share', lo=0, hi=1), _num(row, 'pp_profit_share', lo=-1, hi=1)
    if rs is not None and rs >= 0.001:
        s = f'It accounts for {_pct(rs, 1)} of sector revenue'
        if ps is not None:
            s += (f' and {_pct(ps, 1)} of sector operating profit' if ps >= 0
                  else ', but loses money while its peers profit')
        out.append(s + '.')
    pool = _num(row, '_gate_pool_share', lo=-1, hi=1)
    if pool is not None and abs(pool) >= 0.01:
        out.append(f'Over five years it has been {"gaining" if pool > 0 else "losing"} share of '
                   f'the sector profit pool ({_pct(pool, 1, signed=True)} a year).')
    hhi = _num(row, 'pp_sector_hhi')
    if hhi is not None:
        n = _num(row, 'pp_sector_count')
        cr4 = _num(row, 'pp_sector_cr4', lo=0, hi=1)
        n_txt = f' across {int(n)} companies in the universe' if n else ''
        if cr4 is not None:
            n_txt += f'; the top four hold {_pct(cr4)} of revenue'
        if hhi > HHI_CONCENTRATED:
            out.append(f'The sector is concentrated (HHI {hhi:.2f}{n_txt}), which tends to '
                       f'support pricing power for the leaders.')
        elif hhi > HHI_MODERATE:
            out.append(f'Sector concentration is moderate (HHI {hhi:.2f}{n_txt}).')
        else:
            out.append(f'The sector is fragmented and competitive (HHI {hhi:.2f}{n_txt}), so '
                       f'pricing power is hard to come by.')
    return out


# --- Market ----------------------------------------------------------------

def _rolling_beta(rb, key):
    v = rb.get(key) if isinstance(rb, dict) else None
    if isinstance(v, dict):
        v = v.get('beta')
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


def _summ_market(row, stats):
    out = []
    if row.get('price_data_stale') is True:
        out.append('Price data for this stock is stale, so the market figures below may lag.')
    mom = _num(row, 'momentum_12_1', lo=-1, hi=20)
    m3 = _num(row, 'momentum_3m', lo=-1, hi=20)
    if mom is not None:
        if mom > MOMENTUM_STRONG:
            out.append(f'The shares are in a strong uptrend, up {_pct(mom)} over the past year '
                       f'excluding the latest month.')
        elif mom > MOMENTUM_MILD:
            out.append(f'The shares have drifted higher, up {_pct(mom)} over the past year '
                       f'excluding the latest month.')
        elif mom < -MOMENTUM_STRONG:
            out.append(f'The shares are in a steep downtrend, down {_pct(mom)} over the past '
                       f'year excluding the latest month.')
        elif mom < -MOMENTUM_MILD:
            out.append(f'The shares have drifted lower, down {_pct(mom)} over the past year '
                       f'excluding the latest month.')
        else:
            out.append('The shares are roughly flat over the past year excluding the latest '
                       'month.')
        # A three-month move against the year's trend is worth calling out.
        if m3 is not None and abs(m3) >= MOMENTUM_MILD and abs(mom) > MOMENTUM_MILD \
                and (m3 > 0) != (mom > 0):
            verb = 'rebounded' if m3 > 0 else 'pulled back'
            out[-1] = out[-1][:-1] + f', though it has {verb} {_pct(m3)} over the last three months.'
    pos, off_high = _num(row, 'range_52w_position'), _num(row, 'pct_from_52w_high')
    if pos is not None:
        hi_txt = f', {_pct(off_high)} below the high' if off_high is not None and off_high < 0 else ''
        if pos >= RANGE_HIGH:
            out.append(f'The price sits near the top of its 52-week range{hi_txt}.')
        elif pos <= RANGE_LOW:
            out.append(f'The price sits near the bottom of its 52-week range{hi_txt}.')
        else:
            out.append(f'The price sits mid-way through its 52-week range{hi_txt}.')
    vol, beta = _num(row, 'realized_vol'), _num(row, 'beta_adjusted')
    if beta is not None:
        vol_txt = f'annualised volatility of {_pct(vol)} and ' if vol is not None else ''
        if beta > BETA_HIGH:
            rel = 'amplifies market moves'
        elif beta < BETA_LOW:
            rel = 'is more defensive than the market'
        else:
            rel = 'moves roughly with the market'
        s = f'With {vol_txt}a beta of {beta:.2f}, it {rel}'
        rb = row.get('rolling_betas')
        b1, b5 = _rolling_beta(rb, '1y'), _rolling_beta(rb, '5y')
        stab = _rolling_beta(rb, 'stability')
        if stab is None and b1 is not None and b5 is not None:
            stab = abs(b1 - b5)
        if stab is not None and stab >= BETA_UNSTABLE and b1 is not None and b5 is not None:
            s += (f', though its beta has shifted (1y {b1:.2f} vs 5y {b5:.2f}), so read it '
                  f'loosely')
        out.append(s + '.')
    adv = _num(row, 'avg_dollar_volume_3m')
    vt = _num(row, 'volume_trend', lo=0, hi=50)
    hot = (f', and interest is running hot at {vt:.1f}x its yearly average volume'
           if vt is not None and vt >= 1.5 else '')
    if adv is not None and adv < MIN_ADV_FOR_BUY:
        out.append(f'Trading is thin (about {_money(adv)} a day), below the '
                   f'liquidity floor for a BUY rating; building a position would move the price.')
    elif adv is not None and adv >= ADV_DEEP:
        out.append(f'Liquidity is deep, with about {_money(adv)} traded a day{hot}.')
    elif adv is not None:
        out.append(f'About {_money(adv)} trades a day, enough to build a position{hot}.')
    elif hot:
        out.append(f'Trading interest is running hot, at {vt:.1f}x its yearly average volume.')
    dds = [(_num(row, k, lo=-1, hi=0), lbl) for k, lbl in
           (('drawdown_2020', 'the 2020 crash'), ('drawdown_2022', 'the 2022 bear market'),
            ('drawdown_2008', '2008'))]
    dds = [(v, lbl) for v, lbl in dds if v is not None]
    if dds:
        out.append('In past sell-offs it fell '
                   + _join([f'{_pct(v)} in {lbl}' for v, lbl in dds]) + '.')
    return out


# --- Valuation -------------------------------------------------------------

def _summ_valuation(row, stats):
    out = []
    mos, fv = _num(row, 'mos'), _num(row, '_fv_effective')
    src = _FV_SOURCE_LABEL.get(row.get('_fv_source'), 'model')
    if mos is not None:
        fv_txt = f' of {_price(fv)}' if fv is not None and fv > 0 else ''
        if mos > MOS_WIDE:
            out.append(f'The stock trades {_pct(mos)} below its {src} fair value{fv_txt}, a '
                       f'meaningful margin of safety.')
        elif mos > 0:
            out.append(f'The stock trades only {_pct(mos)} below its {src} fair value{fv_txt}, '
                       f'a thin margin of safety.')
        else:
            cap = ' That is deep enough to cap the rating at PASS.' if mos <= MOS_PASS_CAP else ''
            out.append(f'The price sits above its {src} fair value{fv_txt} (margin of safety '
                       f'{_pct(mos, signed=True)}), so there is no valuation cushion.{cap}')
    disp = _num(row, '_gate_fv_dispersion')
    if disp is not None:
        ok = _gate(row, 'fv_dispersion')
        if ok is None:
            ok = disp <= FV_DISPERSION_MAX
        p10, p90 = _num(row, 'mc_p10_fv'), _num(row, 'mc_p90_fv')
        mc_txt = ''
        if p10 is not None and p90 is not None and 0 < p10 < p90:
            mc_txt = f', and the Monte Carlo range runs {_price(p10)} to {_price(p90)} (P10-P90)'
        if not ok:
            out.append(f'The fair-value models disagree (dispersion {_pct(disp)}){mc_txt}, so '
                       f'treat the estimate as noisy.')
        else:
            out.append(f'The fair-value models corroborate one another (dispersion '
                       f'{_pct(disp)}){mc_txt}.')
    sector = row.get('sector') or 'sector'
    if stats:
        cheap, rich = [], []
        for key, label in (('pe', 'P/E'), ('ev_ebitda', 'EV/EBITDA'), ('pfcf', 'P/FCF')):
            v, med = _num(row, key), stats.get(key)
            if v is None or med is None or v <= 0 or med <= 0:
                continue
            txt = f'{label} {v:.1f} vs {med:.1f}'
            if v <= med * MULT_CHEAP:
                cheap.append(txt)
            elif v >= med * MULT_RICH:
                rich.append(txt)
        if cheap and not rich:
            out.append(f'On multiples it looks cheap against {sector} peers: {_join(cheap)} '
                       f'(sector median).')
        elif rich and not cheap:
            out.append(f'On multiples it looks expensive against {sector} peers: {_join(rich)} '
                       f'(sector median).')
        elif rich and cheap:
            out.append(f'Multiples send mixed signals against {sector} peers: cheaper on '
                       f'{_join(cheap)}, richer on {_join(rich)} (sector median).')
    mvh = _num(row, '_gate_mult_vs_hist', lo=-1, hi=20)
    if mvh is not None:
        ok = _gate(row, 'mult_vs_hist')
        if ok is None:
            ok = mvh < MULT_VS_HIST_CHEAP
        if ok:
            out.append(f'Against its own history it is cheap: the EBIT multiple sits '
                       f'{_pct(mvh)} below its 10-year median.')
        elif mvh > 0:
            out.append(f'Against its own history it is dear: the EBIT multiple sits '
                       f'{_pct(mvh)} above its 10-year median.')
        else:
            out.append(f'The EBIT multiple is close to its own 10-year median '
                       f'({_pct(mvh, signed=True)}).')
    ylds = []
    fy = _num(row, '_gate_fcf_yield', lo=-1, hi=1)
    if fy is not None:
        ok = _gate(row, 'fcf_yield')
        ylds.append((f'an FCF yield of {_pct(fy, 1, signed=fy < 0)}'
                     + (' (above the risk-free rate)' if ok else ' (below the risk-free rate)'
                        if ok is False else ''), ok))
    ee = _num(row, '_gate_ebit_ev', lo=-1, hi=1)
    if ee is not None:
        ok = _gate(row, 'ebit_ev')
        if ok is None:
            ok = ee > EBIT_EV_MIN
        ylds.append((f'an EBIT/EV of {_pct(ee, 1, signed=ee < 0)} (vs the 8% bar)', ok))
    if ylds:
        oks = [ok for _, ok in ylds if ok is not None]
        lead = ('Earnings yields are attractive' if oks and all(oks)
                else 'Earnings yields are thin' if oks and not any(oks)
                else 'Earnings yields are mixed')
        out.append(f'{lead}: {_join([t for t, _ in ylds])}.')
    tgt, price, na = _num(row, 'target_mean'), _num(row, 'price'), _num(row, 'num_analysts')
    if tgt is not None and price is not None and tgt > 0 and price > 0:
        up = tgt / price - 1
        who = f'The {int(na)} covering analysts' if na and na > 1 else 'The Street'
        s = (f'{who} target {_price(tgt)} on average, '
             + (f'{_pct(up)} above the price' if up >= 0.005
                else f'{_pct(up)} below the price' if up <= -0.005 else 'in line with the price'))
        if fv is not None and fv > 0 and abs(tgt / fv - 1) > STREET_GAP:
            s += ', well away from the model\'s fair value'
        out.append(s + '.')
    ig, ive = _num(row, 'implied_growth', lo=-1, hi=2), _num(row, 'implied_vs_estimated', lo=-2, hi=2)
    if ig is not None:
        s = f'The price implies {_pct(ig, 1)} annual cash-flow growth'
        if ive is not None and abs(ive) >= 0.01:
            s += f', {_pp(ive)} {"more" if ive > 0 else "less"} than the model expects'
        out.append(s + '.')
    return out


# --- Profitability ---------------------------------------------------------

def _summ_profitability(row, stats):
    out = []
    roic, wacc, spread = _num(row, 'roic'), _num(row, 'wacc'), _num(row, 'spread')
    if roic is not None and spread is not None:
        wacc_txt = f' of {_pct(wacc, 1)}' if wacc is not None else ''
        if spread > SPREAD_MOAT:
            s = (f'ROIC of {_pct(roic)} clears the cost of capital{wacc_txt} by {_pp(spread)}, '
                 f'the mark of a value creator')
        elif spread > 0:
            s = (f'ROIC of {_pct(roic)} earns above the cost of capital{wacc_txt}, but the '
                 f'{_pp(spread)} spread is short of the 7pp moat bar')
        else:
            s = (f'ROIC of {_pct(roic)} falls short of the cost of capital{wacc_txt}, so '
                 f'growth at these returns destroys value')
        cv = _num(row, 'roic_cv')
        if cv is not None:
            s += ('; returns have been steady' if cv < ROIC_CV_MAX
                  else '; returns have swung a lot from year to year')
            s += f' (CV {cv:.2f})'
        out.append(s + '.')
    gm, fm, fm_x = _num(row, 'gross_margin'), _num(row, 'fcf_margin'), _num(row, 'fcf_margin_ex_sbc')
    if fm is not None:
        gm_txt = f'It runs a {_pct(gm)} gross margin and ' if gm is not None else 'It '
        if fm < 0:
            s = f'{gm_txt}burns cash (FCF margin {_pct(fm, signed=True)})'
        else:
            s = f'{gm_txt}converts {_pct(fm)} of revenue into free cash flow'
        if fm_x is not None and fm - fm_x >= 0.03:
            s += f', or {_pct(fm_x, signed=fm_x < 0)} after stock compensation'
        out.append(s + '.')
    if not _is_financial(row):
        good, bad = [], []
        pio = _num(row, 'piotroski')
        if pio is not None:
            if pio >= PIOTROSKI_STRONG:
                good.append(f'a Piotroski score of {int(pio)}/9')
            elif pio <= PIOTROSKI_WEAK:
                bad.append(f'a weak Piotroski score of {int(pio)}/9')
        cc = _num(row, 'cash_conv', lo=-5, hi=5)
        if cc is not None:
            if cc >= 1.0:
                good.append(f'cash conversion of {cc:.2f}x')
            elif cc < 0.8:
                bad.append(f'cash conversion of only {cc:.2f}x')
        acc = _num(row, 'accruals', lo=-1, hi=1)
        if acc is not None and abs(acc) >= ACCRUALS_MAX:
            bad.append(f'high accruals ({_pct(acc, 1, signed=True)} of assets)')
        if good and not bad:
            out.append(f'Earnings quality looks solid: {_join(good)}.')
        elif bad and not good:
            out.append(f'Earnings quality is a concern: {_join(bad)}.')
        elif good and bad:
            out.append(f'Earnings quality is mixed: {_join(good)}, but {_join(bad)}.')
    else:
        nim, eff = _num(row, 'nim'), _num(row, 'efficiency_ratio')
        parts = []
        if nim is not None:
            parts.append(f'a net interest margin of {_pct(nim, 2)}')
        if eff is not None:
            parts.append(f'an efficiency ratio of {_pct(eff)}'
                         + (' (lean)' if eff < 0.55 else ' (cost-heavy)' if eff > 0.70 else ''))
        if parts:
            out.append(f'As a lender it runs {_join(parts)}.')
    cr = _num(row, 'combined_ratio')
    if cr is not None:
        out.append(f'Its combined ratio of {_pct(cr)} means it '
                   + ('makes an underwriting profit.' if cr < 1 else 'loses money on underwriting.'))
    affo = _num(row, 'affo_margin')
    if affo is not None:
        out.append(f'AFFO margin (a proxy for distributable REIT cash) is {_pct(affo, signed=affo < 0)}.')
    roe, lev = _num(row, 'roe', lo=-5, hi=5), _num(row, 'dupont_leverage', lo=0, hi=100)
    if roe is not None and roe > 0 and _is_financial(row):
        # Balance-sheet leverage is the business model for a lender, not a
        # flattering distortion, so the DuPont caveat below doesn't apply.
        out.append(f'It earns a {_pct(roe)} return on equity.')
    elif roe is not None and roe > 0 and lev is not None:
        # The leverage caveat only matters when leverage is lifting ROE above
        # the return the business itself earns.
        if lev > DUPONT_LEVERAGED and (roic is None or roe > roic):
            out.append(f'ROE of {_pct(roe)} is amplified by {_x(lev)} balance-sheet leverage '
                       f'(DuPont), so ROIC is the cleaner read on the business.')
        else:
            out.append(f'ROE of {_pct(roe)} comes mostly from margins and asset turnover rather '
                       f'than leverage ({_x(lev)} equity multiplier).')
    mvh = _num(row, '_gate_margin_vs_hist', lo=-1, hi=1)
    if mvh is not None and _gate(row, 'margin_vs_hist') is False:
        out.append(f'Operating margin is running {_pp(mvh)} above its own historical average, '
                   f'a peak-margin risk if conditions normalise.')
    costs = []
    rd = _num(row, 'rd_intensity_xbrl', lo=0, hi=5)
    if rd is not None and rd >= 0.005:
        costs.append(f'{_pct(rd, 1)} of revenue on R&D')
    sbc = _num(row, 'sbc_pct_rev_xbrl', lo=0, hi=5)
    if sbc is not None and sbc >= 0.001:
        ok = _gate(row, 'sbc_dilution')
        if ok is None:
            ok = sbc <= SBC_DILUTION_MAX
        costs.append(f'{_pct(sbc, 1)}{"" if costs else " of revenue"} on stock compensation'
                     + ('' if ok else ' (above the 2% gate)'))
    sga = _num(row, 'sga_yoy_change', lo=-1, hi=5)
    if costs:
        s = f'It spends {_join(costs)}'
        if sga is not None and sga > 0.12:
            s += f', and SG&A jumped {_pct(sga)} last year'
        out.append(s + '.')
    elif sga is not None and sga > 0.12:
        out.append(f'SG&A jumped {_pct(sga)} last year, worth watching.')
    return out


# --- Health ----------------------------------------------------------------

def _summ_health(row, stats):
    # Lead with solvency, then anything that caps the rating or flags a trap,
    # then the secondary detail — the cap cuts from the end.
    lead, warn, detail = [], [], []
    fin = _is_financial(row)
    if fin:
        cet1, npl = _num(row, 'cet1_ratio'), _num(row, 'npl_ratio')
        if cet1 is not None:
            lead.append(f'CET1 capital of {_pct(cet1, 1)} '
                        + ('sits comfortably above the ~7% regulatory minimum with buffer.'
                           if cet1 >= 0.10 else 'leaves little room over the ~7% regulatory minimum.'))
        if npl is not None:
            lead.append(f'Non-performing loans are {_pct(npl, 2)} of the book'
                        + (', a clean credit profile.' if npl < 0.01
                           else ', elevated enough to watch.' if npl > 0.03 else '.'))
    else:
        nce, nd = _num(row, 'net_cash_to_mcap'), _num(row, 'net_debt')
        lev = _num(row, 'nd_ebitda')
        ic = _num(row, 'int_cov')
        if nd is not None and nd < 0:
            mc_txt = f' ({_pct(nce)} of market cap)' if nce is not None and nce > 0 else ''
            lead.append(f'The balance sheet carries net cash of {_money(-nd)}{mc_txt}.')
        elif lev is not None and lev < 0:
            lead.append('The company holds more cash than debt.')
        elif lev is not None:
            ok = _gate(row, 'net_debt_ebitda')
            if ok is None:
                ok = lev <= ND_EBITDA_MAX
            cash, debt = _num(row, 'cash', lo=0), _num(row, 'total_debt', lo=0)
            bal = (f' ({_money(cash)} of cash against {_money(debt)} of debt)'
                   if cash is not None and debt is not None and debt > 0 else '')
            if ok:
                s = f'Leverage is conservative at {_x(lev)} net debt to EBITDA{bal}'
            elif lev <= ND_EBITDA_HEAVY:
                s = (f'Leverage is moderate at {_x(lev)} net debt to EBITDA{bal}, above the '
                     f'1.5x gate')
            else:
                s = f'Leverage is heavy at {_x(lev)} net debt to EBITDA{bal}'
            ic_ok = _gate(row, 'int_coverage')
            if ic_ok is None and ic is not None:
                ic_ok = ic > INT_COV_MIN
            if ic is not None:
                s += (f', with interest covered {_x(ic)}' if ic_ok
                      else f', and interest cover is thin at {_x(ic)}')
            lead.append(s + '.')
        elif ic is not None:
            lead.append(f'Interest is covered {_x(ic)}'
                        + ('.' if ic > INT_COV_MIN else ', thinner than the 3x gate.'))
        cr = _num(row, 'cr')
        if cr is not None and cr < 1.0:
            detail.append(f'The current ratio of {cr:.2f} means short-term liabilities exceed '
                          f'current assets.')
        elif cr is not None and cr >= 2.0:
            detail.append(f'Short-term liquidity is ample (current ratio {cr:.2f}).')
        wall = _num(row, 'debt_maturity_wall_yrs', lo=0, hi=100)
        if wall is not None and wall < DEBT_WALL_NEAR:
            detail.append(f'Its debt comes due soon (maturity wall about {wall:.1f} years out), '
                          f'so refinancing terms matter.')
        wcd = _num(row, 'working_capital_days', lo=-365, hi=365)
        if wcd is not None and wcd <= -15:
            detail.append(f'It runs on negative working capital ({wcd:.0f} days), so customers '
                          f'and suppliers effectively fund its operations.')
        elif wcd is not None and wcd >= 120:
            detail.append(f'Working capital ties up about {wcd:.0f} days of sales, a drag on '
                          f'cash generation.')
    flags = []
    z, zone = _num(row, 'altman_z'), row.get('altman_z_zone')
    if not fin and zone == 'distress' and z is not None:
        flags.append(f'an Altman Z of {z:.2f} in the distress zone')
    if row.get('beneish_flag') is True:
        flags.append('a Beneish M-score that flags possible earnings manipulation')
    if flags:
        cap = 'either one caps' if len(flags) > 1 else 'which caps'
        warn.append(f'Warning signs: {_join(flags)}, {cap} the rating at HOLD.')
    elif not fin and zone == 'safe' and z is not None and row.get('beneish_flag') is False:
        detail.insert(0, f'No distress or manipulation flags (Altman Z {z:.1f}, in the safe '
                         f'zone).')
    trap = _num(row, 'trap_score')
    if trap is not None and trap >= TRAP_AMBER:
        lvl = 'high' if trap >= TRAP_RED else 'elevated'
        warn.append(f'The value-trap profile is {lvl} ({round(trap)}/100); see the breakdown '
                    f'below.')
    bm = _num(row, 'beneish_m', lo=-50, hi=50)
    if bm is not None and row.get('beneish_flag') is not True \
            and BENEISH_NEAR < bm <= BENEISH_THRESHOLD:
        detail.append(f'The Beneish M-score ({bm:.2f}) sits just under the manipulation line, '
                      f'so earnings quality bears watching.')
    gw = _num(row, 'goodwill_pct')
    if gw is not None and gw > GOODWILL_HIGH:
        detail.append(f'Goodwill makes up {_pct(gw)} of assets, a soft asset exposed to '
                      f'impairment.')
    ef = _num(row, 'edgar_fields_flagged', lo=0)
    if ef:
        warn.append(f'{int(ef)} reported figure{"s" if ef > 1 else ""} '
                    f'differ{"" if ef > 1 else "s"} by more than 5% between yfinance and SEC '
                    f'filings; see the data-quality note.')
    return lead + warn + detail


# --- Growth ----------------------------------------------------------------

def _summ_growth(row, stats):
    out = []
    c3, c5, c10 = (_num(row, k, lo=-0.9, hi=1.0)
                   for k in ('rev_cagr', 'rev_cagr_5y', 'rev_cagr_10y'))
    spans = [(c, lbl) for c, lbl in ((c3, '3'), (c5, '5'), (c10, '10')) if c is not None]
    if spans:
        body = _join([f'{_pct(c, 1, signed=c < 0)} over {lbl} years' for c, lbl in spans])
        s = f'Revenue has compounded at {body}'
        if c3 is not None and c5 is not None:
            shrinking = c3 < 0 and c5 < 0
            if c3 - c5 > CAGR_SHIFT:
                s += ', so the decline is easing' if shrinking else ', so growth is accelerating'
            elif c5 - c3 > CAGR_SHIFT:
                s += ', so the decline is deepening' if shrinking else ', so growth is decelerating'
        vol = _num(row, '_gate_rev_volatility', lo=0, hi=5)
        if vol is not None and _gate(row, 'rev_volatility') is False:
            s += f', and it has been lumpy (year-to-year swings of {_pct(vol)})'
        out.append(s + '.')
    fg = _num(row, 'fundamental_growth', lo=-1, hi=1)
    rr = _num(row, 'reinvestment_rate', lo=-5, hi=5)
    if fg is not None and rr is not None and abs(rr) < 0.005 and abs(fg) < 0.005:
        # models/ratios.py clamps the rate at 0 when capex + working-capital
        # build does not exceed depreciation — no net reinvestment, not no data.
        out.append('Net reinvestment is nil (capex and working capital no more than cover '
                   'depreciation), so fundamental growth is near zero.')
    elif fg is not None:
        rr_txt = f'reinvesting {_pct(rr)} of profits' if rr is not None else 'its reinvestment'
        verdict = ('enough to sustain growth' if fg > FUND_GROWTH_MIN
                   else 'too little to drive meaningful growth')
        out.append(f'At current returns, {rr_txt} supports about {_pct(fg, 1, signed=fg < 0)} '
                   f'a year of fundamental growth, {verdict}.')
    fcf5 = _num(row, '_gate_fcf_durability', lo=-0.9, hi=1.0)
    if fcf5 is not None:
        out.append(f'Free cash flow has compounded {_pct(fcf5, 1, signed=fcf5 < 0)} a year over '
                   f'five years' + (', short of the 5% durability bar.'
                                    if _gate(row, 'fcf_durability') is False else '.'))
    mt = _num(row, 'margin_trend', lo=-0.5, hi=0.5)
    if mt is not None and abs(mt) >= MARGIN_TREND_MOVE:
        out.append(f'Margins are {"expanding" if mt > 0 else "contracting"} '
                   f'({_pct(mt, 1, signed=True)} trend).')
    lead_ind = []
    btb = _num(row, 'book_to_bill_proxy', lo=0, hi=10)
    if btb is not None:
        lead_ind.append(f'a book-to-bill of {btb:.2f}'
                        + (' (orders outpacing sales)' if btb >= BOOK_TO_BILL_UP
                           else ' (orders lagging sales)' if btb <= BOOK_TO_BILL_DOWN else ''))
    bl = _num(row, 'backlog_to_revenue', lo=0, hi=50)
    if bl is not None:
        lead_ind.append(f'a backlog worth {_pct(bl)} of annual revenue' if bl < 1
                        else f'a backlog worth {bl:.1f}x annual revenue')
    drg = _num(row, 'deferred_rev_growth', lo=-0.9, hi=5)
    if drg is not None and abs(drg) >= 0.02:
        lead_ind.append(f'deferred revenue {"up" if drg > 0 else "down"} {_pct(drg)}')
    ffo = _num(row, 'ffo_growth_5y', lo=-0.9, hi=1)
    if ffo is not None:
        lead_ind.append(f'FFO growth of {_pct(ffo, 1, signed=ffo < 0)} a year over five years')
    fda = _num(row, 'fda_pipeline_count', lo=0)
    if fda:
        lead_ind.append(f'{int(fda)} active clinical trial{"s" if fda > 1 else ""}')
    if lead_ind:
        out.append(f'Forward indicators: {_join(lead_ind)}.')
    cdd, ci = _num(row, 'capex_to_dd_ratio', lo=0, hi=50), _num(row, 'capex_intensity', lo=0, hi=5)
    if cdd is not None:
        ci_txt = f' ({_pct(ci, 1)} of sales)' if ci is not None else ''
        if cdd >= CAPEX_GROWTH:
            out.append(f'Capex runs {_x(cdd)} depreciation{ci_txt}, so it is building capacity '
                       f'for future growth.')
        elif cdd <= CAPEX_HARVEST:
            out.append(f'Capex runs only {_x(cdd)} depreciation{ci_txt}, so the asset base is '
                       f'being harvested rather than grown.')
        else:
            ci_txt = f', {_pct(ci, 1)} of sales' if ci is not None else ''
            out.append(f'Capex roughly matches depreciation ({_x(cdd)}{ci_txt}), a maintenance '
                       f'pace.')
    r40 = _num(row, 'rule_of_40', lo=-500, hi=500)
    if r40 is not None:
        out.append(f'Its Rule of 40 score is {r40:.0f}, '
                   + ('clearing the bar for a healthy growth business.' if r40 >= 40
                      else 'below the 40 that marks a healthy growth business.'))
    sa = _num(row, 'surprise_avg', lo=-2, hi=2)
    if sa is not None and abs(sa) >= SURPRISE_MOVE:
        out.append(f'It has {"beaten" if sa > 0 else "missed"} earnings estimates by '
                   f'{_pct(sa, 1)} on average.')
    ltg = _num(row, 'analyst_ltg', lo=-0.5, hi=1)
    if ltg is not None:
        out.append(f'Analysts expect {_pct(ltg, 1, signed=ltg < 0)} long-term annual earnings '
                   f'growth.')
    return out


# --- Ownership -------------------------------------------------------------

def _summ_ownership(row, stats):
    out = []
    ins, inst = _num(row, 'insider_pct'), _num(row, 'inst_pct')
    if ins is not None:
        if ins >= INSIDER_OWN_MIN:
            s = f'Insiders own {_pct(ins, 1)}, meaningful skin in the game'
        elif ins < 0.01:
            s = f'Insiders hold little stock ({_pct(ins, 1)})'
        else:
            s = f'Insiders own a modest {_pct(ins, 1)}'
        if inst is not None and inst > 1.0:
            s += ', and reported institutional holdings exceed 100% of shares (a sign of heavy share lending)'
        elif inst is not None:
            s += f', and institutions hold {_pct(inst)}'
        out.append(s + '.')
    buys, sells = _num(row, 'insider_buy_count_365d'), _num(row, 'insider_sell_count_365d')
    nv = _num(row, 'insider_net_value')
    nv_txt = (f', net {_money(abs(nv))} {"bought" if nv > 0 else "sold"}'
              if nv is not None and abs(nv) >= 1e5 else '')
    if buys is not None and sells is not None:
        if buys > 0 and sells == 0:
            out.append(f'Insiders have only bought over the past year ({int(buys)} '
                       f'purchase{"s" if buys > 1 else ""}{nv_txt}).')
        elif sells >= 5 and buys == 0:
            out.append(f'Insiders have only sold over the past year ({int(sells)} sales{nv_txt}).')
        elif buys > 0 and sells > 0:
            out.append(f'Insider trading is two-way: {int(buys)} purchase{"s" if buys > 1 else ""} '
                       f'and {int(sells)} sale{"s" if sells > 1 else ""} over the past '
                       f'year{nv_txt}.')
    sy = _num(row, 'shareholder_yield')
    if sy is not None:
        bb, dy = _num(row, 'share_buyback_rate'), _num(row, 'div_yield')
        parts = []
        if bb is not None and bb > 0:
            parts.append(f'{_pct(bb, 1)} in buybacks')
        if dy is not None and dy > 0:
            parts.append(f'{_pct(dy, 1)} in dividends')
        if bb is not None and bb <= -0.001:
            parts[-1:] = [(parts[-1] + ', ' if parts else '')
                          + f'less {_pct(bb, 1)} of net share issuance']
        detail = f' ({_join(parts)})' if parts else ''
        if sy > SHRHLDR_YIELD_MIN:
            out.append(f'It returns {_pct(sy, 1)} a year to shareholders{detail}.')
        elif sy >= 0.001:
            out.append(f'Shareholder yield is a modest {_pct(sy, 1)}{detail}.')
        elif sy <= -0.001:
            out.append(f'Net share issuance leaves shareholder yield negative '
                       f'({_pct(sy, 1, signed=True)}), diluting holders.')
    shr = _num(row, '_gate_share_shrink', lo=-0.5, hi=2)
    if shr is not None and abs(shr) >= 0.005:
        out.append(f'The share count has {"shrunk" if shr < 0 else "grown"} {_pct(shr, 1)} a '
                   f'year over five years'
                   + (', steadily raising each holder\'s stake.' if shr < 0
                      else ', diluting existing holders.'))
    pr, streak = _num(row, 'payout_ratio', lo=0, hi=100), _num(row, 'ddm_consecutive_years', lo=0)
    dg = _num(row, 'dividend_cagr_5y', lo=-0.9, hi=2)
    if pr is not None and pr > 1.0:
        out.append(f'It pays out {_pct(pr)} of earnings, more than it earns, so the dividend '
                   f'is at risk.')
    elif streak is not None and streak >= 10:
        s = f'It has paid a dividend for {int(streak)} consecutive years'
        if dg is not None and abs(dg) >= 0.005:
            s += (f', raising it {_pct(dg, 1)} a year over the last five' if dg > 0
                  else f', though it has been cut {_pct(dg, 1)} a year over the last five')
        out.append(s + '.')
    si = _num(row, 'short_pct_float', lo=0, hi=5)
    if si is not None and si > SHORT_HIGH:
        sr = _num(row, 'short_ratio')
        sr_txt = f', {sr:.1f} days to cover' if sr is not None else ''
        out.append(f'Short interest is elevated at {_pct(si, 1)} of the float{sr_txt}.')
    return out


# --- People ----------------------------------------------------------------

def _summ_people(row, stats):
    out = []
    rpe = _num(row, 'revenue_per_emp')
    emp = _num(row, 'employees', lo=1)
    if rpe is not None and rpe > 0:
        s = (f'With about {int(emp):,} employees, each generates about {_money(rpe)} of revenue'
             if emp else f'Each employee generates about {_money(rpe)} of revenue')
        med = (stats or {}).get('revenue_per_emp')
        if med is not None and med > 0:
            rel = rpe / med
            sec = row.get('sector') or 'sector'
            s += (f', in line with the {sec} median' if 0.85 <= rel <= 1.15
                  else f', {rel:.1f}x the {sec} median')
        g = _num(row, 'rpe_cagr', lo=-0.9, hi=2)
        if g is not None and abs(g) >= 0.02:
            s += (f', and productivity has {"risen" if g > 0 else "fallen"} '
                  f'{_pct(g, 1)} a year')
        out.append(s + '.')
        fpe = _num(row, 'fcf_per_emp')
        if fpe is not None and abs(fpe) >= 1e3:
            out.append(f'That works out to {_money(fpe)} of free cash flow per employee.'
                       if fpe > 0 else
                       f'Free cash flow is negative, at {_money(fpe)} per employee.')
    elif emp:
        out.append(f'It employs about {int(emp):,} people.')
    if row.get('founder_led') is True:
        out.append('The company is founder-led, which often keeps management aligned with '
                   'long-term owners.')
    pay, ratio = _num(row, 'ceo_total_pay'), _num(row, 'ceo_pay_ratio')
    risk = _num(row, 'compensation_risk', lo=1, hi=10)
    risk_txt = ''
    if risk is not None and risk >= COMP_RISK_HIGH:
        risk_txt = f'compensation governance scores poorly ({int(risk)}/10, lower is better)'
    elif risk is not None and risk <= COMP_RISK_LOW:
        risk_txt = f'compensation governance scores well ({int(risk)}/10, lower is better)'
    if pay is not None and pay > 0:
        s = f'CEO pay is {_money(pay)}'
        if ratio is not None:
            s += f' ({ratio:.0f}x revenue per employee)'
        if risk_txt:
            s += f', and {risk_txt}'
        out.append(s + '.')
    elif risk_txt:
        out.append(risk_txt[0].upper() + risk_txt[1:] + '.')
    flags = []
    if row.get('employment_legal_flag') is True:
        flags.append('employment-related legal filings')
    if row.get('layoff_news_signal') is True:
        flags.append('recent layoff news')
    if flags:
        out.append(f'Culture flags: {_join(flags)}.')
    gd = _num(row, 'glassdoor_rating', lo=0, hi=5)
    if gd is not None:
        extra = [f'{_pct(v)} {lbl}' for v, lbl in
                 ((_num(row, 'glassdoor_rec_pct', lo=0, hi=1), 'would recommend it'),
                  (_num(row, 'glassdoor_ceo_pct', lo=0, hi=1), 'approve of the CEO'))
                 if v is not None]
        out.append(f'Employees rate it {gd:.1f}/5 on Glassdoor'
                   + (f'; {_join(extra)}' if extra else '') + '.')
    spe = _num(row, 'sbc_per_emp', lo=0)
    if spe is not None and spe >= 1e3:
        out.append(f'Stock compensation averages {_money(spe)} per employee.')
    if row.get('culture_award_signal') is True:
        out.append('It has recently been recognised with a workplace culture award.')
    return out


_BUILDERS = {
    'sect': _summ_sector,
    'mkt': _summ_market,
    'val': _summ_valuation,
    'prof': _summ_profitability,
    'hlth': _summ_health,
    'growth': _summ_growth,
    'own': _summ_ownership,
    'people': _summ_people,
}


def generate_data_tab_summaries(row, sector_stats=None):
    """Return {tab_key: [sentence, ...]} for every Data sub-tab.

    ``sector_stats`` maps a field to the sector median for the row's sector
    (see ``report_html._sector_stats``); peer comparisons are skipped when it
    is absent. Every key in TAB_KEYS is always present; a tab with nothing
    worth saying gets an empty list.
    """
    stats = sector_stats or {}
    return {k: _BUILDERS[k](row, stats)[:MAX_SENTENCES] for k in TAB_KEYS}
