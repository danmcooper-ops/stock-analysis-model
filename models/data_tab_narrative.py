"""Plain-English summaries for the stock popup's Data sub-tabs.

Each Data sub-tab (Sector, Market, Valuation, Profitability, Health, Growth,
Ownership, People) is a grid of raw numbers. This module turns the same row
into 0-4 sentences per tab saying what those numbers add up to, so the popup
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
MAX_SENTENCES = 4

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
        return f'${v / 1e3:.0f}K'
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
    hhi = _num(row, 'pp_sector_hhi')
    if hhi is not None:
        n = _num(row, 'pp_sector_count')
        n_txt = f' across {int(n)} companies in the universe' if n else ''
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
    mom = _num(row, 'momentum_12_1')
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
    if adv is not None and adv < MIN_ADV_FOR_BUY:
        out.append(f'Trading is thin (about {_money(adv)} a day), below the '
                   f'liquidity floor for a BUY rating; building a position would move the price.')
    else:
        vt = _num(row, 'volume_trend')
        if vt is not None and vt >= 1.5:
            out.append(f'Trading interest is running hot, at {vt:.1f}x its yearly average '
                       f'volume.')
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
        if not ok:
            out.append(f'The fair-value models disagree (dispersion {_pct(disp)}), so treat the '
                       f'estimate as noisy.')
        else:
            out.append(f'The fair-value models corroborate one another (dispersion '
                       f'{_pct(disp)}).')
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
    ig, ive = _num(row, 'implied_growth'), _num(row, 'implied_vs_estimated')
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
    return out


# --- Health ----------------------------------------------------------------

def _summ_health(row, stats):
    out = []
    fin = _is_financial(row)
    if fin:
        cet1, npl = _num(row, 'cet1_ratio'), _num(row, 'npl_ratio')
        if cet1 is not None:
            out.append(f'CET1 capital of {_pct(cet1, 1)} '
                       + ('sits comfortably above the ~7% regulatory minimum with buffer.'
                          if cet1 >= 0.10 else 'leaves little room over the ~7% regulatory minimum.'))
        if npl is not None:
            out.append(f'Non-performing loans are {_pct(npl, 2)} of the book'
                       + (', a clean credit profile.' if npl < 0.01
                          else ', elevated enough to watch.' if npl > 0.03 else '.'))
    else:
        nce, nd = _num(row, 'net_cash_to_mcap'), _num(row, 'net_debt')
        lev = _num(row, 'nd_ebitda')
        ic = _num(row, 'int_cov')
        if nd is not None and nd < 0:
            mc_txt = f' ({_pct(nce)} of market cap)' if nce is not None and nce > 0 else ''
            out.append(f'The balance sheet carries net cash of {_money(-nd)}{mc_txt}.')
        elif lev is not None and lev < 0:
            out.append('The company holds more cash than debt.')
        elif lev is not None:
            ok = _gate(row, 'net_debt_ebitda')
            if ok is None:
                ok = lev <= ND_EBITDA_MAX
            if ok:
                s = f'Leverage is conservative at {_x(lev)} net debt to EBITDA'
            elif lev <= ND_EBITDA_HEAVY:
                s = f'Leverage is moderate at {_x(lev)} net debt to EBITDA, above the 1.5x gate'
            else:
                s = f'Leverage is heavy at {_x(lev)} net debt to EBITDA'
            ic_ok = _gate(row, 'int_coverage')
            if ic_ok is None and ic is not None:
                ic_ok = ic > INT_COV_MIN
            if ic is not None:
                s += (f', with interest covered {_x(ic)}' if ic_ok
                      else f', and interest cover is thin at {_x(ic)}')
            out.append(s + '.')
        elif ic is not None:
            out.append(f'Interest is covered {_x(ic)}'
                       + ('.' if ic > INT_COV_MIN else ', thinner than the 3x gate.'))
        cr = _num(row, 'cr')
        if cr is not None and cr < 1.0:
            out.append(f'The current ratio of {cr:.2f} means short-term liabilities exceed '
                       f'current assets.')
        elif cr is not None and cr >= 2.0:
            out.append(f'Short-term liquidity is ample (current ratio {cr:.2f}).')
    flags = []
    z, zone = _num(row, 'altman_z'), row.get('altman_z_zone')
    if not fin and zone == 'distress' and z is not None:
        flags.append(f'an Altman Z of {z:.2f} in the distress zone')
    if row.get('beneish_flag') is True:
        flags.append('a Beneish M-score that flags possible earnings manipulation')
    if flags:
        cap = 'either one caps' if len(flags) > 1 else 'which caps'
        out.append(f'Warning signs: {_join(flags)}, {cap} the rating at HOLD.')
    elif not fin and zone == 'safe' and z is not None and row.get('beneish_flag') is False:
        out.append(f'No distress or manipulation flags (Altman Z {z:.1f}, in the safe zone).')
    gw = _num(row, 'goodwill_pct')
    if gw is not None and gw > GOODWILL_HIGH:
        out.append(f'Goodwill makes up {_pct(gw)} of assets, a soft asset exposed to '
                   f'impairment.')
    trap = _num(row, 'trap_score')
    if trap is not None and trap >= TRAP_AMBER:
        lvl = 'high' if trap >= TRAP_RED else 'elevated'
        out.append(f'The value-trap profile is {lvl} ({round(trap)}/100); see the breakdown '
                   f'below.')
    return out


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
        out.append(s + '.')
    fg = _num(row, 'fundamental_growth', lo=-1, hi=1)
    rr = _num(row, 'reinvestment_rate', lo=-5, hi=5)
    if fg is not None:
        rr_txt = f'reinvesting {_pct(rr)} of profits' if rr is not None else 'its reinvestment'
        verdict = ('enough to sustain growth' if fg > FUND_GROWTH_MIN
                   else 'too little to drive meaningful growth')
        out.append(f'At current returns, {rr_txt} supports about {_pct(fg, 1, signed=fg < 0)} '
                   f'a year of fundamental growth, {verdict}.')
    mt = _num(row, 'margin_trend', lo=-0.5, hi=0.5)
    if mt is not None and abs(mt) >= MARGIN_TREND_MOVE:
        out.append(f'Margins are {"expanding" if mt > 0 else "contracting"} '
                   f'({_pct(mt, 1, signed=True)} trend).')
    r40 = _num(row, 'rule_of_40')
    if r40 is not None:
        out.append(f'Its Rule of 40 score is {r40:.0f}, '
                   + ('clearing the bar for a healthy growth business.' if r40 >= 40
                      else 'below the 40 that marks a healthy growth business.'))
    sa = _num(row, 'surprise_avg', lo=-2, hi=2)
    if sa is not None and abs(sa) >= SURPRISE_MOVE and len(out) < MAX_SENTENCES:
        out.append(f'It has {"beaten" if sa > 0 else "missed"} earnings estimates by '
                   f'{_pct(sa, 1)} on average.')
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
    if buys is not None and sells is not None:
        if buys > 0 and sells == 0:
            out.append(f'Insiders have only bought over the past year ({int(buys)} purchases).')
        elif sells >= 5 and buys == 0:
            out.append(f'Insiders have only sold over the past year ({int(sells)} sales).')
    sy = _num(row, 'shareholder_yield')
    if sy is not None:
        bb, dy = _num(row, 'share_buyback_rate'), _num(row, 'div_yield')
        parts = []
        if bb is not None and bb > 0:
            parts.append(f'{_pct(bb, 1)} in buybacks')
        if dy is not None and dy > 0:
            parts.append(f'{_pct(dy, 1)} in dividends')
        detail = f' ({_join(parts)})' if parts else ''
        if sy > SHRHLDR_YIELD_MIN:
            out.append(f'It returns {_pct(sy, 1)} a year to shareholders{detail}.')
        elif sy >= 0.001:
            out.append(f'Shareholder yield is a modest {_pct(sy, 1)}{detail}.')
        elif sy <= -0.001:
            out.append(f'Net share issuance leaves shareholder yield negative '
                       f'({_pct(sy, 1, signed=True)}), diluting holders.')
    pr, streak = _num(row, 'payout_ratio'), _num(row, 'ddm_consecutive_years')
    if pr is not None and pr > 1.0:
        out.append(f'It pays out {_pct(pr)} of earnings, more than it earns, so the dividend '
                   f'is at risk.')
    elif streak is not None and streak >= 10:
        out.append(f'It has paid a dividend for {int(streak)} consecutive years.')
    si = _num(row, 'short_pct_float')
    if si is not None and si > SHORT_HIGH and len(out) < MAX_SENTENCES:
        sr = _num(row, 'short_ratio')
        sr_txt = f', {sr:.1f} days to cover' if sr is not None else ''
        out.append(f'Short interest is elevated at {_pct(si, 1)} of the float{sr_txt}.')
    return out


# --- People ----------------------------------------------------------------

def _summ_people(row, stats):
    out = []
    rpe = _num(row, 'revenue_per_emp')
    if rpe is not None and rpe > 0:
        s = f'Each employee generates about {_money(rpe)} of revenue'
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
    pay, ratio = _num(row, 'ceo_total_pay'), _num(row, 'ceo_pay_ratio')
    risk = _num(row, 'compensation_risk')
    if pay is not None and pay > 0:
        s = f'CEO pay is {_money(pay)}'
        if ratio is not None:
            s += f' ({ratio:.0f}x revenue per employee)'
        out.append(s + '.')
    if risk is not None:
        if risk >= COMP_RISK_HIGH:
            out.append(f'Compensation governance scores poorly ({int(risk)}/10, lower is better).')
        elif risk <= COMP_RISK_LOW:
            out.append(f'Compensation governance scores well ({int(risk)}/10, lower is better).')
    flags = []
    if row.get('employment_legal_flag') is True:
        flags.append('employment-related legal filings')
    if row.get('layoff_news_signal') is True:
        flags.append('recent layoff news')
    if flags:
        out.append(f'Culture flags: {_join(flags)}.')
    gd = _num(row, 'glassdoor_rating')
    if gd is not None and len(out) < MAX_SENTENCES:
        out.append(f'Employees rate it {gd:.1f}/5 on Glassdoor.')
    if row.get('culture_award_signal') is True and len(out) < MAX_SENTENCES:
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
