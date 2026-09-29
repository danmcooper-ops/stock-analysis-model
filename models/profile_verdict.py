"""Invest / watch / avoid verdict for the ticker page's one-page Profile tab.

The Profile tab puts every decision-relevant number for one company on a
single page. This module adds the conclusion: a deterministic verdict with
the reasons for and against it, so the page ends in an answer.

It adds no new model. It reads what the pipeline already computed:
- the rating and its caps (``scripts/scoring.py``);
- the margin of safety against the effective fair value;
- the gate inputs, compared at the thresholds ``models/data_tab_narrative``
  already quotes, so the verdict never contradicts a Data-tab summary.

Verdicts:
- ``AVOID``: any hard red flag (see ``_red_flags``).
- ``INVEST``: the model rates it BUY/LEAN BUY, it trades at least ``MIN_MOS``
  below fair value, the valuation is not low-confidence, it earns more than
  its cost of capital, and the weighted checks mostly pass (``MIN_CONVICTION``).
- ``WATCH``: everything else; ``buy_below`` names the price that would make
  it one.
- ``INSUFFICIENT DATA``: no price or no fair value.

A check whose input is missing or non-finite is N/A: it is never a failure,
matching the scoring gates' N/A semantics. Pure, no I/O. The report calls it
at render time (``scripts/report_html._attach_profiles``), so a change here
needs no live run.
"""

import statistics

from models.data_tab_narrative import (
    BENEISH_THRESHOLD,
    INT_COV_MIN,
    MIN_ADV_FOR_BUY,
    MULT_CHEAP,
    MULT_RICH,
    ND_EBITDA_HEAVY,
    ND_EBITDA_MAX,
    PIOTROSKI_STRONG,
    PIOTROSKI_WEAK,
    SPREAD_MOAT,
    TRAP_RED,
    _is_financial,
    _join,
    _money,
    _num,
    _pct,
    _price,
    _x,
)

INVEST, WATCH, AVOID, NO_DATA = 'INVEST', 'WATCH', 'AVOID', 'INSUFFICIENT DATA'
VERDICTS = (INVEST, WATCH, AVOID, NO_DATA)

# --- Thresholds (source in brackets) ---------------------------------------
MIN_MOS = 0.15             # popup MoS cue: green above 15% (the margin to demand)
MOS_AVOID = -0.20          # scoring._rating_cap_for_row: MoS <= -20% caps at PASS
MIN_CONVICTION = 55        # share of weighted checks passing that an INVEST needs
INT_COV_DANGER = 1.5       # below 1.5x operating income barely covers interest
FCF_MARGIN_STRONG = 0.10   # no existing home — chosen for this module
GROWTH_STRONG = 0.07       # revenue CAGR that reads as a growth business
STREET_UPSIDE = 0.15       # analysts' mean target this far above price is support
STREET_DOWNSIDE = -0.05
MOMENTUM_UP = 0.10
MOMENTUM_DOWN = -0.20
MARGIN_TREND_MOVE = 0.01   # data_tab_narrative.MARGIN_TREND_MOVE
SHAREHOLDER_YIELD_HIGH = 0.03
INSIDER_SELL_HEAVY = 0.005  # net insider selling above 0.5% of market cap
CET1_STRONG = 0.10         # bank capital: 10%+ CET1 is comfortably above minimums
NPL_HIGH = 0.03
MAX_REASONS = 5

# Valuation confidence (the Summary PDF's range line). Sheet-only: it never
# feeds the verdict or the rating.
FV_DISPERSION_MAX = 0.15   # scoring fv_dispersion gate: model MAD <= 15%
FV_DISPERSION_WIDE = 0.30  # twice the gate: the models no longer corroborate
DCF_GAP_WIDE = 0.50        # pre-blend DCF this far from the other models' median
DCF_GAP_TIGHT = 0.25
MC_SPREAD_WIDE = 3.0       # Monte Carlo P90 / P10 above 3x: a 3-fold range
CONFIDENCE_LEVELS = ('HIGH', 'MEDIUM', 'LOW')

# Sector-median fields the Profile tab compares against (report_html feeds
# them through _sector_stats); only the multiples feed a check here.
SECTOR_FIELDS = ('pe', 'ev_ebitda', 'pfcf', 'pb', 'roic', 'gross_margin',
                 'fcf_margin', 'div_yield')

_BUYISH = ('BUY', 'LEAN BUY')
# Altman's Z was fit on manufacturers. Regulated, asset-heavy balance sheets
# (utilities, REITs) and lenders sit in its "distress" zone by construction:
# on 2026-09-24, 127 of 513 non-financial distress readings were utilities or
# REITs. There the reading is not reported as a finding at all.
_ALTMAN_SKIP = ('Financial Services', 'Utilities', 'Real Estate')
# Cap reasons already said by another check (MoS has its own).
_CAP_DUPLICATES = ('margin of safety', 'missing price or fair value', 'Altman', 'Beneish',
                   'liquidity')


def _fv(row):
    return _num(row, '_fv_effective', lo=0.0)


def _mc_low(row):
    c = row.get('mc_confidence')
    return isinstance(c, str) and c.upper().startswith('LOW')


def _altman_applies(row):
    return row.get('sector') not in _ALTMAN_SKIP


def _healthy(row):
    """Cash-generative and earning above its cost of capital."""
    spread, fcfm = _num(row, 'spread'), _num(row, 'fcf_margin')
    return spread is not None and spread > 0 and fcfm is not None and fcfm > 0.05


def _altman_txt(row):
    z = _num(row, 'altman_z')
    return 'Altman Z in the distress zone' + (f' ({z:.2f})' if z is not None else '')


def _red_flags(row):
    """Reasons that alone rule a stock out, whatever else it scores."""
    flags = []
    fin = _is_financial(row)
    mos, fv = _num(row, 'mos'), _fv(row)
    if mos is not None and mos <= MOS_AVOID:
        flags.append(f'Priced {_pct(-mos)} above its fair value'
                     + (f' of {_price(fv)}' if fv else ''))
    # A distress reading on a business that plainly earns its keep is a
    # weakness to weigh (see _checks), not a veto.
    if (row.get('altman_z_zone') == 'distress' and _altman_applies(row)
            and not _healthy(row)):
        flags.append(_altman_txt(row))
    bm = _num(row, 'beneish_m')
    if row.get('beneish_flag') is True or (bm is not None and bm > BENEISH_THRESHOLD):
        flags.append('Beneish M-score flags possible earnings manipulation'
                     + (f' ({bm:.2f})' if bm is not None else ''))
    spread, fcfm = _num(row, 'spread'), _num(row, 'fcf_margin')
    if spread is not None and spread < 0 and fcfm is not None and fcfm < 0:
        flags.append('Earns below its cost of capital and burns cash')
    ic, nd = _num(row, 'int_cov'), _num(row, 'net_debt')
    if not fin and ic is not None and ic < INT_COV_DANGER and (nd is None or nd > 0):
        flags.append(f'Operating income covers interest only {_x(ic)}')
    # Last: the specific reasons above make a better headline than the label,
    # which usually just restates one of them.
    if row.get('rating') == 'PASS':
        flags.append('The model rates it PASS')
    return flags


def _mult_vs_sector(row, key, label, med):
    v, m = _num(row, key, lo=0.0), _num(med, key, lo=0.0)
    if v is None or m is None or v == 0 or m == 0:
        return None
    rel = v / m
    txt = f'{label} {_x(v)} vs sector {_x(m)}'
    if rel <= MULT_CHEAP:
        return 1, txt
    if rel >= MULT_RICH:
        return -1, txt
    return 0, txt


def _checks(row, med):
    """Return [(key, weight, status, text)]; status 1 pass, -1 fail, 0 neutral.

    Checks with no usable input are omitted (N/A). Neutral ones count half
    toward conviction and appear in neither list.
    """
    out = []
    fin = _is_financial(row)

    def add(key, w, status, text):
        out.append((key, w, status, text))

    rating = row.get('rating')
    if rating:
        comp = _num(row, '_composite_score')
        gp = row.get('_gates_passed')
        det = ', '.join(p for p in (f'composite {comp:.0f}' if comp is not None else '',
                                    f'gates {gp}' if gp else '') if p)
        txt = f'Model rating {rating}' + (f' ({det})' if det else '')
        add('rating', 3, 1 if rating in _BUYISH else -1 if rating == 'PASS' else 0, txt)

    mos, fv = _num(row, 'mos'), _fv(row)
    if mos is not None and fv is not None:
        where = f'{_pct(mos)} below' if mos >= 0 else f'{_pct(-mos)} above'
        txt = f'Trades {where} fair value of {_price(fv)}'
        add('mos', 3, 1 if mos >= MIN_MOS else -1 if mos < 0 else 0, txt)
    if _mc_low(row):
        add('mc', 1, -1, 'Fair value is low-confidence (wide Monte Carlo spread)')

    if fin:
        roe, re_ = _num(row, 'roe'), _num(row, 'er')
        if roe is not None and re_ is not None:
            d = roe - re_
            add('moat', 3, 1 if d >= 0.03 else -1 if d < 0 else 0,
                f'ROE {_pct(roe, 1)} vs cost of equity {_pct(re_, 1)}')
    else:
        spread, roic, wacc = _num(row, 'spread'), _num(row, 'roic'), _num(row, 'wacc')
        if spread is not None:
            base = (f'ROIC {_pct(roic, 1)} vs WACC {_pct(wacc, 1)}'
                    if roic is not None and wacc is not None else f'ROIC−WACC spread {_pct(spread, 1, True)}')
            add('moat', 3, 1 if spread >= SPREAD_MOAT else -1 if spread < 0 else 0, base)

    fcfm = _num(row, 'fcf_margin', lo=-5, hi=5)
    if fcfm is not None and not fin:
        add('fcf', 2, 1 if fcfm >= FCF_MARGIN_STRONG else -1 if fcfm < 0 else 0,
            f'FCF margin {_pct(fcfm, 1)}' + (' (burns cash)' if fcfm < 0 else ''))

    g = _num(row, 'rev_cagr_5y', lo=-1, hi=3)
    lbl = '5y'
    if g is None:
        g, lbl = _num(row, 'rev_cagr', lo=-1, hi=3), '3y'
    if g is not None:
        add('growth', 2, 1 if g >= GROWTH_STRONG else -1 if g < 0 else 0,
            f'Revenue {"grew" if g >= 0 else "shrank"} {_pct(abs(g), 1)}/yr over {lbl}')

    if fin:
        cet1, npl = _num(row, 'cet1_ratio'), _num(row, 'npl_ratio')
        if cet1 is not None:
            add('capital', 2, 1 if cet1 >= CET1_STRONG else -1, f'CET1 capital ratio {_pct(cet1, 1)}')
        if npl is not None and npl > NPL_HIGH:
            add('npl', 2, -1, f'Non-performing loans {_pct(npl, 1)}')
    else:
        nd, nde, ic = _num(row, 'net_debt'), _num(row, 'nd_ebitda'), _num(row, 'int_cov')
        if nd is not None and nd <= 0:
            add('leverage', 2, 1, f'Net cash of {_money(-nd)}')
        elif nde is not None:
            add('leverage', 2, 1 if nde <= ND_EBITDA_MAX else -1 if nde > ND_EBITDA_HEAVY else 0,
                f'Net debt/EBITDA {_x(nde)}')
        if ic is not None and ic < INT_COV_MIN and (nd is None or nd > 0):
            add('int_cov', 2, -1, f'Interest coverage only {_x(ic)}')

    pio = _num(row, 'piotroski')
    if pio is not None:
        add('piotroski', 1, 1 if pio >= PIOTROSKI_STRONG else -1 if pio <= PIOTROSKI_WEAK else 0,
            f'Piotroski F-score {pio:.0f}/9')
    zone = row.get('altman_z_zone')
    if zone == 'safe' and _altman_applies(row):
        add('altman', 1, 1, 'Altman Z in the safe zone')
    elif zone == 'distress' and _altman_applies(row):
        add('altman', 2, -1, _altman_txt(row))
    if row.get('trap_flag') is True:
        ts = _num(row, 'trap_score')
        reasons = row.get('trap_reasons') or []
        why = f': {reasons[0]}' if reasons and isinstance(reasons[0], str) else ''
        add('trap', 2 if ts is not None and ts >= TRAP_RED else 1, -1,
            f'Value-trap warning{f" (score {ts:.0f})" if ts is not None else ""}{why}')

    for key, label in (('pe', 'P/E'), ('ev_ebitda', 'EV/EBITDA'), ('pfcf', 'P/FCF')):
        r = _mult_vs_sector(row, key, label, med)
        if r is not None:
            add('mult_' + key, 1, r[0], r[1])

    tgt, px = _num(row, 'target_mean', lo=0.0), _num(row, 'price', lo=0.0)
    if tgt is not None and px:
        up = tgt / px - 1
        n = _num(row, 'num_analysts')
        who = f'{n:.0f} analysts\'' if n else 'Analysts\''
        add('street', 1, 1 if up >= STREET_UPSIDE else -1 if up < STREET_DOWNSIDE else 0,
            f'{who} mean target {_price(tgt)} ({_pct(up, 0, True)})')

    nv, mcap = _num(row, 'insider_net_value'), _num(row, 'mcap', lo=0.0)
    buys = _num(row, 'insider_buy_count_365d')
    if nv is not None and nv > 0 and buys:
        add('insider', 1, 1, f'Insiders net buyers ({_money(nv)} over 12 months)')
    elif nv is not None and mcap and -nv / mcap >= INSIDER_SELL_HEAVY:
        add('insider', 1, -1, f'Heavy insider selling ({_money(-nv)} net over 12 months)')

    mom = _num(row, 'momentum_12_1', lo=-1, hi=10)
    if mom is not None:
        add('momentum', 1, 1 if mom >= MOMENTUM_UP else -1 if mom <= MOMENTUM_DOWN else 0,
            f'12-1 month momentum {_pct(mom, 0, True)}')
    mt = _num(row, 'margin_trend', lo=-1, hi=1)
    if mt is not None and abs(mt) >= MARGIN_TREND_MOVE:
        add('margins', 1, 1 if mt > 0 else -1,
            f'Operating margin {"expanding" if mt > 0 else "contracting"} {_pct(abs(mt), 1)}/yr')
    sy = _num(row, 'shareholder_yield', lo=-1, hi=1)
    if sy is not None and sy >= SHAREHOLDER_YIELD_HIGH:
        add('yield', 1, 1, f'Returns {_pct(sy, 1)} a year to shareholders')
    adv = _num(row, 'avg_dollar_volume_3m')
    if adv is not None and adv < MIN_ADV_FOR_BUY:
        add('liquidity', 1, -1, f'Thinly traded ({_money(adv)} a day)')

    for reason in row.get('_rating_cap_reasons') or []:
        if isinstance(reason, str) and not any(d in reason for d in _CAP_DUPLICATES):
            add('cap', 1, -1, f'Rating capped: {reason}')
    return out


def _conviction(checks):
    tot = sum(w for _, w, _, _ in checks)
    if not tot:
        return None
    got = sum(w * (1.0 if s > 0 else 0.5 if s == 0 else 0.0) for _, w, s, _ in checks)
    return round(100 * got / tot)


def _ranked(checks, sign):
    """[(key, text)] of the checks with this status, heaviest first."""
    picked = [(w, i, k, t) for i, (k, w, s, t) in enumerate(checks) if s == sign]
    picked.sort(key=lambda x: (-x[0], x[1]))
    return [(k, t) for _, _, k, t in picked]


def _lc(text):
    """Lower-case a reason's first letter to continue a sentence, unless it
    starts an acronym or a proper name (ROIC, Altman, Beneish, Piotroski)."""
    first = text.split(' ', 1)[0]
    if first[1:2].isupper() or first in ('Altman', 'Beneish', 'Piotroski'):
        return text
    return text[0].lower() + text[1:]


def _invest_blockers(row, checks, conv):
    """Why a stock is not an INVEST, in the order a reader would fix them.

    Empty means every INVEST condition holds. Only meaningful when there is
    a price, a fair value and no red flag.
    """
    out = []
    rating, mos = row.get('rating'), _num(row, 'mos')
    if rating not in _BUYISH:
        out.append(f'the model rates it {rating}' if rating else 'it has no model rating')
    if mos is None or mos < MIN_MOS:
        out.append('no margin of safety' if mos is None or mos <= 0
                   else f'only {_pct(mos)} below fair value')
    if _mc_low(row):
        out.append('the fair value is low-confidence (wide Monte Carlo spread)')
    if any(k == 'moat' and s < 0 for k, _, s, _ in checks):
        out.append('it earns less than its cost of capital')
    if conv is None or conv < MIN_CONVICTION:
        out.append(f'too few checks pass (conviction {conv if conv is not None else 0})')
    return out


def _headline(verdict, row, pros, cons, flags, buy_below, blockers):
    """One sentence answering "should I buy it?"."""
    if verdict == NO_DATA:
        return 'Not enough data to value it: missing price or fair value.'
    if verdict == AVOID:
        return f'Avoid: {_lc(flags[0])}.'
    # The rating is already on the page's header; the headline says why.
    reasons = [t for k, t in pros if k != 'rating']
    if verdict == INVEST:
        return f'Invest: {"; ".join(_lc(t) for t in reasons[:2])}.' if reasons else 'Invest.'
    rating = row.get('rating')
    if rating in _BUYISH:
        lead = f'Watch: {rating}-rated, but {_join(blockers[:2])}'
        mos = _num(row, 'mos')
        if buy_below is not None and (mos is None or mos < MIN_MOS):
            return f'{lead}; it becomes a buy below {_price(buy_below)}.'
        return lead + '.'
    if cons:
        return f'Watch: {_lc(cons[0][1])}.'
    return 'Watch: no strong case either way.'


def _valuation_confidence(row):
    """How far to trust the fair value: a bear/base/bull range plus a level.

    The base is the effective fair value; bear and bull are the Monte Carlo
    P10/P90. The pre-blend DCF is compared with the median of the other
    intrinsic models (growth EPV, RIM, DDM; NAV is an asset floor and
    excluded), which is the same model set scoring's fv_dispersion uses.
    - LOW: any warning sign (low Monte Carlo confidence, wide model
      dispersion, a DCF far from the other models, a 3-fold Monte Carlo
      range).
    - HIGH: every available test is tight.
    - MEDIUM otherwise.
    Missing inputs are skipped; with nothing to judge by, the level is None.
    """
    base = _fv(row)
    if not base:
        return None
    bear, bull = _num(row, 'mc_p10_fv', lo=0.0), _num(row, 'mc_p90_fv', lo=0.0)
    if bear is not None and bull is not None and bear > bull:
        bear, bull = bull, bear
    alts = [v for v in (_num(row, 'epv_growth_fv', lo=0.0), _num(row, 'rim_fv', lo=0.0),
                        _num(row, 'ddm_fv', lo=0.0)) if v]
    alt_median = statistics.median(alts) if alts else None
    dcf = _num(row, '_dcf_fv_preblend', lo=0.0) or _num(row, 'dcf_fv', lo=0.0)
    gap = dcf / alt_median - 1 if dcf and alt_median else None
    disp = _num(row, '_gate_fv_dispersion', lo=0.0)
    mc = row.get('mc_confidence')
    mc = mc.split()[0].upper() if isinstance(mc, str) and mc.strip() else None

    low, tight, judged = [], [], 0
    if mc in CONFIDENCE_LEVELS:
        judged += 1
        if mc == 'LOW':
            low.append('Monte Carlo spread is wide')
        tight.append(mc == 'HIGH')
    if disp is not None:
        judged += 1
        if disp > FV_DISPERSION_WIDE:
            low.append(f'models disagree ({_pct(disp)} dispersion)')
        tight.append(disp <= FV_DISPERSION_MAX)
    if gap is not None:
        judged += 1
        if abs(gap) > DCF_GAP_WIDE:
            low.append(f'DCF is {_pct(abs(gap))} {"above" if gap > 0 else "below"} '
                       f'the other models\' median ({_price(alt_median)})')
        tight.append(abs(gap) <= DCF_GAP_TIGHT)
    if bear and bull:
        judged += 1
        if bull / bear > MC_SPREAD_WIDE:
            low.append(f'bull case is {bull / bear:.1f}x the bear case')
    if not judged:
        level = None
    elif low:
        level = 'LOW'
    elif tight and all(tight) and mc == 'HIGH':
        level = 'HIGH'
    else:
        level = 'MEDIUM'
    return {
        'base': round(base, 2), 'base_src': row.get('_fv_source'),
        'bear': round(bear, 2) if bear else None, 'bull': round(bull, 2) if bull else None,
        'alt_median': round(alt_median, 2) if alt_median else None, 'n_alt': len(alts),
        'gap': round(gap, 4) if gap is not None else None,
        'dispersion': round(disp, 4) if disp is not None else None,
        'level': level, 'why': low,
    }


def profile_verdict(row, sector_medians=None):
    """Return the Profile tab's verdict for one report row.

    ``sector_medians`` maps a field to the median for the row's sector
    (``report_html._sector_stats``); multiples comparisons are skipped
    without it. Never raises on missing or malformed fields.
    """
    med = sector_medians or {}
    fv, px = _fv(row), _num(row, 'price', lo=0.0)
    buy_below = round(fv * (1 - MIN_MOS), 2) if fv else None
    checks = _checks(row, med) if px and fv else []
    pros, cons = _ranked(checks, 1), _ranked(checks, -1)
    conv = _conviction(checks)
    flags = _red_flags(row) if px and fv else []
    blockers = _invest_blockers(row, checks, conv) if px and fv else []

    if not px or not fv:
        verdict = NO_DATA
    elif flags:
        verdict = AVOID
    elif not blockers:
        verdict = INVEST
    else:
        verdict = WATCH
    return {
        'v': verdict,
        'conv': conv,
        'head': _headline(verdict, row, pros, cons, flags, buy_below, blockers),
        'pros': [t for _, t in pros[:MAX_REASONS]],
        'cons': [t for _, t in cons[:MAX_REASONS]],
        'flags': flags,
        # What stands between a WATCH and an INVEST.
        'need': blockers if verdict == WATCH else [],
        'buy_below': buy_below,
        # Bear/base/bull range and how far to trust it (sheet-only).
        'vc': _valuation_confidence(row) if px and fv else None,
        'med': {k: round(v, 4) for k in SECTOR_FIELDS
                if (v := _num(med, k)) is not None},
    }
