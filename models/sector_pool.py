"""Sector profit-pool analytics for the Sector Analysis page.

Pure functions over result rows, built at render time by
``report_html._build_sector_pool_data`` and shipped inside ``SECTOR_POOL``
beside the narrative. The pool is the same one the page already draws:
positive operating income over the rows with revenue and operating income
(``pool_rows``). What this module adds is time: the pool's size, margin and
concentration over the last decade from each company's EDGAR annual
history, the split of its growth into volume and margin, where today's
margin sits in the sector's own range, and which companies took share.

``year_series`` and ``panel_pools`` are shared with the Pool Share gate
(``scripts.scoring._compute_pool_share_trajectory``), so the gate and the
sector view count a year and a consistent panel the same way.

Every series is measured on the companies analysed today, so it shows how
today's sector got here, not the sector as it stood in each year (a company
that left the market is absent from every year). Foreign filers' histories
are translated at one FX rate and so show local-currency growth.
"""

import logging
import math

logger = logging.getLogger(__name__)

HISTORY_YEARS = 10
# A year whose reporters hold less than this share of the revenue that
# could have reported it is incomplete: the latest fiscal year usually is,
# because January-March year-end filers have not reported it yet. It is
# drawn but never used as a growth endpoint, so a filing lag cannot read as
# a fall. See _year_points for what "could have reported" means.
COMPLETE_COVERAGE_RATIO = 0.8
GROWTH_WINDOW_YEARS = 5
# Years averaged at each end of the growth window (see _growth_windows).
ENDPOINT_BLOCK = 3
MIN_GROWTH_SPAN = 3
MIN_PANEL = 3
MIN_CYCLE_YEARS = 5
SHIFT_LIST_N = 5
# Relative HHI change over the growth window that counts as a trend.
HHI_TREND_RATIO = 0.10


def year_series(d):
    """``{int year: float}`` from a history dict.

    Keys are ints in a live run and strings after a JSON round trip, and a
    fiscal-year key may carry a suffix; the first four characters are the
    year. None, NaN and infinite values, unparseable keys and a payload
    that is not a dict at all are dropped: one NaN year (the snapshot store
    keeps NaN as is) would otherwise turn every pool sum it joins into NaN,
    which then passes a ``pool <= 0`` guard."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if v is None or isinstance(v, bool):
            continue
        try:
            x = float(v)
            y = int(str(k)[:4])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out[y] = x
    return out


def panel_pools(histories, y0, y1):
    """Positive operating-income pools at *y0* and *y1* over the consistent
    panel: only the histories that hold both years. Returns
    ``(pool0, pool1, n)``. Negative income clamps to 0, as in the
    single-year ``pp_profit_share`` pass."""
    p0 = p1 = 0.0
    n = 0
    for h in histories:
        if y0 in h and y1 in h:
            p0 += max(h[y0], 0.0)
            p1 += max(h[y1], 0.0)
            n += 1
    return p0, p1, n


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if v != v or v in (float('inf'), float('-inf')):
        return None
    return float(v)


def pool_rows(rows):
    """The sector's pool rows: revenue > 0 and operating income present
    (the predicate the page, the narrative and the cross-sector chart share),
    with one row per issuer. Snapshots written before the issuer collapse
    carry duplicate listings with identical statements; they are dropped by
    the same ``sector|revenue|operating income`` key the cross-sector chart
    uses (``_ppIssuerKey``)."""
    out, seen = [], set()
    for r in rows or []:
        rev, oi = _num(r.get('revenue')), _num(r.get('operating_income'))
        if rev is None or oi is None or rev <= 0:
            continue
        key = (r.get('sector'), rev, oi)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _histories(rows):
    """``[(row, oi_by_year, rev_by_year)]`` for rows with any EDGAR
    operating-income history."""
    out = []
    for r in rows:
        eh = r.get('edgar_history') or {}
        oi = year_series(eh.get('operating_income_history'))
        if not oi:
            continue
        out.append((r, oi, year_series(eh.get('revenue_history'))))
    return out


def _split_stale(hist):
    """Split off histories that stopped well before the sector's current
    fiscal year: ``(current, stale)``.

    A company still trading but no longer filing with the SEC (Sinopec and
    PetroChina after their 2022 ADR delistings; Sony and Vale, whose
    companyfacts history stops years back) keeps its old history, and at ~$940B of today's Energy revenue those two
    alone made every year after 2022 read as a third of the sector missing.
    That is a composition break, not a filing lag, so such rows leave the
    time series altogether. The reference year is the median of each
    company's latest reported year, counted by company rather than weighted
    by revenue: the stale filers can be the largest names (those two
    outweigh the rest of a small panel), and a revenue weight lets them drag
    the reference back to their own last year and hide themselves. A history
    ending more than one year before it is stale; one year behind is an
    ordinary filing lag."""
    if not hist:
        return hist, []
    last = sorted(max(oi) for _, oi, _ in hist)
    ref = last[len(last) // 2]
    current = [h for h in hist if max(h[1]) >= ref - 1]
    stale = [h for h in hist if max(h[1]) < ref - 1]
    return current, stale


def _hhi(values):
    tot = sum(values)
    if tot <= 0:
        return None
    return sum((v / tot) ** 2 for v in values)


def _cagr(a, b, span):
    if a is None or b is None or a <= 0 or b <= 0 or span <= 0:
        return None
    return (b / a) ** (1.0 / span) - 1


def pctile_rank(values, v):
    """Where *v* sits among *values*, 0 (lowest) to 1 (highest): the share
    of the other values below it, ties counted half. The one rank both the
    Cycle Position fact and the sector forces' margin evidence use, so the
    two quote the same percentile for the same margin."""
    if not values:
        return None
    below = sum(1 for x in values if x < v)
    equal = sum(1 for x in values if x == v)
    return (below + 0.5 * max(equal - 1, 0)) / max(len(values) - 1, 1)


def _year_points(hist, today_rev):
    """Per-year aggregates over every row that reported both revenue and
    operating income that year.

    ``coverage`` is the reporters' share of today's sector revenue: what the
    chart's footnote quotes. ``complete`` asks a different question: did the
    companies that could have reported the year do so? A company whose
    history begins later (a recent listing or spin-off) could not, so it
    does not count against the years before it. Measured against today's
    revenue instead, one large 2025 entrant made every earlier Packaged
    Foods year read 43% complete, and the industry had no growth window."""
    years = sorted({y for _, oi, rev in hist for y in oi if y in rev})
    if not years:
        return []
    first = [min((y for y in oi if y in rev), default=None) for _, oi, rev in hist]
    years = [y for y in years if y > years[-1] - HISTORY_YEARS]
    points = []
    for y in years:
        rev = pos = net = cov = eligible = 0.0
        n = 0
        oi_vals = []
        for (r, oi_h, rev_h), y0 in zip(hist, first, strict=True):
            if y0 is not None and y0 <= y:
                eligible += _num(r.get('revenue')) or 0.0
            if y not in oi_h or y not in rev_h or rev_h[y] <= 0:
                continue
            n += 1
            rev += rev_h[y]
            net += oi_h[y]
            pos += max(oi_h[y], 0.0)
            oi_vals.append(max(oi_h[y], 0.0))
            cov += _num(r.get('revenue')) or 0.0
        points.append({
            'year': y, 'n': n, 'revenue': rev, 'pool': pos, 'net': net,
            'margin': (net / rev) if rev > 0 else None,
            'hhi': _hhi(oi_vals) if n >= MIN_PANEL else None,
            'coverage': (cov / today_rev) if today_rev > 0 else None,
            'complete': bool(eligible > 0
                             and cov >= COMPLETE_COVERAGE_RATIO * eligible),
        })
    return points


def _growth_windows(points):
    """Candidate ``(y0, y1, block)`` windows, best first: growth is measured
    between two blocks of *block* complete years ending at *y0* and *y1*.

    Single-year endpoints were the first design and failed on the first
    cyclical sector tried: Energy's window started in FY2020, whose pool was
    a sixth of its neighbours', so the pool "grew" 71% a year and a pipeline
    lost 26 points of pool share from a trough base. Averaging each end over
    ENDPOINT_BLOCK years (2018-20 vs 2023-25) keeps one bad year from
    setting the answer. Then single years over the same span, then longer
    spans (a gap in the middle years: REIT - Industrial's 2018-22, where one
    filer's history is missing), then shorter ones down to MIN_GROWTH_SPAN.
    The caller takes the first whose panel is big enough, so a group whose
    companies mostly arrived late (Independent Power Producers:
    Constellation spun off in 2022) gets a shorter window instead of none."""
    complete = {p['year'] for p in points if p['complete']}
    if not complete:
        return []
    y1 = max(complete)
    out = []
    spans = ([GROWTH_WINDOW_YEARS]
             + list(range(GROWTH_WINDOW_YEARS + 1, HISTORY_YEARS))
             + list(range(GROWTH_WINDOW_YEARS - 1, MIN_GROWTH_SPAN - 1, -1)))
    for span in spans:
        y0 = y1 - span
        for block in ((ENDPOINT_BLOCK, 1) if span == GROWTH_WINDOW_YEARS else (1,)):
            if all((y - k) in complete for y in (y0, y1) for k in range(block)):
                out.append((y0, y1, block))
    return out


def _block_years(y, block):
    return [y - k for k in range(block)]


def _cycle(points):
    """Where the latest complete year's margin sits in the sector's own
    range of complete years."""
    pts = [p for p in points if p['complete'] and p['margin'] is not None]
    if len(pts) < MIN_CYCLE_YEARS:
        return None
    margins = [p['margin'] for p in pts]
    cur = pts[-1]
    pct = pctile_rank(margins, cur['margin'])
    if pct >= 0.8:
        label = 'top'
    elif pct <= 0.2:
        label = 'bottom'
    else:
        label = 'mid'
    return {'year': cur['year'], 'margin': cur['margin'], 'pctile': pct,
            'low': min(margins), 'high': max(margins),
            'median': sorted(margins)[len(margins) // 2],
            'years': len(pts), 'position': label}


def _window_analysis(hist, window, n=SHIFT_LIST_N):
    """Growth decomposition, concentration trend and share shifts over one
    consistent panel: the companies reporting revenue (> 0) and operating
    income in every year of both endpoint blocks, so entries and exits
    cannot read as growth or as share won.

    decomposition: (1 + g_pool) = (1 + g_rev) * (1 + g_margin), margin
    being pool / revenue, CAGRs over the span between the blocks' last
    years. shifts: each company's share of the positive pool in each block;
    shares sum to 1 at both ends, so the changes net to zero. hhi_trend:
    the HHI of those same shares at each end.

    Returns ``(decomposition, hhi_trend, shifts)``, each None for a panel
    under MIN_PANEL or an empty pool."""
    y0, y1, block = window
    b0, b1 = _block_years(y0, block), _block_years(y1, block)
    yrs = b0 + b1
    panel = [(r, oi, rev) for r, oi, rev in hist
             if all(y in oi and (rev.get(y) or 0) > 0 for y in yrs)]
    if len(panel) < MIN_PANEL:
        return None, None, None
    pos0 = [sum(max(oi[y], 0.0) for y in b0) for _, oi, _ in panel]
    pos1 = [sum(max(oi[y], 0.0) for y in b1) for _, oi, _ in panel]
    p0, p1 = sum(pos0), sum(pos1)
    r0 = sum(rev[y] for _, _, rev in panel for y in b0)
    r1 = sum(rev[y] for _, _, rev in panel for y in b1)
    if p0 <= 0 or p1 <= 0:
        return None, None, None
    span = y1 - y0
    base = {'y0': y0, 'y1': y1, 'block': block, 'n': len(panel)}

    decomposition = None
    g_pool, g_rev = _cagr(p0, p1, span), _cagr(r0, r1, span)
    if g_pool is not None and g_rev is not None:
        decomposition = dict(base, pool_cagr=g_pool, revenue_cagr=g_rev,
                             margin_cagr=(1 + g_pool) / (1 + g_rev) - 1,
                             margin0=p0 / r0, margin1=p1 / r1)

    h0, h1 = _hhi(pos0), _hhi(pos1)
    ratio = h1 / h0 - 1
    hhi_trend = dict(base, hhi0=h0, hhi1=h1, trend=(
        'concentrating' if ratio > HHI_TREND_RATIO
        else 'fragmenting' if ratio < -HHI_TREND_RATIO else 'stable'))

    moves = [{'ticker': r.get('ticker'),
              'company_name': r.get('company_name'),
              'rating': r.get('rating'),
              'share0': a / p0, 'share1': b / p1, 'delta': b / p1 - a / p0}
             for (r, _, _), a, b in zip(panel, pos0, pos1, strict=True)]
    shifts = dict(
        base,
        gainers=sorted((m for m in moves if m['delta'] > 0),
                       key=lambda m: -m['delta'])[:n],
        losers=sorted((m for m in moves if m['delta'] < 0),
                      key=lambda m: m['delta'])[:n])
    return decomposition, hhi_trend, shifts


def growth_over(rows, window):
    """The consistent-panel growth decomposition of *rows* over a given
    ``(y0, y1, block)`` window, or None when its panel is too thin. Lets
    one group's growth be compared with another's over the same years."""
    hist, _ = _split_stale(_histories(pool_rows(rows)))
    if not hist or not window:
        return None
    return _window_analysis(hist, tuple(window))[0]


def sector_pool_history(rows):
    """Everything time-based for one sector's rows, or None when no row
    carries EDGAR history.

    Returns ``{'points': [...], 'window': [y0, y1, block] | None,
    'decomposition', 'cycle', 'hhi_trend', 'shifts', 'n_with_history',
    'n_rows', 'stale'}``; ``decomposition``, ``cycle``, ``hhi_trend`` and
    ``shifts`` may be None."""
    prow = pool_rows(rows)
    hist, stale = _split_stale(_histories(prow))
    if not hist:
        return None
    today_rev = sum(_num(r.get('revenue')) or 0.0 for r in prow)
    points = _year_points(hist, today_rev)
    if not points:
        return None
    window, (decomposition, hhi_trend, shifts) = None, (None, None, None)
    for cand in _growth_windows(points):
        found = _window_analysis(hist, cand)
        if found[0] is not None or found[2] is not None:
            window, (decomposition, hhi_trend, shifts) = cand, found
            break
    return {
        'points': points,
        'window': list(window) if window else None,
        'decomposition': decomposition,
        'cycle': _cycle(points),
        'hhi_trend': hhi_trend,
        'shifts': shifts,
        'n_with_history': len(hist),
        'n_rows': len(prow),
        # Histories that stopped filing (see _split_stale), largest first,
        # named so the footnote can say who is missing from every year.
        'stale': [{'ticker': r.get('ticker'), 'last_year': max(oi)}
                  for r, oi, _ in sorted(
                      stale, key=lambda h: -(_num(h[0].get('revenue')) or 0.0))],
    }


# ---------------------------------------------------------------------------
# Below the sector, beside it and behind it: industry sub-pools, the economic
# pool, and the pool's structure and price (Sector Analysis step 2).
# ---------------------------------------------------------------------------

MIN_INDUSTRY_COS = 3
OTHER_INDUSTRY = 'Other'
EP_LIST_N = 3
LORENZ_SHARE = 0.80
# Financials whose balance sheets are funded by deposits or repo, so that
# "invested capital" is the funding of a loan book or trading inventory and
# (ROIC - WACC) x it measures nothing: GS and MS read as the sector's two
# largest value destroyers (-$14B, -$19B) on 2026-10-08, COF -$11B. Used
# when a row has no epv_bridge flag; the flag cannot tell V from COF by
# industry, so the fallback leaves out the whole industry and says so.
BALANCE_SHEET_INDUSTRIES = frozenset({
    'Banks - Regional', 'Banks - Diversified', 'Mortgage Finance',
    'Credit Services', 'Capital Markets'})


def _median(vals):
    vals = sorted(vals)
    if not vals:
        return None
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2


def _quantile(vals, q):
    """Linear-interpolated quantile of a non-empty sorted list."""
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _industry_growth(rs, window):
    """An industry's pool growth: over the sector's *window* when its own
    panel there is big enough, so industries and their sector are compared
    over the same years; else over the industry's own best window, ending
    no later than the sector's. Left free, Software - Infrastructure took a
    window ending FY2026 because Microsoft and Oracle had filed it, and
    against the sector re-measured over those years (in effect NVIDIA, the
    other early filer) a 16.6%/yr pool read as lagging a 21.4% sector."""
    if window:
        dec = growth_over(rs, window)
        if dec:
            return dec, False
    dec = (sector_pool_history(rs) or {}).get('decomposition')
    if dec and window and dec['y1'] > window[1]:
        return None, False
    return dec, bool(dec and window)


def industry_pools(rows, window=None):
    """The sector's pool by industry: where in the sector the money is.

    Industries with fewer than MIN_INDUSTRY_COS companies fold into one
    "Other" group (a two-company industry's margin is two companies, not an
    industry). Each entry carries revenue and pool shares of the sector, net
    weighted margin, median ROIC-WACC spread and its consistent-panel pool
    CAGR, over the sector's growth *window* where it can be
    (``own_window`` marks one that could not, see _industry_growth);
    ordered by pool share."""
    prow = pool_rows(rows)
    groups = {}
    for r in prow:
        groups.setdefault(r.get('industry') or OTHER_INDUSTRY, []).append(r)
    small = [k for k, v in groups.items()
             if len(v) < MIN_INDUSTRY_COS and k != OTHER_INDUSTRY]
    if small:
        other = groups.setdefault(OTHER_INDUSTRY, [])
        for k in small:
            other.extend(groups.pop(k))
    rev_tot = sum(_num(r['revenue']) for r in prow)
    pool_tot = sum(max(_num(r['operating_income']), 0.0) for r in prow)
    if rev_tot <= 0:
        return []
    out = []
    for name, rs in groups.items():
        rev = sum(_num(r['revenue']) for r in rs)
        net = sum(_num(r['operating_income']) for r in rs)
        pos = sum(max(_num(r['operating_income']), 0.0) for r in rs)
        # A balance-sheet financial's spread measures nothing (see
        # economic_profit), so banks show no median rather than one the
        # economic pool beside it refuses to use.
        spreads = [_num(r.get('spread')) for r in rs
                   if not is_balance_sheet_financial(r)]
        top = max(rs, key=lambda r: _num(r['operating_income']))
        try:
            dec, own = _industry_growth(rs, window)
        except Exception as e:            # one odd industry must not cost the rest
            logger.warning('industry pool history failed for %s: %s', name, e)
            dec, own = None, False
        out.append({
            'industry': name, 'n': len(rs),
            'revenue_share': rev / rev_tot,
            'pool_share': (pos / pool_tot) if pool_tot > 0 else None,
            'margin': net / rev if rev > 0 else None,
            'median_spread': _median([s for s in spreads if s is not None]),
            'pool_cagr': dec['pool_cagr'] if dec else None,
            'window': [dec['y0'], dec['y1'], dec['block']] if dec else None,
            'own_window': own,
            'top': {'ticker': top.get('ticker'),
                    'pool_share_in_industry': (max(_num(top['operating_income']), 0.0) / pos
                                               if pos > 0 else None)},
            'folded': name == OTHER_INDUSTRY and bool(small),
        })
    out.sort(key=lambda d: -(d['pool_share'] or 0.0))
    return out


def is_balance_sheet_financial(r):
    """Whether invested capital means nothing for *r* (see
    BALANCE_SHEET_INDUSTRIES). The row's own lender flag when it carries one
    (``epv_bridge``, decided from the companyfacts: a filer tagging net
    interest income), else its industry."""
    bridge = r.get('epv_bridge')
    if bridge is not None:
        return bridge == 'equity'
    return (r.get('sector') == 'Financial Services'
            and r.get('industry') in BALANCE_SHEET_INDUSTRIES)


def _latest_ic(r):
    ic = year_series(r.get('_ic_by_year'))
    if not ic:
        return None
    v = ic[max(ic)]
    return v if v > 0 else None


def economic_profit(r):
    """``(ep, ic, reason)``: economic profit = (ROIC - WACC) x latest
    invested capital, or None with the reason it is not measurable
    ('balance_sheet', 'no_capital', 'no_spread'). ROIC is the row's 5-year median,
    so this is today's capital earning a normalised spread."""
    if is_balance_sheet_financial(r):
        return None, None, 'balance_sheet'
    ic = _latest_ic(r)
    if ic is None:
        return None, None, 'no_capital'
    spread = _num(r.get('spread'))
    if spread is None:
        return None, ic, 'no_spread'
    return spread * ic, ic, None


def universe_totals(all_rows):
    """Universe-wide denominators for the per-sector shares: positive
    operating income, positive economic profit, and market cap over the
    rows that carry one (the pool's price)."""
    prow = pool_rows(all_rows)
    oi_pos = sum(max(_num(r['operating_income']), 0.0) for r in prow)
    ep_pos = oi_pos_measured = 0.0
    for r in prow:
        ep, _, _ = economic_profit(r)
        if ep is None:
            continue
        ep_pos += max(ep, 0.0)
        oi_pos_measured += max(_num(r['operating_income']), 0.0)
    priced = [r for r in prow if (_num(r.get('mcap')) or 0) > 0]
    mcap = sum(_num(r['mcap']) for r in priced)
    pool_priced = sum(max(_num(r['operating_income']), 0.0) for r in priced)
    return {'oi_pos': oi_pos, 'ep_pos': ep_pos, 'oi_pos_measured': oi_pos_measured,
            'pool_multiple': (mcap / pool_priced) if pool_priced > 0 else None}


def economic_pool(rows, universe=None, n=EP_LIST_N):
    """The sector's economic-profit pool beside its accounting pool.

    A sector can hold a large share of the market's operating profit and a
    small share of its economic profit, when the capital behind the profit
    earns little over its cost. Balance-sheet financials and rows without invested capital
    or a spread are left out and counted, never zero-filled. Returns None
    when no row is measurable."""
    prow = pool_rows(rows)
    measured, excluded = [], {'balance_sheet': 0, 'no_capital': 0, 'no_spread': 0}
    for r in prow:
        ep, ic, why = economic_profit(r)
        if why:
            excluded[why] += 1
        else:
            measured.append((r, ep, ic))
    if not measured:
        return None

    def _entry(r, ep, ic):
        return {'ticker': r.get('ticker'), 'company_name': r.get('company_name'),
                'rating': r.get('rating'), 'ep': ep, 'ic': ic,
                'spread': _num(r.get('spread'))}

    total = sum(ep for _, ep, _ in measured)
    ic_tot = sum(ic for _, _, ic in measured)
    pos = sum(ep for _, ep, _ in measured if ep > 0)
    neg = sum(ep for _, ep, _ in measured if ep < 0)
    # The headline pairs two shares of the market, so both are taken over
    # the companies whose economic profit is measurable: Financial Services
    # held 22.7% of all US operating profit, but its banks are not in the
    # economic pool, and setting one against the other compared two sets.
    oi_pos = sum(max(_num(r['operating_income']), 0.0) for r, _, _ in measured)
    uni = universe or {}
    ranked = sorted(measured, key=lambda m: -m[1])
    return {
        'n': len(measured), 'excluded': excluded,
        'total': total, 'created': pos, 'destroyed': neg,
        'ep_on_ic': (total / ic_tot) if ic_tot > 0 else None,
        'share_creating': sum(1 for _, ep, _ in measured if ep > 0) / len(measured),
        'creators': [_entry(*m) for m in ranked[:n] if m[1] > 0],
        'destroyers': [_entry(*m) for m in reversed(ranked[-n:]) if m[1] < 0],
        'share_of_us_oi': ((oi_pos / uni['oi_pos_measured'])
                           if uni.get('oi_pos_measured') else None),
        'share_of_us_ep': (pos / uni['ep_pos']) if uni.get('ep_pos') else None,
    }


def pool_structure(rows, universe=None):
    """Concentration of profit (not revenue), the spread of margins, the
    drag of loss-makers, and what the market pays for the pool."""
    prow = pool_rows(rows)
    if not prow:
        return None
    ois = sorted((max(_num(r['operating_income']), 0.0) for r in prow), reverse=True)
    pool = sum(ois)
    net = sum(_num(r['operating_income']) for r in prow)
    out = {'n': len(prow), 'pool': pool, 'net': net,
           'loss_makers': sum(1 for r in prow if _num(r['operating_income']) < 0)}
    if pool > 0:
        shares = [v / pool for v in ois]
        out['profit_hhi'] = sum(s * s for s in shares)
        out['profit_cr4'] = sum(shares[:4])
        acc, k = 0.0, 0
        for s in shares:
            acc += s
            k += 1
            if acc >= LORENZ_SHARE:
                break
        out['lorenz'] = {'pool_share': LORENZ_SHARE, 'companies': k,
                         'company_share': k / len(prow)}
    # Margins outside +/-100% are a revenue base too small to mean anything
    # (the narrative drops them the same way).
    # Taken from the pool's own figures, so the spread describes exactly the
    # revenue and operating income the pool sums.
    margins = sorted(m for m in (_num(r['operating_income']) / _num(r['revenue'])
                                 for r in prow) if abs(m) <= 1.0)
    if len(margins) >= MIN_PANEL:
        p10, p50, p90 = (_quantile(margins, q) for q in (0.1, 0.5, 0.9))
        out['margins'] = {'p10': p10, 'p50': p50, 'p90': p90, 'spread': p90 - p10}
    priced = [r for r in prow if (_num(r.get('mcap')) or 0) > 0]
    pool_priced = sum(max(_num(r['operating_income']), 0.0) for r in priced)
    if pool_priced > 0:
        out['pool_multiple'] = sum(_num(r['mcap']) for r in priced) / pool_priced
        out['universe_pool_multiple'] = (universe or {}).get('pool_multiple')
    return out
