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

HISTORY_YEARS = 10
# A year whose reporters hold less than this share of the revenue that
# could have reported it is incomplete: the latest fiscal year usually is,
# because January-March year-end filers have not reported it yet. It is
# drawn but never used as a growth endpoint, so a filing lag cannot read as
# a fall. See _year_points for what "could have reported" means.
COMPLETE_COVERAGE_RATIO = 0.8
GROWTH_WINDOW_YEARS = 5
# Years averaged at each end of the growth window (see _growth_window).
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
    year. None values and unparseable keys are dropped."""
    out = {}
    for k, v in (d or {}).items():
        if v is None or isinstance(v, bool):
            continue
        try:
            out[int(str(k)[:4])] = float(v)
        except (TypeError, ValueError):
            continue
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


def _pctile_rank(values, v):
    """Share of *values* strictly below *v*, ties counted half."""
    if not values:
        return None
    below = sum(1 for x in values if x < v)
    equal = sum(1 for x in values if x == v)
    return (below + 0.5 * (equal - 1)) / max(len(values) - 1, 1)


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


def _growth_window(points):
    """``(y0, y1, block)``: growth is measured between two blocks of
    *block* complete years ending at *y0* and *y1*.

    Single-year endpoints were the first design and failed on the first
    cyclical sector tried: Energy's window started in FY2020, whose pool was
    a sixth of its neighbours', so the pool "grew" 71% a year and a pipeline
    lost 26 points of pool share from a trough base. Averaging each end over
    ENDPOINT_BLOCK years (2018-20 vs 2023-25) keeps one bad year from
    setting the answer. Falls back to single years when the history is too
    short for blocks, then to the oldest complete year at least
    MIN_GROWTH_SPAN back. None when no such pair exists."""
    complete = {p['year'] for p in points if p['complete']}
    if not complete:
        return None
    y1 = max(complete)
    y0 = y1 - GROWTH_WINDOW_YEARS
    for block in (ENDPOINT_BLOCK, 1):
        if all((y - k) in complete for y in (y0, y1) for k in range(block)):
            return y0, y1, block
    older = sorted(y for y in complete if y1 - y >= MIN_GROWTH_SPAN)
    return (older[0], y1, 1) if older else None


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
    pct = _pctile_rank(margins, cur['margin'])
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
    window = _growth_window(points)
    decomposition, hhi_trend, shifts = (_window_analysis(hist, window)
                                        if window else (None, None, None))
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
