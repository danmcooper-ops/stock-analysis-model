"""Evidence for each sector's structural headwinds and tailwinds.

The forces themselves are the curated text in ``models/narrative.py``
(``_SECTOR_THESIS_RISKS`` / ``_SECTOR_THESIS_TAILWINDS``), which the Claude
macro narrative also reads; that text is unchanged. This module adds, per
force, what kind of force it is and what would show it acting, then reads
that evidence at render time:

- ``macro``: a FRED series from the macro sidecar (``macro.json``). Status
  comes from where the reading sits in its own history and which way it
  moved over the year, in the direction that strengthens the force.
- ``industry``: one industry's own pool growth against the sector's
  (``models/sector_pool``). A force that is really acting shows up in the
  profits of the industry it names.
- ``margin_cycle``: the sector's net margin within its own decade range, and
  its last move. For forces that act through margins (input costs, pricing
  power, capital discipline).
- none: a force no series measures (antitrust, wildfire liability). Its
  status is "qualitative" and nothing is invented for it.

Metadata is keyed by each force's theme, the text before " — ", so it cannot
drift onto the wrong force when a list is reordered; a test pins that every
force has metadata and every metadata entry a force.

Statuses: active (strong and not fading), building (moving toward strong),
easing (fading), dormant (neither), qualitative, no_data.
"""

import logging
from datetime import date, timedelta

from models.narrative import _SECTOR_THESIS_RISKS, _SECTOR_THESIS_TAILWINDS
from models.sector_pool import (_num, growth_over, is_balance_sheet_financial, pctile_rank,
                                 pool_rows)

logger = logging.getLogger(__name__)

# Status thresholds, in "pressure" terms: a reading's percentile turned so
# that 1 is the strongest the force has been (see _status).
STRONG_PCTILE = 0.70
MOVE_PCTILE = 0.10
EASING_FLOOR = 0.40
# Industry evidence: industry pool CAGR minus the sector's, signed.
INDUSTRY_ACTIVE = 0.03
INDUSTRY_BUILDING = 0.01
SPARK_DAYS = 730
SPARK_POINTS = 48
EXPOSURE_LIST_N = 3
# Companies named on an exposure list hold at least this share of the
# sector's pool: ranked on leverage or margin alone, the most exposed were
# micro-caps nobody reads the sector page for (ABR, TRTX in Real Estate).
# The reach still counts every company.
MIN_LIST_POOL_SHARE = 0.005
MIN_EXPOSURE_ROWS = 6
# Balance weights: an active force counts fully, a building one half.
STATUS_WEIGHT = {'active': 1.0, 'building': 0.5}
# A live force whose exposure is not measured is weighted as if it reached
# a third of the sector, what a measured force's exposed third reaches when
# companies are of even size.
DEFAULT_EXPOSED_SHARE = 1 / 3

# Who feels a force most, by type: (row field, label, higher_is_more_exposed).
# A rate or credit force acts through the balance sheet; a commodity or
# input-cost force hits the thinnest margins first; a demand or capex cycle
# hits the companies whose returns already swing; pricing power is held by
# the widest margins. Forces whose evidence is an industry are felt by that
# industry's companies instead (see exposure()).
EXPOSURE_METRICS = {
    'rates': ('nd_ebitda', 'net debt / EBITDA', True),
    'credit': ('nd_ebitda', 'net debt / EBITDA', True),
    'commodity': ('operating_margin', 'operating margin', False),
    'input_costs': ('operating_margin', 'operating margin', False),
    'trade': ('operating_margin', 'operating margin', False),
    'pricing': ('operating_margin', 'operating margin', True),
    'consumer': ('roic_cv', 'ROIC variability', True),
    'capex_cycle': ('roic_cv', 'ROIC variability', True),
}


def _m(series, sign):
    return {'kind': 'macro', 'series': series, 'sign': sign}


def _i(industry, sign):
    return {'kind': 'industry', 'industry': industry, 'sign': sign}


def _c(sign):
    return {'kind': 'margin_cycle', 'sign': sign}


# sign: +1 when a higher reading strengthens the force, -1 when a lower one
# does (falling rates strengthen a rate-cut tailwind; a weak oil price
# strengthens Energy's commodity headwind).
FORCE_META = {
    'Technology': {
        'Antitrust and platform-regulation risk': ('regulation', 'secular', None),
        'AI disruption of existing software moats':
            ('disruption', 'secular', _i('Software - Application', -1)),
        'Cybersecurity liability and talent cost inflation': ('input_costs', 'secular', _c(-1)),
        'AI capex super-cycle': ('capex_cycle', 'cyclical', _i('Semiconductors', 1)),
        'Software dollar share keeps rising':
            ('demand', 'secular', _i('Software - Infrastructure', 1)),
        'Durable enterprise digitisation': ('demand', 'secular', _i('Software - Application', 1)),
    },
    'Communication Services': {
        'Advertising-cycle exposure and content/regulatory risk': ('demand', 'cyclical', _m('RSAFS', -1)),
        'Cord-cutting acceleration and audience fragmentation':
            ('disruption', 'secular', _i('Telecom Services', -1)),
        'AI-generated content commoditisation': ('disruption', 'secular', None),
        'Retail-media and connected-TV ad surge':
            ('demand', 'secular', _i('Internet Content & Information', 1)),
        'Premium content scarcity value': ('pricing', 'secular', _i('Entertainment', 1)),
        '5G and fixed-wireless monetisation': ('demand', 'secular', _i('Telecom Services', 1)),
    },
    'Consumer Cyclical': {
        'Consumer demand is fundamentally cyclical': ('consumer', 'cyclical', _m('UMCSENT', -1)),
        'E-commerce margin compression and tariff exposure': ('trade', 'secular', _c(-1)),
        'Brand relevance decay': ('disruption', 'secular', None),
        'Rate-cut sensitivity on big-ticket categories': ('rates', 'cyclical', _m('MORTGAGE30US', -1)),
        'Brand consolidation and DTC scale': ('demand', 'secular', _i('Internet Retail', 1)),
        'Travel and experiences super-cycle': ('demand', 'cyclical', _i('Travel Services', 1)),
    },
    'Consumer Defensive': {
        'Slow-motion taste and channel shifts': ('disruption', 'secular', None),
        'Input cost inflation from agricultural commodities': ('input_costs', 'cyclical', _c(-1)),
        'GLP-1 and health-trend demand destruction': ('disruption', 'secular', _i('Packaged Foods', -1)),
        'Pricing power durability': ('pricing', 'secular', _c(1)),
        'Defensive flows in late-cycle markets': ('flows', 'cyclical', _m('BAMLH0A0HYM2', 1)),
        'Emerging-market demographic and category penetration': ('demographics', 'secular', None),
    },
    'Energy': {
        'The commodity cycle': ('commodity', 'cyclical', _m('DCOILWTICO', -1)),
        'ESG-driven capital flight and stranded asset risk': ('regulation', 'secular', None),
        'Geopolitical supply shocks': ('geopolitics', 'cyclical', None),
        'Capital discipline supporting per-barrel returns': ('pricing', 'secular', _c(1)),
        'Structural underinvestment in conventional supply': ('commodity', 'secular', _m('DCOILWTICO', 1)),
        'Geopolitical risk premium': ('geopolitics', 'cyclical', None),
    },
    'Financial Services': {
        'Credit cycles and regulatory capital': ('credit', 'cyclical', _m('BAMLH0A0HYM2', 1)),
        'Fintech disintermediation': ('disruption', 'secular', None),
        'Interest rate regime shifts': ('rates', 'cyclical', _m('DGS10', 1)),
        'Yield-curve normalisation': ('rates', 'cyclical', _m('T10Y2Y', 1)),
        'M&A, IPO, and capital-markets recovery': ('demand', 'cyclical', _i('Capital Markets', 1)),
        'Demographic tailwind for wealth and asset management':
            ('demographics', 'secular', _i('Asset Management', 1)),
    },
    'Healthcare': {
        'Patent cliffs, drug-pricing reform, and reimbursement pressure': ('regulation', 'secular', None),
        'Clinical trial concentration risk': ('regulation', 'secular', None),
        'GLP-1 disruption across sub-sectors': ('disruption', 'secular', _i('Medical Devices', -1)),
        'Aging-demographics demand growth': ('demographics', 'secular', None),
        'GLP-1 demand expansion beyond obesity':
            ('demand', 'secular', _i('Drug Manufacturers - General', 1)),
        'AI-accelerated drug discovery and trial design': ('disruption', 'secular', None),
    },
    'Industrials': {
        'Cyclicality and capital intensity': ('capex_cycle', 'cyclical', _m('INDPRO', -1)),
        'Supply chain reshoring costs and labour shortages': ('input_costs', 'cyclical', _m('UNRATE', -1)),
        'Tariff and trade-policy exposure': ('trade', 'cyclical', None),
        'Reshoring and friend-shoring capex':
            ('capex_cycle', 'secular', _i('Specialty Industrial Machinery', 1)),
        'Defence spending uplift': ('demand', 'secular', _i('Aerospace & Defense', 1)),
        'Infrastructure and grid modernisation':
            ('capex_cycle', 'secular', _i('Electrical Equipment & Parts', 1)),
    },
    'Basic Materials': {
        'Commodity price volatility and cost-curve position': ('commodity', 'cyclical', _m('PCOPPUSDM', -1)),
        'Environmental remediation liability': ('regulation', 'secular', None),
        'Trade policy and tariff whiplash': ('trade', 'cyclical', None),
        'Electrification metals demand': ('commodity', 'secular', _m('PCOPPUSDM', 1)),
        'Supply-side discipline from a decade of underinvestment': ('pricing', 'secular', _c(1)),
        'Dollar weakness as a tailwind to dollar-denominated commodities':
            ('fx', 'cyclical', _m('DTWEXBGS', -1)),
    },
    'Utilities': {
        'Regulatory rate-case risk and rising cost of capital': ('rates', 'cyclical', _m('DGS10', 1)),
        'Grid modernisation capex burden': ('regulation', 'secular', None),
        'Wildfire and climate liability': ('regulation', 'secular', None),
        'AI data-centre electricity demand':
            ('demand', 'secular', _i('Utilities - Independent Power Producers', 1)),
        'Grid modernisation and electrification capex':
            ('capex_cycle', 'secular', _i('Utilities - Regulated Electric', 1)),
        'Defensive yield bid in down markets': ('flows', 'cyclical', _m('BAMLH0A0HYM2', 1)),
    },
    'Real Estate': {
        'Interest-rate sensitivity and tenant-credit risk': ('rates', 'cyclical', _m('DGS10', 1)),
        'Remote work structural vacancy': ('disruption', 'secular', None),
        'Climate and insurance cost escalation': ('regulation', 'secular', None),
        'Data-centre and AI-infrastructure REIT demand': ('demand', 'secular', _i('REIT - Specialty', 1)),
        'Industrial / logistics tailwind from reshoring and e-commerce':
            ('demand', 'secular', _i('REIT - Industrial', 1)),
        'Rate-cut re-rating of cap rates': ('rates', 'cyclical', _m('DGS10', -1)),
    },
}


def split_theme(text):
    """``(theme, detail)``: the text before and after its first " — "."""
    theme, _, detail = text.partition(' — ')
    return theme.strip(), detail.strip()


def sector_forces(sector):
    """The sector's forces with their metadata, headwinds first:
    ``[{'kind', 'theme', 'detail', 'text', 'type', 'horizon', 'indicator'}]``.
    A force without metadata is returned as qualitative rather than dropped."""
    meta = FORCE_META.get(sector, {})
    out = []
    for kind, src in (('headwind', _SECTOR_THESIS_RISKS), ('tailwind', _SECTOR_THESIS_TAILWINDS)):
        for text in src.get(sector, []):
            theme, detail = split_theme(text)
            typ, horizon, ind = meta.get(theme, (None, None, None))
            out.append({'kind': kind, 'theme': theme, 'detail': detail, 'text': text,
                        'type': typ, 'horizon': horizon, 'indicator': ind})
    return out


# --- status -----------------------------------------------------------------

def _status(pressure, move):
    """Status from *pressure* (0..1, 1 = the force at its strongest in the
    reading's own history) and *move* (the change in pressure over the
    year, same units)."""
    if pressure >= STRONG_PCTILE:
        return 'easing' if move <= -MOVE_PCTILE else 'active'
    if move >= MOVE_PCTILE:
        return 'building'
    if move <= -MOVE_PCTILE and pressure >= EASING_FLOOR:
        return 'easing'
    return 'dormant'


def _even_weekly(pts):
    """At most one point per week, the last in it. The sidecar's ``hist`` is
    downsampled to daily for its trailing year and weekly before it
    (macro_dashboard.downsample), so ranking against it as shipped weighted
    the last year ~3.5x; on a weekly grid every week counts once. A monthly
    series is unchanged."""
    if not pts:
        return []
    d0 = pts[0][0]
    by_week = {}
    for d, v in pts:
        by_week[(d - d0).days // 7] = (d, v)
    return [by_week[k] for k in sorted(by_week)]


def _fmt_reading(v, fmt, suffix=''):
    if v is None:
        return 'N/A'
    if fmt == 'pct2':
        s = f'{v:.2f}%'
    elif fmt == 'pct1':
        s = f'{v:.1f}%'
    elif fmt == 'int':
        s = f'{v:,.0f}'
    elif fmt == 'n2':
        s = f'{v:.2f}'
    else:
        s = f'{v:.1f}'
    return s + (suffix or '')


def _fmt_change(v, fmt):
    if v is None:
        return None
    unit = ' pts' if fmt in ('pct1', 'pct2') else ''
    dp = 0 if fmt == 'int' else 2 if fmt in ('pct2', 'n2') else 1
    return f'{v:+,.{dp}f}{unit}'


def _parse_day(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def _macro_evidence(ind, sidecar):
    series = ((sidecar or {}).get('series') or {}).get(ind['series'])
    if not series:
        return 'no_data', {'source': 'macro', 'series': ind['series'],
                           'note': 'series not in this run’s macro data'}
    hist = series.get('hist') or {}
    ds, vs = hist.get('d') or [], hist.get('v') or []
    pts = [(d, v) for d, v in ((_parse_day(d), v) for d, v in zip(ds, vs, strict=False))
           if d is not None and isinstance(v, (int, float))]
    latest = (series.get('latest') or {})
    lv, ld = latest.get('v'), _parse_day(latest.get('d'))
    if lv is None or ld is None or len(pts) < 12:
        return 'no_data', {'source': 'macro', 'series': ind['series'],
                           'note': 'too little history to judge'}
    # Level: the series' own percentile, which the builder took over the full
    # undownsampled history and the Macro tab quotes. Move: both ends ranked
    # on one evenly weighted weekly grid, so the year-ago reading and today's
    # are measured against the same history.
    even = _even_weekly(pts)
    values = [v for _, v in even]
    prior = [v for d, v in even if d <= ld - timedelta(days=365)]
    pctile = series.get('pctile')
    p_now = pctile if isinstance(pctile, (int, float)) else pctile_rank(values, lv)
    p_then = pctile_rank(values, prior[-1]) if prior else None
    sign = ind['sign']
    pressure = p_now if sign > 0 else 1 - p_now
    move = (sign * (pctile_rank(values, lv) - p_then)) if p_then is not None else 0.0
    fmt, suffix = series.get('fmt'), series.get('suffix', '')
    cut = ld - timedelta(days=SPARK_DAYS)
    spark = [v for d, v in pts if d >= cut]
    step = max(1, len(spark) // SPARK_POINTS)
    spark = spark[::step][-SPARK_POINTS:]
    return _status(pressure, move), {
        'source': 'macro', 'series': ind['series'], 'label': series.get('l'),
        'reading': _fmt_reading(lv, fmt, suffix), 'as_of': latest.get('d'),
        'pctile': p_now, 'window': series.get('pct_win'),
        'change_1y': _fmt_change(series.get('chg_1y'), fmt),
        'pressure': pressure, 'move': move, 'spark': spark,
    }


def _industry_evidence(ind, entry, rows=None):
    """The industry's pool growth against the sector's over the SAME years.
    An industry can fall back to its own window (a late spin-off, a filer's
    gap), so the sector's growth is re-measured over the industry's window
    from *rows* rather than quoted from the sector's own window."""
    history = (entry or {}).get('history') or {}
    dec = history.get('decomposition')
    inds = {d['industry']: d for d in ((entry or {}).get('industries') or [])}
    d = inds.get(ind['industry'])
    if d is None:
        return 'no_data', {'source': 'industry', 'industry': ind['industry'],
                           'note': 'fewer than three companies in this industry'}
    if d.get('pool_cagr') is None:
        return 'no_data', {'source': 'industry', 'industry': ind['industry'],
                           'note': 'too few of its companies reported throughout'}
    window = d.get('window')
    if window and (not dec or window != [dec['y0'], dec['y1'], dec['block']]):
        dec = growth_over(rows, window) if rows else None
    if not dec:
        return 'no_data', {'source': 'industry', 'industry': ind['industry'],
                           'note': 'no sector growth over the same years to compare with'}
    diff = d['pool_cagr'] - dec['pool_cagr']
    v = ind['sign'] * diff
    status = ('active' if v >= INDUSTRY_ACTIVE
              else 'building' if v >= INDUSTRY_BUILDING else 'dormant')
    return status, {
        'source': 'industry', 'industry': ind['industry'],
        'industry_cagr': d['pool_cagr'], 'sector_cagr': dec['pool_cagr'],
        'diff': diff, 'pool_share': d.get('pool_share'),
        'window': [dec['y0'], dec['y1'], dec['block']],
    }


def _margin_evidence(ind, entry):
    history = (entry or {}).get('history') or {}
    cyc = history.get('cycle')
    pts = [p for p in history.get('points') or []
           if p.get('complete') and p.get('margin') is not None]
    if not cyc or len(pts) < 2:
        return 'no_data', {'source': 'margin_cycle',
                           'note': 'too few complete years of sector margins'}
    margins = [p['margin'] for p in pts]
    p_now = pctile_rank(margins, pts[-1]['margin'])
    p_prev = pctile_rank(margins, pts[-2]['margin'])
    sign = ind['sign']
    pressure = p_now if sign > 0 else 1 - p_now
    move = sign * (p_now - p_prev)
    return _status(pressure, move), {
        'source': 'margin_cycle', 'year': pts[-1]['year'], 'margin': pts[-1]['margin'],
        'prev_margin': pts[-2]['margin'], 'pctile': p_now,
        'low': cyc['low'], 'high': cyc['high'], 'years': cyc['years'],
        'pressure': pressure, 'move': move,
    }


_EVIDENCE = {'macro': lambda ind, sc, en, rows: _macro_evidence(ind, sc),
             'industry': lambda ind, sc, en, rows: _industry_evidence(ind, en, rows),
             'margin_cycle': lambda ind, sc, en, rows: _margin_evidence(ind, en)}


def market_confirmation(sector, sidecar):
    """The sector ETF's relative strength against the market, from the
    macro sidecar's ``sector_data``: whether price action agrees with the
    balance of forces. None when the sidecar has nothing for the sector."""
    sd = ((sidecar or {}).get('sector_data') or {}).get(sector) or {}
    rs3, rs6 = sd.get('rs_3m'), sd.get('rs_6m')
    if rs3 is None and rs6 is None:
        return None
    return {'etf': sd.get('etf'), 'rs_3m': rs3, 'rs_6m': rs6,
            'trend': sd.get('trend'), 'as_of': (sidecar or {}).get('as_of')}


def _pool_of(rows):
    """``(rows, {ticker: (revenue, positive operating income)})`` over the
    pool's own rows (sector_pool.pool_rows: revenue and operating income
    present, one row per issuer), so exposure counts the same companies as
    the industries table."""
    prow = pool_rows(rows or [])
    return prow, {r.get('ticker'): (_num(r['revenue']), max(_num(r['operating_income']), 0.0))
                  for r in prow}


def _brief(r, value=None):
    return {'ticker': r.get('ticker'), 'company_name': r.get('company_name'),
            'rating': r.get('rating'), 'value': value}


def _shares(members, sizes, rev_tot, pool_tot):
    rev = sum(sizes[r['ticker']][0] for r in members)
    pool = sum(sizes[r['ticker']][1] for r in members)
    return rev / rev_tot, (pool / pool_tot) if pool_tot > 0 else None


def exposure(force, indicator, rows):
    """Which of the sector's companies feel *force* most, and how much of
    the sector they are.

    Reach is the exposed companies' share of the sector's REVENUE: the
    business the force acts on. Their share of the profit pool is shown
    beside it but not used for weight, because for a margin-acting force the
    ranking and the pool are the same ordering: the thin-margin third that a
    commodity downturn hits holds almost none of the pool by construction,
    and the wide-margin third holds most of it, so pool-weighted reach
    tilted every balance toward pricing tailwinds.

    - An industry-evidenced force is felt by that industry's companies.
    - A rate or credit force in a sector of balance-sheet financials is felt
      by them as a group: their net debt / EBITDA is deposit or repo funding
      (scoring masks it), so ranking them on it named banks by a garbage
      ratio. Other companies there are ranked on it as usual.
    - Otherwise the force's type picks a row metric (EXPOSURE_METRICS), and
      the most-exposed third by that metric is the exposed set.

    Returns None when none applies or too few companies carry the metric."""
    prow, sizes = _pool_of(rows)
    rev_tot = sum(v[0] for v in sizes.values())
    pool_tot = sum(v[1] for v in sizes.values())
    if rev_tot <= 0:
        return None

    def _group(members, **kw):
        members = sorted(members, key=lambda r: -sizes[r['ticker']][1])
        reach, pool_share = _shares(members, sizes, rev_tot, pool_tot)
        return dict(kw, n=len(members), reach=reach, pool_share=pool_share,
                    most=[_brief(r, sizes[r['ticker']][1] / pool_tot if pool_tot > 0 else None)
                          for r in members[:EXPOSURE_LIST_N]],
                    least=[])

    if indicator and indicator.get('kind') == 'industry':
        members = [r for r in prow if r.get('industry') == indicator['industry']]
        return _group(members, basis='industry', industry=indicator['industry']) if members else None
    spec = EXPOSURE_METRICS.get(force.get('type'))
    if not spec:
        return None
    field, label, higher = spec
    candidates = prow
    if field == 'nd_ebitda':
        lenders = [r for r in prow if is_balance_sheet_financial(r)]
        if lenders and _shares(lenders, sizes, rev_tot, pool_tot)[0] >= 1 / 3:
            return _group(lenders, basis='group', label='lenders and broker-dealers')
        candidates = [r for r in prow if not is_balance_sheet_financial(r)]
    ranked = [(r, _num(r.get(field))) for r in candidates]
    ranked = [(r, v) for r, v in ranked if v is not None]
    if len(ranked) < MIN_EXPOSURE_ROWS:
        return None
    ranked.sort(key=lambda x: -x[1] if higher else x[1])
    top = [r for r, _ in ranked[:max(1, len(ranked) // 3)]]
    reach, pool_share = _shares(top, sizes, rev_tot, pool_tot)
    named = [(r, v) for r, v in ranked
             if pool_tot > 0 and sizes[r['ticker']][1] / pool_tot >= MIN_LIST_POOL_SHARE]
    if len(named) < 2 * EXPOSURE_LIST_N:
        named = ranked
    return {'basis': 'metric', 'metric': field, 'label': label,
            'higher_is_exposed': higher, 'n': len(ranked),
            'reach': reach, 'pool_share': pool_share,
            'most': [_brief(r, v) for r, v in named[:EXPOSURE_LIST_N]],
            'least': [_brief(r, v) for r, v in named[::-1][:EXPOSURE_LIST_N]]}


def force_balance(forces):
    """Tailwinds against headwinds, each live force (active, or building at
    half weight) weighted by its reach, the share of the sector's revenue
    it acts on (see exposure). Returns the
    two sums and their difference; a measured reach is used where there is
    one, DEFAULT_EXPOSED_SHARE where there is not."""
    sums = {'tailwind': 0.0, 'headwind': 0.0}
    estimated = 0
    for f in forces:
        w = STATUS_WEIGHT.get(f.get('status'))
        if not w:
            continue
        reach = (f.get('exposure') or {}).get('reach')
        if reach is None:
            reach = DEFAULT_EXPOSED_SHARE
            estimated += 1
        sums[f['kind']] += w * reach
    return {'tailwind': sums['tailwind'], 'headwind': sums['headwind'],
            'net': sums['tailwind'] - sums['headwind'], 'estimated': estimated}


def evaluate_forces(sector, sidecar, entry, rows=None):
    """Each of the sector's forces with its status and evidence.

    *sidecar* is the macro.json payload (may be None: macro forces then read
    no_data and the rest still render); *entry* is the sector's
    SECTOR_POOL entry, already carrying ``history`` and ``industries``;
    *rows* the sector's result rows, for each force's exposure.
    Returns ``{'forces': [...], 'market': {...} | None, 'balance', 'as_of'}``."""
    out = []
    for f in sector_forces(sector):
        ind = f.pop('indicator')
        if ind is None:
            status, evidence = 'qualitative', None
        else:
            try:
                status, evidence = _EVIDENCE[ind['kind']](ind, sidecar, entry, rows)
            except Exception as e:      # one bad reading must not cost the section
                logger.warning('force evidence failed for %s / %s: %s', sector, f['theme'], e)
                status, evidence = 'no_data', {'source': ind['kind'], 'note': 'could not be read'}
            if evidence is not None:
                evidence['sign'] = ind['sign']
        try:
            exp = exposure(f, ind, rows or [])
        except Exception as e:
            logger.warning('force exposure failed for %s / %s: %s', sector, f['theme'], e)
            exp = None
        f.update(status=status, evidence=evidence, exposure=exp)
        out.append(f)
    return {'forces': out, 'market': market_confirmation(sector, sidecar),
            'balance': force_balance(out), 'as_of': (sidecar or {}).get('as_of')}
