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
from models.sector_pool import pctile_rank

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
        'Defensive flows in late-cycle markets': ('credit', 'cyclical', _m('BAMLH0A0HYM2', 1)),
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
        'Defensive yield bid in down markets': ('credit', 'cyclical', _m('BAMLH0A0HYM2', 1)),
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


def _industry_evidence(ind, entry):
    history = (entry or {}).get('history') or {}
    dec = history.get('decomposition')
    inds = {d['industry']: d for d in ((entry or {}).get('industries') or [])}
    d = inds.get(ind['industry'])
    if d is None:
        return 'no_data', {'source': 'industry', 'industry': ind['industry'],
                           'note': 'fewer than three companies in this industry'}
    if d.get('pool_cagr') is None or not dec:
        return 'no_data', {'source': 'industry', 'industry': ind['industry'],
                           'note': 'too few of its companies reported throughout'}
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


_EVIDENCE = {'macro': lambda ind, sc, en: _macro_evidence(ind, sc),
             'industry': lambda ind, sc, en: _industry_evidence(ind, en),
             'margin_cycle': lambda ind, sc, en: _margin_evidence(ind, en)}


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


def evaluate_forces(sector, sidecar, entry):
    """Each of the sector's forces with its status and evidence.

    *sidecar* is the macro.json payload (may be None: macro forces then read
    no_data and the rest still render); *entry* is the sector's
    SECTOR_POOL entry, already carrying ``history`` and ``industries``.
    Returns ``{'forces': [...], 'market': {...} | None, 'as_of'}``."""
    out = []
    for f in sector_forces(sector):
        ind = f.pop('indicator')
        if ind is None:
            status, evidence = 'qualitative', None
        else:
            try:
                status, evidence = _EVIDENCE[ind['kind']](ind, sidecar, entry)
            except Exception as e:      # one bad reading must not cost the section
                logger.warning('force evidence failed for %s / %s: %s', sector, f['theme'], e)
                status, evidence = 'no_data', {'source': ind['kind'], 'note': 'could not be read'}
            if evidence is not None:
                evidence['sign'] = ind['sign']
        f.update(status=status, evidence=evidence)
        out.append(f)
    return {'forces': out, 'market': market_confirmation(sector, sidecar),
            'as_of': (sidecar or {}).get('as_of')}
