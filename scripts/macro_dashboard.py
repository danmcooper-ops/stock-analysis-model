"""Builder for the Macro Outlook dashboard tab.

Fetches a curated set of FRED series through data/fred_client.FREDClient,
applies transforms (YoY, MoM diff, scaling), computes numeric context
(latest, prior, 1M/1Y change, historical percentile, z-score), downsamples
long daily histories, and emits:

  - a small inline ``summary`` (regime + headline tiles + sparklines) that
    is Jinja-injected so the tab paints instantly, and
  - a full ``sidecar`` payload written to macro.json and lazy-loaded when
    the tab opens.

All transforms are pure functions on {date: value} dicts so they are
testable offline; network happens only through the injected client.
"""

import sys
import os
from datetime import date, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Each entry: id (FRED series), l (label), sec (section key), freq
# (d/w/m/q), tr (transform: level|yoy|mom_diff), fmt (pct1|pct2|n1|n2|int),
# suffix (display unit appended after the number), good (which direction
# colors green: up|down|none), scale (multiply raw FRED value before
# anything else — FRED units vary per series).
MACRO_SERIES = [
    # --- Rates & Curve ---
    dict(id='DGS10', l='10Y Treasury', sec='rates', freq='d', tr='level',
         fmt='pct2', suffix='', good='none', scale=1),
    dict(id='T10Y2Y', l='10Y–2Y Spread', sec='rates', freq='d', tr='level',
         fmt='pct2', suffix='', good='up', scale=1, zero_line=True),
    dict(id='T10Y3M', l='10Y–3M Spread', sec='rates', freq='d', tr='level',
         fmt='pct2', suffix='', good='up', scale=1, zero_line=True),
    dict(id='DFF', l='Fed Funds (Effective)', sec='rates', freq='d', tr='level',
         fmt='pct2', suffix='', good='none', scale=1),
    # --- Inflation ---
    dict(id='CPIAUCSL', l='CPI (YoY)', sec='inflation', freq='m', tr='yoy',
         fmt='pct1', suffix='', good='down', scale=1),
    dict(id='CPILFESL', l='Core CPI (YoY)', sec='inflation', freq='m', tr='yoy',
         fmt='pct1', suffix='', good='down', scale=1),
    dict(id='PCEPI', l='PCE (YoY)', sec='inflation', freq='m', tr='yoy',
         fmt='pct1', suffix='', good='down', scale=1),
    dict(id='PCEPILFE', l='Core PCE (YoY)', sec='inflation', freq='m', tr='yoy',
         fmt='pct1', suffix='', good='down', scale=1),
    dict(id='T10YIE', l='10Y Breakeven', sec='inflation', freq='d', tr='level',
         fmt='pct2', suffix='', good='none', scale=1),
    dict(id='T5YIFR', l='5y5y Forward Inflation', sec='inflation', freq='d',
         tr='level', fmt='pct2', suffix='', good='none', scale=1),
    # --- Growth & Labor ---
    dict(id='A191RL1Q225SBEA', l='Real GDP (QoQ SAAR)', sec='growth', freq='q',
         tr='level', fmt='pct1', suffix='', good='up', scale=1, zero_line=True),
    dict(id='INDPRO', l='Industrial Production (YoY)', sec='growth', freq='m',
         tr='yoy', fmt='pct1', suffix='', good='up', scale=1, zero_line=True),
    dict(id='RSAFS', l='Retail Sales (YoY)', sec='growth', freq='m', tr='yoy',
         fmt='pct1', suffix='', good='up', scale=1, zero_line=True),
    dict(id='UNRATE', l='Unemployment Rate', sec='growth', freq='m', tr='level',
         fmt='pct1', suffix='', good='down', scale=1),
    dict(id='PAYEMS', l='Nonfarm Payrolls (MoM)', sec='growth', freq='m',
         tr='mom_diff', fmt='int', suffix='K', good='up', scale=1,
         zero_line=True),
    dict(id='ICSA', l='Initial Jobless Claims', sec='growth', freq='w',
         tr='level', fmt='int', suffix='K', good='down', scale=1e-3),
    dict(id='JTSJOL', l='Job Openings', sec='growth', freq='m', tr='level',
         fmt='n1', suffix='M', good='up', scale=1e-3),
    dict(id='SAHMREALTIME', l='Sahm Rule', sec='growth', freq='m', tr='level',
         fmt='n2', suffix='', good='down', scale=1, threshold=0.5),
    # --- Credit & Conditions ---
    dict(id='BAMLC0A0CM', l='IG Corporate OAS', sec='credit', freq='d',
         tr='level', fmt='pct2', suffix='', good='down', scale=1),
    dict(id='BAMLH0A0HYM2', l='High Yield OAS', sec='credit', freq='d',
         tr='level', fmt='pct2', suffix='', good='down', scale=1),
    dict(id='NFCI', l='Chicago Fed NFCI', sec='credit', freq='w', tr='level',
         fmt='n2', suffix='', good='down', scale=1, zero_line=True),
    dict(id='WALCL', l='Fed Balance Sheet ($T)', sec='credit', freq='w',
         tr='level', fmt='n2', suffix='', good='none', scale=1e-6),
    dict(id='M2SL', l='M2 Money Supply (YoY)', sec='credit', freq='m',
         tr='yoy', fmt='pct1', suffix='', good='none', scale=1,
         zero_line=True),
    dict(id='DTWEXBGS', l='Broad Dollar Index', sec='credit', freq='d',
         tr='level', fmt='n1', suffix='', good='none', scale=1),
    # --- Housing & Consumer ---
    dict(id='HOUST', l='Housing Starts (SAAR)', sec='housing', freq='m',
         tr='level', fmt='int', suffix='K', good='up', scale=1),
    dict(id='MORTGAGE30US', l='30Y Mortgage Rate', sec='housing', freq='w',
         tr='level', fmt='pct2', suffix='', good='down', scale=1),
    dict(id='CSUSHPINSA', l='Case–Shiller Home Prices (YoY)', sec='housing',
         freq='m', tr='yoy', fmt='pct1', suffix='', good='none', scale=1),
    dict(id='UMCSENT', l='UMich Consumer Sentiment', sec='housing', freq='m',
         tr='level', fmt='n1', suffix='', good='up', scale=1),
    dict(id='PSAVERT', l='Personal Savings Rate', sec='housing', freq='m',
         tr='level', fmt='pct1', suffix='', good='none', scale=1),
]

SECTIONS = [
    {'k': 'rates', 'l': 'Rates & Curve'},
    {'k': 'inflation', 'l': 'Inflation'},
    {'k': 'growth', 'l': 'Growth & Labor'},
    {'k': 'credit', 'l': 'Credit & Conditions'},
    {'k': 'housing', 'l': 'Housing & Consumer'},
]

# Headline tiles on the Overview sub-tab, in display order.
OVERVIEW_IDS = ['DGS10', 'T10Y2Y', 'PCEPILFE', 'UNRATE', 'BAMLH0A0HYM2',
                'A191RL1Q225SBEA', 'NFCI', 'DTWEXBGS']

DAILY_HISTORY_YEARS = 10
SPARK_POINTS = 60
MIN_SERIES_FOR_DASHBOARD = 5


# ---------------------------------------------------------------------------
# Pure transforms ({date: value} in, {date: value} or scalar out)
# ---------------------------------------------------------------------------

def yoy(obs, tolerance_days=45):
    """Year-over-year % change. Matches each observation against the nearest
    observation ~12 months earlier within a tolerance window — monthly series
    drift in day-of-month, so exact-date matching would drop most points."""
    if not obs:
        return {}
    dates = sorted(obs)
    out = {}
    for d in dates:
        target = d - timedelta(days=365)
        best, best_gap = None, tolerance_days + 1
        for e in dates:
            gap = abs((e - target).days)
            if gap < best_gap:
                best, best_gap = e, gap
            elif e > target + timedelta(days=tolerance_days):
                break
        if best is None or not obs[best]:
            continue
        out[d] = (obs[d] / obs[best] - 1.0) * 100.0
    return out


def mom_diff(obs):
    """First difference between consecutive observations."""
    if not obs:
        return {}
    dates = sorted(obs)
    return {d: obs[d] - obs[prev]
            for prev, d in zip(dates, dates[1:])}


def percentile_rank(values, latest):
    """Fraction of history at or below the latest value, in [0, 1]."""
    if not values or latest is None:
        return None
    n = sum(1 for v in values if v <= latest)
    return n / len(values)


def zscore(values, latest):
    if not values or latest is None or len(values) < 3:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    if var <= 0:
        return 0.0
    return (latest - mean) / var ** 0.5


def downsample(obs, daily_keep_days=365, weekly_step=7):
    """Thin a dense daily series: full density in the trailing year, one
    point per `weekly_step` days beyond. First and last points always kept."""
    if not obs:
        return {}
    dates = sorted(obs)
    last = dates[-1]
    cutoff = last - timedelta(days=daily_keep_days)
    out = {}
    kept = None
    for d in dates:
        if d >= cutoff or kept is None or (d - kept).days >= weekly_step:
            out[d] = obs[d]
            kept = d
    out[dates[0]] = obs[dates[0]]
    out[last] = obs[last]
    return out


def recession_bands(usrec_obs):
    """Compress USREC's monthly 0/1 flags into [[start, end], ...] ISO date
    pairs. An episode still open at the last observation ends there."""
    if not usrec_obs:
        return []
    dates = sorted(usrec_obs)
    bands, start = [], None
    for d in dates:
        in_rec = bool(usrec_obs[d])
        if in_rec and start is None:
            start = d
        elif not in_rec and start is not None:
            bands.append([start.isoformat(), d.isoformat()])
            start = None
    if start is not None:
        bands.append([start.isoformat(), dates[-1].isoformat()])
    return bands


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

def _round(v, fmt):
    if v is None:
        return None
    return round(v, 0 if fmt == 'int' else 3)


def _window_label(dates):
    """Actual span of available history, so percentile context never
    overstates its window (ICE BofA OAS is capped at ~3y by FRED)."""
    if not dates:
        return ''
    span_days = (max(dates) - min(dates)).days
    years = span_days / 365.25
    if years >= 1.5:
        return f'{round(years):d}y'
    months = max(1, round(span_days / 30.44))
    return f'{months}mo'


def _nearest_before(obs, target, max_lookback_days=45):
    candidates = [d for d in obs if d <= target
                  and (target - d).days <= max_lookback_days]
    if not candidates:
        return None
    return max(candidates)


def _build_series_entry(meta, raw_obs, as_of):
    """Transform one raw series into its sidecar entry, or None."""
    scaled = {d: v * meta['scale'] for d, v in raw_obs.items()}
    tr = meta['tr']
    if tr == 'yoy':
        series = yoy(scaled)
    elif tr == 'mom_diff':
        series = mom_diff(scaled)
    else:
        series = scaled
    if len(series) < 2:
        return None

    dates = sorted(series)
    latest_d = max(d for d in dates if d <= as_of) if any(
        d <= as_of for d in dates) else dates[-1]
    latest_v = series[latest_d]
    prior_d = max((d for d in dates if d < latest_d), default=None)

    m1_d = _nearest_before(series, latest_d - timedelta(days=30))
    y1_d = _nearest_before(series, latest_d - timedelta(days=365))

    values = [series[d] for d in dates]
    fmt = meta['fmt']
    hist = series
    if meta['freq'] in ('d', 'w'):
        hist = downsample(series)
    hist_dates = sorted(hist)

    entry = {
        'l': meta['l'], 'sec': meta['sec'], 'fmt': fmt,
        'suffix': meta.get('suffix', ''), 'good': meta['good'],
        'freq': meta['freq'],
        'latest': {'d': latest_d.isoformat(), 'v': _round(latest_v, fmt)},
        'prior': ({'d': prior_d.isoformat(),
                   'v': _round(series[prior_d], fmt)} if prior_d else None),
        'chg_1m': (_round(latest_v - series[m1_d], fmt) if m1_d else None),
        'chg_1y': (_round(latest_v - series[y1_d], fmt) if y1_d else None),
        'pctile': (round(percentile_rank(values, latest_v), 3)
                   if percentile_rank(values, latest_v) is not None else None),
        'pct_win': _window_label(dates),
        'z': (round(zscore(values, latest_v), 2)
              if zscore(values, latest_v) is not None else None),
        'hist': {'d': [d.isoformat() for d in hist_dates],
                 'v': [_round(hist[d], fmt) for d in hist_dates]},
    }
    if meta.get('zero_line'):
        entry['zero_line'] = True
    if meta.get('threshold') is not None:
        entry['threshold'] = meta['threshold']
    return entry


CURVE_YEARS = {'1M': 1 / 12, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2,
               '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30}


def _build_curve(fred, as_of):
    from data.fred_client import CMT_SERIES
    tenors = list(CMT_SERIES)
    snaps = {}
    for key, delta in (('now', 0), ('m1', 30), ('y1', 365)):
        target = as_of - timedelta(days=delta)
        curve = fred.fetch_cmt_curve(as_of=target, with_dates=True)
        if not curve:
            continue
        obs_dates = [curve[t][0] for t in tenors if t in curve]
        snaps[key] = {
            'd': max(obs_dates).isoformat() if obs_dates else None,
            'v': [round(curve[t][1], 3) if t in curve else None
                  for t in tenors],
        }
    if 'now' not in snaps:
        return None
    return {'tenors': tenors,
            'yrs': [round(CURVE_YEARS[t], 4) for t in tenors], **snaps}


def build_macro_payload(fred, regime_result=None, macro_adj=None, as_of=None):
    """Fetch, transform and package everything the Macro Outlook tab needs.

    Returns {'summary': small-inline-dict, 'sidecar': macro.json-dict}, or
    None when too little data arrived to be worth a tab (offline, FRED down).
    Individual series failures are skipped — a partial dashboard beats none.
    """
    as_of = as_of or date.today()
    daily_start = as_of - timedelta(days=int(DAILY_HISTORY_YEARS * 365.25))

    series_out = {}
    for meta in MACRO_SERIES:
        try:
            start = daily_start if meta['freq'] in ('d', 'w') else None
            raw = fred.fetch_series(meta['id'], start=start)
            if not raw:
                continue
            entry = _build_series_entry(meta, raw, as_of)
            if entry:
                series_out[meta['id']] = entry
        except Exception as e:
            print(f"  Macro series {meta['id']} skipped ({e}).")

    if len(series_out) < MIN_SERIES_FOR_DASHBOARD:
        return None

    try:
        curve = _build_curve(fred, as_of)
    except Exception:
        curve = None

    try:
        rec = recession_bands(fred.fetch_series('USREC'))
    except Exception:
        rec = []

    oas_buckets = None
    try:
        now_b = fred.fetch_bucket_oas(as_of=as_of)
        m1_b = fred.fetch_bucket_oas(as_of=as_of - timedelta(days=30))
        if now_b:
            oas_buckets = {'now': {k: round(v, 2) for k, v in now_b.items()},
                           'm1': {k: round(v, 2) for k, v in m1_b.items()}}
    except Exception:
        pass

    regime = None
    if regime_result:
        regime = dict(regime_result)
        if macro_adj:
            regime['adjustments'] = macro_adj

    sections = [dict(s, ids=[m['id'] for m in MACRO_SERIES
                             if m['sec'] == s['k'] and m['id'] in series_out])
                for s in SECTIONS]

    sidecar = {
        'as_of': as_of.isoformat(),
        'keyed': bool(getattr(fred, 'available', False)),
        'regime': regime,
        'recessions': rec,
        'sections': sections,
        'series': series_out,
        'curve': curve,
        'oas_buckets': oas_buckets,
    }

    tiles = []
    for sid in OVERVIEW_IDS:
        s = series_out.get(sid)
        if not s:
            continue
        hist_v = s['hist']['v'][-SPARK_POINTS:]
        tiles.append({'id': sid, 'l': s['l'], 'fmt': s['fmt'],
                      'suffix': s['suffix'], 'good': s['good'],
                      'latest': s['latest'], 'chg_1m': s['chg_1m'],
                      'pctile': s['pctile'], 'pct_win': s['pct_win'],
                      'spark': hist_v})
    summary = {'as_of': sidecar['as_of'], 'regime': regime, 'tiles': tiles}

    return {'summary': summary, 'sidecar': sidecar}
