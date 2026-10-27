"""Bonds vs stocks: what the market pays to own stocks over bonds today.

Display only (Financial Model › Bonds vs Stocks). Nothing here feeds a
rating, ``profile_verdict`` or a snapshot row.

A bond's starting yield is the standard estimate of its return to maturity,
so each bond line is simply its yield. High yield is shown after an expected
credit loss (``HY_CREDIT_LOSS``), stated as its own line so the haircut is
visible rather than buried.

The stock side is cap-weighted over the rendered universe, one row per issuer
(the issuer collapse already ran). Aggregates are Σ flow / Σ market cap, never
an average of ratios, and loss-makers stay in: a row with no usable P/E takes
its net income, so a negative year counts against the total instead of being
dropped (dropping it would flatter the yield). A row whose flow exceeds its
market cap is treated as bad data and left out of that aggregate. A
lender's free cash flow is left out too (``is_balance_sheet_financial``):
deposits flow through it, so it says nothing about what an owner earns.

Earnings yield is compared with the 10-year TIPS yield, not the nominal one:
earnings grow with inflation, so E/P is a real-return proxy. When FRED's
``DFII10`` is missing the real yield is the nominal 10Y less the 10Y
breakeven, which is the same quantity by definition. The ICE effective
yields (``BAMLC0A0CMEY`` / ``BAMLH0A0HYM2EY``) fall back to a Treasury of
roughly matching duration plus the option-adjusted spread, flagged
``derived``.

All yields are fractions (0.05 = 5%); the macro sidecar's percent values are
converted on the way in.
"""

import math

from models.etf_lookthrough import FVP_CLIP
from models.sector_pool import is_balance_sheet_financial

BOND_METHOD = 1
HY_CREDIT_LOSS = 0.015       # 3% default rate x 50% loss given default, a long-run norm
SECTOR_MIN_COMPANIES = 5
SECTOR_MIN_COVERAGE = 0.80   # share of sector cap with a usable earnings figure
# Treasury tenor a corporate index's OAS is added to when its effective yield
# is missing (index durations: IG ~7y, HY ~4y).
_OAS_TENOR = {'ig': '7Y', 'hy': '5Y'}


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if v != v or v in (float('inf'), float('-inf')):
        return None
    return float(v)


def _pct(v):
    v = _num(v)
    return None if v is None else v / 100.0


def _series(sidecar, sid):
    s = ((sidecar or {}).get('series') or {}).get(sid)
    if not s or not s.get('latest'):
        return None
    return s


def _curve_at(curve, tenor, key='now'):
    snap = (curve or {}).get(key) or {}
    try:
        i = curve['tenors'].index(tenor)
    except (KeyError, ValueError):
        return None
    vals = snap.get('v') or []
    return _pct(vals[i]) if i < len(vals) else None


def _line(id_, label, kind, y, *, d=None, chg_1y=None, pctile=None, pct_win=None,
          derived=None, note=None):
    out = {'id': id_, 'l': label, 'kind': kind, 'y': y, 'd': d, 'chg_1y': chg_1y}
    if pctile is not None:
        out['pctile'] = pctile
        out['pct_win'] = pct_win
    if derived:
        out['derived'] = derived
    if note:
        out['note'] = note
    return out


def _from_series(s):
    """``(yield, date, 1y change, pctile, window)`` from a sidecar entry."""
    return (_pct(s['latest'].get('v')), s['latest'].get('d'), _pct(s.get('chg_1y')),
            s.get('pctile'), s.get('pct_win'))


def bond_lines(sidecar):
    """The bond sleeve, shortest first. A line whose inputs are missing is
    left out; the rest still render."""
    curve = (sidecar or {}).get('curve')
    if not curve or not curve.get('now'):
        return []
    as_of = curve['now'].get('d')
    out = []
    for tenor, label in (('3M', '3-month T-bill'), ('2Y', '2-year Treasury'),
                         ('10Y', '10-year Treasury'), ('30Y', '30-year Treasury')):
        y = _curve_at(curve, tenor)
        if y is None:
            continue
        y1 = _curve_at(curve, tenor, 'y1')
        kw = {}
        if tenor == '10Y' and _series(sidecar, 'DGS10'):
            s = _series(sidecar, 'DGS10')
            kw = {'pctile': s.get('pctile'), 'pct_win': s.get('pct_win')}
        out.append(_line('ust' + tenor, label, 'treasury', y, d=as_of,
                         chg_1y=(y - y1) if y1 is not None else None, **kw))

    real = real_10y(sidecar)
    if real is not None:
        out.append(real)

    for key, ey_id, oas_id, label in (('ig', 'BAMLC0A0CMEY', 'BAMLC0A0CM', 'Investment-grade corporate'),
                                      ('hy', 'BAMLH0A0HYM2EY', 'BAMLH0A0HYM2', 'High-yield corporate')):
        s = _series(sidecar, ey_id)
        if s:
            y, d, chg, pc, win = _from_series(s)
            line = _line(key, label, 'corp', y, d=d, chg_1y=chg, pctile=pc, pct_win=win)
        else:
            oas, base = _series(sidecar, oas_id), _curve_at(curve, _OAS_TENOR[key])
            if not oas or base is None:
                continue
            o, d, chg, pc, win = _from_series(oas)
            if o is None:
                continue
            line = _line(key, label, 'corp', base + o, d=d,
                         derived=f"{_OAS_TENOR[key]} Treasury + option-adjusted spread")
        if line['y'] is None:
            continue
        if key == 'hy':
            line['gross'] = line['y']
            line['loss'] = HY_CREDIT_LOSS
            line['y'] = line['y'] - HY_CREDIT_LOSS
            line['note'] = f"after a {HY_CREDIT_LOSS:.1%} expected credit loss"
        out.append(line)
    return out


def real_10y(sidecar):
    """The 10-year TIPS yield line: ``DFII10``, else 10Y less the breakeven."""
    s = _series(sidecar, 'DFII10')
    if s:
        y, d, chg, pc, win = _from_series(s)
        if y is not None:
            return _line('tips10', '10-year TIPS (real)', 'tips', y, d=d, chg_1y=chg,
                         pctile=pc, pct_win=win)
    nom, be = _series(sidecar, 'DGS10'), _series(sidecar, 'T10YIE')
    if not nom or not be:
        return None
    n, d, _, _, _ = _from_series(nom)
    b = _pct(be['latest'].get('v'))
    if n is None or b is None:
        return None
    return _line('tips10', '10-year TIPS (real)', 'tips', n - b, d=d,
                 derived='10Y Treasury − 10Y breakeven inflation')


def _earnings(r, mcap):
    pe = _num(r.get('pe'))
    if pe:
        return mcap / pe
    return _num(r.get('net_income'))


def _flows(r):
    """``(mcap, earnings, fcf, shareholder payout)`` in dollars; each flow is
    None when missing or implausible (larger than the company)."""
    mcap = _num(r.get('mcap'))
    if not mcap or mcap <= 0:
        return None
    e = _earnings(r, mcap)
    fy, sy = _num(r.get('fcf_yield')), _num(r.get('shareholder_yield'))
    if is_balance_sheet_financial(r):
        fy = None

    def _ok(x):
        return x if x is not None and abs(x) <= mcap else None

    return (mcap, _ok(e), _ok(fy * mcap if fy is not None else None),
            _ok(sy * mcap if sy is not None else None))


def stock_aggregate(rows):
    """Cap-weighted yields over *rows*: ``{n, mcap, earnings_yield, fcf_yield,
    shareholder_yield, coverage}``, each yield Σ flow / Σ mcap over the rows
    that have that flow, and ``coverage`` the cap share with earnings."""
    tot = {'cap': 0.0, 'e': [0.0, 0.0], 'f': [0.0, 0.0], 's': [0.0, 0.0]}
    n = 0
    for r in rows or []:
        fl = _flows(r)
        if fl is None:
            continue
        n += 1
        mcap = fl[0]
        tot['cap'] += mcap
        for k, v in zip('efs', fl[1:], strict=True):
            if v is not None:
                tot[k][0] += v
                tot[k][1] += mcap

    def _y(k):
        return tot[k][0] / tot[k][1] if tot[k][1] else None

    return {'n': n, 'mcap': tot['cap'], 'earnings_yield': _y('e'), 'fcf_yield': _y('f'),
            'shareholder_yield': _y('s'),
            'coverage': tot['e'][1] / tot['cap'] if tot['cap'] else 0.0}


def model_read(rows, etf_payload=None):
    """The model's own read of the market: SPY's look-through MoS when the
    ETF payload has it, else the universe cap-weighted the same way (FV/P
    clipped to ``FVP_CLIP``, no fair value at FV/P 1)."""
    spy = ((etf_payload or {}).get('funds') or {}).get('SPY') or {}
    if spy.get('status') == 'ok' and _num(spy.get('mos')) is not None:
        return {'source': 'SPY', 'mos': spy['mos']}
    lo, hi = FVP_CLIP
    v = w = 0.0
    for r in rows or []:
        mcap, fv, p = _num(r.get('mcap')), _num(r.get('_fv_effective')), _num(r.get('price'))
        if not mcap or mcap <= 0:
            continue
        x = min(max(fv / p, lo), hi) if fv and p and fv > 0 and p > 0 else 1.0
        v += mcap * x
        w += mcap
    if not w:
        return None
    return {'source': 'universe', 'mos': 1.0 - w / v}


def _premia(agg, real, ig):
    ey, fy = agg.get('earnings_yield'), agg.get('fcf_yield')
    return {'vs_real': ey - real if ey is not None and real is not None else None,
            'fcf_vs_ig': fy - ig if fy is not None and ig is not None else None}


def bonds_vs_stocks(rows, macro_sidecar, etf_payload=None):
    """The tab's payload, or None without a yield curve."""
    bonds = bond_lines(macro_sidecar)
    if not bonds:
        return None
    by_id = {b['id']: b for b in bonds}
    real = (by_id.get('tips10') or {}).get('y')
    ig = (by_id.get('ig') or {}).get('y')

    stocks = stock_aggregate(rows)
    out = {'method': BOND_METHOD, 'as_of': macro_sidecar['curve']['now'].get('d'),
           'bonds': bonds, 'stocks': stocks, 'model': model_read(rows, etf_payload),
           'premia': _premia(stocks, real, ig), 'hy_credit_loss': HY_CREDIT_LOSS}

    by_sector = {}
    for r in rows or []:
        s = r.get('sector')
        if s:
            by_sector.setdefault(s, []).append(r)
    sectors = []
    for s, rs in by_sector.items():
        agg = stock_aggregate(rs)
        e = {'sector': s, **agg, **_premia(agg, real, ig)}
        if agg['n'] < SECTOR_MIN_COMPANIES:
            e['status'], e['reason'] = 'insufficient', f"{agg['n']} companies"
        elif agg['coverage'] < SECTOR_MIN_COVERAGE:
            e['status'] = 'insufficient'
            e['reason'] = f"{agg['coverage']:.0%} of value has earnings (needs {SECTOR_MIN_COVERAGE:.0%})"
        else:
            e['status'] = 'ok'
        sectors.append(e)
    sectors.sort(key=lambda e: (e['status'] != 'ok',
                                -(e['vs_real'] if e['vs_real'] is not None else -math.inf)))
    out['sectors'] = sectors
    return out
