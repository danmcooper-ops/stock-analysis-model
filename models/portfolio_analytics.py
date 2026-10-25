"""Portfolio-level valuation, risk and attribution.

Three reads of a portfolio grouping, each weighted the way its NAV is
(``data.portfolio_nav.entry_weights``: equal, model or target; a ``held``
portfolio is equal-weighted here, because these numbers are published):

* **Valuation** — the portfolio read through its members' fair values, the
  same look-through the SPDRs get (``models.etf_lookthrough``):
  ``V = Σ w·clip(FV/P, 1/3, 3)`` per $1, MoS ``1 − 1/V``. Like the funds',
  the absolute read is mostly the model's market-wide bias, so the headline
  is ``V / V_universe − 1`` against the equal-weighted universe.
* **Risk** — today's weights over the price parquets (adjusted closes).
  Volatility, Sharpe (over the run's risk-free rate), return and max
  drawdown come from the last year of daily returns. Beta, tracking error
  and correlations come from two years of weekly returns, as the model's
  own beta does: one-year daily betas are noisy (KO read −0.28 against a
  weekly 0.07 on 2026-10-23, AMD 3.26 against 1.87). Plus the basket's
  drawdown through 2008, 2020 and 2022. Ex ante: what today's portfolio
  would have done, not what was held then (that is the NAV's job).
* **Attribution** — each member's contribution to the NAV over a window,
  from the ledger's own steps (``step_contributions``), so it covers what
  was actually held each day. Daily contributions are Carino-linked, so
  they add up to the window's compounded return rather than the plain sum
  of its daily returns (which runs high by the volatility drag: +12.5%
  against +9.3% for a semiconductor basket over 2026's summer).

Pure functions apart from ``load_closes`` (parquet reads).
"""
from __future__ import annotations

import logging
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

from models import etf_lookthrough as lt
from models import portfolio_groups as pg

logger = logging.getLogger(__name__)

RISK_LOOKBACK_BARS = 252        # one year of daily returns
RISK_MIN_BARS = 60
CO_LOOKBACK_WEEKS = 104         # two years of weekly returns: beta, TE, correlation
CO_MIN_WEEKS = 52
TRADING_DAYS = 252
CORR_HIGH = 0.85                # a pair this correlated is close to one bet
TOP_PAIRS = 3
TOP_CONTRIB = 5
MIN_RISK_COVERAGE = 0.80        # weight with a usable return history
CRISES = {'2008': ('2008-01-01', '2009-03-31'),
          '2020': ('2020-01-15', '2020-04-30'),
          '2022': ('2022-01-01', '2022-10-31')}
MARKET = 'SPY'
CASH = '$CASH'


def _f(v, nd=4):
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

def valuation(weights, by_tk, aliases=None):
    """Look-through value of *weights* ({ticker: w}, ``CASH`` valued at 1).

    Returns the ``etf_lookthrough.lookthrough`` result (``status``, ``V``,
    ``mos``, ``valued_w``, ``lifts``/``drags`` …).
    """
    holdings = [{'ticker': t, 'weight': w, 'kind': 'cash' if t == CASH else 'equity'}
                for t, w in weights.items() if w > 0]
    return lt.lookthrough({'fund': None, 'holdings': holdings}, by_tk, aliases or {})


def relative_value(v, base):
    """``V / V_base − 1`` when both reads are usable, else None."""
    if (v or {}).get('status') != 'ok' or (base or {}).get('status') != 'ok':
        return None
    return v['V'] / base['V'] - 1.0


def valuation_summary(v, base):
    """The compact block the report and CLI show."""
    if not v:
        return None
    out = {'status': v.get('status'), 'cov': _f(v.get('valued_w'), 3)}
    if v.get('status') == 'ok':
        out.update(V=_f(v['V']), mos=_f(v['mos']), rel=_f(relative_value(v, base)),
                   lifts=[{'t': e['t'], 'c': _f(e['c'])} for e in v.get('lifts') or ()],
                   drags=[{'t': e['t'], 'c': _f(e['c'])} for e in v.get('drags') or ()])
    else:
        out['reason'] = v.get('reason')
    return out


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

def load_closes(prices_dir, tickers):
    """{ticker: adjusted close Series} for the tickers with a parquet."""
    out = {}
    for t in sorted(set(tickers)):
        path = os.path.join(prices_dir, f'{t}.parquet')
        if not os.path.exists(path):
            continue
        try:
            s = pd.read_parquet(path, columns=['Close'])['Close']
        except Exception as e:  # one bad file must not cost every portfolio its risk
            logger.warning("portfolio risk: %s unreadable (%s)", path, e)
            continue
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        s = s[np.isfinite(s) & (s > 0)]
        if len(s):
            out[t] = s[~s.index.duplicated(keep='last')]
    return out


def _max_drawdown(nav):
    if nav is None or len(nav) < 2:
        return None
    return float((nav / nav.cummax() - 1.0).min())


def _basket(rets, weights):
    """Daily basket return: each day's priced members, weights renormalized
    over them (cash at 0 keeps its share)."""
    w = pd.Series({t: weights[t] for t in rets.columns})
    cash = weights.get(CASH, 0.0)
    avail = rets.notna()
    wsum = avail.mul(w, axis=1).sum(axis=1)
    risky = sum(weights.values()) - cash
    num = rets.fillna(0.0).mul(w, axis=1).sum(axis=1)
    out = num / wsum.where(wsum > 0) * risky / (risky + cash)
    return out.dropna()


def _crisis_drawdowns(closes, weights):
    out = {}
    for name, (a, b) in CRISES.items():
        rets, cov = {}, 0.0
        for t, w in weights.items():
            s = closes.get(t)
            if t == CASH or s is None:
                continue
            win = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
            # Only a member that traded through the whole window counts.
            if len(win) < 20 or win.index[0] > pd.Timestamp(a) + pd.Timedelta(days=10):
                continue
            rets[t] = win.pct_change()
            cov += w
        total = sum(v for k, v in weights.items() if k != CASH)
        if not rets or total <= 0 or cov / total < 0.5:
            out[name] = None
            continue
        r = _basket(pd.DataFrame(rets).iloc[1:], {t: weights[t] for t in rets})
        out[name] = {'dd': _f(_max_drawdown((1.0 + r).cumprod())), 'cov': _f(cov / total, 3)}
    return out


def risk(weights, closes, rf=None, asof=None):
    """Ex-ante risk of *weights* over the trailing year of *closes*.

    *closes* is ``load_closes`` output and must hold ``SPY``; *rf* the
    annual risk-free rate; *asof* (a date string) caps the window so an old
    snapshot is judged on its own history. Returns None when under
    ``MIN_RISK_COVERAGE`` of the risky weight has a usable history.
    """
    spy = closes.get(MARKET)
    if spy is None:
        return None
    if asof:
        cut = pd.Timestamp(asof)
        closes = {t: s[s.index <= cut] for t, s in closes.items()}
        spy = closes[MARKET]
    spy_r = spy.pct_change().iloc[-RISK_LOOKBACK_BARS:].dropna()
    if len(spy_r) < RISK_MIN_BARS:
        return None
    idx = spy_r.index
    members = [t for t in weights if t != CASH]
    risky = sum(weights[t] for t in members)
    cols, have = {}, 0.0
    for t in members:
        s = closes.get(t)
        if s is None:
            continue
        r = s.pct_change().reindex(idx)
        if r.notna().sum() >= RISK_MIN_BARS:
            cols[t] = r
            have += weights[t]
    if risky <= 0 or have / risky < MIN_RISK_COVERAGE:
        return {'status': 'insufficient', 'cov': _f(have / risky if risky else 0.0, 3)}
    rets = pd.DataFrame(cols)
    wts = {**{t: weights[t] for t in cols}, CASH: weights.get(CASH, 0.0)}
    port = _basket(rets, wts)
    vol = float(port.std() * math.sqrt(TRADING_DAYS))
    ann = float((1.0 + port).prod() ** (TRADING_DAYS / len(port)) - 1.0)

    # Co-movement on weekly returns (Friday closes, as of the last bar).
    def _weekly(s):
        return s.resample('W-FRI').last().pct_change()
    spy_w = _weekly(spy).iloc[-CO_LOOKBACK_WEEKS:].dropna()
    wk = pd.DataFrame({t: _weekly(closes[t]).reindex(spy_w.index) for t in cols})
    wk = wk.loc[:, wk.notna().sum() >= CO_MIN_WEEKS]
    beta = te = None
    corr = pd.DataFrame()
    if len(spy_w) >= CO_MIN_WEEKS and not wk.empty:
        pw, sw = _basket(wk, {**{t: wts[t] for t in wk.columns}, CASH: wts[CASH]}).align(
            spy_w, join='inner')
        if len(pw) >= CO_MIN_WEEKS and sw.var() > 0:
            beta = float(np.cov(pw, sw)[0, 1]) / float(sw.var())
            te = float((pw - sw).std() * math.sqrt(52))
        corr = wk.corr(min_periods=CO_MIN_WEEKS)
    pairs = []
    ts = list(corr.columns)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            c = corr.iat[i, j]
            if c == c:
                pairs.append((float(c), ts[i], ts[j]))
    pairs.sort(reverse=True)
    return {
        'status': 'ok', 'cov': _f(have / risky, 3), 'bars': len(port),
        'from': port.index[0].date().isoformat(), 'to': port.index[-1].date().isoformat(),
        'vol': _f(vol), 'ret': _f(ann), 'beta': _f(beta, 3), 'te': _f(te),
        'sharpe': _f((ann - rf) / vol if rf is not None and vol > 0 else None, 3),
        'rf': _f(rf), 'mdd': _f(_max_drawdown((1.0 + port).cumprod())),
        'avg_corr': _f(sum(c for c, _, _ in pairs) / len(pairs) if pairs else None, 3),
        'pairs': [{'a': a, 'b': b, 'r': _f(c, 3)} for c, a, b in pairs[:TOP_PAIRS]],
        'n_high': sum(1 for c, _, _ in pairs if c >= CORR_HIGH),
        'crises': _crisis_drawdowns(closes, {t: weights[t] for t in members}),
    }


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

WINDOWS = (('1M', 30), ('3M', 91), ('YTD', 'ytd'), ('All', None))


def _window_start(end_d, spec):
    if spec is None:
        return None
    if spec == 'ytd':
        return f'{end_d[:4]}-01-01'
    return (date.fromisoformat(end_d) - timedelta(days=spec)).isoformat()


def _carino(r):
    """Carino's linking coefficient ln(1+r)/r (1 at r = 0)."""
    return math.log1p(r) / r if abs(r) > 1e-12 else 1.0


def attribution(series, closes, top=TOP_CONTRIB):
    """Per-member contributions over each window of a ledger series.

    *series* is one portfolio's ledger entries; *closes* is
    ``{market date: {ticker: (bar, close)}}`` (``asof_closes`` over the
    entries' market dates and members). Each step is credited to what the
    previous entry held, with its recorded weights, exactly as the NAV
    step was, and linked with Carino's coefficients so a window's
    contributions sum to its compounded return ``ret``. Returns
    ``{window: {'ret', 'n', 'top', 'bottom'}}`` for the windows the series
    reaches.
    """
    from data.portfolio_nav import step_contributions
    if len(series) < 2:
        return {}
    end_d = series[-1]['d']
    steps = []
    for pe, e in zip(series, series[1:], strict=False):
        held = pe.get('members') or []
        c, ret, _ = step_contributions(held, pe['m'], e['m'], closes, {}, {}, pe.get('w'))
        steps.append((pe['d'], e['d'], c, ret))
    out = {}
    for name, spec in WINDOWS:
        # Same window as portfolio_nav.window_return: from the last point on
        # or before the start date; a series that doesn't reach back that
        # far has no such window.
        start = _window_start(end_d, spec)
        base = [e for e in series if start is None or e['d'] <= start]
        if not base:
            continue
        b = series[0] if start is None else base[-1]
        if not b['nav']:
            continue
        R = series[-1]['nav'] / b['nav'] - 1.0
        if R <= -1.0:
            continue
        K = _carino(R)
        by = {}
        for d0, _, c, ret in steps:
            if d0 < b['d'] or not ret:
                continue
            k = _carino(ret) / K
            for t, v in c.items():
                by[t] = by.get(t, 0.0) + v * k
        ranked = sorted(by.items(), key=lambda kv: -kv[1])
        out[name] = {
            'ret': _f(R),
            'n': len(by),
            'top': [{'t': t, 'c': _f(v)} for t, v in ranked[:top] if v > 0],
            'bottom': [{'t': t, 'c': _f(v)} for t, v in reversed(ranked[-top:]) if v < 0],
        }
    return out


def attribution_closes(series_list, closes_fn):
    """One price query covering every step of every series."""
    tickers, dates = set(), set()
    for s in series_list:
        for e in s:
            dates.add(e['m'])
            tickers.update(e.get('members') or ())
            tickers.update(t for t in e.get('w') or () if t != CASH)
    if not dates or not tickers:
        return {}
    return closes_fn(sorted(tickers), sorted(dates)) or {}


# ---------------------------------------------------------------------------
# Everything for one set of portfolios
# ---------------------------------------------------------------------------

def analyze(portfolios, rows, ledger=None, prices_dir=None, closes_fn=None, asof=None):
    """``{portfolio id: {'w': weighting, 'val': …, 'risk': …, 'attrib': …}}``
    plus ``'_universe'`` (its valuation) — each part None when it cannot be
    computed. *ledger* (the public NAV ledger) feeds attribution; *prices_dir*
    the risk; *closes_fn* (``portfolio_nav.prices_closes_fn``) the attribution
    prices. A ``held`` portfolio is weighted equally here (these are
    published)."""
    from data.portfolio_nav import entry_weights
    by_tk = pg.rows_by_ticker(rows)
    aliases = pg.listing_aliases(by_tk)
    uni = valuation({t: 1.0 for t in by_tk}, by_tk, aliases)
    rfs = sorted(v for v in (pg.js_num(r.get('_risk_free_rate')) for r in rows) if v is not None)
    rf = rfs[len(rfs) // 2] if rfs else None

    weights = {}
    for p in portfolios:
        members = pg.resolve_members(p, by_tk)['members']
        w = None
        if p.get('weighting') in ('model', 'target'):
            w = entry_weights(p, members, by_tk)
        weights[p['id']] = w or ({t: 1.0 / len(members) for t in members} if members else {})

    closes = {}
    if prices_dir and os.path.isdir(prices_dir):
        need = {t for w in weights.values() for t in w if t != CASH} | {MARKET}
        closes = load_closes(prices_dir, need)
    att_closes = {}
    if ledger and closes_fn:
        att_closes = attribution_closes(
            [ledger['portfolios'].get(p['id']) or [] for p in portfolios], closes_fn)

    out = {'_universe': valuation_summary(uni, uni)}
    for p in portfolios:
        w = weights[p['id']]
        res = {'w': p.get('weighting') or 'equal', 'val': None, 'risk': None, 'attrib': None}
        if w:
            res['val'] = valuation_summary(valuation(w, by_tk, aliases), uni)
            if closes:
                try:
                    res['risk'] = risk(w, closes, rf, asof)
                except Exception as e:
                    logger.warning("portfolio risk for %s failed (%s)", p['id'], e)
        if ledger:
            res['attrib'] = attribution(ledger['portfolios'].get(p['id']) or [], att_closes) or None
        out[p['id']] = res
    return out
