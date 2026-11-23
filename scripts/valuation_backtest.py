"""Historical valuation backtest: did the DCF's margin of safety, rebuilt
point in time from EDGAR history, pick stocks that later beat SPY?

The live backtest (backtest.py measure) has 3-6 independent periods because
snapshots start in 2026. Here each company's fiscal year t is valued as of
its filing date (fiscal year end + 90 days, the report's own `fya` rule)
with only what was known then:

  base FCF        owner FCF (OCF − |capex| − SBC, floored as live), capped at
                  1.25x the trailing-5y average positive FCF, as live
  growth          'v1h' = v1's history signals at live weights (FCF CAGR,
                  revenue CAGR, reinvestment × ROIC); 'v2h' = growth model v2's
                  long-run rate (no FCF CAGR, shrunk toward sector base rates
                  measured on outcomes RESOLVED by t). Neither has analyst
                  forecasts (none are stored point in time); v2h's path is
                  flat without them
  discount rate   CAPM on the 10-year Treasury that day (FRED DGS10), a beta
                  from two years of weekly returns against SPY shrunk toward 1,
                  the live ERP; WACC with debt at rf + 2% after 21% tax,
                  clamped to the sector's WACC floor and cap
  price           the close that day with dividends un-adjusted (the parquets
                  are total-return adjusted), times split-restated shares

Outcome: total return over 1 and 3 years in excess of SPY's. Per filing
year (the independent unit) it reports the Spearman IC of MoS against the
outcome, the top-minus-bottom quintile spread, and a bootstrap over years.

Limits: companies listed today only (survivorship: failures are missing,
which flatters every quintile but matters less for a cross-sectional
ranking); no analysts; a simpler WACC than live; non-USD reporters (ADR
share ratios) and Financial Services (no FCF DCF) are left out.

Usage:
  python scripts/valuation_backtest.py                          # newest snapshot
  python scripts/valuation_backtest.py --snapshot output/results_2026-11-20.json
"""
import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import growth_v2  # noqa: E402
from models.dcf import two_stage_ev  # noqa: E402
from scripts import growth_hindcast as gh  # noqa: E402
from scripts.config import (  # noqa: E402
    ERP,
    GROWTH_WEIGHT_FCF,
    GROWTH_WEIGHT_FUNDAMENTAL,
    GROWTH_WEIGHT_REV,
    YIELD_CEILING_MULT,
    _get_sector_config,
)

FILING_LAG_DAYS = 90
HORIZONS_Y = (1, 3)
BETA_WEEKS = 104
DEBT_SPREAD = 0.02
TAX = 0.21
MIN_PER_YEAR = 50
V1H_WEIGHTS = {'fcf_cagr': GROWTH_WEIGHT_FCF, 'rev_cagr': GROWTH_WEIGHT_REV,
               'fundamental': GROWTH_WEIGHT_FUNDAMENTAL}
METHODS = ('v1h', 'v2h')


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def raw_closes(df):
    """Closes with the dividend adjustment undone (splits stay adjusted).

    yfinance's auto-adjusted close scales every bar before an ex-date by
    (1 − D / prior close); walking back from the newest bar, the factor
    before an ex-date is F_after / (1 + F_after · D / adjusted prior close).
    """
    close = df['Close'].astype(float)
    div = df['Dividends'].astype(float) if 'Dividends' in df else pd.Series(0.0, index=df.index)
    f = np.ones(len(close))
    fac = 1.0
    c, d = close.values, div.values
    for i in range(len(c) - 1, -1, -1):
        f[i] = fac
        if d[i] > 0 and i > 0 and c[i - 1] > 0:
            fac = fac / (1 + fac * d[i] / c[i - 1])
    return pd.Series(c / f, index=close.index)


class Prices:
    """As-of closes from output/prices/<T>.parquet (cached per ticker)."""

    def __init__(self, prices_dir):
        self.dir = prices_dir
        self._c = {}

    def get(self, ticker):
        if ticker not in self._c:
            p = os.path.join(self.dir, f'{ticker}.parquet')
            df = None
            if os.path.exists(p):
                try:
                    df = pd.read_parquet(p)
                    df = df[np.isfinite(df['Close']) & (df['Close'] > 0)]
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df = df.assign(Raw=raw_closes(df)) if len(df) else None
                except Exception as e:  # a corrupt parquet costs one ticker
                    print(f"  prices: {ticker} unreadable ({e})", file=sys.stderr)
                    df = None
            self._c[ticker] = df
        return self._c[ticker]

    @staticmethod
    def asof(df, date, col='Close', max_gap=7):
        if df is None:
            return None
        i = df.index.searchsorted(pd.Timestamp(date), side='right') - 1
        if i < 0 or (pd.Timestamp(date) - df.index[i]).days > max_gap:
            return None
        return float(df[col].iloc[i])


def beta(stock, spy, date, weeks=BETA_WEEKS):
    """Weekly-return beta over `weeks` to `date`, shrunk 2/3 + 1/3 toward 1."""
    if stock is None or spy is None:
        return None
    start = pd.Timestamp(date) - timedelta(weeks=weeks)
    s = stock['Close'][(stock.index > start) & (stock.index <= pd.Timestamp(date))].resample('W-FRI').last()
    m = spy['Close'][(spy.index > start) & (spy.index <= pd.Timestamp(date))].resample('W-FRI').last()
    r = pd.concat([s.pct_change(), m.pct_change()], axis=1, sort=True).dropna()
    if len(r) < weeks * 0.6 or r.iloc[:, 1].var() <= 0:
        return None
    b = float(r.cov().iloc[0, 1] / r.iloc[:, 1].var())
    return 2 / 3 * b + 1 / 3


# ---------------------------------------------------------------------------
# Point-in-time valuation
# ---------------------------------------------------------------------------

def _shares_at(eh, date):
    """Newest split-restated share count dated on or before `date`."""
    best = None
    for k, v in (eh.get('shares_history') or {}).items():
        try:
            d = pd.Timestamp(str(k)[:10]) if len(str(k)) >= 10 else pd.Timestamp(f'{int(k)}-12-31')
        except (ValueError, TypeError):
            continue
        if d <= pd.Timestamp(date) and isinstance(v, (int, float)) and v > 0 and (best is None or d > best[0]):
            best = (d, float(v))
    return best[1] if best else None


def fair_value(eh, t, sector, growth, rate, tg):
    """Equity value per share at year t for a given growth and discount rate."""
    fcf = gh.owner_fcf(eh)
    base = fcf.get(t)
    if base is None or base <= 0:
        return None
    pos = [fcf[y] for y in range(t - 4, t + 1) if fcf.get(y, 0) > 0]
    base = min(base, YIELD_CEILING_MULT * float(np.mean(pos))) if pos else base
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ev = two_stage_ev(base, growth, rate, tg)
    if not ev:
        return None
    a = {k: gh.annual(eh.get(k)) for k in ('total_debt_history', 'cash_history',
                                           'st_investments_history')}
    net_debt = (a['total_debt_history'].get(t, 0.0) - a['cash_history'].get(t, 0.0)
                - a['st_investments_history'].get(t, 0.0))
    return ev - net_debt


def availability_date(eh, t):
    fy = (eh.get('fy_ends') or {}).get(str(t))
    try:
        end = pd.Timestamp(fy) if fy else pd.Timestamp(f'{t}-12-31')
    except (ValueError, TypeError):
        end = pd.Timestamp(f'{t}-12-31')
    return end + timedelta(days=FILING_LAG_DAYS)


def build_observations(rows, prices, rf_series, base_rates_by_year):
    """One observation per (ticker, fiscal year) with both MoS and outcomes."""
    spy = prices.get('SPY')
    obs = []
    for r in rows:
        eh, tk, sector = r.get('edgar_history'), r.get('ticker'), r.get('sector') or ''
        if (not isinstance(eh, dict) or sector == 'Financial Services'
                or (eh.get('reporting_currency') or 'USD') != 'USD'):
            continue
        px = prices.get(tk)
        if px is None:
            continue
        cfg = _get_sector_config(sector)
        fcf = gh.owner_fcf(eh)
        for t in sorted(fcf):
            if t - gh.CAGR_SPAN not in fcf:
                continue
            a = availability_date(eh, t)
            p_raw, shares = Prices.asof(px, a, 'Raw'), _shares_at(eh, a)
            p0, s0 = Prices.asof(px, a), Prices.asof(spy, a)
            rf = rf_series.asof(a) if len(rf_series) else None
            if not (p_raw and shares and p0 and s0) or rf is None or not math.isfinite(rf):
                continue
            b = beta(px, spy, a)
            if b is None:
                continue
            mcap = p_raw * shares
            debt = gh.annual(eh.get('total_debt_history')).get(t, 0.0) or 0.0
            re = rf + b * ERP
            wacc = (mcap * re + debt * (rf + DEBT_SPREAD) * (1 - TAX)) / (mcap + debt)
            wacc = min(max(wacc, cfg['wacc_floor']), cfg['wacc_cap'])
            tg = cfg['terminal_growth']
            sig = gh.history_signals(eh, t)
            g1 = growth_v2.history_blend(sig, V1H_WEIGHTS)
            if g1 is None:
                continue
            g1 = growth_v2.clamp(g1, cfg['growth_cap'])
            rates = base_rates_by_year.get(t)
            g2 = (growth_v2.long_run_rate(sig, sector, cfg['growth_cap'], {
                'history_weights': gh.V2_HISTORY_WEIGHTS, 'base_rates': rates,
                'shrink_lambda': gh.V2_SHRINK_LAMBDA})[0] if rates else None)
            rec = {'ticker': tk, 'sector': sector, 't': t, 'year': a.year, 'date': a}
            for name, g in (('v1h', g1), ('v2h', g2)):
                eq = fair_value(eh, t, sector, g, wacc, tg) if g is not None else None
                rec[name] = (eq / shares / p_raw - 1) if eq is not None else None
            for h in HORIZONS_Y:
                d1 = a + timedelta(days=365 * h)
                p1, s1 = Prices.asof(px, d1), Prices.asof(spy, d1)
                rec[f'x{h}'] = (p1 / p0 - s1 / s0) if (p1 and s1) else None
            obs.append(rec)
    return obs


def base_rates_point_in_time(rows):
    """Sector base rates per fiscal year t from outcomes resolved by t."""
    cases = gh.build_cases(rows, 'fcf')
    out = {}
    for t in sorted({c['origin'] for c in cases} | {max(c['origin'] for c in cases) + k
                                                      for k in range(1, 6)}):
        train = [c for c in cases if c['origin'] <= t - gh.HORIZON]
        if len(train) >= 300:
            out[t] = gh._base_rates(train)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _winsor(x, q=0.01):
    lo, hi = np.nanquantile(x, [q, 1 - q])
    return np.clip(x, lo, hi)


def evaluate(obs, n_boot=2000, seed=0):
    """Per method and horizon: per-year IC and quintile spread, then their
    mean over years with a bootstrap CI over years (the independent unit)."""
    rng = np.random.default_rng(seed)
    out = {}
    for m in METHODS:
        for h in HORIZONS_Y:
            per = []
            by = defaultdict(list)
            for o in obs:
                if o.get(m) is not None and o.get(f'x{h}') is not None and math.isfinite(o[m]):
                    by[o['year']].append((o[m], o[f'x{h}']))
            for y in sorted(by):
                v = np.array(by[y])
                if len(v) < MIN_PER_YEAR:
                    continue
                mos, x = v[:, 0], _winsor(v[:, 1])
                q = pd.qcut(pd.Series(mos).rank(method='first'), 5, labels=False).values
                per.append({'year': int(y), 'n': len(v),
                            'ic': float(spearmanr(mos, x).statistic),
                            'q5_q1': float(x[q == 4].mean() - x[q == 0].mean())})
            if not per:
                continue
            ics = np.array([p['ic'] for p in per])
            spr = np.array([p['q5_q1'] for p in per])
            boot = [ics[rng.integers(0, len(ics), len(ics))].mean() for _ in range(n_boot)]
            out[f'{m}_{h}y'] = {
                'years': len(per), 'n_obs': int(sum(p['n'] for p in per)),
                'mean_ic': float(ics.mean()), 'ic_ci90': [float(np.quantile(boot, 0.05)),
                                                          float(np.quantile(boot, 0.95))],
                'ic_pos_share': float(np.mean(ics > 0)),
                'mean_q5_q1': float(spr.mean()), 'per_year': per,
                'overlapping': h > 1,
            }
    return out


def paired(obs, a='v2h', b='v1h', n_boot=2000, seed=0):
    """IC of `a` minus IC of `b` per filing year on the SAME company-years
    (both MoS and the outcome present), averaged with a bootstrap over years."""
    rng = np.random.default_rng(seed)
    out = {}
    for h in HORIZONS_Y:
        by = defaultdict(list)
        for o in obs:
            if all(o.get(k) is not None and math.isfinite(o[k]) for k in (a, b, f'x{h}')):
                by[o['year']].append((o[a], o[b], o[f'x{h}']))
        diffs = []
        for y in sorted(by):
            v = np.array(by[y])
            if len(v) >= MIN_PER_YEAR:
                x = _winsor(v[:, 2])
                diffs.append(float(spearmanr(v[:, 0], x).statistic - spearmanr(v[:, 1], x).statistic))
        if diffs:
            d = np.array(diffs)
            boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
            out[f'{h}y'] = {'years': len(d), 'mean_diff': float(d.mean()),
                            'ci90': [float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))],
                            'pos_share': float(np.mean(d > 0))}
    return out


def print_report(res):
    print(f"\n  VALUATION BACKTEST  ({res['n_obs']} company-years, {res['n_tickers']} "
          f"tickers, filing years {res['year_range']}; survivors, no analysts)")
    print(f"    {'Method/horizon':<16s} {'years':>5s} {'mean IC':>8s} {'90% CI':>17s}"
          f" {'IC>0':>5s} {'Q5−Q1':>7s}")
    for k, v in res['results'].items():
        note = '  (3y windows overlap)' if v['overlapping'] else ''
        print(f"    {k:<16s} {v['years']:>5d} {v['mean_ic']:>+8.3f} "
              f"[{v['ic_ci90'][0]:+.3f},{v['ic_ci90'][1]:+.3f}] {v['ic_pos_share']:>5.0%} "
              f"{v['mean_q5_q1']:>+7.1%}{note}")
    for h, v in res.get('paired', {}).items():
        print(f"    v2h − v1h, same company-years, {h}: {v['mean_diff']:+.3f} "
              f"[{v['ci90'][0]:+.3f},{v['ci90'][1]:+.3f}] over {v['years']} years, "
              f"{v['pos_share']:.0%} positive")
    print("\n  IC: per filing year, Spearman of MoS vs excess total return over SPY.")


def run(rows, prices_dir='output/prices'):
    from data.fred_client import FREDClient
    from datetime import date
    # History up to a month old is fine for a 2010+ backtest.
    obs_rf = FREDClient(max_age_days=30).fetch_series('DGS10', start=date(2005, 1, 1))
    rf = pd.Series({pd.Timestamp(str(k)): v for k, v in obs_rf.items()}, dtype=float)
    rf = (pd.to_numeric(rf, errors='coerce') / 100.0).sort_index().dropna()
    if rf.empty:
        raise SystemExit('no DGS10 history from FRED')
    obs = build_observations(rows, Prices(prices_dir), rf, base_rates_point_in_time(rows))
    years = sorted({o['year'] for o in obs})
    return {'n_obs': len(obs), 'n_tickers': len({o['ticker'] for o in obs}),
            'year_range': f'{years[0]}-{years[-1]}' if years else '-',
            'results': evaluate(obs), 'paired': paired(obs)}


def _load_dotenv():
    """The pipeline's .env parser (FRED_API_KEY), without overriding env."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


def main(argv=None):
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--snapshot')
    ap.add_argument('--results-dir', default='output')
    ap.add_argument('--prices-dir', default='output/prices')
    ap.add_argument('--out', default='output/valuation_backtest.json')
    a = ap.parse_args(argv)
    path = a.snapshot
    if not path:
        from data.snapshot_store import list_snapshot_files
        path = list_snapshot_files(a.results_dir)[-1][1]
    rows = gh._load_rows(path)
    res = run(rows, a.prices_dir)
    res['snapshot'] = os.path.basename(str(path))
    print_report(res)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\n  wrote {a.out}")


if __name__ == '__main__':
    main()
