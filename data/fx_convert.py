"""Currency translation to USD over the FX rate store (data/fx_store.py).

Replaces data/fx_client.py, which converted every value of a year at that
calendar year's last close and passed a value through unconverted when it
had no rate. Translation now follows ASC 830 / IAS 21:

- **flows** (income statement, cash flow, per-share earnings and
  dividends) at the average rate over the fiscal year that produced them;
- **stocks** (balance-sheet instants) at the closing rate on the fiscal
  year's end date;
- **spot** values (yfinance ``info`` totals) at the run date's rate.

A March-year-end yen filer's FY2025 revenue used to take 2025's December
close, nine months after the period; it now takes the April 2024 - March
2025 average.

A year or column with no rate is blanked, never passed through: a yen
figure read as dollars is off by 150x, while a missing year is something
every consumer already handles. Share counts are never converted.

The store is filled by ``scripts/fx_rates.py refresh`` before the nightly
analysis. A currency it does not hold (a filer reporting in one no snapshot
had), or holds only up to a date too early for the lookup, is fetched once
per process on first use (``ensure``), so a dev run needs no setup step.
"""

import logging
import threading
from datetime import date, timedelta

import pandas as pd

from data.fx_store import FXStore, _iso

logger = logging.getLogger(__name__)

FLOW = 'flow'
STOCK = 'stock'
YEAR_DAYS = 364   # a flow's averaging window ends on its period end

_STORE = None
_STORE_LOCK = threading.Lock()
_ENSURED = set()
_ENSURE_LOCK = threading.Lock()


def default_store():
    """The process-wide FXStore (``FX_STORE_DIR`` or output/fx)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = FXStore()
        return _STORE


def ensure(ccy, as_of=None, store=None):
    """Fetch *ccy* into the store once per process when it holds no
    history, or none that reaches *as_of*."""
    ccy = _iso(ccy)
    if ccy in ('', 'USD'):
        return
    store = store or default_store()
    if store.frame(ccy) is not None and (
            as_of is None or store.rate_info(ccy, as_of) is not None):
        return
    with _ENSURE_LOCK:
        key = (store.fx_dir, ccy)
        if key in _ENSURED:
            return
        _ENSURED.add(key)
        logger.info("FX: %s not in the store%s; fetching it", ccy,
                    f' as of {as_of}' if as_of else '')
        store._frames.pop(ccy, None)
        store.refresh([ccy])


def _resolve_end(fy, fy_ends):
    """Period end for fiscal year *fy*: the filer's own date when known,
    else a known end's month and day shifted by the year gap, else 31 Dec."""
    end = fy_ends.get(fy)
    if end:
        return pd.Timestamp(end).date()
    if fy_ends:
        ref_fy = min(fy_ends, key=lambda y: abs(y - fy))
        ref = pd.Timestamp(fy_ends[ref_fy]).date()
        try:
            return ref.replace(year=ref.year + (fy - ref_fy))
        except ValueError:          # 29 Feb into a common year
            return ref.replace(year=ref.year + (fy - ref_fy), day=28)
    return date(int(fy), 12, 31)


def annual_rates(ccy, years, fy_ends=None, store=None):
    """``{fy: (flow_rate, stock_rate)}`` for *years*; either rate may be
    None. *fy_ends* maps fiscal years to their period-end dates."""
    ccy = _iso(ccy)
    years = sorted({int(y) for y in years})
    if ccy == 'USD':
        return {y: (1.0, 1.0) for y in years}
    store = store or default_store()
    fy_ends = {int(k): v for k, v in (fy_ends or {}).items()}
    ends = {fy: _resolve_end(fy, fy_ends) for fy in years}
    if ends:
        ensure(ccy, as_of=max(ends.values()), store=store)
    return {fy: (store.avg_rate(ccy, end - timedelta(days=YEAR_DAYS), end),
                 store.close_rate(ccy, end))
            for fy, end in ends.items()}


def convert_annual(values, rates, kind):
    """``{fy: value}`` in USD. *kind* picks the flow or stock rate; a year
    without that rate is dropped."""
    if not values:
        return values
    i = 0 if kind == FLOW else 1
    out = {}
    for fy, v in values.items():
        try:
            r = (rates.get(int(fy)) or (None, None))[i]
        except (TypeError, ValueError):
            r = None
        if r is None:
            continue
        out[fy] = None if v is None else v * r
    return out


def get_spot_fx_rate(currency, as_of=None, store=None):
    """USD per unit of *currency* as of *as_of* (default today), or None.
    None for USD too, so callers can tell "nothing to convert" apart only
    by the currency, as with the fx_client function this replaces."""
    ccy = _iso(currency)
    if ccy in ('', 'USD'):
        return None
    as_of = as_of or date.today()
    store = store or default_store()
    ensure(ccy, as_of=as_of, store=store)
    return store.rate(ccy, as_of)


def _is_share_count_label(label):
    """True for yfinance's share-count rows ('Diluted Average Shares',
    'Ordinary Shares Number', 'Share Issued'). fx_client also skipped any
    label with 'basic', 'diluted' or 'issued' in it, which left 'Basic EPS',
    'Diluted EPS' and 'Diluted NI Availto Com Stockholders' in the
    reporting currency."""
    s = str(label).lower()
    return 'shares' in s or s == 'share issued'


def convert_statement_df(df, ccy, kind, store=None):
    """A yfinance statement frame (rows = line items, columns = period-end
    dates) in USD, each column at its own period's rate. Share-count rows
    are left alone; a column with no rate is blanked. Returns
    ``(frame, blanked_columns)``; never mutates *df*."""
    if df is None or getattr(df, 'empty', True):
        return df, []
    ccy = _iso(ccy)
    if ccy == 'USD':
        return df, []
    store = store or default_store()
    ensure(ccy, store=store)
    out = df.copy()
    rows = [lbl for lbl in out.index if not _is_share_count_label(lbl)]
    blanked = []
    for col in out.columns:
        try:
            end = pd.Timestamp(col).date()
        except (TypeError, ValueError):
            end = None
        if end is None:
            rate = None
        elif kind == FLOW:
            rate = store.avg_rate(ccy, end - timedelta(days=YEAR_DAYS), end)
        else:
            rate = store.close_rate(ccy, end)
        vals = pd.to_numeric(out.loc[rows, col], errors='coerce')
        if rate is None:
            blanked.append(col)
            out.loc[rows, col] = float('nan')
        else:
            out.loc[rows, col] = vals * rate
    return out, blanked


# -- valuation at today's rate --------------------------------------------------
#
# Translation at historical rates is right for the record, but a fair value is
# a claim on future cash flows, which will translate at future rates; the best
# guess of those is today's (spot). So the valuation inputs are rebased: every
# column of a translated statement is scaled by one factor per kind, today's
# rate over the newest fiscal year's (its average for flows, its close for
# balance sheets). Growth rates are those of the translated history exactly,
# since every year moves by the same factor, while the newest year's level
# sits at today's rate and its returns on capital are the reporting
# currency's own. A yen filer whose FY ended in March, with the yen 5% firmer
# since, is valued on 5% more dollars of earnings.

#
# Two guards, both from 2026-10-26's 283 SEC-sourced foreign filers. A stale
# newest year (Toyota's and Honda's SEC histories end years back) would carry
# years of drift into one factor, so only a fiscal year that ended within
# REBASE_MAX_AGE_DAYS is rebased. And a move past REBASE_MAX_MOVE is mostly
# inflation, not a change in what the business earns: the peso and lira fall
# roughly in step with local prices, so the next year's earnings come in more
# pesos and marking last year's to spot counts the depreciation without the
# inflation (YPF read -93%, Turkcell -41%). Either way the row keeps its
# translated rates and the reason is returned.
REBASE_MAX_AGE_DAYS = 455      # a fiscal year end plus a 20-F's filing lag
REBASE_MAX_MOVE = 0.25         # |factor - 1| for flows or stocks


def spot_rebase_factors(ccy, period_end, as_of, store=None):
    """``{'flow', 'stock', 'spot', 'period_end'}`` for the fiscal year ending
    *period_end*; ``{'skipped': reason, ...}`` when a guard declines it; None
    when a rate is missing or the currency is USD. Only a dict with ``flow``
    rebases anything."""
    ccy = _iso(ccy)
    if ccy in ('', 'USD') or period_end is None:
        return None
    store = store or default_store()
    end = pd.Timestamp(period_end).date()
    as_of = as_of or date.today()
    if (as_of - end).days > REBASE_MAX_AGE_DAYS:
        return {'skipped': 'stale', 'period_end': end.isoformat()}
    spot = get_spot_fx_rate(ccy, as_of, store=store)
    avg = store.avg_rate(ccy, end - timedelta(days=YEAR_DAYS), end)
    close = store.close_rate(ccy, end)
    if not spot or not avg or not close:
        return None
    flow, stock = spot / avg, spot / close
    if max(abs(flow - 1), abs(stock - 1)) > REBASE_MAX_MOVE:
        return {'skipped': 'large_move', 'period_end': end.isoformat(),
                'move': round(flow - 1, 4)}
    return {'flow': flow, 'stock': stock, 'spot': spot,
            'period_end': end.isoformat()}


def newest_period_end(yf_data):
    """Date of the newest column of the income statement (else the balance
    sheet), or None."""
    for key in ('income_statement', 'income_stmt', 'balance_sheet'):
        df = (yf_data or {}).get(key)
        if df is not None and not getattr(df, 'empty', True):
            try:
                return max(pd.Timestamp(c) for c in df.columns).date()
            except (TypeError, ValueError):
                continue
    return None


def _scale_df(df, factor):
    out = df.copy()
    rows = [lbl for lbl in out.index if not _is_share_count_label(lbl)]
    for col in out.columns:
        out.loc[rows, col] = pd.to_numeric(out.loc[rows, col], errors='coerce') * factor
    return out


def rebase_statements(yf_data, factors):
    """*yf_data* with its statement frames scaled by *factors* (flows by
    ``flow``, the balance sheet by ``stock``); share counts and ``info`` are
    left alone (``info`` totals are already at the run date's rate). Never
    mutates the input."""
    if not factors or 'flow' not in factors:
        return yf_data
    out = dict(yf_data)
    for key, kind in (('balance_sheet', STOCK), ('income_statement', FLOW),
                      ('income_stmt', FLOW), ('cash_flow', FLOW)):
        df = out.get(key)
        if df is not None and not getattr(df, 'empty', True):
            out[key] = _scale_df(df, factors[kind])
    return out
