"""FX (foreign exchange) conversion helpers.

Shared between the SEC EDGAR XBRL historical pipeline (annual fiscal-year
conversion) and the yfinance live-data pipeline (spot conversion of
balance_sheet / income_statement / cash_flow + price / market cap).

Without this layer, foreign-domiciled tickers produce per-share fair-value
estimates in the local reporting currency that are then compared against a
USD-denominated price — yielding the "wild divergence" between DCF / DDM /
EPV / RIM / NAV that 36.5% of the universe exhibits.

Public API:
- ``_get_fx_rates_to_usd(currency)``: cached {fiscal_year: rate_to_USD} dict
  built from yfinance year-end closes. Used by the EDGAR pipeline for
  historical year-by-year conversion. Returns ``{}`` when yfinance has no
  data for the pair (silent pass-through).
- ``_apply_fx_annual(values, fx_rates)``: multiply each ``{year: value}``
  by the matching FX rate. Years missing from ``fx_rates`` pass through.
- ``get_spot_fx_rate(currency)``: return the most-recent cached rate for
  ``currency``. Walks back year-by-year from the current year if the
  latest entry is missing. Returns ``None`` (not 1.0) on failure so callers
  can distinguish "no conversion needed" from "fetch failed".
- ``apply_fx_to_statement_df(df, rate)``: multiply every numeric cell of a
  yfinance balance_sheet / income_statement / cash_flow DataFrame by
  ``rate``. Skips share-count rows (label contains "Shares" / "Issued" /
  "Diluted"). Returns a new DataFrame; never mutates input.
"""

import logging
from datetime import datetime as _dt

logger = logging.getLogger(__name__)


# Module-level FX cache. Keyed by ISO currency code; value is {fiscal_year: rate_to_USD}.
# Built lazily on first request per currency so a 50-foreign-ticker batch hits
# yfinance at most ~10 times instead of once per ticker.
_FX_CACHE = {}


def _get_fx_rates_to_usd(currency):
    """Return {fiscal_year: ccy→USD rate} using yfinance year-end closes.

    Falls back to {} if yfinance is unavailable or the pair has no data,
    in which case downstream conversion silently passes through values
    unchanged. That's acceptable for HKD (pegged ~7.78) and small CCY
    drift but introduces error for volatile currencies — callers should
    log ``reporting_currency`` so consumers can audit when this happens.
    """
    if currency in _FX_CACHE:
        return _FX_CACHE[currency]
    if currency == 'USD':
        _FX_CACHE[currency] = {}
        return {}
    try:
        import yfinance as yf
        pair = f'{currency}USD=X'
        df = yf.download(pair, period='max', interval='1d',
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            _FX_CACHE[currency] = {}
            return {}
        # yfinance returns a MultiIndex-columned DataFrame even for a single
        # ticker; pluck the Close column and squeeze to a Series.
        if 'Close' in df.columns.get_level_values(0):
            close = df['Close']
        else:
            close = df.iloc[:, [0]]
        if hasattr(close, 'squeeze'):
            close = close.squeeze('columns') if close.ndim == 2 else close
        rates = {}
        for ts, val in close.items():
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            # Iteration is chronological, so later in-year quotes overwrite
            # earlier ones — year-end close wins.
            rates[ts.year] = v
        _FX_CACHE[currency] = rates
        return rates
    except Exception as e:
        logger.warning(f"FX: rate fetch failed for {currency}: {e}")
        # Don't cache the failure: a transient yfinance error at run start
        # would otherwise leave every ticker in this currency unconverted
        # for the whole 3-6h run. Next ticker retries the fetch.
        return {}


def _apply_fx_annual(values, fx_rates):
    """Multiply each {year: value} by the matching FX rate.

    Years missing from fx_rates are passed through unchanged — pre-2003 EUR
    or other thin-history pairs will retain native-currency magnitudes,
    which is at least directionally correct for trend signal and never
    worse than dropping the year.
    """
    if not values or not fx_rates:
        return values
    out = {}
    for y, v in values.items():
        try:
            y_int = int(y)
        except (TypeError, ValueError):
            out[y] = v
            continue
        rate = fx_rates.get(y_int)
        out[y] = v * rate if (rate is not None and v is not None) else v
    return out


def get_spot_fx_rate(currency):
    """Return the most-recent CCY→USD rate, or ``None`` on failure.

    Walks back from the current year through any year the cache has
    populated, so a stale cache (built earlier in the session) still
    serves the latest known year. Distinct ``None`` return value lets
    callers distinguish "USD or unknown — no conversion needed" from
    "fetch failed — pass through with a warning flag".
    """
    if not currency or currency == 'USD':
        return None
    rates = _get_fx_rates_to_usd(currency)
    if not rates:
        return None
    current_year = _dt.now().year
    # Walk back at most 10 years for stale-cache safety. Even thin-history
    # pairs (e.g., recently-introduced ISO codes) typically have at least
    # one year of data in the last decade.
    for y in range(current_year, current_year - 10, -1):
        if y in rates:
            r = rates[y]
            if r is not None and r > 0:
                return float(r)
    return None


_SHARE_COUNT_HINTS = ('shares', 'issued', 'diluted', 'outstanding', 'basic')


def _is_share_count_label(label):
    """Heuristic: does this row label name a share count (not a dollar value)?"""
    if label is None:
        return False
    s = str(label).lower()
    return any(h in s for h in _SHARE_COUNT_HINTS)


def apply_fx_to_statement_df(df, rate):
    """Return a copy of ``df`` with every numeric cell multiplied by ``rate``.

    Skips rows whose label looks like a share count (those are unitless and
    should stay in their original magnitude). Returns ``df`` unchanged when
    ``rate`` is None, missing, or ``df`` is empty / None.
    """
    if df is None or rate is None:
        return df
    if hasattr(df, 'empty') and df.empty:
        return df
    try:
        import pandas as pd  # local import — keeps the module light if pandas isn't needed
    except Exception as e:
        logger.debug(f"FX: pandas import failed, statement left unconverted: {e}")
        return df
    try:
        out = df.copy()
    except Exception as e:
        logger.debug(f"FX: statement copy failed, left unconverted: {e}")
        return df
    for idx in out.index:
        if _is_share_count_label(idx):
            continue
        try:
            out.loc[idx] = out.loc[idx].apply(
                lambda v: v * rate if (v is not None and pd.notna(v)) else v
            )
        except Exception as e:
            # If a row resists multiplication (non-numeric / mixed types),
            # leave it as-is rather than corrupting the whole frame.
            logger.debug(f"FX: row conversion failed for {idx}: {e}")
            continue
    return out
