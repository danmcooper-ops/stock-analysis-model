# tests/test_price_history_payloads.py
"""Regression tests for the report's price-history payloads.

The chart's MAX range once silently clipped every series to the last 20
years (over half the universe shared the same artificial start date), and
Close-only write-through stubs from YFinanceClient._maybe_persist_prices
never got upgraded to full history by the refresh pass. These tests pin the
fixed behavior: full parquet history ships to the report, sub-$1 closes keep
enough precision to chart, and the stub detector tells seeded files apart
from real downloads.
"""
from datetime import date

import numpy as np
import pandas as pd

from scripts.download_prices import _parquet_is_stub, download_ticker
from scripts.report_html import _load_price_payloads


def _write_prices(path, years=25, cols=('Open', 'High', 'Low', 'Close', 'Volume'),
                  start_price=50.0):
    idx = pd.bdate_range(end=pd.Timestamp('2026-08-28'), periods=int(years * 252))
    rng = np.random.default_rng(7)
    close = start_price * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))))
    close = np.maximum(close, 0.02)
    frame = {
        'Open': close, 'High': close * 1.01, 'Low': close * 0.99,
        'Close': close, 'Volume': np.full(len(idx), 1e6),
    }
    df = pd.DataFrame({c: frame[c] for c in cols}, index=idx)
    df.to_parquet(path)
    return idx


def test_price_payloads_ship_full_history(tmp_path):
    idx = _write_prices(str(tmp_path / 'OLD.parquet'), years=25)
    prices, px, _vol = _load_price_payloads([{'ticker': 'OLD'}], str(tmp_path))
    assert prices is not None and 'OLD' in px
    # No 20-year clip: the dates axis reaches the parquet's first bar.
    assert prices['dates'][0] == idx[0].strftime('%Y-%m-%d')
    assert prices['dates'][-1] == idx[-1].strftime('%Y-%m-%d')
    shard = px['OLD']
    assert shard['i0'] == 0
    assert len(shard['p']) == len(prices['dates'])


def test_sub_dollar_closes_keep_precision(tmp_path):
    _write_prices(str(tmp_path / 'PNY.parquet'), years=2, start_price=0.09)
    _prices, px, _vol = _load_price_payloads([{'ticker': 'PNY'}], str(tmp_path))
    vals = [v for v in px['PNY']['p'] if v is not None]
    assert vals
    # 2-decimal rounding would quantize a $0.09 series onto {0.08,0.09,0.10};
    # sub-$1 closes ship with 4 decimals instead.
    assert any(round(v, 2) != v for v in vals)


def test_parquet_stub_detection(tmp_path):
    full = str(tmp_path / 'FULL.parquet')
    stub = str(tmp_path / 'STUB.parquet')
    _write_prices(full, years=1)
    _write_prices(stub, years=1, cols=('Close',))
    assert not _parquet_is_stub(full)
    assert _parquet_is_stub(stub)


def test_refresh_leaves_fresh_full_files_alone(tmp_path, monkeypatch):
    # A current full-history file short-circuits before any network call.
    _write_prices(str(tmp_path / 'FULL.parquet'), years=1)
    monkeypatch.setattr('scripts.download_prices.date',
                        type('D', (), {'today': staticmethod(lambda: date(2026, 8, 28))}))
    assert download_ticker('FULL', str(tmp_path), delay=0,
                           refresh=True, max_age_days=7) == 'fresh'


def test_refresh_redownloads_stub_files(tmp_path, monkeypatch):
    # A fresh-but-Close-only stub must go back to the network for max history.
    _write_prices(str(tmp_path / 'STUB.parquet'), years=1, cols=('Close',))
    monkeypatch.setattr('scripts.download_prices.date',
                        type('D', (), {'today': staticmethod(lambda: date(2026, 8, 28))}))

    def _boom(*a, **k):
        raise RuntimeError('network attempted — stub correctly not skipped')
    monkeypatch.setattr('scripts.download_prices.yf.Ticker', _boom)
    result = download_ticker('STUB', str(tmp_path), delay=0,
                             refresh=True, max_age_days=7)
    assert result.startswith('error:')
