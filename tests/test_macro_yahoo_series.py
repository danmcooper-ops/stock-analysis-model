# tests/test_macro_yahoo_series.py
"""data.macro_client.fetch_yahoo_closes: the Macro tab's gold and silver cards.

FRED dropped the LBMA precious-metals fixes in 2022, so those two cards come
from Yahoo. The fetcher must return the {date: value} shape a FRED series has
and degrade to {} on any failure, so a Yahoo outage costs only those cards.
"""
from datetime import date

import pandas as pd

from data import macro_client


class _Ticker:
    def __init__(self, hist=None, exc=None):
        self.hist, self.exc, self.kwargs = hist, exc, None

    def history(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.hist


def _patch(monkeypatch, ticker):
    monkeypatch.setattr(macro_client, '_yf_session', lambda: None)
    monkeypatch.setattr(macro_client.yf, 'Ticker', lambda symbol, session=None: ticker)


def test_closes_as_date_dict_dropping_nan_and_nonpositive(monkeypatch):
    idx = pd.to_datetime(['2026-10-20', '2026-10-21', '2026-10-22', '2026-10-23'])
    t = _Ticker(pd.DataFrame({'Close': [2401.5, float('nan'), 0.0, 2410.25]}, index=idx))
    _patch(monkeypatch, t)
    out = macro_client.fetch_yahoo_closes('GC=F', start=date(2006, 10, 23))
    assert out == {date(2026, 10, 20): 2401.5, date(2026, 10, 23): 2410.25}
    assert t.kwargs == {'start': '2006-10-23'}


def test_no_start_reads_ten_years(monkeypatch):
    t = _Ticker(pd.DataFrame({'Close': [30.1]}, index=pd.to_datetime(['2026-10-23'])))
    _patch(monkeypatch, t)
    macro_client.fetch_yahoo_closes('SI=F')
    assert t.kwargs == {'period': '10y'}


def test_failure_is_empty(monkeypatch):
    _patch(monkeypatch, _Ticker(exc=RuntimeError('Too Many Requests')))
    assert macro_client.fetch_yahoo_closes('GC=F') == {}


def test_empty_history_is_empty(monkeypatch):
    _patch(monkeypatch, _Ticker(pd.DataFrame()))
    assert macro_client.fetch_yahoo_closes('GC=F') == {}
