# tests/test_cagr_complex_guard.py
"""CAGR endpoints must be positive, and no payload may carry a complex number.

On 2026-09-08 HBNC (Horizon Bancorp) reported a NEGATIVE FY2025 total revenue
of -$26,966,000 — a bank booking a securities-portfolio restructuring loss.
The revenue-per-employee CAGR in ``_run_narratives`` guarded its EARLIEST
endpoint but not its LATEST one, so it evaluated a negative base to a
fractional power. Python 3 answers that with a COMPLEX number instead of
raising, and the value propagated silently: the results-JSON writer
stringified it into the snapshot, and ``report_html`` then died on

    TypeError: Object of type complex is not JSON serializable

at the very END of a 13.6-hour pipeline run, after every expensive phase had
already completed.

These tests lock in both halves of the fix — the arithmetic guard that stops
a complex value being produced, and the serialization backstops that keep one
from destroying a whole render if some other call site ever regresses.

Synthetic records only; no network, no filesystem.
"""

import json
import logging
import types

import numpy as np
import pytest

from scripts.analyze_stock import _make_json_safe, _run_narratives
from scripts.report_html import _json_default, _sanitize


def _record(ticker, revenue_history, employees=1000):
    return {
        'ticker': ticker,
        'sector': 'Financial Services',
        'employees': employees,
        'revenue': 200e6,
        'edgar_history': {'revenue_history': dict(revenue_history)},
    }


def _narrate(records):
    """Run the narrative phase over `records` with everything optional off."""
    args = types.SimpleNamespace(macro=False)
    _run_narratives(records, args, {}, None, None, None, None)
    return records


def _complex_values(obj, path=()):
    """Yield (dotted-path, value) for every complex leaf in a structure."""
    if isinstance(obj, (complex, np.complexfloating)) and not isinstance(obj, (bool, int, float)):
        yield '.'.join(str(p) for p in path), obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _complex_values(v, path + (k,))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _complex_values(v, path + (i,))


# --- The arithmetic guard --------------------------------------------------

class TestRpeCagrEndpoints:

    # HBNC's actual reported series (FY2025 restructuring loss).
    HBNC = {2021: 180_000_000, 2022: 190_000_000, 2023: 200_000_000,
            2024: 68_410_000, 2025: -26_966_000}

    def test_negative_latest_revenue_yields_none_not_complex(self):
        r = _narrate([_record('HBNC', self.HBNC)])[0]
        assert r['rpe_cagr'] is None
        assert not isinstance(r['rpe_cagr'], complex)

    def test_negative_earliest_revenue_yields_none(self):
        hist = {2021: -5_000_000, 2022: 190_000_000, 2023: 200_000_000}
        r = _narrate([_record('NEGSTART', hist)])[0]
        assert r['rpe_cagr'] is None

    def test_zero_latest_revenue_yields_none(self):
        hist = {2021: 100_000_000, 2022: 190_000_000, 2023: 0}
        r = _narrate([_record('ZEROEND', hist)])[0]
        assert r['rpe_cagr'] is None

    def test_positive_series_still_computes_the_cagr(self):
        """The guard must not suppress the ordinary case."""
        hist = {2021: 100_000_000, 2022: 110_000_000, 2023: 121_000_000}
        r = _narrate([_record('GOODCO', hist, employees=500)])[0]
        # Employees cancel out of the ratio: (121/100) ** (1/2) - 1 = 10%.
        assert r['rpe_cagr'] == pytest.approx(0.10, abs=1e-9)

    def test_no_record_in_a_mixed_universe_carries_a_complex_value(self):
        """A synthetic universe of pathological histories stays JSON-clean."""
        records = [
            _record('HBNC', self.HBNC),
            _record('NEGSTART', {2021: -5e6, 2022: 1.9e8, 2023: 2.0e8}),
            _record('BOTHNEG', {2021: -5e6, 2022: 1.0e8, 2023: -2.0e7}),
            _record('ZEROEND', {2021: 1.0e8, 2022: 1.9e8, 2023: 0}),
            _record('ZEROSTART', {2021: 0, 2022: 1.9e8, 2023: 2.0e8}),
            _record('TINYEND', {2021: 1.0e8, 2022: 1.9e8, 2023: 1.0}),
            _record('GOODCO', {2021: 1.0e8, 2022: 1.1e8, 2023: 1.21e8}),
            _record('SHORT', {2021: 1.0e8, 2022: -1.0e8}),
            _record('NOEMP', self.HBNC, employees=None),
        ]
        _narrate(records)
        offenders = [(rec['ticker'], p, v)
                     for rec in records for p, v in _complex_values(rec)]
        assert offenders == []

    def test_the_whole_universe_survives_the_json_writers(self):
        """End to end: narrate → results JSON → report payload, no TypeError."""
        records = [
            _record('HBNC', self.HBNC),
            _record('BOTHNEG', {2021: -5e6, 2022: 1.0e8, 2023: -2.0e7}),
            _record('GOODCO', {2021: 1.0e8, 2022: 1.1e8, 2023: 1.21e8}),
        ]
        _narrate(records)

        # The canonical results JSON (analyze_stock's writer).
        rows = [{k: _make_json_safe(v) for k, v in r.items()} for r in records]
        text = json.dumps(rows, default=str)
        assert 'j)' not in text          # no stringified complex leaked through

        # The report payload (report_html's writer).
        json.dumps(_sanitize(records), default=_json_default)


# --- The serialization backstops -------------------------------------------

class TestComplexBackstops:

    def test_make_json_safe_nulls_a_complex_value(self, caplog):
        with caplog.at_level(logging.WARNING, logger='analyze_stock'):
            assert _make_json_safe(complex(-0.0878, 0.2082)) is None
        assert 'complex' in caplog.text

    def test_make_json_safe_no_longer_stringifies_a_complex_value(self):
        """The old fallthrough wrote "(-0.0878+0.2082j)" into a numeric field."""
        out = _make_json_safe({'rpe_cagr': complex(-0.0878, 0.2082)})
        assert out == {'rpe_cagr': None}

    def test_json_default_coerces_complex_instead_of_raising(self, caplog):
        payload = {'ticker': 'HBNC', 'rpe_cagr': complex(-0.0878, 0.2082)}
        with caplog.at_level(logging.WARNING, logger='report_html'):
            text = json.dumps(payload, default=_json_default)
        assert json.loads(text)['rpe_cagr'] is None
        assert 'complex' in caplog.text

    def test_json_default_still_raises_for_genuinely_unknown_types(self):
        """The backstop is narrow on purpose — it must not mask novel bugs."""
        class Weird:
            pass

        with pytest.raises(TypeError):
            json.dumps({'x': Weird()}, default=_json_default)

    def test_sanitize_names_the_ticker_and_field(self, caplog):
        rows = [{'ticker': 'HBNC', 'rpe_cagr': complex(-0.0878, 0.2082)}]
        with caplog.at_level(logging.WARNING, logger='report_html'):
            out = _sanitize(rows)
        assert out == [{'ticker': 'HBNC', 'rpe_cagr': None}]
        assert 'HBNC.rpe_cagr' in caplog.text

    def test_sanitize_nulls_a_numpy_complex(self):
        rows = [{'ticker': 'X', 'v': np.complex128(complex(1, 2))}]
        assert _sanitize(rows) == [{'ticker': 'X', 'v': None}]

    def test_sanitize_nulls_a_stringified_complex_from_a_stale_snapshot(self, caplog):
        rows = [{'ticker': 'HBNC',
                 'rpe_cagr': '(-0.08779424569020489+0.20820501072235512j)'}]
        with caplog.at_level(logging.WARNING, logger='report_html'):
            out = _sanitize(rows)
        assert out[0]['rpe_cagr'] is None
        assert 'HBNC.rpe_cagr' in caplog.text

    @pytest.mark.parametrize('text', [
        '(a joke)', 'Djibouti', '(not a number j)', '', '(', 'j)',
        'Series J)', '(1, 2)',
    ])
    def test_sanitize_leaves_ordinary_strings_alone(self, text):
        assert _sanitize({'x': text}) == {'x': text}
