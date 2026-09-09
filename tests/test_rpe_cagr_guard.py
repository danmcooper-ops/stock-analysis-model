"""Revenue-per-employee CAGR must never produce a complex number.

Regression coverage for the 2026-09-08 nightly run, which completed 13.6
hours of analysis and then died rendering the report:

    TypeError: Object of type complex is not JSON serializable
    when serializing dict item 'rpe_cagr'

HBNC (Horizon Bancorp) tagged a NEGATIVE 2025 total revenue of -26,966,000 —
a bank booking a securities-portfolio restructuring loss. The guard required
the EARLIEST year to be positive but not the LATEST, so the ratio went
negative and, in Python 3, a negative float raised to a fractional power
returns a complex number rather than raising. It then flowed all the way to
the report writer: the JSON path's _make_json_safe stringified it, while
report_html's stricter _json_default raised and took the whole render with it.

A CAGR to a negative endpoint is undefined, so None is the correct answer.
"""

import pytest

from scripts.analyze_stock import _run_narratives


def _record(ticker, employees, revenue_by_year):
    return {
        'ticker': ticker,
        'sector': 'Financial Services',
        'employees': employees,
        'edgar_history': {'revenue_history': dict(revenue_by_year)},
    }


def _rpe_cagr_for(record):
    """Run the narrative pass over one record and return its rpe_cagr."""
    results = [record]
    _run_narratives(results, _Args(), {}, None, {}, {}, {})
    return results[0].get('rpe_cagr')


class _Args:
    """Minimal stand-in for the argparse namespace the pass reads."""
    macro = False
    universe = 'us'


class TestRpeCagrNegativeEndpoint:
    def test_negative_latest_revenue_yields_none_not_complex(self):
        """The HBNC case: 2011 positive, 2025 negative."""
        rec = _record('HBNC', 465, {
            2011: 68412000, 2012: 85537000, 2013: 87289000,
            2024: 246969000, 2025: -26966000,
        })
        value = _rpe_cagr_for(rec)
        assert not isinstance(value, complex)
        assert value is None

    def test_negative_earliest_revenue_yields_none(self):
        rec = _record('TEST', 100, {
            2020: -5000000, 2021: 1000000, 2025: 8000000,
        })
        value = _rpe_cagr_for(rec)
        assert not isinstance(value, complex)
        assert value is None

    def test_all_positive_revenue_still_computes(self):
        """The guard must not suppress the ordinary case."""
        rec = _record('TEST', 100, {
            2020: 100000000, 2021: 110000000, 2025: 200000000,
        })
        value = _rpe_cagr_for(rec)
        assert isinstance(value, float)
        # 2x over 5 years ~= 14.9%/yr, independent of the constant headcount.
        assert value == pytest.approx(2 ** (1 / 5) - 1, rel=1e-6)


def test_no_record_ever_carries_a_complex_rpe_cagr():
    """Whole-pass invariant: nothing complex reaches the report writers."""
    results = [
        _record('A', 465, {2011: 68412000, 2025: -26966000}),
        _record('B', 100, {2020: -1.0, 2025: -2.0}),
        _record('C', 250, {2020: 5000000, 2025: 9000000}),
        _record('D', 0, {2020: 5000000, 2025: 9000000}),
    ]
    _run_narratives(results, _Args(), {}, None, {}, {}, {})
    assert all(not isinstance(r.get('rpe_cagr'), complex) for r in results)
