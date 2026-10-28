"""Series maths. Every helper must return None rather than a plausible
default when there is not enough history — a currency with two years of data
must read as "insufficient", never as "zero volatility"."""

import math

import pytest

from models import series as sr


def _ramp(n, start=100.0, step=1.0):
    return [('p%03d' % i, start + i * step) for i in range(n)]


def test_tail_requires_full_window():
    s = _ramp(5)
    assert sr.tail(s, 5) == s
    assert sr.tail(s, 6) is None
    assert sr.tail(s, 0) is None
    assert sr.tail([], 1) is None


def test_z_score_of_a_ramp_is_positive_at_the_top():
    z = sr.z_score(_ramp(20), 20)
    assert z is not None and z > 1.0


def test_z_score_none_on_flat_series():
    # A genuinely fixed rate has zero dispersion; dividing by it would
    # manufacture an infinite valuation signal.
    flat = [('p%d' % i, 100.0) for i in range(30)]
    assert sr.z_score(flat, 30) is None


def test_z_score_none_when_window_unavailable():
    assert sr.z_score(_ramp(5), 60) is None


def test_pct_deviation_sign():
    s = _ramp(10, 100.0, 1.0)          # 100..109, mean 104.5
    dev = sr.pct_deviation(s, 10)
    assert dev == pytest.approx(109 / 104.5 - 1)
    assert dev > 0


def test_pct_change_and_skip_month_variant():
    s = [('a', 100.0), ('b', 110.0), ('c', 121.0)]
    assert sr.pct_change(s, 1) == pytest.approx(0.1)
    assert sr.pct_change(s, 2) == pytest.approx(0.21)
    # From 2-back to 1-back: the most recent point is skipped entirely.
    assert sr.pct_change_between(s, 2, 1) == pytest.approx(0.1)


def test_pct_change_guards():
    s = [('a', 100.0), ('b', 110.0)]
    assert sr.pct_change(s, 5) is None
    assert sr.pct_change_between(s, 1, 1) is None      # needs older > newer
    assert sr.pct_change([('a', 0.0), ('b', 5.0)], 1) is None   # zero base


def test_log_returns_skip_non_positive_prices():
    s = [('a', 100.0), ('b', 0.0), ('c', 110.0)]
    rets = sr.log_returns(s)
    assert len(rets) == 1
    assert rets[0] == pytest.approx(math.log(1.1))


def test_realized_vol_is_annualized():
    # Alternating +/-1% daily moves.
    s = []
    v = 100.0
    for i in range(60):
        s.append(('p%02d' % i, v))
        v *= 1.01 if i % 2 == 0 else (1 / 1.01)
    vol = sr.realized_vol(s, 50, periods_per_year=252)
    assert vol is not None and 0.05 < vol < 0.5


def test_realized_vol_zero_for_flat_series():
    flat = [('p%d' % i, 100.0) for i in range(40)]
    assert sr.realized_vol(flat, 30) == 0.0


def test_max_drawdown_is_positive_fraction():
    s = [('a', 100.0), ('b', 120.0), ('c', 60.0), ('d', 90.0)]
    assert sr.max_drawdown(s, 4) == pytest.approx(0.5)


def test_max_drawdown_zero_on_monotonic_rise():
    assert sr.max_drawdown(_ramp(10), 10) == 0.0


def test_moving_average():
    assert sr.moving_average(_ramp(4, 1.0, 1.0), 4) == pytest.approx(2.5)
    assert sr.moving_average(_ramp(3), 10) is None


def test_stdev_needs_two_points():
    assert sr.stdev([1.0]) is None
    assert sr.stdev([]) is None
    assert sr.stdev([1.0, 3.0]) == pytest.approx(math.sqrt(2))
