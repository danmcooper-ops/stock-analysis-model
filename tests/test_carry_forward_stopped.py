"""Carry-forward tickers that stopped trading are dropped from the run.

Yahoo keeps serving a delisted symbol's last quote in ``.info``, and
carry-forward bypasses Phase 1's mcap/spread filters, so an acquired company
used to be re-rated on a frozen price every night (JHG at $51.95 from
2026-07-02 to 08-04; its parquet held 3 bars, all before 07-03). These pin
the rule — a parquet more than CARRY_FORWARD_MAX_PRICE_LAG_BARS SPY trading
days behind SPY — and the cases where it must stay off.
"""

import io
import logging
import sys
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.analyze_stock as A  # noqa: E402
from scripts.analyze_stock import (_drop_stopped_carry_forwards,  # noqa: E402
                                   stopped_trading_carry_forwards)
from scripts.config import CARRY_FORWARD_MAX_PRICE_LAG_BARS  # noqa: E402

pytest.importorskip('duckdb')

RUN_DAY = date(2026, 9, 21)                      # a Monday
SPY_DAYS = pd.bdate_range(end=pd.Timestamp(RUN_DAY), periods=400)


def _write(tmp_path, ticker, days):
    df = pd.DataFrame({'Close': [100.0 + i for i in range(len(days))]},
                      index=pd.DatetimeIndex(days, name='Date'))
    df.to_parquet(tmp_path / f"{ticker}.parquet")


def _ending_bars_ago(n, periods=300):
    """Business days ending *n* SPY bars before SPY's last bar."""
    end = SPY_DAYS[-1 - n]
    return SPY_DAYS[SPY_DAYS <= end][-periods:]


@pytest.fixture
def prices(tmp_path):
    _write(tmp_path, 'SPY', SPY_DAYS)
    return tmp_path


# --- the rule ---------------------------------------------------------------

def test_live_ticker_is_kept_and_dead_one_dropped(prices):
    _write(prices, 'LIVE', _ending_bars_ago(0))
    _write(prices, 'DEAD', _ending_bars_ago(40))
    out = stopped_trading_carry_forwards({'LIVE', 'DEAD'}, str(prices), RUN_DAY)
    assert set(out) == {'DEAD'}
    last_bar, lag = out['DEAD']
    assert last_bar == SPY_DAYS[-41].date().isoformat() and lag == 40


def test_lag_boundary_is_counted_in_spy_trading_days(prices):
    n = CARRY_FORWARD_MAX_PRICE_LAG_BARS
    _write(prices, 'EDGE', _ending_bars_ago(n))
    _write(prices, 'OVER', _ending_bars_ago(n + 1))
    out = stopped_trading_carry_forwards({'EDGE', 'OVER'}, str(prices), RUN_DAY)
    assert set(out) == {'OVER'}


def test_the_one_bar_file_yahoo_leaves_for_a_delisted_symbol(prices):
    """period="max" for a dead symbol returns only its final quote, which the
    >60-bar freshness gate ignores — that is why price_data_stale never
    fired on these rows. The rule must not need history."""
    _write(prices, 'JHG', [SPY_DAYS[-55]])
    assert set(stopped_trading_carry_forwards({'JHG'}, str(prices), RUN_DAY)) == {'JHG'}


def test_a_ticker_with_no_parquet_is_never_dropped(prices):
    assert stopped_trading_carry_forwards({'NOFILE'}, str(prices), RUN_DAY) == {}


@pytest.mark.parametrize('spy_end_ago_days', [None, 8])
def test_rule_is_off_without_a_current_spy(tmp_path, spy_end_ago_days):
    _write(tmp_path, 'DEAD', _ending_bars_ago(40))
    if spy_end_ago_days is not None:
        cutoff = pd.Timestamp(RUN_DAY) - pd.Timedelta(days=spy_end_ago_days)
        _write(tmp_path, 'SPY', SPY_DAYS[SPY_DAYS <= cutoff])
    assert stopped_trading_carry_forwards({'DEAD'}, str(tmp_path), RUN_DAY) == {}


def test_a_night_whose_download_failed_wholesale_drops_nothing(tmp_path):
    """Every file — SPY included — a few days behind: lag is measured
    against SPY's last bar, not the run date."""
    behind = SPY_DAYS[:-3]
    _write(tmp_path, 'SPY', behind)
    _write(tmp_path, 'AAA', behind)
    assert stopped_trading_carry_forwards({'AAA'}, str(tmp_path), RUN_DAY) == {}


def test_a_past_run_date_ignores_later_bars(prices):
    """--run-date re-run: a ticker that stopped AFTER the session date was
    live then; one that had already stopped was not."""
    past = SPY_DAYS[-101].date()                        # 100 bars back
    _write(prices, 'LATER', _ending_bars_ago(50))       # stopped after `past`
    _write(prices, 'EARLY', _ending_bars_ago(150))      # stopped 50 bars before
    out = stopped_trading_carry_forwards({'LATER', 'EARLY'}, str(prices), past)
    assert set(out) == {'EARLY'} and out['EARLY'][1] == 50


def test_unset_inputs_are_a_no_op(prices):
    assert stopped_trading_carry_forwards(set(), str(prices), RUN_DAY) == {}
    assert stopped_trading_carry_forwards({'X'}, None, RUN_DAY) == {}
    assert stopped_trading_carry_forwards({'X'}, str(prices), None) == {}


# --- the run's wrapper ------------------------------------------------------

class _Prov:
    def __init__(self):
        self.events = []

    def record_event(self, etype, ticker=None, source=None, detail=None):
        self.events.append((etype, ticker, source, detail))


def test_each_drop_is_logged_and_recorded(prices, caplog):
    _write(prices, 'DEAD', _ending_bars_ago(40))
    _write(prices, 'LIVE', _ending_bars_ago(0))
    prov = _Prov()
    with caplog.at_level(logging.WARNING, logger=A.logger.name), \
            redirect_stdout(io.StringIO()) as buf:
        dropped = _drop_stopped_carry_forwards({'DEAD', 'LIVE'}, str(prices),
                                               RUN_DAY, prov)
    assert dropped == {'DEAD'}
    assert any('DEAD: dropped from carry-forward' in r.getMessage()
               and '40 SPY trading days' in r.getMessage() for r in caplog.records)
    assert prov.events == [('carry_forward_stopped', 'DEAD', 'prices',
                            {'last_bar': SPY_DAYS[-41].date().isoformat(),
                             'lag_bars': 40})]
    assert 'dropped 1 ticker(s)' in buf.getvalue()


def test_a_mass_stop_reads_as_a_failed_refresh(prices, caplog):
    """Hundreds of names do not delist in one night; a stale price set does."""
    carry = {f'T{i:02d}' for i in range(30)}
    for t in carry:
        _write(prices, t, _ending_bars_ago(40))
    with caplog.at_level(logging.WARNING, logger=A.logger.name):
        assert _drop_stopped_carry_forwards(carry, str(prices), RUN_DAY,
                                            guard_floor=10) == set()
    assert any('failed price refresh' in r.getMessage() for r in caplog.records)


def test_a_failing_check_keeps_every_ticker(monkeypatch, caplog):
    def _boom(*a, **k):
        raise RuntimeError('duckdb exploded')
    monkeypatch.setattr(A, 'stopped_trading_carry_forwards', _boom)
    with caplog.at_level(logging.WARNING, logger=A.logger.name):
        assert _drop_stopped_carry_forwards({'A'}, 'x', RUN_DAY) == set()
    assert any('keeping every carry-forward' in r.getMessage() for r in caplog.records)


# --- wired into Phase 1 -----------------------------------------------------

def test_phase1_never_fetches_a_stopped_carry_forward(prices, monkeypatch):
    """The dead carry-forward is neither fetched nor qualified, even though
    Yahoo would hand back a healthy-looking frozen quote for it."""
    from tests.test_phase1_prefetch import _args, _FakeSEC, _FakeYF, _Prov as _P

    _write(prices, 'LIVE', _ending_bars_ago(0))
    _write(prices, 'DEAD', [SPY_DAYS[-55]])
    monkeypatch.setattr(A, 'calculate_roic', lambda d: {'roic_median_5y': 0.15})
    monkeypatch.setattr(A, 'calculate_wacc', lambda *a, **k: 0.08)
    monkeypatch.setattr(A, 'select_cost_of_equity',
                        lambda *a, **k: (0.09, 'capm', None))
    monkeypatch.setattr(A, '_convert_financials_to_usd', lambda d, **k: (d, {}))
    monkeypatch.setattr(A, '_fresh_local_prices', lambda *a, **k: None)
    monkeypatch.setattr(A, 'prior_snapshot_file',
                        lambda *a, **k: ('2026-09-18', 'output/results_2026-09-18.json'))
    monkeypatch.setattr(A, '_load_carry_forward_rows',
                        lambda *a, **k: [{'ticker': 'LIVE'}, {'ticker': 'DEAD'}])

    yf = _FakeYF(mcaps={'LIVE': 1e8, 'DEAD': 1e8})    # both below the floor
    with redirect_stdout(io.StringIO()) as buf:
        out = A._run_phase1_screen(
            _args(), _P(), ['OTHER'], {'OTHER': 'quality'}, yf, None,
            _FakeSEC(ciks=()), 0.04, 0.045, prices_dir=str(prices),
            phase1_workers=1)
    assert 'DEAD' not in yf.fetched
    assert 'LIVE' in out['qualifying']            # carry-forward still bypasses mcap
    assert 'DEAD' not in out['qualifying']
    assert 'Carry-forward: 1 ticker(s)' in buf.getvalue()


# --- the report -------------------------------------------------------------
# Snapshots written before the Phase-1 rule still hold stopped tickers; the
# render leaves them out with the same rule.

def _report_rows():
    return [{'ticker': 'LIVECO', 'price': 10.0, 'rating': 'HOLD'},
            {'ticker': 'DEADCO', 'price': 51.95, 'rating': 'LEAN BUY'},
            {'ticker': 'NOFILE', 'price': 5.0, 'rating': 'PASS'}]


def _report_prices(prices):
    _write(prices, 'LIVECO', _ending_bars_ago(0))
    _write(prices, 'DEADCO', [SPY_DAYS[-55]])


def test_report_leaves_out_stopped_rows(prices, caplog):
    from scripts.report_html import _drop_stopped_rows
    _report_prices(prices)
    with caplog.at_level(logging.WARNING, logger='report_html'), \
            redirect_stdout(io.StringIO()):
        kept = _drop_stopped_rows(_report_rows(), str(prices), RUN_DAY)
    assert [r['ticker'] for r in kept] == ['LIVECO', 'NOFILE']
    assert any('DEADCO: left out of the report' in r.getMessage()
               for r in caplog.records)


def test_report_needs_a_run_date_to_judge(prices):
    """An old snapshot rendered without its date must not be judged against
    today's parquets — names live on its date would vanish."""
    from scripts.report_html import _drop_stopped_rows
    _report_prices(prices)
    rows = _report_rows()
    assert _drop_stopped_rows(rows, str(prices), None) == rows
    assert _drop_stopped_rows(rows, None, RUN_DAY) == rows


def test_report_mass_stop_renders_every_row(prices):
    from scripts.report_html import _drop_stopped_rows
    rows = [{'ticker': f'T{i:02d}'} for i in range(30)]
    for r in rows:
        _write(prices, r['ticker'], _ending_bars_ago(40))
    assert _drop_stopped_rows(rows, str(prices), RUN_DAY) == rows


def test_rendered_report_omits_the_stopped_row(prices, tmp_path):
    from scripts.report_html import build_html
    _report_prices(prices)
    site = tmp_path / 'site'
    site.mkdir()
    out = site / 'report.html'
    with redirect_stdout(io.StringIO()):
        build_html(_report_rows(), str(out), prices_dir=str(prices),
                   run_date=RUN_DAY)
    html = out.read_text(encoding='utf-8')
    assert 'LIVECO' in html and 'NOFILE' in html
    assert 'DEADCO' not in html
