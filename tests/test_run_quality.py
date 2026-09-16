"""analyze_stock: tickers that lose their SEC ticker-map entry are surfaced."""

import json
import logging
from datetime import date
from types import SimpleNamespace

from scripts import analyze_stock as a


def _prior():
    return [
        {'ticker': 'LEG', 'data_source': 'sec_xbrl+yfinance'},   # dropped from map
        {'ticker': 'AAPL', 'data_source': 'sec_xbrl+yfinance'},  # still mapped
        {'ticker': 'ONLY', 'data_source': 'sec_xbrl'},           # dropped from map
        {'ticker': 'MNSKY', 'data_source': 'yfinance'},          # never had SEC
        {'ticker': None, 'data_source': 'sec_xbrl'},
        'not a row',
    ]


MAP = {'AAPL': '0000320193', 'MNSKY': '0000000001'}


def test_lost_sec_tickers_only_prior_sec_rows_missing_from_map():
    assert a.lost_sec_tickers(_prior(), MAP) == ['LEG', 'ONLY']


def test_lost_sec_tickers_ignores_a_failed_map_load():
    assert a.lost_sec_tickers(_prior(), {}) == []
    assert a.lost_sec_tickers(_prior(), None) == []


def _write_prior(tmp_path, rows, d='2026-09-08'):
    out = tmp_path / 'output'
    out.mkdir(exist_ok=True)
    (out / f'results_{d}.json').write_text(json.dumps({'results': rows}), encoding='utf-8')
    return str(out)


def test_check_warns_per_ticker_from_json_prior(tmp_path, caplog):
    out = _write_prior(tmp_path, _prior()[:4])
    with caplog.at_level(logging.WARNING, logger='analyze_stock'):
        lost = a._check_lost_sec_tickers(date(2026, 9, 9), MAP, results_dir=out)
    assert lost == ['LEG', 'ONLY']
    assert 'LEG: dropped from SEC ticker map (had SEC data on 2026-09-08)' in caplog.text


def test_check_reads_prior_from_store(tmp_path):
    from data.snapshot_store import sync_snapshot_file
    out = _write_prior(tmp_path, _prior()[:4])
    assert sync_snapshot_file(f'{out}/results_2026-09-08.json') is True
    assert a._check_lost_sec_tickers(date(2026, 9, 9), MAP, results_dir=out) == ['LEG', 'ONLY']


def test_check_no_prior_or_broken_prior_is_quiet(tmp_path, caplog):
    out = tmp_path / 'output'
    out.mkdir()
    assert a._check_lost_sec_tickers(date(2026, 9, 9), MAP, results_dir=str(out)) == []
    (out / 'results_2026-09-08.json').write_text('{broken', encoding='utf-8')
    with caplog.at_level(logging.WARNING, logger='analyze_stock'):
        assert a._check_lost_sec_tickers(date(2026, 9, 9), MAP, results_dir=str(out)) == []
    assert 'lost-SEC check failed' in caplog.text


def test_same_day_snapshot_is_not_the_prior(tmp_path):
    out = _write_prior(tmp_path, _prior()[:4], d='2026-09-09')
    assert a._check_lost_sec_tickers(date(2026, 9, 9), MAP, results_dir=out) == []


def test_quality_summary_reports_lost_sec(caplog):
    counter = SimpleNamespace(fabricated=0, total=0)
    with caplog.at_level(logging.WARNING, logger='analyze_stock'):
        a._run_quality_summary(0.04, 'fred', counter, None, lost_sec=['LEG', 'XOM'])
    assert 'RUN QUALITY: 2 ticker(s) lost SEC history' in caplog.text
    assert 'LEG, XOM' in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='analyze_stock'):
        a._run_quality_summary(0.04, 'fred', counter, None)
    assert 'lost SEC history' not in caplog.text
