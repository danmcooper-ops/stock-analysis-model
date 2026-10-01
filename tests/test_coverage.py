# tests/test_coverage.py
"""The snapshot coverage floor (data/coverage.py, scripts/check_coverage.py).

2026-09-30: 739 rows against 2,514 the day before. The database publish
refused it; the archive and the Pages publish did not ask, so the live
report was replaced by a snapshot missing 71% of the universe."""
import json
import logging
from types import SimpleNamespace

import pytest

from data.coverage import MIN_ROW_RATIO, coverage_check, format_coverage, read_coverage
from scripts import check_coverage


def test_floor_is_the_database_publish_floor():
    from data.db import publish
    assert publish.MIN_ROW_RATIO == MIN_ROW_RATIO == 0.7


def test_the_2026_09_30_night_is_degraded():
    cov = coverage_check(739, 2514, '2026-09-29')
    assert cov['degraded'] is True and cov['ratio'] == pytest.approx(0.294, abs=1e-3)
    assert format_coverage(cov) == \
        'COVERAGE degraded rows=739 prior_rows=2514 (2026-09-29) ratio=0.294 min=0.70'


def test_an_ordinary_night_is_ok():
    cov = coverage_check(2498, 2514, '2026-09-29')
    assert cov['degraded'] is False and cov['note'] is None
    assert format_coverage(cov).startswith('COVERAGE ok rows=2498')


@pytest.mark.parametrize('rows,prior,degraded', [(1760, 2514, False), (1759, 2514, True)])
def test_floor_edge(rows, prior, degraded):
    assert coverage_check(rows, prior)['degraded'] is degraded


def test_no_prior_or_explicit_tickers_never_degrade():
    cov = coverage_check(8, 0)
    assert cov['degraded'] is False and 'no prior snapshot' in cov['note'] and cov['ratio'] is None
    cov = coverage_check(8, 2514, '2026-09-29', applicable=False)
    assert cov['degraded'] is False and 'explicit ticker list' in cov['note']
    assert '[floor not applied: explicit ticker list]' in format_coverage(cov)
    assert format_coverage(None) == 'COVERAGE unknown (no provenance.coverage block)'


def _snapshot(tmp_path, name, coverage=None):
    meta = {'date': '2026-09-30', 'count': 1, 'provenance': {}, 'results': [{'ticker': 'A'}]}
    if coverage is not None:
        meta['provenance']['coverage'] = coverage
    p = tmp_path / name
    p.write_text(json.dumps(meta), encoding='utf-8')
    return str(p)


def test_read_coverage_round_trips_and_tolerates_old_snapshots(tmp_path):
    cov = coverage_check(739, 2514, '2026-09-29')
    assert read_coverage(_snapshot(tmp_path, 'results_2026-09-30.json', cov)) == cov
    assert read_coverage(_snapshot(tmp_path, 'results_2026-09-29.json')) is None


def test_cli_exit_codes(tmp_path, capsys):
    degraded = _snapshot(tmp_path, 'results_2026-09-30.json', coverage_check(739, 2514, '2026-09-29'))
    assert check_coverage.main([degraded]) == 4
    assert capsys.readouterr().out.startswith('COVERAGE degraded rows=739')
    ok = _snapshot(tmp_path, 'results_2026-10-01.json', coverage_check(2500, 2514, '2026-09-30'))
    assert check_coverage.main([ok]) == 0
    old = _snapshot(tmp_path, 'results_2026-09-29.json')
    assert check_coverage.main([old]) == 0
    assert 'COVERAGE unknown' in capsys.readouterr().out
    assert check_coverage.main([str(tmp_path / 'missing.json')]) == 1
    assert 'COVERAGE unreadable' in capsys.readouterr().out


def test_quality_summary_leads_with_a_degraded_coverage(caplog):
    import scripts.analyze_stock as a
    counter = SimpleNamespace(fabricated=0, total=0)
    cov = coverage_check(739, 2514, '2026-09-29')
    with caplog.at_level(logging.WARNING, logger='analyze_stock'):
        a._run_quality_summary(0.05, 'live', counter, None, coverage=cov)
    assert 'RUN QUALITY: coverage DEGRADED' in caplog.text
    assert '739 rows against 2514 on 2026-09-29' in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='analyze_stock'):
        a._run_quality_summary(0.05, 'live', counter, None, coverage=coverage_check(2500, 2514))
    assert 'coverage' not in caplog.text
