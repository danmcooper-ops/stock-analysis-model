"""Resumable runs (scripts/run_checkpoint.py) and --run-date parsing."""

import math
import os
import pickle
import types
from datetime import date, timedelta

import pytest

from scripts.analyze_stock import _parse_run_date
from scripts.run_checkpoint import RunCheckpoint, code_version, fingerprint


def _args(**kw):
    base = dict(universe='us', mcap_min=300e6, min_spread=0.0, input=None,
                validation=None, tickers=None, macro=True, prices_dir='output/prices',
                workers=4, no_screen_cache=False, no_resume=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


FP = fingerprint('2026-09-11', _args(), code='abc')


def test_progress_survives_a_restart(tmp_path):
    root = str(tmp_path)
    ck = RunCheckpoint('2026-09-11', FP, root=root)
    assert not ck.resumed
    ck.record_screened_out('TINY', 'quality')
    row = {'ticker': 'AAPL', 'dcf_sens_range': (90.0, 130.0), 'pe': float('nan'),
           'rolling_betas': {'1y': 1.1}}
    ck.record_phase2('AAPL', row)
    ck.record_phase2('ZTR', None, {'reason': 'ROIC or WACC unavailable'})

    again = RunCheckpoint('2026-09-11', FP, root=root)
    assert again.resumed
    assert again.screened_out('TINY') and not again.screened_out('AAPL')
    saved, skip = again.phase2_record('AAPL')
    assert skip is None and saved['dcf_sens_range'] == (90.0, 130.0)
    assert math.isnan(saved['pe'])
    assert again.phase2_record('ZTR') == (None, {'reason': 'ROIC or WACC unavailable'})
    assert again.phase2_record('MSFT') is None
    assert again.counts() == {'screened_out': 1, 'phase2': 2}


@pytest.mark.parametrize('changed', [
    dict(run_date='2026-09-12'),
    dict(args=_args(mcap_min=0)),
    dict(args=_args(tickers=['AAPL'])),
    dict(code='def'),
])
def test_different_date_options_or_code_discard_progress(tmp_path, changed):
    root = str(tmp_path)
    ck = RunCheckpoint('2026-09-11', FP, root=root)
    ck.record_phase2('AAPL', {'ticker': 'AAPL'})
    fp = fingerprint(changed.get('run_date', '2026-09-11'), changed.get('args', _args()),
                     code=changed.get('code', 'abc'))
    other = RunCheckpoint('2026-09-11', fp, root=root)
    assert not other.resumed and other.phase2_record('AAPL') is None
    assert not os.path.exists(os.path.join(root, '2026-09-11', 'phase2.pkl'))


def test_speed_only_options_do_not_invalidate(tmp_path):
    assert fingerprint('d', _args(workers=1, no_screen_cache=True), code='x') == \
        fingerprint('d', _args(workers=8), code='x')


def test_torn_records_are_ignored(tmp_path):
    root = str(tmp_path)
    ck = RunCheckpoint('2026-09-11', FP, root=root)
    ck.record_phase2('AAPL', {'ticker': 'AAPL'})
    ck.record_screened_out('TINY')
    d = os.path.join(root, '2026-09-11')
    blob = pickle.dumps(('MSFT', {'ticker': 'MSFT'}, None))
    with open(os.path.join(d, 'phase2.pkl'), 'ab') as f:
        f.write(blob[:len(blob) // 2])          # killed mid-write
    with open(os.path.join(d, 'phase1.jsonl'), 'a', encoding='utf-8') as f:
        f.write('{"t": "HAL')                    # killed mid-line
    again = RunCheckpoint('2026-09-11', FP, root=root)
    assert again.phase2_record('AAPL') is not None and again.phase2_record('MSFT') is None
    assert again.screened_out('TINY') and not again.screened_out('HAL')


def test_unserialisable_row_is_skipped_not_raised(tmp_path):
    ck = RunCheckpoint('2026-09-11', FP, root=str(tmp_path))
    ck.record_phase2('BAD', {'fn': lambda: 1})
    assert ck.phase2_record('BAD') is None
    ck.record_phase2('OK', {'ticker': 'OK'})
    assert RunCheckpoint('2026-09-11', FP, root=str(tmp_path)).phase2_record('OK') is not None


def test_unwritable_root_never_raises(tmp_path):
    blocker = tmp_path / 'file'
    blocker.write_text('x', encoding='utf-8')
    ck = RunCheckpoint('2026-09-11', FP, root=str(blocker))
    ck.record_screened_out('TINY')
    ck.record_phase2('AAPL', {'ticker': 'AAPL'})
    assert ck.counts() == {'screened_out': 0, 'phase2': 0}


def test_clear_removes_the_date_and_empty_root(tmp_path):
    root = tmp_path / '.checkpoint'
    ck = RunCheckpoint('2026-09-11', FP, root=str(root))
    ck.record_phase2('AAPL', {'ticker': 'AAPL'})
    ck.clear()
    assert not root.exists()


def test_code_version_tracks_source_changes(tmp_path):
    pkg = tmp_path / 'scripts'
    pkg.mkdir()
    (pkg / 'a.py').write_text('x = 1\n', encoding='utf-8')
    v1 = code_version(root=str(tmp_path), dirs=('scripts',))
    (pkg / 'a.py').write_text('x = 2\n', encoding='utf-8')
    assert code_version(root=str(tmp_path), dirs=('scripts',)) != v1


def test_parse_run_date():
    assert _parse_run_date(None) == date.today()
    assert _parse_run_date('2026-09-11') == date(2026, 9, 11)
    with pytest.raises(ValueError, match='YYYY-MM-DD'):
        _parse_run_date('09/11/2026')
    with pytest.raises(ValueError, match='future'):
        _parse_run_date((date.today() + timedelta(days=1)).isoformat())
