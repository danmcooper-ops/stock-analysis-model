"""The snapshot's provenance.scoring describes the code the run loaded.

analyze_stock stamps it hours after starting; on 2026-10-23 a run that
started on 9357ad79c was stamped 386b0005d, merged while it ran. _main now
takes the fingerprint at startup and _write_outputs writes that one.
"""
import datetime as dt

import pytest

import scripts.analyze_stock as az


class _Stop(Exception):
    pass


class _Prov:
    def run_block(self, results):
        return {}


def _capture_snapshot(monkeypatch, tmp_path):
    seen = {}

    def fake_write(path, data):
        seen['meta'] = data
        raise _Stop

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(az, 'write_snapshot_file', fake_write)
    return seen


def _write(**kw):
    az._write_outputs([], dt.date(2026, 10, 23), _Prov(), 0.05, 'live',
                      None, None, None, 'prices', **kw)


def test_write_outputs_stamps_the_startup_fingerprint(monkeypatch, tmp_path):
    seen = _capture_snapshot(monkeypatch, tmp_path)
    monkeypatch.setattr(az, '_scoring_fingerprint_now',
                        lambda: pytest.fail('must not re-fingerprint at write time'))
    fp = {'params_hash': 'aaaaaaaaaaaa', 'git_sha': 'start', 'git_dirty': False}
    with pytest.raises(_Stop):
        _write(scoring_fp=fp)
    assert seen['meta']['provenance']['scoring'] == fp


def test_write_outputs_without_one_takes_it_then(monkeypatch, tmp_path):
    seen = _capture_snapshot(monkeypatch, tmp_path)
    fp = {'params_hash': 'bbbbbbbbbbbb', 'git_sha': 'now', 'git_dirty': None}
    monkeypatch.setattr(az, '_scoring_fingerprint_now', lambda: fp)
    with pytest.raises(_Stop):
        _write()
    assert seen['meta']['provenance']['scoring'] == fp


def test_main_fingerprints_before_anything_else(monkeypatch):
    order = []
    monkeypatch.setattr(az, '_scoring_fingerprint_now', lambda: order.append('fp') or {})

    def setup():
        order.append('setup')
        raise _Stop
    monkeypatch.setattr(az, '_run_setup', setup)
    with pytest.raises(_Stop):
        az._main()
    assert order == ['fp', 'setup']


def test_failed_fingerprint_is_logged_not_raised(monkeypatch, caplog):
    import scripts.param_set as ps

    def boom():
        raise RuntimeError('no git')
    monkeypatch.setattr(ps, 'scoring_fingerprint', boom)
    assert az._scoring_fingerprint_now() is None
    assert 'scoring fingerprint failed' in caplog.text
