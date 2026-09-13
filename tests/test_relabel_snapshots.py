"""scripts/relabel_snapshots.py: weekend-dated snapshots and duplicates."""

import json

from data.snapshot_store import list_snapshot_files, read_snapshot
from scripts import relabel_snapshots as rs


def _snap(d, day, rows=None, prov=None, gz=False):
    payload = {'date': day, 'risk_free_rate': 0.04, 'count': 1,
               'results': rows or [{'ticker': 'AAA', 'price': 10.0, 'pe': float('nan')}]}
    if prov is not None:
        payload['provenance'] = prov
    name = f'results_{day}.json' + ('.gz' if gz else '')
    path = d / name
    if gz:
        from data.snapshot_store import write_snapshot_file
        write_snapshot_file(str(path), payload)
    else:
        path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_relabel_rewrites_date_and_records_origin(tmp_path):
    arch, out = tmp_path / 'arch', tmp_path / 'output'
    arch.mkdir(), out.mkdir()
    _snap(arch, '2026-09-05', prov={'run_started_at': '2026-09-05T11:49:15+00:00'})
    _snap(out, '2026-09-05', rows=[{'ticker': 'AAA', 'price': 99.0}])   # re-rendered later
    assert rs.main(['--dest', str(arch), '--results-dir', str(out),
                    '--map', '2026-09-05=2026-09-04']) == 0
    assert [d for d, _ in list_snapshot_files(str(arch))] == ['2026-09-04']
    new = read_snapshot(str(arch / 'results_2026-09-04.json.gz'))
    assert new['date'] == '2026-09-04'
    assert new['provenance']['relabeled_from'] == '2026-09-05'
    assert new['provenance']['run_started_at'] == '2026-09-05T11:49:15+00:00'
    # The local copy is rewritten from the ARCHIVE content, not its own.
    local = read_snapshot(str(out / 'results_2026-09-04.json'))
    assert local == new and local['results'][0]['price'] == 10.0
    assert not (out / 'results_2026-09-05.json').exists()


def test_retire_moves_duplicates_out_of_every_listing(tmp_path):
    arch, out = tmp_path / 'arch', tmp_path / 'output'
    arch.mkdir(), out.mkdir()
    _snap(arch, '2026-07-02'), _snap(arch, '2026-07-03')
    _snap(out, '2026-07-03')
    assert rs.main(['--dest', str(arch), '--results-dir', str(out), '--retire', '2026-07-03']) == 0
    assert [d for d, _ in list_snapshot_files(str(arch))] == ['2026-07-02']
    assert (arch / 'retired' / 'results_2026-07-03.json.gz').exists()
    assert read_snapshot(str(arch / 'retired' / 'results_2026-07-03.json.gz'))['date'] == '2026-07-03'
    assert list_snapshot_files(str(out)) == []
    assert (out / 'retired' / 'results_2026-07-03.json').exists()


def test_refusals(tmp_path, capsys):
    arch = tmp_path / 'arch'
    arch.mkdir()
    _snap(arch, '2026-09-04'), _snap(arch, '2026-09-05'), _snap(arch, '2026-09-08')
    # target already archived
    assert rs.main(['--dest', str(arch), '--map', '2026-09-05=2026-09-04']) == 1
    # source is a trading day
    assert rs.main(['--dest', str(arch), '--map', '2026-09-08=2026-09-04']) == 1
    # target is not a trading day (Labor Day)
    assert rs.main(['--dest', str(arch), '--map', '2026-09-05=2026-09-07']) == 1
    # missing source, and a date used twice
    assert rs.main(['--dest', str(arch), '--retire', '2026-05-16']) == 1
    assert rs.main(['--dest', str(arch), '--retire', '2026-09-05', '--map', '2026-09-05=2026-09-03']) == 1
    assert 'REFUSED' in capsys.readouterr().out
    assert sorted(p.name for p in arch.iterdir()) == [
        'results_2026-09-04.json', 'results_2026-09-05.json', 'results_2026-09-08.json']


def test_dry_run_writes_nothing(tmp_path):
    arch = tmp_path / 'arch'
    arch.mkdir()
    _snap(arch, '2026-08-30'), _snap(arch, '2026-08-15', gz=True)
    before = sorted(p.name for p in arch.iterdir())
    assert rs.main(['--dest', str(arch), '--map', '2026-08-30=2026-08-28',
                    '--retire', '2026-08-15', '--dry-run']) == 0
    assert sorted(p.name for p in arch.iterdir()) == before


def test_gz_source_and_nan_survive(tmp_path):
    import math
    arch = tmp_path / 'arch'
    arch.mkdir()
    _snap(arch, '2026-08-15', gz=True)
    assert rs.main(['--dest', str(arch), '--map', '2026-08-15=2026-08-14']) == 0
    row = read_snapshot(str(arch / 'results_2026-08-14.json.gz'))['results'][0]
    assert math.isnan(row['pe'])
    assert not (arch / 'results_2026-08-15.json.gz').exists()
