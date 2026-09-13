# tests/test_snapshot_blobs.py
"""The .gz archive form stores ``edgar_history`` once per distinct value.

Covers the content-addressed blob store in data/snapshot_store.py, the
verified archive/migration paths in scripts/archive_snapshot.py, relabeling
into ``retired/``, and staging blobs out of a blob-less partial clone
(scripts/stage_snapshot_blobs.py).
"""

import copy
import gzip
import hashlib
import json
import os
import shutil
import subprocess

import pytest

import data.snapshot_store as ss
from data.snapshot_store import (
    BLOB_DIRNAME, clear_blob_cache, list_snapshot_files, load_snapshot_file,
    read_snapshot, snapshot_blob_refs, write_snapshot_file,
)
from scripts import archive_snapshot as arch
from scripts import relabel_snapshots as rs

HIST_A = {'revenue_history': {'2023': 1, '2024': 2}, 'years_available': 2,
          'margin': float('nan')}
HIST_B = {'revenue_history': {'2023': 5, '2024': 7}, 'years_available': 2}


def _snap(day='2026-09-03', hist_b=HIST_B):
    return {
        'date': day, 'risk_free_rate': 0.04, 'count': 3,
        'results': [
            {'ticker': 'AAA', 'rating': 'BUY', 'edgar_history': copy.deepcopy(HIST_A),
             'pe': float('inf')},
            {'ticker': 'BBB', 'rating': 'PASS', 'edgar_history': copy.deepcopy(hist_b)},
            {'ticker': 'CCC', 'rating': 'PASS', 'edgar_history': None},
        ],
    }


def _blob_files(root):
    return sorted(os.path.relpath(os.path.join(d, f), root)
                  for d, _, fs in os.walk(root) for f in fs)


def _raw_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


def _same(a, b):
    """Equality that treats NaN as equal to itself (canonical encoding)."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_blob_cache()
    yield
    clear_blob_cache()


# --- write / read -----------------------------------------------------------

def test_gz_write_stores_references_and_reads_back_whole(tmp_path):
    p = tmp_path / 'results_2026-09-03.json.gz'
    write_snapshot_file(str(p), _snap())

    rows = _raw_gz(p)['results']
    assert set(rows[0]['edgar_history']) == {'$blob'}
    assert set(rows[1]['edgar_history']) == {'$blob'}
    assert rows[2]['edgar_history'] is None
    assert len(_blob_files(tmp_path / BLOB_DIRNAME)) == 2
    assert _same(read_snapshot(str(p)), _snap())
    # key order survives, so re-serializing reproduces the plain encoding
    assert list(read_snapshot(str(p))['results'][0]) == list(_snap()['results'][0])


def test_blob_name_is_the_hash_of_its_content(tmp_path):
    p = tmp_path / 'results_2026-09-03.json.gz'
    write_snapshot_file(str(p), _snap())
    for key, sha in snapshot_blob_refs(str(p)):
        fp = tmp_path / BLOB_DIRNAME / key / sha[:2] / f'{sha}.json.gz'
        with gzip.open(fp, 'rb') as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == sha


def test_plain_write_stays_self_contained(tmp_path):
    p = tmp_path / 'results_2026-09-03.json'
    write_snapshot_file(str(p), _snap())
    assert not (tmp_path / BLOB_DIRNAME).exists()
    assert b'$blob' not in p.read_bytes()


def test_write_does_not_modify_the_callers_data(tmp_path):
    data = _snap()
    write_snapshot_file(str(tmp_path / 'results_2026-09-03.json.gz'), data)
    assert _same(data, _snap())


def test_gz_write_is_deterministic(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(), b.mkdir()
    write_snapshot_file(str(a / 'results_2026-09-03.json.gz'), _snap())
    write_snapshot_file(str(b / 'results_2026-09-03.json.gz'), _snap())
    assert (a / 'results_2026-09-03.json.gz').read_bytes() == \
        (b / 'results_2026-09-03.json.gz').read_bytes()
    fa, fb = _blob_files(a / BLOB_DIRNAME), _blob_files(b / BLOB_DIRNAME)
    assert fa == fb
    for f in fa:
        assert (a / BLOB_DIRNAME / f).read_bytes() == (b / BLOB_DIRNAME / f).read_bytes()


def test_unchanged_history_is_stored_once_across_snapshots(tmp_path):
    write_snapshot_file(str(tmp_path / 'results_2026-09-02.json.gz'), _snap('2026-09-02'))
    assert len(_blob_files(tmp_path / BLOB_DIRNAME)) == 2
    write_snapshot_file(str(tmp_path / 'results_2026-09-03.json.gz'), _snap('2026-09-03'))
    assert len(_blob_files(tmp_path / BLOB_DIRNAME)) == 2       # nothing new
    changed = dict(HIST_B, years_available=3)
    write_snapshot_file(str(tmp_path / 'results_2026-09-04.json.gz'),
                        _snap('2026-09-04', hist_b=changed))
    assert len(_blob_files(tmp_path / BLOB_DIRNAME)) == 3       # only BBB's new value


def test_discovery_helpers_return_rehydrated_rows(tmp_path):
    write_snapshot_file(str(tmp_path / 'results_2026-09-02.json.gz'), _snap('2026-09-02'))
    write_snapshot_file(str(tmp_path / 'results_2026-09-03.json'), _snap('2026-09-03'))
    found = list_snapshot_files(str(tmp_path))
    assert [d for d, _ in found] == ['2026-09-02', '2026-09-03']
    for _, path in found:
        _, rows = load_snapshot_file(path)
        assert rows[1]['edgar_history'] == HIST_B


def test_rows_never_share_history_objects(tmp_path):
    """The cache holds text; a caller mutating one row must not leak."""
    p = str(tmp_path / 'results_2026-09-03.json.gz')
    write_snapshot_file(p, _snap())
    first = read_snapshot(p)
    first['results'][1]['edgar_history']['years_available'] = 99
    assert read_snapshot(p)['results'][1]['edgar_history']['years_available'] == 2


def test_missing_blob_fails_loudly(tmp_path):
    p = tmp_path / 'results_2026-09-03.json.gz'
    write_snapshot_file(str(p), _snap())
    lone = tmp_path / 'copied' / 'alone'        # neither it nor its parent has blobs/
    lone.mkdir(parents=True)
    shutil.copy(p, lone / p.name)
    with pytest.raises(FileNotFoundError, match='not found'):
        read_snapshot(str(lone / p.name), blob_cache=False)


def test_corrupt_blob_fails_loudly(tmp_path):
    p = tmp_path / 'results_2026-09-03.json.gz'
    write_snapshot_file(str(p), _snap())
    victim = tmp_path / BLOB_DIRNAME / _blob_files(tmp_path / BLOB_DIRNAME)[0]
    with open(victim, 'wb') as fh:
        with gzip.GzipFile(fileobj=fh, mode='wb', mtime=0) as gz:
            gz.write(b'{"revenue_history":{}}')
    with pytest.raises(FileNotFoundError, match='corrupt'):
        read_snapshot(str(p), blob_cache=False)


def test_rewrite_repairs_a_corrupt_blob(tmp_path):
    p = tmp_path / 'results_2026-09-03.json.gz'
    write_snapshot_file(str(p), _snap())
    victim = tmp_path / BLOB_DIRNAME / _blob_files(tmp_path / BLOB_DIRNAME)[0]
    victim.write_bytes(b'garbage')
    write_snapshot_file(str(p), _snap())
    assert _same(read_snapshot(str(p), blob_cache=False), _snap())


def test_retired_snapshot_resolves_against_the_archive_root(tmp_path):
    (tmp_path / 'retired').mkdir()
    write_snapshot_file(str(tmp_path / 'results_2026-09-02.json.gz'), _snap('2026-09-02'))
    write_snapshot_file(str(tmp_path / 'retired' / 'results_2026-07-03.json.gz'),
                        _snap('2026-07-03'))
    assert not (tmp_path / 'retired' / BLOB_DIRNAME).exists()
    assert _same(read_snapshot(str(tmp_path / 'retired' / 'results_2026-07-03.json.gz')),
                 _snap('2026-07-03'))


# --- archive_snapshot -------------------------------------------------------

@pytest.mark.parametrize('seps', [(',', ':'), (', ', ': ')])
def test_archive_externalizes_and_reproduces_source_bytes(tmp_path, seps):
    """Both encodings real snapshots use; NaN/Infinity included."""
    out, dest = tmp_path / 'output', tmp_path / 'archive'
    out.mkdir(), dest.mkdir()
    src = out / 'results_2026-09-03.json'
    src.write_bytes(json.dumps(_snap(), separators=seps).encode())
    stats = {}

    path, raw, gz, level = arch.archive_snapshot(str(src), str(dest), stats_out=stats)

    assert stats == {'refs': 2, 'new_blobs': 2, 'new_bytes': stats['new_bytes'],
                     'verbatim': False}
    assert raw == src.stat().st_size and level == 'ok'
    back = read_snapshot(path, blob_cache=False)
    assert json.dumps(back, separators=seps).encode() == src.read_bytes()
    assert [f for f in os.listdir(dest) if '.tmp.' in f] == []
    assert sorted(arch.blob_paths(path, str(dest))) == sorted(
        os.path.join(BLOB_DIRNAME, f) for f in _blob_files(dest / BLOB_DIRNAME))


def test_archive_copies_a_non_canonical_source_verbatim(tmp_path):
    out, dest = tmp_path / 'output', tmp_path / 'archive'
    out.mkdir(), dest.mkdir()
    src = out / 'results_2026-09-03.json'
    src.write_text(json.dumps(_snap(), indent=1), encoding='utf-8')
    stats = {}
    path = arch.archive_snapshot(str(src), str(dest), stats_out=stats)[0]
    assert stats['verbatim'] is True
    with gzip.open(path, 'rb') as fh:
        assert fh.read() == src.read_bytes()


def test_archive_fails_when_a_written_blob_is_corrupt(tmp_path, monkeypatch):
    out, dest = tmp_path / 'output', tmp_path / 'archive'
    out.mkdir(), dest.mkdir()
    src = out / 'results_2026-09-03.json'
    write_snapshot_file(str(src), _snap())

    def bad_blob_write(path, payload):
        with open(path, 'wb') as fh:
            with gzip.GzipFile(fileobj=fh, mode='wb', mtime=0) as gz:
                gz.write(payload[:-5])

    monkeypatch.setattr(ss, '_gzip_bytes_to', bad_blob_write)
    with pytest.raises(OSError):
        arch.archive_snapshot(str(src), str(dest))
    assert not (dest / 'results_2026-09-03.json.gz').exists()
    assert [f for f in os.listdir(dest) if '.tmp.' in f] == []


def test_main_writes_the_blob_list(tmp_path, capsys):
    out, dest = tmp_path / 'output', tmp_path / 'archive'
    out.mkdir(), dest.mkdir()
    write_snapshot_file(str(out / 'results_2026-09-03.json'), _snap())
    listing = tmp_path / 'blobs.txt'
    assert arch.main(['--results-dir', str(out), '--dest', str(dest),
                      '--list-blobs', str(listing)]) == arch.EXIT_OK
    lines = listing.read_text(encoding='utf-8').split()
    assert len(lines) == 2 and all((dest / p).exists() for p in lines)
    assert '2 history refs' in capsys.readouterr().out


def test_externalize_dir_migrates_and_is_idempotent(tmp_path):
    archive = tmp_path / 'archive'
    (archive / 'retired').mkdir(parents=True)
    # the pre-change archive: history inline in .gz (json.dump defaults) and plain
    for p, day in ((archive / 'results_2026-09-01.json.gz', '2026-09-01'),
                   (archive / 'retired' / 'results_2026-07-03.json.gz', '2026-07-03')):
        with gzip.open(p, 'wb') as fh:
            fh.write(json.dumps(_snap(day)).encode())
    (archive / 'results_2026-09-02.json').write_text(json.dumps(_snap('2026-09-02')),
                                                    encoding='utf-8')
    originals = {d: _snap(d) for d in ('2026-09-01', '2026-09-02', '2026-07-03')}

    assert arch.externalize_dir(str(archive), log=lambda *_: None) == (2, 0, 0)
    assert (archive / 'results_2026-09-02.json').exists()            # plain not included
    assert arch.externalize_dir(str(archive), include_plain=True, remove_plain=True,
                                log=lambda *_: None) == (1, 2, 0)
    assert not (archive / 'results_2026-09-02.json').exists()
    assert len(_blob_files(archive / BLOB_DIRNAME)) == 2               # shared by all three
    for day, path in list_snapshot_files(str(archive)):
        assert _same(read_snapshot(path, blob_cache=False), originals[day])
    assert _same(read_snapshot(str(archive / 'retired' / 'results_2026-07-03.json.gz')),
                 originals['2026-07-03'])
    assert arch.externalize_dir(str(archive), include_plain=True,
                                log=lambda *_: None) == (0, 3, 0)


# --- relabel ----------------------------------------------------------------

def test_relabel_and_retire_with_blobs(tmp_path):
    archive = tmp_path / 'archive'
    archive.mkdir()
    write_snapshot_file(str(archive / 'results_2026-09-05.json.gz'), _snap('2026-09-05'))
    write_snapshot_file(str(archive / 'results_2026-07-03.json.gz'), _snap('2026-07-03'))
    assert rs.main(['--dest', str(archive), '--map', '2026-09-05=2026-09-04',
                    '--retire', '2026-07-03']) == 0
    new = read_snapshot(str(archive / 'results_2026-09-04.json.gz'), blob_cache=False)
    assert new['results'][1]['edgar_history'] == HIST_B
    retired = archive / 'retired' / 'results_2026-07-03.json.gz'
    assert _same(read_snapshot(str(retired), blob_cache=False)['results'],
                 _snap()['results'])
    assert not (archive / 'retired' / BLOB_DIRNAME).exists()


# --- staging from a blob-less partial clone ---------------------------------

def _git(*args, cwd=None):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which('git') is None, reason='git not installed')
def test_stage_blobs_from_a_blobless_partial_clone(tmp_path):
    from scripts.stage_snapshot_blobs import stage_blobs

    origin = tmp_path / 'origin'
    origin.mkdir()
    _git('init', '-q', '-b', 'data/snapshots', cwd=origin)
    _git('config', 'uploadpack.allowFilter', 'true', cwd=origin)
    _git('config', 'uploadpack.allowAnySHA1InWant', 'true', cwd=origin)
    for day in ('2026-09-02', '2026-09-03'):
        write_snapshot_file(str(origin / f'results_{day}.json.gz'), _snap(day))
    _git('add', '.', cwd=origin)
    _git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'snap', cwd=origin)

    clone = tmp_path / 'clone'
    _git('clone', '-q', '--filter=blob:none', '--depth', '1', '--no-checkout',
         '--single-branch', '-b', 'data/snapshots', f'file://{origin}', str(clone))
    out = tmp_path / 'output'
    out.mkdir()
    staged = []
    for day in ('2026-09-02', '2026-09-03'):
        nm = f'results_{day}.json.gz'
        with open(out / nm, 'wb') as fh:
            subprocess.run(['git', '-C', str(clone), 'show', f'HEAD:{nm}'],
                           stdout=fh, check=True)
        staged.append(str(out / nm))

    assert stage_blobs(str(clone), str(out), staged, log=lambda *_: None) == 2
    for path in staged:
        assert _same(read_snapshot(path, blob_cache=False)['results'], _snap()['results'])
    # already present and intact: nothing to fetch the second time
    assert stage_blobs(str(clone), str(out), staged, log=lambda *_: None) == 0
