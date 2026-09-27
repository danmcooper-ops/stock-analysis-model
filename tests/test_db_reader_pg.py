"""DbStore against a live database: parity with the DuckDB store (P4).

The same snapshots are ingested into a DuckDB SnapshotStore and published to
Postgres; every reader must return the same answer from both. Then the real
call sites run with ``SNAPSHOT_STORE_BACKEND=postgres``.

Runs only with ``TEST_DATABASE_URL``; test data uses November 2031 and tickers
starting ``ZZR`` and is deleted afterwards.
"""
import math
import os

import pytest

pytestmark = pytest.mark.pg

DSN = os.environ.get('TEST_DATABASE_URL')
if not DSN:
    pytest.skip('TEST_DATABASE_URL not set', allow_module_level=True)

psycopg = pytest.importorskip('psycopg')

from data.db import reader  # noqa: E402
from data.db.codec import values_equal  # noqa: E402
from data.db.columns import COLUMNS  # noqa: E402
from data.db.connect import connect  # noqa: E402
from data.db.publish import DirectTransport, build_load, publish  # noqa: E402
from data.snapshot_store import SnapshotStore, write_snapshot_file  # noqa: E402

DATES = ['2031-11-10', '2031-11-11', '2031-11-12', '2031-11-13']
NUM = next(k for k, t in COLUMNS.items() if t == 'double precision' and k not in ('mos',))
BIG = next(k for k, t in COLUMNS.items() if t == 'bigint')
RATINGS = {   # ticker -> rating per date (None = no rating that day, '-' = ticker absent)
    'ZZRA': ['BUY', 'BUY', 'HOLD', 'HOLD'],
    'ZZRB': ['PASS', None, 'PASS', 'BUY'],
    'ZZRC': ['HOLD', 'HOLD', 'HOLD', '-'],          # gone on the last day: last_known_rows fallback
    'ZZRD': ['-', 'LEAN BUY', '', 'LEAN BUY'],
}


def _snapshot(i, d):
    rows = []
    for t, seq in RATINGS.items():
        if seq[i] == '-':
            continue
        rows.append({'ticker': t, 'rating': seq[i], 'mos': 0.03932028370017462 * (i + 1), NUM: math.nan if i % 2 else -math.inf,
                     BIG: 10 ** 12 + i, 'shares_out': 1.5e9, 'mcap': 2.25e11, 'data_source': 'sec_xbrl',
                     'roic_by_year': {'2030': 0.1 * i, '2031': math.nan},
                     '_rating_cap_reasons': ['a\x00b', 'liquidity'], 'p4_new_key': {'$nf': 'x'},
                     'edgar_history': {'years_available': 10 + i, 'operating_income_history': [1.0, None]}})
    return {'date': d, 'risk_free_rate': 0.0425, 'risk_free_rate_source': 'fred', 'count': len(rows),
            'provenance': {'timings': {'phase1': 12.5}}, 'results': rows}


def _cleanup(con):
    with con.transaction():
        for sql in ("DELETE FROM core.rating_changes WHERE run_date >= '2031-11-01' AND run_date < '2031-12-01'",
                    "DELETE FROM core.latest_results WHERE ticker_id IN "
                    "(SELECT ticker_id FROM core.tickers WHERE ticker LIKE 'ZZR%')",
                    "DELETE FROM core.results WHERE run_date >= '2031-11-01' AND run_date < '2031-12-01'",
                    "DELETE FROM core.runs WHERE run_date >= '2031-11-01' AND run_date < '2031-12-01'",
                    "DELETE FROM core.tickers WHERE ticker LIKE 'ZZR%'"):
            con.execute(sql)


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """Snapshots on disk, in a DuckDB store and published to Postgres."""
    out = tmp_path_factory.mktemp('snapshots')
    con = connect(DSN, autocommit=True)
    _cleanup(con)
    duck_path = str(out / 'snapshots.duckdb')
    with SnapshotStore(duck_path) as duck:
        for i, d in enumerate(DATES):
            data = _snapshot(i, d)
            path = str(out / f'results_{d}.json')
            write_snapshot_file(path, data)
            duck.ingest_json(path, replace=True)
            publish(build_load(data, d), DirectTransport(con), min_row_ratio=0)
    duck = SnapshotStore(duck_path, read_only=True)
    db = reader.DbStore(DirectTransport(con), results_dir=str(out))
    yield out, duck, db, con
    duck.close()
    _cleanup(con)
    con.close()


def _in_window(dates):
    return [d for d in dates if d.startswith('2031-11')]


def _duck_view(v, nested=False):
    """*v* as the DuckDB store keeps it: its JSON columns null a NaN nested in
    a dict or list (``_json_safe``), where the database keeps it as the file has it."""
    if isinstance(v, float) and nested and math.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: _duck_view(x, True) for k, x in v.items()}
    if isinstance(v, list):
        return [_duck_view(x, True) for x in v]
    return v


def _rows_equal(a, b, view=lambda v: v):
    assert [r['ticker'] for r in a] == [r['ticker'] for r in b]
    for ra, rb in zip(a, b, strict=True):
        diff = [k for k in set(ra) | set(rb) if not values_equal(ra.get(k), view(rb.get(k)))]
        assert diff == [], (ra['ticker'], diff)


def test_dates_match(world):
    _, duck, db, _ = world
    assert _in_window(db.dates()) == duck.dates() == DATES
    assert _in_window(db.dates(before=DATES[2])) == DATES[:2]
    assert all(db.has_date(d) for d in DATES) and not db.has_date('2031-11-14')


@pytest.mark.parametrize('columns', [
    ['ticker', 'shares_out', 'mcap'], ['ticker', 'data_source'], ['rating'],
    ['rating', 'mos', NUM, BIG], ['rating', 'no_such_key', 'roic_by_year', '_rating_cap_reasons', 'p4_new_key'],
])
def test_rows_match(world, columns):
    _, duck, db, _ = world
    for i, d in enumerate(DATES):
        got = db.rows(d, columns)
        _rows_equal(duck.rows(d, columns), got, view=_duck_view)
        by_file = sorted(_snapshot(i, d)['results'], key=lambda r: r['ticker'])     # the source of truth
        _rows_equal([{'ticker': r['ticker'], **{c: r.get(c) for c in columns if c != 'ticker'}} for r in by_file], got)


@pytest.mark.parametrize('before,lookback', [('2031-11-14', 4), ('2031-11-14', 2), ('2031-11-12', 2), ('2031-11-11', 1)])
def test_last_known_rows_match(world, before, lookback):
    _, duck, db, _ = world
    cols = ['rating', 'mos', NUM, '_rating_cap_reasons']
    pd_, rd, nd = duck.last_known_rows(before, cols, max_lookback=lookback)
    pp, rp, npp = db.last_known_rows(before, cols, max_lookback=lookback)
    assert (pd_, nd) == (pp, npp)
    assert rd.keys() == rp.keys()
    for t in rd:
        assert all(values_equal(rd[t][c], _duck_view(rp[t][c])) for c in cols), t


@pytest.mark.parametrize('before', [None, '2031-11-12', '2031-11-14'])
def test_rating_history_matches(world, before):
    _, duck, db, _ = world
    got = {t: h for t, h in db.rating_history(before=before).items() if t.startswith('ZZR')}
    assert got == duck.rating_history(before=before)


def test_run_meta_matches(world):
    _, duck, db, _ = world
    for d in DATES:
        a, b = duck.run_meta(d), db.run_meta(d)
        assert sorted(k for k in set(a) | set(b) if not values_equal(a.get(k), b.get(k))) == []


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv('SNAPSHOT_STORE_BACKEND', 'postgres')
    monkeypatch.setenv('SUPABASE_DB_URL', DSN)
    for v in ('SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'SUPABASE_READER_URL'):
        monkeypatch.delenv(v, raising=False)
    reader.reset_backend_state()
    yield
    reader.reset_backend_state()


def test_call_sites_read_the_database(world, backend, monkeypatch):
    out, duck, _, _ = world
    opened = []
    real = reader.open_db_store
    monkeypatch.setattr(reader, 'open_db_store', lambda *a: opened.append(1) or real(*a))
    from scripts.analyze_stock import _load_carry_forward_rows
    from scripts.report_html import _prev_ratings_from_store, _rating_history_from_store
    from scripts.track_portfolio import _load_prior_by_ticker
    d = DATES[-1]
    carry = _load_carry_forward_rows(d, str(out / f'results_{d}.json'))
    _rows_equal(sorted(carry, key=lambda r: r['ticker']), duck.rows(d, ['ticker', 'shares_out', 'mcap']))
    prior = _load_prior_by_ticker(str(out / f'results_{d}.json'))
    assert {t: r['rating'] for t, r in prior.items()} == {'ZZRA': 'HOLD', 'ZZRB': 'BUY', 'ZZRD': 'LEAN BUY'}
    primary, drivers = _prev_ratings_from_store(str(out), '2031-11-14', list(reversed(DATES)), [])
    assert primary == DATES[-1] and drivers['ZZRC']['rating'] == 'HOLD'      # via the fallback look-back
    hist = _rating_history_from_store(str(out), '2031-11-14', DATES[2:])      # fewer files than the database
    assert {t: h for t, h in hist.items() if t.startswith('ZZR')} == duck.rating_history(before='2031-11-14')
    assert len(opened) >= 4


def test_a_rewritten_file_falls_back_to_the_json(world, backend):
    out, _, _, _ = world
    from scripts.track_portfolio import _load_prior_by_ticker
    d = DATES[0]
    path = str(out / f'results_{d}.json')
    data = _snapshot(0, d)
    data['results'][0]['rating'] = 'PASS'                                     # rewritten after publishing
    write_snapshot_file(path, data)
    try:
        store = reader.open_db_store(str(out))
        assert store is not None and not store.has_date(d)
        assert _load_prior_by_ticker(path)['ZZRA']['rating'] == 'PASS'
    finally:
        write_snapshot_file(path, _snapshot(0, d))
