"""P6 cutover: the nightly database check, the readiness gate, the DB_PRIMARY
switch in run.sh, and the restore drill (``pg`` tests at the end)."""
import datetime as dt
import json
import os
import re
import shutil
import subprocess

import pytest

from scripts import db_night_check as nc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_SH = os.path.join(REPO, 'scheduled-tasks', 'cloud-daily-stock-analysis', 'run.sh')


# --- rating-history parity ----------------------------------------------------

def test_identical_histories_match():
    h = {'AAA': [['2026-09-01', 'BUY'], ['2026-09-10', 'HOLD']], 'BBB': [['2026-09-01', 'PASS']]}
    assert nc.rating_history_parity(h, h, '2026-09-20') == \
        ([], {'tickers': 2, 'common_start': '2026-09-01', 'only_before_start': 0})


def test_sources_starting_on_different_days_compare_from_the_later():
    """The database holds every run; the cache began later. A since-date on
    or before the later start is "before" on both sides."""
    db = {'AAA': [['2026-04-20', 'BUY']],                           # BUY all along
          'BBB': [['2026-04-20', 'HOLD'], ['2026-09-15', 'BUY']],
          'OLD': [['2026-05-01', 'PASS']]}                           # left the universe before the cache
    cache = {'AAA': [['2026-09-01', 'BUY']],
             'BBB': [['2026-09-01', 'HOLD'], ['2026-09-15', 'BUY']]}
    mismatches, stats = nc.rating_history_parity(db, cache, '2026-09-20')
    assert mismatches == [] and stats['common_start'] == '2026-09-01' and stats['only_before_start'] == 1


@pytest.mark.parametrize('cache, why', [
    ({'AAA': [['2026-09-01', 'BUY'], ['2026-09-12', 'HOLD']]}, 'a change only the cache saw'),
    ({'AAA': [['2026-09-01', 'BUY']], 'NEW': [['2026-09-12', 'BUY']]}, 'a ticker only the cache saw'),
    ({'AAA': [['2026-09-01', 'PASS']]}, 'a different rating'),
])
def test_real_differences_are_mismatches(cache, why):
    db = {'AAA': [['2026-09-01', 'BUY']]}
    mismatches, _ = nc.rating_history_parity(db, cache, '2026-09-20')
    assert mismatches, why


def test_parity_is_judged_as_of_the_cache_day():
    db = {'AAA': [['2026-09-01', 'BUY'], ['2026-09-21', 'HOLD']]}   # a change after the cache's last day
    cache = {'AAA': [['2026-09-01', 'BUY']]}
    assert nc.rating_history_parity(db, cache, '2026-09-20')[0] == []
    assert nc.rating_history_parity(db, cache, '2026-09-21')[0]
    assert nc.rating_history_parity({}, {}, '2026-09-21') == \
        ([], {'tickers': 0, 'common_start': None, 'only_before_start': 0})


def test_by_ticker_sorts_rpc_rows():
    rows = [['B', '2026-09-02', 'BUY'], ['A', '2026-09-03', 'HOLD'], ['A', '2026-09-01', 'BUY']]
    assert nc._by_ticker(rows) == {'A': [['2026-09-01', 'BUY'], ['2026-09-03', 'HOLD']],
                                   'B': [['2026-09-02', 'BUY']]}


# --- the streak -------------------------------------------------------------------

def _green(d, **kw):
    return {'run_date': d, 'publish_rc': 0, 'db_check_ok': True, 'parity_ok': True, **kw}


def test_trading_days_skip_weekends_and_holidays():
    days = list(nc.trading_days_back(dt.date(2026, 9, 8)))[:3]     # Tue after Labor Day
    assert days == [dt.date(2026, 9, 8), dt.date(2026, 9, 4), dt.date(2026, 9, 3)]


def test_streak_counts_consecutive_green_trading_days():
    all_days = [d.isoformat() for d in list(nc.trading_days_back(dt.date(2026, 10, 30)))[:26]]
    days = all_days[:25]
    recs = [_green(d) for d in days]
    assert nc.streak(recs, dt.date(2026, 10, 30)) == (20, None)
    assert nc.streak(recs, dt.date(2026, 10, 30), need=30) == (25, (all_days[25], 'no record'))
    broken = [r for r in recs if r['run_date'] != days[4]]         # one night never recorded
    assert nc.streak(broken, dt.date(2026, 10, 30)) == (4, (days[4], 'no record'))
    red = [dict(r, parity_ok=False) if r['run_date'] == days[2] else r for r in recs]
    assert nc.streak(red, dt.date(2026, 10, 30)) == (2, (days[2], 'parity mismatch'))
    for field, value, why in (('publish_rc', 1, '06a exited 1'), ('db_check_ok', False, 'db check failed'),
                              ('parity_ok', None, 'parity not checked')):
        bad = [dict(r, **{field: value}) if r['run_date'] == days[0] else r for r in recs]
        assert nc.streak(bad, dt.date(2026, 10, 30)) == (0, (days[0], why))


def test_gate_line():
    assert nc.gate_line(20, 20, None).startswith('DB_CUTOVER_STREAK 20/20 (ready: set DB_PRIMARY=1')
    assert nc.gate_line(3, 20, ('2026-10-01', 'no record')) == \
        'DB_CUTOVER_STREAK 3/20 (not ready; last break 2026-10-01: no record)'


# --- the commands, against a fake transport --------------------------------------

class FakeTransport:
    def __init__(self, db_hist=(), records=()):
        self.db_hist, self.records, self.calls = list(db_hist), list(records), []

    def call(self, fn, args, idempotent=False):
        self.calls.append((fn, args))
        if fn == 'rating_history':
            return [r for r in self.db_hist if r[1] < args['p_before']]
        if fn == 'record_night_check':
            green = args['p_publish_rc'] == 0 and args['p_db_check_ok'] and args['p_parity_ok'] is True
            self.records = [r for r in self.records if r['run_date'] != args['p_run_date']] + [{
                'run_date': args['p_run_date'], 'publish_rc': args['p_publish_rc'],
                'db_check_ok': args['p_db_check_ok'], 'parity_ok': args['p_parity_ok']}]
            return {'run_date': args['p_run_date'], 'green': green}
        if fn == 'night_checks':
            return [r for r in self.records if r['run_date'] >= args['p_since']]
        raise AssertionError(fn)


def _cache(tmp_path, hist, last):
    (tmp_path / 'rating_history.json').write_text(json.dumps({'last_scanned': last, 'hist': hist}),
                                                  encoding='utf-8')


def _args(tmp_path, **kw):
    import types
    base = dict(date='2026-10-01', publish_rc=0, results_dir=str(tmp_path), need=20,
                status_file=str(tmp_path / 'status.txt'), until=None)
    return types.SimpleNamespace(**{**base, **kw})


def test_record_green_and_red(tmp_path, monkeypatch, capsys):
    import scripts.check_snapshot_store as css
    monkeypatch.setattr(css, 'check_database', lambda *a: ([], ['database holds 2026-10-01: 3 rows']))
    _cache(tmp_path, {'AAA': [['2026-09-01', 'BUY']]}, '2026-09-30')
    t = FakeTransport(db_hist=[['AAA', '2026-09-01', 'BUY']])
    assert nc.cmd_record(_args(tmp_path), t) == 0
    fn, args = t.calls[-1]
    assert fn == 'record_night_check' and args['p_parity_ok'] is True and args['p_db_check_ok'] is True
    assert args['p_details']['parity_as_of'] == '2026-09-30'
    assert 'night 2026-10-01: green' in capsys.readouterr().out
    # 06a failed: recorded, not green
    assert nc.cmd_record(_args(tmp_path, publish_rc=1), t) == 1
    # a parity mismatch is recorded with examples
    t2 = FakeTransport(db_hist=[['AAA', '2026-09-01', 'HOLD']])
    assert nc.cmd_record(_args(tmp_path), t2) == 1
    details = t2.calls[-1][1]['p_details']
    assert details['mismatches'] == 1 and details['examples'][0][0] == 'AAA'


def test_record_without_the_cache_is_not_green(tmp_path, monkeypatch):
    import scripts.check_snapshot_store as css
    monkeypatch.setattr(css, 'check_database', lambda *a: ([], []))
    t = FakeTransport()
    assert nc.cmd_record(_args(tmp_path), t) == 1
    args = t.calls[-1][1]
    assert args['p_parity_ok'] is None and 'not checked' in args['p_details']['parity']


def test_a_stale_cache_is_not_evidence(tmp_path):
    _cache(tmp_path, {'AAA': [['2026-09-01', 'BUY']]}, '2026-09-21')
    t = FakeTransport(db_hist=[['AAA', '2026-09-01', 'BUY']])
    assert nc.cache_lag('2026-09-30', '2026-10-01') == 1                  # the normal night
    ok, details = nc.check_parity(t, str(tmp_path), '2026-09-28')          # 5 trading days behind
    assert ok is True
    n_calls = len(t.calls)
    ok, details = nc.check_parity(t, str(tmp_path), '2026-09-29')          # 6: stale
    assert ok is None and 'cache stale since 2026-09-21 (6 trading days behind)' in details['parity']
    assert len(t.calls) == n_calls                                          # nothing compared


def test_status_appends_the_gate_line(tmp_path, capsys):
    days = [d.isoformat() for d in list(nc.trading_days_back(dt.date(2026, 10, 30)))[:20]]
    t = FakeTransport(records=[_green(d) for d in days])
    assert nc.cmd_status(_args(tmp_path, until='2026-10-30'), t) == 0
    assert (tmp_path / 'status.txt').read_text(encoding='utf-8').startswith('DB_CUTOVER_STREAK 20/20 (ready')
    t = FakeTransport(records=[_green(d) for d in days[:5]])
    assert nc.cmd_status(_args(tmp_path, until='2026-10-30'), t) == 1


# --- run.sh: DB_PRIMARY ------------------------------------------------------------

def _fn(name):
    src = open(RUN_SH, encoding='utf-8').read()
    m = re.search(rf'^{name}\(\) {{\n.*?^}}\n', src, re.S | re.M)
    assert m, f'{name}() not found in run.sh'
    return m.group(0)


@pytest.mark.skipif(shutil.which('bash') is None, reason='needs bash')
@pytest.mark.parametrize('primary, rc, says', [('0', 0, 'skipping'), ('1', 1, 'DB_PRIMARY=1 but')])
def test_db_publish_without_secrets(primary, rc, says):
    script = (f'set -uo pipefail\nSMOKE=0; DRY_RUN=0; DB_PRIMARY={primary}; PYTHON=false; RESULTS=x\n'
              + _fn('db_publish') + 'db_publish\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith('SUPABASE_')}
    r = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == rc and says in r.stdout


@pytest.mark.skipif(shutil.which('bash') is None, reason='needs bash')
@pytest.mark.parametrize('env, want', [
    ({}, ''),                                                               # no secrets: files only
    ({'SUPABASE_URL': 'u'}, ''),
    ({'SUPABASE_URL': 'u', 'SUPABASE_SERVICE_ROLE_KEY': 'k'}, 'postgres'),
    ({'SUPABASE_URL': 'u', 'SUPABASE_SERVICE_ROLE_KEY': 'k', 'SNAPSHOT_STORE_BACKEND': 'duckdb'}, 'duckdb'),
])
def test_run_sh_selects_the_database_readers_with_the_secrets(env, want):
    src = open(RUN_SH, encoding='utf-8').read()
    m = re.search(r'^if \[ -n "\$\{SUPABASE_URL:-\}" \].*?^fi\n', src, re.S | re.M)
    assert m and 'SNAPSHOT_STORE_BACKEND' in m.group(0)
    base = {k: v for k, v in os.environ.items() if not k.startswith(('SUPABASE_', 'SNAPSHOT_STORE'))}
    r = subprocess.run(['bash', '-c', m.group(0) + 'printf %s "${SNAPSHOT_STORE_BACKEND:-}"'],
                       env={**base, **env}, capture_output=True, text=True, timeout=30)
    assert r.stdout == want


def test_run_sh_makes_06a_blocking_only_as_primary():
    src = open(RUN_SH, encoding='utf-8').read()
    assert 'DB_BLOCKING=0; [ "$DB_PRIMARY" = 1 ] && DB_BLOCKING=1' in src
    assert 'run_step 06a-db-publish "$DB_BLOCKING" db_publish\nDB_RC=$?' in src
    # the archive (06) comes after 06a and is not gated on it
    assert src.index('run_step 06a-db-publish') < src.index('run_step 06-archive')
    assert 'RESULT FAILED at db-publish (DB_PRIMARY=1; archived)' in src
    assert '--publish-rc "$DB_RC"' in src


@pytest.mark.parametrize('name', ['postgres', 'template1', 'livedb', 'bad-name'])
def test_drill_refuses_live_or_odd_database_names(name):
    from scripts import db_restore_drill as drill
    with pytest.raises(SystemExit):
        drill.create_drill_db('postgresql://u:p@127.0.0.1:1/livedb', name, replace=True)


# --- live database ---------------------------------------------------------------

DSN = os.environ.get('TEST_DATABASE_URL')
pg = pytest.mark.skipif(not DSN, reason='TEST_DATABASE_URL not set')
D = ['2031-12-15', '2031-12-16', '2031-12-17']


def _cleanup(con):
    with con.transaction():
        con.execute("DELETE FROM core.night_checks WHERE run_date >= '2031-12-01'")
        con.execute("DELETE FROM core.rating_changes WHERE run_date >= '2031-12-01'")
        con.execute("DELETE FROM core.latest_results WHERE ticker_id IN "
                    "(SELECT ticker_id FROM core.tickers WHERE ticker LIKE 'ZZR%')")
        con.execute("DELETE FROM core.results WHERE run_date >= '2031-12-01'")
        con.execute("DELETE FROM core.runs WHERE run_date >= '2031-12-01'")
        con.execute("DELETE FROM core.tickers WHERE ticker LIKE 'ZZR%'")


@pytest.fixture
def live():
    pytest.importorskip('psycopg')
    from data.db.connect import connect
    con = connect(DSN, autocommit=True)
    _cleanup(con)
    yield con
    _cleanup(con)
    con.close()


@pg
@pytest.mark.pg
def test_night_check_rpcs(live):
    from data.db.publish import DirectTransport
    t = DirectTransport(live)
    assert t.call('record_night_check', {'p_run_date': D[0], 'p_publish_rc': 0, 'p_db_check_ok': True,
                                         'p_parity_ok': None, 'p_details': {'x': 1}})['green'] is False
    assert t.call('record_night_check', {'p_run_date': D[0], 'p_publish_rc': 0, 'p_db_check_ok': True,
                                         'p_parity_ok': True, 'p_details': {}})['green'] is True   # re-record
    got = t.call('night_checks', {'p_since': '2031-12-01'})
    assert [(r['run_date'], r['parity_ok']) for r in got] == [(D[0], True)]
    for role, allowed in (('anon', False), ('authenticated', False), ('service_role', True)):
        for fn in ('pipeline.record_night_check(date, integer, boolean, boolean, jsonb)',
                   'pipeline.night_checks(date)'):
            assert live.execute('SELECT has_function_privilege(%s, %s, %s)',
                                (role, fn, 'EXECUTE')).fetchone()[0] is allowed, (role, fn)


@pg
@pytest.mark.pg
def test_restore_drill_rebuilds_and_catches_a_difference(live, tmp_path):
    """Publish three days, rebuild them in a scratch database from the files,
    and compare; then tamper with the rebuilt copy and compare again."""
    from data.db.connect import connect
    from data.db.publish import DirectTransport, build_load, publish
    from data.snapshot_store import write_snapshot_file
    from scripts import db_restore_drill as drill
    ratings = [('BUY', 'HOLD'), ('HOLD', 'HOLD'), ('HOLD', 'PASS')]
    for d, (a, b) in zip(D, ratings, strict=True):
        data = {'date': d, 'risk_free_rate': 0.04, 'results': [
            {'ticker': 'ZZRA', 'rating': a, 'mos': 0.1, 'edgar_history': {'rev': [1.5]}},
            {'ticker': 'ZZRB', 'rating': b, 'price': 12.25}]}
        write_snapshot_file(str(tmp_path / f'results_{d}.json'), data)
        publish(build_load(data, d), DirectTransport(live), min_row_ratio=0)
    name = 'restore_drill_test'
    try:
        rc = drill.main(['--source', 'archive', '--results-dir', str(tmp_path), '--server-dsn', DSN,
                         '--drill-db', name, '--replace-db', '--keep', '--since', D[0]])
        assert rc == 0
        rebuilt = connect(drill.with_dbname(DSN, name), autocommit=True)
        try:
            assert drill.compare(rebuilt, live, D)[0] == []
            rebuilt.execute("UPDATE core.results SET mos = 0.2 WHERE run_date = %s", (D[1],))
            problems, _ = drill.compare(rebuilt, live, D)
            assert any(p.startswith(f'{D[1]}: rows differ') for p in problems)
        finally:
            rebuilt.close()
    finally:
        drill.drop_drill_db(DSN, name)
    assert live.execute('SELECT 1 FROM pg_database WHERE datname = %s', (name,)).fetchone() is None
