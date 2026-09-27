"""The P5 scale harness (tests/load/scale.py): its pure helpers offline, and
one tiny end-to-end pass against a live database (``pg``)."""
import datetime as dt
import importlib.util
import json
import os
from collections import Counter

import pytest

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'load', 'scale.py')
_spec = importlib.util.spec_from_file_location('scale_harness', _PATH)
scale = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scale)


def test_business_days_skip_weekends():
    days = scale.business_days(dt.date(2027, 1, 1), 6)          # a Friday
    assert days[0] == dt.date(2027, 1, 1)
    assert all(d.weekday() < 5 for d in days)
    assert days[1] == dt.date(2027, 1, 4)
    assert list(scale._days_between(dt.date(2027, 1, 4), dt.date(2027, 1, 11))) == \
        scale.business_days(dt.date(2027, 1, 4), 5)


def test_ratings_change_about_two_percent_a_day():
    """The generator's rating walk is tuned to the real churn (a median of 53
    changes a night over ~2.5k tickers, ~2%)."""
    n, days, changes = 2000, 120, 0
    for i in range(n):
        seed, runlen = scale.ticker_params(i)
        assert 20 <= runlen <= 80
        prev = scale.rating_for(seed, runlen, 0)
        for d in range(1, days):
            cur = scale.rating_for(seed, runlen, d)
            changes += cur != prev
            prev = cur
    rate = changes / (n * (days - 1))
    assert 0.015 < rate < 0.03, rate
    assert scale.ticker_params(7) == scale.ticker_params(7)          # deterministic
    first = Counter(scale.rating_for(*scale.ticker_params(i), 0) for i in range(n))
    assert set(first) == set(scale.RATINGS) and min(first.values()) > n / 8


def test_percentile_and_summary():
    xs = [float(x) for x in range(1, 101)]
    assert scale.percentile(xs, 95) == 95 and scale.percentile(xs, 50) == 50 and scale.percentile([3.0], 95) == 3
    s = scale.summarize([5.0, 1.0, 2.0])
    assert s['first_ms'] == 5.0 and s['max_ms'] == 5.0 and s['n'] == 3


def test_partitions_in_plan():
    plan = [{'Plan': {'Node Type': 'Append', 'Subplans Removed': 2, 'Plans': [
        {'Node Type': 'Index Scan', 'Relation Name': 'results_2027'},
        {'Node Type': 'Seq Scan', 'Relation Name': 'results_2028'},
        {'Node Type': 'Seq Scan', 'Relation Name': 'tickers'}]}}]
    assert scale.partitions_in_plan(plan) == ({'results_2027', 'results_2028'}, 2)
    assert scale.node_types(plan) == {'Append', 'Index Scan', 'Seq Scan'}


def test_synthetic_snapshot_relabels_template_rows():
    template = {'date': '2026-09-25', 'risk_free_rate': 0.04,
                'results': [{'ticker': 'AAA', 'rating': 'BUY', 'mos': 0.1}, {'ticker': 'BBB', 'rating': None}, 'junk']}
    snap = scale.synthetic_snapshot(template, 5, 3, dt.date(2027, 1, 7))
    assert snap['date'] == '2027-01-07' and snap['risk_free_rate'] == 0.04
    assert [r['ticker'] for r in snap['results']] == [f'SYN{i:05d}' for i in range(5)]
    assert snap['results'][1]['rating'] is None                         # an unrated template stays unrated
    for i in (0, 2, 4):
        assert snap['results'][i]['rating'] == scale.rating_for(*scale.ticker_params(i), 3)
        assert snap['results'][i]['mos'] == 0.1
    assert template['results'][0]['ticker'] == 'AAA'                      # template untouched


# --- live database -------------------------------------------------------------

DSN = os.environ.get('TEST_DATABASE_URL')


@pytest.mark.pg
@pytest.mark.skipif(not DSN, reason='TEST_DATABASE_URL not set')
def test_harness_end_to_end_tiny(tmp_path):
    pytest.importorskip('psycopg')
    from data.db.connect import connect
    from data.snapshot_store import write_snapshot_file
    con = connect(DSN, autocommit=True)
    # Checked before the try: its cleanup must never run on a database this
    # test did not populate (a scale run, or real runs before mid-2027).
    if con.execute("SELECT to_regclass('bench.syn') IS NOT NULL OR EXISTS "
                   "(SELECT 1 FROM core.runs WHERE run_date < '2027-06-01')").fetchone()[0]:
        con.close()
        pytest.skip('database already holds runs before mid-2027 or a scale run; not touching it')
    try:
        tpl = tmp_path / 'results_2026-12-30.json'
        write_snapshot_file(str(tpl), {'date': '2026-12-30', 'risk_free_rate': 0.04, 'results': [
            {'ticker': f'ZZS{i}', 'rating': ('BUY', 'HOLD', None)[i % 3], 'mos': 0.1 * i, 'price': 10.0 + i,
             'sector': 'Tech'} for i in range(6)]})
        args = ['--dsn', DSN, '--template', str(tpl), '--tickers', '12', '--days', '30', '--samples', '5',
                '--readers', '2', '--direct', '--no-restart']
        out = tmp_path / 'r.json'
        assert scale.main(['seed', *args]) == 0
        assert scale.main(['generate', *args]) == 0
        n = con.execute('SELECT count(*) FROM core.results r JOIN bench.syn s USING (ticker_id)').fetchone()[0]
        assert n == 12 * 30
        # the SQL rating walk is the Python one
        got = con.execute('SELECT s.i, d.day_index, r.rating FROM core.results r JOIN bench.syn s USING (ticker_id) '
                          'JOIN bench.days d USING (run_date) WHERE r.rating IS NOT NULL').fetchall()
        assert got and all(rt == scale.rating_for(*scale.ticker_params(i), di) for i, di, rt in got)
        # change points equal what publish_run's rule gives, recomputed in Python
        cps = con.execute('SELECT count(*) FROM core.rating_changes rc JOIN bench.syn s USING (ticker_id)').fetchone()[0]
        unrated = {r[0] for r in con.execute('SELECT s.i FROM bench.syn s JOIN bench.templates t USING (tpl) '
                                             'WHERE t.rating IS NULL').fetchall()}
        assert len(unrated) == 4
        want = 0
        for i in range(12):
            if i in unrated:
                continue
            seq = [scale.rating_for(*scale.ticker_params(i), d) for d in range(30)]
            want += 1 + sum(a != b for a, b in zip(seq, seq[1:], strict=False))
        assert cps == want
        for step in ('publish', 'explain', 'size', 'bench', 'load'):
            assert scale.main([step, *args, '--out', str(out)]) == 0
            rep = json.loads(out.read_text(encoding='utf-8'))[step]
            if step == 'explain':
                assert rep['all_pruned'] and rep['ticker_history_index_only'], rep
            if step == 'load':
                assert rep['atomic'] and rep['row_counts_seen'][-1] == 12 and not rep['reader_errors'], rep
            if step == 'bench':
                assert rep['ticker_history_5y']['rows'] == 31 and rep['export']['rows'] == 12
    finally:
        with con.transaction():
            con.execute("DELETE FROM core.rating_changes WHERE ticker_id IN "
                        "(SELECT ticker_id FROM core.tickers WHERE ticker LIKE 'SYN%' OR ticker LIKE 'ZZS%')")
            con.execute("DELETE FROM core.latest_results WHERE ticker_id IN "
                        "(SELECT ticker_id FROM core.tickers WHERE ticker LIKE 'SYN%' OR ticker LIKE 'ZZS%')")
            con.execute("DELETE FROM core.results WHERE run_date = '2026-12-30' OR "
                        "(run_date >= '2027-01-04' AND run_date < '2027-06-01')")
            con.execute("DELETE FROM core.runs r WHERE r.run_date = '2026-12-30' OR "
                        "(r.run_date >= '2027-01-04' AND r.run_date < '2027-06-01')")
            con.execute("DELETE FROM core.tickers WHERE ticker LIKE 'SYN%' OR ticker LIKE 'ZZS%'")
            con.execute('DROP SCHEMA IF EXISTS bench CASCADE')
        con.close()
