"""scripts/score_history.py: the composite re-scored with today's model, and
how the render turns it into score_prev / score_chg / score_hist."""
import json

from scripts import report_html as rh
from scripts import score_history as sh
from scripts.scoring import compute_continuous_scores


def test_wanted_dates_take_one_per_week_plus_the_recent_runs():
    days = ['2026-06-30', '2026-07-06', '2026-07-08', '2026-07-10',
            '2026-07-13', '2026-07-17', '2026-07-20', '2026-07-21']
    got = sh.wanted_dates(days, '2026-07-21', recent=2)
    # Before the floor is dropped; each ISO week keeps its last date; the two
    # newest are kept whatever their week.
    assert '2026-06-30' not in got
    assert got == ['2026-07-10', '2026-07-17', '2026-07-20', '2026-07-21']


def test_wanted_dates_ignore_dates_after_the_run():
    got = sh.wanted_dates(['2026-07-10', '2026-07-17', '2026-07-24'], '2026-07-20')
    assert got == ['2026-07-10', '2026-07-17']


def _cache(scores, h='abc'):
    return {'version': sh.CACHE_VERSION, 'params_hash': h, 'built': None, 'scores': scores}


def test_series_aligns_weekly_points_and_takes_the_prior_run():
    c = _cache({'2026-07-10': {'A': 40.4, 'B': 10.0},
                '2026-07-17': {'A': 45.6},
                '2026-07-20': {'A': 50.2, 'B': 12.0},
                '2026-07-21': {'A': 51.0, 'B': 13.0}})
    s = sh.series(c, '2026-07-21', 'abc')
    # Tonight is not in the history (the render appends tonight's own score);
    # the week of 07-20 is represented by its last date before the run.
    assert s['dates'] == ['2026-07-10', '2026-07-17', '2026-07-20']
    assert s['hist']['A'] == [40, 46, 50]
    assert s['hist']['B'] == [10, None, 12]
    assert s['prev_date'] == '2026-07-20' and s['prev']['A'] == 50.2
    assert s['stale'] is False


def test_series_is_stale_under_another_model():
    s = sh.series(_cache({'2026-07-10': {'A': 40.0}}, h='old'), '2026-07-21', 'new')
    assert s['stale'] is True


def test_update_rescores_only_missing_dates_and_rebuilds_on_a_new_model(tmp_path, monkeypatch):
    calls = []

    def fake_rescore(results_dir, d, params, archive_dir=None):
        calls.append(d)
        return {'A': 50.0}

    monkeypatch.setattr(sh, 'rescore_date', fake_rescore)
    monkeypatch.setattr(sh, 'available_dates',
                        lambda *a, **k: {'2026-07-10', '2026-07-17'})
    path = str(tmp_path / sh.CACHE_NAME)
    c, n = sh.update(str(tmp_path), '2026-07-17', cache_path=path,
                     params_hash='h1', params={})
    assert n == 2 and sorted(calls) == ['2026-07-10', '2026-07-17']
    calls.clear()
    c, n = sh.update(str(tmp_path), '2026-07-17', cache_path=path,
                     params_hash='h1', params={})
    assert n == 0 and calls == []
    c, n = sh.update(str(tmp_path), '2026-07-17', cache_path=path,
                     params_hash='h2', params={})
    assert n == 2 and c['params_hash'] == 'h2'
    assert json.load(open(path, encoding='utf-8'))['params_hash'] == 'h2'


def test_rescore_date_matches_the_scoring_function(tmp_path):
    from data.snapshot_store import write_snapshot_file
    rows = [{'ticker': t, 'sector': 'Technology', 'roic': r, 'mos': m,
             'price': 10.0, '_fv_effective': 12.0}
            for t, r, m in (('A', 0.30, 0.2), ('B', 0.05, -0.3), ('C', 0.15, 0.05))]
    write_snapshot_file(str(tmp_path / 'results_2026-07-10.json'),
                        {'date': '2026-07-10', 'results': [dict(r) for r in rows]})
    got = sh.rescore_date(str(tmp_path), '2026-07-10', None)
    compute_continuous_scores(rows)
    assert got == {r['ticker']: round(r['_composite_score'], 1) for r in rows}


def test_score_fields_prefer_the_rescored_prior_score():
    ss = {'stale': False, 'prev': {'A': 55.0}, 'hist': {'A': [50, 54]}}
    f = rh._score_fields({'ticker': 'A', '_composite_score': 58.25},
                         {'_composite_score': 40.0}, ss)
    assert f == {'score_prev': 55.0, 'score_chg': 3.2, 'score_prev_basis': 'rescored',
                 'score_hist': [50, 54]}


def test_score_fields_fall_back_to_the_stored_score_when_stale():
    ss = {'stale': True, 'prev': {'A': 55.0}, 'hist': {'A': [50, 54]}}
    f = rh._score_fields({'ticker': 'A', '_composite_score': 58.0},
                         {'_composite_score': 40.04}, ss)
    assert f['score_prev'] == 40.0 and f['score_prev_basis'] == 'stored'
    assert f['score_chg'] == 18.0 and f['score_hist'] is None


def test_score_fields_without_any_prior():
    f = rh._score_fields({'ticker': 'N', '_composite_score': 58.0}, None, None)
    assert f == {'score_prev': None, 'score_chg': None, 'score_prev_basis': None,
                 'score_hist': None}
