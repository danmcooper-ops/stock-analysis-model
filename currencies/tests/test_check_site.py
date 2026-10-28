"""The post-deploy check (scripts/check_site.py), offline.

Its job is to tell a healthy-but-late site from a broken one, so the cases
that matter are the ones the cloud routine it replaced got wrong: a renamed
repository's Pages URL, a build that lands hours after the check fires, and
a day the build gate skipped on purpose.
"""

import json
import os

import pytest

from scripts import check_site as cs


def _payload(**ratings):
    """results.json shape: code -> (rating, composite[, valuation])."""
    rows = []
    for code, spec in ratings.items():
        row = {'code': code, 'rating': spec[0], '_composite_score': spec[1]}
        if len(spec) > 2:
            row['_score_pillar_valuation'] = spec[2]
        rows.append(row)
    return {'date': '2026-10-15', 'results': rows}


@pytest.mark.parametrize('remote', [
    'https://github.com/danmcooper-ops/Currency-Screener-',
    'https://github.com/danmcooper-ops/Currency-Screener-.git',
    'git@github.com:danmcooper-ops/Currency-Screener-.git',
    'http://local_proxy@127.0.0.1:1234/git/danmcooper-ops/Currency-Screener-',
])
def test_site_url_follows_the_remote(remote):
    slug = cs.parse_remote(remote)
    if slug is None:  # a non-GitHub proxy remote falls back to the constant
        slug = cs.parse_remote(cs.FALLBACK_REMOTE)
    assert slug == 'danmcooper-ops/Currency-Screener-'
    assert cs.site_url(slug) == \
        'https://danmcooper-ops.github.io/Currency-Screener-/'


def test_site_url_lowercases_the_owner():
    assert cs.site_url('Dan/Repo') == 'https://dan.github.io/Repo/'


@pytest.mark.parametrize('status,size,kind', [
    (200, 683_504, 'ok'),
    (200, 16_000, 'readme'),
    (404, 9_115, 'missing'),
    (None, 0, 'missing'),
    (200, 900_000, 'odd'),
])
def test_classify_page(status, size, kind):
    assert cs.classify_page(status, size) == kind


def test_previous_day_skips_today_and_picks_the_newest_earlier(tmp_path):
    for day in ('2026-10-13', '2026-10-14', '2026-10-15'):
        (tmp_path / day).mkdir()
        (tmp_path / day / 'results.json').write_text('{}', encoding='utf-8')
    (tmp_path / '2026-10-12').mkdir()          # no results.json: ignored
    (tmp_path / 'notes').mkdir()
    assert cs.previous_day(str(tmp_path), '2026-10-15') == '2026-10-14'
    assert cs.previous_day(str(tmp_path), '2026-10-13') is None
    assert cs.previous_day(str(tmp_path / 'absent'), '2026-10-15') is None


def test_diff_reports_rating_changes_and_universe_moves():
    prev = _payload(HUF=('NEUTRAL', 38.8), JPY=('LONG', 60.0), ARS=('SHORT', 9))
    cur = _payload(HUF=('LEAN LONG', 39.2), JPY=('LONG', 61.0), EGP=('NEUTRAL', 40))
    d = cs.diff_results(prev, cur)
    assert d['changes'] == [{'code': 'HUF', 'old': 'NEUTRAL', 'new': 'LEAN LONG',
                             'old_score': 38.8, 'new_score': 39.2}]
    assert d['added'] == ['EGP'] and d['removed'] == ['ARS']


def test_diff_with_no_changes_is_empty():
    p = _payload(HUF=('NEUTRAL', 38.8))
    assert cs.diff_results(p, p) == {'changes': [], 'added': [], 'removed': []}


def test_top_n_orders_by_composite_and_ignores_missing_scores():
    p = _payload(A=('LONG', 50), B=('LONG', 70), C=('NEUTRAL', None), D=('LONG', 60))
    assert [c for c, _, _ in cs.top_n(p, 2)] == ['B', 'D']


@pytest.mark.parametrize('score,inverted', [(90.4, False), (50.0, False),
                                            (49.9, True), (5.0, True)])
def test_jpy_valuation_floor(score, inverted):
    assert cs.jpy_valuation(_payload(JPY=('LONG', 60, score))) == (score, inverted)


def test_missing_jpy_score_is_not_called_inverted():
    assert cs.jpy_valuation(_payload(JPY=('LONG', 60))) == (None, False)
    assert cs.jpy_valuation(_payload()) == (None, False)


def test_data_quality():
    assert cs.data_quality({'provenance': {'stale_slots': 0,
                                           'event_counts': {}}}) == []
    issues = cs.data_quality({'provenance': {'stale_slots': 2,
                                             'event_counts': {'stale': 2}}})
    assert len(issues) == 2 and '2 stale' in issues[0]
    assert cs.data_quality({}) == ['manifest carries no provenance block']


def _run(created, status='completed', conclusion='success'):
    return {'created_at': created, 'status': status, 'conclusion': conclusion}


def test_runs_today_uses_the_new_york_date():
    runs = [_run('2026-10-15T23:04:42Z'),   # 19:04 New York, 10-15
            _run('2026-10-16T03:30:00Z'),   # 23:30 New York, still 10-15
            _run('2026-10-14T23:04:42Z')]
    assert len(cs.runs_today(runs, '2026-10-15')) == 2


def test_not_landed_skipped_when_ecb_published_nothing():
    """A TARGET holiday is a correct skip even if a run failed."""
    assert cs.not_landed_status(False, [_run('x', conclusion='failure')]) \
        == 'SKIPPED'


def test_not_landed_failed_only_when_nothing_is_still_going():
    failed = _run('x', conclusion='failure')
    assert cs.not_landed_status(True, [failed]) == 'FAILED'
    assert cs.not_landed_status(True, [failed, _run('y', status='queued',
                                                    conclusion=None)]) == 'LATE'


@pytest.mark.parametrize('runs', [[], [_run('x', status='in_progress',
                                            conclusion=None)],
                                  [_run('x')]])
def test_not_landed_otherwise_late(runs):
    """No run yet (GitHub starts crons hours late), running, or gated."""
    assert cs.not_landed_status(True, runs) == 'LATE'
    assert cs.not_landed_status(None, runs) == 'LATE'


def test_worst_status_wins():
    assert cs.worst(['OK', 'JPY-INVERTED']) == 'JPY-INVERTED'
    assert cs.worst(['LATE', 'SITE-WRONG']) == 'SITE-WRONG'
    assert cs.worst([]) == 'OK'


def test_report_leads_with_changes_and_flags_jpy():
    cur = _payload(JPY=('LONG', 60, 30.0), HUF=('LEAN LONG', 39.2))
    prev = _payload(JPY=('LONG', 60, 90.0), HUF=('NEUTRAL', 38.8))
    ctx = {'today': '2026-10-15', 'status': 'JPY-INVERTED', 'reason': 'r',
           'site': 's', 'results': cur, 'prev_date': '2026-10-14',
           'diff': cs.diff_results(prev, cur), 'top': cs.top_n(cur),
           'jpy': cs.jpy_valuation(cur), 'issues': []}
    md = cs.render_report(ctx)
    assert md.index('Rating changes') < md.index('Top 5')
    assert 'HUF: NEUTRAL -> LEAN LONG' in md
    assert 'BELOW 50' in md


def test_report_says_when_there_is_nothing_to_compare():
    cur = _payload(JPY=('LONG', 60, 90.0))
    md = cs.render_report({'today': 'd', 'status': 'OK', 'reason': 'r',
                           'site': 's', 'results': cur, 'diff': None,
                           'top': cs.top_n(cur), 'jpy': cs.jpy_valuation(cur)})
    assert 'comparison unavailable' in md


def test_archive_and_prune(tmp_path):
    h = str(tmp_path)
    cs.archive(h, '2026-10-15', {'a': 1}, {'run_date': '2026-10-15'})
    cs.archive(h, '2024-01-02', {'a': 0}, None)
    assert json.loads((tmp_path / '2026-10-15' / 'results.json')
                      .read_text(encoding='utf-8')) == {'a': 1}
    assert not (tmp_path / '2024-01-02' / 'manifest.json').exists()
    cs.prune(h, '2026-10-15')
    assert sorted(os.listdir(h)) == ['2026-10-15']
