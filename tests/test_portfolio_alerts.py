"""Portfolio alerts v2: Action/Watch/FYI classification, systemic days,
the digest (text / markdown / JSON) and the GitHub-issue poster."""
import json
import subprocess

import pytest

from models import portfolio_groups as pg
from scripts import portfolio_digest as pdg
from scripts import portfolios as cli

DAY = '2026-09-25'


def row(t, rating, price=10.0, name=None, **kw):
    r = {'ticker': t, 'rating': rating, 'price': price,
         'company_name': f'{t} Corp' if name is None else name, 'sector': 'Technology'}
    r.update(kw)
    return r


def by(*rows):
    return pg.rows_by_ticker(list(rows))


def classify(today, prior, **kw):
    return pg.classify_changes(today, prior, DAY, **kw)


def kinds(entries):
    return sorted((e['ticker'], e['kind'], e['level']) for e in entries)


# A stable background so a handful of moves never reads as a flood day.
BG = [row(f'Z{i:03d}', 'HOLD') for i in range(60)]


def with_bg(*rows):
    return by(*rows, *BG)


class TestClassify:
    def test_buy_line_crossings_are_action_same_side_is_fyi(self):
        prior = with_bg(row('UP', 'HOLD'), row('DN', 'LEAN BUY'), row('SS', 'BUY'), row('PP', 'HOLD'))
        today = with_bg(row('UP', 'LEAN BUY'), row('DN', 'PASS'), row('SS', 'LEAN BUY'), row('PP', 'PASS'))
        e, st = classify(today, prior)
        assert kinds(e) == [('DN', 'exited_buy', 'action'), ('PP', 'same_side', 'fyi'),
                            ('SS', 'same_side', 'fyi'), ('UP', 'entered_buy', 'action')]
        assert not st['flood']
        msg = {x['ticker']: x['message'] for x in e}
        assert msg['UP'] == 'UP moved into the buy zone: HOLD → LEAN BUY'
        assert msg['DN'] == 'DN fell out of the buy zone: LEAN BUY → PASS'

    def test_reversal_within_window_is_watch(self):
        prior = with_bg(row('R', 'HOLD'), row('OLD', 'HOLD'))
        today = with_bg(row('R', 'BUY'), row('OLD', 'BUY'))
        hist = {'R': [['2026-09-10', 'BUY'], ['2026-09-22', 'HOLD']],       # left the zone 3 days ago
                'OLD': [['2026-08-01', 'BUY'], ['2026-09-01', 'HOLD']]}     # 24 days ago: too old
        e, _ = classify(today, prior, history=hist)
        got = {x['ticker']: x for x in e}
        assert (got['R']['kind'], got['R']['level'], got['R']['reversal_of']) == ('reversal', 'watch', '2026-09-22')
        assert got['R']['message'].endswith('(reverses the 2026-09-22 move)')
        assert got['OLD']['kind'] == 'entered_buy'
        # A same-direction earlier crossing is not a reversal.
        e, _ = classify(today, prior, history={'R': [['2026-09-20', 'HOLD'], ['2026-09-21', 'BUY'],
                                                    ['2026-09-22', 'HOLD']]})
        assert {x['ticker']: x['kind'] for x in e}['R'] == 'reversal'

    def test_missing_data_is_a_data_gap_not_a_signal(self):
        prior = with_bg(row('G', 'BUY', _composite_score=70.0), row('P', 'BUY'))
        today = with_bg(row('G', 'PASS', price=None, _composite_score=30.0),
                        row('P', 'HOLD'))
        prior['P'].update(price=None)
        e, _ = classify(today, prior)
        assert kinds(e) == [('G', 'data_gap', 'fyi'), ('P', 'data_gap', 'fyi')]   # no score_drop for G
        assert "today's row is missing" in {x['ticker']: x['message'] for x in e}['G']
        assert "the prior run's row" in {x['ticker']: x['message'] for x in e}['P']
        # Identity loss alone (name and sector blank) also counts.
        assert pg.data_missing(row('X', 'BUY', name='', sector=''))
        assert not pg.data_missing(row('X', 'BUY', name='', sector='Energy'))

    def test_flood_days(self):
        prior = by(*[row(f'T{i}', 'HOLD') for i in range(20)])
        today = by(*[row(f'T{i}', 'PASS' if i < 3 else 'HOLD') for i in range(20)])
        _, st = classify(today, prior)
        assert st['flood'] and st['cause'] == 'model' and st['changed_share'] == 0.15
        assert pg.systemic_message(st).startswith('Model-wide shift: 15% of the universe (3 stocks)')
        today = by(*[row(f'T{i}', 'HOLD', price=None if i < 5 else 10.0) for i in range(20)])
        _, st = classify(today, prior)
        assert st['cause'] == 'data'
        assert pg.systemic_message(st).startswith('Data problem: 5 of 20 rows')
        assert pg.systemic_message({'flood': False}) is None

    def test_flood_tags_crossings(self):
        prior = by(*[row(f'T{i}', 'HOLD') for i in range(5)])
        today = by(*[row(f'T{i}', 'BUY') for i in range(5)])
        e, st = classify(today, prior)
        assert st['flood'] and all(x.get('flood') for x in e)

    def test_score_drop_and_fv_jump(self):
        prior = with_bg(row('S', 'HOLD', _composite_score=60.0, _fv_effective=100.0),
                        row('F', 'HOLD', _fv_effective=100.0), row('Q', 'HOLD', _fv_effective=100.0))
        today = with_bg(row('S', 'HOLD', _composite_score=49.0, _fv_effective=100.0),
                        row('F', 'HOLD', _fv_effective=151.0), row('Q', 'HOLD', _fv_effective=140.0))
        e, _ = classify(today, prior)
        assert kinds(e) == [('F', 'fv_jump', 'watch'), ('S', 'score_drop', 'watch')]
        assert 'fair value moved +51% ($100.00 → $151.00)' in [x for x in e if x['ticker'] == 'F'][0]['message']

    def test_explain_attaches_why_to_rating_moves(self):
        prior, today = with_bg(row('U', 'HOLD')), with_bg(row('U', 'BUY'))
        e, _ = classify(today, prior, explain=lambda p, r: [f"{p['rating']}→{r['rating']}"])
        assert e[0]['why'] == ['HOLD→BUY']
        e, _ = classify(today, prior, explain=lambda p, r: None)
        assert 'why' not in e[0]

    def test_earnings_soon(self):
        rows = by(row('E', 'HOLD', earnings_next_date='2026-09-30'),
                  row('L', 'HOLD', earnings_next_date='2026-10-09'),
                  row('P', 'HOLD', earnings_next_date='2026-09-20'), row('N', 'HOLD'))
        assert [x['ticker'] for x in pg.earnings_soon(['E', 'L', 'P', 'N'], rows, DAY)] == ['E']

    def test_modes_and_validation(self):
        e = {'kind': 'same_side', 'level': 'fyi'}
        assert pg.level_for(e, 'buy_line') == 'fyi' and pg.level_for(e, 'all') == 'watch'
        assert pg.level_for({'kind': 'entered_buy', 'level': 'action'}, 'off') is None
        p = pg.normalize_portfolio({'id': 'a', 'name': 'A'}, today=DAY)
        assert p['alerts'] == 'buy_line'
        with pytest.raises(ValueError, match='alerts must be one of'):
            pg.normalize_portfolio({'id': 'a', 'name': 'A', 'alerts': 'loud'}, today=DAY)


# ---------------------------------------------------------------- digest

def _digest(tmp_path=None, flood=False, alerts=None):
    return {'version': 1, 'date': DAY, 'prev_date': '2026-09-24',
            'systemic': {'flood': flood, 'cause': 'data' if flood else None,
                         'message': 'Data problem: 9 of 10 rows are missing a price' if flood else None},
            'portfolios': [{'id': 'semis', 'name': 'Semis', 'mode': 'buy_line', 'missing': [],
                            'stats': {'n': 3, 'ratings': {'BUY': 1, 'LEAN BUY': 0, 'HOLD': 2, 'PASS': 0},
                                      'median_mos': -0.5, 'median_score': 40, 'median_spread': 0.3,
                                      'top_sector': 'Technology', 'top_sector_weight': 1.0,
                                      'concentrated': True, 'upgrades': 1, 'downgrades': 0},
                            'alerts': alerts if alerts is not None else [
                                {'ticker': 'AVGO', 'kind': 'entered_buy', 'level': 'action',
                                 'message': 'AVGO moved into the buy zone: HOLD → BUY',
                                 'why': ['Composite crossed 60']},
                                {'ticker': 'TSM', 'kind': 'score_drop', 'level': 'watch',
                                 'message': 'TSM composite score dropped 12.0 pts (60.0 → 48.0)'},
                                {'ticker': 'NVDA', 'kind': 'same_side', 'level': 'fyi',
                                 'message': 'NVDA HOLD → PASS'}]}]}


class TestDigest:
    def test_text(self):
        t = pdg.render_text(_digest(flood=True))
        assert t.startswith('Portfolio alerts 2026-09-25 (vs 2026-09-24)\n\n!! Data problem')
        assert 'ACTION AVGO moved into the buy zone: HOLD → BUY' in t
        assert '           · Composite crossed 60' in t
        assert 'WATCH  TSM composite score dropped' in t
        assert '(1 quieter change: ' in t and 'NVDA' not in t
        assert t.rstrip().endswith('1 action, 1 watch, 1 FYI')

    def test_markdown_and_title(self):
        md = pdg.render_markdown(_digest(flood=True), 'https://x.github.io/r/')
        assert md.startswith('## Portfolio alerts — 2026-09-25')
        assert '(https://x.github.io/r/#s=%7B%22v%22%3A%22pf%22%7D)' in md
        assert '> ⚠️ **Data problem:**' in md
        assert '**Action**\n- **AVGO** moved into the buy zone: HOLD → BUY\n  - Composite crossed 60' in md
        assert '<details><summary>1 quieter change</summary>' in md
        assert pdg.title(_digest(flood=True)) == 'Portfolio alerts — 2026-09-25 (1 action, 1 watch) — systemic day'

    def test_markdown_stays_under_githubs_limit(self):
        fyi = [{'ticker': f'T{i}', 'kind': 'same_side', 'level': 'fyi', 'message': 'x' * 200}
               for i in range(2000)]
        md = pdg.render_markdown(_digest(alerts=fyi))
        assert len(md) <= pdg.MAX_BODY + 200 and '… and 1940 more' in md

    def test_should_post(self):
        assert pdg.should_post(_digest())
        quiet = _digest(alerts=[{'ticker': 'N', 'kind': 'same_side', 'level': 'fyi', 'message': 'N'}])
        assert not pdg.should_post(quiet)
        assert pdg.should_post(dict(quiet, systemic={'flood': True}))


class FakeGh:
    """Records gh invocations; answers `issue list` from *existing*."""
    def __init__(self, existing=(), label_exists=False):
        self.calls, self.existing, self.label_exists = [], list(existing), label_exists

    def __call__(self, cmd, check, capture_output, text):
        self.calls.append(cmd[1:])
        if cmd[1:3] == ['issue', 'list']:
            out = json.dumps(self.existing)
        elif cmd[1:3] == ['label', 'create'] and self.label_exists:
            raise subprocess.CalledProcessError(1, cmd)
        elif cmd[1:3] == ['issue', 'create']:
            out = 'https://github.com/o/r/issues/9\n'
        else:
            out = ''
        return subprocess.CompletedProcess(cmd, 0, stdout=out)


class TestPost:
    def test_creates_labels_and_closes_older(self):
        gh = FakeGh(existing=[{'number': 5, 'title': 'Portfolio alerts — 2026-09-24', 'state': 'OPEN'},
                              {'number': 3, 'title': 'Portfolio alerts — 2026-09-23', 'state': 'CLOSED'}],
                    label_exists=True)
        assert pdg.post(_digest(), 'o/r', run=gh) == 'created https://github.com/o/r/issues/9'
        verbs = [c[:2] for c in gh.calls]
        assert verbs == [['issue', 'list'], ['label', 'create'], ['issue', 'create'], ['issue', 'close']]
        create = gh.calls[2]
        assert create[create.index('--title') + 1] == 'Portfolio alerts — 2026-09-25 (1 action, 1 watch)'
        assert gh.calls[3][2] == '5'

    def test_idempotent_quiet_and_dry_run(self):
        gh = FakeGh(existing=[{'number': 9, 'title': 'Portfolio alerts — 2026-09-25 (1 action)', 'state': 'OPEN'}])
        assert pdg.post(_digest(), 'o/r', run=gh) == 'already posted for 2026-09-25'
        assert [c[:2] for c in gh.calls] == [['issue', 'list']]
        gh = FakeGh()
        quiet = _digest(alerts=[])
        assert pdg.post(quiet, 'o/r', run=gh) == 'quiet day: nothing to post' and gh.calls == []
        gh = FakeGh()
        assert pdg.post(_digest(), 'o/r', dry_run=True, run=gh).startswith('dry run: would create')
        assert [c[:2] for c in gh.calls] == [['issue', 'list']]


# ---------------------------------------------------------------- CLI

class TestCli:
    def _days(self, tmp_path):
        from data.snapshot_store import write_snapshot_file
        res = tmp_path / 'out'
        res.mkdir()
        prior = [row('AAA', 'HOLD'), row('BBB', 'BUY'), *BG]
        today = [row('AAA', 'BUY'), row('BBB', 'BUY'), *BG]
        write_snapshot_file(str(res / 'results_2026-09-24.json'), {'results': prior})
        write_snapshot_file(str(res / 'results_2026-09-25.json'), {'results': today})
        return res

    def test_alerts_json_markdown_and_edit(self, tmp_path, capsys):
        res = self._days(tmp_path)
        f = tmp_path / 'pf.json'
        cli.main(['--file', str(f), 'create', 'p', '--name', 'P', '--tickers', 'AAA,BBB'])
        cli.main(['--file', str(f), 'edit', 'p', '--alerts', 'all'])
        assert pg.load_portfolios(str(f))['portfolios'][0]['alerts'] == 'all'
        capsys.readouterr()
        cli.main(['--file', str(f), 'alerts', '--results-dir', str(res),
                  '--json', str(tmp_path / 'a.json'), '--markdown', str(tmp_path / 'a.md'),
                  '--pages-url', 'https://x/'])
        out = capsys.readouterr().out
        assert 'ACTION AAA moved into the buy zone: HOLD → BUY' in out and '[alerts: all]' in out
        d = json.loads((tmp_path / 'a.json').read_text(encoding='utf-8'))
        assert d['date'] == DAY and d['prev_date'] == '2026-09-24' and not d['systemic']['flood']
        assert [(a['ticker'], a['level']) for a in d['portfolios'][0]['alerts']] == [('AAA', 'action')]
        assert '**AAA** moved into the buy zone' in (tmp_path / 'a.md').read_text(encoding='utf-8')

    def test_replay(self, tmp_path, capsys):
        res = self._days(tmp_path)
        cli.main(['--file', str(tmp_path / 'pf.json'), 'alerts', '--replay', '--results-dir', str(res)])
        out = capsys.readouterr().out
        assert '2026-09-25' in out and '1 run(s): ACTION median 1/run' in out


class TestStatsLine:
    BASE = {'n': 3, 'ratings': {'HOLD': 2, 'PASS': 1}, 'median_mos': -0.5, 'median_score': 40,
            'median_spread': 0.3, 'top_sector': 'Technology', 'top_sector_weight': 1.0,
            'concentrated': False}

    def test_no_sector_suffix(self):
        line = pdg.stats_line(dict(self.BASE, no_sector=2))
        assert line.endswith('top sector Technology 100%; 2 without sector data')
        assert 'Unknown' not in line

    def test_no_suffix_when_zero_or_absent(self):
        # Digests written before no_sector existed render unchanged.
        assert pdg.stats_line(dict(self.BASE, no_sector=0)) == pdg.stats_line(self.BASE)
        assert 'sector data' not in pdg.stats_line(self.BASE)
