"""Portfolio groupings: definitions, rules, membership, CLI and report wiring."""
import base64
import json
import os
import re
import shutil
import subprocess

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models import portfolio_groups as pg
from scripts import portfolios as cli

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'portfolio_rule_cases.json')
TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'report.html')


def _cases():
    with open(FIXTURE, encoding='utf-8') as f:
        return json.load(f)


def _pf(**kw):
    base = {'id': 'p', 'name': 'P', 'tickers': [], 'exclude': [], 'rule': None}
    base.update(kw)
    return pg.normalize_portfolio(base, today='2026-09-26')


# ---------------------------------------------------------------- validation

class TestNormalize:
    def test_minimal_portfolio_gets_defaults(self):
        p = _pf(tickers=['nvda', ' AMD ', 'NVDA'])
        assert p['tickers'] == ['AMD', 'NVDA']
        assert p['created'] == '2026-09-26'
        assert p['color'] is None and p['rule'] is None

    @pytest.mark.parametrize('bad_id', ['', 'Has Caps', '-lead', 'x' * 41, 'a_b', None])
    def test_bad_ids_rejected(self, bad_id):
        with pytest.raises(ValueError, match='id'):
            _pf(id=bad_id)

    def test_duplicate_ids_rejected(self):
        doc = {'version': 1, 'portfolios': [{'id': 'a', 'name': 'A'}, {'id': 'a', 'name': 'B'}]}
        with pytest.raises(ValueError, match='duplicate'):
            pg.normalize(doc)

    def test_unsupported_version(self):
        with pytest.raises(ValueError, match='version'):
            pg.normalize({'version': 2, 'portfolios': []})

    def test_bad_color(self):
        with pytest.raises(ValueError, match='color'):
            _pf(color='red')

    def test_rule_without_constraints_rejected(self):
        # It would silently turn the portfolio into the whole universe.
        with pytest.raises(ValueError, match='no constraints'):
            _pf(rule={'ratings': None, 'sectors': None, 'countries': None, 'cf': []})

    @pytest.mark.parametrize('clause, msg', [
        ({'key': 'mos'}, 'min and/or max'),
        ({'key': 'mos', 'min': 'x'}, 'finite number'),
        ({'key': 'mos', 'min': float('inf')}, 'finite number'),
        ({'key': 'mos', 'min': 0.1, 'txt': None, 'bogus': 1}, 'only key, min and max'),
        ({'key': 'industry', 'txt': '  '}, 'empty text'),
        ({'min': 1}, "column 'key'"),
    ])
    def test_bad_clauses(self, clause, msg):
        with pytest.raises(ValueError, match=msg):
            _pf(rule={'cf': [clause]})

    def test_unknown_rule_key(self):
        with pytest.raises(ValueError, match='unknown key'):
            _pf(rule={'ratings': ['BUY'], 'industries': ['x']})

    def test_text_clause_uppercased_and_lists_sorted(self):
        p = _pf(rule={'ratings': ['LEAN BUY', 'BUY', 'BUY'], 'cf': [{'key': 'industry', 'txt': 'oil '}]})
        assert p['rule']['ratings'] == ['BUY', 'LEAN BUY']
        assert p['rule']['cf'] == [{'key': 'industry', 'txt': 'OIL'}]
        assert p['rule']['sectors'] is None

    def test_slugify(self):
        assert pg.slugify('Energy BUYs!') == 'energy-buys'
        assert pg.slugify('***') == 'portfolio'
        assert len(pg.slugify('a' * 80)) == 40


# ---------------------------------------------------------------- rules

class TestRules:
    @pytest.mark.parametrize('case', _cases()['cases'], ids=lambda c: c['name'])
    def test_fixture_cases(self, case):
        rows = _cases()['rows']
        rule = pg.normalize_rule(case['rule'])
        got = [r['ticker'] for r in rows if pg.rule_matches(rule, r)]
        assert got == case['expect']

    @pytest.mark.parametrize('v, want', [
        (None, None), (True, 1.0), (False, 0.0), (3, 3.0), (2.5, 2.5),
        (float('nan'), None), (float('inf'), None), ('  12 ', 12.0), ('', 0.0),
        ('1e3', 1000.0), ('.5', 0.5), ('5.', 5.0), ('Infinity', None), ('NaN', None),
        ('1_000', None), ('abc', None), ([1], None), ({}, None),
    ])
    def test_js_num(self, v, want):
        assert pg.js_num(v) == want

    def test_empty_rule_matches_nothing(self):
        assert not pg.rule_matches(None, {'ticker': 'X'})


# ---------------------------------------------------------------- membership

ROWS = [
    {'ticker': 'AAA', 'rating': 'BUY', 'sector': 'Energy'},
    {'ticker': 'BBB', 'rating': 'HOLD', 'sector': 'Energy'},
    {'ticker': 'CCC', 'rating': 'BUY', 'sector': 'Technology'},
]


class TestMembership:
    def test_union_of_picks_and_rule_minus_exclude(self):
        p = _pf(tickers=['CCC', 'GONE'], exclude=['AAA'],
                rule={'sectors': ['Energy']})
        res = pg.resolve_members(p, pg.rows_by_ticker(ROWS))
        assert res['members'] == ['BBB', 'CCC']
        assert res['missing'] == ['GONE']
        assert res['ruled'] == ['BBB']

    def test_exclude_beats_a_hand_pick(self):
        p = _pf(tickers=['AAA'], exclude=['AAA'])
        assert pg.resolve_members(p, pg.rows_by_ticker(ROWS))['members'] == []

    def test_ticker_in_several_portfolios(self):
        pfs = [_pf(id='a', tickers=['AAA']),
               _pf(id='b', rule={'ratings': ['BUY']}),
               _pf(id='c', tickers=['ZZZ'])]
        idx = pg.membership_index(pfs, ROWS)
        assert idx == {'AAA': ['a', 'b'], 'CCC': ['b']}

    @settings(max_examples=60, deadline=None)
    @given(picks=st.lists(st.sampled_from(['AAA', 'BBB', 'CCC', 'DDD']), unique=True),
           excl=st.lists(st.sampled_from(['AAA', 'BBB', 'CCC', 'DDD']), unique=True),
           use_rule=st.booleans())
    def test_properties(self, picks, excl, use_rule):
        rule = {'ratings': ['BUY']} if use_rule else None
        p = _pf(tickers=picks, exclude=excl, rule=rule)
        by_tk = pg.rows_by_ticker(ROWS)
        res = pg.resolve_members(p, by_tk)
        # Exclude always wins; members are always in the universe.
        assert not set(res['members']) & set(excl)
        assert set(res['members']) <= set(by_tk)
        # Order of the definition's lists never matters.
        q = _pf(tickers=list(reversed(picks)), exclude=list(reversed(excl)), rule=rule)
        assert pg.resolve_members(q, by_tk) == res


# ---------------------------------------------------------------- file IO

class TestFile:
    def test_missing_file_is_empty(self, tmp_path):
        assert pg.load_portfolios(str(tmp_path / 'nope.json'))['portfolios'] == []

    def test_save_round_trip_materializes_colors(self, tmp_path):
        path = str(tmp_path / 'sub' / 'pf.json')
        doc = {'version': 1, 'portfolios': [{'id': 'b', 'name': 'B'}, {'id': 'a', 'name': 'A', 'color': '#ABCDEF'}]}
        pg.save_portfolios(doc, path)
        back = pg.load_portfolios(path)
        assert [p['id'] for p in back['portfolios']] == ['b', 'a']
        assert back['portfolios'][0]['color'] == pg.PALETTE[0]
        assert back['portfolios'][1]['color'] == '#abcdef'
        assert not [f for f in os.listdir(tmp_path / 'sub') if f.endswith('.tmp')]

    def test_invalid_doc_is_not_written(self, tmp_path):
        path = tmp_path / 'pf.json'
        with pytest.raises(ValueError):
            pg.save_portfolios({'version': 1, 'portfolios': [{'id': 'BAD', 'name': 'x'}]}, str(path))
        assert not path.exists()

    def test_revision_is_content_hash(self):
        a = {'version': 1, 'portfolios': [{'id': 'a', 'name': 'A', 'created': '2026-01-01', 'tickers': ['X', 'Y']}]}
        b = {'version': 1, 'portfolios': [{'id': 'a', 'name': 'A', 'created': '2026-01-01', 'tickers': ['Y', 'X']}]}
        assert pg.revision(a) == pg.revision(b)
        b['portfolios'][0]['name'] = 'B'
        assert pg.revision(a) != pg.revision(b)


# ---------------------------------------------------------------- merge / share

class TestMergeAndShare:
    def test_decode_share_link(self):
        p = {'id': 'x', 'name': 'X', 'tickers': ['aaa']}
        tok = base64.urlsafe_b64encode(json.dumps(p).encode()).decode().rstrip('=')
        doc, base = pg.decode_share(f'https://h/report.html#pf={tok}')
        assert doc['portfolios'][0]['tickers'] == ['AAA'] and base is None

    def test_decode_export_keeps_base_rev(self):
        doc, base = pg.decode_share(json.dumps({'version': 1, 'base_rev': 'abc', 'portfolios': []}))
        assert base == 'abc' and doc['portfolios'] == []

    def test_merge_adds_and_refuses_clashes(self):
        cur = [_pf(id='a', tickers=['X'])]
        inc = [_pf(id='a', tickers=['Y']), _pf(id='b')]
        with pytest.raises(ValueError, match='a'):
            pg.merge_portfolios(cur, inc)
        out = pg.merge_portfolios(cur, inc, overwrite=True)
        assert [(p['id'], p['tickers']) for p in out] == [('a', ['Y']), ('b', [])]
        assert pg.merge_portfolios(cur, [cur[0]]) == cur

    def test_diff_lines(self):
        old = [_pf(id='a', tickers=['X']), _pf(id='gone')]
        new = [_pf(id='a', name='A2', tickers=['Y']), _pf(id='new', rule={'ratings': ['BUY']})]
        lines = pg.diff_portfolios(old, new)
        assert lines[0].startswith('+ new') and lines[1].startswith('- gone')
        assert "    name: 'P' -> 'A2'" in lines
        assert '    tickers +Y' in lines and '    tickers -X' in lines


# ---------------------------------------------------------------- CLI

def _run(path, *argv):
    return cli.main(['--file', str(path), *argv])


class TestCli:
    def test_create_add_remove_rule(self, tmp_path, capsys):
        f = tmp_path / 'pf.json'
        _run(f, 'create', 'semis', '--name', 'Semis', '--tickers', 'nvda,amd')
        _run(f, 'create', '--name', 'Energy BUYs', '--sector', 'Energy',
             '--rating', 'BUY,LEAN BUY', '--min', 'mcap=2e9', '--contains', 'industry=oil')
        _run(f, 'add', 'semis', 'avgo')
        _run(f, 'remove', 'semis', 'AMD')
        _run(f, 'remove', 'energy-buys', 'XOM')
        doc = pg.load_portfolios(str(f))
        semis, energy = doc['portfolios']
        assert semis['tickers'] == ['AVGO', 'NVDA']
        assert energy['exclude'] == ['XOM']
        assert energy['rule'] == {'ratings': ['BUY', 'LEAN BUY'], 'sectors': ['Energy'], 'countries': None,
                                  'cf': [{'key': 'mcap', 'min': 2e9, 'max': None},
                                         {'key': 'industry', 'txt': 'OIL'}]}
        assert all(p['color'] for p in doc['portfolios'])
        # Adding a ticker back lifts its exclude.
        _run(f, 'add', 'energy-buys', 'XOM')
        assert pg.load_portfolios(str(f))['portfolios'][1]['exclude'] == []

    def test_create_duplicate_and_unknown_id(self, tmp_path):
        f = tmp_path / 'pf.json'
        _run(f, 'create', 'a', '--name', 'A')
        with pytest.raises(SystemExit, match='already exists'):
            _run(f, 'create', 'a', '--name', 'A')
        with pytest.raises(SystemExit, match="no portfolio 'zz'"):
            _run(f, 'add', 'zz', 'X')

    def test_import_fast_forward_vs_merge(self, tmp_path, capsys):
        f = tmp_path / 'pf.json'
        _run(f, 'create', 'a', '--name', 'A', '--tickers', 'X')
        _run(f, 'create', 'b', '--name', 'B')
        cur = pg.load_portfolios(str(f))
        # The browser's export: edited a, deleted b, made c — from this file.
        exp = {'version': 1, 'base_rev': pg.revision(cur), 'portfolios': [
            dict(cur['portfolios'][0], tickers=['X', 'Y']),
            {'id': 'c', 'name': 'C', 'rule': {'ratings': ['BUY']}}]}
        e = tmp_path / 'export.json'
        e.write_text(json.dumps(exp), encoding='utf-8')
        _run(f, 'import', str(e), '--dry-run')
        out = capsys.readouterr().out
        assert 'fast-forward' in out and '- b (B)' in out and '    tickers +Y' in out
        assert [p['id'] for p in pg.load_portfolios(str(f))['portfolios']] == ['a', 'b']  # dry run
        # From an older file revision it merges, and the edit to a clashes.
        exp['base_rev'] = 'old'
        e.write_text(json.dumps(exp), encoding='utf-8')
        with pytest.raises(SystemExit, match='differ from the file: a'):
            _run(f, 'import', str(e))
        _run(f, 'import', str(e), '--overwrite')
        doc = pg.load_portfolios(str(f))
        assert [p['id'] for p in doc['portfolios']] == ['a', 'b', 'c']
        assert doc['portfolios'][0]['tickers'] == ['X', 'Y']

    def test_show_against_snapshot(self, tmp_path, capsys):
        from data.snapshot_store import write_snapshot_file
        res = tmp_path / 'out'
        res.mkdir()
        write_snapshot_file(str(res / 'results_2026-09-14.json'),
                            {'results': ROWS + [{'ticker': 'DDD', 'rating': 'BUY', 'sector': 'Energy'}]})
        f = tmp_path / 'pf.json'
        _run(f, 'create', 'e', '--name', 'E', '--sector', 'Energy', '--tickers', 'CCC,GONE')
        _run(f, 'remove', 'e', 'DDD')
        capsys.readouterr()
        _run(f, 'show', 'e', '--results-dir', str(res))
        out = capsys.readouterr().out
        assert 'Snapshot 2026-09-14' in out and '3 members, 1 not in universe' in out
        assert re.search(r'CCC .* pick', out) and re.search(r'AAA .* rule', out)
        assert 'DDD' not in out.split('rule:')[1].split('\n', 1)[1].replace('not in universe', '')


# ---------------------------------------------------------------- report

class TestReport:
    def test_rows_carry_membership_and_payload(self, tmp_path, monkeypatch):
        from scripts.report_html import build_html
        pf = tmp_path / 'pf.json'
        pg.save_portfolios({'version': 1, 'portfolios': [
            {'id': 'mine', 'name': 'Mine', 'tickers': ['AAA', 'NOPE']},
            {'id': 'buys', 'name': 'BUYs', 'rule': {'ratings': ['BUY']}}]}, str(pf))
        monkeypatch.setenv('PORTFOLIOS_FILE', str(pf))
        out = tmp_path / 'report.html'
        build_html([dict(r) for r in ROWS], str(out), prices_dir=None)
        html = out.read_text(encoding='utf-8')
        m = re.search(r'var PF_PUB=\(function\(\)\{var p=(.*?);return', html)
        payload = json.loads(m.group(1))
        assert payload['rev'] == pg.revision(pg.load_portfolios(str(pf)))
        assert [p['id'] for p in payload['portfolios']] == ['mine', 'buys']
        assert payload['portfolios'][0]['missing'] == ['NOPE']
        data = json.loads(re.search(r'var DATA=(\[.*?\]);\n', html).group(1))
        assert {d['ticker']: d['pf'] for d in data} == {'AAA': ['mine', 'buys'], 'BBB': [], 'CCC': ['buys']}

    def test_bad_file_renders_without_portfolios(self, tmp_path, monkeypatch):
        from scripts.report_html import build_html
        pf = tmp_path / 'pf.json'
        pf.write_text('{"version": 1, "portfolios": [{"id": "BAD"}]}', encoding='utf-8')
        monkeypatch.setenv('PORTFOLIOS_FILE', str(pf))
        out = tmp_path / 'report.html'
        build_html([dict(r) for r in ROWS], str(out), prices_dir=None)
        html = out.read_text(encoding='utf-8')
        payload = json.loads(re.search(r'var PF_PUB=\(function\(\)\{var p=(.*?);return', html).group(1))
        assert payload['portfolios'] == [] and 'BAD' in payload['error']

    @pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
    def test_js_evaluator_matches_fixture(self, tmp_path):
        """Run the report's own _num/_pfRuleMatch over the shared fixture."""
        src = open(TEMPLATE, encoding='utf-8').read()
        num = re.search(r'^function _num\(v\)\{.*?\}$', src, re.M).group(0)
        start = src.index('function _pfRuleMatch(r,d){')
        end = src.index('\n}\n', start) + 3
        js = (num + '\n' + src[start:end]
              + '\nvar fx=' + json.dumps(_cases()) + ';\n'
              + 'var bad=fx.cases.filter(function(c){var got=fx.rows.filter(function(d){'
              + 'return _pfRuleMatch(c.rule,d);}).map(function(d){return d.ticker;});'
              + 'return JSON.stringify(got)!==JSON.stringify(c.expect);}).map(function(c){return c.name;});\n'
              + 'if(bad.length){console.log(JSON.stringify(bad));process.exit(1);}\n')
        script = tmp_path / 'parity.js'
        script.write_text(js, encoding='utf-8')
        r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
