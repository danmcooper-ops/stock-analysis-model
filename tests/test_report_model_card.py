"""The ticker page's Model card (_gvRender): composite against the rating
lines, one row per category with its failing gates as chips, and the gates
with short targets. Display only."""
import json
import os
import re
import shutil
import subprocess

import pytest

from score_band_js import band_js

from scripts import param_set
from scripts.scoring import gate_metadata

TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')
node = pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')


def _src():
    return open(TEMPLATE, encoding='utf-8').read()


def _fn(src, name):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M).group(0)


def test_every_gate_has_a_short_target():
    for g in gate_metadata()['gates']:
        assert g['target'], g['key']
        assert len(g['target']) <= 24, g


def test_rating_lines_follow_params():
    lines = gate_metadata({'rating_threshold_buy': 60})['ratings']
    assert [x['rating'] for x in lines] == ['BUY', 'LEAN BUY', 'HOLD']
    assert [x['min'] for x in lines] == [60, 39, 25]


def test_display_metadata_does_not_move_the_scoring_hash(monkeypatch):
    from scripts import scoring
    before = param_set.scoring_params_hash()
    monkeypatch.setitem(scoring._GATE_DISPLAY['spread'], 'target', '> 99%')
    assert param_set.scoring_params_hash() == before


def _render(tmp_path, row):
    src = _src()
    gm = gate_metadata()
    js = '\n'.join([
        'var GM=' + json.dumps(gm) + ';',
        'function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}',
        'function _attr(s){return esc(s).replace(/"/g,"&quot;");}',
        'function isCapped(d){return!!(d&&d.rating_raw&&d.rating&&d.rating!==d.rating_raw);}',
        'function _gvState(){return {};}',
        re.search(r'^var _GV_RC=.*$', src, re.M).group(0),
        _fn(src, '_gvFmt'), _fn(src, '_gvGates'), _fn(src, '_gvLines'), _fn(src, '_gvRender'),
        'var row=' + json.dumps(row) + ';',
        'console.log(JSON.stringify({h:_gvRender(row),'
        'l:[24,40,72].map(function(c){return _gvLines(c,GM.ratings);})}));',
    ])
    script = tmp_path / 't.js'
    script.write_text(band_js(_src()) + js, encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


def _row():
    gm = gate_metadata()
    moat = [g for g in gm['gates'] if g['category'] == 'Moat']
    row = {'ticker': 'X', 'rating': 'BUY', '_composite_score_raw': 72.0,
           '_rating_cap_reasons': ['ADV < $1M']}
    for c in gm['categories']:
        row[c['scoreKey']] = 80.0
    for g in gm['gates']:
        row[g['key']], row[g['gpKey']], row[g['scoreKey']] = 0.2, True, 90.0
    fail, na = moat[1], moat[2]
    row[fail['key']], row[fail['gpKey']], row[fail['scoreKey']] = 2.26, False, 0.0
    row[na['key']], row[na['gpKey']], row[na['scoreKey']] = None, None, None
    return row, fail, na


@node
def test_folded_row_is_name_passes_and_score_and_na_is_not_a_failure(tmp_path):
    row, fail, na = _row()
    h = _render(tmp_path, row)['h']
    # Folded, a category is one line (the iOS app's): name, gates passed,
    # score. The bar and the weight sit in the same header and show only
    # once it is opened (CSS: .gv-shut hides them); no failing-gate chips.
    assert 'gv-chip' not in h
    moat = h.split('data-gvc="Moat"')[1].split('</div>')[0]
    order = [moat.index(k) for k in ('gv-cat-t', 'gv-trk', 'gv-cat-w', 'gv-cat-p', 'gv-cat-s', 'ppf-chev')]
    assert order == sorted(order)
    moat_n = sum(g['category'] == 'Moat' for g in gate_metadata()['gates'])
    assert f'{moat_n - 2}/{moat_n - 1} pass' in moat and 'gv-allpass' not in moat
    assert '1 N/A, counted 0' in moat
    # The failing gate is listed among the gates, with its value and target.
    item = h.split('class="gv-item fail"')[1].split('</div>')[0]
    assert fail['label'] in item and '226.0%' in item
    assert "doesn't apply" in h
    assert '% of score' in h and '* 30% weight' not in h
    assert '\u26a0 Flag</span><ul><li>ADV &lt; $1M</li></ul>' in h


def test_folded_row_hides_the_bar_and_weight():
    src = _src()
    assert '.gv-shut .gv-cat .gv-trk,.gv-shut .gv-cat-w{display:none;}' in src
    assert '.gv-shut .gv-cat{grid-template-columns:minmax(0,1fr) auto auto 12px;}' in src


@node
def test_score_line_names_the_nearest_lines(tmp_path):
    row, _, _ = _row()
    out = _render(tmp_path, row)
    l24, l40, l72 = out['l']
    assert l24 == [{'k': 'short', 'r': 'HOLD', 'm': 25, 'n': '1'}]
    assert l40 == [{'k': 'above', 'r': 'LEAN BUY', 'm': 39, 'n': '1'},
                   {'k': 'short', 'r': 'BUY', 'm': 57, 'n': '17'}]
    assert l72 == [{'k': 'above', 'r': 'BUY', 'm': 57, 'n': '15'}]
    assert '72<small> / 100' in out['h']
    # The card names the line's score, never the rating.
    assert 'above the 57 line' in out['h']
    for word in ('>BUY<', 'LEAN BUY', 'gv-sc-r'):
        assert word not in out['h']


@node
def test_score_keeps_a_decimal_when_rounding_would_cross_a_line(tmp_path):
    row, _, _ = _row()
    row['_composite_score_raw'] = 56.6
    row['rating'] = 'LEAN BUY'
    assert '56.6<small>' in _render(tmp_path, row)['h']
