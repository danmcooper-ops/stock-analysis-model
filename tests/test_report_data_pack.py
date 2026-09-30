# tests/test_report_data_pack.py
"""The report's row payload ships column-wise (report_html.pack_rows) and the
template's _unpackRows rebuilds it. The page must see exactly the rows it
saw before packing: absent keys stay absent, nulls stay null, and floats
change only past the tenth significant digit."""
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.report_html import build_html, pack_rows, unpack_rows

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'

ROWS = [
    {'ticker': 'AAA', 'price': 123.45678901234, 'pe': None, 'hist': [0.1 + 0.2, 1.0, None],
     'nested': {'x': 2.0 / 3.0, 's': 'a'}, 'only_a': 1},
    {'ticker': 'BBB', 'price': 12345.67, 'pe': 14.2, 'hist': [], 'nested': {}},
    {'ticker': 'CCC', 'price': 0.001234567890123, 'pe': 7, 'hist': [2.5],
     'nested': {'x': None}, 'mostly': True},
    {'ticker': 'DDD', 'price': None, 'pe': 3.0, 'hist': None, 'nested': None, 'mostly': False},
    {'ticker': 'EEE', 'pe': 1.5, 'hist': [], 'nested': {}, 'mostly': None},
]


def _trimmed(v):
    if isinstance(v, float):
        return float(f'{v:.10g}') if math.isfinite(v) else v
    if isinstance(v, dict):
        return {k: _trimmed(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_trimmed(x) for x in v]
    return v


def test_round_trip_keeps_absent_null_and_values():
    packed = pack_rows(ROWS)
    assert unpack_rows(json.loads(json.dumps(packed))) == [_trimmed(r) for r in ROWS]
    # Rare keys list where they are present, common ones where they are not.
    k = packed['k']
    assert packed['p'][str(k.index('only_a'))] == [0]
    assert packed['a'][str(k.index('price'))] == [4]
    assert str(k.index('ticker')) not in packed['p'] and str(k.index('ticker')) not in packed['a']
    # Each key name appears once however many rows carry it.
    assert json.dumps(packed).count('"ticker"') == 1


def test_float_trim_is_beyond_display_precision():
    packed = pack_rows([{'a': 0.1 + 0.2, 'b': 12345.678901234567, 'c': 7, 'd': float('nan'),
                         'e': 2345678901234.5}])
    col = {k: packed['c'][i][0] for i, k in enumerate(packed['k'])}
    assert col['a'] == 0.3
    assert col['b'] == 12345.6789
    assert col['c'] == 7 and isinstance(col['c'], int)
    assert math.isnan(col['d'])
    assert abs(col['e'] - 2345678901234.5) / 2345678901234.5 < 1e-9


def test_empty_payload():
    assert unpack_rows(pack_rows([])) == []


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_template_decoder_matches_python(tmp_path):
    src = TEMPLATE.read_text(encoding='utf-8')
    fn = re.search(r'(function _unpackRows\(p\)\{.*?\n\})\n', src, re.S).group(1)
    packed = pack_rows(ROWS)
    js = (fn + '\nvar P=' + json.dumps(packed) + ';\n'
          'var R=_unpackRows(P);\n'
          # Absent keys must be absent, not undefined-valued.
          'console.log(JSON.stringify({rows:R,has:R.map(function(r){return Object.keys(r).sort();}),'
          'legacy:_unpackRows([{"ticker":"Z"}])}));\n')
    script = tmp_path / 'unpack.js'
    script.write_text(js, encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    want = unpack_rows(packed)
    assert out['rows'] == want
    assert out['has'] == [sorted(w) for w in want]
    assert out['legacy'] == [{'ticker': 'Z'}]


def test_rendered_report_ships_packed_rows(tmp_path):
    out = tmp_path / 'report.html'
    build_html([{'ticker': 'AAA', 'price': 1.0, 'rating': 'HOLD'},
                {'ticker': 'BBB', 'price': 2.5, 'rating': 'BUY'}], str(out), prices_dir=None)
    html = out.read_text(encoding='utf-8')
    m = re.search(r'var DATA=_unpackRows\((\{.*?\})\);\n', html)
    assert m, 'packed DATA not found'
    rows = unpack_rows(json.loads(m.group(1)))
    assert [r['ticker'] for r in rows] == ['AAA', 'BBB']
    assert rows[1]['price'] == 2.5
