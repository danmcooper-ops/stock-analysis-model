"""The currency_report_<date>.json contract the stock report draws.

The parent repo's report (Investing › Currencies) reads this file and
nothing else from this project, so its shape is pinned here: a change that
breaks it must bump REPORT_DATA_VERSION, which the reader checks.
"""

import json
import math
from datetime import date

from scripts import report_html
from scripts.scoring import gate_metadata


def _rows():
    return [
        {'code': 'JPY', 'name': 'Japanese yen', 'regime': 'float',
         'rating': 'LONG', 'rating_raw': 'LONG', '_composite_score': 71.5,
         'reer_z20': -2.1, 'spark': [['2026-01-02', 0.0064], ['2026-10-27', 0.0066]],
         'internal_working_value': 1},
        {'code': 'HKD', 'name': 'Hong Kong dollar', 'regime': 'peg',
         'rating': 'NEUTRAL', 'rating_raw': 'LONG', '_rating_cap': 'NEUTRAL',
         '_rating_cap_reasons': ['Pegged regime'], '_composite_score': float('nan'),
         'vol_1y': float('inf')},
    ]


def test_report_data_shape():
    d = report_html.report_data(_rows(), date(2026, 10, 27), {'generated_at': 'x'})
    assert d['version'] == report_html.REPORT_DATA_VERSION
    assert d['run_date'] == '2026-10-27'
    assert d['total'] == 2
    assert d['gate_meta']['ratings'] == gate_metadata()['ratings']
    assert [r['code'] for r in d['rows']] == ['JPY', 'HKD']
    # Only the page's fields travel; working values stay in results JSON.
    assert 'internal_working_value' not in d['rows'][0]
    # Every gate's value, pass flag and score key is present on each row.
    g = d['gate_meta']['gates'][0]
    assert {g['key'], g['gpKey'], g['scoreKey']} <= set(d['rows'][0])
    assert d['world_svg'].startswith('<svg') and d['world_svg'].endswith('</svg>')


def test_non_finite_values_become_null():
    d = report_html.report_data(_rows(), date(2026, 10, 27))
    assert d['rows'][1]['_composite_score'] is None
    assert d['rows'][1]['vol_1y'] is None


def test_write_report_data_is_strict_json(tmp_path):
    path = report_html.write_report_data(_rows(), str(tmp_path), date(2026, 10, 27))
    assert path.endswith('currency_report_2026-10-27.json')
    with open(path, encoding='utf-8') as f:
        loaded = json.load(f, parse_constant=lambda c: math.nan)
    assert loaded['rows'][0]['code'] == 'JPY'
    assert not list(tmp_path.glob('*.tmp'))


def test_build_html_writes_the_data_file_beside_the_page(tmp_path):
    page = tmp_path / 'currency_screener_2026-10-27.html'
    report_html.build_html(_rows(), str(page), run_date=date(2026, 10, 27))
    assert page.exists()
    assert (tmp_path / 'currency_report_2026-10-27.json').exists()
