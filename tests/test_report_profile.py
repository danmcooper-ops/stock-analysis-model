# tests/test_report_profile.py
"""The ticker page's one-page Profile tab (templates/report.html).

Source-level assertions for the wiring (tab order, pane routing, print
path), a render check that the verdict rides the details/ parts, and a
Node-evaluated run of the renderer on a full row and an empty one, since a
throw there blanks the whole tab.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.report_html import build_html

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn(src, name):
    """Source of a top-level `function name(...){...}` (ends at the first
    line that is exactly `}`)."""
    start = src.index('function ' + name + '(')
    return src[start:src.index('\n}\n', start) + 2]


def test_profile_tab_sits_after_summary_and_routes_to_its_pane():
    src = _tpl()
    assert "var DT_PROFILE={k:'profile',l:'Profile'};" in src
    assert "return [{k:'overview',l:'Overview'},DT_PROFILE,DT_FIN].concat(_detDataItems());" in src
    assert '<div id="d-pane-profile" class="d-pane"><div id="d-profile"></div></div>' in src
    sw = _fn(src, 'swDetTab')
    assert "k===DT_PROFILE.k?'d-pane-profile'" in sw
    assert '_renderDetProfile(_pd)' in sw
    assert "if(k===DT_PROFILE.k)return 'profile';" in _fn(src, '_detGroupOf')


def test_profile_rerenders_when_its_history_shard_lands():
    body = _fn(_tpl(), '_renderDetProfile')
    assert '_ensureHist([tk]' in body
    assert '_dpTicker===tk&&_detTab===DT_PROFILE.k' in body


def test_print_goes_through_a_hidden_frame_on_one_letter_page():
    src = _tpl()
    assert 'onclick="exportProfilePdf()"' in src
    exp = _fn(src, 'exportProfilePdf')
    assert "fr.id='pro-print-frame';" in exp
    assert 'fr.contentWindow.print()' in exp
    doc = _fn(src, '_proPrintDoc')
    # The frame reuses the tab's own stylesheet, so paper and screen agree.
    assert "document.getElementById('pro-css')" in doc
    assert '@page{size:letter portrait;margin:8mm}' in doc
    assert 'b.style.zoom=' in doc          # shrink-to-fit fallback
    assert '<style id="pro-css">' in src


def test_dark_mode_restores_semantic_colours_past_the_blanket_rule():
    src = _tpl()
    assert '[data-theme="dark"] .pro{--pro-pos:' in src
    assert '[data-theme="dark"] #d-profile .pos,' in src
    assert '[data-theme="dark"] #d-profile .neg,' in src


def _row(**over):
    row = {
        'ticker': 'PRO', 'company_name': 'Profile Co', 'sector': 'Technology',
        'industry': 'Software', 'country': 'United States',
        'price': 100.0, 'mcap': 5e10, '_fv_effective': 130.0, '_fv_source': 'dcf',
        'dcf_fv': 130.0, 'mos': 0.23, 'rating': 'BUY', 'rating_raw': 'BUY',
        '_rating_cap_reasons': [], '_composite_score': 64.0, '_gates_passed': '9/12',
        'roic': 0.25, 'wacc': 0.09, 'spread': 0.16, 'fcf_margin': 0.22,
        'rev_cagr_5y': 0.11, 'net_debt': -5e9, 'piotroski': 7, 'pe': 20.0,
        'altman_z_zone': 'safe', 'mc_confidence': 'MEDIUM (CV 28%)',
    }
    row.update(over)
    return row


def test_verdict_rides_the_details_parts(tmp_path):
    out = tmp_path / 'index.html'
    build_html([_row(), _row(ticker='NOPX', price=None, mos=None)], str(out), prices_dir=None)
    parts = json.loads((tmp_path / 'details_index.json').read_text(encoding='utf-8'))['parts']
    details = {}
    for p in parts:
        details.update(json.loads((tmp_path / 'details' / f'{p}.json').read_text(encoding='utf-8')))
    assert details['PRO']['profile']['v'] == 'INVEST'
    assert details['NOPX']['profile']['v'] == 'INSUFFICIENT DATA'
    # Not inlined into the page: the verdict costs index.html nothing.
    assert '"buy_below"' not in out.read_text(encoding='utf-8')


_JS_PRELUDE = r"""
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function _attr(s){return esc(s).replace(/"/g,'&quot;');}
function _num(v){if(v==null)return null;if(typeof v==='number')return isFinite(v)?v:null;var n=+v;return isFinite(n)?n:null;}
function fds(v){var n=_num(v);if(n==null)return'N/A';return (n<0?'-':'')+'$'+(Math.abs(n)/1e9).toFixed(1)+'B';}
function _trFmtSh(v){return String(v);}
function isCapped(d){return!!(d&&d.rating_raw&&d.rating&&d.rating!==d.rating_raw);}
var HIST={},_HIST_SET={PRO:1};
function _histHas(tk){return !!(_HIST_SET[tk]&&HIST[tk]);}
var RUN_DATE='2026-09-24',_DETAILS_AVAILABLE=true,_DETAILS_LOADED=true;
var GM={categories:[{name:'Moat',scoreKey:'_score_moat',weight:0.3}],
        gates:[{category:'Moat',key:'_gate_spread',gpKey:'_gp_spread',label:'Spread',fmt:'pct1',threshold:'> 7%'},
               {category:'Moat',key:'_gate_x',gpKey:'_gp_x',label:'X',fmt:'ratio'}]};
"""


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_renderer_handles_a_full_row_and_an_empty_one(tmp_path):
    from models.profile_verdict import profile_verdict
    src = _tpl()
    fns = ''.join(_fn(src, n) + '\n' for n in (
        '_proFmt', '_proKv', '_proT', '_proBlock', '_proMed', '_proRel', '_proFootball',
        '_proSpark', '_proHistHtml', '_proScorecard', '_proRisks', '_proHtml'))
    consts = ("var _PRO_VCLS={'INVEST':'pro-v-inv','WATCH':'pro-v-wat','AVOID':'pro-v-avo',"
              "'INSUFFICIENT DATA':'pro-v-na'};\n")
    assert consts.strip() in src
    full = _row(dcf_sens_range=[110, 150], mc_p10_fv=90, mc_p90_fv=170, target_low=80,
                target_high=160, target_mean=125, low_52w=70, high_52w=120, epv_fv=60,
                _gate_spread=0.16, _gp_spread=True, _score_moat=72,
                trap_reasons=['<script>x</script>'], sector_tailwinds=[{'text': 'AI demand'}],
                roic_by_year={'2023': 0.2, '2024': 0.22, '2025': 0.25})
    full['profile'] = profile_verdict(full)
    hist = {'rev': {'2023-12-31': 8e9, '2024-12-31': 9e9, '2025-12-31': 10e9},
            'op': {'2023-12-31': 2e9, '2024-12-31': 2.4e9, '2025-12-31': 3e9},
            'ni': {'2023-12-31': 1e9, '2024-12-31': 1.5e9, '2025-12-31': 2e9}}
    js = (_JS_PRELUDE + consts + fns
          + f'HIST.PRO={json.dumps(hist)};\n'
          + f'var a=_proHtml({json.dumps(full)});\n'
          + "var b=_proHtml({ticker:'EMPTY'});\n"
          + "var c=_proHtml({ticker:'NAN',price:NaN,mos:'x',pe:Infinity,_fv_effective:null,"
            "dcf_sens_range:'bad',trap_reasons:[null,{text:'t'}]});\n"
          + 'console.log(JSON.stringify([a,b,c]));\n')
    script = tmp_path / 'profile.js'
    script.write_text(js, encoding='utf-8')
    r = subprocess.run(['node', str(script)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    a, b, c = json.loads(r.stdout)
    assert 'pro-v-inv' in a and '>INVEST<' in a
    assert 'class="pro-ff"' in a              # fair-value range chart
    assert 'Ten-year record' in a and 'FY25' in a
    assert 'pro-g pass' in a                  # scorecard chip
    assert '<script>x' not in a and '&lt;script&gt;x' in a   # reasons are escaped
    assert 'Tailwind: AI demand' in a
    for empty in (b, c):
        assert 'class="pro"' in empty
        assert 'Valuation' not in empty      # blocks with no data disappear
