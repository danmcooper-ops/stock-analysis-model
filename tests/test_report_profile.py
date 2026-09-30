# tests/test_report_profile.py
"""The ticker page's one-page Summary PDF (templates/report.html).

Source-level assertions for the wiring (the "Summary" button beside the
ticker, the print path), a render check that the verdict rides the details/
parts, and a Node-evaluated run of the renderer on a full row and an empty
one, since a throw there prints a blank page.
"""
import json
import re
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


def test_there_is_no_profile_tab():
    """The one-pager is a PDF only: no tab, no pane, no on-screen renderer."""
    src = _tpl()
    for gone in ('DT_PROFILE', 'd-pane-profile', 'id="d-profile"', '_renderDetProfile',
                 'pro-print-btn', 'pro-tools'):
        assert gone not in src, gone


def test_summary_button_sits_right_of_the_ticker_symbol():
    """The button sits directly after the ticker symbol in the header band,
    not on the sub-tab row."""
    src = _tpl()
    band = re.search(r'<div class="dh-left">(.*?)</div></div>', src).group(1)
    tk, btn = band.index('id="d-tk"'), band.index('id="d-sum"')
    assert tk < btn < band.index('class="det-nav-pos"')
    button = band[band.rindex('<button', 0, btn):band.index('</button>', btn)]
    assert 'onclick="exportProfilePdf()"' in button
    assert button.endswith('Summary')      # text label after the download glyph
    assert '<svg' in button and 'aria-hidden="true"' in button
    assert 'sa-chrome' in button           # chrome colours survive dark mode
    assert '#det-modal .d-sum-btn{--sac:var(--chrome-heading);' in src
    assert 'd-tabs-row' not in src


def test_pdf_waits_for_the_verdict_history_and_prices():
    exp = _fn(_tpl(), 'exportProfilePdf')
    details = exp.index('if(_DETAILS_AVAILABLE&&!_DETAILS_LOADED){_loadDetails(exportProfilePdf);return;}')
    hist = exp.index('_ensureHist([tk],exportProfilePdf)')
    px = exp.index('_pfWithPx([tk]')
    assert details < hist < px < exp.index('print()')


def test_print_goes_through_a_hidden_frame_on_one_letter_page():
    src = _tpl()
    exp = _fn(src, 'exportProfilePdf')
    assert "fr.id='pro-print-frame';" in exp
    assert 'fr.contentWindow.print()' in exp
    assert 'width:8.5in;height:11in' in exp   # measured at real page size
    doc = _fn(src, '_proPrintDoc')
    assert "document.getElementById('pro-css')" in doc
    assert '@page{size:letter portrait;margin:8mm}' in doc
    # Shrink-to-fit steps the root font size; everything is sized in em.
    assert 'r.style.fontSize=s+"px"' in doc and 'b.scrollHeight>H' in doc
    sheet = src[src.index('<style id="pro-css">'):]
    sheet = sheet[:sheet.index('</style>')]
    # Neither prints the same everywhere: Safari left a multi-column page
    # half empty and dropped blocks, so the layout uses neither.
    for text in (sheet, doc):
        assert 'zoom:' not in text and 'column-count' not in text
        assert not re.search(r'(?<![-\w])columns\s*:', text)


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
        '_proPxAt', '_proPriceChart', '_proSpark', '_proHistHtml', '_proPeers',
        '_proScorecard', '_proRisks', '_proBalance', '_proRangeLine', '_proHtml'))
    consts = ("var _PRO_VCLS={'INVEST':'pro-v-inv','WATCH':'pro-v-wat','AVOID':'pro-v-avo',"
              "'INSUFFICIENT DATA':'pro-v-na'};\n")
    assert consts.strip() in src
    full = _row(dcf_sens_range=[110, 150], mc_p10_fv=90, mc_p90_fv=170, target_low=80,
                target_high=160, target_mean=125, low_52w=70, high_52w=120, epv_fv=60,
                _gate_spread=0.16, _gp_spread=True, _score_moat=72,
                trap_reasons=['<script>x</script>'], sector_tailwinds=[{'text': 'AI demand'}],
                roic_by_year={'2023': 0.2, '2024': 0.22, '2025': 0.25},
                epv_growth_fv=60.0, rim_fv=58.0, ddm_fv=51.0, _dcf_fv_preblend=130.0,
                _gate_fv_dispersion=0.35)
    full['profile'] = profile_verdict(full)
    hist = {'rev': {'2023-12-31': 8e9, '2024-12-31': 9e9, '2025-12-31': 10e9},
            'op': {'2023-12-31': 2e9, '2024-12-31': 2.4e9, '2025-12-31': 3e9},
            'ni': {'2023-12-31': 1e9, '2024-12-31': 1.5e9, '2025-12-31': 2e9},
            'ocf': {'2023-12-31': 1.5e9, '2024-12-31': 2e9, '2025-12-31': 2.6e9},
            'capex': {'2023-12-31': -3e8, '2024-12-31': -3.5e8, '2025-12-31': -4e8},
            'dna': {'2023-12-31': 2e8, '2024-12-31': 2.2e8, '2025-12-31': 2.4e8},
            'shares': {'2023-12-31': 5e8, '2024-12-31': 4.9e8, '2025-12-31': 4.8e8}}
    peer = _row(ticker='PEER', company_name='Peer Co', mcap=4e10, pe=18.0)
    n = 300
    prices = {'dates': [f'{2025 + i // 252}-01-{1 + i % 28:02d}' for i in range(n)],
              'prices': {'PRO': [50 + i * 0.2 for i in range(n)],
                         'SPY': [400 + i * 0.5 for i in range(n)]}}
    js = (_JS_PRELUDE + consts + fns
          + f'HIST.PRO={json.dumps(hist)};\n'
          + f'var DATA={json.dumps([full, peer])};\n'
          + f'var PRICES={json.dumps(prices)};\n'
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
    assert 'class="pro-px"' in a              # price history vs S&P 500
    assert 'Price return vs S&amp;P 500' in a
    assert '<td>PEER</td>' in a and 'pro-self' in a   # peers, company first
    # Valuation confidence: the fair value as a range, with the DCF's gap to
    # the other models and a level that is a word, not just a colour.
    assert 'Intrinsic value <b>$90.00 \u2013 $170</b>' in a
    assert 'Base <b>$130</b> (DCF)' in a and 'Bear <b>$90.00</b>' in a and 'Bull <b>$170</b>' in a
    assert 'Other models\u2019 median <b>$58.00</b> (DCF +124%)' in a
    assert 'pro-conf-LOW">LOW<' in a
    assert 'DCF vs other models' in a and 'Model dispersion (MAD)' in a
    assert 'range $90.00\u2013$170' in a                 # fair-value tile
    # Per-share and cash quality in the ten-year record, and the bridge.
    for row in ('FCF per share', 'FCF / net income', 'CFO / EBITDA'):
        assert f'<td>{row}</td>' in a, row
    assert 'class="pro-bridge"' in a and 'FCF per share' in a and '\u00f7 share count' in a
    cols = a[a.index('class="pro-cols"'):]
    assert cols.count('<section class="pro-b"') >= 5
    for empty in (b, c):
        assert 'class="pro"' in empty
        # Blocks with no data disappear.
        for gone in ('Valuation models', 'Ten-year', 'class="pro-px"', 'Peers',
                     'pro-range', 'pro-bridge', 'DCF vs other models'):
            assert gone not in empty, gone
