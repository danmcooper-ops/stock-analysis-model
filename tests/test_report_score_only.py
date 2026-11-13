"""The report shows the composite score, never a rating (2026-11-12).

The template draws its labels in JavaScript, so a search of the rendered
HTML for "BUY" proves nothing. These are source-level guards on the hooks
that used to show a rating, plus the band helpers run in node. The engine
still computes, stores and publishes every rating field (see
tests/test_app_feed.py); only what the page shows changed."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from score_band_js import band_js

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def test_no_rating_pill_on_any_stock():
    src = _tpl()
    # rb() drew a rating's name in a pill; every stock now gets sb()/sbv().
    for call in ('rb(d.rating', 'rb(x.rating', 'rb(pd.rating', 'rb(stock.rating',
                 'rb(r,d)+', 'rb(v)}'):
        assert call not in src, call
    assert "{k:'rating',l:'Rating'" not in src
    assert src.count("{k:'_composite_score',l:'Score',f:'score'}") == 8


def test_ticker_header_shows_the_score_not_the_rating():
    src = _tpl()
    assert "rEl.textContent='';" in src and "rEl.style.display='none';" in src
    assert "cEl.textContent=cs!=null?'Score '+Math.round(cs)" in src
    assert 'Model rating' not in src
    assert "'Model: <b>'+esc(d.rating" not in src


def test_visible_labels_name_score_bands():
    src = _tpl()
    for label in ('> 57+</label>', '> 39–56</label>', '> 25–38</label>', '> &lt;25</label>'):
        assert label in src
    for gone in ('> BUY</label>', 'All Ratings', '>Upgrades</button>', '>Downgrades</button>',
                 "'BUY %'", 'BUY rated', 'Rating mix', 'Recent rating changes',
                 "colored by actual rating", 'Today\\u2019s Rating'):
        assert gone not in src, gone


def test_analyst_answers_in_scores():
    src = _tpl()
    assert 'Never call a stock BUY, LEAN BUY, HOLD or PASS' in src
    assert "Rating values: BUY, LEAN BUY, HOLD, PASS" not in src
    assert '.then(_aiNoRating)' in src


def test_alerts_show_their_score_wording():
    src = _tpl()
    assert 'a.display' in src and 's.display||s.message' in src


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_band_helpers(tmp_path):
    src = _tpl()
    js = band_js(src) + (
        "var RC={'BUY':'#1','LEAN BUY':'#2','HOLD':'#3','PASS':'#4'};"
        "var GM={ratings:[{rating:'BUY',min:57},{rating:'LEAN BUY',min:39},{rating:'HOLD',min:25}]};"
        "console.log(JSON.stringify(["
        "['BUY','LEAN BUY','HOLD','PASS'].map(bandLabel),"
        "[bandKey(57),bandKey(56.9),bandKey(39),bandKey(24.9),bandKey(null)],"
        "scoreCross({_composite_score:41,score_prev:36.2}),"
        "scoreCross({_composite_score:55,score_prev:58}),"
        "scoreCross({_composite_score:41,score_prev:40}),"
        "scoreCross({_composite_score:41}),"
        "crossText({_composite_score:41,score_prev:36.2}),"
        "crossText({_composite_score:41,score_prev:36.2,score_prev_basis:'stored'}),"
        "bandsAsRange(['BUY','LEAN BUY']),bandsAsRange(['HOLD','PASS']),bandsAsRange(['BUY','PASS']),"
        "sb({_composite_score:64.2,rating:'HOLD',rating_raw:'BUY',_rating_cap_reasons:['x']})"
        "]));")
    script = tmp_path / 'b.js'
    script.write_text(js, encoding='utf-8')
    out = subprocess.run(['node', str(script)], capture_output=True, text=True, check=True).stdout
    import json
    r = json.loads(out)
    assert r[0] == ['57+', '39–56', '25–38', '<25']
    assert r[1] == ['BUY', 'LEAN BUY', 'LEAN BUY', 'PASS', None]
    assert r[2:6] == [1, -1, 0, None]
    assert r[6] == 'Crossed above 39 (36.2 → 41.0)'
    assert r[7].endswith('the scoring model changed since the prior run')
    assert r[8:11] == ['score ≥ 39', 'score < 39', None]
    assert re.search(r'>64</span><span class="rb-cap"', r[11])
    assert 'BUY' not in r[11].split('title=')[0]
