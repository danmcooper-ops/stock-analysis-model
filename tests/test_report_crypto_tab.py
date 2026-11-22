"""Investing › Crypto: scripts/crypto_market.py's report staged beside the
page as crypto.json and drawn as a market table. Staging, wiring, the
display-only rule, and the renderer evaluated in Node with its helpers
lifted from the template."""
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from scripts import report_html

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'templates' / 'report.html'


def _tpl():
    return TEMPLATE.read_text(encoding='utf-8')


def _fn_src(name, src):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M).group(0)


def _row(sym, name, rank, **kw):
    r = {'id': name.lower(), 'symbol': sym, 'name': name, 'rank': rank, 'price': 1.0,
         'market_cap': 1e9, 'weight': 0.1, 'volume_24h': 1e8, 'chg_24h': 0.01, 'chg_7d': 0.02,
         'chg_30d': 0.03, 'ytd': 0.04, 'chg_1y': 0.05, 'vol_30d': 0.5, 'vol_1y': 0.6,
         'max_dd_1y': -0.4, 'off_high_1y': -0.2, 'ath_off': -0.5, 'ath_date': '2025-10-06',
         'corr_btc_1y': 0.8, 'corr_spy_1y': 0.3, 'fdv': 1.1e9, 'turnover': 0.1,
         'circulating_supply': 900, 'max_supply': 1000, 'supply_issued': 0.9,
         'history_days': 366, 'spark': [['2025-11-20', 1.0], ['2026-11-20', 2.0]]}
    r.update(kw)
    return r


REPORT = {
    'version': 1, 'run_date': '2026-11-20', 'generated': '2026-11-20 17:02 EST', 'total': 3,
    'top_n': 3, 'source': 'CoinGecko', 'errors': [],
    'ui': {'price': 'Last price <usd>.', 'chg_7d': 'Seven days.'},
    'excluded': [{'id': 'tether', 'symbol': 'USDT', 'name': 'Tether', 'rank': 3,
                  'reason': 'stablecoin (Stablecoins)'}],
    'rows': [
        _row('BTC', 'Bitcoin', 1, price=60863.0, market_cap=1.22e12, weight=0.8, chg_7d=-0.063,
             chg_1y=-0.28, corr_btc_1y=1.0),
        _row('ETH', 'Ethereum', 2, price=1745.5, market_cap=2.1e11, weight=0.15, chg_7d=0.041,
             chg_1y=0.12, corr_spy_1y=None),
        _row('DOGE', 'Dogecoin', 13, price=0.0571, market_cap=8.7e9, weight=0.05, chg_7d=0.0, volume_24h=689559.0,
             max_supply=None, supply_issued=None),
    ],
}


def _write(d, dates, **over):
    for day in dates:
        (d / f'crypto_report_{day}.json').write_text(
            json.dumps(dict(REPORT, run_date=day, **over)), encoding='utf-8')


def test_stage_picks_newest_on_or_before_the_run(tmp_path):
    src, out = tmp_path / 'crypto', tmp_path / 'out'
    src.mkdir()
    out.mkdir()
    _write(src, ['2026-11-18', '2026-11-20', '2026-11-21'])
    meta = report_html._stage_crypto_report(str(out), date(2026, 11, 20), str(src))
    assert meta == {'date': '2026-11-20', 'total': 3,
                    'index': [['BTC', 'Bitcoin'], ['ETH', 'Ethereum'], ['DOGE', 'Dogecoin']]}
    assert json.loads((out / 'crypto.json').read_text(encoding='utf-8'))['run_date'] == '2026-11-20'


def test_stage_removes_a_stale_copy_and_refuses_another_shape(tmp_path):
    src, out = tmp_path / 'crypto', tmp_path / 'out'
    src.mkdir()
    out.mkdir()
    (out / 'crypto.json').write_text('{}', encoding='utf-8')
    _write(src, ['2026-11-20'], version=2)
    assert report_html._stage_crypto_report(str(out), date(2026, 11, 20), str(src)) is None
    assert not (out / 'crypto.json').exists()


def test_versions_agree_and_default_source(monkeypatch):
    from models import crypto_market
    assert report_html.CRYPTO_REPORT_VERSION == crypto_market.REPORT_VERSION
    monkeypatch.delenv('CRYPTO_MODEL_OUTPUT', raising=False)
    assert Path(report_html._crypto_output_dir()) == ROOT / 'output' / 'crypto'


def test_subtab_is_wired():
    src = _tpl()
    assert "{k:'ccy',l:'Currencies'}].concat(CRYPTO_META?[{k:'crypto',l:'Crypto'}]:[])" in src
    assert '<div id="m-crypto-view" style="display:none"></div>' in src
    show = _fn_src('_mShow', src)
    assert "k==='crypto'" in show and 'renderCryptoView()' in show
    assert "var CRYPTO_META={{ crypto_meta|default('null') }};" in src
    assert "_loadSidecar('crypto.json'," in _fn_src('renderCryptoView', src)
    assert "_kxOn()?'crypto'" in _fn_src('_altFiltKind', src)
    assert "G.push({h:'Crypto',k:'crypto'" in src
    # The Currencies table's look reaches the crypto table.
    assert '#cx .cx-tbl,#kx .cx-tbl{' in src


def test_nightly_runs_it_and_publish_ships_the_sidecar():
    sh = (ROOT / 'scripts' / 'run_daily.sh').read_text(encoding='utf-8')
    assert 'scripts/crypto_market.py --run-date "$RUNDATE"' in sh
    assert 'SIDE_PIDS+=("crypto:$!")' in sh
    assert 'cp output/crypto.json "$docs/crypto.json"' in sh
    assert 'rm -f "$docs/crypto.json"' in sh


def _block(src):
    start = src.index('var _KX=null,')
    return src[start:src.index('\n/* A sector page', start)]


def test_display_only_never_writes_data():
    assert not re.search(r'\b(DATA|FD)\b', _block(_tpl()))


def _render(after):
    if shutil.which('node') is None:
        pytest.skip('node not installed')
    src = _tpl()
    pick = [re.search(r'^function _attr\(s\)\{.*?\}$', src, re.M).group(0)]
    fm = src[src.index('var _FM_SUB='):src.index('function _fmApply(')]
    fd = src[src.index('var _FD_RC='):src.index('function _fdJump(')]
    stub = ("var Q='',_altN={},HOST={};function _filtCountSync(){}"
            "var document={getElementById:function(id){return id==='f-search'?{value:Q}:(HOST[id]=HOST[id]||{innerHTML:''});},"
            "documentElement:{classList:{contains:function(){return false;}}}};"
            "var localStorage={getItem:function(){return null;},setItem:function(){}};")
    js = '\n'.join([stub, _fn_src('_esc', src), _fn_src('_qHelp', src), _fn_src('_cxSpark', src),
                    *pick, fm, fd, _block(src), f'_KX=_kxPrep({json.dumps(REPORT)});', after])
    return subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout


def test_dashboard():
    html = _render('process.stdout.write(_kxDash(_KX));')
    assert '<span>Market cap</span><b>$1.44T</b>' in html
    assert '<span>Bitcoin share</span><b>80%</b>' in html
    assert '<span>Up over 7d</span><b>1 of 3</b>' in html
    assert 'Best 7 days' in html and '_kxPick(&quot;ETH&quot;)' in html
    assert 'USDT' in html and 'stablecoin (Stablecoins)' in html
    for word in ('BUY', 'LONG', 'Score'):
        assert word not in html


def test_table_sort_search_and_detail():
    out = _render(
        "var r={};_kxDraw();r.table=HOST['kx-body'].innerHTML;"
        "r.order=_kxRows().map(function(x){return x.symbol;});"
        "Q='doge';r.q=_kxRows().map(function(x){return x.symbol;});Q='';"
        "_kxSortBy('corr_spy_1y');r.nulls=_kxRows().map(function(x){return x.symbol;});"
        "_kxSortBy('corr_spy_1y');r.nulls2=_kxRows().map(function(x){return x.symbol;});"
        "_kxSortBy('name');r.name=_kxRows().map(function(x){return x.symbol;});"
        "r.n=_altN.crypto;"
        "_kxSel='DOGE';_kxDetail();r.det=HOST['kx-det'].innerHTML;"
        "process.stdout.write(JSON.stringify(r));")
    r = json.loads(out)
    assert r['order'] == ['BTC', 'ETH', 'DOGE']          # market cap, high to low
    assert r['q'] == ['DOGE']
    assert r['nulls'][-1] == 'ETH' and r['nulls2'][-1] == 'ETH'   # missing sorts last
    assert r['name'] == ['BTC', 'DOGE', 'ETH']           # text sorts A to Z first
    assert r['n'] == [3, 3]
    t = r['table']
    assert '$60,863' in t and '$1,746' in t and '$0.0571' in t
    assert '$1.22T' in t and '$210.0B' in t
    assert '$690K' in t
    assert '>80%<' in t and '>5.0%<' in t          # shares under 10% keep a decimal
    assert '<span class="bv-neg">-6.3%</span>' in t and '<span class="bv-pos">+4.1%</span>' in t
    assert 'data-col-tip-body="Last price &lt;usd&gt;."' in t
    d = r['det']
    assert 'DOGE · Dogecoin' in d and '<b>no cap</b>' in d and '#13 by market cap' in d
    assert 'cx-spark' in d
