"""The ticker page's Charts tab › Future (_dtcFutureModel, _dtcFutureSVG,
_dtcFutureNotes): the iOS app's FutureOutlook in the report. A no-drift
volatility cone, the model's bear–bull band and fair value, analyst
targets, the next earnings date and the reverse DCF's implied growth.
Display only."""
import json
import math
import os
import re
import shutil
import subprocess

import pytest

TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')
node = pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')


def _src():
    return open(TEMPLATE, encoding='utf-8').read()


def _fn(src, name):
    return re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M).group(0)


def _closes(n, swing=0.01, start='2026-01-02'):
    """Closes alternating ±swing a day (log), one a calendar day, from 100."""
    import datetime as dt
    d0 = dt.date.fromisoformat(start)
    out, p = [], 100.0
    for i in range(n):
        out.append({'d': (d0 + dt.timedelta(days=i)).isoformat(), 'c': p})
        p *= math.exp(swing) if i % 2 == 0 else math.exp(-swing)
    return out


ROW = {'price': 100, 'mc_p10_fv': 80, 'mc_p90_fv': 180, 'dcf_fv': 130,
       'target_low': 70, 'target_mean': 110, 'target_high': 160,
       'implied_growth': -0.02, 'fcf_growth': 0.03, 'earnings_date_est': True,
       'profile': {'buy_below': 95}}


def _run(tmp_path, cases):
    """cases: [(closes, row)] -> the model's plain fields, plus the SVG and
    notes for each."""
    src = _src()
    js = '\n'.join([
        re.search(r'^function _num\(.*$', src, re.M).group(0),
        'function esc(s){return String(s);}',
        'function fd(v){return "$"+v.toFixed(2);}',
        _fn(src, '_dtcFutureModel'), _fn(src, '_dtcFutureSVG'), _fn(src, '_dtcFutureNotes'),
        'var cases=' + json.dumps(cases) + ';',
        'console.log(JSON.stringify(cases.map(function(c){',
        '  var f=_dtcFutureModel(c[0],c[1]);if(!f)return null;',
        '  return {price:f.price,vol:f.vol,cone:f.cone.map(function(k){return [k.l1,k.h1,k.l2,k.h2];}),',
        '    fv:f.fv,bear:f.bear,bull:f.bull,buyBelow:f.buyBelow,targets:f.targets,earnings:!!f.earnings,',
        '    est:f.earningsEst,implied:f.implied,model:f.modelGrowth,rr:f.rewardToRisk,base:f.change(f.fv),',
        '    svg:_dtcFutureSVG(f,c[0].slice(-90),640),notes:_dtcFutureNotes(f)};',
        '})));',
    ])
    p = tmp_path / 'f.js'
    p.write_text(js)
    return json.loads(subprocess.run(['node', str(p)], capture_output=True, text=True, check=True).stdout)


@node
def test_cone_is_past_year_volatility_with_no_drift(tmp_path):
    f = _run(tmp_path, [[_closes(400), ROW]])[0]
    assert abs(f['vol'] - 0.01 * math.sqrt(252)) < 0.002
    assert len(f['cone']) == 13
    l1, h1, l2, h2 = f['cone'][0]
    assert l1 == h1 == f['price']
    l1, h1, l2, h2 = f['cone'][-1]
    assert abs(l1 * h1 - f['price'] ** 2) < 1e-6      # symmetric in logs
    assert l2 < l1 and h2 > h1


@node
def test_reads_the_rows_forward_fields(tmp_path):
    f = _run(tmp_path, [[_closes(400), dict(ROW, earnings_next_date='2027-03-15')]])[0]
    assert (f['fv'], f['bear'], f['bull'], f['buyBelow']) == (130, 80, 180, 95)
    assert f['targets'] == {'low': 70, 'mean': 110, 'high': 160}
    assert f['earnings'] and f['est']
    assert (f['implied'], f['model']) == (-0.02, 0.03)
    assert abs(f['base'] - 0.30) < 1e-9
    assert abs(f['rr'] - 1.5) < 1e-9                 # (130-100)/(100-80)
    n = f['notes']
    assert 'implies <b>-2.0%</b> a year growth; the model expects <b>+3.0%</b>' in n
    assert '1.5× the downside to the bear value' in n
    assert 'Earnings ' in f['svg'] and '(est.)' in f['svg']
    for lbl in ('Fair value', 'Buy below', 'Today', 'Low 70', 'Avg 110', 'High 160'):
        assert lbl in f['svg'], lbl


@node
def test_fallbacks_and_edges(tmp_path):
    out = _run(tmp_path, [
        # dcf_sens_range stands in for the Monte Carlo band; _fv_effective leads.
        [_closes(400), {'price': 100, '_fv_effective': 120, 'dcf_fv': 999, 'dcf_sens_range': [90, 150]}],
        # Below the bear value: no reward-to-risk, a plain sentence instead.
        [_closes(400), dict(ROW, price=75)],
        # Above fair value: no modelled upside.
        [_closes(400), dict(ROW, price=140)],
        # Too little history for a cone; still a chart.
        [_closes(30), {}],
        [[], ROW],
    ])
    a, below, above, short, empty = out
    assert (a['fv'], a['bear'], a['bull']) == (120, 90, 150) and not a['earnings']
    assert below['rr'] is None and 'below even the model’s bear value' in below['notes']
    assert above['rr'] is None and 'above the model’s fair value' in above['notes']
    assert short['vol'] is None and short['cone'] == [] and 'Too little price history' in short['notes']
    assert empty is None


@node
def test_earnings_flag_reads_numbers_and_strings(tmp_path):
    out = _run(tmp_path, [[_closes(400), dict(ROW, earnings_date_est=v)] for v in (1, 'true', 0, False)])
    assert [f['est'] for f in out] == [True, True, False, False]


def test_future_is_a_price_range_after_max():
    src = _src()
    btns = _fn(src, '_dtcButtons')
    assert "concat(_dtcMetric==='price'?['FUTURE']:[])" in btns and "(r==='FUTURE'?'Future':r)" in btns
    render = _fn(src, '_dtcRender')
    assert "if(def.src==='price'&&_dtcRange==='FUTURE'){_dtcFutureRender(tk,d,el,tb,fut);return;}" in render
    # Fundamentals are annual: a metric switch leaves Future for 1Y.
    assert "['1M','3M','6M','FUTURE'].indexOf(_dtcRange)>=0" in _fn(src, 'dtcSetMetric')
    # Future draws the company alone.
    assert "_dtcRange!=='FUTURE'" in _fn(src, '_dtcBenchOn')
