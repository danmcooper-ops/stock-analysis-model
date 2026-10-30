"""Charts › Stock Comparison: history rebuilt from filings and price.

Annual ratios come straight from the hist/ shards; daily multiples combine
each day's actual close (px/ closes are dividend-adjusted, undone with the
shard's ``dv`` events) with the latest fiscal year already filed (``fya``:
the year's real end from ``fy_ends`` plus the filing window).
"""
import json
import os
import re
import shutil
import subprocess
from datetime import date

import pandas as pd
import pytest

from data.splits import dividend_events_from_prices, split_events_from_prices
from scripts.report_html import _hist_avail_and_dividends

TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')
node = pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')


def _prices(tmp_path, tk='T'):
    idx = pd.to_datetime(['2024-01-02', '2024-03-01', '2024-06-03', '2025-02-03'])
    pd.DataFrame({'Close': [10.0, 11, 12, 13], 'Dividends': [0, 0.25, 0, 0.3],
                  'Stock Splits': [0, 0, 2.0, 0]}, index=idx).to_parquet(tmp_path / f'{tk}.parquet')
    return str(tmp_path)


def test_dividend_and_split_events_share_one_read(tmp_path):
    d = _prices(tmp_path)
    assert dividend_events_from_prices(d, 'T') == [(date(2024, 3, 1), 0.25), (date(2025, 2, 3), 0.3)]
    assert dividend_events_from_prices(d, 'T', since=date(2025, 1, 1)) == [(date(2025, 2, 3), 0.3)]
    assert split_events_from_prices(d, 'T') == [(date(2024, 6, 3), 2.0)]
    assert dividend_events_from_prices(d, 'NOPE') == []


def test_availability_uses_real_year_ends_and_the_filing_window(tmp_path):
    d = _prices(tmp_path)
    th = {'rev': {'2024-12-31': 1.0, '2025-12-31': 2.0}}
    # A March year-end filer: FY2025 ended 2025-03-29, filed by late June.
    _hist_avail_and_dividends(th, {'fy_ends': {'2025': '2025-03-29'}}, 'T', d)
    assert th['fya'] == {'2024': '2025-03-31', '2025': '2025-06-27'}  # 2024: Dec 31 fallback
    assert th['dv'][-1] == ['2025-02-03', 0.3]
    # A filer reporting in another currency gets the 20-F's 120 days.
    th2 = {'rev': {'2025-12-31': 1.0}}
    _hist_avail_and_dividends(th2, {'fy_ends': {'2025': '2025-12-31'}, 'reporting_currency': 'JPY'}, 'T', None)
    assert th2['fya'] == {'2025': '2026-04-30'} and 'dv' not in th2


def test_sec_history_records_fiscal_year_ends(monkeypatch):
    from data.sec_xbrl_client import SECXBRLClient
    c = SECXBRLClient(cik_map={}, name_map={}, email='t@e.com', request_delay=0)
    rev = [{'form': '10-K', 'fy': y, 'fp': 'FY', 'val': 1e9, 'filed': f'{y}-05-20',
            'start': f'{y - 1}-03-31', 'end': f'{y}-03-29'} for y in (2024, 2025)]
    monkeypatch.setattr(c, 'fetch_company_facts',
                        lambda tk: {'facts': {'us-gaap': {'Revenues': {'units': {'USD': rev}}}}})
    h = c.fetch_historical_financials('TEST')
    assert h['fy_ends'] == {'2024': '2024-03-29', '2025': '2025-03-29'}


def _fns(*names):
    src = open(TEMPLATE, encoding='utf-8').read()
    out = []
    for n in names:
        out.append(re.search(r'^function ' + re.escape(n) + r'\(.*?^\}', src, re.S | re.M).group(0))
    m = re.search(r'^var _PH_METRICS=\[.*?^\];', src, re.S | re.M).group(0)
    return m + '\n' + '\n'.join(out)


@node
def test_daily_multiples_use_real_closes_and_filed_years(tmp_path):
    # Real closes 100 every day; a $1 dividend goes ex on 2025-06-02, so
    # Yahoo's adjusted closes before it read 99. EPS: FY2024 = 5 (public
    # 2025-03-31), FY2025 = 10 (public 2026-03-31).
    stubs = r"""
var PRICES={dates:['2025-03-28','2025-03-31','2025-05-30','2025-06-02','2026-03-30','2026-03-31'],
            prices:{T:[99,99,99,100,100,100]}};
var HIST={T:{epsd:{'2024-12-31':5,'2025-12-31':10},shares:{'2024-12-31':1e6,'2025-12-31':1e6},
             equity:{'2024-12-31':2e7,'2025-12-31':5e7},
             fya:{'2024':'2025-03-31','2025':'2026-03-31'},dv:[['2025-06-02',1]]}};
var _phRange='MAX';
function _phRangeCutoff(){return null;}
"""
    js = stubs + _fns('_phMetricDef', '_phRawCloses', '_phGetHpxPts').replace('var _PH_METRICS', 'var _PH_RAW={};var _PH_METRICS') + r"""
console.log(JSON.stringify({raw:_phRawCloses('T'),pe:_phGetHpxPts('T','pe_h'),pb:_phGetHpxPts('T','pb_h'),dy:_phGetHpxPts('T','dy')}));
"""
    f = tmp_path / 't.js'
    f.write_text(js, encoding='utf-8')
    r = subprocess.run(['node', str(f)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    assert [round(x, 6) for x in o['raw']] == [100] * 6
    pe = {p['d']: p['c'] for p in o['pe']}
    assert '2025-03-28' not in pe                      # no year filed yet
    assert pe['2025-03-31'] == 20 and pe['2026-03-30'] == 20   # FY2024's EPS
    assert pe['2026-03-31'] == 10                      # FY2025 from its filing
    assert {p['d']: p['c'] for p in o['pb']}['2026-03-31'] == 2
    dy = {p['d']: p['c'] for p in o['dy']}
    assert dy['2025-05-30'] == 0 and dy['2025-06-02'] == 0.01
