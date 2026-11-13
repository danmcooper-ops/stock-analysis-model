"""The report's score-band helpers (templates/report.html, "SCORE, NOT
RATING"), as one JavaScript prelude for the tests that run pieces of the
template in node. Put it FIRST in a script, so a test's own stubs (esc, RC,
...) defined after it still win."""
import re

BAND_FUNCS = ('_num', '_bandMins', 'bandKey', 'bandOf', '_bandFmt', 'bandLabel',
              'bandTitle', 'scoreCross', 'crossText', 'crossShort', 'bandsAsRange',
              'sbv', 'sb', 'bandPill', 'scoreSeries', 'isCapped', 'capTip', 'capTipAttr')


def _fn(src, name):
    one = re.search(r'^function ' + re.escape(name) + r'\(.*\}$', src, re.M)
    if one and one.group(0).count('{') == one.group(0).count('}'):
        return one.group(0)
    m = re.search(r'^function ' + re.escape(name) + r'\(.*?^\}', src, re.S | re.M)
    assert m, name
    return m.group(0)


def band_js(src):
    """The helpers, plus _BAND_KEYS and a default esc for capTipAttr."""
    keys = re.search(r'^var _BAND_KEYS=.*$', src, re.M).group(0)
    return '\n'.join(['function esc(s){return String(s==null?"":s);}', keys]
                     + [_fn(src, n) for n in BAND_FUNCS]) + '\n'
