# tests/test_tooltips.py
"""Guard: every gate column header must have a narrative tooltip.

The report renders each gate header with a `data-gv-tip` key equal to the
gate key minus its `_gate_` prefix, and looks the text up in the template's
`var TT={...}` map (empty string when absent). A gate added to scoring.py
without a matching TT entry ships a blank tooltip — this has regressed before
(see "Add tooltip handlers for nine missing TT keys"), so pin it.
"""
import os
import re


from scripts.scoring import gate_metadata

_TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')


def _tt_keys():
    html = open(_TEMPLATE, encoding='utf-8').read()
    body = re.search(r'var TT=\{(.*?)\n\};', html, re.S).group(1)
    return set(re.findall(r'^([A-Za-z_][A-Za-z0-9_]*):', body, re.M))


def test_every_gate_header_has_a_tooltip():
    tt = _tt_keys()
    missing = [g['key'].replace('_gate_', '')
               for g in gate_metadata()['gates']
               if g['key'].replace('_gate_', '') not in tt]
    assert not missing, f'gate headers with no TT entry: {missing}'


def test_new_2026_07_gates_have_tooltips():
    tt = _tt_keys()
    for key in ('ebit_ev', 'incr_roic', 'margin_vs_hist', 'insider_buying',
                'pool_share'):
        assert key in tt, f'missing tooltip for {key}'
        # non-trivial narrative text, not a stub
        body = open(_TEMPLATE, encoding='utf-8').read()
        entry = re.search(rf"^{key}:'(.*?)',$", body, re.M)
        assert entry and len(entry.group(1)) > 80, f'{key} tooltip too short'
