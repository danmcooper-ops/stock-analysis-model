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


def _html():
    return open(_TEMPLATE, encoding='utf-8').read()


def _column_keys():
    """Every metric key the report hangs a tooltip on.

    Three surfaces, all keyed the same way: the Financial Data tables (TC),
    the detail popup's data tabs (DT_TABS, which look the key up with
    ``tt(c.k)``) and the portfolio members presets. Anything here without a
    TT entry renders a row or header with no explanation at all — the People
    tab shipped with fifteen of them.
    """
    html = _html()
    out = {}

    def _cols(blob, where):
        for k in re.findall(r"\{k:'([^']+)'", blob):
            out.setdefault(k, where)

    tc = re.search(r'\nvar TC=\{(.*?)\n\};', html, re.S).group(1)
    for name, blob in re.findall(r"^\s*([a-z_0-9]+):\[(.*?)\],?\s*$", tc, re.M | re.S):
        _cols(blob, 'TC.' + name)
    for name, blob in re.findall(r"^TC\.([a-z_0-9]+)=\[(.*?)\];\s*$", html, re.M | re.S):
        _cols(blob, 'TC.' + name)
    dt = re.search(r'var DT_TABS=\[(.*?)\n\];', html, re.S).group(1)
    for tab in re.split(r"\n \{k:'", dt)[1:]:
        _cols(tab, 'DT_TABS.' + tab.split("'")[0])
    pf = re.search(r'var _PFM_PRESETS=\[(.*?)\n\];', html, re.S).group(1)
    _cols(pf, '_PFM_PRESETS')
    return out


def test_every_table_and_popup_column_has_a_tooltip():
    tt = _tt_keys()
    missing = sorted((k, w) for k, w in _column_keys().items()
                     if k not in tt and not k.startswith('_r'))
    assert not missing, f'column keys with no TT entry: {missing}'


def test_dkv_labels_resolve_to_real_tooltips():
    """bkv()/km cells map a row LABEL to a TT key — a typo'd or removed key
    silently drops the tooltip rather than failing."""
    html = _html()
    body = re.search(r'var DKV_TIPS=\{(.*?)\n\};', html, re.S).group(1)
    tt = _tt_keys()
    dead = sorted({k for _, k in re.findall(r"'([^']+)'\s*:\s*'([^']+)'", body)
                   if k not in tt})
    assert not dead, f'DKV_TIPS points at TT keys that do not exist: {dead}'


def test_gate_tooltips_agree_with_their_own_pass_rule():
    """The popup prints the gate's threshold directly under its tooltip, so a
    tooltip that states a different number contradicts itself on screen
    (net_debt_ebitda said 2x against a 1.5x gate; sbc_dilution said 3%
    against 2%)."""
    html = _html()
    mismatched = []
    for g in gate_metadata()['gates']:
        key = g['key'].replace('_gate_', '')
        entry = re.search(rf"^{key}:'(.*?)',$", html, re.M)
        if not entry:
            continue
        stated = re.findall(r'[Pp]asses[^.;]*?(\d+(?:\.\d+)?)\s*(?:%|×|x)', entry.group(1))
        wanted = set(re.findall(r'\d+(?:\.\d+)?', g.get('threshold') or ''))
        if stated and not (set(stated) & wanted):
            mismatched.append((key, stated, g.get('threshold')))
    assert not mismatched, f'tooltip contradicts the gate threshold: {mismatched}'


def test_tooltips_stay_succinct():
    """A tooltip is a hover, not an essay: past ~60 words it outgrows the
    360px box and stops being read."""
    tt_html = _html()
    body = re.search(r'var TT=\{(.*?)\n\};', tt_html, re.S).group(1)
    long = [(k, len(v.split()))
            for k, v in re.findall(r"^\s*'?([A-Za-z_][A-Za-z0-9_]*)'?:'(.*?)',?$", body, re.M)
            if len(v.split()) > 60]
    assert not long, f'tooltips over 60 words: {long}'
