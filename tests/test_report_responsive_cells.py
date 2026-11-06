# tests/test_report_responsive_cells.py
"""Guard: ticker-page blocks that clipped their own text between phone and
wide desktop.

A headless sweep of every view at iPhone, iPad and desktop sizes found three
blocks whose nowrap text ran past an overflow:hidden box and was cut
mid-figure: the Snapshot strip (iPad mini / Air: "$243.42 – $348"), The
Model's five gate columns (every width under ~1900px: "Spread79.2%", the
threshold notes) and the Peers table on phones (ROIC and MoS unreachable).
Each fix is a line of CSS that is easy to drop in a cleanup, so pin them.
"""
import os
import re

_TEMPLATE = os.path.join(os.path.dirname(__file__), '..', 'templates', 'report.html')


def _css():
    return open(_TEMPLATE, encoding='utf-8').read()


def test_snapshot_cells_wrap_on_tablets():
    m = re.search(r'@media \(min-width:601px\) and \(max-width:1199px\)\{(.*?)\n\}', _css(), re.S)
    assert m, 'tablet rule for the Snapshot strip is gone'
    body = m.group(1)
    assert '.km-cell{min-width:0' in body
    assert 'white-space:normal' in body


def test_gate_columns_stack_instead_of_clipping():
    """The Model card's columns are ~150-225px on a wide page: a category row
    stacks its bar under its name there, and the short targets wrap rather
    than being cut off (test_report_model_card has the rest)."""
    css = _css()
    assert re.search(r'\.gv-col\{[^}]*container-type:inline-size', css)
    assert re.search(r'@container \(max-width:\d+px\)\{\s*\.gv-cat\{grid-template-columns', css)
    assert not re.search(r'\.gv-[a-z]+\{[^}]*text-overflow:ellipsis', css)


def test_peer_tables_pan_rather_than_clip():
    assert re.search(r'\.ps-col\{overflow-x:auto;', _css())
