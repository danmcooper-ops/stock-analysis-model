# tests/test_report_table_template.py
"""Every data table in the report is drawn as Stocks (.matrix-table) is.

The TABLE TEMPLATE block at the end of the report's last stylesheet gives
each table class the navy header, cell grid, zebra and hover. A new table
class must either be listed there or be one of the tables that already
draw that look themselves (or the print-only Summary PDF).
"""
import re
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'

# Tables that already are the Stocks look, or print-only (#pro-css).
EXEMPT = {'matrix-table', 'dt', 'heatmap-table', 'cx-tbl', 'pro-tbl', 'pro-rets'}


def _src():
    return TEMPLATE.read_text(encoding='utf-8')


def _template_block(src):
    start = src.index('/* TABLE TEMPLATE')
    return src[start:src.index('</style>', start)]


def _template_classes(block):
    m = re.search(r':is\(([^)]*)\)\{--tt-cell', block)
    assert m, 'TABLE TEMPLATE selector list not found'
    return {s.strip().lstrip('.') for s in m.group(1).split(',')}


def test_template_is_the_last_stylesheet_rule_set():
    src = _src()
    block_at = src.index('/* TABLE TEMPLATE')
    # Only the print-only Summary sheet and iframe exports may come later.
    later = [m.start() for m in re.finditer(r'<style>', src) if m.start() > block_at]
    assert later == [], 'a later stylesheet would outrank the table template'


def test_every_table_class_uses_the_template():
    src = _src()
    covered = _template_classes(_template_block(src))
    used = set()
    for m in re.finditer(r'<table class=\\?["\']([^"\'\\]+)', src):
        used.add(m.group(1).split()[0])
    missing = sorted(c for c in used if c not in covered and c not in EXEMPT)
    assert not missing, f'tables without the Stocks template: {missing}'


def test_header_is_navy_with_white_labels_in_both_themes():
    block = _template_block(_src())
    assert 'background:var(--hdr-bg) !important' in block
    assert 'color:var(--hdr-fg) !important' in block
    assert '[data-theme="dark"]' in block
