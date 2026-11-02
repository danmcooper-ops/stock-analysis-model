"""The Filters panel's Saved filters section: named snapshots of the
panel kept in this browser, applied through _applyFilterState."""
from pathlib import Path

TPL = (Path(__file__).resolve().parent.parent / 'templates' / 'report.html').read_text(encoding='utf-8')


def test_section_leads_the_panel():
    saved = TPL.index('data-fsec="saved"')
    assert saved < TPL.index('data-fsec="rating"')
    assert 'id="fsum-saved"' in TPL and 'id="sf-list"' in TPL
    assert 'onsubmit="sfSave();return false;"' in TPL


def test_snapshots_are_local_and_normalised():
    assert "var _SF_KEY='stock_saved_filters_v1';" in TPL
    body = TPL[TPL.index('function _sfSnap('):TPL.index('function _sfKey(')]
    # The top bar's search and a dashboard drill are not part of a saved filter.
    assert 'search' not in body and 'dashFilter' not in body
    assert '_sfKey(x.st)===cur' in TPL


def test_apply_replaces_every_filter_and_summary_names_it():
    body = TPL[TPL.index('function sfApply('):TPL.index('function sfDelete(')]
    assert '_applyFilterState(x.st)' in body
    assert "saved:(_sfCurrent()||{}).name||''" in TPL
    assert '_sfRender();' in TPL[TPL.index('function _fsecSync('):TPL.index('function _fsecSet(')]
