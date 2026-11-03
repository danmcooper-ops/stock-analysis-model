"""Question marks in templates/report.html: cards keep their own ?, and no
page carries a page-level ? (removed 2026-11-02)."""
from pathlib import Path

TPL = Path(__file__).resolve().parent.parent / 'templates' / 'report.html'


def _tpl():
    return TPL.read_text(encoding='utf-8')


def test_no_page_question_mark():
    src = _tpl()
    for gone in ('page-help-btn', 'help-wrap', 'id="help-pop"', '.help-pop', '_HELP_TXT', '_helpPlace',
                 'toggleHelp', 'closeHelp', 'pg-help'):
        assert gone not in src, gone


def test_every_card_question_mark_is_level_with_its_title():
    """_qAlign centres a card's corner ? on its title, or moves it into the
    title's row when the title sits above the card."""
    src = _tpl()
    align = src.split('function _qAlign(){', 1)[1].split('\n}\n', 1)[0]
    assert "'.qhelp-corner,.note-q'" in align
    assert '_qIntoRow(q,lab,' in align


def test_alignment_reruns_after_renders_on_a_timer():
    """Views render by innerHTML and swap sub-tabs by data-tab; a timer, not
    requestAnimationFrame, so an unpainted tab still gets the pass."""
    src = _tpl()
    sched = src.split('function _qSchedule(){', 1)[1].split('\n}\n', 1)[0]
    assert 'setTimeout(' in sched and 'requestAnimationFrame' not in sched
    assert "attributeFilter:['data-tab','class','hidden']" in src


def test_portfolio_page_does_not_repeat_its_name():
    """The portfolio's tab names it; its page carries no heading, and the
    Performance chart calls its own line "Portfolio"."""
    src = _tpl()
    det = src.split('function _pfDetailHtml(id){', 1)[1].split('\n}\n', 1)[0]
    assert 'p.name' not in det and '<h2>' not in det
    assert 'class="pfc-pf" style="stroke:\'+_pfEsc(_pfCol(p.color))+\'"/></svg>Portfolio</span>' in src
    assert "row('Portfolio','pf')" in src
