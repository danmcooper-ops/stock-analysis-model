"""Portfolio alerts in score terms: every alert keeps its ``message`` (the
email and the digest print it verbatim) and gains ``display``, the same event
as the score line it crossed, which the report and the app show."""
from models import portfolio_groups as pg

DAY = '2026-09-25'
BG = [{'ticker': f'Z{i:03d}', 'rating': 'HOLD', 'price': 10.0, 'company_name': 'Z',
       'sector': 'Tech', '_composite_score': 30.0} for i in range(60)]


def _r(t, rating, score, **kw):
    d = {'ticker': t, 'rating': rating, 'price': 10.0, 'company_name': f'{t} Corp',
         'sector': 'Technology', '_composite_score': score}
    d.update(kw)
    return d


def _classify(today, prior):
    return pg.classify_changes(pg.rows_by_ticker([today, *BG]),
                               pg.rows_by_ticker([prior, *BG]), DAY)[0]


def test_lines_come_from_the_scoring_thresholds():
    assert pg.rating_lines() == {'BUY': 57.0, 'LEAN BUY': 39.0, 'HOLD': 25.0, 'PASS': 0.0}


def test_move_phrase_names_the_line_crossed():
    assert pg.score_move_phrase('HOLD', 'LEAN BUY', 36.2, 41.0) == 'crossed above 39 (36.2 → 41.0)'
    assert pg.score_move_phrase('BUY', 'LEAN BUY', 58.0, 55.1) == 'fell below 57 (58.0 → 55.1)'
    assert pg.score_move_phrase('LEAN BUY', 'PASS') == 'fell below 39'
    assert pg.score_move_phrase('PASS', 'BUY') == 'crossed above 57'


def test_rating_sets_as_score_ranges():
    assert pg.ratings_as_score_range(['BUY']) == 'score ≥ 57'
    assert pg.ratings_as_score_range(['BUY', 'LEAN BUY']) == 'score ≥ 39'
    assert pg.ratings_as_score_range(['HOLD']) == 'score 25–39'
    assert pg.ratings_as_score_range(['HOLD', 'PASS']) == 'score < 39'
    assert pg.ratings_as_score_range(['BUY', 'PASS']) is None
    assert pg.ratings_as_score_range([]) is None


def test_entering_the_buy_zone_keeps_its_message_and_adds_a_display():
    e = _classify(_r('AAA', 'LEAN BUY', 41.0), _r('AAA', 'HOLD', 36.2))
    a = next(x for x in e if x['ticker'] == 'AAA')
    assert a['kind'] == 'entered_buy' and a['level'] == 'action'
    assert a['message'] == 'AAA moved into the buy zone: HOLD → LEAN BUY'
    assert a['display'] == 'AAA crossed above 39 (36.2 → 41.0)'
    assert a['score_from'] == 36.2 and a['score_to'] == 41.0


def test_a_move_within_a_side_reads_as_its_line():
    e = _classify(_r('AAA', 'LEAN BUY', 55.1), _r('AAA', 'BUY', 58.0))
    a = next(x for x in e if x['ticker'] == 'AAA')
    assert a['kind'] == 'same_side'
    assert a['message'] == 'AAA BUY → LEAN BUY'
    assert a['display'] == 'AAA fell below 57 (58.0 → 55.1)'


def test_no_display_names_a_rating():
    e = _classify(_r('AAA', 'PASS', 20.0), _r('AAA', 'BUY', 60.0))
    for a in e:
        if 'display' in a:
            for word in ('BUY', 'HOLD', 'PASS'):
                assert word not in a['display']


def test_rule_change_in_score_terms():
    rule = {'ratings': ['BUY'], 'sectors': None, 'countries': None, 'cf': []}
    prev, now = _r('AAA', 'BUY', 58.2), _r('AAA', 'LEAN BUY', 55.1)
    assert pg.explain_rule_change(rule, prev, now) == ['rating BUY → LEAN BUY']
    assert pg.explain_rule_change(rule, prev, now, score_terms=True) == \
        ['score 58.2 → 55.1 (rule: score ≥ 57)']


def test_systemic_display_names_no_rating():
    st = {'flood': True, 'cause': 'model', 'changed_share': 0.4, 'changed': 1500}
    assert 'changed rating' in pg.systemic_message(st)
    d = pg.systemic_display(st)
    assert 'crossed a score line' in d and 'rating' not in d
    assert pg.systemic_display({'flood': False}) is None
