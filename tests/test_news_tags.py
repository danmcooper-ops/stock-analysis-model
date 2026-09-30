"""Tag table tests.

The first two tests are the guard on ``layoff_news_signal`` — a typed column
in ``core.results`` that feeds ``models/narrative.py`` and
``models/data_tab_narrative.py``. They re-declare the pre-refactor keyword
sets literally, so a change to ``data/news_tags`` that alters either boolean
fails here rather than silently shifting a published column.
"""
import json
import pathlib

from data.news_tags import (
    CULTURE_POS_KEYWORDS,
    LAYOFF_KEYWORDS,
    TAG_KEYWORDS,
    TAG_LABELS,
    tag_headline,
)

# Copied from scripts/analyze_stock.py as it stood before the tagger landed.
# This is a fixed record of the old behaviour, not an import — an edit to the
# module must not be able to edit its own test.
LEGACY_LAYOFF = {
    'layoff', 'lay off', 'laid off', 'job cut', 'workforce reduction',
    'redundan', 'downsiz', 'restructur', 'reorg',
}
LEGACY_CULTURE_POS = {
    'best place', 'top employer', 'great place to work',
    'best company', 'culture award',
}

_FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'news_titles.json'


def _titles():
    return json.loads(_FIXTURE.read_text(encoding='utf-8'))


def test_legacy_keyword_sets_frozen():
    """The two published-column keyword sets have not drifted."""
    assert set(LAYOFF_KEYWORDS) == LEGACY_LAYOFF
    assert set(CULTURE_POS_KEYWORDS) == LEGACY_CULTURE_POS


def test_layoff_equivalence_on_corpus():
    """`layoffs` reproduces the old substring pass over real headlines."""
    for title in _titles():
        legacy = any(kw in title.lower() for kw in LEGACY_LAYOFF)
        assert ('layoffs' in tag_headline(title)) is legacy, title


def test_culture_equivalence_on_corpus():
    """`culture_award` reproduces the old substring pass over real headlines."""
    for title in _titles():
        legacy = any(kw in title.lower() for kw in LEGACY_CULTURE_POS)
        assert ('culture_award' in tag_headline(title)) is legacy, title


def test_prefix_keywords_still_match_word_families():
    """The deliberate prefixes must not become word-boundary matches."""
    assert 'layoffs' in tag_headline('Acme announces restructuring charge')
    assert 'layoffs' in tag_headline('Reorganization of the retail unit')
    assert 'layoffs' in tag_headline('Redundancies announced in EMEA')
    assert 'layoffs' in tag_headline('Downsizing its logistics footprint')


def test_culture_award_phrases():
    assert 'culture_award' in tag_headline('Named a Best Place to Work 2026')
    assert 'culture_award' in tag_headline('Top employer in the Midwest')
    assert tag_headline('Quarterly dividend declared') == ('earnings', 'dividend')


def test_empty_and_missing_titles():
    assert tag_headline('') == ()
    assert tag_headline(None) == ()


def test_tags_are_declaration_ordered_and_labelled():
    tags = tag_headline('Q3 earnings beat; board raises guidance and dividend')
    assert tags == ('earnings', 'guidance', 'dividend')
    assert set(TAG_LABELS) == set(TAG_KEYWORDS)
