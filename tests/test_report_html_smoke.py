# tests/test_report_html_smoke.py
"""Smoke tests for the HTML report builder.

Renders synthetic result rows through build_html end-to-end (Jinja2 template
included) and asserts the render succeeds and carries the expected markers.
Deliberately not a byte-golden: the template changes constantly; these tests
only pin "renders without exception, non-trivial output, rows present".
"""
import math

from scripts.report_html import build_html


def _rich_row():
    """A row with the optional model/quality/news blocks populated, so the
    row-context code exercises its non-default branches."""
    return {
        'ticker': 'RICH',
        'company_name': 'Rich Industries',
        'description': 'Rich Industries makes very profitable widgets.',
        'ceo': 'Ada Founder',
        'ceo_bio': 'Founded the company in a garage.',
        'founder_led': True,
        'sector': 'Technology',
        'industry': 'Software',
        'country': 'United States',
        'price': 100.0,
        'mcap': 50e9,
        'spread': 0.08,
        'mos': 0.25,
        'roic': 0.18,
        'wacc': 0.10,
        'dcf_fv': 133.0,
        '_fv_effective': 130.0,
        '_fv_source': 'blend',
        'dcf_sens_range': (110.0, 150.0),
        'rating': 'BUY',
        'rating_raw': 'BUY',
        '_rating_cap_reasons': [],
        '_composite_score': 61.0,
        '_score_valuation': 72.0,
        '_score_quality': 65.0,
        '_score_moat': 58.0,
        '_score_growth': 55.0,
        '_score_ownership': 50.0,
        '_gates_passed': 11,
        'piotroski': 8,
        'pe': 22.0,
        'ev_ebitda': 14.0,
        'fcf': 2.5e9,
        'fcf_margin': 0.25,
        'gross_margin': 0.62,
        'revenue': 10e9,
        'operating_income': 3e9,
        'operating_margin': 0.30,
        'pp_revenue_share': 0.4,
        'pp_profit_share': 0.5,
        'roe': 0.25,
        'roa': 0.12,
        'shares_out': 500e6,
        'insider_pct': 0.06,
        'div_yield': 0.01,
        'insider_transactions': [
            {'date': '2026-07-15', 'name': 'Ada Founder', 'type': 'P',
             'shares': 10000},
        ],
        'financial_summary': ['Revenue grew 14%.', 'Margins expanded.'],
        'news_headlines': [
            {'title': 'Rich wins big contract', 'url': 'https://example.com/a'},
        ],
        'news_sentiment': 0.6,
        'legal_filings': [{'date': '2026-06-01', 'type': '8-K'}],
        'ddm_eligible': True,
        'ddm_fv': 90.0,
        'ddm_growth': 0.05,
        'epv_fv': 88.0,
        'epv_mos': 0.05,
        'rim_fv': 115.0,
        'rim_mos': 0.13,
        'altman_z': 4.2,
        'altman_z_zone': 'safe',
        'beneish_m': -2.5,
        'beneish_flag': False,
        'trap_flag': False,
        '_trap_components': {'value': 3, 'quality': 2},
        'high_52w': 120.0,
        'low_52w': 70.0,
        'culture_narrative': 'A famously frugal engineering culture.',
        'employees': 12000,
        'glassdoor_rating': 4.4,
        'macro_regime': 'expansion',
        'analyst_rec': 'buy',
        'num_analysts': 25,
        'target_mean': 140.0,
        # NaN / stringified-infinity exercise the _sanitize normalization.
        'net_cash_to_mcap': math.nan,
        'deferred_rev_growth': 'Infinity',
        # edgar_history exercises the hist.json sidecar builder, including the
        # int-year and date-string key normalization.
        'edgar_history': {
            'revenue_history': {2021: 8e9, '2022': 9e9, '2023-12-31': 10e9},
            'earnings_history': {2022: 1.6e9, 2023: 2.0e9},
        },
    }


def _sparse_row():
    """A row with almost everything missing — the report must still render."""
    return {'ticker': 'SPRS', 'price': None, 'rating': None}


def test_build_html_renders_synthetic_rows(tmp_path):
    out = tmp_path / 'report.html'
    build_html([_rich_row(), _sparse_row()], str(out), prices_dir=None)
    assert out.exists()
    html = out.read_text(encoding='utf-8')
    # The template alone is ~9,800 lines; anything much smaller means the
    # render broke partway.
    assert len(html) > 100 * 1024
    assert 'RICH' in html
    assert 'SPRS' in html
    assert 'Rich Industries' in html


def test_build_html_writes_detail_and_hist_sidecars(tmp_path):
    out = tmp_path / 'report.html'
    build_html([_rich_row(), _sparse_row()], str(out), prices_dir=None)
    # Heavy popup-only fields are stripped into details.json; the rich row's
    # edgar_history feeds hist.json.
    assert (tmp_path / 'details.json').exists()
    assert (tmp_path / 'hist.json').exists()


def test_build_html_renders_empty_results(tmp_path):
    out = tmp_path / 'report_empty.html'
    build_html([], str(out), prices_dir=None)
    assert out.exists()
    assert len(out.read_text(encoding='utf-8')) > 100 * 1024
    # No rows means no sidecars — and no crash.
    assert not (tmp_path / 'details.json').exists()
    assert not (tmp_path / 'hist.json').exists()


def test_build_html_plumbs_macro_narrative_through_summary(tmp_path):
    """The Claude macro narrative rides MACRO_SUM inline. Its strings are
    model output, so a </script> inside one must never terminate the script
    block (dumps_for_script escapes HTML-significant characters)."""
    hostile = 'Growth is slowing.</script><script>alert(1)'
    macro_payload = {
        'summary': {
            'as_of': '2026-08-22', 'regime': None, 'tiles': [],
            'narrative': {
                'paragraphs': [hostile],
                'headwinds': ['Curve inverted'], 'tailwinds': [],
                'sectors': [{'sector': 'Technology', 'stance': 'neutral',
                             'outlook': 'Flat.'}],
                'model': 'claude-opus-5', 'generated_at': '2026-08-22T09:00:00+00:00',
            },
        },
        'sidecar': {'as_of': '2026-08-22', 'series': {'DGS10': {}}},
    }
    out = tmp_path / 'report.html'
    build_html([_rich_row()], str(out), prices_dir=None,
               macro_payload=macro_payload)
    html = out.read_text(encoding='utf-8')
    assert 'Curve inverted' in html
    assert hostile not in html            # raw </script> never lands verbatim
    assert 'Growth is slowing.' in html   # ...but the content does


def test_epv_tooltips_keyed_to_row_fields():
    """The detail panel looks tooltips up by the row key it renders; an
    orphaned key (epv_p_fv vs the row's epv_pfv) left the P/EPV column
    without a tooltip, and the growth-adjusted label reused the zero-growth
    text."""
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / 'templates' / 'report.html'
    html = src.read_text(encoding='utf-8')
    assert 'epv_p_fv' not in html
    assert re.search(r"^epv_pfv:'", html, re.M)
    assert re.search(r"^epv_growth_fv:'", html, re.M)
    assert "'EPV Growth-Adj':'epv_growth_fv'" in html
    assert 'gate passes below 1.2' not in html
