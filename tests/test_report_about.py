"""The About card: the model in one paragraph, the same words as the iOS
app's welcome card and the App Store description."""
from scripts.report_html import build_html

ABOUT = ("CitytoSummit’s Stock Analysis Model values about 3,900 US stocks "
         "each night with several fair-value models and scores each one 0–100 "
         "on quality, growth and valuation.")


def test_about_card_ships_and_opens_from_the_sidebar(tmp_path):
    out = tmp_path / 'report.html'
    build_html([{'ticker': 'AAA', 'price': 1.0, 'rating': 'HOLD'}], str(out),
               prices_dir=None)
    html = out.read_text(encoding='utf-8')
    assert 'id="about-overlay"' in html
    assert ABOUT in html
    assert 'not investment advice.' in html
    assert 'onclick="sbClose();aboutOpen()"' in html
    # once by itself on a first visit, and Escape closes it
    assert "localStorage.getItem('stock_about_seen_v1')" in html
    assert "if(_aboutOpen()){" in html
