import json

from scripts.report_portfolio_html import build_portfolio_html
from scripts.safe_json import dumps_for_script


def test_dumps_for_script_escapes_script_end_tag():
    payload = {"name": "</script><script>alert(1)</script>"}

    encoded = dumps_for_script(payload)

    assert "</script>" not in encoded.lower()
    assert json.loads(encoded) == payload


def test_dumps_for_script_escapes_html_significant_chars():
    payload = {"value": "<tag data-x='1'>& text"}

    encoded = dumps_for_script(payload)

    assert "<" not in encoded
    assert ">" not in encoded
    assert "&" not in encoded
    assert json.loads(encoded) == payload


def test_portfolio_report_does_not_emit_raw_script_breakout(tmp_path):
    payload = "</script><script>window.XSS=1</script>"
    out = tmp_path / "portfolio.html"

    build_portfolio_html({
        "portfolio_name": payload,
        "benchmark": payload,
        "macro_regime": payload,
        "holdings": [{"ticker": "XSS", "company_name": payload}],
        "alerts": [{"ticker": "XSS", "message": payload, "severity": "HIGH"}],
        "concentration": {"sector_weights": {payload: 1.0}},
    }, str(out))

    html = out.read_text()
    assert payload not in html
    assert "\\u003c/script\\u003e" in html
    assert "&lt;/script&gt;" in html
