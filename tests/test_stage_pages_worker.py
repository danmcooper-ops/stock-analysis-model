"""Tests for scripts/stage_pages_worker.py (the login Worker for Cloudflare Pages)."""

import pytest

from scripts import stage_pages_worker as spw

TEAM = 'stockteam.cloudflareaccess.com'
AUD = 'a' * 64


def test_render_fills_both_placeholders_in_the_real_template():
    with open(spw.TEMPLATE, encoding='utf-8') as fh:
        template = fh.read()
    out = spw.render(template, TEAM, AUD)
    assert f"const ACCESS_TEAM_DOMAIN = '{TEAM}';" in out
    assert f"const ACCESS_AUD = '{AUD}';" in out
    assert "'__ACCESS_TEAM_DOMAIN__'" not in out and "'__ACCESS_AUD__'" not in out


def test_render_normalises_case_and_whitespace():
    out = spw.render("x='__ACCESS_TEAM_DOMAIN__'; y='__ACCESS_AUD__';", ' StockTeam.CloudflareAccess.com ', 'A' * 64)
    assert out == f"x='{TEAM}'; y='{AUD}';"


@pytest.mark.parametrize('team,aud', [
    (None, AUD),
    ('', AUD),
    ('stockteam', AUD),                              # bare team name
    ('https://stockteam.cloudflareaccess.com', AUD),  # URL, not host
    ('evil.com', AUD),
    ("x.cloudflareaccess.com'; alert(1); '", AUD),   # quote injection
    (TEAM, None),
    (TEAM, 'abc'),
    (TEAM, 'g' * 64),
])
def test_render_rejects_bad_config(team, aud):
    with pytest.raises(ValueError):
        spw.render("'__ACCESS_TEAM_DOMAIN__' '__ACCESS_AUD__'", team, aud)


def test_render_rejects_template_without_placeholders():
    with pytest.raises(ValueError, match='not found'):
        spw.render('export default {}', TEAM, AUD)


def test_main_writes_worker(tmp_path, monkeypatch):
    monkeypatch.setenv('CF_ACCESS_TEAM_DOMAIN', TEAM)
    monkeypatch.setenv('CF_ACCESS_AUD', AUD)
    assert spw.main([str(tmp_path)]) == 0
    assert TEAM in (tmp_path / '_worker.js').read_text(encoding='utf-8')


def test_main_refuses_without_config(tmp_path, monkeypatch):
    monkeypatch.delenv('CF_ACCESS_TEAM_DOMAIN', raising=False)
    monkeypatch.delenv('CF_ACCESS_AUD', raising=False)
    assert spw.main([str(tmp_path)]) == 1
    assert not (tmp_path / '_worker.js').exists()
