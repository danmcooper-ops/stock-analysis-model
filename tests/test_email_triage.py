# tests/test_email_triage.py
"""scripts/email_triage.py: the morning triage email. Nothing opens a socket."""

import os

from scripts import email_triage as et


def test_subject_is_first_line():
    s, body = et.subject_and_body('TRIAGE 2026-11-27: OK\nHealth: all clear')
    assert s == 'TRIAGE 2026-11-27: OK'
    assert body.endswith('\n')


def test_subject_skips_blank_and_heading_marks():
    s, _ = et.subject_and_body('\n\n# TRIAGE 2026-11-27: FAILED\nCause: x\n')
    assert s == 'TRIAGE 2026-11-27: FAILED'


def test_empty_report_gets_a_subject():
    s, _ = et.subject_and_body('')
    assert s == 'TRIAGE: empty report'


def test_load_env_keeps_existing_values(tmp_path, monkeypatch):
    env = tmp_path / '.env'
    env.write_text('# c\nTRIAGE_T_A=from_file\nTRIAGE_T_B = b \nbad line\n')
    monkeypatch.setenv('TRIAGE_T_A', 'already')
    monkeypatch.delenv('TRIAGE_T_B', raising=False)
    et.load_env(str(env))
    assert os.environ['TRIAGE_T_A'] == 'already'
    assert os.environ['TRIAGE_T_B'] == 'b'
    monkeypatch.delenv('TRIAGE_T_B')


def test_print_mode_sends_nothing(tmp_path, capsys, monkeypatch):
    rpt = tmp_path / 'triage.md'
    rpt.write_text('TRIAGE 2026-11-27: OK\nHealth: all clear\n')
    monkeypatch.setattr(et, 'send', lambda *a, **k: (_ for _ in ()).throw(AssertionError('sent')))
    assert et.main([str(rpt), '--print', '--to', 'x@example.com']) == 0
    out = capsys.readouterr().out
    assert 'Subject: TRIAGE 2026-11-27: OK' in out


def test_send_called_with_settings(tmp_path, monkeypatch):
    rpt = tmp_path / 'triage.md'
    rpt.write_text('TRIAGE 2026-11-27: DIED\n')
    monkeypatch.setattr(et, 'load_env', lambda p: None)
    monkeypatch.setenv('SMTP_USER', 'me@example.com')
    monkeypatch.setenv('SMTP_APP_PASSWORD', 'pw')
    monkeypatch.delenv('EMAIL_TO', raising=False)
    monkeypatch.delenv('EMAIL_TRIAGE', raising=False)
    seen = {}
    monkeypatch.setattr(et, 'send', lambda msg, host, port, user, pw: seen.update(
        subject=msg['Subject'], to=msg['To'], host=host, port=port))
    assert et.main([str(rpt)]) == 0
    assert seen == {'subject': 'TRIAGE 2026-11-27: DIED', 'to': 'me@example.com',
                    'host': et.DEFAULT_HOST, 'port': et.DEFAULT_PORT}


def test_missing_credentials_is_an_error(tmp_path, monkeypatch):
    rpt = tmp_path / 'triage.md'
    rpt.write_text('TRIAGE 2026-11-27: OK\n')
    monkeypatch.setattr(et, 'load_env', lambda p: None)
    monkeypatch.delenv('SMTP_USER', raising=False)
    monkeypatch.delenv('SMTP_APP_PASSWORD', raising=False)
    monkeypatch.delenv('EMAIL_TRIAGE', raising=False)
    assert et.main([str(rpt)]) == 1
