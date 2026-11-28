#!/usr/bin/env python3
"""Email the morning triage report (scheduled-tasks/daily-triage/SKILL.md).

The triage task writes ``output/triage_<RUNDATE>.md`` and then calls this, so
every triage run (OK nights included) ends with one email. The report's first
line (``TRIAGE <RUNDATE>: <VERDICT>``) is the subject; the whole file is the
body.

The triage runs as a Claude desktop-app task, not under ``run_daily.sh``, so
nothing has exported ``.env``; this script reads it itself, with the same
rules (KEY=VALUE lines, # comments, a value already in the environment wins).
Sending reuses scripts/email_summary.py: the same SMTP_USER,
SMTP_APP_PASSWORD, SMTP_HOST, SMTP_PORT and EMAIL_TO settings (recipient
defaults to SMTP_USER). EMAIL_TRIAGE=0 turns it off.

    python scripts/email_triage.py output/triage_2026-11-27.md
    python scripts/email_triage.py output/triage_2026-11-27.md --print
"""
import argparse
import os
import re
import smtplib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.email_summary import DEFAULT_HOST, DEFAULT_PORT, REPO, message, send  # noqa: E402

SUBJECT_MAX = 150


def load_env(path):
    """Export KEY=VALUE lines from ``path``; values already set win."""
    try:
        f = open(path, encoding='utf-8')
    except OSError:
        return
    with f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = (s.strip() for s in line.split('=', 1))
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) and k not in os.environ:
                os.environ[k] = v


def subject_and_body(text):
    lines = text.strip('\n').splitlines()
    first = next((ln.strip().lstrip('#').strip() for ln in lines if ln.strip()), '')
    subject = first[:SUBJECT_MAX] or 'TRIAGE: empty report'
    return subject, text if text.endswith('\n') else text + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('report', help='the triage report (output/triage_<RUNDATE>.md)')
    ap.add_argument('--to', help='recipient (default EMAIL_TO, else SMTP_USER)')
    ap.add_argument('--print', dest='show', action='store_true', help='print the email instead of sending it')
    a = ap.parse_args(argv)

    load_env(os.path.join(REPO, '.env'))
    if os.environ.get('EMAIL_TRIAGE', '1') == '0' and not a.show:
        print('email: EMAIL_TRIAGE=0 — not sent')
        return 0
    try:
        with open(a.report, encoding='utf-8') as f:
            text = f.read()
    except OSError as exc:
        print(f'email: cannot read {a.report}: {exc}', file=sys.stderr)
        return 1
    subject, body = subject_and_body(text)

    user = os.environ.get('SMTP_USER', '').strip()
    password = os.environ.get('SMTP_APP_PASSWORD', '').strip()
    to = a.to or os.environ.get('EMAIL_TO', '').strip() or user
    if a.show:
        print(f'To: {to or "(unset)"}\nSubject: {subject}\n\n{body}')
        return 0
    if not (user and password):
        print('email: SMTP_USER / SMTP_APP_PASSWORD not set (.env) — not sent', file=sys.stderr)
        return 1
    host = os.environ.get('SMTP_HOST', '').strip() or DEFAULT_HOST
    port = int(os.environ.get('SMTP_PORT', '').strip() or DEFAULT_PORT)
    try:
        send(message(subject, body, user, to), host, port, user, password)
    except smtplib.SMTPAuthenticationError as exc:
        print(f'email: {host} rejected the login for {user} ({exc.smtp_code})', file=sys.stderr)
        return 1
    except (smtplib.SMTPException, OSError) as exc:
        print(f'email: send to {to} via {host}:{port} failed: {exc}', file=sys.stderr)
        return 1
    print(f'email: sent "{subject}" to {to}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
