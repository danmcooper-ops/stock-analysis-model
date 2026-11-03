#!/usr/bin/env python3
"""Email the nightly run's summary to the owner's inbox.

``run_daily.sh`` calls this from ``finish()``, right after it writes
``output/run_summary_<RUNDATE>.json``, so every run that reaches the end of
the script mails its outcome — ok, degraded, failed or a market-closed skip —
whether or not the Claude scheduled task that launched it finishes its own
summary. A run that dies before ``finish()`` (killed, crashed, refused by the
lock) sends nothing.

The digest is built from two files the run already writes: the summary JSON
(status, each step's rc and duration, notes) and the daily log, from which it
lifts the report sections the runbook
(``scheduled-tasks/daily-stock-analysis/SKILL.md``) asks about: gate N/A
coverage, the momentum check, the portfolio report's flags, portfolio alerts,
the store and database checks, coverage, run quality and the publish result.
The log is appended to by every ``--from`` resume of a date, so each section
is taken from its last occurrence.

Mail goes through SMTP with STARTTLS (or implicit TLS on port 465). The
defaults fit iCloud Mail, which needs an app-specific password, never the
Apple ID password. Settings come from the environment (``run_daily.sh``
exports ``.env``):

    SMTP_USER            sender login, also the From address
    SMTP_APP_PASSWORD    app-specific password
    SMTP_HOST            default smtp.mail.me.com
    SMTP_PORT            default 587 (465 = implicit TLS)
    EMAIL_TO             recipient, default SMTP_USER
    EMAIL_SUMMARY=0      turns the email off

Without SMTP_USER/SMTP_APP_PASSWORD it prints why and exits 0, so a dev box
behaves as before. Stdlib only.

    python scripts/email_summary.py --date 2026-11-02
    python scripts/email_summary.py --date 2026-11-02 --print   # show, do not send
"""
import argparse
import html
import json
import os
import re
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.expanduser('~/Library/Logs/StockModel')
DEFAULT_HOST = 'smtp.mail.me.com'
DEFAULT_PORT = 587
SEND_ATTEMPTS = 2
RETRY_WAIT_SEC = 30

# The runbook's thresholds (SKILL.md, Step 3).
SECTOR_LIMIT_PCT = 35.0
CORR_LIMIT = 0.85
DRAWDOWN_LIMIT_PCT = -50.0
MOMENTUM_R_LIMIT = 0.15
MOMENTUM_P_LIMIT = 0.05
# BUY/LEAN BUY's mean trailing return has to beat HOLD/PASS's by this much
# (percentage points) to count as "clearly higher".
MOMENTUM_GAP_PP = 5.0
# The portfolio report's drawdown table covers BUY and LEAN BUY and prints
# no rating, only the composite; the rating is the composite's bucket alone
# (scripts/scoring.py, rating_threshold_buy), so a BUY is a score of 57 up.
BUY_SCORE = 57.0

_START = re.compile(r'^--- \[(\w+)\] \d\d:\d\d:\d\d: ')
_END = re.compile(r'^--- \[(\w+)\] rc=')
_EARNINGS = re.compile(r'^\s*WATCH\s+(\S+) reports earnings on \d{4}-(\d\d-\d\d)\s*$')
_BUCKET = re.compile(r'^(BUY|LEAN BUY|HOLD|PASS)\s+(\d+)\s+(-?[\d.]+)%')
_SPEARMAN = re.compile(r'Spearman r = ([+-]?[\d.]+)\s+\(p=([\d.]+)\)')
_SECTOR_ROW = re.compile(r'^\s{2}(\S.*?)\s{2,}(\d+)\s+([\d.]+)%')
_PAIR = re.compile(r'\*\*\*\s+(\S+) <-> (\S+)\s+r = ([\d.]+)')
_DRAWDOWN = re.compile(r'^\s{2}(\S+)\s+.*?\s(-?[\d.]+)%\s.*?\s([\d.]+)\s+\*\*\* SEVERE DROP')


# ---------------------------------------------------------------- log parsing
def log_sections(text):
    """{step name: [lines]} for each ``--- [name]`` block, last occurrence wins.

    A block runs from its start line to its ``rc=`` line; a ``[run_daily]``
    line straight after it (a note the step's failure raised) is kept too."""
    sections, cur, name = {}, None, None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _START.match(line)
        if m:
            name, cur = m.group(1), []
        elif cur is not None and _END.match(line):
            j = i + 1
            while j < len(lines) and lines[j].startswith('[run_daily]'):
                cur.append(lines[j])
                j += 1
            sections[name] = cur
            cur = name = None
        elif cur is not None:
            cur.append(line)
        i += 1
    return sections


def _strip_logging(line):
    """'2026-11-02 22:26:21,727 WARNING analyze_stock: RUN QUALITY: x' -> 'RUN QUALITY: x'."""
    k = line.find('RUN QUALITY:')
    return line[k:] if k >= 0 else line.strip()


def run_quality(sections):
    lines = sections.get('analyze') or []
    rq = [_strip_logging(ln) for ln in lines if 'RUN QUALITY:' in ln]
    skip = [ln.strip() for ln in lines if 'Screen skip cache:' in ln and 'without a fetch' in ln]
    return rq, skip[-1:] if skip else []


def gate_block(lines):
    """The gate N/A table as printed, plus its JUMP rows."""
    body = [ln for ln in lines if ln.strip()]
    jumps = [ln.strip() for ln in body if 'JUMP' in ln and '⚠' in ln]
    return body, jumps


def momentum(lines):
    """Bucket rows, the Spearman line and the runbook's two momentum flags."""
    buckets = {}
    for ln in lines:
        m = _BUCKET.match(ln.strip())
        if m:
            buckets[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    table = [ln for ln in lines if _BUCKET.match(ln.strip()) or ln.startswith(('Rating ', 'SPY '))]
    sp = next((ln.strip() for ln in lines if 'Spearman r' in ln), None)
    flags = []
    m = _SPEARMAN.search(sp or '')
    if m and float(m.group(1)) > MOMENTUM_R_LIMIT and float(m.group(2)) < MOMENTUM_P_LIMIT:
        flags.append(f'Spearman r {m.group(1)} (p={m.group(2)}) is above +{MOMENTUM_R_LIMIT} — '
                     'the ratings are tracking momentum')

    def _mean(keys):
        n = sum(buckets[k][0] for k in keys if k in buckets)
        return sum(buckets[k][0] * buckets[k][1] for k in keys if k in buckets) / n if n else None
    top, bottom = _mean(('BUY', 'LEAN BUY')), _mean(('HOLD', 'PASS'))
    if top is not None and bottom is not None and top - bottom > MOMENTUM_GAP_PP:
        flags.append(f'BUY/LEAN BUY returned {top:.1f}% over the trailing year against '
                     f'{bottom:.1f}% for HOLD/PASS')
    return table, sp, flags


def portfolio_report(lines):
    """Runbook flags: sector share of the BUY/LEAN BUY bucket, r > 0.85 pairs,
    and BUYs with a 2020 drawdown worse than -50%."""
    sectors, in_conc = [], False
    for ln in lines:
        if 'CONCENTRATION SUMMARY' in ln:
            in_conc = True
        elif in_conc and ('DRAWDOWN RESILIENCE' in ln or 'with no sector data' in ln):
            in_conc = False
        elif in_conc:
            m = _SECTOR_ROW.match(ln)
            if m and m.group(1) != 'Sector':
                sectors.append((m.group(1).strip(), float(m.group(3))))
    pairs = [(a, b, float(r)) for a, b, r in (m.groups() for m in map(_PAIR.search, lines) if m)
             if float(r) > CORR_LIMIT]
    drops = []
    for ln in lines:
        m = _DRAWDOWN.match(ln)
        if m and float(m.group(2)) < DRAWDOWN_LIMIT_PCT and float(m.group(3)) >= BUY_SCORE:
            drops.append(m.group(1))
    return sectors, pairs, drops


def portfolio_alerts(lines):
    """The alerts text, with each portfolio's earnings WATCH lines folded into one."""
    out, earn = [], []

    def flush():
        if earn:
            out.append(f"  WATCH  earnings within 7 days ({len(earn)}): {', '.join(earn)}")
            earn.clear()
    for ln in lines:
        if ln.startswith('wrote ') or ln.startswith('portfolios: '):
            continue
        m = _EARNINGS.match(ln)
        if m:
            earn.append(f'{m.group(1)} {m.group(2)}')
            continue
        if not ln.strip() or not ln.startswith(' '):
            flush()
        elif ln.lstrip().startswith('(') and 'quieter change' in ln:
            flush()
        out.append(ln)
    flush()
    while out and not out[-1].strip():
        out.pop()
    return out


# ---------------------------------------------------------------- digest
def _dur(sec):
    sec = int(sec or 0)
    h, m = divmod(sec // 60, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m{sec % 60:02d}s'


def build(summary, log_text):
    """(subject, body) for one run."""
    rundate = summary.get('rundate', '?')
    status = str(summary.get('status', 'unknown'))
    steps = summary.get('steps') or []
    failed = [s for s in steps if s.get('rc') not in (0, 'skipped', 'dry-run')]
    sections = log_sections(log_text)

    subject = f'Stock model {rundate}: {status.upper()}'
    if failed:
        subject += ' — failed: ' + ', '.join(s['step'] for s in failed)
    notes = summary.get('notes') or []
    cov = next((n for n in notes if n.startswith('COVERAGE')), None)
    if cov and 'not published' in cov:
        subject += ' — under the coverage floor, not published'

    b = [f"Status: {status.upper()}   run date {rundate}   "
         f"finished {str(summary.get('finished_at', '?')).replace('T', ' ')}   "
         f"total {_dur(summary.get('total_seconds'))}", '']
    if status == 'skipped':
        return subject, '\n'.join(b + ['Market closed — run skipped.'] + [f'  {n}' for n in notes]) + '\n'

    if cov and 'not published' in cov:
        b += ['!! ' + cov, "!! Tonight's ratings are unreliable; both sites keep the last good report.",
              f'!! Re-run once the source recovers: scripts/run_daily.sh --from analyze --date {rundate}', '']

    if failed:
        b.append('FAILED STEPS')
        for s in failed:
            b.append(f"  {s['step']} rc={s['rc']} ({'blocking' if s.get('blocking') else 'non-blocking'})")
            tail = [ln for ln in sections.get(s['step'], []) if ln.strip()][-4:]
            b += [f'      {ln.strip()[:200]}' for ln in tail]
        b.append('')
    if notes:
        b += ['NOTES'] + [f'  - {n}' for n in notes] + ['']

    # Flags first: what a reader has to act on.
    flags = []
    gate_lines, jumps = gate_block(sections.get('gate_na_report', []))
    flags += [f'Gate N/A jump: {j}' for j in jumps]
    mom_table, sp, mom_flags = momentum(sections.get('validate_ratings', []))
    flags += mom_flags
    sectors, pairs, drops = portfolio_report(sections.get('portfolio_report', []))
    flags += [f'Sector concentration: {n} is {p:.1f}% of the BUY/LEAN BUY bucket (limit {SECTOR_LIMIT_PCT:.0f}%)'
              for n, p in sectors if p > SECTOR_LIMIT_PCT]
    store = sections.get('store_check', [])
    flags += [ln.strip() for ln in store if ln.startswith('PROBLEM')]
    pa_raw = sections.get('portfolio_alerts', [])
    flags = [ln.strip() for ln in pa_raw if ln.startswith('!!')] + flags
    b += ['FLAGS'] + ([f'  ! {f}' for f in flags] or ['  none']) + ['']

    pa = portfolio_alerts(pa_raw)
    if pa:
        b += ['YOUR PORTFOLIOS'] + [f'  {ln}' if ln else '' for ln in pa] + ['']

    b.append('CHECKS')
    if gate_lines:
        b.append(f"  Gate N/A: {len(jumps)} jump(s) ≥10 pts vs prior")
    if sp:
        b.append(f'  Momentum: {sp}')
    if sectors:
        top = max(sectors, key=lambda s: s[1])
        b.append(f'  Largest sector in BUY/LEAN BUY: {top[0]} {top[1]:.1f}%')
    if pairs:
        b.append(f"  Correlated pairs r > {CORR_LIMIT}: {len(pairs)} — "
                 + ', '.join(f'{a}/{c}' for a, c, _ in pairs[:12]) + (' …' if len(pairs) > 12 else ''))
    if drops:
        b.append(f'  BUYs with a 2020 drawdown worse than {DRAWDOWN_LIMIT_PCT:.0f}% ({len(drops)}): '
                 + ', '.join(drops))
    if store:
        b.append(f"  Snapshot store: {next((ln.strip() for ln in store if ln.strip()), '?')}")
    # Its PROBLEM lines are already under FAILED STEPS when it failed.
    db = next((ln.strip() for ln in sections.get('db_check', []) if ln.startswith('DB_CUTOVER_STREAK')), None)
    if db:
        b.append(f'  Database: {db}')
    cov_line = next((ln.strip() for ln in sections.get('coverage', []) if ln.startswith('COVERAGE')), None)
    if cov_line:
        b.append(f'  Coverage: {cov_line}')
    pub = [ln.strip() for ln in sections.get('publish', []) if ln.startswith('publish:')]
    pub += [ln.strip() for ln in sections.get('publish_cloudflare', []) if ln.startswith('live:')]
    b += [f'  Publish: {ln}' for ln in pub] or ['  Publish: not run']
    b.append('')

    rq, skip = run_quality(sections)
    if rq or skip:
        b += ['RUN QUALITY'] + [f'  {ln[:400]}' for ln in rq + skip] + ['']

    if gate_lines:
        b += ['GATE N/A COVERAGE'] + gate_lines + ['']
    if mom_table:
        b += ['MOMENTUM CHECK'] + mom_table + ([sp] if sp else []) + ['']

    b.append('STEPS')
    for s in steps:
        b.append(f"  {s['step']:<22} rc={str(s.get('rc')):<8} {_dur(s.get('seconds'))}")
    return subject, '\n'.join(b) + '\n'


def message(subject, body, sender, to):
    msg = EmailMessage()
    msg['Subject'], msg['From'], msg['To'] = subject, sender, to
    msg.set_content(body)
    # Monospace, so the tables line up in a mail client.
    msg.add_alternative('<pre style="font-family:Menlo,Consolas,monospace;font-size:12px">'
                        f'{html.escape(body)}</pre>', subtype='html')
    return msg


def send(msg, host, port, user, password, attempts=SEND_ATTEMPTS, wait=RETRY_WAIT_SEC, smtp=smtplib):
    """Send, retrying a transient failure once; a rejected login is not retried."""
    ctx = ssl.create_default_context()
    for attempt in range(1, attempts + 1):
        try:
            if port == 465:
                conn = smtp.SMTP_SSL(host, port, timeout=30, context=ctx)
            else:
                conn = smtp.SMTP(host, port, timeout=30)
            with conn as s:
                if port != 465:
                    s.starttls(context=ctx)
                s.login(user, password)
                s.send_message(msg)
            return
        except smtplib.SMTPAuthenticationError:
            raise
        except (smtplib.SMTPException, OSError) as exc:
            if attempt == attempts:
                raise
            print(f'email: attempt {attempt} failed ({exc}); retrying in {wait}s', file=sys.stderr)
            time.sleep(wait)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--date', required=True, help='the run date, YYYY-MM-DD')
    ap.add_argument('--summary', help='run_summary JSON (default output/run_summary_<date>.json)')
    ap.add_argument('--log', help='daily log (default ~/Library/Logs/StockModel/daily_<date>.log)')
    ap.add_argument('--to', help='recipient (default EMAIL_TO, else SMTP_USER)')
    ap.add_argument('--print', dest='show', action='store_true', help='print the email instead of sending it')
    a = ap.parse_args(argv)

    if os.environ.get('EMAIL_SUMMARY', '1') == '0' and not a.show:
        print('email: EMAIL_SUMMARY=0 — not sent')
        return 0
    summary_path = a.summary or os.path.join(REPO, 'output', f'run_summary_{a.date}.json')
    log_path = a.log or os.path.join(LOGDIR, f'daily_{a.date}.log')
    try:
        with open(summary_path, encoding='utf-8') as f:
            summary = json.load(f)
    except (OSError, ValueError) as exc:
        print(f'email: cannot read {summary_path}: {exc}', file=sys.stderr)
        return 1
    try:
        with open(log_path, encoding='utf-8', errors='replace') as f:
            log_text = f.read()
    except OSError as exc:
        print(f'email: no log at {log_path} ({exc}); sending the summary alone', file=sys.stderr)
        log_text = ''
    subject, body = build(summary, log_text)

    user = os.environ.get('SMTP_USER', '').strip()
    password = os.environ.get('SMTP_APP_PASSWORD', '').strip()
    to = a.to or os.environ.get('EMAIL_TO', '').strip() or user
    if a.show:
        print(f'To: {to or "(unset)"}\nSubject: {subject}\n\n{body}')
        return 0
    if not (user and password):
        print('email: SMTP_USER / SMTP_APP_PASSWORD not set (.env) — not sent')
        return 0
    host = os.environ.get('SMTP_HOST', '').strip() or DEFAULT_HOST
    port = int(os.environ.get('SMTP_PORT', '').strip() or DEFAULT_PORT)
    try:
        send(message(subject, body, user, to), host, port, user, password)
    except smtplib.SMTPAuthenticationError as exc:
        print(f'email: {host} rejected the login for {user} ({exc.smtp_code}) — '
              'check SMTP_APP_PASSWORD (an app-specific password, not the Apple ID one)', file=sys.stderr)
        return 1
    except (smtplib.SMTPException, OSError) as exc:
        print(f'email: send to {to} via {host}:{port} failed: {exc}', file=sys.stderr)
        return 1
    print(f'email: sent "{subject}" to {to}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
