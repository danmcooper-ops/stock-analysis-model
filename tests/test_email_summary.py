# tests/test_email_summary.py
"""scripts/email_summary.py: the nightly run's summary email.

The digest is lifted from the daily log's step sections, so these tests pin
the parsing against log text in the shape run_daily.sh writes, and the send
path against a fake SMTP module (nothing here opens a socket)."""

import json
import smtplib

import pytest

from scripts import email_summary as es

LOG = """\
=== run_daily start Mon Nov  2 17:06:26 EST 2026 RUNDATE=2026-11-02 from=preflight dry_run=0 ===

--- [pull] 17:35:23: pull_main
old attempt line
--- [pull] rc=0

--- [pull] 17:36:00: pull_main
You have not agreed to the Xcode license agreements.
--- [pull] rc=1
[run_daily] git pull --ff-only failed — rendering from the current checkout (stale templates possible)

--- [analyze] 17:40:00: python scripts/analyze_stock.py
  Screen skip cache: 2896 ticker(s) skipped without a fetch; 4847 remembered for the next run
2026-11-02 22:26:21,727 WARNING analyze_stock: RUN QUALITY: 408 tickers priced without a local regression beta
--- [analyze] rc=0

--- [portfolio_alerts] 22:36:31: python scripts/portfolios.py alerts
portfolios: stopped trading since 2026-10-30: ACVA
Portfolio alerts 2026-11-02 (vs 2026-10-30)

!! Systemic day (data): 40% of the universe re-rated

Paper book [paper]: 3 stocks  today: +1 up / -0 down
  BUY:3  median MoS +26%
  WATCH  ANET reports earnings on 2026-11-03
  ACTION  XYZ rating BUY → HOLD
  WATCH  CF reports earnings on 2026-11-04
  (4 quieter changes: moves within a side or on missing data)

0 action, 2 watch, 4 FYI
wrote output/portfolio_alerts_2026-11-02.txt
--- [portfolio_alerts] rc=0

--- [portfolio_report] 22:38:16: python scripts/portfolio_report.py
  HIGH CORRELATION PAIRS (r > 0.85):
    *** TNK <-> TK  r = 0.916  — consider overlap / redundancy
  3. CONCENTRATION SUMMARY  (BUY + LEAN BUY bucket)
  Sector                            Count  % of bucket   BUY   LEAN BUY
  ------------------------------------------------------------------------
  Industrials                         201       41.1%    20        181   ##########
  Technology                          188       19.7%    35        153   #########
  1 with no sector data (excluded; % is of the 952 with a sector)
  4. DRAWDOWN RESILIENCE  (ranked by 2020 drawdown, worst first)
  Ticker   Sector                         2020 DD   2022 DD   2008 DD   Comp Score  Flag
  MTDR     Energy                          -93.9%    -34.6%     n/a           54.3  *** SEVERE DROP
  BBW      Consumer Cyclical               -81.3%    -43.4%    -77.6%         70.4  *** SEVERE DROP
  ROST     Consumer Cyclical               -41.4%    -37.5%    -46.5%         59.7  *** SEVERE DROP
--- [portfolio_report] rc=0

--- [gate_na_report] 22:38:36: python scripts/gate_na_report.py
  Gate                   Category      N/A   N/A %  Masked %  Missing %   Δ pts  Flags
  MoS                    Valuation     870   22.6%      0.0%      22.6%   +12.2  ⚠ JUMP
--- [gate_na_report] rc=0

--- [validate_ratings] 22:38:37: python scripts/validate_ratings.py
Rating          N   Mean Ret    Med Ret    vs SPY    Hit%
BUY           100     40.0%      2.0%    -7.4%   35.2%
LEAN BUY      100     30.0%     11.4%   +11.0%   50.5%
HOLD          100     10.0%     17.3%   +64.6%   57.2%
PASS          100      5.0%      2.7%  +144.3%   42.8%
  Spearman r = +0.210  (p=0.001)
--- [validate_ratings] rc=0

--- [store_check] 22:38:46: python scripts/check_snapshot_store.py
PROBLEM: store holds no rows for 2026-11-02
--- [store_check] rc=1

--- [coverage] 22:38:54: coverage_check
COVERAGE ok rows=3854 prior_rows=3843 (2026-10-30) ratio=1.003 min=0.70
--- [coverage] rc=0

--- [publish] 22:38:55: publish
publish: https://example.github.io/ -> 200
--- [publish] rc=0
"""

SUMMARY = {
    'rundate': '2026-11-02', 'status': 'degraded', 'finished_at': '2026-11-02T22:40:33',
    'total_seconds': 20047,
    'steps': [{'step': 'pull', 'rc': 1, 'seconds': 0, 'blocking': False},
              {'step': 'analyze', 'rc': 0, 'seconds': 17459, 'blocking': True},
              {'step': 'store_check', 'rc': 1, 'seconds': 0, 'blocking': False}],
    'notes': ['git pull --ff-only failed — rendering from the current checkout (stale templates possible)'],
}


def test_sections_take_the_last_occurrence_and_the_note_after_rc():
    sec = es.log_sections(LOG)
    assert sec['pull'] == ['You have not agreed to the Xcode license agreements.',
                           '[run_daily] git pull --ff-only failed — rendering from the current '
                           'checkout (stale templates possible)']


def test_subject_names_status_and_failed_steps():
    subject, _ = es.build(SUMMARY, LOG)
    assert subject == 'Stock model 2026-11-02: DEGRADED — failed: pull, store_check'


def test_flags_follow_the_runbook_rules():
    _, body = es.build(SUMMARY, LOG)
    flags = body.split('FLAGS\n', 1)[1].split('\n\n', 1)[0]
    assert '!! Systemic day (data)' in flags                       # banner leads
    assert 'Gate N/A jump' in flags and 'MoS' in flags             # ⚠ JUMP row
    assert 'Spearman r +0.210' in flags                            # r > 0.15, p < 0.05
    assert 'BUY/LEAN BUY returned 35.0%' in flags                  # beats HOLD/PASS by > 5pp
    assert 'Industrials is 41.1%' in flags and 'Technology' not in flags
    assert 'PROBLEM: store holds no rows' in flags


def test_only_buys_count_toward_the_2020_drawdown_flag():
    # MTDR scores 54.3 (LEAN BUY); ROST's 2020 drawdown is within the limit.
    _, body = es.build(SUMMARY, LOG)
    assert 'BUYs with a 2020 drawdown worse than -50% (1): BBW' in body


def test_earnings_watch_lines_fold_into_one_and_others_stay_verbatim():
    out = es.portfolio_alerts(es.log_sections(LOG)['portfolio_alerts'])
    text = '\n'.join(out)
    assert '  ACTION  XYZ rating BUY → HOLD' in out
    assert 'WATCH  earnings within 7 days (2): ANET 11-03, CF 11-04' in text
    assert 'reports earnings' not in text
    assert not any(ln.startswith(('wrote ', 'portfolios: ')) for ln in out)
    assert text.index('earnings within') < text.index('quieter change')


def test_run_quality_and_publish_lines():
    _, body = es.build(SUMMARY, LOG)
    assert 'RUN QUALITY: 408 tickers priced without a local regression beta' in body
    assert 'Screen skip cache: 2896 ticker(s) skipped without a fetch' in body
    assert 'Publish: publish: https://example.github.io/ -> 200' in body


def test_coverage_floor_leads_the_email():
    s = dict(SUMMARY, notes=['COVERAGE degraded rows=739 prior_rows=2514 ratio=0.29 min=0.70 — '
                             'archived, but not published; the sites keep the last good report'])
    subject, body = es.build(s, LOG)
    assert subject.endswith('under the coverage floor, not published')
    assert body.split('\n')[2].startswith('!! COVERAGE degraded')
    assert '--from analyze --date 2026-11-02' in body


def test_a_skipped_run_is_a_short_email():
    s = {'rundate': '2026-11-27', 'status': 'skipped', 'finished_at': '2026-11-27T17:00:01',
         'total_seconds': 0, 'steps': [{'step': 'preflight', 'rc': 10, 'seconds': 0, 'blocking': True}],
         'notes': ['market closed — run skipped']}
    subject, body = es.build(s, '')
    assert subject.startswith('Stock model 2026-11-27: SKIPPED')
    assert 'Market closed — run skipped.' in body and 'FLAGS' not in body


def test_message_carries_a_monospace_html_part():
    msg = es.message('s', 'a < b\n', 'me@example.com', 'you@example.com')
    html_part = msg.get_body(('html',)).get_content()
    assert '<pre' in html_part and 'a &lt; b' in html_part
    assert msg.get_body(('plain',)).get_content() == 'a < b\n'


class _FakeSMTP:
    sent, fail = [], []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.tls = True

    def login(self, user, password):
        if _FakeSMTP.fail:
            raise _FakeSMTP.fail.pop(0)

    def send_message(self, msg):
        _FakeSMTP.sent.append((self.host, self.port, msg['To']))


class _FakeModule:
    SMTP = SMTP_SSL = _FakeSMTP


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakeSMTP.sent, _FakeSMTP.fail = [], []


def test_send_retries_a_transient_failure_once():
    _FakeSMTP.fail = [smtplib.SMTPServerDisconnected('dropped')]
    es.send(es.message('s', 'b', 'me@x', 'you@x'), 'h', 587, 'u', 'p', wait=0, smtp=_FakeModule)
    assert _FakeSMTP.sent == [('h', 587, 'you@x')]


def test_send_does_not_retry_a_rejected_login():
    _FakeSMTP.fail = [smtplib.SMTPAuthenticationError(535, b'bad')]
    with pytest.raises(smtplib.SMTPAuthenticationError):
        es.send(es.message('s', 'b', 'me@x', 'you@x'), 'h', 587, 'u', 'p', wait=0, smtp=_FakeModule)
    assert _FakeSMTP.sent == []


@pytest.fixture
def run_files(tmp_path):
    summary = tmp_path / 'run_summary_2026-11-02.json'
    summary.write_text(json.dumps(SUMMARY), encoding='utf-8')
    log = tmp_path / 'daily.log'
    log.write_text(LOG, encoding='utf-8')
    return ['--date', '2026-11-02', '--summary', str(summary), '--log', str(log)]


def test_main_without_credentials_sends_nothing_and_succeeds(run_files, monkeypatch, capsys):
    for k in ('SMTP_USER', 'SMTP_APP_PASSWORD', 'EMAIL_SUMMARY'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(es, 'send', lambda *a, **k: pytest.fail('must not send'))
    assert es.main(run_files) == 0
    assert 'not set' in capsys.readouterr().out


def test_main_sends_to_email_to(run_files, monkeypatch):
    monkeypatch.setenv('SMTP_USER', 'me@icloud.example')
    monkeypatch.setenv('SMTP_APP_PASSWORD', 'abcd-efgh')
    monkeypatch.setenv('EMAIL_TO', 'inbox@example.com')
    monkeypatch.delenv('EMAIL_SUMMARY', raising=False)
    calls = []
    monkeypatch.setattr(es, 'send', lambda msg, host, port, user, pw: calls.append((msg, host, port)))
    assert es.main(run_files) == 0
    msg, host, port = calls[0]
    assert (host, port) == ('smtp.mail.me.com', 587)
    assert msg['To'] == 'inbox@example.com' and msg['From'] == 'me@icloud.example'
    assert msg['Subject'].startswith('Stock model 2026-11-02: DEGRADED')


def test_main_reports_a_send_failure_with_rc_1(run_files, monkeypatch):
    monkeypatch.setenv('SMTP_USER', 'me@x')
    monkeypatch.setenv('SMTP_APP_PASSWORD', 'p')
    monkeypatch.delenv('EMAIL_SUMMARY', raising=False)

    def boom(*a, **k):
        raise smtplib.SMTPAuthenticationError(535, b'bad')
    monkeypatch.setattr(es, 'send', boom)
    assert es.main(run_files) == 1


def test_email_summary_0_turns_it_off(run_files, monkeypatch):
    monkeypatch.setenv('EMAIL_SUMMARY', '0')
    monkeypatch.setenv('SMTP_USER', 'me@x')
    monkeypatch.setenv('SMTP_APP_PASSWORD', 'p')
    monkeypatch.setattr(es, 'send', lambda *a, **k: pytest.fail('must not send'))
    assert es.main(run_files) == 0


def test_run_daily_emails_after_writing_the_summary():
    src = open(es.os.path.join(es.REPO, 'scripts', 'run_daily.sh'), encoding='utf-8').read()
    body = src[src.index('finish() {'):]
    body = body[:body.index('\n}\n')]
    assert body.index('write_summary') < body.index('email_summary')
