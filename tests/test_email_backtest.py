# tests/test_email_backtest.py
"""scripts/email_backtest.py: the weekly backtest's results email.

The digest is built from status.txt, the summary JSON and a few step logs in
the shapes run.sh and backtest.py write; nothing here opens a socket."""

import json

from scripts import email_backtest as eb

STATUS_OK = """\
01-venv rc=0 seconds=1
04-prices rc=0 seconds=983
06-measure rc=0 seconds=137
08-archive rc=0 seconds=7
RUNDATE 2026-11-15
SUMMARY /nowhere/backtest_summary_2026-11-15.json
ELAPSED_SECONDS 1352
SOFT_FAILURES none
PUBLISHED pushed 9369d65b
RESULT OK
"""


def _ic(mean, t, lo=None, hi=None, n=4):
    return {'n_snapshots': 66, 'mean_ic': mean, 'frac_positive': 0.8, 'effective_n': n,
            't_stat_effective': t, 'bootstrap': {'ci_low': lo, 'ci_high': hi}}


def _buckets(buy, lean, hold, pss):
    return {r: {'n': 10, 'mean_excess': m, 'median_excess': m, 'hit_rate': 0.5}
            for r, m in zip(eb.RATINGS, (buy, lean, hold, pss), strict=True)}


SUMMARY = {
    'composite_ic': {'30': _ic(0.078, 1.93, 0.033, 0.125), '90': _ic(0.126, 2.47, n=1)},
    'rating_buckets': {'30': _buckets(0.007, 0.012, 0.011, -0.015)},
    'rating_bucket_inference': {'30': {'spread': {'mean': 0.0236, 'ci_low': 0.0043, 'ci_high': 0.0444}}},
    'regimes': [{'params_hash': 'pre-fingerprint', 'first': '2026-07-06', 'last': '2026-09-25',
                 'snapshots': 53, 'composite_ic': {'30': _ic(0.053, 1.34)}}],
    'rescored_current': {'params_hash': 'ef97d734', 'snapshots': 66,
                         'composite_ic': {'30': _ic(0.075, 1.35, 0.01, 0.147)}},
    'coverage': [
        {'run_date': '2026-07-06', 'horizon': 30, 'coverage': 0.95, 'gone_before_snapshot': 5,
         'delisted_by_rating': {'HOLD': {'merger': 4}, 'PASS': {'performance': 1}},
         'unpriced_by_rating': {'BUY': 4, 'PASS': 41}},
        {'run_date': '2026-07-10', 'horizon': 30, 'coverage': 0.9478, 'gone_before_snapshot': 2},
    ],
    'deferred': [{'run_date': '2026-10-16', 'horizon': 30, 'reason': 'SPY bars end 2026-11-13'}],
    'readiness': {'30': {'n_matured': 68, 'effective_n': 4, 'date_calibrate': '2027-03-03',
                         'date_ic_test': '2027-10-29'}},
    'skipped_snapshots': [],
}

LOGS = {
    '04-prices': 'SPY prices end 2026-11-13\n2655 of 2681 ticker(s) have a price file (99.0%, floor 90%)\n',
    '04b-backfill': '[backfill] 68 ticker(s) without usable prices: 30 fetched, 0 deferred\n',
    '07-compare': 'snapshots measured: 61 -> 66\nNOTICE: model pooled\nno regressions\n',
    '08-archive': ('commit built (not yet pushed): 9369d65b Weekly backtest: 2026-11-15\n'
                   'A\treturns/2026-10-08_h30.json\nM\treturns/2026-09-22_h30.json\n'
                   'A\tprice_backfill/X.json\n'),
}


def test_ok_digest_carries_the_runbook_sections():
    subject, body = eb.build('2026-11-15', STATUS_OK, LOGS, SUMMARY)
    assert subject == 'Weekly backtest 2026-11-15: OK'
    assert 'Published: pushed 9369d65b' in body
    assert 'is under the |t| >= 2 bar' in body
    assert '90% CI [+0.033, +0.125]' in body and 'no CI (under 3 independent periods)' in body
    assert 'BUY+LEAN BUY - PASS +2.36% [+0.43%, +4.44%]' in body
    assert 'pre-fingerprint' in body and 'ef97d734 re-scored over 66' in body
    assert 'Lowest coverage 0.948 (2026-07-10 +30d)' in body
    assert 'merger 4, performance 1' in body and 'excluded): 7' in body
    assert '2026-10-16 +30d' in body
    assert 'no regressions' in body and 'NOTICE: model pooled' in body
    # Only returns/ sidecars are counted, not the price backfill cache.
    assert 'returns/ sidecars: 1 added, 1 changed' in body
    assert '30d buckets out of order: BUY < LEAN BUY' in body


def test_reversal_names_last_week_when_it_persists():
    prior = {'rating_buckets': {'30': _buckets(0.004, 0.006, 0.007, -0.018)}}
    _, body = eb.build('2026-11-15', STATUS_OK, LOGS, SUMMARY, prior)
    assert 'BUY < LEAN BUY (also last week)' in body


def test_regressions_reach_the_subject():
    logs = dict(LOGS, **{'07-compare': 'REGRESSION: 2026-10-16 +30d deferred two weeks running\n'})
    subject, body = eb.build('2026-11-15', STATUS_OK, logs, SUMMARY)
    assert subject.endswith('— 1 regression(s)')
    assert '! REGRESSION: 2026-10-16 +30d deferred two weeks running' in body


def test_failed_prices_quotes_problems_and_tail():
    status = ('04-prices rc=1 seconds=900\nRUNDATE 2026-11-15\nSOFT_FAILURES none\n'
              'RESULT FAILED at 04-prices\n')
    logs = {'04-prices': 'AAA: empty response\nPROBLEM: only 60% of tickers have a price file\n'}
    subject, body = eb.build('2026-11-15', status, logs, None)
    assert subject == 'Weekly backtest 2026-11-15: FAILED at 04-prices'
    assert 'PROBLEM: only 60% of tickers' in body
    assert 'LAST LINES OF 04-prices.log' in body and 'AAA: empty response' in body
    assert 'Nothing was measured or published past that step.' in body


def test_run_without_result_reads_as_not_finished():
    stale = STATUS_OK.replace('2026-11-15', '2026-11-08')
    subject, body = eb.build('2026-11-15', stale, {}, None, run_log='== 04-prices\nKilled\n')
    assert subject == 'Weekly backtest 2026-11-15: DID NOT FINISH'
    assert 'status.txt is for 2026-11-08' in body and 'Killed' in body


def test_dry_run_says_nothing_was_published():
    status = STATUS_OK.replace('PUBLISHED pushed 9369d65b', 'PUBLISHED no (DRY_RUN)').replace(
        'RESULT OK', 'RESULT OK-DRYRUN')
    _, body = eb.build('2026-11-15', status, LOGS, SUMMARY)
    assert 'Dry run — nothing was published.' in body


def test_prior_summary_is_the_newest_before_the_date(tmp_path):
    for d, v in (('2026-11-01', 1), ('2026-11-08', 2), ('2026-11-15', 3)):
        (tmp_path / f'backtest_summary_{d}.json').write_text(json.dumps({'v': v}), encoding='utf-8')
    assert eb.prior_summary(str(tmp_path), '2026-11-15') == {'v': 2}
    assert eb.prior_summary(str(tmp_path), '2026-11-01') is None


def test_main_prints_without_smtp(tmp_path, capsys, monkeypatch):
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'output').mkdir()
    (tmp_path / 'status.txt').write_text(STATUS_OK.replace('/nowhere', str(tmp_path / 'output')),
                                         encoding='utf-8')
    (tmp_path / 'output' / 'backtest_summary_2026-11-15.json').write_text(json.dumps(SUMMARY),
                                                                          encoding='utf-8')
    for k, v in LOGS.items():
        (tmp_path / 'logs' / f'{k}.log').write_text(v, encoding='utf-8')
    monkeypatch.delenv('SMTP_USER', raising=False)
    monkeypatch.delenv('SMTP_APP_PASSWORD', raising=False)
    assert eb.main(['--date', '2026-11-15', '--work', str(tmp_path), '--log', '/dev/null', '--print']) == 0
    out = capsys.readouterr().out
    assert 'Subject: Weekly backtest 2026-11-15: OK' in out and 'Lowest coverage' in out
    # Not --print and no credentials: says so and sends nothing.
    assert eb.main(['--date', '2026-11-15', '--work', str(tmp_path), '--log', '/dev/null']) == 0
    assert 'not sent' in capsys.readouterr().out
