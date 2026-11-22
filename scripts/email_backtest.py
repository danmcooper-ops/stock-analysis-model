#!/usr/bin/env python3
"""Email the weekly backtest's results to the owner's inbox.

``scheduled-tasks/weekly-backtest/mac_run.sh`` calls this once
``cloud-weekly-backtest/run.sh`` has exited, so every run that gets that far —
OK, failed at a step, or dead before it could finish ``status.txt`` — mails
its outcome, whether or not the Claude scheduled task that launched it writes
its own summary. A run killed by a signal, a SMOKE run and a DRY_RUN send
nothing (mac_run.sh decides).

The digest follows the runbook's summary
(``scheduled-tasks/cloud-weekly-backtest/SKILL.md`` §4) and is built from
files the run already writes, under the work directory:

    status.txt                       steps, PUBLISHED, RESULT
    output/backtest_summary_<d>.json IC, buckets, models, coverage, deferred
    logs/04-prices.log               the price gate (and PROBLEM: lines)
    logs/04b-backfill.log            the Tiingo backfill tally
    logs/07-compare.log              REGRESSION: / NOTICE: lines
    logs/08-archive.log              returns/ sidecars added or changed

SMTP settings, the sender and the retry are scripts/email_summary.py's
(SMTP_USER, SMTP_APP_PASSWORD, SMTP_HOST, SMTP_PORT, EMAIL_TO; EMAIL_SUMMARY=0
turns this email off too). Stdlib only, so mac_run.sh's bare python3 runs it.

    python scripts/email_backtest.py --date 2026-11-15
    python scripts/email_backtest.py --date 2026-11-15 --print   # show, do not send
"""
import argparse
import glob
import json
import os
import re
import smtplib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.email_summary import DEFAULT_HOST, DEFAULT_PORT, message, send  # noqa: E402

WORK = os.environ.get('STOCK_MODEL_WORK') or os.path.expanduser(
    '~/Library/Application Support/StockModel/backtest')
LOGDIR = os.path.expanduser('~/Library/Logs/StockModel')
RATINGS = ('BUY', 'LEAN BUY', 'HOLD', 'PASS')
TAIL_LINES = 30
_STEP = re.compile(r'^(\S+) rc=(\S+) seconds=(\d+)$')


# ---------------------------------------------------------------- inputs
def read_status(text):
    """{'steps': [(name, rc, seconds)], 'RUNDATE': .., 'RESULT': .., ...}."""
    out = {'steps': []}
    for line in text.splitlines():
        m = _STEP.match(line.strip())
        if m:
            out['steps'].append((m.group(1), m.group(2), int(m.group(3))))
            continue
        key, _, val = line.strip().partition(' ')
        if key.isupper() and key:
            out[key] = val
    return out


def _read(path):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return f.read()
    except OSError:
        return ''


def _tail(text, n=TAIL_LINES):
    return [ln.rstrip() for ln in text.splitlines() if ln.strip()][-n:]


def prior_summary(out_dir, date):
    """The newest backtest summary dated before *date* in *out_dir*, or None."""
    cands = sorted(p for p in glob.glob(os.path.join(out_dir, 'backtest_summary_*.json'))
                   if os.path.basename(p)[len('backtest_summary_'):-len('.json')] < date)
    if not cands:
        return None
    try:
        with open(cands[-1], encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- formatting
def _pct(x, signed=True):
    if x is None:
        return 'n/a'
    return f'{x * 100:+.2f}%' if signed else f'{x * 100:.1f}%'


def _num(x, fmt='+.3f'):
    return 'n/a' if x is None else format(x, fmt)


def _dur(sec):
    sec = int(sec or 0)
    h, m = divmod(sec // 60, 60)
    return f'{h}h{m:02d}m' if h else f'{m}m{sec % 60:02d}s'


def ic_lines(ic):
    """One line per horizon of a composite_ic block."""
    out = []
    for h in sorted(ic or {}, key=int):
        e = ic[h]
        bs = e.get('bootstrap') or {}
        ci = (f"90% CI [{_num(bs.get('ci_low'))}, {_num(bs.get('ci_high'))}]"
              if bs.get('ci_low') is not None else 'no CI (under 3 independent periods)')
        out.append(f"  {h:>3}d  IC {_num(e.get('mean_ic'))}  t(eff) {_num(e.get('t_stat_effective'), '.2f')}"
                   f"  eff n {e.get('effective_n')}  IC>0 {_pct(e.get('frac_positive'), False)}"
                   f"  snapshots {e.get('n_snapshots')}  {ci}")
    return out


def ic_verdict(ic):
    """The runbook's headline: t(eff) against |t| = 2."""
    e = (ic or {}).get('30') or {}
    t = e.get('t_stat_effective')
    if t is None:
        return '30d composite IC: not measured'
    real = 'clears' if abs(t) >= 2 else 'is under'
    return (f"30d composite IC {_num(e.get('mean_ic'))} at t(eff) {t:.2f} {real} the |t| >= 2 bar"
            + ('' if abs(t) >= 2 else ' — not yet distinguishable from zero'))


def bucket_order_broken(buckets):
    """{horizon: [out-of-order pairs]} where mean excess is not BUY > LEAN BUY > HOLD > PASS."""
    out = {}
    for h, b in (buckets or {}).items():
        means = [(r, (b.get(r) or {}).get('mean_excess')) for r in RATINGS]
        means = [(r, m) for r, m in means if m is not None]
        bad = [f'{a} < {c}' for (a, x), (c, y) in zip(means, means[1:], strict=False) if x < y]
        if bad:
            out[h] = bad
    return out


def bucket_lines(buckets, inference):
    out = []
    for h in sorted(buckets or {}, key=int):
        b, inf = buckets[h], (inference or {}).get(h) or {}
        out.append(f'  {h}d')
        for r in RATINGS:
            e = b.get(r)
            if not e:
                continue
            i = inf.get(r) or {}
            ci = (f"  per-snapshot {_pct(i.get('mean'))} [{_pct(i.get('ci_low'))}, {_pct(i.get('ci_high'))}]"
                  if i.get('ci_low') is not None else '')
            out.append(f"    {r:<9} mean {_pct(e.get('mean_excess'))}  median {_pct(e.get('median_excess'))}"
                       f"  beat SPY {_pct(e.get('hit_rate'), False)}  n {e.get('n')}{ci}")
        sp = inf.get('spread') or {}
        if sp.get('mean') is not None:
            ci = (f" [{_pct(sp.get('ci_low'))}, {_pct(sp.get('ci_high'))}]"
                  if sp.get('ci_low') is not None else '')
            out.append(f"    BUY+LEAN BUY - PASS {_pct(sp.get('mean'))}{ci}")
    return out


def coverage_lines(cov):
    if not cov:
        return []
    low = min(cov, key=lambda r: r.get('coverage', 1))
    kinds, gone, unpriced = {}, 0, {}
    for r in cov:
        for v in (r.get('delisted_by_rating') or {}).values():
            for kind, n in (v.items() if isinstance(v, dict) else [('?', v)]):
                kinds[kind] = kinds.get(kind, 0) + (n if isinstance(n, int) else len(n))
        g = r.get('gone_before_snapshot') or 0
        gone += g if isinstance(g, int) else len(g)
        for rating, n in (r.get('unpriced_by_rating') or {}).items():
            unpriced[rating] = unpriced.get(rating, 0) + (n if isinstance(n, int) else len(n))
    out = [f"  Lowest coverage {low.get('coverage', 0):.3f} ({low.get('run_date')} +{low.get('horizon')}d)"]
    out.append('  Delisted rows measured to last close: '
               + (', '.join(f'{k} {n}' for k, n in sorted(kinds.items())) or 'none'))
    out.append(f'  Rows already delisted at the snapshot (excluded): {gone}')
    out.append('  Unpriced rows: ' + (', '.join(f'{r} {unpriced[r]}' for r in RATINGS if r in unpriced) or 'none'))
    return out


def regime_lines(regimes, rescored):
    out = []
    for g in regimes or []:
        ic = g.get('composite_ic') or {}
        parts = [f"{h}d IC {_num((ic[h] or {}).get('mean_ic'))} (t {_num((ic[h] or {}).get('t_stat_effective'), '.2f')})"
                 for h in sorted(ic, key=int) if ic.get(h)]
        out.append(f"  {g.get('params_hash')}  {g.get('first')}..{g.get('last')}  "
                   f"{g.get('snapshots')} snapshot(s)  " + ', '.join(parts))
    if rescored:
        out.append(f"  Current model {rescored.get('params_hash')} re-scored over "
                   f"{rescored.get('snapshots')} snapshot(s) (in-sample if its weights were tuned on this corpus):")
        out += ic_lines(rescored.get('composite_ic'))
        out += bucket_lines(rescored.get('rating_buckets'), rescored.get('rating_bucket_inference'))
    return out


def readiness_lines(rd):
    out = []
    for h in sorted(rd or {}, key=int):
        e = rd[h]
        out.append(f"  {h:>3}d  matured {e.get('n_matured')}  eff n {e.get('effective_n')}"
                   f"  calibrate {e.get('date_calibrate') or 'ready'}  IC test {e.get('date_ic_test') or 'ready'}")
    return out


def archive_lines(text):
    added = sum(1 for ln in text.splitlines() if re.match(r'^A\treturns/', ln))
    changed = sum(1 for ln in text.splitlines() if re.match(r'^M\treturns/', ln))
    commit = next((ln.split(': ', 1)[1] for ln in text.splitlines()
                   if ln.startswith('commit built')), None)
    out = []
    if commit:
        out.append(f'  Commit: {commit}')
    if added or changed:
        out.append(f'  returns/ sidecars: {added} added, {changed} changed')
    return out


# ---------------------------------------------------------------- digest
def build(date, status_text, logs, summary=None, prior=None, run_log=''):
    """(subject, body). *logs* maps a step name ('04-prices', ...) to its log text."""
    st = read_status(status_text)
    result = st.get('RESULT')
    if st.get('RUNDATE') != date or not result:
        subject = f'Weekly backtest {date}: DID NOT FINISH'
        body = [f'The run for {date} ended without a RESULT in status.txt '
                f"(status.txt is for {st.get('RUNDATE') or 'no run'}): a crash, a kill, or the lock "
                'refusing a second copy. Nothing can be assumed measured or published.', '',
                'LAST LINES OF THE RUN LOG'] + [f'  {ln}' for ln in _tail(run_log)]
        return subject, '\n'.join(body) + '\n'

    published = st.get('PUBLISHED', 'unknown')
    subject = f'Weekly backtest {date}: {result}'
    b = [f'Result: {result}   run date {date}   elapsed {_dur(st.get("ELAPSED_SECONDS"))}',
         f'Published: {published}']
    if result == 'OK-DRYRUN':
        b.append('Dry run — nothing was published.')
    soft = st.get('SOFT_FAILURES', 'none')
    if soft and soft != 'none':
        b.append(f'Soft failures: {soft}')
    b.append('')

    if result.startswith('FAILED'):
        step = result[len('FAILED at '):].split(' ', 1)[0]
        log_key = next((k for k in logs if k == step or k.endswith(step) or step.endswith(k)), step)
        text = logs.get(log_key, '')
        probs = [ln.strip() for ln in text.splitlines() if ln.startswith('PROBLEM')]
        if probs:
            b += ['PROBLEMS'] + [f'  {p}' for p in probs] + ['']
        b += [f'LAST LINES OF {log_key}.log'] + [f'  {ln}' for ln in _tail(text)] + ['']
        if step not in ('archive', '08-archive'):
            b += ['Nothing was measured or published past that step.', '']
        if not summary:
            b += _steps(st)
            return subject, '\n'.join(b) + '\n'

    s = summary or {}
    compare = logs.get('07-compare', '')
    regressions = [ln.strip() for ln in compare.splitlines() if ln.startswith('REGRESSION')]
    notices = [ln.strip() for ln in compare.splitlines() if ln.startswith('NOTICE')]
    if regressions:
        subject += f' — {len(regressions)} regression(s)'

    flags = list(regressions)
    broken = bucket_order_broken(s.get('rating_buckets'))
    prior_broken = bucket_order_broken((prior or {}).get('rating_buckets'))
    for h, bad in sorted(broken.items(), key=lambda kv: int(kv[0])):
        again = ' (also last week)' if h in prior_broken else ''
        flags.append(f"{h}d buckets out of order: {', '.join(bad)}{again}")
    if s.get('skipped_snapshots'):
        flags.append(f"{len(s['skipped_snapshots'])} snapshot(s) skipped (missing gate fields?)")

    b += ['HEADLINE', f'  {ic_verdict(s.get("composite_ic"))}', '']
    b += ['FLAGS'] + ([f'  ! {f}' for f in flags] or ['  none']) + ['']
    b += ['COMPOSITE-SCORE RANK IC (as recorded)'] + ic_lines(s.get('composite_ic')) + ['']
    b += ['RATING BUCKETS (excess return vs SPY, as recorded)'] + bucket_lines(
        s.get('rating_buckets'), s.get('rating_bucket_inference')) + ['']
    rl = regime_lines(s.get('regimes'), s.get('rescored_current'))
    if rl:
        b += ['SCORING MODELS'] + rl + ['']

    cov = coverage_lines(s.get('coverage'))
    tally = next((ln.strip() for ln in logs.get('04b-backfill', '').splitlines()
                  if ln.startswith('[backfill]')), None)
    gate = [ln.strip() for ln in logs.get('04-prices', '').splitlines()
            if ln.startswith('SPY prices end') or 'have a price file' in ln]
    if cov or tally or gate:
        b += ['COVERAGE AND ATTRITION'] + cov + [f'  {g}' for g in gate]
        if tally:
            b.append(f'  {tally}')
        b.append('')

    deferred = s.get('deferred') or []
    b += ['DEFERRED PAIRS (normal: the eval date has no SPY bar yet)']
    b += [f"  {d.get('run_date')} +{d.get('horizon')}d — {d.get('reason')}" for d in deferred] or ['  none']
    b.append('')
    b += ['WEEK OVER WEEK'] + [f'  {ln}' for ln in regressions or ['no regressions']]
    b += [f'  {ln}' for ln in notices] + ['']
    b += ['READINESS'] + readiness_lines(s.get('readiness')) + ['']
    b += ['ARCHIVE', f'  PUBLISHED {published}'] + archive_lines(logs.get('08-archive', '')) + ['']
    b += _steps(st)
    return subject, '\n'.join(b) + '\n'


def _steps(st):
    return ['STEPS'] + [f'  {n:<18} rc={rc:<4} {_dur(sec)}' for n, rc, sec in st['steps']]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--date', required=True, help='the run date (RUNDATE), YYYY-MM-DD')
    ap.add_argument('--work', default=WORK, help='the run.sh work directory')
    ap.add_argument('--log', help='full run log (default ~/Library/Logs/StockModel/weekly_<date>.log)')
    ap.add_argument('--to', help='recipient (default EMAIL_TO, else SMTP_USER)')
    ap.add_argument('--print', dest='show', action='store_true', help='print the email instead of sending it')
    a = ap.parse_args(argv)

    if os.environ.get('EMAIL_SUMMARY', '1') == '0' and not a.show:
        print('email: EMAIL_SUMMARY=0 — not sent')
        return 0
    status_text = _read(os.path.join(a.work, 'status.txt'))
    logs = {os.path.basename(p)[:-len('.log')]: _read(p)
            for p in glob.glob(os.path.join(a.work, 'logs', '*.log'))}
    out_dir = os.path.join(a.work, 'output')
    summary = None
    st = read_status(status_text)
    spath = st.get('SUMMARY') or os.path.join(out_dir, f'backtest_summary_{a.date}.json')
    if os.path.isfile(spath):
        try:
            with open(spath, encoding='utf-8') as f:
                summary = json.load(f)
        except (OSError, ValueError) as exc:
            print(f'email: cannot read {spath}: {exc}', file=sys.stderr)
    run_log = _read(a.log or os.path.join(LOGDIR, f'weekly_{a.date}.log'))
    subject, body = build(a.date, status_text, logs, summary, prior_summary(out_dir, a.date), run_log)

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
