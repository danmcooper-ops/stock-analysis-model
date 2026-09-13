# Stock Analysis Snapshots

Historical `results_YYYY-MM-DD.json` files from the daily analysis pipeline.
Each top-level `results_<date>.json[.gz]` is the snapshot for that NYSE
session. Newer days are archived gzipped (`scripts/archive_snapshot.py`).
Every reader handles both forms.

## Relabeled and retired snapshots (2026-09-13)

Fourteen snapshots were named after the calendar day their run *started*: a
weekend or a market holiday, after the previous session had closed. Their
prices are that previous session's closes. The mislabel distorted the
backtest: a Sunday label measured forward returns from Monday's close, and
days whose real session was also archived were counted twice. They were fixed
with `scripts/relabel_snapshots.py`, using this branch's copies as the source.

**Relabeled to the session they hold.** The top-level `date` was rewritten,
and `provenance.relabeled_from` / `provenance.relabel_reason` record the old
label:

| Was | Now | Evidence |
|---|---|---|
| 2026-05-03 (Sun) | 2026-05-01 (Fri) | weekend label; Friday not otherwise archived |
| 2026-06-07 (Sun) | 2026-06-05 (Fri) | weekend label; Friday not otherwise archived |
| 2026-07-25 (Sat) | 2026-07-24 (Fri) | run started Sat 16:49 UTC |
| 2026-08-15 (Sat) | 2026-08-14 (Fri) | run started Sat 08:47 UTC |
| 2026-08-30 (Sun) | 2026-08-28 (Fri) | run started Sun 19:47 UTC |
| 2026-09-05 (Sat) | 2026-09-04 (Fri) | run started Sat 11:49 UTC |

**Retired** to `retired/`. No snapshot reader lists that folder. Each one
duplicates a session that is already archived:
2026-04-26, 2026-05-02 (same session as the 05-03 file above), 2026-05-09,
2026-05-16 (partial run, 854 rows), 2026-05-23, 2026-05-30, and the holidays
2026-06-19 and 2026-07-03.

`scripts/market_open.py` now skips weekend and holiday runs, and the pipeline
fixes its run date at start (`--run-date`), so neither problem should recur.

## Known gaps

These sessions have no snapshot, and **will not be reconstructed**. The
pipeline reads live yfinance fundamentals, analyst data, the risk-free rate,
the macro regime, news and the ticker universe, none of which can be
recovered as of a past date. A rebuilt file would put look-ahead data into
the backtest corpus. Backtest readiness is span-based, so the gaps do not
block calibration.

- 2026-04: 27, 28, 30
- 2026-05: 04, 11, 18
- 2026-06: 01, 03, 08, 12, 15, 17, 30
- 2026-07: 09
- 2026-08: 06, 20, 24, 27
- 2026-09: 10

(2026-09-11 was produced after the fact on 2026-09-13, over the weekend.
Friday's close was still the newest price bar then, the same situation as the
relabeled weekend runs.)
