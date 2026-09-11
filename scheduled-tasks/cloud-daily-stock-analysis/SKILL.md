---
name: cloud-daily-stock-analysis
description: End-of-day stock analysis as a Claude Code cloud Routine — a fresh container each weekday runs run.sh, archives the snapshot to data/snapshots and publishes the report to GitHub Pages
---

You are running the end-of-day stock analysis routine **in a Claude Code cloud
session**. Nothing persists between runs except what is on GitHub, so the
whole pipeline is packaged in `scheduled-tasks/cloud-daily-stock-analysis/run.sh`
next to this file. Your job is to start it, wait for it, and write the run
summary from its logs. Do not re-implement the steps by hand.

This replaced the Mac runbook (`../daily-stock-analysis/SKILL.md`) on
**2026-09-10**, the day after the local working copy — venv, price cache,
worktrees and the `~/.claude/scheduled-tasks` symlinks — was deleted. That
runbook is kept for reference and for a future Mac rebuild (`../RECOVERY.md`);
the two must not run on the same day, because both append to `data/snapshots`
and both force-push `pages-live`.

## Execution mode
- **Fully autonomous.** Nobody is watching. Never pause for confirmation.
  Make reasonable choices for anything ambiguous and note them in the summary.
- The only outward actions permitted are the ones `run.sh` performs: a commit
  on `data/snapshots` (today's `results_<date>.json.gz` plus
  `rating_history.json`) and a force-push of the single-commit `pages-live`
  branch. Never push to `main` or any other branch, never open a PR, never
  edit scoring/config files.

## What the script does (so you can read its logs)
`run.sh` is the Mac runbook's steps in order, with a stateless bootstrap:

| step | log | blocking? | notes |
|---|---|---|---|
| preflight | `00-preflight.log` | — | `scripts/market_open.py`; exit 10 = market closed → the script exits 0 immediately and `status.txt` says `SKIPPED market closed`. **That is a successful run**; report the one-line reason and stop. |
| 01-venv | | yes | `.venv` + `pip install -e ".[dev]"` (~1 min) |
| 02-stage-snapshots | | yes | blob-less, checkout-less clone of `data/snapshots`; materialises the newest `SNAPSHOT_HISTORY` (10) snapshots and `rating_history.json` into `output/` so carry-forward, Yesterday's Rating, the rate-change look-back, rating history and gate N/A deltas all work exactly as they did locally |
| 03-prices | | no | full price history for every ticker in the newest prior snapshot + benchmarks (~2,300 tickers, 25–45 min) |
| 04-analyze | | **yes** | `analyze_stock.py --macro --universe us --min-spread 0 --mcap-min 300e6` — **3–6 hours**. SEC companyfacts are re-downloaded every run (the on-disk cache does not survive the container) |
| 05a–05d enrich | | no | FDIC, REIT, XBRL, FDA pipeline — same as the Mac runbook 1b–1e |
| 05e-prices-topup | | no | full history for Phase-2 entrants that only got a Close-only stub during the run |
| 05f-rerender | | yes | `rescore_and_render.py` so the HTML carries every enrichment |
| 06-archive | | **yes** | `archive_snapshot.py` (gzip + SHA-256 round-trip + 80 MiB guard), commit `Snapshot: <date>` on top of the remote tip, push with retries. rc 2 = over the hard guard, not pushed |
| 07a–07c reports | | no | portfolio concentration/drawdown, gate N/A coverage + deltas, trailing-momentum sanity check — **their logs are the body of your summary** |
| 08-publish | | no* | rebuilds `pages-live` (index.html, prices_meta/hist/details/macro sidecars, `px/` and `vol/` shards by manifest) as one fresh commit, force-pushes it, then polls the live URL for today's date. *A publish failure does not fail the analysis (the snapshot is safe); report it and note that re-running only step 8 is possible by hand |

Everything lands under `$REPO/.cloud-run/`: `status.txt` (one `step rc=N seconds=S`
line per step, then `RUNDATE`, `SOFT_FAILURES`, `RESULT ...`) and `logs/<step>.log`.

## Steps for you

### 1. Get the checkout
The session should already contain the repository at `/home/user/stock-analysis-model`
(the Routine's environment). If it does not, clone it:
```
git clone https://github.com/danmcooper-ops/stock-analysis-model.git /home/user/stock-analysis-model
```
Then make sure you are on the latest `main` — the render reads
`templates/report.html` from the working tree, so a stale checkout re-publishes
old UI (the Mac runbook's Step 0.5):
```
cd /home/user/stock-analysis-model && git fetch origin main && git checkout -q main && git reset -q --hard origin/main && git log --oneline -1
```

### 2. Start the script in the background and wait for it
Run it as a **background** Bash command (the Bash tool's `run_in_background`),
so the session is woken when it exits; the run takes **4–8 hours**:
```
cd /home/user/stock-analysis-model && bash scheduled-tasks/cloud-daily-stock-analysis/run.sh > .cloud-run.log 2>&1
```
Do not poll with `sleep` loops. While waiting you may `tail` `.cloud-run/status.txt`
occasionally to confirm progress, but do nothing else to the repo. Never start
a second copy.

API keys come from the environment (`SEC_EMAIL`, `FMP_API_KEY`, `TIINGO_API_KEY`,
`FINNHUB_API_KEY`, `ANTHROPIC_API_KEY`, `FRED_API_KEY`). They are configured on
the cloud environment, not in the repo; if `SEC_EMAIL` is unset the script
falls back to a placeholder contact address and everything else degrades as
documented in the Mac runbook (no `ANTHROPIC_API_KEY` = no macro narrative,
still a successful run). Mention in the summary which keys were absent.

### 3. Read the outcome
When the background command finishes, read `.cloud-run/status.txt` first.
- `SKIPPED market closed` → report "market closed — run skipped" plus the
  printed reason. Done.
- `RESULT OK` → full summary (below).
- `RESULT OK-BUT-PUBLISH-FAILED` → full summary, publish failure called out
  with the tail of `logs/08-publish.log`.
- `RESULT FAILED at ...` → say which step, quote the last ~30 lines of that
  step's log, and state plainly what did **not** happen (no snapshot archived
  / nothing published). Do not retry the whole pipeline; a second 6-hour run
  in the same session rarely helps and can double-post. If only the push in
  06-archive failed (`git push` in the log, the `.json.gz` exists in
  `.cloud-run/snapshots-data/`), retry just that push once:
  `git -C .cloud-run/snapshots-data push https://github.com/danmcooper-ops/stock-analysis-model.git HEAD:refs/heads/data/snapshots`.
  If the push was rejected for lack of credentials, call the
  `mcp__Claude_Code_Remote__add_repo` tool with `owner: danmcooper-ops`,
  `repo: stock-analysis-model`, `access: push` and retry once.

### 4. Write the run summary
Lead with the result line and the run date, then, in this order:
1. **Archive:** the `Snapshot: <date>` commit hash and archive size from
   `logs/06-archive.log` (call out a WARNING past the 50 MiB soft guard).
2. **Gate N/A coverage** — paste the full table from `logs/07b-gate-na-report.log`
   and call out any **⚠ JUMP** (a gate's N/A share rose ≥ 10 points vs the
   prior snapshot: a data source degraded today). **⚠ HIGH** alone is
   baseline; only mention a gate that is *newly* HIGH.
3. **Momentum sanity check** — the rating-bucket table and Spearman r from
   `logs/07c-validate-ratings.log`. It compares today's ratings with the
   *past* 12 months of returns; the correlation is expected to be negative
   (a value model buys laggards) — flag it only if positive and significant
   (r > +0.15, p < 0.05). This is not a measure of accuracy; accuracy is the
   weekly backtest's job.
4. **Portfolio report** — from `logs/07a-portfolio-report.log`: any sector
   > 35% of the BUY/LEAN BUY bucket, any correlated pair (r > 0.85) that is
   not an obvious duplicate (GOOG/GOOGL), any BUY with a 2020 drawdown worse
   than −50%.
5. **Publish** — the live URL and whether it served today's date; the deploy
   workflow otherwise.
6. **Run quality** — the analysis log's closing run-quality summary
   (fabricated/fallback input counts), the number of tickers screened /
   qualifying, any step in `SOFT_FAILURES` with one line on why, missing API
   keys, and total elapsed time.

Keep it factual; the summary is the only record of the run.

## Success criteria
A run the preflight skipped is a success. Otherwise:
- `output/stock_analysis_results_<date>.html` and `output/results_<date>.json` exist
- `results_<date>.json.gz` was committed and pushed to `data/snapshots` (rc 0, under the size guard)
- the gate N/A table and the momentum check are in the summary
- the publish completed, or its failure is reported clearly

## Running it by hand / changing the schedule
The Routine lives in the claude.ai Routines list under the name
**Daily stock analysis (cloud)** (id `trig_017kRuovqnvA3hk31Zd4zBMe`, created
2026-09-10 from a Claude Code session, environment "Default"); fire it out of
schedule from there, or from a session with the
`mcp__Claude_Code_Remote__fire_trigger` tool. Sessions it fires carry no MCP
connector tools, so the `add_repo` fallback above may not exist there; if a
push is refused for credentials, recreate the Routine from the claude.ai
Routines UI with this repository attached as a source. The schedule is
`0 21 * * 1-5` UTC (17:00 New York in summer, 16:00 in winter — always after
the close; cloud cron is UTC-only, so the local hour drifts with DST instead
of ever landing before the bell). To test the script itself without touching
GitHub: `SMOKE=1 FORCE=1 DRY_RUN=1 bash scheduled-tasks/cloud-daily-stock-analysis/run.sh`
runs eight tickers end to end and pushes nothing.

## Notes
- `data/yf_session.py` impersonates `chrome116` here (`YF_IMPERSONATE`,
  exported by the script): the newer Chrome TLS profiles are reset by the
  cloud egress proxy, and a plain client is 429'd by Yahoo.
- `.cloud-run/` is gitignored scratch; the blob-less snapshot clone inside it
  never downloads more than the staged days plus today's upload.
- `01-venv` routes pip through the egress proxy on purpose. The proxy's
  noProxy set contains `pypi.org` and `files.pythonhosted.org`, so pip would
  otherwise reach PyPI directly — and direct egress answers the request but
  never streams the body (a 1 MiB wheel stalled at 0 bytes for 60s), failing
  every install with `ReadTimeoutError`. `pip_no_proxy()` drops just those two
  hosts from `no_proxy`; the localhost/link-local/internal entries stay, and
  when no proxy is configured the list is left alone.
- The weekly backtest is a separate routine and is **not** covered here.
