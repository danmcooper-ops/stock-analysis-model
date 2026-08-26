---
name: daily-stock-analysis
description: Run full stock analysis and publish updated report to GitHub Pages
---

You are running the end-of-day stock analysis routine. Execute the following steps in order. Stop and report an error if any step fails.

## Execution mode
- **Always run fully autonomously (auto mode).** Do not pause for confirmation or ask clarifying questions — this is an unattended scheduled run. Make reasonable choices for any ambiguity and note them in the run summary. Only "write" actions explicitly described in the steps below (committing/pushing snapshots, publishing the report) are permitted; do not take other outward-facing or destructive actions.
- **Always run on the latest available model.** Use the most capable current Claude model for this routine; do not pin to or fall back to an older model.

## Paths
- **Main repo:** `$HOME/Desktop/Workspace Folder`
- **Pages worktree:** `$HOME/Desktop/Workspace Folder/.claude/worktrees/pages-live` (branch `pages-live`; used only by the publish routine in Step 8)
- **Snapshots worktree:** `$HOME/Desktop/Workspace Folder/.claude/worktrees/snapshots-data`
- **Python:** `$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python`
- **SSL fix:** set `SSL_CERT_FILE` to the output of `.venv/bin/python -m certifi` before running any Python script
- **Historical snapshots (git branch `data/snapshots`):** `$HOME/Desktop/Workspace Folder/.claude/worktrees/snapshots-data/results_*.json`

## IMPORTANT: Command format
All Python script invocations **must be sent as a single-line semicolon-separated Bash command** (not multi-line). This is required for permission matching to work. Use the exact format shown in each step below.

## Steps

### 0. Refresh the price cache
The `output/prices` parquets never refresh themselves; the analysis, the report's px/vol chart shards, and the validation steps all read them, so they must be brought current at the start of each run. Run as a **single Bash call** (all on one line, semicolons not newlines):
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/download_prices.py --output-dir output/prices --max-age-days 2 --tickers $(ls output/prices/*.parquet | sed 's|.*/||;s|\.parquet||') SPY QQQ IWM DIA
```
Refreshes every already-cached ticker whose last bar is older than 2 days, plus the four benchmark indices explicitly (SPY/QQQ/IWM/DIA feed the report's index-comparison lines; the weekly backtest job only refreshes SPY). Idempotent — current files are skipped, so the day after a full refresh this is nearly a no-op. **Typical runtime: 15–40 min on a normal weekday; up to ~60 min after weekends/gaps.** Do not run it concurrently with Step 1 — Step 1 reads these files.

A non-zero exit code (or partial ticker failures — Yahoo outages, delisted names) should be reported but does **not** block the remaining steps: the analysis still works on slightly-stale bars, which was the status quo before this step existed. Individual delisted-ticker errors in the output are routine noise, not failures.

### 0.5 Fast-forward main so the render uses merged template work
PRs merged on GitHub are invisible to this run until the local checkout is updated: the pipeline reads `templates/report.html` from the working tree at render time, so a stale `main` silently re-publishes old UI. (Bitten 2026-08-25: local `main` was 6 commits behind `origin/main` and the nightly render reverted the merged Macro Outlook nav order on the live site.) Run as a **single Bash call**:
```
cd "$HOME/Desktop/Workspace Folder"; git fetch origin main; git pull --ff-only origin main
```
If the pull fails (non-fast-forward divergence, or uncommitted changes that conflict), do **not** force-update or discard anything: report the divergence prominently in the run summary and continue the run on the current checkout. A stale render is recoverable later with `rescore_and_render.py` + a republish; a forced update can destroy local work.

### 1. Run the analysis
Run as a **single Bash call** (all on one line, semicolons not newlines):
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/analyze_stock.py --macro --prices-dir output/prices --universe us --min-spread 0 --mcap-min 300e6
```
This expands the ticker universe from ~500 S&P/Dow stocks to all US-listed equities (~7,000–10,000 tickers from SEC EDGAR), then applies two Phase-1 filters before the expensive Phase 2 deep analysis: (1) market cap ≥ $300M (drops micro-caps and shells that can't be meaningfully valued) and (2) ROIC > WACC (positive economic spread — the business creates value). **Expected runtime: 3–6 hours.** The SEC listings are cached locally for 7 days (`data/cache/us_listings.csv`) so Phase 1 startup is fast on subsequent runs.

This produces `output/stock_analysis_results_YYYY-MM-DD.html` and `output/results_YYYY-MM-DD.json` (where YYYY-MM-DD is today's date). Confirm both files exist before continuing.

If the script exits non-zero, do not proceed with any further steps.

### 1b. Enrich Financial Services records with FDIC call-report data
Run as a **single Bash call** (all on one line, semicolons not newlines):
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/enrich_fdic.py "output/results_$(date +%Y-%m-%d).json"
```
Joins NIM / Efficiency Ratio / CET1 / NPL / Deposit Beta from the FDIC BankFind Suite API into each mapped bank's record (writes `nim`, `efficiency_ratio`, `cet1_ratio`, `npl_ratio`, `deposit_beta`, `fdic_cert`, `fdic_repdte` in place). `deposit_beta` = Δ(cost of deposits) / Δ(fed funds) across the 2021→23 hiking cycle, from two historical FDIC quarters (EDEP/DEP). Mapping lives in `data/ticker_fdic_map.py` (~45 US-chartered banks); responses cached for 30 days under `data/cache/fdic/`. The script is idempotent — strips prior enrichment before running, and a staleness guard rejects records older than 2024 so any wrong CERT fails closed.

A non-zero exit code should be reported but does **not** block the remaining steps — the analysis itself is still valid even if FDIC enrichment fails (e.g. API outage). The Financials sector banner will simply show `—` for the four FDIC-backed KPIs.

### 1c. Enrich Real Estate records with FFO Growth + AFFO Margin proxies
Run as a **single Bash call** (all on one line, semicolons not newlines):
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/enrich_reit.py "output/results_$(date +%Y-%m-%d).json"
```
Computes two REIT-specific proxies from each Real Estate stock's existing `edgar_history` (no network calls):
- `ffo_growth_5y` — 5-yr CAGR of operating cash flow (proxy for FFO growth)
- `affo_margin` — (CFO − capex) / revenue, latest year (proxy for AFFO margin)

The NAREIT FFO/AFFO definitions aren't in standardized XBRL; these CFO-based proxies cover ~85% of the REIT universe and match the right qualitative ranking (storage / data-center REITs land in the 50%+ AFFO band; healthcare / hotels at 15-25%). Idempotent.

A non-zero exit code should be reported but does **not** block the remaining steps. Pure local computation — failure is unlikely unless the JSON is malformed.

### 1d. Enrich sector-specific KPIs via SEC XBRL (Phases 3–6)
Covers nine sectors in a single pass: Technology / Healthcare / Communication Services / Industrials / Consumer Cyclical / Consumer Defensive / Energy / Utilities / Basic Materials. Run as a **single Bash call** (all on one line, semicolons not newlines):
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/enrich_xbrl.py "output/results_$(date +%Y-%m-%d).json"
```
For every target-sector stock with a known SEC CIK, fetches the companyfacts blob once and derives:

**Phase 3 — Tech / Healthcare / Comm:**
- `rd_intensity_xbrl` — R&D / Revenue (XBRL is more reliable than yfinance income statement)
- `sbc_pct_rev_xbrl` — Stock-based comp / Revenue
- `fcf_margin_ex_sbc` — (CFO − Capex − SBC) / Revenue
- `net_cash_to_mcap` — (Cash + ST Inv − Debt) / Market Cap
- `deferred_rev_growth` — YoY change in contract-with-customer liability

**Phase 4 — Industrials:**
- `capex_intensity` — abs(Capex) / Revenue (pure local from edgar_history)
- `backlog_to_revenue` — RemainingPerformanceObligation / Revenue

**Phase 5 — Consumer Cyclical / Defensive:**
- `inventory_days` — Inventory / (COGS / 365)
- `working_capital_days` — Inventory Days + AR Days − AP Days (Cash Conversion Cycle)
- `brand_spend_pct_rev` — AdvertisingExpense / Revenue

**Phase 6 — Energy / Utilities / Materials:**
- `capex_to_dd_ratio` — abs(Capex) / D&A (universal capital-reinvestment discipline ratio; for E&P filers, D&A includes depletion of reserves)

**Phase 7 — cross-sector quick wins + Real Estate + insurers** (Real Estate and insurer-only Financial Services records were added to the target set; banks in Financial Services are skipped here since they're enriched via FDIC):
- `rule_of_40` — 5yr revenue CAGR + FCF-ex-SBC margin, in points (Technology; pure local)
- `book_to_bill_proxy` — (ΔBacklog + Revenue) / Revenue from the RPO series (Industrials; pure local)
- `brand_spend_trend` — annualized change in AdvertisingExpense / Revenue (Consumer Cyclical/Defensive; pure local)
- `debt_maturity_wall_yrs` — principal-weighted average years to maturity from XBRL `LongTermDebtMaturitiesRepaymentsOfPrincipal*` buckets (Real Estate)
- `combined_ratio` / `float_cost` — insurers: `1 − UnderwritingIncomeLoss / PremiumsEarnedNet`, and combined − 1 (Financial Services insurers)

Bounded by SEC's 10 req/sec rate limit; ~250 seconds wall clock for ~1,500 mapped tickers on a cold in-memory cache. Idempotent — strips prior enrichment first.

A non-zero exit code should be reported but does **not** block the remaining steps. Stocks for which the SEC XBRL fetch fails just show "—" for the affected KPIs in their sector banners.

### 1e. Enrich Healthcare drug/biotech records with FDA pipeline depth
Run as a **single Bash call** (all on one line, semicolons not newlines):
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/enrich_pipeline.py "output/results_$(date +%Y-%m-%d).json"
```
Counts each drug/biotech company's active sponsored interventional trials on the ClinicalTrials.gov API v2 and writes `fda_pipeline_count`. Scoped to drug/biotech industries (devices/payers/services are skipped so they don't distort the sector median). Sponsor name comes from `data/ticker_sponsor_map.py` when mapped, else the cleaned company name; the lookup matches on the lead-sponsor field so unrelated companies don't collide. Responses cached for 30 days under `data/cache/clinicaltrials/`. Idempotent — strips prior enrichment first; API/network failures degrade to "—" rather than raising.

A non-zero exit code should be reported but does **not** block the remaining steps. The Healthcare banner simply shows `—` for FDA Pipeline Count if the enrichment fails.

### 1f. Re-render the HTML report so banners pick up the enrichment
Must run **after** 1b, 1c, 1d, and 1e so the final HTML reflects every enriched field. Run as a **single Bash call**:
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/rescore_and_render.py "output/results_$(date +%Y-%m-%d).json"
```
This overwrites `output/stock_analysis_results_YYYY-MM-DD.html` with a render that includes every Phase 1–4 enrichment. If any of the enrichment steps failed (no fields populated), this step is still safe — the affected banners just fall back to "—".

### 2. Commit today's snapshot to the data/snapshots branch
Run each as a **separate** Bash call (single line each):
```
cp "output/results_$(date +%Y-%m-%d).json" "$HOME/Desktop/Workspace Folder/.claude/worktrees/snapshots-data/"
```
```
git -C "$HOME/Desktop/Workspace Folder/.claude/worktrees/snapshots-data" add "results_$(date +%Y-%m-%d).json"
```
```
git -C "$HOME/Desktop/Workspace Folder/.claude/worktrees/snapshots-data" commit -m "Snapshot: $(date +%Y-%m-%d)"
```
```
git -C "$HOME/Desktop/Workspace Folder/.claude/worktrees/snapshots-data" push origin data/snapshots
```
This persists the snapshot to GitHub so it is never lost if the local worktree is deleted. A non-zero exit code should be reported but does **not** block the remaining steps.

### 3. Run portfolio concentration and drawdown report
Run as a **single Bash call** (all on one line, semicolons not newlines):
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/portfolio_report.py --results-dir output/ --prices-dir output/prices
```
This saves `output/portfolio_report_YYYY-MM-DD.txt` automatically in addition to printing to the console. Print the full output in the run summary. Flag: any sector > 35% of the BUY/LEAN BUY bucket (concentration risk), any highly correlated pair (r > 0.85) that are not obvious duplicates (e.g. GOOG/GOOGL), and any BUY-rated stock with a 2020 drawdown worse than -50%. A non-zero exit code should be reported but does **not** block remaining steps.

### 4. Report gate N/A coverage
Run as a **single Bash call** (all on one line, semicolons not newlines):
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/gate_na_report.py "output/results_$(date +%Y-%m-%d).json"
```
Prints, for each of the 26 model gate columns, how many records have an N/A raw value (`_gate_*` is null) and the percentage of the universe affected, plus the day-over-day delta vs the prior snapshot. **Print the full table in the run summary.** Flags to act on:
- **⚠ JUMP** — a gate's N/A share rose ≥ 10 points vs the prior snapshot. This is the signal that a data source degraded *today* (e.g. the 2026-07-22 run silently dropped ~100 tickers to fetch timeouts). Call it out prominently in the run summary.
- **⚠ HIGH** — a gate is ≥ 40% N/A. Several gates are structurally high (sector-inapplicable metrics, short-history requirements like 10Y Rev CAGR), so HIGH alone is baseline, not a problem — only flag it in the summary if a gate is newly HIGH or its set of HIGH gates changed vs recent runs.

Must run after 1f (so it measures the final enriched/rescored snapshot). A non-zero exit code should be reported but does **not** block the remaining steps.

### 5. Validate ratings against trailing price returns
Run as a **single Bash call** (all on one line, semicolons not newlines):
```
PYTHON="$HOME/Desktop/Workspace Folder/.claude/worktrees/phase-1-api/.venv/bin/python"; SSL_CERT_FILE=$("$PYTHON" -m certifi); export SSL_CERT_FILE; cd "$HOME/Desktop/Workspace Folder"; "$PYTHON" scripts/validate_ratings.py --snapshot "output/results_$(date +%Y-%m-%d).json" --prices-dir output/prices
```
This checks whether today's BUY/LEAN BUY/HOLD/PASS ratings correlate with the past 12 months of actual price returns. Print the full output in the run summary. Key things to flag:
- If BUY-rated stocks had significantly *higher* trailing returns than HOLD/PASS, the model may be chasing momentum rather than identifying value — worth reviewing
- The Spearman correlation between composite score and trailing return is expected to be **negative** (value model buys laggards); flag it if it turns positive and significant (r > +0.15, p < 0.05)
A non-zero exit code should be reported but does **not** block the remaining steps.

### 8. Publish the report
Read `$HOME/.claude/scheduled-tasks/publish-stock-report/SKILL.md` and execute the steps in that file. That routine copies seven artifacts into the `pages-live` worktree — `output/stock_analysis_results_RUNDATE.html` → `docs/index.html`, plus `prices_meta.json`, `hist.json`, `details.json`, `macro.json` (optional — present only when the run reached FRED), and the `vol/` and `px/` shard directories (the HTML lazy-loads all of those at runtime, so everything must ship together; the dense `prices.json` was retired 2026-08-11 in favor of per-ticker `px/` shards) — then amends that branch's single commit and force-pushes it. **`main` is not touched by this routine.** The old sweet-gauss copy→merge→fast-forward flow is retired.

Note that the publish routine uses the **run-START date** (RUNDATE), not `$(date)` — if the 3–6 hour analysis crossed midnight, `$(date)` names the wrong file.

This is run as the final step of the analysis routine, but the publish routine is a **separate failure surface**: if it fails, the analysis itself is still considered successful (today's JSON and HTML exist locally and the snapshot has been pushed). Report the publish failure but do not retroactively fail the analysis run. The publish routine can be re-invoked manually to retry without re-running analysis.

## Success criteria
- `output/stock_analysis_results_YYYY-MM-DD.html` was created today
- `output/results_YYYY-MM-DD.json` was committed to `data/snapshots` and pushed
- Gate N/A coverage table (per-gate N/A % + deltas) is included in the run summary, with any ⚠ JUMP flags called out
- Trailing validation output (rating buckets + Spearman r) is included in the run summary
- The publish routine completed successfully (or its failure was reported clearly)

## Notes
- The script reads API keys from `.env` in the main repo root — do not commit that file
- SSL certificate errors on macOS are fixed by setting `SSL_CERT_FILE` to certifi's bundle (see Step 1)
- Use `git -C <path>` for all git commands so you don't need to change directories
- The `phase-1-api` venv has all required packages (yfinance, pandas, openpyxl, jinja2, lxml, certifi)
- If the `pages-live` or `snapshots-data` worktrees are missing, recreate them:
  - `git worktree add .claude/worktrees/pages-live pages-live`
  - `git worktree add .claude/worktrees/snapshots-data data/snapshots`
