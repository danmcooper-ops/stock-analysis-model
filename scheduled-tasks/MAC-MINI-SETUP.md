# Moving the daily run to a Mac mini

A checklist for taking the nightly analysis off the cloud Routine
(`cloud-daily-stock-analysis/`) and running it on an always-on Mac mini with
`scripts/run_daily.sh`. It builds on `RECOVERY.md` (the rebuild steps) and
`daily-stock-analysis/SKILL.md` (the Mac runbook). It does not repeat them.

The gain is that the machine keeps its state. `output/prices/`,
`data/cache/sec_facts/`, `snapshots.duckdb`, the rating history and the
checkpoints stay on disk between runs, so nothing has to be restored, the
Yahoo profile doesn't need pinning, and there's no proxy. The run will not
get much faster. Most of the 12–20 hours is spent waiting on Yahoo's and SEC's
rate limits, not on computing.

**Never run both.** The cloud Routine and the Mac routine each commit to
`data/snapshots` and force-push `pages-live`. The cloud Routine gets paused in
section 6, and not before the Mac has done a clean dry run.

---

## 1. Hardware and macOS

- [ ] Any M-series Mac mini. 16 GB of memory is enough (Phase 1 caps
      in-flight companyfacts at ~370 MB) and 24 GB gives headroom. 512 GB of
      storage covers the price parquets, the DuckDB store, and the full
      `data/snapshots` worktree for years.
- [ ] Wired Ethernet, not Wi-Fi.
- [ ] **Time zone America/New_York** (System Settings → General → Date & Time).
      `run_daily.sh` takes RUNDATE from local `date +%F`, and the snapshot name
      and the market-open gate both depend on it. The cloud script had to force
      `TZ=America/New_York` for the same reason.
- [ ] Energy settings for an unattended machine:
      ```bash
      sudo pmset -a sleep 0 disksleep 0 autorestart 1 womp 1
      pmset -g    # confirm: sleep 0, autorestart 1
      ```
      `autorestart` is "Start up automatically after a power failure".
      `run_daily.sh` also runs under `caffeinate`.
- [ ] **Decide how the Mac comes back after a power cut.** With FileVault on,
      a restart stops at the unlock screen, so no user session runs and no
      scheduled task fires until someone logs in. Either turn on automatic
      login (this needs FileVault off), or accept that a power cut means a
      missed night until you log in. A missed weekday can be re-run the
      next morning with `scripts/run_daily.sh --from analyze --date YYYY-MM-DD`.
- [ ] A small UPS is worth having. An interrupted analysis resumes from
      `output/.checkpoint/`, but only after someone restarts it.
- [ ] Turn off automatic macOS updates that restart the machine
      (System Settings → General → Software Update → Automatic updates: turn off
      "Install macOS updates"). Update by hand on a weekend.
- [ ] Keep the repo **off iCloud Drive** (not under `~/Desktop` or
      `~/Documents`). iCloud once evicted snapshot files and filled `output/vol/`
      with conflict copies (`publish-stock-report/SKILL.md`).

## 2. Checkout, venv and keys

Follow `RECOVERY.md` → **Rebuild**, steps 1–5, as written:

- [ ] Clone to `~/Projects/Workspace Folder`. `scripts/run_daily.sh` hardcodes
      `REPO="/Users/danmcooper/Projects/Workspace Folder"`, so a different
      user name or path means editing that line and the three runbooks.
- [ ] Create both worktrees, `pages-live` and `data/snapshots`
      (`.claude/worktrees/…`).
- [ ] Build the venv at `~/.venvs/stock-model`, outside the repo. Any Python
      ≥ 3.11 works. `RECOVERY.md` uses the python.org 3.14 build.
      `certifi` must import.
- [ ] Write `.env` from your password manager: `SEC_EMAIL` is required, plus
      `FMP_API_KEY`, `TIINGO_API_KEY`, `FINNHUB_API_KEY`, `FRED_API_KEY` and
      `MACRO_ANTHROPIC_API_KEY`. Add `SUPABASE_URL` and
      `SUPABASE_SERVICE_ROLE_KEY` if you want section 3's cache restore or the
      database steps. `chmod 600 .env`.
- [ ] Git can push without prompting, to both `data/snapshots` and
      `pages-live`. Use an SSH key or a credential-helper token; launchd and
      Claude Code sessions can't answer a password prompt.
      Test: `git -C .claude/worktrees/snapshots-data push --dry-run origin data/snapshots`.
- [ ] Don't set `YF_IMPERSONATE`. The default `chrome` profile works outside
      the cloud proxy. `chrome116` was only ever for the cloud proxy.

## 3. Seed the local state

The cloud run keeps its state in Supabase Storage and on `data/snapshots`.
Pull it down once instead of rebuilding it cold. Run everything below from
the repo root, with the venv's Python as `$PYTHON`.

- [ ] **Price parquets.** If the Supabase secrets are in the environment, this
      takes minutes instead of hours:
      ```bash
      set -a; . ./.env; set +a
      "$PYTHON" scripts/price_cache.py restore --prices-dir output/prices
      ```
      Without the secrets, use `RECOVERY.md` step 6 (a full `download_prices.py
      --universe us`, which takes hours but can be resumed). Either way, do it
      **before** the first run. The nightly `prices` step only refreshes
      parquets that already exist.
- [ ] **SEC companyfacts cache, with its watermark:**
      `"$PYTHON" scripts/sec_cache.py restore`. It's optional, since a cold
      cache only costs one slower run, but it brings `_state.json`, so filing
      sweeps resume from the right day.
- [ ] **Recent snapshots**, for carry-forward, Yesterday's Rating and gate
      N/A deltas. Copy the newest 10 into `output/` along with the blobs they
      reference:
      ```bash
      SNAP=.claude/worktrees/snapshots-data
      ls "$SNAP"/results_*.json.gz | sort | tail -10 | xargs -I{} cp {} output/
      rsync -a "$SNAP/blobs/" output/blobs/
      ```
- [ ] **The state files the cloud run kept on the branch:**
      ```bash
      cp "$SNAP/rating_history.json" output/
      cp "$SNAP/portfolio_nav.json"  output/
      mkdir -p data/cache && cp "$SNAP/screen_skip.json" data/cache/
      ```
      Without `screen_skip.json` the first Phase-1 screen runs cold, which adds
      about 1h20m (4h15m instead of 2h54m).
- [ ] **DuckDB index:** `RECOVERY.md` step 7
      (`ingest_snapshots.py --results-dir .claude/worktrees/snapshots-data`).
- [ ] Run `RECOVERY.md` → **Check it worked**. The parquet count should be
      in the thousands, not 4.

## 4. Bring `run_daily.sh` up to date with the cloud script

`run_daily.sh` went dormant on 2026-09-10. Everything added to the pipeline
since then went into `cloud-daily-stock-analysis/run.sh` only. Before the Mac
takes over, port these steps or decide to drop them:

- [ ] **Price top-up after the analysis** (cloud `05e-prices-topup`): full
      history for tickers that entered Phase 2 with only a Close-only stub.
      Without it their charts and the stopped-trading rule stay thin until
      the next night's `prices` step.
- [ ] **Portfolio alerts digest to `data/snapshots`.** The GitHub workflow
      `.github/workflows/portfolio-alerts.yml` reads `portfolio_alerts.json`
      from that branch. `run_daily.sh` writes the file to `output/` but never
      commits it, so the alert issues would stop.
- [ ] **`rating_history.json`, `portfolio_nav.json` and `screen_skip.json` on
      `data/snapshots`.** They're no longer needed to carry state (the Mac
      keeps them locally), but committing them keeps the branch complete for
      the weekly backtest, `RECOVERY.md` and a later move back to the cloud.
      Keep the cloud script's guards: skipped on smoke runs, and
      `screen_skip.json` only when it's over 10 KB.
- [ ] **Supabase steps**, if you use them: `06a-db-publish`
      (`scripts/db_publish.py`, set `DB_DEFER_PUBLISH=1` for the run),
      `07e-db-check` (`scripts/db_night_check.py record`, which the
      `DB_CUTOVER_STREAK` needs every night), and `SNAPSHOT_STORE_BACKEND`
      (the Mac can simply stay on the local `duckdb` store). The Mac can reach
      Postgres directly over TCP, which the cloud container couldn't.
- [ ] **Cloudflare deploy** (`08b-publish-cloudflare`), if the Cloudflare
      Pages site is live. It needs the `CLOUDFLARE_*` / `CF_ACCESS_*` variables
      in `.env` and wrangler (Node) installed.
- [ ] **Optional:** keep saving caches to Supabase (`price_cache.py save`,
      `sec_cache.py save`) as an off-site backup, so a dead Mac or a move back
      to the cloud starts warm.
- [ ] Remove the **DORMANT** banner from `daily-stock-analysis/SKILL.md` and add
      one to `cloud-daily-stock-analysis/SKILL.md`. Update the task table in
      this directory's `README.md` in the same commit.

## 5. Schedule it

- [ ] **Daily:** the existing arrangement is a Claude Code scheduled task,
      symlinked into this directory (`RECOVERY.md` step 4), so Claude writes
      the run summary. This needs the Claude app running and signed in under
      the logged-in user. Schedule it for **weekdays 17:00 New York**, the same
      slot as the cloud Routine (21:00 UTC in summer). A run that goes past
      midnight keeps its start date.
      Alternative: a launchd agent that runs `scripts/run_daily.sh` directly.
      It's more robust, but produces no written summary. The summary is still
      in `output/run_summary_<date>.json`.
- [ ] **Weekly backtest: leave it on the cloud Routine.**
      `cloud-weekly-backtest` reads only `data/snapshots` and commits only a
      summary on Sundays, so it doesn't compete with the Mac's weekday run. The
      Mac's `weekly-backtest/` job is older: it runs calibration and assumes
      the iCloud-era `output/` symlink. Don't load its plist.
- [ ] Remove the `.env` values and secrets you no longer need from the cloud
      environment only **after** section 6.

## 6. Cut over

- [ ] Dry run on the Mac while the cloud Routine is still live:
      `scripts/run_daily.sh --dry-run`. It prints the plan and runs nothing.
- [ ] A short real run that pushes nothing: the S&P 500 + Dow universe from
      `RECOVERY.md` → **Check it worked**, a few hundred tickers.
- [ ] **Pause the cloud Routine** ("Daily stock analysis (cloud)") in the
      claude.ai Routines UI. Check that it shows as disabled.
- [ ] Enable the Mac's scheduled task.
- [ ] After the first night, check:
  - [ ] `output/run_summary_<date>.json` has `status: ok`
  - [ ] `results_<date>.json.gz` is on `data/snapshots`, with no second
        `Snapshot: <date>` commit from the cloud
  - [ ] the Pages site shows the new date
  - [ ] a `Portfolio alerts — <date>` issue appears when there are alerts
  - [ ] the next Sunday's cloud backtest picks up the Mac's snapshots
  - [ ] Phase 1's `facts_stats` show mostly disk hits (the SEC cache is doing
        its job) and the `prices` step takes minutes, not an hour

## Going back to the cloud

Pause the Mac task first, then resume the cloud Routine. If section 4's
branch commits and the Supabase cache saves were kept up, the cloud run
starts warm without further work.
