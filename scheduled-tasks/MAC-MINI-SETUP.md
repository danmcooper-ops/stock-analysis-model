# Moving the daily run and the weekly backtest to a Mac mini

A checklist for taking both cloud Routines off the cloud and running them on
an always-on Mac mini. The nightly analysis (`cloud-daily-stock-analysis/`)
moves to `scripts/run_daily.sh`. The Sunday backtest
(`cloud-weekly-backtest/`) moves to the same `run.sh` it runs today, run
from the Mac. It builds on `RECOVERY.md` (the rebuild steps) and
`daily-stock-analysis/SKILL.md` (the Mac runbook). It does not repeat them.

The gain is that the machine keeps its state. `output/prices/`,
`data/cache/sec_facts/`, `snapshots.duckdb`, the rating history and the
checkpoints stay on disk between runs, so nothing has to be restored, the
Yahoo profile doesn't need pinning, and there's no proxy. The run will not
get much faster. Most of the 12–20 hours is spent waiting on Yahoo's and SEC's
rate limits, not on computing.

**Never run both copies of the same job.** The daily jobs each commit to
`data/snapshots` and force-push `pages-live`. The weekly jobs each commit a
`Weekly backtest: <date>` summary and rewrite the `returns/` sidecars. Each
cloud Routine gets paused in section 7, and not before its Mac replacement has
done a clean dry run. The two jobs can move on different weekends.

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
- [ ] **Weekly:** see section 6.
- [ ] Remove the secrets from the cloud environment only **after** both
      Routines are paused (section 7). The weekly Routine still needs
      `TIINGO_API_KEY` until then.

## 6. The weekly backtest

**Use `cloud-weekly-backtest/run.sh`, not the old Mac job.** The Mac's
`weekly-backtest/weekly_backtest.sh` and its plist are older than the cloud
routine, and they:
- run `calibrate` every week, which the current runbook forbids below
  `MIN_EFFECTIVE_N`;
- skip the readiness census, the Tiingo delisting backfill, the
  week-over-week `compare`, and the commit to `data/snapshots` (no summary
  from it reached the branch after 2026-07-13);
- assume the iCloud-era `output/` symlink.

Retire them: don't load the plist, and mark them superseded in this
directory's `README.md`.

`run.sh` is already mostly portable. It takes its checkout from
`STOCK_MODEL_REPO`, its scratch directory from `STOCK_MODEL_WORK`, stages
its own corpus from a blob-less clone, and pushes with plumbing that retries
on a moved branch tip. On the Mac it needs:

- [ ] **A Python ≥ 3.11 first on `PATH`.** Its `01-venv` step runs
      `python3 -m venv "$REPO/.venv"`. Under launchd or a bare shell,
      `python3` can be macOS's `/usr/bin/python3`, which is too old for the
      pinned dependencies. Put the python.org or Homebrew build first on
      `PATH`. The script builds its own `.venv` in the repo, separate from the
      daily's `~/.venvs/stock-model`, and that's fine.
- [ ] **A CA bundle.** The script only sets `SSL_CERT_FILE` when it finds the
      cloud's `/root/.ccr/ca-bundle.crt`. On the Mac, export it from `certifi`
      as `run_daily.sh` does. The python.org build fails HTTPS verification
      without it.
- [ ] **`YF_IMPERSONATE=chrome`.** The script defaults to `chrome116`, which
      was only needed for the cloud proxy.
- [ ] **`TIINGO_API_KEY` in the environment.** `run.sh` does not read `.env`.
      Without the key, step 04b is skipped and delisted names go unmeasured.
      Source `.env` in the wrapper below.
- [ ] **Keep the price cache between Sundays.** `stage_corpus` runs
      `rm -rf "$SNAP" "$OUT"`, and the prices live in `$OUT/prices`. The
      container never noticed, but on the Mac it means a cold 1–2 hour
      download every week. Change the script to keep `$OUT/prices` and clear
      everything else. `returns/` and `price_backfill/` are re-staged from the
      branch, which stays their record. Download
      freshness is judged from parquet content, so a kept file that is
      behind is re-fetched, never trusted. Do the same in the cloud copy so
      the two stay one script.
- [ ] **A small wrapper**, e.g. `scheduled-tasks/weekly-backtest/mac_run.sh`:
      source `.env`, set the three variables above, set
      `STOCK_MODEL_WORK="$HOME/Library/Application Support/StockModel/backtest"`
      (outside the repo, off iCloud), then `exec` `run.sh`. Commit it here.
- [ ] **Git push access** to `data/snapshots` from that user. This is the
      same credential as the daily. `run.sh` pushes to the HTTPS URL in
      `PUSH_REMOTE`, so a token in the macOS keychain credential helper works.
      For SSH, set `PUSH_REMOTE`/`CLONE_REMOTE` to the `git@github.com:` form.
- [ ] **Schedule:** Sundays, as now. Friday's daily run can last into
      Saturday afternoon, so any time Sunday is clear. Use the same mechanism
      you chose for the daily. A Claude Code scheduled task gets you the written
      summary described in `cloud-weekly-backtest/SKILL.md` §3–4 (point it at
      the Mac paths). A launchd agent can reuse `com.stockmodel.weekly.plist`
      with its program switched to the wrapper and `PATH` set in an
      `EnvironmentVariables` dict, since launchd doesn't source your shell
      profile.
- [ ] **Disk:** the scratch directory holds the staged corpus (~1.5 GB, from
      2026-07-06 on and growing about 17 MB a night) plus the price parquets.
      The clone is re-fetched from GitHub each Sunday, about 1.5 GB of
      download.
- [ ] **Smoke test** before the cut-over. It pushes nothing:
      `SMOKE=1 bash scheduled-tasks/weekly-backtest/mac_run.sh`, then
      `cat "$STOCK_MODEL_WORK/status.txt"`.

## 7. Cut over

- [ ] Dry run on the Mac while the cloud Routine is still live:
      `scripts/run_daily.sh --dry-run`. It prints the plan and runs nothing.
- [ ] A short real run that pushes nothing: the S&P 500 + Dow universe from
      `RECOVERY.md` → **Check it worked**, a few hundred tickers.
- [ ] **Pause the cloud Routine** ("Daily stock analysis (cloud)") in the
      claude.ai Routines UI. Check that it shows as disabled.
- [ ] Enable the Mac's daily task.
- [ ] Before a Sunday, **pause the weekly cloud Routine** too, then enable the
      Mac's weekly task. Swap the DORMANT banners between
      `cloud-weekly-backtest/SKILL.md` and `weekly-backtest/SKILL.md`, and
      point the latter at the wrapper.
- [ ] After the first night, check:
  - [ ] `output/run_summary_<date>.json` has `status: ok`
  - [ ] `results_<date>.json.gz` is on `data/snapshots`, with no second
        `Snapshot: <date>` commit from the cloud
  - [ ] the Pages site shows the new date
  - [ ] a `Portfolio alerts — <date>` issue appears when there are alerts
  - [ ] the next Sunday's backtest picks up the Mac's snapshots
  - [ ] Phase 1's `facts_stats` show mostly disk hits (the SEC cache is doing
        its job) and the `prices` step takes minutes, not an hour
- [ ] After the first Mac Sunday, check:
  - [ ] `status.txt` ends `RESULT OK`, with exactly one
        `Weekly backtest: <date>` commit on `data/snapshots` (none from the
        cloud)
  - [ ] `07-compare` found last week's cloud summary and reports no new
        `REGRESSION:` lines. A different machine is not a model change, so a
        `NOTICE` about the scoring model means something else changed.
  - [ ] `04b-backfill` ran (the Tiingo key reached it)
  - [ ] on the second Mac Sunday, `04-prices` takes minutes, which shows the
        cache survived

## Going back to the cloud

Pause the Mac task first, then resume the matching cloud Routine. Do this
per job. If section 4's
branch commits and the Supabase cache saves were kept up, the cloud run
starts warm without further work.
