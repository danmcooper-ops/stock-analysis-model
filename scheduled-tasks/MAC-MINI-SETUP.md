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

**Two scripts do most of this** (`mac-mini/`):
- `pack_old_mac.sh` runs on the old Mac. It inventories it and packs what
  has to move (section 0).
- `bootstrap_mini.sh` runs on the new one. It does sections 2 and 3 and can
  be re-run. `--check` reports what is left without changing anything.

---

## 0. Moving from the old Mac (the MacBook Air)

Almost nothing needs to move. The code, every archived snapshot and the
state files are on GitHub. The price and companyfacts caches the cloud
Routine has saved every night since 2026-09-10 are in Supabase Storage, and
they are newer than anything the Air holds. Only three things can exist
nowhere else:
- **`.env`**, the API keys;
- **work not on GitHub**: uncommitted changes, unpushed commits, stashes;
- **portfolio edits made in the report**, which stay in that browser's
  localStorage until exported.

Don't use Migration Assistant for this. It would copy the Air's whole
account, including the old venv, stale caches, the Claude scheduled tasks
and launchd jobs, onto a machine meant to run one job cleanly.

On the **Air**, in Terminal, from any checkout of the repo that has this
branch, or by downloading the script alone:
- [ ] `bash scheduled-tasks/mac-mini/pack_old_mac.sh --inventory-only` and read
      the report. It lists every checkout it finds, the Trash included, with
      its git state, `.env` key names (never values), caches, and the Air's
      scheduled tasks and launchd jobs. macOS may ask to let Terminal read
      Desktop/Documents: allow it, or the search misses checkouts there.
- [ ] Push anything it flags as **not on GitHub**.
- [ ] Export portfolio edits: open the report in the browser you use,
      Portfolios → Manage → Export. Bring the file across.
- [ ] If `.env` isn't found (the checkout was deleted on 2026-09-09), check
      `~/.Trash` and your password manager. The cloud Routine's environment
      settings (claude.ai, the environment menu → Edit) at least list which
      keys it uses. Re-issue any key you can't recover.
- [ ] Pack: `bash scheduled-tasks/mac-mini/pack_old_mac.sh`. It writes
      `~/StockModelTransfer`, holding `.env` (mode 600) and the inventory.
      Add `--with-sec-cache` / `--with-prices` only if you won't give the
      mini the Supabase keys. Even then the Air's prices are weeks old and get
      re-downloaded in full; they only carry the ticker list.
- [ ] Move it. The simplest way is over the network: on the mini, System
      Settings → General → Sharing → **Remote Login** on, then on the Air
      re-run with `--to <you>@<mini-name>.local`. Or use AirDrop or a USB
      drive. Never put the folder in iCloud Drive; the script refuses to
      write there.
- [ ] Nothing to disable yet. The Air's scheduled tasks and launchd job point
      at the checkout deleted on 2026-09-09, and the runbooks are DORMANT
      anyway. Remove them before you retire the Air; the script prints the
      commands.

On the **mini**:
- [ ] Sign in with your Apple ID, set the time zone, energy settings and
      FileVault choice (section 1). Install the Xcode command-line tools
      (`xcode-select --install`), a Python ≥ 3.11 (python.org or
      `brew install python@3.13`), and log git in to GitHub
      (`brew install gh && gh auth login`).
- [ ] Get `bootstrap_mini.sh` onto the mini. It's in the pack's repo, or
      clone first: `git clone https://github.com/danmcooper-ops/stock-analysis-model.git "$HOME/Projects/Workspace Folder"`.
      Then run
      `bash "$HOME/Projects/Workspace Folder/scheduled-tasks/mac-mini/bootstrap_mini.sh" --transfer ~/StockModelTransfer`.
      It works through sections 2 and 3. Fix what it reports and re-run until
      the summary reads `0 to fix`.
- [ ] Import the portfolio export:
      `~/.venvs/stock-model/bin/python scripts/portfolios.py import <file>`.
- [ ] Delete `~/StockModelTransfer` on both Macs once `.env` is installed.

---

## 1. Hardware and macOS

- [ ] Any M-series Mac mini. 16 GB of memory is enough (Phase 1 caps
      in-flight companyfacts at ~370 MB) and 24 GB gives headroom. 512 GB of
      storage covers the price parquets, the DuckDB store, and the full
      `data/snapshots` worktree for years.
- [ ] Wired Ethernet, not Wi-Fi.
- [ ] **Time zone America/New_York** (System Settings → General → Date & Time).
      Both scripts now force `TZ=America/New_York` for the run date, but the
      schedulers fire in the machine's zone, so keep them the same.
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

- [ ] Clone to `~/Projects/Workspace Folder`. The scripts find the repo from
      their own path. The runbooks (`daily-stock-analysis/SKILL.md`,
      `weekly-backtest/SKILL.md`) and the weekly plist use that path, so a
      different one means editing them.
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
      (`ingest_snapshots.py --results-dir .claude/worktrees/snapshots-data --db output/snapshots.duckdb`).
- [ ] Run `RECOVERY.md` → **Check it worked**. The parquet count should be
      in the thousands, not 4.

## 4. `run_daily.sh` is up to date with the cloud script (done)

`run_daily.sh` went dormant on 2026-09-10. It now carries every step the
cloud `run.sh` gained since then, so there is nothing to port. What it does
beyond the old runbook:

- **Reads `.env`** with `analyze_stock.py`'s rules (the environment wins), so
  the Supabase, Cloudflare and Tiingo keys reach every step. It also forces
  `TZ=America/New_York` and finds the repo from its own path, with no
  hard-coded `/Users/...`.
- **Price top-up** after the enrichment (cloud `05e`), for the Phase-2
  entrants that only got a Close-only stub.
- **Portfolio alerts before the archive**, and `portfolio_alerts.json`,
  `rating_history.json`, `portfolio_nav.json` and `screen_skip.json` (only
  when it's over 10 KB) committed with the snapshot. The GitHub alert issues
  keep working, and the branch stays complete for the backtest and a move
  back.
- **Branch sharing:** it fast-forwards the `data/snapshots` worktree before
  archiving, and rebases and retries if the weekly backtest's commit lands
  first.
- **Supabase**, only when the keys are set:
  - `db_publish` before the archive (non-blocking until `DB_PRIMARY=1`)
    and `db_check` after the reports, for the `DB_CUTOVER_STREAK`;
  - the price and companyfacts caches saved to Storage as an off-site backup.

  Readers stay on the local DuckDB store. Set `SNAPSHOT_STORE_BACKEND=postgres`
  only if you want the database to serve them.
- **Coverage floor** (cloud `05h`, added to `run_daily.sh` on 2026-10-03):
  `scripts/check_coverage.py` runs as the `coverage` step just before the
  publish. A run with fewer than 70% of the prior run's rows is archived but
  not published to either site, and ends `degraded` with the `COVERAGE` line
  in `notes`. `--force` publishes it anyway. Being part of the publish step,
  it also gates a `--from publish` resume.
- **Cloudflare Pages deploy** after the GitHub publish, only when
  `CLOUDFLARE_API_TOKEN`/`CLOUDFLARE_ACCOUNT_ID`/`CF_PAGES_PROJECT` are set.
  It needs Node (`npx`) and the `CF_ACCESS_*` keys, and refuses to deploy
  without the login. `_worker.js` is deployed but never committed to
  `pages-live`.

The cut-over banners were swapped in PR #304 (2026-10-03): the
**DORMANT** banner moved from `daily-stock-analysis/SKILL.md` to
`cloud-daily-stock-analysis/SKILL.md`, with this directory's `README.md`
table updated in the same commit.

## 5. Schedule it

- [ ] **Daily:** a Claude desktop app scheduled task,
      `the-stock-analysis-model`, so Claude writes the run summary. Its
      `~/.claude/scheduled-tasks/the-stock-analysis-model/` folder must be a
      real folder holding a **copy** of the tracked
      `the-stock-analysis-model/SKILL.md`. The app refuses a symlinked task
      file ("symlink detected before open; refusing to open"); see
      `RECOVERY.md` step 4. This needs the Claude app running and signed in under
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

The Mac runs **the same `cloud-weekly-backtest/run.sh` as the cloud**,
through `weekly-backtest/mac_run.sh`. The old `weekly_backtest.sh` is
deleted: it ran `calibrate` every week, skipped the readiness census, the
Tiingo backfill and the week-over-week `compare`, and published nothing to
`data/snapshots` after 2026-07-13. The runbook is `weekly-backtest/SKILL.md`.

Already done in the repo:
- `mac_run.sh` supplies what the container used to:
  - a python3 ≥ 3.11 first on `PATH` (python.org, then Homebrew; override
    with `PYTHON3=`);
  - certifi's CA bundle;
  - `YF_IMPERSONATE=chrome`;
  - the keys in `.env`, including `TIINGO_API_KEY`;
  - `STOCK_MODEL_WORK=~/Library/Application Support/StockModel/backtest`.

  Every `run.sh` knob (`SMOKE`, `DRY_RUN`, `RUNDATE`, …) passes through. The
  log goes to `~/Library/Logs/StockModel/weekly_<date>.log`.
- `run.sh` now **keeps `$OUT/prices` between runs** and re-stages everything
  else. `returns/` and `price_backfill/` come back from the branch, which
  stays their record. The step-04 gate counts a kept file only if this run
  wrote it or it is already current, so a throttled Sunday cannot pass on
  last week's files. That makes no difference in the cloud, whose directory
  starts empty. Tested with two smoke runs: the second found all 12 files
  current and downloaded nothing.
- `com.stockmodel.weekly.plist` runs `mac_run.sh` in place (Sundays 20:00),
  with install steps in its header.

On the Mac:
- [ ] **Git push access** to `data/snapshots`. This is the same credential as
      the daily. `run.sh` pushes to the HTTPS URL in `PUSH_REMOTE`, so a token
      in the macOS keychain credential helper works. For SSH, set
      `PUSH_REMOTE`/`CLONE_REMOTE` to the `git@github.com:` form.
- [ ] **Schedule:** Sundays. Friday's daily run can last into Saturday
      afternoon, and the two now rebase over each other if they meet. Either a
      Claude Code scheduled task following `weekly-backtest/SKILL.md` (with a
      written summary), or the plist (without one; edit its two paths if the
      repo isn't at `~/Projects/Workspace Folder`).
- [ ] **Disk:** the work directory holds the staged corpus (~1.5 GB, from
      2026-07-06 on and growing about 17 MB a night) and the price parquets.
      The corpus is re-cloned from GitHub each Sunday, about 1.5 GB of
      download. The repo also gets its own `.venv` (gitignored), separate
      from the daily's `~/.venvs/stock-model`.
- [ ] **Smoke test** before the cut-over. It pushes nothing:
      `SMOKE=1 scheduled-tasks/weekly-backtest/mac_run.sh`, then
      `cat ~/Library/Application\ Support/StockModel/backtest/status.txt`.
      `RESULT OK` is the pass. A `07-compare` soft failure is expected on the
      8-ticker smoke corpus.

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
      `cloud-weekly-backtest/SKILL.md` and `weekly-backtest/SKILL.md`.
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
