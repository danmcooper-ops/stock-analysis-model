# StockModel: the iPhone/iPad app

A SwiftUI viewer of the nightly run. It shows exactly what the pipeline
published and never re-runs a model. The plan, milestones and the
data-licence decision are in [design/ios-app-plan.md](../design/ios-app-plan.md).

The Swift sources here are a starter set. They were written without a Swift
toolchain at hand, so the first build may need small fixes.

## Set up (about 20 minutes)

1. **Create the project.** In Xcode, choose File → New → Project → iOS → App.
   - Product Name `StockModel`, Interface SwiftUI, Language Swift, Storage
     None, Testing System XCTest.
   - Save it in this `ios/` folder, so the project sits at
     `ios/StockModel.xcodeproj` beside these sources.
   - Uncheck "Create Git repository": the repo already is one.
2. **Use these sources.** Delete the template's `ContentView.swift` and
   `StockModelApp.swift` (Move to Trash). Drag the `StockModel/` subfolders
   (`App`, `Auth`, `Feed`, `Model`, `Features`, `Shared`) onto the StockModel
   group, with "Create groups" and the app target ticked. Drag
   `StockModelTests/RowTableTests.swift` into the test target, then delete the
   template's own test file.
3. **Target settings** (StockModel target):
   - General → Minimum Deployments: **iOS 17.0**. Supported Destinations:
     iPhone and iPad.
   - Build Settings → **Swift Language Version: Swift 5**, and, on Xcode 26,
     **Default Actor Isolation: nonisolated**. The sources use explicit
     `@MainActor` and `actor`; Swift 6's strict checking can be turned on later.
   - Info tab: add `App Transport Security Settings` → `Allows Local
     Networking` = YES (plain-HTTP localhost and `*.local`). Add `Privacy -
     Local Network Usage Description` = "Loads the report from your Mac mini."
4. **Serve a feed locally.** Get a published `docs/` folder from the Mac mini.
   It must come from a render after 2026-11-01, so that it contains `app/`.

   ```bash
   rsync -a mini.local:stock-analysis-model/.claude/worktrees/pages-live/docs/ ~/stock-docs/
   ```

   ```bash
   python3 -m http.server 8000 -d ~/stock-docs
   ```

5. **Run** on an iPhone simulator. Settings → Feed = Local
   (`http://localhost:8000/`) → Reload. The Stocks tab fills, and ⌘U runs the
   tests.

## Sources

| Folder | What it does |
|---|---|
| `Feed/` | `FeedClient` (manifest-first download, SHA-256 checked, cached per build, offline fallback), `RowTable` (columnar port of `unpack_rows`), file types for `manifest`/`gates`/`portfolios`/`px`/`hist` |
| `Auth/` | `AuthProvider`, `AccessAuth` (Keychain), `AccessSignInView` (Cloudflare Access one-time code in a web view) |
| `Model/` | `StockRow` (typed, N/A-preserving accessors over one row), `Rating` |
| `Features/` | Stocks list (search, rating/sector filter, sort), ticker page (scores, gates, fair-value range, 5-year chart vs SPY, 10-year record), Changes, Portfolios, Settings |

## Rules

- **Never mix builds.** `px/` shards index into that deploy's
  `prices_meta.json`. `FeedClient` caches per `manifest.build` and checks the
  live build before fetching a shard.
- **N/A is never 0.** Absent, null and non-finite all show "—".
- **No credentials in the project.** The Access token lives in the Keychain.
  Sign-out deletes it and every cached build.
- **No one else is invited until the data licence covers it** (plan, Part C).

## Using the Mac mini or Cloudflare

- **Mac mini:** set Feed = Mac mini with `http://<mini>.local:8765/`, the
  live-quotes service (MAC-MINI-SETUP.md §8). The app follows its redirect to
  the newest immutable `/b/<build>/` copy.
- **Cloudflare:** set Feed = Cloudflare and tap Sign in. The site's Access
  login sends a code to an allowlisted email, and the session lasts the Access
  app's session duration (30 days). See `cloudflare/README.md`, "The
  iPhone/iPad app".
