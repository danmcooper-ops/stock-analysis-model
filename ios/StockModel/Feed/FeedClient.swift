import CryptoKit
import Foundation

enum FeedError: Error, LocalizedError {
    case signInRequired
    case http(Int, String)
    case unsupportedVersion(Int)
    case hashMismatch(String)
    case buildChanged

    var errorDescription: String? {
        switch self {
        case .signInRequired: return "Sign in to load the report."
        case .http(let code, let path): return "HTTP \(code) for \(path)."
        case .unsupportedVersion(let v): return "This app reads feed version \(Manifest.supportedVersion); the site has \(v). Update the app."
        case .hashMismatch(let f): return "\(f) did not match the manifest; try again after the publish settles."
        case .buildChanged: return "A newer report was published; reloading."
        }
    }
}

/// One loaded build: everything the list and pages need up front.
struct FeedSnapshot: Sendable {
    let manifest: Manifest
    let rows: RowTable
    let gates: GateMeta
    let portfolios: PortfolioFeed
}

/// Fetches the feed and caches it per build under
/// Application Support/feed/<build>/. Rule: files from two builds are never
/// mixed (px/ shards are offsets into that build's prices_meta.json).
actor FeedClient {
    let baseURL: URL
    private let token: String?
    private let session: URLSession
    private let root: URL
    private var checkedBuild: (build: String, at: Date)?

    init(baseURL: URL, token: String?) {
        self.baseURL = baseURL
        self.token = token
        let cfg = URLSessionConfiguration.default
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData   // freshness is the manifest's job
        cfg.timeoutIntervalForRequest = 30
        session = URLSession(configuration: cfg)
        root = FeedClient.cacheRoot
    }

    static var cacheRoot: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("feed", isDirectory: true)
    }

    /// Deletes every cached build (sign-out; vendor terms require it).
    static func clearCache() {
        try? FileManager.default.removeItem(at: cacheRoot)
    }

    // MARK: Build-level files

    /// Newest build from the network, else the newest cached one (offline).
    func load() async throws -> FeedSnapshot {
        do {
            let data = try await get("app/manifest.json")
            let manifest = try JSONDecoder().decode(Manifest.self, from: data)
            guard manifest.version == Manifest.supportedVersion else {
                throw FeedError.unsupportedVersion(manifest.version)
            }
            let dir = buildDir(manifest.build)
            if !FileManager.default.fileExists(atPath: dir.appendingPathComponent("manifest.json").path) {
                try await download(manifest, into: dir)
            }
            prune(keeping: manifest.build)
            checkedBuild = (manifest.build, .now)
            return try snapshot(from: dir)
        } catch let e as URLError where e.code == .notConnectedToInternet || e.code == .timedOut
                    || e.code == .networkConnectionLost || e.code == .cannotConnectToHost {
            if let dir = newestCachedBuild() { return try snapshot(from: dir) }
            throw e
        }
    }

    private func download(_ manifest: Manifest, into dir: URL) async throws {
        let tmp = root.appendingPathComponent(".tmp-\(manifest.build)", isDirectory: true)
        try? FileManager.default.removeItem(at: tmp)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        for (name, sha) in manifest.files {
            let data = try await get("app/\(name)")
            guard Self.sha256(data) == sha else { throw FeedError.hashMismatch(name) }
            try data.write(to: tmp.appendingPathComponent(name))
        }
        // prices_meta.json is not in the manifest but belongs to the same deploy.
        let meta = try await get("prices_meta.json")
        try meta.write(to: tmp.appendingPathComponent("prices_meta.json"))
        try JSONEncoder().encode(manifest).write(to: tmp.appendingPathComponent("manifest.json"))
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.moveItem(at: tmp, to: dir)   // the build appears whole or not at all
    }

    private func snapshot(from dir: URL) throws -> FeedSnapshot {
        let dec = JSONDecoder()
        return FeedSnapshot(
            manifest: try dec.decode(Manifest.self, from: Data(contentsOf: dir.appendingPathComponent("manifest.json"))),
            rows: try RowTable(data: Data(contentsOf: dir.appendingPathComponent("rows.json"))),
            gates: try dec.decode(GateMeta.self, from: Data(contentsOf: dir.appendingPathComponent("gates.json"))),
            portfolios: try dec.decode(PortfolioFeed.self, from: Data(contentsOf: dir.appendingPathComponent("portfolios.json"))))
    }

    // MARK: Per-ticker shards (lazy, cached inside the build)

    func pricesMeta(build: String) throws -> PricesMeta {
        try JSONDecoder().decode(PricesMeta.self,
                                 from: Data(contentsOf: buildDir(build).appendingPathComponent("prices_meta.json")))
    }

    func px(_ ticker: String, build: String) async throws -> PxShard? {
        try await ensureLive(build)
        return try await shard("px/\(ticker).json", build: build).map { try JSONDecoder().decode(PxShard.self, from: $0) }
    }

    func hist(_ ticker: String, build: String) async throws -> HistShard? {
        try await shard("hist/\(ticker).json", build: build).map { try JSONDecoder().decode(HistShard.self, from: $0) }
    }

    /// nil when the site has no such shard (404): not every ticker has one.
    private func shard(_ path: String, build: String) async throws -> Data? {
        let file = buildDir(build).appendingPathComponent(path)
        if let d = try? Data(contentsOf: file) { return d }
        do {
            let d = try await get(path)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try d.write(to: file)
            return d
        } catch FeedError.http(404, _) {
            return nil
        }
    }

    /// A px/ shard only makes sense against its own build's dates axis, and
    /// the site serves whatever deploy is live. So before fetching one (at
    /// most once every 5 minutes) confirm the live build is still ours;
    /// if not, throw so the app reloads instead of mixing builds.
    private func ensureLive(_ build: String) async throws {
        if let c = checkedBuild, c.build == build, Date.now.timeIntervalSince(c.at) < 300 { return }
        let data = try await get("app/manifest.json")
        let live = try JSONDecoder().decode(Manifest.self, from: data).build
        guard live == build else { throw FeedError.buildChanged }
        checkedBuild = (build, .now)
    }

    // MARK: Plumbing

    private func get(_ path: String) async throws -> Data {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        if let token { req.setValue(token, forHTTPHeaderField: "cf-access-token") }
        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse else { throw FeedError.http(0, path) }
        // An expired or missing session ends on Access's login page.
        if http.url?.host?.hasSuffix("cloudflareaccess.com") == true || http.statusCode == 401 || http.statusCode == 403 {
            throw FeedError.signInRequired
        }
        guard http.statusCode == 200 else { throw FeedError.http(http.statusCode, path) }
        return data
    }

    private func buildDir(_ build: String) -> URL { root.appendingPathComponent(build, isDirectory: true) }

    private func cachedBuilds() -> [URL] {
        let fm = FileManager.default
        let dirs = (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        return dirs.filter { !$0.lastPathComponent.hasPrefix(".") &&
                             fm.fileExists(atPath: $0.appendingPathComponent("manifest.json").path) }
            .sorted { modified($0) > modified($1) }
    }

    private func newestCachedBuild() -> URL? { cachedBuilds().first }

    /// Keep the current build and one before it.
    private func prune(keeping build: String) {
        for (i, dir) in cachedBuilds().filter({ $0.lastPathComponent != build }).enumerated() where i >= 1 {
            try? FileManager.default.removeItem(at: dir)
        }
    }

    private func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
