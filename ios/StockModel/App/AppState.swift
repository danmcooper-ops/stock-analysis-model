import Foundation
import Observation

/// Where the feed comes from (Settings). Local and Mini need no login.
enum FeedSource: String, CaseIterable, Identifiable {
    case local = "Local (Simulator)"
    case mini = "Mac mini (home)"
    case cloudflare = "Cloudflare"
    var id: String { rawValue }
    var needsLogin: Bool { self == .cloudflare }
}

@MainActor
@Observable
final class AppState {
    // Settings (persisted in UserDefaults; no secrets here).
    var source: FeedSource {
        didSet { UserDefaults.standard.set(source.rawValue, forKey: "source") }
    }
    var localURL: String { didSet { UserDefaults.standard.set(localURL, forKey: "localURL") } }
    var miniURL: String { didSet { UserDefaults.standard.set(miniURL, forKey: "miniURL") } }
    var cloudflareURL: String { didSet { UserDefaults.standard.set(cloudflareURL, forKey: "cloudflareURL") } }

    // Feed state.
    private(set) var snapshot: FeedSnapshot?
    private(set) var rows: [StockRow] = []
    private(set) var loading = false
    var error: String?
    var showSignIn = false

    let access = AccessAuth()
    private(set) var client: FeedClient?

    init() {
        let d = UserDefaults.standard
        source = d.string(forKey: "source").flatMap(FeedSource.init(rawValue:)) ?? .local
        localURL = d.string(forKey: "localURL") ?? "http://localhost:8000/"
        // The live-quotes service's root; "/" redirects to the newest
        // immutable /b/<build>/ copy, which refresh() resolves.
        miniURL = d.string(forKey: "miniURL") ?? "http://mini.local:8765/"
        cloudflareURL = d.string(forKey: "cloudflareURL") ?? "https://stock-analysis-model.pages.dev/"
    }

    var baseURL: URL? {
        switch source {
        case .local: return URL(string: localURL)
        case .mini: return URL(string: miniURL)
        case .cloudflare: return URL(string: cloudflareURL)
        }
    }

    var build: String? { snapshot?.manifest.build }

    func refresh() async {
        guard var base = baseURL else { error = "Set a valid URL in Settings."; return }
        if source.needsLogin && !access.isSignedIn { showSignIn = true; return }
        loading = true
        defer { loading = false }
        if source == .mini {
            do { base = try await Self.resolveMiniBuild(base) }
            catch { self.error = "Mac mini: \(error.localizedDescription)"; return }
        }
        let c = FeedClient(baseURL: base, token: source.needsLogin ? access.token : nil)
        client = c
        do {
            let s = try await c.load()
            snapshot = s
            rows = s.rows.rows
            error = nil
        } catch FeedError.signInRequired {
            access.signOut()
            showSignIn = true
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// scripts/live_quotes.py answers "/" with a 302 to /b/<build>/, a copy
    /// of docs/ that never changes, so pinning to it can never mix builds.
    static func resolveMiniBuild(_ root: URL) async throws -> URL {
        var req = URLRequest(url: root)
        req.httpMethod = "HEAD"
        let (_, resp) = try await URLSession.shared.data(for: req)
        guard let final = resp.url?.absoluteString, final.contains("/b/"),
              let url = URL(string: final.hasSuffix("/") ? final : final + "/") else {
            throw URLError(.badServerResponse)
        }
        return url
    }

    func didSignIn(token: String) {
        access.store(token: token)
        showSignIn = false
        Task { await refresh() }
    }

    func signOut() {
        access.signOut()
        FeedClient.clearCache()
        snapshot = nil
        rows = []
    }

    func row(_ ticker: String) -> StockRow? { rows.first { $0.ticker == ticker } }
}
