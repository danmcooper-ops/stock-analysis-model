import Foundation
import Observation

/// How the feed's requests are signed in. Cloudflare Access today; Sign in
/// with Apple + a Worker-checked token if the app ever goes public, behind
/// the same protocol.
@MainActor
protocol AuthProvider: AnyObject {
    var isSignedIn: Bool { get }
    /// Add credentials to a request (no-op when there are none).
    func authorize(_ request: inout URLRequest)
    /// Forget the credential (sign-out, or the server said it expired).
    func signOut()
}

/// The local `python3 -m http.server` or the Mac mini on the home network.
@MainActor
final class NoAuth: AuthProvider {
    var isSignedIn: Bool { true }
    func authorize(_ request: inout URLRequest) {}
    func signOut() {}
}

/// Cloudflare Access (cloudflare/README.md, "The iPhone/iPad app"). The
/// sign-in sheet (AccessSignInView) runs Access's own email + one-time-code
/// login in a web view and hands over the CF_Authorization cookie; requests
/// then carry it as the `cf-access-token` header, which Access accepts in
/// place of the cookie. Its lifetime is the Access app's session duration.
@MainActor
@Observable
final class AccessAuth: AuthProvider {
    static let account = "cf-access-token"
    private(set) var token: String?

    init() { token = KeychainStore.get(Self.account) }

    var isSignedIn: Bool { token != nil }

    func authorize(_ request: inout URLRequest) {
        if let token { request.setValue(token, forHTTPHeaderField: "cf-access-token") }
    }

    func store(token: String) {
        self.token = token
        KeychainStore.set(token, for: Self.account)
    }

    func signOut() {
        token = nil
        KeychainStore.remove(Self.account)
    }
}
