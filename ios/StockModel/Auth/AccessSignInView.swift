import SwiftUI
import WebKit

/// Loads the site in a web view. Cloudflare Access redirects to its login
/// (email, then the one-time code), then back to the site, setting the
/// CF_Authorization cookie on the site's host. Once that cookie appears it
/// is handed to `onToken` and the sheet can close.
struct AccessSignInView: UIViewRepresentable {
    let siteURL: URL
    let onToken: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(host: siteURL.host ?? "", onToken: onToken) }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()   // the token lives in the Keychain, not here
        let view = WKWebView(frame: .zero, configuration: config)
        view.navigationDelegate = context.coordinator
        view.load(URLRequest(url: siteURL.appendingPathComponent("app/manifest.json")))
        return view
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKNavigationDelegate {
        let host: String
        let onToken: (String) -> Void
        private var done = false

        init(host: String, onToken: @escaping (String) -> Void) {
            self.host = host
            self.onToken = onToken
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            guard !done else { return }
            webView.configuration.websiteDataStore.httpCookieStore.getAllCookies { cookies in
                // HttpOnly cookies are visible to the cookie store.
                guard let c = cookies.first(where: {
                    $0.name == "CF_Authorization" && self.host.hasSuffix($0.domain.trimmingCharacters(in: CharacterSet(charactersIn: ".")))
                }) else { return }
                DispatchQueue.main.async {
                    guard !self.done else { return }
                    self.done = true
                    self.onToken(c.value)
                }
            }
        }
    }
}
