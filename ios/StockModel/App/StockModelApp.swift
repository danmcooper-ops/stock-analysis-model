import SwiftUI

// Replace the StockModelApp.swift and ContentView.swift that Xcode's
// template generated with this file.
@main
struct StockModelApp: App {
    @State private var state = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(state)
                .task { await state.refresh() }
        }
    }
}

struct RootView: View {
    @Environment(AppState.self) private var state
    @Environment(\.scenePhase) private var phase

    var body: some View {
        @Bindable var state = state
        TabView {
            StocksView()
                .tabItem { Label("Stocks", systemImage: "list.bullet.rectangle") }
            ChangesView()
                .tabItem { Label("Changes", systemImage: "arrow.left.arrow.right") }
            PortfoliosView()
                .tabItem { Label("Portfolios", systemImage: "folder") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gear") }
        }
        .onChange(of: phase) { _, p in
            if p == .active { Task { await state.refresh() } }
        }
        .sheet(isPresented: $state.showSignIn) {
            if let url = state.baseURL {
                NavigationStack {
                    AccessSignInView(siteURL: url) { state.didSignIn(token: $0) }
                        .navigationTitle("Sign in")
                        .navigationBarTitleDisplayMode(.inline)
                        .toolbar {
                            ToolbarItem(placement: .cancellationAction) {
                                Button("Cancel") { state.showSignIn = false }
                            }
                        }
                }
            }
        }
    }
}

/// "as of" line, with a warning once the build is older than 26 h.
struct AsOfBanner: View {
    let manifest: Manifest?
    var body: some View {
        if let m = manifest {
            HStack(spacing: 6) {
                if m.isStale() { Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange) }
                Text("As of \(m.runDate ?? "?") · \(m.rows) stocks")
                if m.isStale() { Text("· not updated").foregroundStyle(.orange) }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}
