import SwiftUI

/// Today's rating changes; buy-line crossings (into/out of BUY or LEAN BUY,
/// the portfolio alerts' Action level) first.
struct ChangesView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        let changed = state.rows.filter(\.ratingChanged)
        let crossings = changed.filter(\.crossedBuyLine)
        let within = changed.filter { !$0.crossedBuyLine }
        NavigationStack {
            List {
                Section { AsOfBanner(manifest: state.snapshot?.manifest) }
                Section("Crossed the buy line (\(crossings.count))") {
                    ForEach(crossings) { r in NavigationLink(value: r.ticker) { ChangeRow(row: r) } }
                }
                Section("Other changes (\(within.count))") {
                    ForEach(within) { r in NavigationLink(value: r.ticker) { ChangeRow(row: r) } }
                }
            }
            .navigationTitle("Changes")
            .navigationDestination(for: String.self) { tk in
                if let r = state.row(tk) { TickerView(row: r) }
            }
            .refreshable { await state.refresh() }
        }
    }
}

struct ChangeRow: View {
    let row: StockRow
    var body: some View {
        HStack {
            Text(row.ticker).font(.headline)
            Text(row.name ?? "").font(.caption).foregroundStyle(.secondary).lineLimit(1)
            Spacer()
            Text(row.priorRating?.rawValue ?? "—").font(.caption).foregroundStyle(row.priorRating?.color ?? .secondary)
            Image(systemName: "arrow.right").font(.caption2)
            Text(row.rating?.rawValue ?? "—").font(.caption.bold()).foregroundStyle(row.rating?.color ?? .secondary)
        }
    }
}

/// Published portfolio definitions; members are the rows whose `pf` holds the id.
struct PortfoliosView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        NavigationStack {
            List(state.snapshot?.portfolios.portfolios ?? []) { p in
                let members = state.rows.filter { $0.portfolios.contains(p.id) }
                NavigationLink {
                    List(members.sorted { ($0.score ?? -1) > ($1.score ?? -1) }) { r in
                        NavigationLink { TickerView(row: r) } label: { StockRowView(row: r) }
                    }
                    .navigationTitle(p.name)
                } label: {
                    HStack {
                        Circle().fill(Color(hexString: p.color) ?? .blue).frame(width: 10, height: 10)
                        VStack(alignment: .leading) {
                            Text(p.name)
                            if let d = p.description { Text(d).font(.caption).foregroundStyle(.secondary).lineLimit(2) }
                        }
                        Spacer()
                        Text("\(members.count)").foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Portfolios")
        }
    }
}

struct SettingsView: View {
    @Environment(AppState.self) private var state

    var body: some View {
        @Bindable var state = state
        NavigationStack {
            Form {
                Section("Source") {
                    Picker("Feed", selection: $state.source) {
                        ForEach(FeedSource.allCases) { Text($0.rawValue).tag($0) }
                    }
                    switch state.source {
                    case .local: TextField("URL", text: $state.localURL).urlField()
                    case .mini: TextField("URL", text: $state.miniURL).urlField()
                    case .cloudflare: TextField("URL", text: $state.cloudflareURL).urlField()
                    }
                    Button("Reload") { Task { await state.refresh() } }
                }
                if state.source.needsLogin {
                    Section("Account") {
                        if state.access.isSignedIn {
                            Button("Sign out and clear data", role: .destructive) { state.signOut() }
                        } else {
                            Button("Sign in") { state.showSignIn = true }
                        }
                    }
                }
                Section("This build") {
                    LabeledContent("Run date", value: state.snapshot?.manifest.runDate ?? "—")
                    LabeledContent("Build", value: state.build ?? "—")
                    LabeledContent("Stocks", value: state.snapshot.map { "\($0.manifest.rows)" } ?? "—")
                    if let e = state.error { Text(e).foregroundStyle(.red).font(.caption) }
                    Button("Clear cache") { FeedClient.clearCache() }
                }
            }
            .navigationTitle("Settings")
        }
    }
}

private extension View {
    func urlField() -> some View {
        self.keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
    }
}
