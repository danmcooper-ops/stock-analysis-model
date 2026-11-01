import SwiftUI

enum StockSort: String, CaseIterable, Identifiable {
    case score = "Score", mos = "MoS", cap = "Market cap", ticker = "Ticker"
    var id: String { rawValue }
}

struct StocksView: View {
    @Environment(AppState.self) private var state
    @State private var query = ""
    @State private var ratings: Set<Rating> = []
    @State private var sector: String?
    @State private var sort: StockSort = .score

    private var sectors: [String] {
        Array(Set(state.rows.compactMap(\.sector))).sorted()
    }

    private var shown: [StockRow] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        let out = state.rows.filter { r in
            (q.isEmpty || r.ticker.lowercased().hasPrefix(q) || (r.name?.lowercased().contains(q) ?? false))
                && (ratings.isEmpty || r.rating.map { ratings.contains($0) } == true)
                && (sector == nil || r.sector == sector)
        }
        // N/A sorts last, whatever the direction.
        func desc(_ a: Double?, _ b: Double?) -> Bool {
            switch (a, b) {
            case let (x?, y?): return x > y
            case (_?, nil): return true
            default: return false
            }
        }
        switch sort {
        case .score: return out.sorted { desc($0.score, $1.score) }
        case .mos: return out.sorted { desc($0.marginOfSafety, $1.marginOfSafety) }
        case .cap: return out.sorted { desc($0.marketCap, $1.marketCap) }
        case .ticker: return out.sorted { $0.ticker < $1.ticker }
        }
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(shown) { r in
                        NavigationLink(value: r.ticker) { StockRowView(row: r) }
                    }
                } header: {
                    AsOfBanner(manifest: state.snapshot?.manifest)
                }
            }
            .listStyle(.plain)
            .overlay {
                if state.rows.isEmpty {
                    if state.loading { ProgressView("Loading…") }
                    else if let e = state.error { ContentUnavailableView("Couldn't load", systemImage: "wifi.exclamationmark", description: Text(e)) }
                }
            }
            .searchable(text: $query, prompt: "Ticker or name")
            .refreshable { await state.refresh() }
            .navigationTitle("Stocks")
            .navigationDestination(for: String.self) { tk in
                if let r = state.row(tk) { TickerView(row: r) } else { Text("Not in this run") }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Picker("Sort", selection: $sort) {
                            ForEach(StockSort.allCases) { Text($0.rawValue).tag($0) }
                        }
                        Section("Rating") {
                            ForEach(Rating.allCases) { r in
                                Button {
                                    if ratings.contains(r) { ratings.remove(r) } else { ratings.insert(r) }
                                } label: {
                                    Label(r.rawValue, systemImage: ratings.contains(r) ? "checkmark.circle.fill" : "circle")
                                }
                            }
                        }
                        Picker("Sector", selection: $sector) {
                            Text("All sectors").tag(String?.none)
                            ForEach(sectors, id: \.self) { Text($0).tag(String?.some($0)) }
                        }
                    } label: {
                        Image(systemName: "line.3.horizontal.decrease.circle")
                    }
                }
            }
        }
    }
}

struct StockRowView: View {
    let row: StockRow
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(row.ticker).font(.headline)
                Text(row.name ?? "").font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text(Fmt.price(row.price)).font(.subheadline.monospacedDigit())
                Text("MoS \(Fmt.pct(row.marginOfSafety, digits: 0))").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            RatingPill(rating: row.rating, score: row.score)
        }
    }
}

struct RatingPill: View {
    let rating: Rating?
    let score: Double?
    var body: some View {
        VStack(spacing: 1) {
            Text(rating?.rawValue ?? "N/A").font(.caption2.bold())
            Text(Fmt.num(score, digits: 0)).font(.caption2.monospacedDigit())
        }
        .foregroundStyle(.white)
        .frame(width: 64)
        .padding(.vertical, 4)
        .background(rating?.color ?? .gray, in: RoundedRectangle(cornerRadius: 6))
    }
}
