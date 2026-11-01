import Charts
import SwiftUI

struct TickerView: View {
    @Environment(AppState.self) private var state
    let row: StockRow

    @State private var px: [PricePoint] = []
    @State private var spy: [PricePoint] = []
    @State private var hist: HistShard?
    @State private var chartError: String?

    var body: some View {
        List {
            Section { header }
            Section("Fair value") { FairValueBar(row: row) }
            if let gates = state.snapshot?.gates {
                Section("Scores") {
                    ForEach(gates.categories) { c in
                        LabeledContent(c.name, value: Fmt.num(row.categoryScore(c), digits: 0))
                    }
                }
                ForEach(gates.categories) { c in
                    Section(c.name) {
                        ForEach(gates.gates.filter { $0.category == c.name }) { g in
                            GateRowView(gate: g, row: row)
                        }
                    }
                }
            }
            Section("Price, 5 years (vs S&P 500)") { priceChart }
            if let hist { Section("Record") { HistoryTable(hist: hist) } }
            if !row.flags.isEmpty {
                Section("Flags") { ForEach(row.flags, id: \.self) { Label($0, systemImage: "flag") } }
            }
            Section("Returns and spread") {
                LabeledContent("ROIC", value: Fmt.pct(row.roic))
                LabeledContent("WACC", value: Fmt.pct(row.wacc))
                LabeledContent("Spread", value: Fmt.pct(row.spread))
                LabeledContent("P/E", value: Fmt.multiple(row.pe))
                LabeledContent("EV/EBITDA", value: Fmt.multiple(row.evEbitda))
                LabeledContent("Dividend yield", value: Fmt.pct(row.dividendYield))
                LabeledContent("Market cap", value: Fmt.cap(row.marketCap))
            }
        }
        .navigationTitle(row.ticker)
        .task(id: row.ticker) { await loadShards() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(row.name ?? row.ticker).font(.title3.bold())
            Text([row.sector, row.industry, row.country].compactMap { $0 }.joined(separator: " · "))
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Text(Fmt.price(row.price)).font(.title2.monospacedDigit())
                Spacer()
                RatingPill(rating: row.rating, score: row.score)
            }
            HStack {
                Text("Gates passed \(Fmt.num(row.gatesPassed, digits: 0))")
                if row.ratingChanged, let p = row.priorRating {
                    Text("· was \(p.rawValue)")
                }
            }
            .font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var priceChart: some View {
        if let chartError {
            Text(chartError).font(.caption).foregroundStyle(.secondary)
        } else if px.isEmpty {
            ProgressView().frame(maxWidth: .infinity)
        } else {
            // Both rebased to 0% at the window start, as the report does.
            let base = px.first!.close
            let spyBase = spy.first?.close
            Chart {
                ForEach(px) { p in
                    LineMark(x: .value("Date", p.date), y: .value("Return", p.close / base - 1),
                             series: .value("Series", row.ticker))
                        .foregroundStyle(row.rating?.color ?? .blue)
                }
                if let spyBase {
                    ForEach(spy) { p in
                        LineMark(x: .value("Date", p.date), y: .value("Return", p.close / spyBase - 1),
                                 series: .value("Series", "SPY"))
                            .foregroundStyle(.gray)
                    }
                }
                if let fv = row.fairValue, let last = px.last {
                    // Fair value as a level relative to the same base.
                    RuleMark(y: .value("Fair value", fv / base - 1))
                        .foregroundStyle(.green.opacity(0.6))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                        .annotation(position: .top, alignment: .leading) {
                            Text("FV \(Fmt.price(fv)) vs \(Fmt.price(last.close))").font(.caption2)
                        }
                }
            }
            .chartYAxis { AxisMarks(format: FloatingPointFormatStyle<Double>.Percent()) }
            .frame(height: 220)
        }
    }

    private func loadShards() async {
        guard let client = state.client, let build = state.build else { return }
        do {
            let meta = try await client.pricesMeta(build: build)
            let cutoff = Calendar.current.date(byAdding: .year, value: -5, to: .now) ?? .distantPast
            func series(_ s: PxShard?) -> [PricePoint] {
                (s?.points(dates: meta.dates) ?? []).compactMap { p in
                    guard let d = Self.day.date(from: p.date), d >= cutoff else { return nil }
                    return PricePoint(date: d, close: p.close)
                }
            }
            px = series(try await client.px(row.ticker, build: build))
            spy = series(meta.indexPx?["SPY"])
            if px.isEmpty { chartError = "No price history for \(row.ticker)." }
            hist = try await client.hist(row.ticker, build: build)
        } catch FeedError.buildChanged {
            await state.refresh()
        } catch {
            chartError = error.localizedDescription
        }
    }

    static let day: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        return f
    }()
}

struct PricePoint: Identifiable {
    let date: Date
    let close: Double
    var id: Date { date }
}

/// Price against the fair-value range: Monte Carlo P10–P90 with the
/// effective fair value marked.
struct FairValueBar: View {
    let row: StockRow
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent("Fair value", value: Fmt.price(row.fairValue))
            LabeledContent("Margin of safety", value: Fmt.pct(row.marginOfSafety))
            LabeledContent("Range (P10–P90)", value: "\(Fmt.price(row.fairValueP10)) – \(Fmt.price(row.fairValueP90))")
            if let src = row.fairValueSource { LabeledContent("Source", value: src) }
            if let lo = row.fairValueP10, let hi = row.fairValueP90, let p = row.price, hi > lo {
                GeometryReader { g in
                    let span = max(hi, p) - min(lo, p)
                    let x = { (v: Double) in (v - min(lo, p)) / span * g.size.width }
                    ZStack(alignment: .leading) {
                        Capsule().fill(.green.opacity(0.25))
                            .frame(width: x(hi) - x(lo), height: 8).offset(x: x(lo))
                        if let fv = row.fairValue {
                            Rectangle().fill(.green).frame(width: 2, height: 14).offset(x: x(fv))
                        }
                        Circle().fill(.primary).frame(width: 10, height: 10).offset(x: x(p) - 5)
                    }
                }
                .frame(height: 16)
            }
        }
    }
}

struct GateRowView: View {
    let gate: GateMeta.Gate
    let row: StockRow
    var body: some View {
        let pass = row.gatePassed(gate)
        HStack {
            Image(systemName: pass == true ? "checkmark.circle.fill" : pass == false ? "xmark.circle.fill" : "minus.circle")
                .foregroundStyle(pass == true ? .green : pass == false ? .red : .secondary)
            VStack(alignment: .leading) {
                Text(gate.label)
                if let t = gate.threshold { Text(t).font(.caption2).foregroundStyle(.secondary) }
            }
            Spacer()
            Text(pass == nil ? "N/A" : Fmt.gate(row.gateValue(gate), fmt: gate.fmt))
                .monospacedDigit().foregroundStyle(.secondary)
        }
    }
}

/// The last 10 fiscal years of a few headline lines from hist/.
struct HistoryTable: View {
    let hist: HistShard
    private let lines: [(key: String, label: String)] = [
        ("rev", "Revenue"), ("op", "Operating income"), ("ni", "Net income"),
        ("ocf", "Operating cash flow"), ("capex", "Capex"), ("shares", "Shares"),
    ]
    var body: some View {
        let years = Array(Array(Set(lines.flatMap { hist.series[$0.key]?.keys.map { String($0.prefix(4)) } ?? [] }))
            .sorted().suffix(10))
        ScrollView(.horizontal) {
            Grid(alignment: .trailing, horizontalSpacing: 12, verticalSpacing: 4) {
                GridRow {
                    Text("").gridColumnAlignment(.leading)
                    ForEach(years, id: \.self) { Text($0).font(.caption.bold()) }
                }
                ForEach(lines, id: \.key) { line in
                    let byYear = Dictionary((hist.series[line.key] ?? [:]).map { (String($0.key.prefix(4)), $0.value) },
                                            uniquingKeysWith: { a, _ in a })
                    GridRow {
                        Text(line.label).font(.caption)
                        ForEach(years, id: \.self) { y in
                            Text(line.key == "shares" ? Fmt.cap(byYear[y]).replacingOccurrences(of: "$", with: "")
                                                      : Fmt.cap(byYear[y]))
                                .font(.caption.monospacedDigit())
                        }
                    }
                }
            }
        }
    }
}
