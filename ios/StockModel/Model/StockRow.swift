import SwiftUI

/// BUY / LEAN BUY / HOLD / PASS: the composite score's bucket
/// (scripts/scoring.py; BUY ≥ 57, LEAN BUY ≥ 39, HOLD ≥ 25).
enum Rating: String, CaseIterable, Identifiable, Sendable {
    case buy = "BUY", leanBuy = "LEAN BUY", hold = "HOLD", pass = "PASS"
    var id: String { rawValue }

    /// The report's rating colours (templates/report.html `RC`).
    var color: Color {
        switch self {
        case .buy: return Color(hex: 0x1a9850)
        case .leanBuy: return Color(hex: 0x74c476)
        case .hold: return Color(hex: 0xfd8d3c)
        case .pass: return Color(hex: 0xde2d26)
        }
    }

    /// BUY and LEAN BUY are the buy side (the portfolio alerts' "buy line").
    var isBuySide: Bool { self == .buy || self == .leanBuy }

    var order: Int { Rating.allCases.firstIndex(of: self) ?? 99 }
}

extension Color {
    init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xff) / 255,
                  green: Double((hex >> 8) & 0xff) / 255,
                  blue: Double(hex & 0xff) / 255)
    }

    /// "#RRGGBB" from the feed (gate categories, portfolios).
    init?(hexString: String?) {
        guard var s = hexString else { return nil }
        if s.hasPrefix("#") { s.removeFirst() }
        guard s.count == 6, let v = UInt32(s, radix: 16) else { return nil }
        self.init(hex: v)
    }
}

/// One row of the run: a (table, index) view, never a copied dictionary.
/// Every accessor returns nil for N/A; display "—", never 0.
struct StockRow: Identifiable, Hashable, Sendable {
    let table: RowTable
    let index: Int

    static func == (a: StockRow, b: StockRow) -> Bool { a.index == b.index && a.ticker == b.ticker }
    func hash(into h: inout Hasher) { h.combine(index); h.combine(ticker) }

    var id: String { ticker }

    func raw(_ key: String) -> JSONValue? { table.value(key, row: index) }
    func number(_ key: String) -> Double? { raw(key)?.double }
    func text(_ key: String) -> String? { raw(key)?.string }

    var ticker: String { text("ticker") ?? "?" }
    var name: String? { text("company_name") }
    var sector: String? { text("sector") }
    var industry: String? { text("industry") }
    var country: String? { text("country") }

    var rating: Rating? { text("rating").flatMap(Rating.init(rawValue:)) }
    var priorRating: Rating? { text("rating_prev").flatMap(Rating.init(rawValue:)) }
    var ratingChanged: Bool { rating != nil && priorRating != nil && rating != priorRating }
    /// Crossed into or out of BUY/LEAN BUY since the prior run.
    var crossedBuyLine: Bool {
        guard let r = rating, let p = priorRating else { return false }
        return r.isBuySide != p.isBuySide
    }

    var score: Double? { number("_composite_score") }
    var price: Double? { number("price") }
    /// The fair value the MoS was computed against (may be a blend).
    var fairValue: Double? { number("_fv_effective") }
    var fairValueSource: String? { text("_fv_source") }
    var marginOfSafety: Double? { number("mos") }
    var fairValueP10: Double? { number("mc_p10_fv") }
    var fairValueP90: Double? { number("mc_p90_fv") }
    var roic: Double? { number("roic") }
    var wacc: Double? { number("wacc") }
    var spread: Double? { number("spread") }
    var marketCap: Double? { number("mcap") }
    var pe: Double? { number("pe") }
    var evEbitda: Double? { number("ev_ebitda") }
    var dividendYield: Double? { number("div_yield") }
    var gatesPassed: Double? { number("_gates_passed") }

    /// Warning flags (no longer caps: the rating is the score's bucket).
    var flags: [String] {
        (raw("_rating_cap_reasons")?.array ?? []).compactMap(\.string)
    }

    /// Portfolio ids this row belongs to.
    var portfolios: [String] { (raw("pf")?.array ?? []).compactMap(\.string) }

    func categoryScore(_ c: GateMeta.Category) -> Double? { number(c.scoreKey) }

    /// A gate's pass state: true pass, false fail, nil N/A.
    func gatePassed(_ g: GateMeta.Gate) -> Bool? { raw(g.gpKey)?.bool }
    func gateValue(_ g: GateMeta.Gate) -> Double? { number(g.key) }
}

extension RowTable {
    var rows: [StockRow] { (0..<count).map { StockRow(table: self, index: $0) } }
}
