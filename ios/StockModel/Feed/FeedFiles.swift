import Foundation

/// `app/manifest.json`. Written last by the pipeline; names every file by
/// SHA-256. The app polls only this file.
struct Manifest: Codable, Equatable, Sendable {
    /// The feed shape the app understands (report_html.APP_FEED_VERSION).
    static let supportedVersion = 1

    let version: Int
    let runDate: String?
    let generatedAt: String?
    let build: String
    let rows: Int
    let files: [String: String]

    enum CodingKeys: String, CodingKey {
        case version, build, rows, files
        case runDate = "run_date"
        case generatedAt = "generated_at"
    }

    var generatedDate: Date? {
        generatedAt.flatMap { ISO8601DateFormatter().date(from: $0) }
    }

    /// True once the build is older than `hours` (a missed night).
    func isStale(now: Date = .now, hours: Double = 26) -> Bool {
        guard let g = generatedDate else { return true }
        return now.timeIntervalSince(g) > hours * 3600
    }
}

/// `app/gates.json` (scripts/scoring.py `gate_metadata`).
struct GateMeta: Codable, Sendable {
    struct Gate: Codable, Identifiable, Sendable {
        let key: String        // _gate_*: the raw metric value
        let gpKey: String      // _gp_*: true pass / false fail / null N/A
        let scoreKey: String   // _score_*
        let label: String
        let threshold: String?
        let category: String
        let fmt: String?
        let weight: Double
        var id: String { key }
    }
    struct Category: Codable, Identifiable, Sendable {
        let name: String
        let weight: Double
        let dark: String
        let light: String
        let scoreKey: String
        var id: String { name }
    }
    let gates: [Gate]
    let categories: [Category]
}

/// `app/portfolios.json` (the report's PF_PUB). Only what v1 shows; unknown
/// keys are ignored. Membership lives on each row's `pf`.
struct PortfolioFeed: Codable, Sendable {
    struct Portfolio: Codable, Identifiable, Sendable {
        let id: String
        let name: String
        let description: String?
        let color: String?
    }
    let portfolios: [Portfolio]
}

/// `prices_meta.json`: the shared dates axis for every `px/` shard, plus
/// index series (SPY) embedded whole.
struct PricesMeta: Decodable, Sendable {
    let dates: [String]
    let indexPx: [String: PxShard]?
    enum CodingKeys: String, CodingKey {
        case dates
        case indexPx = "index_px"
    }
}

/// `px/<TICKER>.json`: closes from `dates[i0]` on (dividend-adjusted).
/// Only meaningful against the prices_meta.json of the same build.
struct PxShard: Decodable, Sendable {
    let i0: Int
    let p: [Double?]

    func points(dates: [String]) -> [(date: String, close: Double)] {
        p.enumerated().compactMap { j, v in
            let i = i0 + j
            guard let v, v.isFinite, i < dates.count else { return nil }
            return (dates[i], v)
        }
    }
}

/// `hist/<TICKER>.json`: annual EDGAR series, short key -> {date: value}
/// (rev, ni, ocf, capex, op, shares, debt, cash, equity, ...). Other keys
/// (fya, dv) have other shapes and are skipped here.
struct HistShard: Decodable, Sendable {
    let series: [String: [String: Double]]

    init(from decoder: Decoder) throws {
        let raw = try [String: JSONValue](from: decoder)
        var out: [String: [String: Double]] = [:]
        for (k, v) in raw {
            guard let obj = v.object else { continue }
            let nums = obj.compactMapValues { $0.double }
            if !nums.isEmpty && nums.count == obj.count { out[k] = nums }
        }
        series = out
    }

    /// The series sorted oldest first.
    func sorted(_ key: String) -> [(date: String, value: Double)] {
        (series[key] ?? [:]).sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }
}
