import Foundation

/// Display formatting. Every formatter takes an optional and shows "—" for
/// N/A, never 0.
enum Fmt {
    static let dash = "—"

    static func price(_ v: Double?) -> String {
        guard let v else { return dash }
        return v.formatted(.currency(code: "USD").precision(.fractionLength(2)))
    }

    /// Fractions as percent: 0.153 -> "15.3%".
    static func pct(_ v: Double?, digits: Int = 1) -> String {
        guard let v else { return dash }
        return v.formatted(.percent.precision(.fractionLength(digits)))
    }

    static func num(_ v: Double?, digits: Int = 1) -> String {
        guard let v else { return dash }
        return v.formatted(.number.precision(.fractionLength(digits)))
    }

    static func multiple(_ v: Double?) -> String {
        guard let v else { return dash }
        return v.formatted(.number.precision(.fractionLength(1))) + "x"
    }

    /// 2.9e12 -> "$2.9T".
    static func cap(_ v: Double?) -> String {
        guard let v else { return dash }
        let a = abs(v)
        let (d, s): (Double, String) = a >= 1e12 ? (1e12, "T") : a >= 1e9 ? (1e9, "B") : a >= 1e6 ? (1e6, "M") : (1, "")
        return "$" + (v / d).formatted(.number.precision(.fractionLength(1))) + s
    }

    /// A gate's raw value by its `fmt` from gates.json (pct1 / ratio / int).
    static func gate(_ v: Double?, fmt: String?) -> String {
        switch fmt {
        case let f? where f.hasPrefix("pct"): return pct(v, digits: Int(f.dropFirst(3)) ?? 1)
        case "ratio": return num(v, digits: 2)
        case "int": return num(v, digits: 0)
        default: return num(v, digits: 2)
        }
    }
}
