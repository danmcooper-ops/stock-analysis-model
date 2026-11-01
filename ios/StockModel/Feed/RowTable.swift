import Foundation

/// Column-wise rows, decoded from `app/rows.json` (scripts/report_html.py
/// `pack_rows`; Python mirror `unpack_rows`, JS mirror `_unpackRows`).
///
/// Shape: `n` rows; `k[i]` is a key; `c[i]` holds that key's values for the
/// rows that have it, in row order. Those rows are `p[i]` (present indices)
/// or all rows except `a[i]` (absent indices), or all rows when neither is
/// given. An absent key stays absent; an explicit null stays null.
///
/// It stays columnar: ~4.4k rows x ~300 keys as dictionaries would cost far
/// more memory and decode time than one array per key.
struct RowTable: Sendable {
    let count: Int
    let keys: [String]
    /// `cells[keyIndex][row]`: nil = key absent on that row.
    private let cells: [[JSONValue?]]
    private let keyIndex: [String: Int]

    struct Packed: Decodable {
        let n: Int
        let k: [String]
        let c: [[JSONValue]]
        let p: [String: [Int]]
        let a: [String: [Int]]
    }

    enum DecodeError: Error, Equatable {
        case lengthMismatch(key: String, expected: Int, got: Int)
        case indexOutOfRange(key: String, index: Int)
    }

    init(data: Data) throws {
        try self.init(packed: JSONDecoder().decode(Packed.self, from: data))
    }

    init(packed: Packed) throws {
        count = packed.n
        keys = packed.k
        var idx: [String: Int] = [:]
        var cols: [[JSONValue?]] = []
        cols.reserveCapacity(packed.k.count)
        for (i, key) in packed.k.enumerated() {
            idx[key] = i
            let values = packed.c[i]
            var col = [JSONValue?](repeating: nil, count: packed.n)
            let rows: [Int]
            if let present = packed.p[String(i)] {
                rows = present
            } else if let absent = packed.a[String(i)] {
                let miss = Set(absent)
                rows = (0..<packed.n).filter { !miss.contains($0) }
            } else {
                rows = Array(0..<packed.n)
            }
            guard rows.count == values.count else {
                throw DecodeError.lengthMismatch(key: key, expected: rows.count, got: values.count)
            }
            for (j, r) in rows.enumerated() {
                guard r >= 0 && r < packed.n else { throw DecodeError.indexOutOfRange(key: key, index: r) }
                col[r] = values[j]
            }
            cols.append(col)
        }
        cells = cols
        keyIndex = idx
    }

    /// The value of `key` on `row`; nil when the row lacks the key.
    func value(_ key: String, row: Int) -> JSONValue? {
        guard let i = keyIndex[key], row >= 0, row < count else { return nil }
        return cells[i][row]
    }

    func has(_ key: String, row: Int) -> Bool { value(key, row: row) != nil }

    /// One row as a dictionary: for tests and debugging, not for lists.
    func dictionary(row: Int) -> [String: JSONValue] {
        var out: [String: JSONValue] = [:]
        for (i, key) in keys.enumerated() {
            if let v = cells[i][row] { out[key] = v }
        }
        return out
    }
}
