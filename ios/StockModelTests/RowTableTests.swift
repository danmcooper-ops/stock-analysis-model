import XCTest
@testable import StockModel

/// The Swift decoder must rebuild exactly what scripts/report_html.py
/// `unpack_rows` does. `packed` and `expected` were produced by the Python
/// side (pack_rows / unpack_rows of the same four rows); regenerate both if
/// the packing changes (tests/test_app_feed.py pins its shape).
final class RowTableTests: XCTestCase {
    let packed = #"""
    {"n":4,"k":["ticker","price","rating","pe","pf","only_a","_rating_cap_reasons","rating_prev"],"c":[["AAA","BBB","CCC","DDD"],[123.45,2.5,null],["BUY","HOLD","PASS","LEAN BUY"],[null,14.2,7],[["semis"],[],[],["semis"]],[1],[["Beneish"]],["BUY","PASS","HOLD"]],"p":{"5":[0],"6":[0]},"a":{"1":[2],"3":[3],"7":[0]}}
    """#

    let expected = #"""
    [{"ticker":"AAA","price":123.45,"rating":"BUY","pe":null,"pf":["semis"],"only_a":1,"_rating_cap_reasons":["Beneish"]},{"ticker":"BBB","price":2.5,"rating":"HOLD","pe":14.2,"pf":[],"rating_prev":"BUY"},{"ticker":"CCC","rating":"PASS","pe":7,"pf":[],"rating_prev":"PASS"},{"ticker":"DDD","price":null,"rating":"LEAN BUY","pf":["semis"],"rating_prev":"HOLD"}]
    """#

    func testMatchesPythonUnpack() throws {
        let table = try RowTable(data: Data(packed.utf8))
        let want = try JSONDecoder().decode([[String: JSONValue]].self, from: Data(expected.utf8))
        XCTAssertEqual(table.count, want.count)
        for i in 0..<table.count {
            XCTAssertEqual(table.dictionary(row: i), want[i], "row \(i)")
        }
    }

    func testAbsentIsNotNull() throws {
        let table = try RowTable(data: Data(packed.utf8))
        XCTAssertNil(table.value("price", row: 2))                     // absent
        XCTAssertEqual(table.value("price", row: 3), .null)            // explicit null
        XCTAssertNil(table.value("rating_prev", row: 0))
        XCTAssertNil(table.value("no_such_key", row: 0))
    }

    func testStockRowAccessorsKeepNA() throws {
        let rows = try RowTable(data: Data(packed.utf8)).rows
        XCTAssertEqual(rows[0].rating, .buy)
        XCTAssertNil(rows[0].pe)                  // null is N/A, never 0
        XCTAssertNil(rows[2].price)               // absent is N/A
        XCTAssertEqual(rows[0].flags, ["Beneish"])
        XCTAssertEqual(rows[3].portfolios, ["semis"])
        XCTAssertTrue(rows[1].ratingChanged)      // BUY -> HOLD
        XCTAssertTrue(rows[1].crossedBuyLine)
        XCTAssertFalse(rows[2].ratingChanged)     // PASS -> PASS
        XCTAssertTrue(rows[3].ratingChanged)      // HOLD -> LEAN BUY
        XCTAssertTrue(rows[3].crossedBuyLine)
    }

    func testRejectsMisalignedColumns() {
        let bad = #"{"n":2,"k":["a"],"c":[[1]],"p":{},"a":{}}"#
        XCTAssertThrowsError(try RowTable(data: Data(bad.utf8)))
    }

    func testManifestStaleness() throws {
        let json = #"{"version":1,"run_date":"2026-10-31","generated_at":"2026-11-01T06:00:00+00:00","build":"abc","rows":4,"files":{}}"#
        let m = try JSONDecoder().decode(Manifest.self, from: Data(json.utf8))
        let t0 = ISO8601DateFormatter().date(from: "2026-11-01T06:00:00Z")!
        XCTAssertFalse(m.isStale(now: t0.addingTimeInterval(3600)))
        XCTAssertTrue(m.isStale(now: t0.addingTimeInterval(27 * 3600)))
    }

    func testFormattersShowDashForNA() {
        XCTAssertEqual(Fmt.price(nil), "—")
        XCTAssertEqual(Fmt.pct(nil), "—")
        XCTAssertEqual(Fmt.cap(nil), "—")
    }
}
