# Plan: Four Model Improvements

## 1. Sector-Relative Scoring for Gross Margin & Accruals

**Problem:** Gross Margin > 40% gate fails banks, utilities, retailers regardless of quality (52% pass). Accruals < 5% is the harshest gate (58% pass).

**Approach:** Add sector-relative percentile scoring. Keep softened absolute thresholds for screening (pass/fail), but score relative to sector peers.

### Changes:

**`scripts/scoring.py`:**
- Add `is_sector_relative` as a 6th element to SCORING_GATES tuples
- Soften SCREENING_GATES thresholds:
  - Gross Margin: `v > 0.40` → `v > 0.25` (still filters commodity businesses)
  - Accruals: `abs(v) < 0.05` → `abs(v) < 0.08` (less harsh)
- Change SCORING_GATES for both to use sector percentile:
  - Gross Margin: `is_sector_relative=True`, score = percentile within sector
  - Accruals: `is_sector_relative=True`, score = percentile within sector (lower is better, so invert)
- Modify `compute_continuous_scores()` to support sector-relative percentile computation:
  - Group values by `r.get('sector')`, compute percentile rank within each sector
  - Fallback to global percentile if sector has < 5 stocks

**`scripts/config.py`:**
- Add `MIN_SECTOR_STOCKS_SCORING = 5` constant

---

## 2. Replace Rubber-Stamp Gates

**Problem:** Cash Conv (95% pass) and Analyst LTG (95% pass) barely discriminate.

### Cash Conv → Interest Coverage

**Why:** Interest Coverage tests debt-servicing ability — a real stress test. Only ~60-70% of stocks will pass > 3×, providing better discrimination.

**`scripts/scoring.py`:**
- SCREENING_GATES: Replace `('Quality: Cash Conv', 'cash_conv', ...)` with `('Quality: Int Coverage', 'int_cov', lambda v, r: v > 3.0 if v is not None else None)`
- SCORING_GATES: Replace cash_conv entry with `('Quality: Int Coverage', 'int_cov', 'Quality', lambda v, r, pct: _score_linear(min(v, 20), 1.0, 10.0), False)` — cap at 20 to avoid outlier distortion

**`scripts/report_html.py`:**
- Replace `_gate_cash_conv` / `_gp_cash_conv` / `_score_cash_conv` with `_gate_int_coverage` / `_gp_int_coverage` / `_score_int_coverage`
- Update gate_meta: label='Int Cov', threshold='IC > 3×', fmt='ratio'

**`scripts/report_excel.py`:**
- Replace gate rows, column widths, threshold labels

**`templates/report.html`:**
- Update TT tooltip for `int_coverage` gate

### Analyst LTG → Fundamental Growth

**Why:** Fundamental Growth = ROIC × Reinvestment Rate. Grounded in actual capital efficiency, not analyst optimism. More discriminating.

**`scripts/scoring.py`:**
- SCREENING_GATES: Replace `('Growth: Analyst LTG', ...)` with `('Growth: Fund. Growth', 'fundamental_growth', lambda v, r: v > 0.03 if v is not None else None)`
- SCORING_GATES: Replace analyst_ltg entry with `('Growth: Fund. Growth', 'fundamental_growth', 'Growth', lambda v, r, pct: _score_linear(v, 0.0, 0.15), False)` — no longer relative

**Same pattern for report_html.py, report_excel.py, report.html** — replace all analyst_ltg gate references with fund._growth equivalents.

---

## 3. Shareholder Yield (New Quality Gate)

**Problem:** No gate captures capital return to shareholders (dividends + buybacks).

**Approach:** Compute total shareholder yield from yfinance cash flow data, add as a Quality gate.

### Changes:

**`scripts/analyze_stock.py`:**
- Add `_compute_shareholder_yield(yf_data, mcap)` function:
  - Dividend: `abs(cash_flow['Common Stock Dividend Paid'])` (latest year)
  - Buyback: `abs(cash_flow['Repurchase Of Capital Stock'])` (latest year)
  - Total yield = (dividend + buyback) / market_cap
  - Return the yield as a decimal (e.g. 0.04 = 4%)
- Call it in the per-ticker analysis block, store as `'shareholder_yield'` in result dict

**`scripts/scoring.py`:**
- SCREENING_GATES: Add `('Quality: Shrhldr Yield', 'shareholder_yield', lambda v, r: v > 0.0 if v is not None else None)`
- SCORING_GATES: Add `('Quality: Shrhldr Yield', 'shareholder_yield', 'Quality', lambda v, r, pct: _score_linear(v, -0.02, 0.06), False)`

**`scripts/report_html.py`:**
- Add `shareholder_yield`, `_gate_shrhldr_yield`, `_gp_shrhldr_yield`, `_score_shrhldr_yield` to chart_data
- Add gate_meta entry: label='Shrhldr Yld', threshold='Yield > 0%', category='Quality', fmt='pct1'

**`scripts/report_excel.py`:**
- Add gate rows, column widths, threshold labels

**`templates/report.html`:**
- Add TT tooltip for `shrhldr_yield` and `shareholder_yield`
- Add to Overview and relevant table tabs

---

## 4. Weight Calibration via Grid Search

**Problem:** 30/25/25/20 weights are heuristic. We have quality/poor labels to optimize against.

**Approach:** Grid search over weight combinations, maximize Cohen's d between quality and poor groups.

### Changes:

**`scripts/calibrate.py`:**
- Add `optimize_weights(results)` function:
  - Generate weight grid: all combos of (0.10, 0.15, 0.20, 0.25, 0.30, 0.35) that sum to 1.0
  - For each weight set: recompute composite scores, compute Cohen's d between quality vs poor
  - Return best weights and Cohen's d
- Run against latest results JSON

**`scripts/config.py`:**
- Update the 4 weight constants with calibrated values

---

## Execution Order

1. Sector-relative scoring (scoring.py changes)
2. Replace rubber-stamp gates (scoring.py + report files)
3. Shareholder yield (analyze_stock.py + scoring.py + report files)
4. Weight calibration (calibrate.py, then config.py)
5. Run full model and rebuild reports
6. Compare Cohen's d before vs after
