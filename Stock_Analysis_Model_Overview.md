# Stock Analysis Model — Overview

A quantitative framework that screens the entire US equity market every day and
ranks companies on a disciplined value-investing framework (Graham / Buffett /
Munger). For each company it produces a **BUY / LEAN BUY / HOLD / PASS** rating
backed by a full valuation, quality, and moat breakdown.

---

## How it works — the daily pipeline

1. **Universe.** Pulls the full list of US-listed equities (~9,000+ tickers) from
   SEC EDGAR.
2. **Phase 1 screen.** Two cheap filters run before any expensive analysis:
   - **Market cap ≥ $300M** — drops micro-caps and shells that can't be
     meaningfully valued.
   - **ROIC > WACC** — keeps only businesses earning a *positive economic
     spread* (they create value rather than destroy it).
3. **Phase 2 deep analysis.** Every survivor gets a full multi-model valuation,
   scoring, and rating.
4. **Sector enrichment.** Sector-specific KPIs are layered on so a bank isn't
   judged by the same yardstick as a software company.
5. **Outputs.** An interactive HTML report, an Excel workbook, and a JSON
   snapshot — the snapshot is archived every day so a full historical time
   series accumulates.

> **Latest run (2026-06-19):** 2,193 companies fully analyzed from a starting
> universe of ~9,100 tickers.

---

## Valuation engine

Each company is valued with several **independent** models, then cross-checked
against one another for confidence:

| Model | What it captures |
|---|---|
| **DCF** (discounted cash flow) | Owner-earnings intrinsic value, with Monte-Carlo simulation on key inputs |
| **RIM** (residual income model) | Book value plus the value of returns above the cost of capital |
| **EPV** (earnings power value) | Graham/Greenwald zero-growth "floor" value |
| **DDM** (dividend discount / H-model) | Fair value for established dividend payers |
| **CAPM** | Cost of equity and the WACC hurdle rate |
| **Relative multiples** | P/Fair-Value, P/FCF, P/Tangible-Book vs. peers |
| **NAV** (net asset value) | Asset-based value for REITs |

---

## Scoring framework

A **0–100 composite score** blends five weighted categories:

| Category | Weight | What it measures |
|---|:--:|---|
| **Moat** | **40%** | ROIC spread vs. WACC, ROIC consistency & trend, gross & free-cash-flow margins |
| **Valuation** | **20%** | Margin of safety vs. DCF, P/FV, P/FCF, P/TBV, EPV floor, RIM margin of safety |
| **Quality** | **20%** | Interest coverage, net debt/EBITDA, accruals, cash conversion, ROE, Piotroski F-score |
| **Growth** | **10%** | Revenue & free-cash-flow durability, margin trend, fundamental growth |
| **Ownership** | **10%** | Shareholder yield, buyback rate, share-count direction (dilution vs. shrink) |

Alongside the score, every company is graded against **26 pass/fail gates**
spanning the same five categories — a fast "how many quality tests does it
clear" read (e.g. *26/26*). A **Monte-Carlo uncertainty penalty** trims the
composite when the fair-value estimate is wide, so speculative valuations can't
score as confidently as well-supported ones.

### Rating bands

| Rating | Composite score |
|---|:--:|
| **BUY** | ≥ 60 |
| **LEAN BUY** | ≥ 43 |
| **HOLD** | ≥ 29 |
| **PASS** | < 29 |

The model is deliberately **value-oriented**: it rewards high-quality businesses
trading *below* intrinsic value, so its top picks tend to be out-of-favor
laggards rather than momentum darlings. Daily validation confirms a *negative*
correlation between composite score and trailing 12-month return — which is the
expected, intended behavior for a value strategy.

---

## Data sources

- **SEC EDGAR** — filings, XBRL financial statements, insider transactions, legal disclosures
- **yfinance** — prices and market data
- **FDIC BankFind** — bank call-report metrics
- **ClinicalTrials.gov** — drug/biotech pipeline depth
- **Finnhub / Tiingo / FMP** — supplementary fundamentals and news

---

## Sector-aware KPIs

Generic ratios don't fairly compare across industries, so the model layers in
metrics that actually distinguish leaders within each sector:

- **Banks** — net interest margin, efficiency ratio, CET1 capital, non-performing loans, deposit beta
- **REITs** — FFO growth, AFFO margin, debt-maturity wall, net asset value
- **Technology** — R&D intensity, stock-based comp as % of revenue, FCF margin ex-SBC, net cash, Rule of 40
- **Healthcare** — R&D intensity, FDA pipeline depth (active clinical trials)
- **Industrials** — capex intensity, backlog-to-revenue, book-to-bill
- **Consumer** — inventory days, cash-conversion cycle, brand/advertising spend
- **Energy / Utilities / Materials** — capex-to-D&A reinvestment discipline
- **Insurers** — combined ratio, cost of float

---

## Outputs & cadence

- **Interactive HTML report** — sortable screening tables, per-company deep-dive
  panels, sector analysis, multi-stock price charts, and a personal
  flagging/watchlist feature.
- **Excel workbook + daily JSON snapshot** — the snapshot is committed to a
  history branch each day, building a longitudinal dataset over time.
- **Fully automated**, runs end-of-day.

---

## Roadmap

The accumulating daily snapshots are intended to train and calibrate the model
over time, sharpening it toward actionable, real-money BUY / HOLD / PASS
recommendations.

---

*Disclaimer: This is a quantitative research and educational tool. It is not
investment advice, a recommendation, or an offer to buy or sell any security.
Ratings are model outputs derived from historical and estimated data, which may
contain errors or omissions. Always do your own research and consult a licensed
financial professional before making investment decisions.*
