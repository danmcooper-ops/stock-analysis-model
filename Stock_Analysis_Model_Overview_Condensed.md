# Stock Analysis Model — Overview

A quantitative framework that screens the entire US equity market daily and ranks companies on a value-investing framework (Graham / Buffett / Munger), producing a **BUY / LEAN BUY / HOLD / PASS** rating for each name with a full valuation, quality, and moat breakdown. *Latest run (2026-06-19): 2,193 companies analyzed from ~9,100 tickers.*

**Daily pipeline:** Universe (~9,000+ US tickers from SEC EDGAR) → **Phase 1 screen** (market cap ≥ $300M; ROIC > WACC, i.e. positive economic spread) → **Phase 2 deep analysis** (full valuation, scoring, rating on survivors) → **sector enrichment** → outputs (interactive HTML, Excel, daily JSON snapshot archived to a history branch).

### Valuation engine — each company valued by several independent models, cross-checked

| Model | Captures | | Model | Captures |
|---|---|---|---|---|
| **DCF** | Owner-earnings intrinsic value, Monte-Carlo on inputs | | **DDM** | Fair value for dividend payers |
| **RIM** | Book value + returns above cost of capital | | **CAPM** | Cost of equity / WACC hurdle |
| **EPV** | Graham/Greenwald zero-growth floor | | **Multiples / NAV** | P/FV, P/FCF, P/TBV; NAV for REITs |

### Scoring — 0–100 composite across five weighted categories

| Category | Weight | Measures | Rating | Composite |
|---|:--:|---|---|:--:|
| **Moat** | **40%** | ROIC spread vs WACC, ROIC consistency & trend, gross/FCF margins | **BUY** | ≥ 60 |
| **Valuation** | **20%** | Margin of safety vs DCF, P/FV, P/FCF, P/TBV, EPV floor, RIM MoS | **LEAN BUY** | ≥ 43 |
| **Quality** | **20%** | Interest coverage, net debt/EBITDA, accruals, cash conv, ROE, Piotroski | **HOLD** | ≥ 29 |
| **Growth** | **10%** | Revenue & FCF durability, margin trend, fundamental growth | **PASS** | < 29 |
| **Ownership** | **10%** | Shareholder yield, buybacks, share-count direction | | |

Every company is also graded against **26 pass/fail gates** (same five categories) for a fast "tests cleared" read, and a **Monte-Carlo uncertainty penalty** trims the score when fair-value estimates are wide. The model is deliberately **value-oriented** — it rewards quality businesses trading *below* intrinsic value, so picks skew toward out-of-favor laggards (validation confirms an intended *negative* correlation between score and trailing 12-month return).

### Data sources & sector-aware KPIs

**Sources:** SEC EDGAR (filings, XBRL, insider, legal) · yfinance (prices) · FDIC BankFind (bank metrics) · ClinicalTrials.gov (pipeline) · Finnhub / Tiingo / FMP (fundamentals, news).

Generic ratios can't fairly compare across industries, so sector-specific metrics are layered in:

- **Banks** — NIM, efficiency ratio, CET1, NPL, deposit beta
- **REITs** — FFO growth, AFFO margin, debt-maturity wall, NAV
- **Technology** — R&D intensity, SBC % of revenue, FCF margin ex-SBC, net cash, Rule of 40
- **Healthcare** — R&D intensity, FDA pipeline depth (active trials)
- **Industrials** — capex intensity, backlog/revenue, book-to-bill
- **Consumer** — inventory days, cash-conversion cycle, brand spend
- **Energy / Utilities / Materials** — capex-to-D&A reinvestment discipline · **Insurers** — combined ratio, cost of float

**Output:** interactive HTML report (sortable tables, per-company deep-dives, sector analysis, multi-stock charts, personal watchlist), Excel workbook, and daily JSON snapshots that accumulate into a longitudinal dataset for ongoing calibration. Fully automated, runs end-of-day.

---
*Not investment advice. A quantitative research/educational tool; ratings are model outputs from historical and estimated data that may contain errors. Do your own research and consult a licensed professional before investing.*
