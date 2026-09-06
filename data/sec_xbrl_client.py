"""SEC EDGAR XBRL Company Facts client.

Fetches structured financial data from the SEC XBRL API to:
1. Cross-validate yfinance financial statements (flag discrepancies > 5%)
2. Provide 10+ years of revenue/earnings history for extended CAGR analysis

Free API — no authentication required, just User-Agent with email.
Uses only stdlib (urllib + json + datetime).
"""

import gzip
import json
import logging
import ssl
import time
import urllib.error
import urllib.request
from datetime import date as _date
from datetime import datetime as _dt
from datetime import timedelta as _timedelta

from data.sec_facts_cache import SECFactsCache, _norm_cik
from data.throttle import Throttle

logger = logging.getLogger(__name__)


class _Absent:
    """Sentinel: SEC answered that the resource does not exist (403/404).

    Distinct from a failed request. The daily filing index is absent on
    weekends and holidays, which genuinely means 'nothing was filed'; a
    timeout or 5xx means 'unknown', and the two must not be confused or a
    sweep would advance its watermark past filings it never read.
    """

    def __repr__(self):
        return '<absent>'


ABSENT = _Absent()

def _ssl_context():
    """Return an SSL context using certifi certs if available, else system certs."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception as e:
        logger.debug(f"sec_xbrl: certifi unavailable, using system trust store: {e}")
        # Fall back to the system trust store — NEVER disable verification:
        # unverified TLS would let a MITM feed fabricated financial data
        # into the pipeline silently.
        return ssl.create_default_context()

_SSL_CTX = _ssl_context()

from models.field_keys import (
    _get, REVENUE_KEYS, NET_INCOME_KEYS, TOTAL_ASSETS_KEYS,
    EQUITY_KEYS, DEBT_KEYS, OPERATING_CF_KEYS, GROSS_PROFIT_KEYS, INTEREST_KEYS,
)

# FX helpers moved to data/fx_client.py so the yfinance live-data pipeline
# can share them. Re-exported here so the existing call sites below (line
# ~787) and any external imports continue to work unchanged.
from data.fx_client import (  # noqa: F401
    _FX_CACHE,
    _get_fx_rates_to_usd,
    _apply_fx_annual,
)


class SECXBRLClient:
    """Fetch and interpret XBRL Company Facts from SEC EDGAR."""

    _COMPANY_FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    # Daily list of everything disseminated on a date, pipe-delimited as
    # CIK|Company Name|Form Type|Date Filed|File Name. Drives cache
    # invalidation: companyfacts carries no ETag/Last-Modified, so this
    # is the only way to know a blob changed without refetching it.
    _DAILY_INDEX_URL = ('https://www.sec.gov/Archives/edgar/daily-index/'
                        '{year}/QTR{quarter}/master.{yyyymmdd}.idx')

    # Form types that revise the XBRL facts this client reads, with their /A
    # amendments (matched on the base form, so 10-K/A counts as 10-K).
    #
    # Derived by enumerating the `form` field on every fact in the
    # companyfacts blobs of a spread of filers rather than from the filing
    # rules, because the two disagree. Measured contributions of statement
    # (us-gaap / ifrs-full) facts:
    #   10-K, 10-Q          domestic periodic reports — the bulk
    #   20-F, 40-F          foreign annual reports
    #   6-K                 substantial for foreign issuers (Shell 3.5k facts,
    #                       BTI 3.9k, ORIX 5.8k) — NOT skippable, even though
    #                       an individual recent 6-K often has not landed yet
    #   8-K                 real but largely duplicative (AAPL 337 client-read
    #                       facts, JPM 252): an earnings-release 8-K previews
    #                       figures the following 10-Q restates, so almost no
    #                       datapoint is 8-K-only. Included anyway — it keeps
    #                       the freshest quarter from lagging its 10-Q by days,
    #                       and the whole filter still only evicts ~3% of a
    #                       cache per day (below), so it is nearly free.
    # DEF 14A supplies a handful of stray tagged values (5 for JPM) that the
    # periodic reports also carry; not worth evicting for.
    #
    # Measured cost of the filter as a whole: ~294 filers/day file one of
    # these forms, ~251 of them listed companies — about 3% of SEC's ~8,000
    # listed CIKs, so ~97% of a warm cache survives each run.
    _FACT_BEARING_FORMS = ('10-K', '10-Q', '20-F', '40-F', '6-K', '8-K')

    # Walking more index days than this costs more than it saves, so a
    # longer gap falls back to the cache's own age backstop.
    _MAX_INDEX_SWEEP_DAYS = 30

    # Multiple possible US-GAAP tags per financial concept.
    # Ordered by prevalence — first match wins.
    _XBRL_TAG_MAP = {
        'revenue': [
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'RevenueFromContractWithCustomerIncludingAssessedTax',
            'SalesRevenueNet',
            'SalesRevenueGoodsNet',
            # Pre-ASC 606 (FY2017 and earlier) totals. The 2018 taxonomy
            # retired these in favour of RevenueFromContractWithCustomer*,
            # and companyfacts keeps the old years under the old tags only —
            # so a filer that had used one of them started its revenue
            # history in 2018 and read N/A on 10Y Rev CAGR and Rev
            # Volatility however long it had filed (HCA, ODFL, PLD, VLO, EQT,
            # BAH, MAN, RHI ... 64 filers with 11+ years of balance sheets
            # in the 2026-09-03 snapshot). Values merge per period end, and
            # a tie on filed date keeps the earlier tag, so these can never
            # displace a total tagged above. Only whole-company totals are
            # listed: gross-of-provision and component lines (patient
            # service revenue before bad debts, lease revenue, food and
            # beverage, advertising, home building) are deliberately absent
            # because a component stitched onto a later total fabricates a
            # growth step at the tag boundary.
            'SalesRevenueServicesNet',
            'RealEstateRevenueNet',
            'HealthCareOrganizationRevenue',
            'HealthCareOrganizationPatientServiceRevenueLessProvisionForBadDebts',
            'RegulatedAndUnregulatedOperatingRevenue',
            'ElectricUtilityRevenue',
            'RegulatedOperatingRevenue',
            'RegulatedOperatingRevenueGas',
            'OilAndGasRevenue',
            'OilAndGasSalesRevenue',
            'RefiningAndMarketingRevenue',
            'RevenueOilAndGasServices',
            'ContractsRevenue',
            'RevenueFromContractWithCustomerExcludingAssessedTaxTransferredAtAPointInTime',
        ],
        # Bank revenue components. Banks tag no revenue total: their
        # ASC 606 line is fee income only (BKU: $22M against $1.1B of net
        # revenue), and the standard "total revenue" a bank reports — and
        # yfinance's totalRevenue — is net interest income + noninterest
        # income. _resolve_revenue_annual sums these two where both are
        # tagged; JPM, COF and WFC, which tag Revenues as well, show the
        # sum equals that line. Insurers, brokers and asset managers do not
        # tag NoninterestIncome, so the derivation cannot fire on them.
        'net_interest_income': ['InterestIncomeExpenseNet'],
        'noninterest_income': ['NoninterestIncome'],
        'net_income': [
            'NetIncomeLoss',
            'NetIncomeLossAvailableToCommonStockholdersBasic',
            'ProfitLoss',
        ],
        'total_assets': [
            'Assets',
        ],
        'total_equity': [
            'StockholdersEquity',
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        ],
        # Split views of the same line, consumed by _resolve_equity_annual:
        # parent-attributable equity is what yfinance's 'Stockholders Equity'
        # row (and every per-share model) means. A filer that only tags the
        # NCI-inclusive total gets minority interest subtracted back out.
        'equity_parent': ['StockholdersEquity'],
        'equity_incl_nci': [
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
        ],
        # Debt components are extracted SEPARATELY (not alias-merged) and
        # combined by _resolve_total_debt_annual: US-GAAP LongTermDebt
        # includes current maturities while LongTermDebtNoncurrent does not,
        # so merging them lets a current-inclusive total win the merge and
        # then get the current portion added a second time.
        'ltd_noncurrent': [
            'LongTermDebtNoncurrent',
            'LongTermDebtAndCapitalLeaseObligations',
        ],
        'ltd_total': [
            'LongTermDebt',
        ],
        'debt_current_total': [
            'DebtCurrent',          # short-term borrowings + current LTD
        ],
        'ltd_current': [
            'LongTermDebtCurrent',
        ],
        'st_borrowings': [
            'ShortTermBorrowings',
            'CommercialPaper',
            'BankOverdrafts',
        ],
        'total_liabilities': [
            'Liabilities',
        ],
        'retained_earnings': [
            'RetainedEarningsAccumulatedDeficit',
        ],
        'operating_income': [
            'OperatingIncomeLoss',
        ],
        'cash': [
            'CashAndCashEquivalentsAtCarryingValue',
        ],
        # The securities-inclusive total on its own (never alias-merged with
        # 'cash'): build_yfinance_shape composes the yfinance
        # 'Cash Cash Equivalents And Short Term Investments' row from it, or
        # from cash + st_investments when the filer tags the pieces.
        'cash_incl_sti': [
            'CashCashEquivalentsAndShortTermInvestments',
        ],
        # Claims ahead of common equity in the EV→equity bridge.
        # (Short-term investments: see the existing 'st_investments' concept.)
        'preferred_stock': [
            'PreferredStockValue',
            'PreferredStockValueOutstanding',
        ],
        'operating_cash_flow': [
            'NetCashProvidedByUsedInOperatingActivities',
            # Filers with discontinued operations (DIS, JCI, DRI, T post-
            # WarnerMedia, HON) tag the continuing-ops subtotal INSTEAD of
            # the total, and most large caps used this element for their
            # FY2014-2017 10-Ks (AAPL, MSFT, CSCO, INTC, TXN, ...). Without
            # it those years are holes in the OCF/FCF history. Listed second
            # so the total wins whenever a filing carries both.
            'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
        ],
        'capex': [
            'PaymentsToAcquirePropertyPlantAndEquipment',
            # Many large domestic filers (LRCX, AOS, ...) report capex
            # under the "ProductiveAssets" line rather than PP&E.
            'PaymentsToAcquireProductiveAssets',
            'PaymentsToAcquireOtherProductiveAssets',
            # LLY ($7.8B/yr) and ADP tag purchases of PP&E under the
            # "Other" PP&E element only — without it their FCF is None.
            'PaymentsToAcquireOtherPropertyPlantAndEquipment',
            'PaymentsForCapitalImprovements',
            'CapitalExpendituresIncurringObligation',
        ],
        # D&A from the cash-flow statement. Needed by
        # calculate_fundamental_growth (Reinvestment Rate × ROIC) and the
        # owner-earnings growth-capex add-back — both dead on the XBRL path
        # until these rows existed in build_yfinance_shape's frames.
        'd_and_a': [
            'DepreciationDepletionAndAmortization',
            'DepreciationAndAmortization',
            'DepreciationAmortizationAndAccretionNet',
            'Depreciation',
        ],
        # Working-capital endpoints for the ΔWC term of the reinvestment
        # rate (and the current ratio, which was N/A for XBRL records).
        'current_assets': [
            'AssetsCurrent',
        ],
        'current_liabilities': [
            'LiabilitiesCurrent',
        ],
        'gross_profit': [
            'GrossProfit',
        ],
        'interest_expense': [
            # The 2024 US-GAAP taxonomy deprecated InterestExpense in favour
            # of InterestExpenseNonoperating; MSFT, JNJ, AMZN, KO and others
            # switched for FY2024+. Without this tag the latest-year row is
            # NaN and calculate_wacc / interest coverage fall back silently.
            'InterestExpenseNonoperating',
            'InterestExpense',
            # The 2024 US-GAAP taxonomy retired InterestExpense in favour of
            # the operating/non-operating split; VZ, T, MSFT, HD, NVDA, ...
            # now tag only this element, leaving the latest year blank for
            # WACC cost of debt, interest coverage and the FCFF add-back.
            'InterestExpenseNonoperating',
            'InterestAndDebtExpense',
            'InterestExpenseDebt',
        ],
        'dividends_paid': [
            'PaymentsOfDividendsCommonStock',
            'PaymentsOfDividends',
            'DividendsCommonStockCash',
        ],
        'shares_outstanding': [
            'CommonStockSharesOutstanding',
            'CommonStockSharesOutstandingBasic',
        ],
        # Needed by the build_yfinance_shape adapter so calculate_roic /
        # calculate_wacc can derive a real (not default 21%) tax rate.
        'tax_provision': [
            'IncomeTaxExpenseBenefit',
        ],
        'pretax_income': [
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesAndMinorityInterest',
        ],
        # --- GAAP statement presentation (popup statement tabs only) -------
        # Everything below exists so the Income Statement, Balance Sheet and
        # Cash Flow tabs can follow Reg S-X / ASC 220-210-230 ordering rather
        # than listing whatever happened to be extracted. Coverage measured
        # over a 12-filer sample (AAPL/MSFT/JPM/KO/XOM/UNH/PG/NVDA/T/WMT/
        # BRK-B/O); the fraction noted is how many tagged the concept at all.
        # None of these feed scoring — rows simply drop when a filer is
        # silent, which is why partial coverage is acceptable here.
        'cost_of_revenue': [                       # 8/12
            'CostOfRevenue',
            'CostOfGoodsAndServicesSold',
            'CostOfGoodsSold',
            'CostOfServices',
        ],
        'rd_expense': ['ResearchAndDevelopmentExpense'],          # 4/12 (tech/pharma)
        'sga_expense': [                           # 9/12
            'SellingGeneralAndAdministrativeExpense',
            'GeneralAndAdministrativeExpense',
        ],
        'selling_marketing': ['SellingAndMarketingExpense'],      # 2/12
        'total_opex': ['OperatingExpenses', 'CostsAndExpenses'],  # 7/12
        'other_nonop': [                           # 9/12
            'NonoperatingIncomeExpense',
            'OtherNonoperatingIncomeExpense',
        ],
        # As-FILED per-share figures. These are restated by the filer for
        # stock splits, unlike the raw share counts — which is exactly why
        # the statement tab shows these rather than dividing net income by
        # period-end shares (that version stepped 4x at Apple's 2020 split).
        'eps_basic': ['EarningsPerShareBasic'],                   # 10/12
        'eps_diluted': ['EarningsPerShareDiluted'],               # 10/12
        'wavg_basic': ['WeightedAverageNumberOfSharesOutstandingBasic'],    # 11/12
        'wavg_diluted': ['WeightedAverageNumberOfDilutedSharesOutstanding'],# 10/12
        'st_investments': [                        # 6/12
            'ShortTermInvestments',
            'AvailableForSaleSecuritiesDebtSecuritiesCurrent',
            'MarketableSecuritiesCurrent',
        ],
        'receivables': [                           # 8/12
            'AccountsReceivableNetCurrent',
            'ReceivablesNetCurrent',
        ],
        'inventory': ['InventoryNet'],                            # 9/12
        'ppe_net': ['PropertyPlantAndEquipmentNet'],              # 9/12
        'goodwill': ['Goodwill'],                                 # 11/12
        'intangibles': [                           # 8/12
            'IntangibleAssetsNetExcludingGoodwill',
            'FiniteLivedIntangibleAssetsNet',
        ],
        'accounts_payable': [                      # 8/12
            'AccountsPayableCurrent',
            'AccountsPayableAndAccruedLiabilitiesCurrent',
        ],
        # The balance-sheet identity check: assets must equal this.
        'liab_and_equity': ['LiabilitiesAndStockholdersEquity'],  # 11/12
        'cs_apic': [                               # 11/12
            'CommonStocksIncludingAdditionalPaidInCapital',
            'AdditionalPaidInCapital',
            'AdditionalPaidInCapitalCommonStock',
        ],
        'aoci': ['AccumulatedOtherComprehensiveIncomeLossNetOfTax'],  # 11/12
        'treasury_stock': ['TreasuryStockValue', 'TreasuryStockCommonValue'],  # 6/12
        'minority_interest': ['MinorityInterest'],                # 7/12
        # ASC 230 requires the three-section presentation; operating already
        # exists above as 'operating_cash_flow'.
        'investing_cf': ['NetCashProvidedByUsedInInvestingActivities'],   # 11/12
        'financing_cf': ['NetCashProvidedByUsedInFinancingActivities'],   # 11/12
        'sbc_cf': ['ShareBasedCompensation'],                     # 8/12
        'deferred_tax': ['DeferredIncomeTaxExpenseBenefit'],      # 11/12
        'buybacks': ['PaymentsForRepurchaseOfCommonStock'],       # 10/12
        # Gross proceeds from issuing stock — nets against buybacks in
        # _compute_shareholder_yield so net diluters don't read as returners.
        'share_issuance': ['ProceedsFromIssuanceOfCommonStock'],
        'debt_issued': [                           # 8/12
            'ProceedsFromIssuanceOfLongTermDebt',
            'ProceedsFromIssuanceOfDebt',
        ],
        'debt_repaid': [                           # 8/12
            'RepaymentsOfLongTermDebt',
            'RepaymentsOfDebt',
        ],
        'acquisitions': ['PaymentsToAcquireBusinessesNetOfCashAcquired'],  # 10/12
        'net_change_cash': [                       # 11/12
            'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect',
            'CashAndCashEquivalentsPeriodIncreaseDecrease',
        ],
        # ASC 230-10-45-24: filers with foreign operations report the effect
        # of exchange-rate changes on cash as its own reconciling line. Without
        # it the three sections don't foot to the net change for any
        # multinational (KO's FY2025 gap was $321M).
        'fx_effect_cash': [
            'EffectOfExchangeRateOnCashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
            'EffectOfExchangeRateOnCashAndCashEquivalents',
        ],
        # IncreaseDecreaseInOperatingCapital was probed and tagged by only
        # 1/12 filers, so working-capital movement is shown as a derived
        # plug (OCF less the tagged add-backs) rather than a tagged row.
    }

    # IFRS taxonomy tag names. 20-F / 40-F filers (foreign issuers) report under
    # facts['ifrs-full'] instead of facts['us-gaap']. Keys mirror _XBRL_TAG_MAP
    # so the same concept lookups work across both taxonomies.
    _XBRL_TAG_MAP_IFRS = {
        'revenue': [
            'Revenue',
            'RevenueFromContractsWithCustomers',
        ],
        'net_income': [
            'ProfitLoss',
            'ProfitLossAttributableToOwnersOfParent',
        ],
        'total_assets': [
            'Assets',
        ],
        'total_equity': [
            'Equity',
            'EquityAttributableToOwnersOfParent',
        ],
        'equity_parent': ['EquityAttributableToOwnersOfParent'],
        'equity_incl_nci': ['Equity'],
        # Debt components mirror the US-GAAP split (see _XBRL_TAG_MAP).
        # IFRS filers tag borrowings by current/noncurrent classification;
        # 'Borrowings' (undifferentiated total) is the ltd_total-tier
        # fallback for filers that only tag the combined figure.
        'ltd_noncurrent': [
            'NoncurrentBorrowings',
            'BorrowingsNoncurrent',
            'LongtermBorrowings',
        ],
        'ltd_total': [
            'Borrowings',
        ],
        'debt_current_total': [
            'CurrentBorrowings',
            'BorrowingsCurrent',
            'ShorttermBorrowings',
        ],
        'ltd_current': [],
        'st_borrowings': [],
        'total_liabilities': [
            'Liabilities',
        ],
        'retained_earnings': [
            'RetainedEarnings',
        ],
        'operating_income': [
            'ProfitLossFromOperatingActivities',
            'OperatingProfitLoss',
        ],
        'cash': [
            'CashAndCashEquivalents',
        ],
        # IFRS has no standard combined cash + short-term-investments total
        # and preferred capital sits inside IssuedCapital; left untagged.
        'preferred_stock': [],
        'operating_cash_flow': [
            'CashFlowsFromUsedInOperatingActivities',
        ],
        'capex': [
            'PurchaseOfPropertyPlantAndEquipment',
            'PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities',
            # SAP / SONY / many EU filers report CapEx as additions on PPE
            'AdditionsOtherThanThroughBusinessCombinationsPropertyPlantAndEquipment',
            'PaymentsForPropertyPlantAndEquipment',
            # SAP combined PPE + intangibles tag — better than nothing
            'PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets',
            'AcquisitionsThroughBusinessCombinationsPropertyPlantAndEquipment',
        ],
        'd_and_a': [
            'DepreciationAndAmortisationExpense',
            # Cash-flow-statement adjustment line — the more commonly tagged
            # IFRS location for D&A.
            'AdjustmentsForDepreciationAndAmortisationExpense',
            'DepreciationExpense',
        ],
        'current_assets': [
            'CurrentAssets',
        ],
        'current_liabilities': [
            'CurrentLiabilities',
        ],
        'gross_profit': [
            'GrossProfit',
        ],
        'interest_expense': [
            'FinanceCosts',
            'InterestExpense',
        ],
        'dividends_paid': [
            'DividendsPaid',
            'DividendsPaidClassifiedAsFinancingActivities',
            'DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities',
        ],
        'shares_outstanding': [
            'NumberOfSharesOutstanding',
            'NumberOfSharesIssued',
            'WeightedAverageNumberOfOrdinarySharesOutstanding',
        ],
        'tax_provision': [
            'IncomeTaxExpenseContinuingOperations',
        ],
        'pretax_income': [
            'ProfitLossBeforeTax',
        ],
        # --- GAAP statement presentation, IFRS equivalents ----------------
        # Mirrors the block added to _XBRL_TAG_MAP so 20-F/40-F filers get
        # the same statement rows. IFRS presentation differs in places (no
        # AOCI as such — 'OtherReserves' is the closest analogue; equity is
        # 'IssuedCapital' + 'SharePremium'), so these are approximations
        # chosen to populate the same line, not exact GAAP counterparts.
        'cost_of_revenue': ['CostOfSales'],
        'rd_expense': ['ResearchAndDevelopmentExpense'],
        'sga_expense': [
            'SellingGeneralAndAdministrativeExpense',
            'AdministrativeExpense',
        ],
        'selling_marketing': ['DistributionCosts', 'SellingExpense'],
        'total_opex': ['OperatingExpense'],
        'other_nonop': ['OtherOperatingIncomeExpense'],
        'eps_basic': ['BasicEarningsLossPerShare'],
        'eps_diluted': ['DilutedEarningsLossPerShare'],
        'wavg_basic': ['WeightedAverageNumberOfOrdinarySharesOutstanding'],
        'wavg_diluted': ['WeightedAverageNumberOfDilutedOrdinarySharesOutstanding'],
        'st_investments': ['OtherCurrentFinancialAssets'],
        'receivables': ['TradeAndOtherCurrentReceivables'],
        'inventory': ['Inventories'],
        'ppe_net': ['PropertyPlantAndEquipment'],
        'goodwill': ['Goodwill'],
        'intangibles': ['IntangibleAssetsOtherThanGoodwill'],
        'accounts_payable': ['TradeAndOtherCurrentPayables'],
        'liab_and_equity': ['EquityAndLiabilities'],
        'cs_apic': ['IssuedCapital', 'SharePremium'],
        'aoci': ['OtherReserves'],
        'treasury_stock': ['TreasuryShares'],
        'minority_interest': ['NoncontrollingInterests'],
        'investing_cf': ['CashFlowsFromUsedInInvestingActivities'],
        'financing_cf': ['CashFlowsFromUsedInFinancingActivities'],
        'sbc_cf': ['ShareBasedPaymentExpense'],
        'deferred_tax': ['DeferredTaxExpenseIncome'],
        'buybacks': ['PaymentsToAcquireOrRedeemEntitysShares'],
        'share_issuance': ['ProceedsFromIssuingShares'],
        'debt_issued': ['ProceedsFromBorrowings'],
        'debt_repaid': ['RepaymentsOfBorrowings'],
        'acquisitions': [
            'CashFlowsUsedInObtainingControlOfSubsidiariesOrOtherBusinessesClassifiedAsInvestingActivities',
        ],
        'net_change_cash': ['IncreaseDecreaseInCashAndCashEquivalents'],
        'fx_effect_cash': [
            'EffectOfExchangeRateChangesOnCashAndCashEquivalents',
        ],
    }

    _TAXONOMIES = (('us-gaap', _XBRL_TAG_MAP), ('ifrs-full', _XBRL_TAG_MAP_IFRS))

    # Map XBRL concept names to yfinance field_keys lists for cross-validation
    _YF_KEY_MAP = {
        'revenue': REVENUE_KEYS,
        'net_income': NET_INCOME_KEYS,
        'total_assets': TOTAL_ASSETS_KEYS,
        'total_equity': EQUITY_KEYS,
        'total_debt': DEBT_KEYS,
        'operating_cash_flow': OPERATING_CF_KEYS,
        'gross_profit': GROSS_PROFIT_KEYS,
        'interest_expense': INTEREST_KEYS,
    }

    # Which yfinance statement contains each concept
    _YF_STATEMENT_MAP = {
        'revenue': 'income_statement',
        'net_income': 'income_statement',
        'total_assets': 'balance_sheet',
        'total_equity': 'balance_sheet',
        'total_debt': 'balance_sheet',
        'operating_cash_flow': 'cash_flow',
        'gross_profit': 'income_statement',
        'interest_expense': 'income_statement',
    }

    def __init__(self, cik_map, name_map,
                 email='stockanalysis@example.com', request_delay=1.0,
                 facts_cache=None):
        """
        Args:
            cik_map: dict {ticker: zero-padded CIK} from SECLegalClient.
            name_map: dict {ticker: company name} from SECLegalClient.
            email: Contact email for SEC User-Agent header.
            request_delay: Seconds between requests.
            facts_cache: persistent companyfacts cache. A SECFactsCache, a
                directory path, or True for the default location; None (the
                default) keeps the per-process in-memory cache only, which is
                what the tests and one-shot callers want.
        """
        self._ua = f'StockAnalyzer/1.0 ({email})'
        self._throttle = Throttle(request_delay)
        self._cache = {}          # ticker -> raw company facts JSON
        self._cik_map = cik_map
        self._name_map = name_map
        self._facts_cache = self._resolve_facts_cache(facts_cache)

    @staticmethod
    def _resolve_facts_cache(facts_cache):
        if facts_cache is None or facts_cache is False:
            return None
        if facts_cache is True:
            return SECFactsCache()
        if isinstance(facts_cache, str):
            return SECFactsCache(cache_dir=facts_cache)
        return facts_cache

    # ------------------------------------------------------------------
    # Throttling & requests
    # ------------------------------------------------------------------

    def _request_bytes(self, url, timeout=20, absent_codes=()):
        """GET returning the decoded body, or None on failure.

        Asks for gzip: companyfacts blobs are ~14x smaller on the wire that
        way (a 3.9 MB blob ships as ~270 KB), and urllib sends no
        Accept-Encoding of its own, so without this every fetch pulls the
        full uncompressed JSON.

        *absent_codes* are HTTP statuses that mean "no such resource" for the
        caller rather than a failure; they return :data:`ABSENT` and are not
        logged. Everything else that goes wrong returns None.
        """
        self._throttle()
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': self._ua, 'Accept-Encoding': 'gzip'})
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=timeout) as resp:
                raw = resp.read()
                if resp.headers.get('Content-Encoding') == 'gzip':
                    raw = gzip.decompress(raw)
                return raw
        except urllib.error.HTTPError as e:
            if e.code in absent_codes:
                return ABSENT
            if e.code == 429:
                time.sleep(5)
            return None
        except Exception as e:
            logger.warning(f"SEC XBRL: request failed for {url}: {e}")
            return None

    def _request_json(self, url, timeout=20):
        """GET request returning parsed JSON, or None on failure."""
        raw = self._request_bytes(url, timeout=timeout)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError as e:
            logger.warning(f"SEC XBRL: bad JSON from {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # Core data extraction
    # ------------------------------------------------------------------

    def fetch_company_facts(self, ticker):
        """Fetch full XBRL company facts JSON from SEC EDGAR.

        Looks in the per-process dict, then the persistent cache when one is
        configured, and only then the network — so a run re-downloads a blob
        only when its filer has filed since it was stored (see
        :meth:`refresh_stale_facts`).

        Returns:
            dict: The raw companyfacts JSON, or None on failure.
        """
        if ticker in self._cache:
            return self._cache[ticker]

        cik = self._cik_map.get(ticker)
        if not cik:
            self._cache[ticker] = None
            return None

        if self._facts_cache is not None:
            cached = self._facts_cache.get(cik)
            if cached is not None:
                self._cache[ticker] = cached
                return cached

        url = self._COMPANY_FACTS_URL.format(cik=cik)
        data = self._request_json(url)
        # Don't cache request failures: a transient timeout would otherwise
        # read as "this ticker has no XBRL data" for the rest of the run.
        if data is not None:
            self._cache[ticker] = data
            if self._facts_cache is not None:
                self._facts_cache.put(cik, data)
        return data

    # ------------------------------------------------------------------
    # Persistent-cache freshness
    # ------------------------------------------------------------------

    def _daily_index_ciks(self, day):
        """CIKs with a fact-bearing filing disseminated on *day*.

        Returns a set when the index was read, an empty set when SEC says
        there is no index for that day (weekends, holidays — nothing was
        filed), or None when the request failed for any other reason. The
        caller must not advance its watermark past a None, or that day's
        filings are never seen.
        """
        url = self._DAILY_INDEX_URL.format(
            year=day.year, quarter=(day.month - 1) // 3 + 1,
            yyyymmdd=day.strftime('%Y%m%d'))
        raw = self._request_bytes(url, absent_codes=(403, 404))
        if raw is ABSENT:
            return set()
        if raw is None:
            return None
        ciks = set()
        for line in raw.decode('latin-1').splitlines():
            parts = line.split('|')
            if len(parts) != 5:
                continue
            form = parts[2].strip().upper()
            # "10-K/A" and friends count; the base form is before the slash.
            if form.split('/')[0] not in self._FACT_BEARING_FORMS:
                continue
            norm = _norm_cik(parts[0])
            if norm:
                ciks.add(norm)
        return ciks

    def refresh_stale_facts(self, today=None, max_days=None):
        """Evict cached companyfacts whose filers have filed since the last sweep.

        companyfacts carries no ETag or Last-Modified, so there is no cheap way
        to ask whether one blob changed. SEC's daily index answers the question
        for every filer at once: one ~0.4 MB request per calendar day lists
        everything disseminated, and only the fact-bearing forms matter.

        Days are walked from the last recorded sweep up to yesterday (today's
        index is incomplete while the day is still running, and is picked up on
        the next run). A gap longer than *max_days* is not walked — the cache's
        own age backstop covers that case instead.

        Returns a summary dict; never raises, since a failed sweep must degrade
        to serving slightly staler facts rather than break the run.
        """
        cache = self._facts_cache
        if cache is None:
            return {'swept': 0, 'invalidated': 0, 'reason': 'no persistent cache'}
        today = today or _date.today()
        through = today - _timedelta(days=1)
        last = cache.last_sweep()
        if last is None:
            # First sweep: nothing was cached under a known-good watermark, so
            # just start the clock. Entries age out via the backstop until the
            # next run has a real window to walk.
            cache.record_sweep(through)
            return {'swept': 0, 'invalidated': 0, 'reason': 'first sweep'}
        if last >= through:
            return {'swept': 0, 'invalidated': 0, 'reason': 'already current'}

        max_days = max_days or self._MAX_INDEX_SWEEP_DAYS
        days = (through - last).days
        if days > max_days:
            logger.warning(
                "SEC XBRL: %d days since the last filing-index sweep (max %d); "
                "skipping it — cached facts fall back to the age backstop",
                days, max_days)
            return {'swept': 0, 'invalidated': 0, 'reason': f'gap {days}d > {max_days}d'}

        # The watermark may only advance over an unbroken run of resolved
        # days: the first day whose index could not be read stops it, so that
        # day is retried next run instead of its filings being skipped.
        filed, swept, resolved_through = set(), 0, last
        stalled_on = None
        day = last + _timedelta(days=1)
        while day <= through:
            got = self._daily_index_ciks(day)
            if got is None:
                stalled_on = day
                break
            filed |= got
            swept += 1
            resolved_through = day
            day += _timedelta(days=1)

        removed = cache.invalidate(filed)
        pruned = cache.prune_expired()
        if resolved_through > last:
            cache.record_sweep(resolved_through)
        if stalled_on is not None:
            logger.warning("SEC XBRL: filing index unreadable for %s; sweeping "
                           "resumes there next run", stalled_on)
        logger.info("SEC XBRL: filing-index sweep %s..%s — %d filer(s) filed, "
                    "%d cached blob(s) evicted", last + _timedelta(days=1),
                    resolved_through, len(filed), removed)
        return {'swept': swept, 'filers': len(filed), 'invalidated': removed,
                'pruned': pruned, 'through': resolved_through.isoformat(),
                'stalled_on': stalled_on.isoformat() if stalled_on else None}

    def facts_cache_stats(self):
        """Persistent-cache counters for the run-quality summary, or None."""
        return None if self._facts_cache is None else self._facts_cache.stats()

    def get_filing_provenance(self, ticker):
        """Latest annual-filing metadata from the cached companyfacts blob.

        Reads only the in-memory cache populated by fetch_company_facts —
        never triggers a network request, so it returns None if called
        before the facts were fetched (or after they were evicted).

        Scans the revenue/net_income concepts (present for virtually every
        filer) for annual-report entries and returns the most recently
        filed one as {'cik', 'accn', 'form', 'filed', 'fy'}, or None.
        """
        facts = self._cache.get(ticker)
        if not facts:
            return None
        best = None
        try:
            for taxonomy_key, tag_map in self._TAXONOMIES:
                taxo = facts.get('facts', {}).get(taxonomy_key, {})
                for concept in ('revenue', 'net_income'):
                    for tag in tag_map.get(concept, ()):
                        units = (taxo.get(tag) or {}).get('units', {})
                        for entries in units.values():
                            for e in entries:
                                if e.get('form') not in ('10-K', '20-F', '40-F'):
                                    continue
                                filed = e.get('filed') or ''
                                if best is None or filed > best['filed']:
                                    best = {'accn': e.get('accn'),
                                            'form': e.get('form'),
                                            'filed': filed,
                                            'fy': e.get('fy')}
        except Exception as e:
            logger.debug(f"SEC XBRL: filing provenance scan failed for {ticker}: {e}")
            return None
        if best is None:
            return None
        best['cik'] = self._cik_map.get(ticker)
        return best

    # Concepts whose alias tags are the whole-company total or components of
    # it, so when one filing carries several of them for the same period the
    # LARGEST of the aliases is the total. Revenue is the one such concept in
    # the map: NJR tags RegulatedAndUnregulatedOperatingRevenue ($2.6B, the
    # operating revenues line) alongside
    # RevenueFromContractWithCustomerExcludingAssessedTax ($0.8B, the ASC 606
    # slice) every year since 2019, and the first-tag tie-break read the
    # slice as revenue — a 3x understatement that only surfaced once the
    # pre-2019 totals joined the series. The first tag in the list
    # ('Revenues', the explicit GAAP total) still wins every tie outright:
    # for E&Ps it is SMALLER than the gas-sales alias because derivative
    # losses sit inside it (EQT 2021: $3.1B vs $6.8B), and that total is the
    # figure wanted. Not generalised beyond revenue: for net income the
    # NCI-inclusive ProfitLoss is larger than the parent-only figure the
    # models want, so first-tag-wins stays the default elsewhere.
    _TIE_BREAK_MAX = frozenset({'revenue'})

    def _extract_annual_values(self, facts_json, tag_list, form_filter='10-K',
                               units_key='USD', taxonomy_key='us-gaap',
                               tie_break='first'):
        """Extract annual values for a concept from XBRL facts.

        Tries each tag in tag_list until one has data.  Filters for the
        specified form type.  Handles the complexity of XBRL entries which
        include segment breakdowns and prior-year restatements alongside
        total annual figures.

        tie_break: how to choose between tags that carry a value for the
            same period end with the same filed date — 'first' keeps the
            earlier tag in tag_list; 'max' keeps the first tag when it is
            among them and otherwise the largest value (see _TIE_BREAK_MAX).

        Selection logic per fiscal year:
        1. Only 10-K / 20-F / 40-F filings with fp='FY'
        2. Period must span ~1 full year (340-400 days)
        3. Period end year must match the filing's fiscal year (or fy+1
           for Jan/Feb year-ends labelled by the year the period starts)
        4. Entries are grouped by period END date. The value comes from
           the latest filing (restatements win); the fiscal-year label
           comes from the EARLIEST filing that carried the period — the
           one in which it was the current year. Keying on each entry's
           own 'fy' mislabels Jan/Feb year-end filers (HD, LOW, TGT, KR,
           DG, ROST, ...): the newest 10-K's prior-year comparative
           carries the newest fy, ties with the current year on filed
           date, and wins on JSON order, so every year in the history
           shifted back by one and the "latest" year was stale.

        Args:
            facts_json: Raw companyfacts JSON.
            tag_list: List of XBRL tag names to try (US-GAAP or IFRS).
            form_filter: Filing form type ('10-K', '20-F', or '40-F').
            taxonomy_key: 'us-gaap' (default) or 'ifrs-full' for foreign issuers.

        Returns:
            dict {fiscal_year (int): value (float)} sorted by year,
            or empty dict if no data found.
        """
        if not facts_json:
            return {}

        us_gaap = facts_json.get('facts', {}).get(taxonomy_key, {})

        # Merge values across all matching tags. Companies often switch
        # XBRL tags over time (e.g. SalesRevenueNet → RevenueFromContract...)
        # Grouped by period end date — see the docstring for why the
        # entry's own 'fy' cannot be the key.
        by_end = {}  # {end: {'val': (filed, value), 'label': (filed, fy)}}

        for rank, tag in enumerate(tag_list):
            concept = us_gaap.get(tag)
            if not concept:
                continue

            units = concept.get('units', {})
            entries = units.get(units_key, [])
            if not entries:
                continue

            # Filter for annual filings with full-year periods
            candidates = []
            for e in entries:
                form = e.get('form', '')
                if form not in (form_filter, '20-F', '40-F'):
                    continue
                fy = e.get('fy')
                fp = e.get('fp', '')
                val = e.get('val')
                filed = e.get('filed', '')
                start = e.get('start', '')
                end = e.get('end', '')
                if fy is None or val is None or not end:
                    continue
                # Only full-year figures
                if fp not in ('FY', 'CY'):
                    continue

                # Two types of XBRL entries:
                # 1. Duration (income/cash flow): has start + end
                # 2. Point-in-time (balance sheet): has end only
                #
                # For duration entries: filter by period ~1 year (340-400 days)
                # For point-in-time: filter by end year matching fiscal year
                # Both: require end year to match fy (or fy+1 for Jan FY-ends)
                try:
                    if end:
                        d_end = _dt.strptime(end, '%Y-%m-%d')
                        end_year = d_end.year
                        if end_year != fy and end_year != fy + 1:
                            continue
                    if start and end:
                        d_start = _dt.strptime(start, '%Y-%m-%d')
                        d_end = _dt.strptime(end, '%Y-%m-%d')
                        days = (d_end - d_start).days
                        if days < 340 or days > 400:
                            continue
                except (ValueError, TypeError):
                    pass  # If dates can't be parsed, keep the entry

                candidates.append((fy, filed, val, end))

            # Per period end: latest filed value (restatements supersede),
            # earliest filed fiscal-year label (the original filing). Ties
            # on filed date keep the first tag in tag_list, unless the
            # concept opted into the largest-value rule (tie_break='max').
            for fy, filed, val, end in candidates:
                rec = by_end.get(end)
                if rec is None:
                    by_end[end] = {'val': (filed, val), 'label': (filed, fy),
                                   'rank': rank}
                    continue
                if filed > rec['val'][0]:
                    rec['val'] = (filed, val)
                    rec['rank'] = rank
                elif (tie_break == 'max' and filed == rec['val'][0]
                        and rec['rank'] > 0 and val > rec['val'][1]):
                    rec['val'] = (filed, val)
                    rec['rank'] = rank
                if filed < rec['label'][0]:
                    rec['label'] = (filed, fy)

        if not by_end:
            return {}

        # Two period ends can only share a label when a filer's original
        # filing is missing from companyfacts; keep the later period.
        merged = {}  # {fy_label: (end, value)}
        for end, rec in by_end.items():
            label = rec['label'][1]
            prev = merged.get(label)
            if prev is None or end > prev[0]:
                merged[label] = (end, rec['val'][1])

        return {fy: info[1] for fy, info in sorted(merged.items())}

    def _extract_periodic_values(self, facts_json, tag_list, units_key='USD',
                                 point_in_time=False, taxonomy_key='us-gaap'):
        """Extract per-period (quarterly + annual) values for a flow/stock concept.

        For *flow* concepts (revenue, net income, OCF, capex, etc.), accepts
        entries with start+end durations of ~1 quarter (80–100 days) or
        ~1 year (340–400 days). YTD partials (6/9-month cumulative figures
        filed in Q2/Q3 10-Qs) are dropped so plotted magnitudes are
        comparable across points.

        Then, for each annual entry at end-date X, looks for ~3 quarterly
        entries with end-dates in (X − 380d, X). If found, derives Q4 =
        FY − sum(Q1+Q2+Q3) and replaces the annual point with four
        quarterly points. Years without quarterly coverage keep the
        annual point.

        For *stock* (point-in-time) concepts like shares outstanding, set
        ``point_in_time=True``: every dated entry is accepted (deduped by
        latest filed per end date).

        Returns:
            dict {period_end (YYYY-MM-DD str): value (float)} sorted by date.
        """
        if not facts_json:
            return {}

        us_gaap = facts_json.get('facts', {}).get(taxonomy_key, {})

        # Stage 1: collect entries deduped by (end, kind) where kind is "Q",
        # "FY", or "PT" (point-in-time). XBRL filings include prior-period
        # comparatives — different filings restate the same period, so we
        # keep the latest filed version per (end, kind).
        merged = {}  # {(end_str, kind): (filed_str, val)}

        for tag in tag_list:
            concept = us_gaap.get(tag)
            if not concept:
                continue
            entries = concept.get('units', {}).get(units_key, [])
            if not entries:
                continue

            for e in entries:
                form = e.get('form', '')
                if form not in ('10-K', '10-Q', '20-F', '40-F'):
                    continue
                val = e.get('val')
                filed = e.get('filed', '')
                start = e.get('start', '')
                end = e.get('end', '')
                if val is None or not end:
                    continue

                if point_in_time:
                    kind = 'PT'
                else:
                    if not start:
                        continue
                    try:
                        d_start = _dt.strptime(start, '%Y-%m-%d')
                        d_end = _dt.strptime(end, '%Y-%m-%d')
                    except (ValueError, TypeError):
                        continue
                    days = (d_end - d_start).days
                    if 80 <= days <= 100:
                        kind = 'Q'
                    elif 340 <= days <= 400:
                        kind = 'FY'
                    else:
                        # YTD partial (180/270d) or other — drop.
                        continue

                key = (end, kind)
                prev = merged.get(key)
                if prev is None or filed > prev[0]:
                    merged[key] = (filed, val)

        if not merged:
            return {}

        if point_in_time:
            # Collapse to {end_date: val}.
            return {end: info[1] for (end, _k), info in sorted(merged.items())}

        # Stage 2: derive Q4 for each annual entry where 3 quarterly entries
        # ending in the prior ~380 days are present. End-date keys are
        # YYYY-MM-DD strings, so chronological sort = lexicographic sort.
        quarters = sorted([end for (end, k) in merged if k == 'Q'])
        annuals = sorted([end for (end, k) in merged if k == 'FY'])

        out = {}  # {end_date_str: val}

        # Add all quarterly entries as-is.
        for q_end in quarters:
            out[q_end] = merged[(q_end, 'Q')][1]

        # For each annual: find ~3 quarterly entries within the prior year.
        # If ≥3 found, derive a Q4 point at the annual's end-date.
        # Otherwise keep the annual point as-is at its end-date.
        for fy_end in annuals:
            try:
                fy_d = _dt.strptime(fy_end, '%Y-%m-%d')
            except ValueError:
                continue
            # Take quarterlies within (fy_end − 380d, fy_end]
            in_window = []
            for q_end in quarters:
                if q_end > fy_end:
                    continue
                try:
                    q_d = _dt.strptime(q_end, '%Y-%m-%d')
                except ValueError:
                    continue
                gap = (fy_d - q_d).days
                if 0 < gap <= 380:
                    in_window.append((q_end, merged[(q_end, 'Q')][1]))

            if len(in_window) >= 3:
                # Take the 3 most recent (largest gap excluded if 4 found).
                in_window.sort(key=lambda x: x[0])
                q3 = in_window[-3:]
                q_sum = sum(v for _, v in q3)
                fy_val = merged[(fy_end, 'FY')][1]
                # Derived Q4 sits at the annual end-date.
                out[fy_end] = fy_val - q_sum
            else:
                # No quarterly coverage — keep the annual at year-end. (Mixed
                # magnitude is acceptable: pre-2009 or foreign filers will
                # only have annual points.)
                out[fy_end] = merged[(fy_end, 'FY')][1]

        return dict(sorted(out.items()))

    # ------------------------------------------------------------------
    # Dual-taxonomy + currency-aware concept resolution
    # ------------------------------------------------------------------

    def _detect_currency(self, facts_json, concept):
        """Return the reporting currency for a concept across both taxonomies.

        Inspects the unit keys present on each candidate tag, returning the
        first ISO-shaped currency code (3 uppercase letters, not 'shares'
        or 'pure'). Prefers USD if available, else falls back to whichever
        currency the foreign filer actually uses.

        Returns None if no currency-keyed entries exist for any candidate tag.
        """
        if not facts_json:
            return None
        facts = facts_json.get('facts', {})
        for taxonomy_key, tag_map in self._TAXONOMIES:
            ns = facts.get(taxonomy_key, {})
            if not ns:
                continue
            for tag in tag_map.get(concept, []):
                units = ns.get(tag, {}).get('units', {})
                if 'USD' in units:
                    return 'USD'
                for k in units:
                    if len(k) == 3 and k.isalpha() and k.isupper():
                        return k
        return None

    def _extract_concept_annual(self, facts_json, concept, units_key=None):
        """Try US-GAAP first, fall back to IFRS, auto-detect currency.

        Returns (values_dict, taxonomy_key, currency) tuple. values_dict is
        empty if no entries were found in either taxonomy.

        units_key: explicit override (e.g. 'shares' for share counts). If
        None, the method auto-picks USD or the foreign filer's currency.
        """
        ccy = units_key or self._detect_currency(facts_json, concept) or 'USD'
        tie_break = 'max' if concept in self._TIE_BREAK_MAX else 'first'
        for taxonomy_key, tag_map in self._TAXONOMIES:
            tags = tag_map.get(concept, [])
            if not tags:
                continue
            vals = self._extract_annual_values(
                facts_json, tags, units_key=ccy, taxonomy_key=taxonomy_key,
                tie_break=tie_break)
            if vals:
                return vals, taxonomy_key, ccy
        return {}, None, ccy

    def _resolve_revenue_annual(self, facts_json):
        """Whole-company revenue per fiscal year, (values, currency).

        The generic revenue aliases, then — for filers that tag both net
        interest income and noninterest income, i.e. banks — the larger of
        that sum and the alias value for each year. Banks tag no revenue
        total in the alias list: 144 of 224 banks in the 2026-09-03 snapshot
        had no revenue at all and the other 80 carried their ASC 606 fee
        slice, which is why 78 of them "passed" Margin Advantage on
        operating margins above 100%. Where a bank does tag Revenues it is
        this same net figure (JPM 2025: 95.4B + 87.0B = 182.4B), so the max
        is a no-op there and only replaces the fee slice.
        """
        rev, _tax, ccy = self._extract_concept_annual(facts_json, 'revenue')
        nii, _t2, nii_ccy = self._extract_concept_annual(facts_json, 'net_interest_income')
        nonint, _t3, _c3 = self._extract_concept_annual(facts_json, 'noninterest_income')
        if not nii or not nonint:
            return rev, ccy
        out = dict(rev)
        for fy in set(nii) & set(nonint):
            bank_rev = nii[fy] + nonint[fy]
            if out.get(fy) is None or bank_rev > out[fy]:
                out[fy] = bank_rev
        return dict(sorted(out.items())), (ccy if rev else nii_ccy)

    def _resolve_equity_annual(self, facts_json):
        """Parent-attributable equity per fiscal year, (values, currency).

        The 'total_equity' alias list merges StockholdersEquity with the
        NCI-inclusive variant, so a filer that tags only the inclusive total
        (or restates it in a later filing) hands every per-share model
        equity that belongs partly to minority holders. Per year:

        1. parent-only tag when present;
        2. otherwise the NCI-inclusive total minus tagged minority interest;
        3. otherwise the inclusive total as-is (best available).
        """
        parent, _t, ccy = self._extract_concept_annual(facts_json, 'equity_parent')
        incl, _t2, ccy2 = self._extract_concept_annual(facts_json, 'equity_incl_nci')
        if not parent and not incl:
            return {}, ccy
        nci, _t3, _c3 = self._extract_concept_annual(facts_json, 'minority_interest')
        out = {}
        for fy in sorted(set(parent) | set(incl)):
            if parent.get(fy) is not None:
                out[fy] = parent[fy]
                continue
            total = incl.get(fy)
            if total is None:
                continue
            mi = nci.get(fy)
            if mi is None:
                logger.debug(
                    f"SEC XBRL: FY{fy} equity is NCI-inclusive with no "
                    f"MinorityInterest tag; using the total as-is")
                out[fy] = total
            else:
                out[fy] = total - mi
        return out, (ccy if parent else ccy2)

    def _extract_concept_periodic(self, facts_json, concept, units_key=None,
                                  point_in_time=False):
        """Periodic equivalent of _extract_concept_annual.

        Returns (values_dict, taxonomy_key, currency) tuple.
        """
        ccy = units_key or self._detect_currency(facts_json, concept) or 'USD'
        for taxonomy_key, tag_map in self._TAXONOMIES:
            tags = tag_map.get(concept, [])
            if not tags:
                continue
            vals = self._extract_periodic_values(
                facts_json, tags, units_key=ccy,
                point_in_time=point_in_time, taxonomy_key=taxonomy_key)
            if vals:
                return vals, taxonomy_key, ccy
        return {}, None, ccy

    _DEBT_COMPONENT_CONCEPTS = (
        'ltd_noncurrent', 'ltd_total', 'debt_current_total',
        'ltd_current', 'st_borrowings',
    )

    def _resolve_total_debt_annual(self, facts_json, units_key='USD',
                                   taxonomy_key='us-gaap'):
        """Per-year total debt from separately-extracted components.

        Priority per fiscal year (never fabricate zero for a levered filer):
          1. noncurrent LTD + current-debt total (clean split, no overlap)
          2. noncurrent LTD + current LTD + short-term borrowings
          3. total LTD (already includes current maturities) + ST borrowings
        Years where none of the tiers resolve are OMITTED — debt UNKNOWN
        for that year is not debt = 0.

        Returns:
            (values {fy: float}, tagged bool). tagged=False means NO debt
            component concept carries any entries at all — the filer is
            genuinely unlevered (or entirely untagged) and callers may
            treat debt as 0; tagged=True with a missing year means unknown.
        """
        tag_map = dict(self._TAXONOMIES).get(taxonomy_key, {})

        def _c(concept):
            tags = tag_map.get(concept, [])
            if not tags:
                return {}
            return self._extract_annual_values(
                facts_json, tags, units_key=units_key,
                taxonomy_key=taxonomy_key) or {}

        ltd_nc = _c('ltd_noncurrent')
        ltd_total = _c('ltd_total')
        debt_cur = _c('debt_current_total')
        ltd_cur = _c('ltd_current')
        stb = _c('st_borrowings')

        tagged = any((ltd_nc, ltd_total, debt_cur, ltd_cur, stb))
        out = {}
        for y in sorted(set(ltd_nc) | set(ltd_total) | set(debt_cur)
                        | set(ltd_cur) | set(stb)):
            if ltd_nc.get(y) is not None:
                if debt_cur.get(y) is not None:
                    out[y] = ltd_nc[y] + debt_cur[y]
                else:
                    out[y] = (ltd_nc[y] + (ltd_cur.get(y) or 0)
                              + (stb.get(y) or 0))
            elif ltd_total.get(y) is not None:
                out[y] = ltd_total[y] + (stb.get(y) or 0)
        return out, tagged

    def _resolve_total_debt_concept(self, facts_json):
        """Dual-taxonomy, currency-aware total-debt resolution.

        The _extract_concept_annual equivalent for the composed total-debt
        figure: tries US-GAAP first, falls back to IFRS, auto-detecting the
        reporting currency from whichever debt component is tagged.

        Returns (values {fy: float}, taxonomy_key, currency, tagged).
        """
        ccy = None
        for concept in self._DEBT_COMPONENT_CONCEPTS:
            ccy = ccy or self._detect_currency(facts_json, concept)
        ccy = ccy or 'USD'
        any_tagged = False
        for taxonomy_key, _tag_map in self._TAXONOMIES:
            vals, tagged = self._resolve_total_debt_annual(
                facts_json, units_key=ccy, taxonomy_key=taxonomy_key)
            any_tagged = any_tagged or tagged
            if vals:
                return vals, taxonomy_key, ccy, True
        return {}, None, ccy, any_tagged

    # ------------------------------------------------------------------
    # Public API: Validation
    # ------------------------------------------------------------------

    def validate_against_yfinance(self, ticker, yf_financials):
        """Cross-check yfinance financial data against SEC EDGAR XBRL.

        Compares the most recent annual values for key financial concepts.
        Flags discrepancies where the percentage difference exceeds 5%.

        Args:
            ticker: Stock ticker symbol.
            yf_financials: dict with 'balance_sheet', 'income_statement',
                          'cash_flow', 'info' from yfinance.

        Returns:
            dict with validation results, or None on failure.
        """
        facts = self.fetch_company_facts(ticker)
        if not facts:
            return None

        discrepancies = []
        fields_checked = 0
        fields_flagged = 0

        for concept, yf_keys in self._YF_KEY_MAP.items():
            # Get EDGAR value (most recent annual). Total debt is a composed
            # figure (LTD + current debt), not a single tag — resolve it the
            # same way build_yfinance_shape does so the comparison against
            # yfinance's current-inclusive 'Total Debt' row is apples-to-apples.
            if concept == 'total_debt':
                annual_vals, _tagged = self._resolve_total_debt_annual(facts)
            else:
                xbrl_tags = self._XBRL_TAG_MAP.get(concept, [])
                annual_vals = self._extract_annual_values(facts, xbrl_tags)
            if not annual_vals:
                continue

            edgar_year = max(annual_vals.keys())
            edgar_val = annual_vals[edgar_year]

            # Get yfinance value from the corresponding statement
            stmt_key = self._YF_STATEMENT_MAP.get(concept)
            if not stmt_key:
                continue
            stmt = yf_financials.get(stmt_key)
            if stmt is None or stmt.empty:
                continue

            # Get latest year column
            try:
                latest_col = stmt.iloc[:, 0]
                yf_val = _get(latest_col, yf_keys)
            except (IndexError, KeyError):
                continue

            if yf_val is None or edgar_val is None:
                continue

            # Compute percentage difference
            max_abs = max(abs(yf_val), abs(edgar_val))
            if max_abs == 0:
                pct_diff = 0.0
            else:
                pct_diff = abs(yf_val - edgar_val) / max_abs

            flagged = pct_diff > 0.05
            fields_checked += 1
            if flagged:
                fields_flagged += 1

            discrepancies.append({
                'field': concept,
                'yfinance': yf_val,
                'edgar': edgar_val,
                'edgar_year': edgar_year,
                'pct_diff': round(pct_diff, 4),
                'flagged': flagged,
            })

        if fields_checked == 0:
            return None

        return {
            'validated': True,
            'discrepancies': discrepancies,
            'fields_checked': fields_checked,
            'fields_flagged': fields_flagged,
            'edgar_quality_score': max(0, 100 - (20 * fields_flagged)),
        }

    # ------------------------------------------------------------------
    # Public API: Historical financials
    # ------------------------------------------------------------------

    def fetch_historical_financials(self, ticker, min_years=10):
        """Fetch long-duration revenue and earnings history from EDGAR XBRL.

        Flow concepts (revenue, net income, OCF, capex, etc.) are returned
        as one value per fiscal year keyed by integer year — sourced from
        10-K / 20-F / 40-F annual filings (US-GAAP or IFRS taxonomy).
        Shares outstanding remains point-in-time keyed by period-end date
        ("YYYY-MM-DD"), since balance-sheet items carry useful intra-year
        detail.

        Foreign-currency filings (EUR, JPY, DKK, etc.) are auto-detected
        and converted to USD using year-end FX rates from yfinance, so
        the returned values are always USD-denominated.

        Args:
            ticker: Stock ticker symbol.
            min_years: Minimum years desired (informational, not enforced).

        Returns:
            dict with revenue_history, earnings_history, years_available,
            reporting_currency, fx_converted, or None on failure.
        """
        facts = self.fetch_company_facts(ticker)
        if not facts:
            return None

        def _flow(concept):
            vals, _tax, ccy = self._extract_concept_annual(facts, concept)
            return vals, ccy

        rev, rev_ccy             = self._resolve_revenue_annual(facts)
        ni,  ni_ccy              = _flow('net_income')
        ocf, ocf_ccy             = _flow('operating_cash_flow')
        capex, capex_ccy         = _flow('capex')
        gp, gp_ccy               = _flow('gross_profit')
        intexp, intexp_ccy       = _flow('interest_expense')
        div, div_ccy             = _flow('dividends_paid')
        opinc, opinc_ccy         = _flow('operating_income')
        # Remaining income-statement + cash-flow lines. Only the popup's
        # statement tabs consume these; every ratio/model above still reads
        # the series it already read, so adding them changes no scoring.
        pretax, pretax_ccy       = _flow('pretax_income')
        taxprov, taxprov_ccy     = _flow('tax_provision')
        dna, dna_ccy             = _flow('d_and_a')
        # Balance-sheet series (annual FY-keyed like the flows — one
        # fiscal-year-end instant per year). Total debt is composed from
        # the component concepts via the shared resolver.
        cash_h, cash_ccy         = _flow('cash')
        assets_h, assets_ccy     = _flow('total_assets')
        ca_h, ca_ccy             = _flow('current_assets')
        cl_h, cl_ccy             = _flow('current_liabilities')
        liabs_h, liabs_ccy       = _flow('total_liabilities')
        equity_h, equity_ccy     = self._resolve_equity_annual(facts)
        retearn_h, retearn_ccy   = _flow('retained_earnings')
        # GAAP presentation lines. Grouped by statement; every one of these
        # is optional — the statement tabs drop a row whose series is empty.
        cogs_h, _c1     = _flow('cost_of_revenue')
        rd_h, _c2       = _flow('rd_expense')
        sga_h, _c3      = _flow('sga_expense')
        selmkt_h, _c4   = _flow('selling_marketing')
        opex_h, _c5     = _flow('total_opex')
        othnonop_h, _c6 = _flow('other_nonop')
        stinv_h, _c7    = _flow('st_investments')
        recv_h, _c8     = _flow('receivables')
        invty_h, _c9    = _flow('inventory')
        ppe_h, _c10     = _flow('ppe_net')
        gw_h, _c11      = _flow('goodwill')
        intang_h, _c12  = _flow('intangibles')
        ap_h, _c13      = _flow('accounts_payable')
        lae_h, _c14     = _flow('liab_and_equity')
        csapic_h, _c15  = _flow('cs_apic')
        aoci_h, _c16    = _flow('aoci')
        treas_h, _c17   = _flow('treasury_stock')
        minint_h, _c18  = _flow('minority_interest')
        icf_h, _c19     = _flow('investing_cf')
        fcf_h, _c20     = _flow('financing_cf')
        sbc_h, _c21     = _flow('sbc_cf')
        deftax_h, _c22  = _flow('deferred_tax')
        buyback_h, _c23 = _flow('buybacks')
        dissued_h, _c24 = _flow('debt_issued')
        drepaid_h, _c25 = _flow('debt_repaid')
        acq_h, _c26     = _flow('acquisitions')
        chgcash_h, _c27 = _flow('net_change_cash')
        fxeff_h, _c28   = _flow('fx_effect_cash')
        # ----- Derive missing income-statement series -----
        # build_yfinance_shape fills the ROW-level statements from accounting
        # identities when a filer never tags the line (HCA, ZTS, ADP, BMY,
        # MRK, LLY, COP, CLX ... tag pretax income but no OperatingIncomeLoss;
        # ~380 filers tag CostOfRevenue but no GrossProfit). The multi-year
        # series here did not, so for those filers every history-backed
        # consumer — Mult vs Hist, Pool Share, Margin vs Hist, the
        # int_cov_edgar fallback and the gross-margin trend — read N/A while
        # the point-in-time figures resolved fine. Same rules as the shape
        # builder, never overwriting a tagged value:
        #   pretax    = net income + tax provision           (last resort)
        #   operating = pretax + interest expense            (other income ~ 0)
        #   gross     = revenue - cost of revenue
        # Every operand is in the reporting currency, so this sits before the
        # FX pass; the derived years are re-sorted so the series stay
        # year-ordered like every other history dict.
        for _y in set(ni) | set(taxprov):
            if (pretax.get(_y) is None and ni.get(_y) is not None
                    and taxprov.get(_y) is not None):
                pretax[_y] = ni[_y] + taxprov[_y]
        for _y in set(pretax):
            if opinc.get(_y) is None and pretax.get(_y) is not None:
                opinc[_y] = pretax[_y] + (intexp.get(_y) or 0)
        for _y in set(rev) & set(cogs_h):
            if (gp.get(_y) is None and rev.get(_y) is not None
                    and cogs_h.get(_y) is not None):
                gp[_y] = rev[_y] - cogs_h[_y]
        pretax = dict(sorted(pretax.items()))
        opinc = dict(sorted(opinc.items()))
        gp = dict(sorted(gp.items()))

        # A classified balance sheet splits borrowings across the current and
        # non-current captions, but the shared resolver only returns the
        # composed total. Rebuild the two halves from the same component
        # concepts it uses, applying its precedence: DebtCurrent already
        # bundles short-term borrowings with current maturities, so the
        # ltd_current + st_borrowings sum is only a fallback. Likewise
        # LongTermDebt INCLUDES current maturities while LongTermDebtNoncurrent
        # does not — subtract the current portion when only the former exists.
        dct_h, _d1 = _flow('debt_current_total')
        ltdc_h, _d2 = _flow('ltd_current')
        stb_h, _d3 = _flow('st_borrowings')
        ltdnc_h, _d4 = _flow('ltd_noncurrent')
        ltdt_h, _d5 = _flow('ltd_total')
        _yrs = set(dct_h) | set(ltdc_h) | set(stb_h) | set(ltdnc_h) | set(ltdt_h)
        debt_cur_h, debt_nc_h = {}, {}
        for _y in _yrs:
            if _y in dct_h:
                debt_cur_h[_y] = dct_h[_y]
            elif _y in ltdc_h or _y in stb_h:
                debt_cur_h[_y] = ltdc_h.get(_y, 0.0) + stb_h.get(_y, 0.0)
            if _y in ltdnc_h:
                debt_nc_h[_y] = ltdnc_h[_y]
            elif _y in ltdt_h:
                # LongTermDebt is current-inclusive; net out current maturities
                # when they're separately tagged, else take it as reported.
                debt_nc_h[_y] = ltdt_h[_y] - ltdc_h.get(_y, 0.0)
        debt_h, _dtx, debt_ccy, debt_tagged = \
            self._resolve_total_debt_concept(facts)
        shares, _tax_s, _ccy_s   = self._extract_concept_periodic(
            facts, 'shares_outstanding', units_key='shares', point_in_time=True)
        if not shares:
            # Cover-page share count (dei taxonomy). Many filers (KO, JNJ,
            # most ADRs) never tag the us-gaap balance-sheet concepts but
            # always file this one — it's mandatory on 10-K/10-Q/20-F covers.
            shares = self._extract_periodic_values(
                facts, ['EntityCommonStockSharesOutstanding'],
                units_key='shares', point_in_time=True, taxonomy_key='dei')

        if not rev and not ni:
            return None

        # The reporting currency is whichever non-USD currency appears on the
        # primary income-statement concepts. If revenue is in JPY but a US-GAAP
        # subsidiary tag happens to carry USD on, say, dividends, treat the
        # filer as JPY.
        # Deliberately NOT widened to the statement-tab series added below:
        # this list decides whether every OTHER series gets FX-converted, so
        # a new concept flipping the detection would move published scores.
        # The new series ride the detected rate; they don't vote on it.
        currencies = [c for c in (rev_ccy, ni_ccy, ocf_ccy, capex_ccy,
                                  gp_ccy, intexp_ccy, div_ccy, opinc_ccy) if c]
        reporting_ccy = next((c for c in currencies if c != 'USD'), 'USD')
        fx_converted = reporting_ccy != 'USD'

        # Per-share amounts live under a compound unit ("USD/shares"), and
        # weighted share counts under "shares" — neither is discoverable by
        # the currency auto-detect, so both need an explicit units_key. They
        # sit here rather than with the other _flow calls because the unit
        # string depends on the reporting currency resolved just above.
        eps_b, _e1, _e2 = self._extract_concept_annual(
            facts, 'eps_basic', units_key=reporting_ccy + '/shares')
        eps_d, _e3, _e4 = self._extract_concept_annual(
            facts, 'eps_diluted', units_key=reporting_ccy + '/shares')
        wavg_b, _e5, _e6 = self._extract_concept_annual(
            facts, 'wavg_basic', units_key='shares')
        wavg_d, _e7, _e8 = self._extract_concept_annual(
            facts, 'wavg_diluted', units_key='shares')

        if fx_converted:
            fx = _get_fx_rates_to_usd(reporting_ccy)
            rev    = _apply_fx_annual(rev, fx)
            ni     = _apply_fx_annual(ni, fx)
            ocf    = _apply_fx_annual(ocf, fx)
            capex  = _apply_fx_annual(capex, fx)
            gp     = _apply_fx_annual(gp, fx)
            intexp = _apply_fx_annual(intexp, fx)
            div    = _apply_fx_annual(div, fx)
            opinc  = _apply_fx_annual(opinc, fx)
            pretax  = _apply_fx_annual(pretax, fx)
            taxprov = _apply_fx_annual(taxprov, fx)
            dna     = _apply_fx_annual(dna, fx)
            # Balance-sheet instants use the same year-end-close rate as the
            # flows — exact for calendar-year fiscal ends, and the same
            # months-off approximation already accepted for flow series on
            # mid-year fiscal ends.
            cash_h = _apply_fx_annual(cash_h, fx)
            debt_h = _apply_fx_annual(debt_h, fx)
            assets_h  = _apply_fx_annual(assets_h, fx)
            ca_h      = _apply_fx_annual(ca_h, fx)
            cl_h      = _apply_fx_annual(cl_h, fx)
            liabs_h   = _apply_fx_annual(liabs_h, fx)
            equity_h  = _apply_fx_annual(equity_h, fx)
            retearn_h = _apply_fx_annual(retearn_h, fx)
            # GAAP statement lines. Per-share amounts convert at the same
            # rate (a currency-per-share unit scales exactly like currency);
            # weighted share COUNTS do not, same as `shares` below.
            cogs_h    = _apply_fx_annual(cogs_h, fx)
            rd_h      = _apply_fx_annual(rd_h, fx)
            sga_h     = _apply_fx_annual(sga_h, fx)
            selmkt_h  = _apply_fx_annual(selmkt_h, fx)
            opex_h    = _apply_fx_annual(opex_h, fx)
            othnonop_h = _apply_fx_annual(othnonop_h, fx)
            stinv_h   = _apply_fx_annual(stinv_h, fx)
            recv_h    = _apply_fx_annual(recv_h, fx)
            invty_h   = _apply_fx_annual(invty_h, fx)
            ppe_h     = _apply_fx_annual(ppe_h, fx)
            gw_h      = _apply_fx_annual(gw_h, fx)
            intang_h  = _apply_fx_annual(intang_h, fx)
            ap_h      = _apply_fx_annual(ap_h, fx)
            lae_h     = _apply_fx_annual(lae_h, fx)
            csapic_h  = _apply_fx_annual(csapic_h, fx)
            aoci_h    = _apply_fx_annual(aoci_h, fx)
            treas_h   = _apply_fx_annual(treas_h, fx)
            minint_h  = _apply_fx_annual(minint_h, fx)
            icf_h     = _apply_fx_annual(icf_h, fx)
            fcf_h     = _apply_fx_annual(fcf_h, fx)
            sbc_h     = _apply_fx_annual(sbc_h, fx)
            deftax_h  = _apply_fx_annual(deftax_h, fx)
            buyback_h = _apply_fx_annual(buyback_h, fx)
            dissued_h = _apply_fx_annual(dissued_h, fx)
            drepaid_h = _apply_fx_annual(drepaid_h, fx)
            acq_h     = _apply_fx_annual(acq_h, fx)
            chgcash_h = _apply_fx_annual(chgcash_h, fx)
            fxeff_h   = _apply_fx_annual(fxeff_h, fx)
            debt_cur_h = _apply_fx_annual(debt_cur_h, fx)
            debt_nc_h  = _apply_fx_annual(debt_nc_h, fx)
            eps_b     = _apply_fx_annual(eps_b, fx)
            eps_d     = _apply_fx_annual(eps_d, fx)
            # shares are unit-counts, not currency — leave alone.

        # Untagged debt across every component concept = genuinely unlevered
        # filer: explicit zeros over the revenue span (else net debt would
        # read unknown instead of -cash). A tagged filer with missing years
        # keeps those years absent (unknown ≠ 0).
        if not debt_tagged and not debt_h:
            debt_h = {y: 0.0 for y in (rev or ni)}

        # Unchanged on purpose: years_available gates backfill refetch and
        # model eligibility upstream, so the statement-tab series must not
        # be able to inflate it.
        all_series = [rev, ni, ocf, capex, gp, intexp, div, shares, opinc,
                      debt_h, cash_h]
        years_available = max((len(s) for s in all_series if s), default=0)

        return {
            'revenue_history':          rev,
            'earnings_history':         ni,
            'operating_cf_history':     ocf,
            'capex_history':            capex,
            'gross_profit_history':     gp,
            'interest_expense_history': intexp,
            'dividends_paid_history':   div,
            'operating_income_history': opinc,
            'total_debt_history':       debt_h,
            'cash_history':             cash_h,
            'shares_history':           shares,
            # Statement-tab series (popup Balance Sheet / Income Statement /
            # Cash Flow). Consumed only by the report's hist.json sidecar.
            'pretax_income_history':    pretax,
            'tax_provision_history':    taxprov,
            'd_and_a_history':          dna,
            'total_assets_history':     assets_h,
            'current_assets_history':   ca_h,
            'current_liabilities_history': cl_h,
            'total_liabilities_history': liabs_h,
            'total_equity_history':     equity_h,
            'retained_earnings_history': retearn_h,
            # GAAP presentation lines (popup statement tabs only).
            'cost_of_revenue_history':  cogs_h,
            'rd_expense_history':       rd_h,
            'sga_expense_history':      sga_h,
            'selling_marketing_history': selmkt_h,
            'total_opex_history':       opex_h,
            'other_nonop_history':      othnonop_h,
            'eps_basic_history':        eps_b,
            'eps_diluted_history':      eps_d,
            'wavg_basic_history':       wavg_b,
            'wavg_diluted_history':     wavg_d,
            'st_investments_history':   stinv_h,
            'receivables_history':      recv_h,
            'inventory_history':        invty_h,
            'ppe_net_history':          ppe_h,
            'goodwill_history':         gw_h,
            'intangibles_history':      intang_h,
            'accounts_payable_history': ap_h,
            'liab_and_equity_history':  lae_h,
            'cs_apic_history':          csapic_h,
            'aoci_history':             aoci_h,
            'treasury_stock_history':   treas_h,
            'minority_interest_history': minint_h,
            'debt_current_history':     debt_cur_h,
            'debt_noncurrent_history':  debt_nc_h,
            'investing_cf_history':     icf_h,
            'financing_cf_history':     fcf_h,
            'sbc_cf_history':           sbc_h,
            'deferred_tax_history':     deftax_h,
            'buybacks_history':         buyback_h,
            'debt_issued_history':      dissued_h,
            'debt_repaid_history':      drepaid_h,
            'acquisitions_history':     acq_h,
            'net_change_cash_history':  chgcash_h,
            'fx_effect_cash_history':   fxeff_h,
            'years_available':          years_available,
            'reporting_currency':       reporting_ccy,
            'fx_converted':             fx_converted,
        }

    def build_yfinance_shape(self, ticker, year_limit=None):
        """Construct a yfinance-shaped financials dict from SEC XBRL data.

        Returned shape matches what models/ratios.py expects:
        {balance_sheet, income_statement, cash_flow, info}, with statement
        DataFrames indexed by line-item label and columned by fiscal-year-end
        Timestamp (latest year first).

        info is sparse: only carries `symbol` and a `_source` tag. Callers
        that want CAPM / market-cap-weighted WACC should merge yfinance's
        info dict on top of this result.

        Args:
            ticker: Stock ticker symbol.
            year_limit: Optional cap on the number of year columns kept
                (latest first). Defaults to None — return all available
                history (typically 10-16 years). For a value-investor /
                DCF pipeline this is preferable: through-cycle ROIC and
                tax rates smooth out one-off years and are a more defensible
                input to long-horizon terminal-value calculations.

        Returns:
            dict or None if no XBRL data is available for ticker.
        """
        import pandas as pd

        facts = self.fetch_company_facts(ticker)
        if not facts:
            return None

        def _ann(concept):
            tags = self._XBRL_TAG_MAP.get(concept, [])
            return self._extract_annual_values(facts, tags) if tags else {}

        # Income statement (flow concepts)
        revenue, _rev_ccy = self._resolve_revenue_annual(facts)
        net_income    = _ann('net_income')
        op_income     = _ann('operating_income')
        gross_profit  = _ann('gross_profit')
        interest_exp  = _ann('interest_expense')
        tax_provision = _ann('tax_provision')
        pretax_income = _ann('pretax_income')

        # Cash flow (flow concepts)
        op_cf         = _ann('operating_cash_flow')
        capex         = _ann('capex')
        d_and_a       = _ann('d_and_a')
        # SBC + shareholder-return rows: without these, the owner-earnings
        # SBC haircut in run_forward_dcf and the buyback half of
        # _compute_shareholder_yield silently never fire on the XBRL path
        # (the same failure mode the d_and_a row above once had).
        sbc           = _ann('sbc_cf')
        dividends     = _ann('dividends_paid')
        buybacks      = _ann('buybacks')
        issuance      = _ann('share_issuance')

        # Balance sheet (point-in-time concepts). Their entries in XBRL only
        # carry end dates, no durations — _extract_annual_values' duration
        # filter conditional skips them naturally, and the fy match keeps the
        # right period-end value per fiscal year.
        equity, _eq_ccy = self._resolve_equity_annual(facts)
        cash          = _ann('cash')
        # Liquid assets beyond bank cash. The 'cash' concept prefers the
        # equivalents-only tag, so filers that park liquidity in marketable
        # securities (AAPL, GOOGL, MSFT) read as far more levered than they
        # are in every net-debt bridge. Compose the securities-inclusive row:
        # the inclusive tag when the filer reports it, else cash + current
        # investments when both are tagged for the year. Years where neither
        # resolves stay None so consumers fall back to the bare cash row
        # rather than reading a phantom zero.
        cash_incl     = _ann('cash_incl_sti')
        st_inv        = _ann('st_investments')
        cash_sti = {}
        for _y in set(cash) | set(cash_incl):
            if cash_incl.get(_y) is not None:
                cash_sti[_y] = cash_incl[_y]
            elif cash.get(_y) is not None and st_inv.get(_y) is not None:
                cash_sti[_y] = cash[_y] + st_inv[_y]
        minority      = _ann('minority_interest')
        preferred     = _ann('preferred_stock')
        assets        = _ann('total_assets')
        curr_assets   = _ann('current_assets')
        curr_liabs    = _ann('current_liabilities')
        liabilities   = _ann('total_liabilities')
        ret_earnings  = _ann('retained_earnings')
        # Goodwill + intangibles: without these rows the NAV model's
        # tangible-book strip finds nothing to subtract on the XBRL path, so
        # nav_fv / p_tbv silently degrade to plain book value for every US
        # filer (the same failure mode the SBC / buyback rows above once had).
        goodwill      = _ann('goodwill')
        intangibles   = _ann('intangibles')
        # Period-end share count (an instant, same date as the equity above)
        # so per-share book metrics divide like-dated numerator and
        # denominator instead of year-old equity by today's float.
        shares_fye, _t_sh, _u_sh = self._extract_concept_annual(
            facts, 'shares_outstanding', units_key='shares')
        # Total debt is composed from separately-tagged components (LTD +
        # current debt) — a single-tag read understates leverage by the
        # short-term portion for most filers.
        debt, debt_tagged = self._resolve_total_debt_annual(facts)

        # Need at least revenue or net income to consider the data usable.
        if not revenue and not net_income:
            return None

        years = sorted(set(revenue) | set(net_income) | set(op_income) |
                       set(equity) | set(debt) | set(assets) | set(pretax_income),
                       reverse=True)
        if not years:
            return None

        # Truncate to the most recent year_limit years (latest first).
        # XBRL routinely returns 10–16 years; a long window pulls in pre-
        # crisis filings that yfinance no longer reports, producing ROIC
        # values that diverge meaningfully from the recent window.
        if year_limit and year_limit > 0:
            years = years[:year_limit]

        # ----- Derive missing line items from related XBRL fields -----
        # XBRL tagging varies across companies — many file pretax_income but not
        # operating_income (NKE), or operating_income but not pretax_income (REITs
        # like BXP). The accounting identity is pretax ≈ operating - interest_exp
        # (treating other_income/expense as ≈ 0). Use it to fill missing entries
        # so calculate_roic / calculate_wacc can derive a real tax rate instead
        # of the 21% default. Never overwrite a value that XBRL already provided.
        for y in years:
            op = op_income.get(y)
            pti = pretax_income.get(y)
            intexp = interest_exp.get(y) or 0
            if op is None and pti is not None:
                op_income[y] = pti + intexp
            elif pti is None and op is not None:
                pretax_income[y] = op - intexp
            # Last-resort derivation: pretax = net_income + tax_provision
            if pretax_income.get(y) is None:
                ni = net_income.get(y)
                tx = tax_provision.get(y)
                if ni is not None and tx is not None:
                    pretax_income[y] = ni + tx

        # Untagged debt across every component concept = genuinely unlevered
        # filer -> an explicit 0 per year, so net debt reads -cash instead of
        # unknown. A TAGGED filer with a missing year stays None (unknown).
        if not debt_tagged:
            debt = {y: 0.0 for y in years}

        # Total liabilities: fall back to the Assets − Equity identity for
        # years where the filer doesn't tag the 'Liabilities' total. Unblocks
        # Debt/Equity and Altman Z on the XBRL path.
        for y in years:
            if liabilities.get(y) is None:
                a, e = assets.get(y), equity.get(y)
                if a is not None and e is not None:
                    liabilities[y] = a - e

        # Plain cash falls back to the combined tag for filers that only
        # report the total. The reverse is deliberately NOT done: a year
        # with no investments tag leaves the combined row None, and
        # consumers fall back to the bare cash row via CASH_KEYS order
        # rather than reading a phantom "inclusive" figure equal to cash.
        for y in years:
            if cash.get(y) is None and cash_sti.get(y) is not None:
                cash[y] = cash_sti[y]

        cols = [pd.Timestamp(year=y, month=12, day=31) for y in years]

        income_df = pd.DataFrame({
            col: {
                'Total Revenue':    revenue.get(y),
                'Net Income':       net_income.get(y),
                'Operating Income': op_income.get(y),
                'Gross Profit':     gross_profit.get(y),
                'Interest Expense': interest_exp.get(y),
                'Tax Provision':    tax_provision.get(y),
                'Pretax Income':    pretax_income.get(y),
            } for y, col in zip(years, cols, strict=False)
        })

        balance_df = pd.DataFrame({
            col: {
                'Stockholders Equity':       equity.get(y),
                'Total Debt':                debt.get(y),
                'Cash And Cash Equivalents': cash.get(y),
                'Cash Cash Equivalents And Short Term Investments': cash_sti.get(y),
                'Minority Interest':         minority.get(y),
                'Preferred Stock Equity':    preferred.get(y),
                'Total Assets':              assets.get(y),
                'Current Assets':            curr_assets.get(y),
                'Current Liabilities':       curr_liabs.get(y),
                'Total Liabilities Net Minority Interest': liabilities.get(y),
                'Retained Earnings':         ret_earnings.get(y),
                'Goodwill':                  goodwill.get(y),
                'Other Intangible Assets':   intangibles.get(y),
                'Ordinary Shares Number':    shares_fye.get(y),
            } for y, col in zip(years, cols, strict=False)
        })

        # Outflows follow the yfinance sign convention: stored NEGATIVE
        # (capex, dividends paid, buybacks). XBRL Payments* tags are
        # debit-positive, so negate. SBC and issuance proceeds stay positive,
        # also matching yfinance. Consumers that derive FCF as OCF + Capex
        # (_fcf_series_from_cashflow) and those that abs() the value
        # (calculate_fundamental_growth, the owner-earnings add-back,
        # _compute_shareholder_yield) all read correctly under this
        # convention.
        def _neg(v):
            return -abs(v) if v is not None else None

        cash_flow_df = pd.DataFrame({
            col: {
                'Operating Cash Flow':           op_cf.get(y),
                'Capital Expenditure':           _neg(capex.get(y)),
                'Depreciation And Amortization': d_and_a.get(y),
                'Stock Based Compensation':      sbc.get(y),
                'Common Stock Dividend Paid':    _neg(dividends.get(y)),
                'Repurchase Of Capital Stock':   _neg(buybacks.get(y)),
                'Issuance Of Capital Stock':     issuance.get(y),
            } for y, col in zip(years, cols, strict=False)
        })

        return {
            'balance_sheet':    balance_df,
            'income_statement': income_df,
            'cash_flow':        cash_flow_df,
            'info':             {'symbol': ticker, '_source': 'sec_xbrl'},
            'growth_estimates': None,
            'earnings_history': None,
        }
