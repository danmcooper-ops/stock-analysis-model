"""Generate a value-investing workflow checklist PDF.

Output: docs/value_investing_workflow.pdf
"""
from pathlib import Path
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
OUT = ROOT / "docs" / "value_investing_workflow.pdf"

CHECK = "☐"  # ☐
ARROW = "→"  # →
TIMES = "×"  # ×

# (heading, optional intro paragraph, [ (kind, text) ... ])
# kind: "check" | "sub" | "para" | "rule"
SECTIONS = [
    (
        "Value-Investing Workflow",
        "subtitle",
        "Built from the model's actual scoring weights — Moat 40% / Quality 20% / "
        "Valuation 20% / Ownership 15% / Growth 5% — and the fields rendered in "
        "templates/report.html.",
        [],
    ),
    (
        "Stage 0  —  Pre-screen (kill in 60 seconds)",
        None,
        "Open the report and check these three first. Any single fail → stop.",
        [
            ("check", "Fraud / distress filter: beneish_flag is False AND altman_z ≥ 2.99"),
            ("check", "Confidence floor: _mc_confidence_label ≠ LOW (mc_cv < 40%)"),
            ("check", "EDGAR data integrity: edgar_quality_score ≥ 80 and edgar_fields_flagged is empty for revenue / net income"),
        ],
    ),
    (
        "Stage 1  —  MOAT  (40% weight, the most important block)",
        None,
        "A value buy with no moat is a value trap. Need 3 of 4 to pass.",
        [
            ("check", "roic_cv < 30%   (consistency — the single best moat proxy in this model)"),
            ("check", "spread (ROIC − WACC) > 7%"),
            ("check", "gross_margin_avg_5y > 35%   (pricing power)"),
            ("check", "fcf_margin > 12%   (capital efficiency)"),
            ("sub", "Supporting context (read, don't gate):"),
            ("check", "Scan roic_by_year chart — trend flat/up, or eroding?"),
            ("check", "pp_margin_advantage positive vs _sector_median_opm"),
            ("check", "pp_sector_hhi / pp_sector_cr4 — concentrated industry favors incumbents"),
        ],
    ),
    (
        "Stage 2  —  QUALITY  (20% weight, earnings must be real)",
        None,
        "Need 4 of 5 to pass. These confirm reported numbers aren't manufactured.",
        [
            ("check", "cash_conv ≥ 0.85   (cash backs earnings)"),
            ("check", "accruals absolute value < 8%   (low working-capital games)"),
            ("check", "int_cov > 3×   (debt service safe)"),
            ("check", "nd_ebitda ≤ 1.5×   (leverage discipline)"),
            ("check", "piotroski ≥ 7 / 9"),
            ("sub", "Red-flag escalation (one failure → re-read the 10-K notes):"),
            ("check", "sbc_pct_rev > 2%   — dilution headwind"),
            ("check", "goodwill_pct rising   — acquisition-driven “growth”"),
        ],
    ),
    (
        "Stage 3  —  OWNERSHIP & CAPITAL ALLOCATION  (15% weight)",
        None,
        "Capital allocation tells you whether management is on your side.",
        [
            ("check", "insider_pct ≥ 5%   (skin in the game)"),
            ("check", "shareholder_yield > 2%   (dividends + net buybacks)"),
            ("check", "shares_cagr_5y < 0   (count is shrinking, not growing)"),
            ("check", "sbc_pct_rev ≤ 2%   (buybacks aren't just offsetting dilution)"),
            ("check", "Form 4 panel: insider_buy_count_90d ≥ insider_sell_count_90d OR insider_net_value ≥ 0"),
            ("check", "founder_led is True   → bonus signal, not required"),
        ],
    ),
    (
        "Stage 4  —  VALUATION  (20% weight, only after 1–3 pass)",
        None,
        "Don't anchor on price first. Most value traps clear this stage but fail moat/quality.",
        [
            ("sub", "Primary gates (need 3 of 4):"),
            ("check", "mos > 10%   (DCF margin of safety)"),
            ("check", "_price_fv (price / dcf_fv) < 1.0"),
            ("check", "pfcf between 8× and 20×"),
            ("check", "pb ≤ 5×"),
            ("sub", "Cross-checks (concordance check, not gate):"),
            ("check", "epv_fv and rim_fv within ±25% of dcf_fv — divergence means assumptions are doing the heavy lifting"),
            ("check", "Reverse DCF: implied_growth < your honest forecast for the business"),
            ("check", "If dividend payer: ddm_eligible True AND ddm_fv corroborates DCF"),
            ("check", "mc_p10_fv (10th percentile) still above current price — downside FV still favorable"),
        ],
    ),
    (
        "Stage 5  —  GROWTH  (5% weight, least important, but a tiebreaker)",
        None,
        None,
        [
            ("check", "rev_cagr_10y > 2%   (no secular decline)"),
            ("check", "fcf_cagr_5y > 5%"),
            ("check", "gross_margin_trend ≥ 0   (not eroding)"),
            ("check", "fundamental_growth (reinvestment × ROIC) > 3%   — the only growth number that ties to capital allocation"),
        ],
    ),
    (
        "Stage 6  —  Composite & Decision",
        None,
        None,
        [
            ("check", "_composite_score ≥ 60 → BUY band;  40–59 → LEAN BUY;  < 40 → PASS"),
            ("check", "_gates_passed count ≥ 15 / 23   (sanity-check vs composite — they should agree)"),
            ("check", "No thesis_breaker flag from narrative.py"),
            ("check", "Macro regime (macro_regime) not hostile for the sector — if it is, halve position size"),
            ("sub", "Position sizing (model already computes this in portfolio.py):"),
            ("check", "Weight = MoS × MC confidence × composite/100, clamped 1–8%"),
            ("check", "Check concentration_flag and sector HHI before adding"),
        ],
    ),
    (
        "Stage 7  —  Ongoing monitoring  (per holding, monthly)",
        None,
        "The portfolio_tracker.py module already produces these alerts — just act on them.",
        [
            ("check", "rating_change_alert fires → re-run full checklist"),
            ("check", "thesis_break_alert fires → exit unless you can defend the original thesis in writing"),
            ("check", "price_alert (drawdown) → check moat metrics first; price drops with intact moat = add, with eroding ROIC = trim"),
            ("check", "Quarterly: rerun and watch roic_by_year, gross_margin_trend, sbc_pct_rev, shares_cagr_5y — these are the slow leaks"),
        ],
    ),
    (
        "Priority cheat-sheet  (when time-boxed)",
        None,
        "If you only get 5 minutes per name, look at these in order:",
        [
            ("check", "1.  roic_cv and spread — is there a moat?"),
            ("check", "2.  cash_conv and accruals — are earnings real?"),
            ("check", "3.  mos and _price_fv — am I being paid to take the risk?"),
            ("check", "4.  insider_pct and shareholder_yield — is management aligned?"),
            ("check", "5.  _composite_score — does the model agree?"),
            ("para", "Everything else is supporting evidence."),
        ],
    ),
]


class WorkflowPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Value-Investing Workflow", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "stock-analysis-model  ·  scoring weights: Moat 40 / Quality 20 / Valuation 20 / Ownership 15 / Growth 5", align="C")
        self.set_text_color(0, 0, 0)


def build():
    pdf = WorkflowPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVuMono", "", str(FONT_DIR / "DejaVuSansMono.ttf"))
    pdf.add_page()

    # Title block
    pdf.set_font("DejaVu", "B", 20)
    pdf.cell(0, 10, "Value-Investing Workflow", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 5, SECTIONS[0][2])
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for heading, _flag, intro, items in SECTIONS[1:]:
        # Section heading
        if pdf.get_y() > 250:
            pdf.add_page()
        pdf.set_font("DejaVu", "B", 13)
        pdf.set_fill_color(235, 240, 248)
        pdf.cell(0, 8, "  " + heading, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        if intro:
            pdf.set_font("DejaVu", "", 10)
            pdf.set_text_color(70, 70, 70)
            pdf.multi_cell(0, 5, intro)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

        for kind, text in items:
            pdf.set_x(pdf.l_margin)
            if kind == "check":
                pdf.set_font("DejaVu", "", 11)
                pdf.multi_cell(0, 6, f"{CHECK}   {text}", new_x="LMARGIN", new_y="NEXT")
            elif kind == "sub":
                pdf.ln(1)
                pdf.set_x(pdf.l_margin)
                pdf.set_font("DejaVu", "B", 10)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
            elif kind == "para":
                pdf.set_font("DejaVu", "", 10)
                pdf.set_text_color(70, 70, 70)
                pdf.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
                pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
