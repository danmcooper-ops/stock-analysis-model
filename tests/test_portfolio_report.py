# tests/test_portfolio_report.py
"""The nightly top-picks report (step 07a) never treats a missing sector as a
sector: no "Unknown" row, group or cluster, and the excluded names are
counted."""
import pandas as pd

from scripts import portfolio_report as pr


def _stocks():
    return [
        {'ticker': 'A', 'sector': 'Tech', 'rating': 'BUY', '_composite_score': 70},
        {'ticker': 'B', 'sector': 'Tech', 'rating': 'LEAN BUY', '_composite_score': 60},
        {'ticker': 'C', 'sector': 'Energy', 'rating': 'BUY', '_composite_score': 65},
        {'ticker': 'X', 'sector': None, 'rating': 'BUY', '_composite_score': 62},
        {'ticker': 'Y', 'sector': '', 'rating': 'LEAN BUY', '_composite_score': 55},
    ]


def test_concentration_summary_excludes_missing_sector(capsys):
    pr.print_concentration_summary(_stocks(), total_rated=100)
    out = capsys.readouterr().out
    assert 'Unknown' not in out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith(('Tech', 'Energy'))]
    # % of the 3 names with a sector, not of all 5
    assert '66.7%' in lines[0] and '33.3%' in lines[1]
    assert '2 with no sector data' in out


def test_concentration_summary_all_sectored_has_no_footer(capsys):
    pr.print_concentration_summary(_stocks()[:3], total_rated=100)
    assert 'no sector data' not in capsys.readouterr().out


def test_cluster_analysis_does_not_group_missing_sector(capsys):
    tks = ['A', 'B', 'C', 'X', 'Y']
    corr = pd.DataFrame(0.95, index=tks, columns=tks)   # everything correlated
    groups = pr.print_cluster_analysis(corr, _stocks())
    out = capsys.readouterr().out
    assert set(groups) == {'Tech', 'Energy'}
    assert 'Unknown' not in out and 'X <-> Y' not in out
    assert '2 ticker(s) with no sector data' in out


def test_summary_counts_real_sectors_only(capsys):
    pr.print_summary(_stocks(), high_corr_pairs=[], drawdown_rows=[], missing_tickers=[],
                     results_date='2026-09-25', ratings_filter=['BUY', 'LEAN BUY'])
    out = ' '.join(capsys.readouterr().out.split())   # the summary is word-wrapped
    assert 'spanning 2 sectors' in out
    assert 'Tech at 66.7% of the bucket (2 with no sector data excluded)' in out
    assert 'Unknown' not in out
