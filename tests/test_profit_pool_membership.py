# tests/test_profit_pool_membership.py
"""Guard: the profit pool counts each company once, on numbers that are real.

Both defects below were live on the 2026-09-14 run, and both landed on one
sector, which read 52.5% of the universe's operating profit on 16.5% of its
revenue at a 78.5% blended operating margin:

  * the totals summed one row per TICKER, so Freddie Mac went in 22 times
    (FMCC plus 21 preferred series, each row carrying the parent's whole
    $23.3B / $121.8B income statement), Fannie Mae 14 times, Alphabet twice
    via GOOGL/GOOG and ~40 ADR/foreign-ordinary pairs twice each — $3,249B of
    phantom operating income in an $8,237B universe.
  * yfinance's Operating Income and Total Revenue lines disagree about what
    they measure for some financials, giving FMCC a 523% operating margin and
    GS 152% against $17.2B of net income. Those rows were excluded from the
    margin RANKINGS but deliberately left in the totals, where one of them can
    set a whole sector bar's height.

Neither fails loudly: the page renders, the arithmetic is self-consistent, and
only the conclusion is wrong. So pin the membership rules.
"""
import pytest

from scripts.analyze_stock import (assign_pool_membership, _pp_norm_name,
                                   _pp_primary_key)
from scripts.config import PP_MAX_SANE_OP_MARGIN, PP_MARGIN_INCOMPARABLE_SECTORS


def _row(ticker, rev, oi, name=None, sector='Technology', amihud=1.0, **kw):
    d = {
        'ticker': ticker,
        'company_name': name or (ticker + ' Inc'),
        'sector': sector,
        'revenue': rev,
        'operating_income': oi,
        'operating_margin': (oi / rev) if (rev and oi is not None) else None,
        'amihud_illiquidity': amihud,
        'mcap': 1e9,
    }
    d.update(kw)
    return d


def _members(rows):
    return [r['ticker'] for r in rows if r['pp_pool_member']]


class TestDuplicateListings:
    def test_one_issuer_on_many_lines_counts_once(self):
        """Freddie Mac's shape: the common line plus preferred series, each
        carrying the parent's income statement."""
        rows = [_row('FMCC', 23.3e9, 10e9, name='Freddie Mac',
                     sector='Financial Services', amihud=20.0)]
        rows += [_row('FMCC%d' % i, 23.3e9, 10e9, name='Freddie Mac',
                      sector='Financial Services', amihud=16675.0)
                 for i in range(21)]
        s = assign_pool_membership(rows)
        assert _members(rows) == ['FMCC']
        assert s['duplicates'] == 21
        for r in rows[1:]:
            assert r['pp_excluded_reason'] == 'duplicate_listing'
            assert r['pp_duplicate_of'] == 'FMCC'

    def test_the_primary_is_the_line_the_market_trades(self):
        """Amihud illiquidity, not ticker order: it is the one identity-free
        measure of which listing is the real one, and — unlike mcap — it is
        not shared across a poisoned share-count cluster."""
        rows = [_row('GOOG', 350e9, 129e9, name='Alphabet Inc.', amihud=0.021),
                _row('GOOGL', 350e9, 129e9, name='Alphabet Inc.', amihud=0.014)]
        assign_pool_membership(rows)
        assert _members(rows) == ['GOOGL']
        assert rows[0]['pp_duplicate_of'] == 'GOOGL'

    def test_case_and_corporate_form_do_not_split_an_issuer(self):
        """One company's listings disagree about punctuation and suffixes:
        'Nomura Holdings Inc' on NMR, 'NOMURA HOLDINGS INC.' on NRSCF."""
        rows = [_row('NMR', 20e9, 2e9, name='Nomura Holdings Inc',
                     sector='Financial Services', amihud=16.1),
                _row('NRSCF', 20e9, 2e9, name='NOMURA HOLDINGS INC.',
                     sector='Financial Services', amihud=164862.0)]
        assign_pool_membership(rows)
        assert _members(rows) == ['NMR']

    def test_a_duplicate_keeps_its_primarys_economics(self):
        """Membership governs DENOMINATORS only. A second listing is a real
        security with the primary's income statement, so the caller still
        computes its shares and it must not score differently — BRK-A and
        BRK-B cannot diverge."""
        rows = [_row('BRK-A', 371e9, 87e9, name='Berkshire Hathaway Inc.',
                     sector='Financial Services', amihud=0.398),
                _row('BRK-B', 371e9, 87e9, name='Berkshire Hathaway Inc. New',
                     sector='Financial Services', amihud=0.029)]
        assign_pool_membership(rows)
        assert _members(rows) == ['BRK-B']
        # not an artifact, so nothing nulls its numbers downstream
        assert rows[0]['pp_excluded_reason'] == 'duplicate_listing'

    def test_two_companies_are_never_merged_by_name_alone(self):
        """The join needs the same sector, the same normalised name AND the
        same two statement lines to the cent."""
        rows = [_row('AAA', 100e9, 10e9, name='Acme Group'),
                _row('BBB', 100e9, 10.5e9, name='Acme Group'),
                _row('CCC', 100e9, 10e9, name='Apex Group'),
                _row('DDD', 100e9, 10e9, name='Acme Group',
                     sector='Industrials')]
        assign_pool_membership(rows)
        assert sorted(_members(rows)) == ['AAA', 'BBB', 'CCC', 'DDD']

    def test_normalisation_keeps_the_identity_tokens(self):
        assert _pp_norm_name('Berkshire Hathaway Inc. New') == 'berkshire hathaway'
        assert _pp_norm_name('NOMURA HOLDINGS INC.') == 'nomura holdings' \
            or _pp_norm_name('NOMURA HOLDINGS INC.') == 'nomura'
        assert _pp_norm_name('Alphabet Inc.') == 'alphabet'
        assert _pp_norm_name('') == ''
        assert _pp_norm_name(None) == ''

    def test_primary_choice_is_deterministic_without_liquidity(self):
        """Missing Amihud must not make the pick depend on row order."""
        a = _row('ZZZZ', 10e9, 1e9, name='Same Co', amihud=None, mcap=5e9)
        b = _row('AA', 10e9, 1e9, name='Same Co', amihud=None, mcap=5e9)
        assert _pp_primary_key(b) < _pp_primary_key(a)


class TestMarginArtifacts:
    def test_an_impossible_margin_leaves_the_totals(self):
        """FMCC's shape: $121.8B of 'operating income' on $23.3B of revenue."""
        rows = [_row('FMCC', 23.3e9, 121.8e9, sector='Financial Services'),
                _row('JPM', 182e9, 72e9, sector='Financial Services')]
        s = assign_pool_membership(rows)
        assert _members(rows) == ['JPM']
        assert rows[0]['pp_excluded_reason'] == 'om_artifact'
        assert s['artifacts'] == 1

    def test_a_large_negative_margin_is_an_artifact_too(self):
        rows = [_row('BUST', 1e6, -5e6)]
        assign_pool_membership(rows)
        assert rows[0]['pp_excluded_reason'] == 'om_artifact'

    def test_the_sane_band_is_inclusive_at_the_edge(self):
        rows = [_row('EDGE', 100.0, 100.0)]
        assign_pool_membership(rows)
        assert abs(rows[0]['operating_margin']) == PP_MAX_SANE_OP_MARGIN
        assert rows[0]['pp_pool_member'] is True

    def test_a_duplicate_of_an_artifact_is_an_artifact(self):
        """The regression this ordering exists for: every listing of an issuer
        carries the same income statement, so FMCCL inheriting FMCC's $121.8B
        put 12.7% of the sector pool back on a 523% margin while FMCC itself
        was correctly excluded."""
        rows = [_row('FMCC', 23.3e9, 121.8e9, name='Freddie Mac',
                     sector='Financial Services', amihud=20.0),
                _row('FMCCL', 23.3e9, 121.8e9, name='Freddie Mac',
                     sector='Financial Services', amihud=16675.0)]
        assign_pool_membership(rows)
        assert _members(rows) == []
        assert [r['pp_excluded_reason'] for r in rows] == ['om_artifact'] * 2
        # still says which listing it shadows
        assert rows[1]['pp_duplicate_of'] == 'FMCC'


class TestIneligibleRows:
    @pytest.mark.parametrize('kw', [
        {'rev': 0, 'oi': 5e9},
        {'rev': None, 'oi': 5e9},
        {'rev': 100e9, 'oi': None},
    ])
    def test_a_row_without_both_statement_lines_is_not_a_member(self, kw):
        rows = [_row('X', **kw)]
        s = assign_pool_membership(rows)
        assert rows[0]['pp_pool_member'] is False
        assert rows[0]['pp_excluded_reason'] == 'no_financials'
        assert s['no_financials'] == 1

    def test_a_row_without_a_sector_is_not_a_member(self):
        rows = [_row('X', 100e9, 10e9, sector=None)]
        assign_pool_membership(rows)
        assert rows[0]['pp_pool_member'] is False

    def test_fields_are_set_on_every_row(self):
        """Defaults-first, so a stale value from a prior rescore of a snapshot
        can never survive (cf. _compute_pool_share_trajectory)."""
        rows = [_row('X', 100e9, 10e9, pp_pool_member=True,
                     pp_excluded_reason='om_artifact', pp_duplicate_of='STALE')]
        assign_pool_membership(rows)
        assert rows[0]['pp_duplicate_of'] is None
        assert rows[0]['pp_excluded_reason'] is None


class TestMarginComparability:
    def test_financials_are_flagged_not_dropped(self):
        """A bank's revenue is already net of interest expense, so the ratio
        is not a margin — but the profit dollars are real, so the row stays in
        the pool and the page carries the caveat."""
        rows = [_row('JPM', 182e9, 72e9, sector='Financial Services')]
        assign_pool_membership(rows)
        assert rows[0]['pp_margin_comparable'] is False
        assert rows[0]['pp_pool_member'] is True

    def test_other_sectors_are_comparable(self):
        rows = [_row('NVDA', 130e9, 81e9, sector='Technology')]
        assign_pool_membership(rows)
        assert rows[0]['pp_margin_comparable'] is True

    def test_the_incomparable_list_matches_the_gate_mask(self):
        """scripts.scoring._appl_non_financial masks Moat: Margin Advantage on
        the same fact. If one grows a sector the other must learn about it."""
        from scripts.scoring import _appl_non_financial
        for sector in PP_MARGIN_INCOMPARABLE_SECTORS:
            assert _appl_non_financial({'sector': sector}) is False


def test_a_nameless_row_groups_only_with_itself():
    """An empty company_name normalises to nothing, which is no evidence of
    identity — it must not join two companies that happen to share a sector
    and a pair of statement lines."""
    rows = [_row('AAA', 100e9, 10e9, name=''),
            _row('BBB', 100e9, 10e9, name=None)]
    assign_pool_membership(rows)
    assert sorted(_members(rows)) == ['AAA', 'BBB']
