# tests/test_issuers.py
"""One row per issuer (data/issuers.py) and the SEC issuer map behind it.

The universe listed some companies several times — ADR beside an OTC
ordinary line, dash-less preferreds and notes, a filer's ETNs, second share
classes — and each copy voted again in sector medians, peer percentiles and
profit pools while contradicting the others on the page.
"""
import io
import json

import pytest

from data import us_listings
from data.issuers import (alias_map, collapse_duplicate_listings,
                          is_otc_preferred_symbol, one_listing_per_issuer)
from models import portfolio_groups as pg
from scripts.analyze_stock import _collapse_listings, _run_sector_exit_multiples

ISSUERS = {
    # SEC order puts BIPH (notes) ahead of BIP (the traded units).
    'BIPH': {'cik': '1', 'exchange': 'NYSE', 'rank': 10},
    'BIP':  {'cik': '1', 'exchange': 'NYSE', 'rank': 11},
    # ADR on NYSE beside the OTC ordinary line.
    'NVO':   {'cik': '2', 'exchange': 'NYSE', 'rank': 20},
    'NONOF': {'cik': '2', 'exchange': 'OTC', 'rank': 21},
    # OTC-only issuer: the dormant ADR is listed first.
    'DLMAY': {'cik': '3', 'exchange': 'OTC', 'rank': 30},
    'DLMAF': {'cik': '3', 'exchange': 'OTC', 'rank': 31},
    # Share classes.
    'BRK-B': {'cik': '4', 'exchange': 'NYSE', 'rank': 40},
    'BRK-A': {'cik': '4', 'exchange': 'NYSE', 'rank': 41},
    'AAPL':  {'cik': '5', 'exchange': 'Nasdaq', 'rank': 50},
}


def _row(t, dv=None, **kw):
    return dict({'ticker': t, 'avg_dollar_volume_3m': dv}, **kw)


class TestCollapse:
    def test_exchange_then_dollar_volume_then_sec_order(self):
        rows = [_row('BIPH', 0.2e6), _row('BIP', 28e6), _row('NONOF', 90e6),
                _row('NVO', 1e6), _row('DLMAY', 0.01e6), _row('DLMAF', 3.6e6),
                _row('BRK-B', 5e9), _row('BRK-A', 5e9), _row('AAPL', 9e9)]
        kept, folded = collapse_duplicate_listings(rows, ISSUERS)
        assert [r['ticker'] for r in kept] == ['BIP', 'NVO', 'DLMAF', 'BRK-B', 'AAPL']
        assert folded == {'BIPH': 'BIP', 'NONOF': 'NVO', 'DLMAY': 'DLMAF',
                          'BRK-A': 'BRK-B'}
        assert next(r for r in kept if r['ticker'] == 'BRK-B')['listing_aliases'] == ['BRK-A']
        assert 'listing_aliases' not in next(r for r in kept if r['ticker'] == 'AAPL')

    def test_missing_or_bad_dollar_volume_falls_back_to_sec_order(self):
        rows = [_row('DLMAF', None), _row('DLMAY', float('nan'))]
        kept, _ = collapse_duplicate_listings(rows, ISSUERS)
        assert [r['ticker'] for r in kept] == ['DLMAY']

    def test_unknown_tickers_untouched_and_stale_aliases_reset(self):
        rows = [_row('ZZZZ', listing_aliases=['OLD']), _row('AAPL', listing_aliases=['X'])]
        kept, folded = collapse_duplicate_listings(rows, ISSUERS)
        assert [r['ticker'] for r in kept] == ['ZZZZ', 'AAPL'] and folded == {}
        assert all('listing_aliases' not in r for r in kept)

    def test_empty_map_is_a_no_op(self):
        rows = [_row('BRK-A'), _row('BRK-B')]
        assert collapse_duplicate_listings(rows, {})[0] == rows

    def test_alias_map_reads_rows_back(self):
        kept, folded = collapse_duplicate_listings([_row('BRK-A'), _row('BRK-B')], ISSUERS)
        assert alias_map(kept) == folded == {'BRK-A': 'BRK-B'}

    def test_one_listing_per_issuer_keeps_order(self):
        assert one_listing_per_issuer(['NONOF', 'AAPL', 'NVO', 'ZZZZ', 'BRK-A'], ISSUERS) \
            == ['AAPL', 'NVO', 'ZZZZ', 'BRK-A']
        assert one_listing_per_issuer(['A', 'B'], None) == ['A', 'B']


class TestOtcPreferredSymbols:
    MAP = {
        # Ameren Illinois: every public line is a preferred.
        'AILIH': {'cik': '18654', 'exchange': 'OTC', 'rank': 1},
        'AILLM': {'cik': '18654', 'exchange': 'OTC', 'rank': 2},
        # Fannie Mae: common plus preferreds, all OTC.
        'FNMA':  {'cik': '310522', 'exchange': 'OTC', 'rank': 3},
        'FNMAO': {'cik': '310522', 'exchange': 'OTC', 'rank': 4},
        'FNMAS': {'cik': '310522', 'exchange': 'OTC', 'rank': 5},
        # OTC common, foreign ordinary and ADR lines.
        'FMCB':  {'cik': '7', 'exchange': 'OTC', 'rank': 6},
        'DLMAF': {'cik': '8', 'exchange': 'OTC', 'rank': 7},
        'DLMAY': {'cik': '8', 'exchange': 'OTC', 'rank': 8},
        # Exchange-listed five-letter classes.
        'GOOGL': {'cik': '9', 'exchange': 'Nasdaq', 'rank': 9},
        'BELFB': {'cik': '10', 'exchange': 'Nasdaq', 'rank': 10},
    }

    @pytest.mark.parametrize('ticker,expected', [
        ('AILIH', True), ('AILLM', True), ('FNMAO', True),
        ('FNMAS', False),          # S is ambiguous: left to the issuer fold
        ('FNMA', False), ('FMCB', False), ('DLMAF', False), ('DLMAY', False),
        ('GOOGL', False), ('BELFB', False), ('ZZZZP', False),  # unknown
    ])
    def test_symbol_rule(self, ticker, expected):
        assert is_otc_preferred_symbol(ticker, self.MAP) is expected

    def test_preferred_only_issuer_has_no_row(self):
        rows = [_row(t, 1.0) for t in ('AILIH', 'AILLM', 'FMCB')]
        kept, folded = collapse_duplicate_listings(rows, self.MAP)
        assert [r['ticker'] for r in kept] == ['FMCB']
        assert folded == {'AILIH': None, 'AILLM': None}

    def test_common_survives_its_preferreds(self):
        rows = [_row('FNMAO', 9e9), _row('FNMA', 16e6), _row('FNMAS', 1e4)]
        kept, folded = collapse_duplicate_listings(rows, self.MAP)
        assert [r['ticker'] for r in kept] == ['FNMA']
        assert kept[0]['listing_aliases'] == ['FNMAS']      # never a dropped preferred
        assert folded == {'FNMAO': None, 'FNMAS': 'FNMA'}

    def test_exit_multiples_skip_them_too(self):
        assert one_listing_per_issuer(['AILIH', 'AILLM', 'FMCB', 'GOOGL'], self.MAP) \
            == ['FMCB', 'GOOGL']

    def test_pipeline_records_the_drop(self):
        class Prov:
            def __init__(self):
                self.events = []

            def record_event(self, *a):
                self.events.append(a)
        rows = [_row('AILIH'), _row('FMCB')]
        prov = Prov()
        _collapse_listings(rows, self.MAP, prov)
        assert [r['ticker'] for r in rows] == ['FMCB']
        assert prov.events == [('listing_dropped_preferred', 'AILIH', 'sec_issuers')]


class TestPipeline:
    def test_collapse_runs_in_place_and_records_provenance(self):
        class Prov:
            events = []

            def record_event(self, *a):
                self.events.append(a)
        rows = [_row('BRK-A', 1.0), _row('BRK-B', 2.0), _row('AAPL', 3.0)]
        same = rows
        prov = Prov()
        assert _collapse_listings(rows, ISSUERS, prov) == {'BRK-A': 'BRK-B'}
        assert same is rows and [r['ticker'] for r in rows] == ['BRK-B', 'AAPL']
        assert prov.events == [('listing_folded', 'BRK-A', 'sec_issuers', {'kept': 'BRK-B'})]
        assert _collapse_listings(rows, {}, prov) == {}

    def test_exit_multiples_count_each_issuer_once(self):
        def cache(ee, sector='Tech'):
            return {'yf_data': {'info': {'enterpriseToEbitda': ee, 'sector': sector}}}
        issuers = {f'D{i}': {'cik': 'dup', 'exchange': 'OTC', 'rank': i} for i in range(6)}
        qualifying = [f'D{i}' for i in range(6)] + [f'U{i}' for i in range(5)]
        screen = {t: cache(30.0) for t in qualifying[:6]}
        screen.update({f'U{i}': cache(10.0 + i) for i in range(5)})
        with_dups = _run_sector_exit_multiples(qualifying, screen, 0.0)
        one_each = _run_sector_exit_multiples(qualifying, screen, 0.0, issuer_map=issuers)
        assert with_dups['sector_exit_multiples']['Tech'] > \
            one_each['sector_exit_multiples']['Tech']


class TestPortfolioAliases:
    PF = {'id': 'core', 'name': 'Core', 'tickers': ['BRK-A', 'GOOG', 'AAPL'],
          'exclude': []}

    def test_folded_ticker_resolves_to_kept_row(self):
        rows = [{'ticker': 'BRK-B', 'listing_aliases': ['BRK-A']},
                {'ticker': 'GOOGL', 'listing_aliases': ['GOOG']},
                {'ticker': 'AAPL'}]
        res = pg.resolve_members(self.PF, pg.rows_by_ticker(rows))
        assert res['members'] == ['AAPL', 'BRK-B', 'GOOGL'] and res['missing'] == []

    def test_exclude_follows_the_fold(self):
        pf = dict(self.PF, exclude=['GOOG'])
        rows = [{'ticker': 'GOOGL', 'listing_aliases': ['GOOG']}, {'ticker': 'AAPL'},
                {'ticker': 'BRK-B', 'listing_aliases': ['BRK-A']}]
        assert pg.resolve_members(pf, pg.rows_by_ticker(rows))['members'] == ['AAPL', 'BRK-B']

    def test_fold_night_raises_no_membership_events(self):
        prev = [{'ticker': t, 'rating': 'HOLD'} for t in ('BRK-A', 'BRK-B', 'GOOG', 'GOOGL', 'AAPL')]
        now = [{'ticker': 'BRK-B', 'rating': 'HOLD', 'listing_aliases': ['BRK-A']},
               {'ticker': 'GOOGL', 'rating': 'HOLD', 'listing_aliases': ['GOOG']},
               {'ticker': 'AAPL', 'rating': 'HOLD'}]
        assert pg.membership_events([self.PF], pg.rows_by_ticker(now),
                                    pg.rows_by_ticker(prev)) == []


class TestIssuerMap:
    PAYLOAD = {'fields': ['cik', 'name', 'ticker', 'exchange'],
               'data': [[1, 'Brookfield', 'BIPH', 'NYSE'], [1, 'Brookfield', 'BIP', 'NYSE'],
                        [2, 'Novo', 'nvo', 'NYSE'], [2, 'Novo', 'NONOF', 'OTC'],
                        [3, 'Shell co', 'XX', None]]}

    def test_fetch_parses_and_caches(self, tmp_path, monkeypatch):
        calls = []

        def fake_urlopen(req, **kw):
            calls.append(req.full_url)
            return io.BytesIO(json.dumps(self.PAYLOAD).encode())
        monkeypatch.setattr(us_listings.urllib.request, 'urlopen', fake_urlopen)
        cache = str(tmp_path / 'issuers.csv')
        m = us_listings.fetch_issuer_map(cache_path=cache)
        assert m['BIPH'] == {'cik': '1', 'exchange': 'NYSE', 'rank': 0}
        assert m['NVO']['cik'] == '2' and m['NONOF']['exchange'] == 'OTC'
        assert m['XX']['exchange'] is None
        assert us_listings.fetch_issuer_map(cache_path=cache) == m   # from cache
        assert len(calls) == 1

    def test_fetch_failure_without_cache_is_empty(self, tmp_path, monkeypatch):
        def boom(*a, **kw):
            raise OSError('offline')
        monkeypatch.setattr(us_listings.urllib.request, 'urlopen', boom)
        assert us_listings.fetch_issuer_map(cache_path=str(tmp_path / 'x.csv')) == {}

    def test_disabled_by_env(self, monkeypatch):
        from scripts.analyze_stock import _load_issuer_map
        monkeypatch.setenv('ISSUER_COLLAPSE', '0')
        assert _load_issuer_map() == {}


def test_report_ships_aliases_and_evaluator_resolves_them():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert "'listing_aliases': r.get('listing_aliases')" in \
        (root / 'scripts' / 'report_html.py').read_text(encoding='utf-8')
    tpl = (root / 'templates' / 'report.html').read_text(encoding='utf-8')
    assert 'var canon=function(t){return byTk[t]?t:(alias[t]||t);};' in tpl


@pytest.mark.parametrize('ticker', ['BRK-A', 'BRK-B'])
def test_listing_rank_is_deterministic_on_ties(ticker):
    rows = [_row('BRK-A', 1.0), _row('BRK-B', 1.0)]
    kept, _ = collapse_duplicate_listings(rows if ticker == 'BRK-A' else rows[::-1], ISSUERS)
    assert [r['ticker'] for r in kept] == ['BRK-B']
