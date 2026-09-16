"""scripts/gate_na_report.py: masked vs missing N/A split and its flags."""

import json
import sys

import pytest

from scripts import gate_na_report
from scripts.gate_na_report import _na_pcts
from scripts.scoring import APPLICABILITY_FIELDS, GATES, _gate_key

FS = 'Financial Services'
SPREAD = _gate_key('Moat: Spread')             # masked for FS
MOS = _gate_key('Valuation: MoS')              # no predicate
INT_COV = _gate_key('Quality: Int Coverage')   # masked for FS
FCF_YIELD = _gate_key('Valuation: FCF Yield')  # masked for FS


def _row(ticker, sector, **gates):
    return {'ticker': ticker, 'sector': sector, **gates}


def _today():
    """6 rows: Spread N/A only on FS rows (masked-only), MoS N/A on two
    non-FS rows (missing-only), Int Cov N/A on one FS + one non-FS (mixed),
    FCF Yield N/A on two non-FS rows and one FS row."""
    return [
        _row('F1', FS, **{SPREAD: None, MOS: 0.2, INT_COV: None, FCF_YIELD: None}),
        _row('F2', FS, **{SPREAD: None, MOS: 0.1, INT_COV: 5.0, FCF_YIELD: 0.04}),
        _row('A', 'Technology', **{SPREAD: 0.1, MOS: None, INT_COV: None, FCF_YIELD: None}),
        _row('B', 'Technology', **{SPREAD: 0.1, MOS: None, INT_COV: 8.0, FCF_YIELD: None}),
        _row('C', 'Industrials', **{SPREAD: 0.1, MOS: 0.3, INT_COV: 9.0, FCF_YIELD: 0.05}),
        _row('D', 'Industrials', **{SPREAD: 0.1, MOS: 0.3, INT_COV: 9.0, FCF_YIELD: 0.05}),
    ]


def _prior():
    """Same combined N/A on FCF Yield (3/6) but all of it masked FS rows."""
    rows = _today()
    for r, sector, fcf in zip(rows, [FS, FS, FS, 'Technology', 'Industrials', 'Industrials'],
                              [None, None, None, 0.03, 0.05, 0.05], strict=True):
        r['sector'] = sector
        r[FCF_YIELD] = fcf
    return rows


def _gates(*keys):
    return [{'key': k, 'label': k, 'category': 'X'} for k in keys]


def test_masked_only_gate():
    c = _na_pcts(_today(), _gates(SPREAD))[SPREAD]
    assert (c['na'], c['masked'], c['missing']) == (2, 2, 0)
    assert c['masked_pct'] == pytest.approx(100 / 3) and c['missing_pct'] == 0


def test_missing_only_gate():
    c = _na_pcts(_today(), _gates(MOS))[MOS]
    assert (c['na'], c['masked'], c['missing']) == (2, 0, 2)


def test_mixed_gate():
    c = _na_pcts(_today(), _gates(INT_COV))[INT_COV]
    assert (c['na'], c['masked'], c['missing']) == (2, 1, 1)
    assert c['na_pct'] == pytest.approx(c['masked_pct'] + c['missing_pct'])


def test_unknown_key_and_non_dict_rows_count_as_missing():
    rows = _today() + ['not a row']
    c = _na_pcts(rows, _gates('_gate_nope', SPREAD))
    assert (c['_gate_nope']['masked'], c['_gate_nope']['missing']) == (0, 7)
    assert (c[SPREAD]['masked'], c[SPREAD]['missing']) == (2, 1)


def test_jump_keys_on_missing_not_combined(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'output'
    out.mkdir()
    for d, rows in (('2026-01-01', _prior()), ('2026-01-02', _today())):
        (out / f'results_{d}.json').write_text(json.dumps({'results': rows}),
                                              encoding='utf-8')
    before = _na_pcts(_prior(), _gates(FCF_YIELD))[FCF_YIELD]
    after = _na_pcts(_today(), _gates(FCF_YIELD))[FCF_YIELD]
    assert before['na_pct'] == after['na_pct'] == 50.0   # combined: no move
    assert (before['missing'], after['missing']) == (0, 2)

    monkeypatch.setattr(sys, 'argv', ['gate_na_report', str(out / 'results_2026-01-02.json')])
    gate_na_report.main()
    lines = capsys.readouterr().out.splitlines()
    fcf = next(line for line in lines if line.strip().startswith('FCF Yield'))
    assert '⚠ JUMP' in fcf and '+33.3' in fcf
    assert '50.0%' in fcf and '16.7%' in fcf and '33.3%' in fcf
    spread = next(line for line in lines if line.strip().startswith('Spread'))
    assert 'JUMP' not in spread and 'HIGH' not in spread and '+0.0' in spread
    # (gates absent from these rows also move with the FS mix, so no exact count)
    assert any('missing share jumped' in line for line in lines)


def test_high_flag_ignores_masked_share(tmp_path, monkeypatch, capsys):
    rows = [_row(f'F{i}', FS, **{SPREAD: None}) for i in range(5)] + \
           [_row('A', 'Technology', **{SPREAD: None})]
    p = tmp_path / 'results_2026-01-02.json'
    p.write_text(json.dumps(rows), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['gate_na_report', str(p), '--high', '50'])
    gate_na_report.main()
    out = capsys.readouterr().out
    spread = next(line for line in out.splitlines() if line.strip().startswith('Spread'))
    assert '100.0%' in spread and 'HIGH' not in spread   # 83% masked, 17% missing
    assert 'no prior snapshot for deltas' in out


class _Recorder(dict):
    def __init__(self):
        super().__init__()
        self.seen = set()

    def get(self, key, default=None):
        self.seen.add(key)
        return default


def test_applicability_fields_cover_every_predicate():
    """A predicate reading a key outside APPLICABILITY_FIELDS would make the
    store-backed prior misclassify masked rows as missing."""
    rec = _Recorder()
    for g in GATES:
        if g.applicable is not None:
            g.applicable(rec)
    assert rec.seen <= set(APPLICABILITY_FIELDS)
    assert rec.seen == set(APPLICABILITY_FIELDS)
