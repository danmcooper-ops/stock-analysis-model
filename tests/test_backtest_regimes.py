# tests/test_backtest_regimes.py
"""The backtest across a change of scoring model.

`measure` scores the rating each snapshot recorded, so after a weight change
the weekly headline pools two models. Snapshots now carry
``provenance.scoring.params_hash``; `measure` reports the corpus per model
and adds the current model re-scored over every snapshot.
"""

import copy
import json

import pytest

import scripts.backtest as bt
from scripts import backtest_cloud as bc
from scripts import param_set
from scripts.param_set import default_params, scoring_params_hash
from tests.test_backtest_store import DATES, TICKERS, corpus  # noqa: F401  (fixture)


def test_params_hash_is_stable_and_tracks_every_input(monkeypatch):
    h = scoring_params_hash()
    assert h == scoring_params_hash() and len(h) == 12
    p = default_params()
    p['score_weight_moat'] += 0.01
    assert scoring_params_hash(p) != h
    # A gate change (weight or threshold) is a model change too.
    import scripts.scoring as sc
    gates = list(sc.GATES)
    gates[0] = gates[0]._replace(weight=gates[0].weight + 1)
    monkeypatch.setattr(sc, 'GATES', gates)
    assert scoring_params_hash() != h


def test_fingerprint_shape():
    fp = param_set.scoring_fingerprint()
    assert set(fp) == {'params_hash', 'git_sha'}
    assert fp['params_hash'] == scoring_params_hash()


def test_snapshot_params_hash_defaults_to_pre_fingerprint():
    assert bt.snapshot_params_hash({'date': '2026-07-06'}) == bt.UNFINGERPRINTED
    assert bt.snapshot_params_hash(
        {'provenance': {'scoring': {'params_hash': 'abc123def456'}}}) == 'abc123def456'


def _m(run_date, h, ph, n=12):
    details = [{'ticker': f'T{i}', 'rating': 'BUY' if i % 2 else 'PASS',
                '_composite_score': float(i), 'excess_return': 0.01 * i}
               for i in range(n)]
    return {'run_date': run_date, 'horizon': h, 'details': details, 'params_hash': ph}


def test_model_regimes_split_the_corpus_by_hash():
    ms = [_m('2026-07-06', 30, 'pre-fingerprint'), _m('2026-07-07', 30, 'pre-fingerprint'),
          _m('2026-10-01', 30, 'aaaaaaaaaaaa')]
    regs = bt.model_regimes(ms)
    assert [(r['params_hash'], r['snapshots'], r['first'], r['last']) for r in regs] == [
        ('pre-fingerprint', 2, '2026-07-06', '2026-07-07'),
        ('aaaaaaaaaaaa', 1, '2026-10-01', '2026-10-01')]
    assert regs[0]['composite_ic']['30']['mean_ic'] == pytest.approx(1.0)
    assert set(regs[0]['rating_buckets']['30']) == {'BUY', 'PASS'}


def test_measure_reports_regimes_and_the_rescored_view(corpus, monkeypatch):  # noqa: F811
    class _NoNet:
        def fetch_history(self, *a, **k):
            return None

    # Stamp the newest snapshot as rated by a different model.
    path = corpus / f'results_{DATES[-1]}.json'
    snap = json.loads(path.read_text(encoding='utf-8'))
    snap['provenance'] = {'scoring': {'params_hash': 'newmodel0001'}}
    path.write_text(json.dumps(snap), encoding='utf-8')
    monkeypatch.setattr(bt, 'USE_SNAPSHOT_STORE', False)

    report = {}
    metrics = bt.run_backtest(str(corpus), [30], _NoNet(), prices_dir=str(corpus / 'prices'),
                              since=None, cache_dir=str(corpus / 'returns'), report=report)
    assert {m['params_hash'] for m in metrics} == {'pre-fingerprint', 'newmodel0001'}
    kept = report['_kept']
    expected = bt._evaluate_params_on_snapshots(copy.deepcopy(kept), default_params(), [30])

    rescored = bt.rescored_current_view(kept, [30])
    summary = bt.build_measure_summary(metrics, [30], None, '2026-10-04', {}, report,
                                       rescored=rescored)
    assert [r['params_hash'] for r in summary['regimes']] == ['pre-fingerprint', 'newmodel0001']
    assert summary['provenance']['current_params_hash'] == scoring_params_hash()
    assert summary['rescored_current']['params_hash'] == scoring_params_hash()
    assert summary['rescored_current']['snapshots'] == len(DATES)
    assert summary['rescored_current']['composite_ic'] == {
        str(h): v for h, v in bt.composite_ic_summary(expected).items()}
    assert '_kept' not in json.dumps(summary, default=str)
    bt.print_regimes_and_rescored(summary['regimes'], rescored)   # smoke: no crash


def test_compare_notices_a_model_change_without_failing():
    base = {'snapshots': [], 'skipped_snapshots': [], 'unmeasured': [], 'coverage': []}
    prior = dict(base, provenance={'current_params_hash': 'aaa'})
    cur = dict(base, provenance={'current_params_hash': 'bbb'},
               regimes=[{'params_hash': 'aaa', 'snapshots': 40, 'first': '2026-07-06', 'last': '2026-10-02'},
                        {'params_hash': 'bbb', 'snapshots': 3, 'first': '2026-10-05', 'last': '2026-10-07'}])
    notes = bc.model_notices(cur, prior)
    assert any('aaa -> bbb' in n for n in notes)
    assert any('pools aaa (40' in n for n in notes)
    assert bc.compare_summaries(cur, prior) == []                 # not a regression
    assert bc.model_notices(dict(base, provenance={'current_params_hash': 'aaa'}), prior) == []
    assert TICKERS                                                 # fixture module imported
