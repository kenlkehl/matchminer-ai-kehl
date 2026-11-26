"""Scoring contract of the retrospective distillation-fidelity evaluation, on fabricated data: P@20 and the 0-6
score, the (c) and (d) reranks, the merged union and the per-criterion sample, label polarity, and the decision-model
metrics (AUROC, AUPRC, Brier, ECE, calibration intercept and slope, clustered bootstrap); the full-denominator mode
(corpus identity, lists, metrics without per-criterion inputs, by ranker); the orchestrator's default plan."""
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest

from matchminer_ai.evaluation.retro_fidelity import lists as rf_lists
from matchminer_ai.evaluation.retro_fidelity import metrics as rf_metrics
from matchminer_ai.evaluation.retro_fidelity import prepare as rf_prepare
from matchminer_ai.labeling.files import write_json, write_jsonl
from matchminer_ai.training.trialspace import stage

YES = dict(reasonableness='yes', cancer_type='yes', burden='no', treatment_history='yes', biomarker='yes')
NO = dict(reasonableness='no', cancer_type='yes', burden='yes', treatment_history='yes', biomarker='yes')


def frame(rows):
    return pd.DataFrame(rows, columns=['workflow', 'query_id', 'ranker', 'rank', 'patient_id', 'cohort_id', 'score'])


def test_match_score_is_zero_unless_reasonable_and_weights_biomarker_twice():
    assert rf_metrics.match_score(YES) == 1 + 1 + 0 + 1 + 2
    assert rf_metrics.match_score(NO) == 0
    from matchminer_ai.matching.match_labels import match_score
    assert match_score(YES) == rf_metrics.match_score(YES) and match_score(NO) == 0


def test_precision_at_20_counts_labeled_pairs_and_short_lists():
    rows = [('patient_centric', 'q1', 'embeddinggemma2', r, 'q1', f'c{r}', 0.0) for r in range(1, 21)]
    rows += [('patient_centric', 'q2', 'embeddinggemma2', r, 'q2', f'c{r}', 0.0) for r in range(1, 6)]  # 5 only
    teacher = {('q1', f'c{r}'): dict(answers=YES if r <= 5 else NO, score=rf_metrics.match_score(YES if r <= 5 else NO))
               for r in range(1, 20)}  # c20 of q1 unlabeled
    teacher.update({('q2', f'c{r}'): dict(answers=YES, score=5) for r in range(1, 6)})
    cells = rf_metrics.per_query(frame(rows), teacher)
    q1 = cells[('patient_centric', 'embeddinggemma2', 'q1')]
    assert (q1['n'], q1['labeled'], q1['unlabeled']) == (20, 19, 1)
    assert q1['precision'] == pytest.approx(5 / 19) and q1['mean_score'] == pytest.approx(25 / 19)
    table = rf_metrics.retrieval_metrics(frame(rows), teacher, draws=200)['table']['patient_centric/embeddinggemma2']
    assert table['queries'] == 2 and table['queries_with_fewer_than_top'] == 1 and table['unlabeled_pairs'] == 1
    assert table['precision_at_top']['mean'] == pytest.approx((5 / 19 + 1) / 2)


def test_rerank_orders_by_probability_with_retriever_rank_ties_and_keeps_top():
    candidates = pd.DataFrame(dict(workflow='trial_centric', query_id='c', rank=range(1, 26),
                                   patient_id=[f'p{i}' for i in range(1, 26)], cohort_id='c'))
    probability = {(f'p{i}', 'c'): (0.9 if i in (25, 3) else 0.1) for i in range(1, 26)}
    out = rf_lists.rerank(candidates, probability)
    assert len(out) == 20 and list(out.patient_id[:3]) == ['p3', 'p25', 'p1']
    assert list(out['rank']) == list(range(1, 21)) and set(out.ranker) == {'trialspace_round2_decision'}
    assert out.retriever_rank.iloc[1] == 25


def test_union_keeps_each_pair_once_with_every_list_position():
    lists = frame([('patient_centric', 'p1', 'embeddinggemma2', 1, 'p1', 'c1', 0.5),
                   ('patient_centric', 'p1', 'trialspace_round2', 3, 'p1', 'c1', 0.4),
                   ('trial_centric', 'c1', 'trialspace_round2', 2, 'p1', 'c1', 0.4),
                   ('trial_centric', 'c1', 'trialspace_round2', 1, 'p2', 'c1', 0.6)])
    pairs = rf_lists.union(lists, {'c1': 'NCT0'})
    assert [(p['patient_id'], p['cohort_id']) for p in pairs] == [('p1', 'c1'), ('p2', 'c1')]
    assert len(pairs[0]['sources']) == 3 and {s['direction'] for s in pairs[0]['sources']} == {
        'patient_to_cohort', 'cohort_to_patient'}
    assert pairs[0]['split'] == 'val' and pairs[0]['trial_id'] == 'NCT0'


def test_criteria_sample_reaches_each_stratum_target_once_per_line():
    items = [dict(number=n, heading='', text=f'line {n}', criterion_type='inclusion' if n % 2 else 'exclusion',
                  boilerplate=False) for n in range(1, 4)]
    cohorts = {f'c{i}': dict(trial_id='NCT0', items=items) for i in range(6)}  # siblings with the same criteria
    lists = frame([('patient_centric', 'p1', r, k, 'p1', f'c{k}', 0.0)
                   for r in ('embeddinggemma2', 'trialspace_round2') for k in range(6)])
    rows, counts = rf_lists.criteria_sample(lists, cohorts, [('patient_centric', 'embeddinggemma2'),
                                                             ('patient_centric', 'trialspace_round2')], 3, seed=1)
    assert len(rows) == 3 and len({r['line_key'] for r in rows}) == 3  # p1's lines once, whichever sibling holds them
    assert counts['patient_centric/embeddinggemma2']['lines'] == 3 and counts['patient_centric/trialspace_round2']['lines'] == 0
    again, _ = rf_lists.criteria_sample(lists, cohorts, [('patient_centric', 'embeddinggemma2')], 3, seed=1,
                                        taken=[r['line_key'] for r in rows])
    assert again == []


def test_label_polarity_of_boilerplate_and_criteria():
    assert rf_metrics.boilerplate_yes(False) is True and rf_metrics.boilerplate_yes(True) is False
    assert rf_metrics.criterion_yes('yes') is True and rf_metrics.criterion_yes('no') is False
    with pytest.raises(ValueError):
        rf_metrics.boilerplate_yes('no')
    with pytest.raises(ValueError):
        rf_metrics.criterion_yes('maybe')


def test_auroc_and_average_precision_match_sklearn_with_ties():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.3
    p = np.round(np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1), 2)  # ties
    assert rf_metrics.auroc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert rf_metrics.average_precision(y, p) == pytest.approx(average_precision_score(y, p))
    assert rf_metrics.average_precision(~y, 1 - p) == pytest.approx(average_precision_score(~y, 1 - p))
    assert np.isnan(rf_metrics.auroc(np.ones(5, bool), np.arange(5)))


def test_calibration_of_a_calibrated_and_an_overconfident_model():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 40000)
    y = rng.random(40000) < p
    intercept, slope = rf_metrics.calibration(y, p)
    assert abs(intercept) < 0.05 and abs(slope - 1) < 0.05
    table, ece = rf_metrics.reliability(y, p, 10)
    assert len(table) == 10 and sum(b['n'] for b in table) == 40000 and ece < 0.02
    sharp = 1 / (1 + np.exp(-3 * rf_metrics.logit(p)))  # overconfident: recalibration slope about 1/3
    assert rf_metrics.calibration(y, sharp)[1] == pytest.approx(1 / 3, abs=0.03)
    shifted = 1 / (1 + np.exp(-(rf_metrics.logit(p) + 1)))  # over-predicts yes: negative intercept
    assert rf_metrics.calibration(y, shifted)[0] == pytest.approx(-1, abs=0.05)
    point = rf_metrics.point_metrics(np.array([1, 0, 1, 0], bool), np.array([0.9, 0.1, 0.6, 0.4]))
    assert point['brier'] == pytest.approx((0.01 + 0.01 + 0.16 + 0.16) / 4) and point['auroc'] == 1.0


def test_clustered_bootstrap_resamples_whole_clusters():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(50), 20)
    p = rng.random(1000)
    y = rng.random(1000) < p
    out = rf_metrics.clustered_bootstrap(y, p, clusters, draws=200, seed=3)
    assert out['n'] == 1000 and out['clusters'] == 50 and out['prevalence_yes'] == pytest.approx(y.mean())
    for name in ('auroc', 'auprc_yes', 'auprc_no', 'brier', 'ece', 'calibration_intercept', 'calibration_slope'):
        cell = out[name]
        assert cell['low'] <= cell['value'] <= cell['high'], name
    assert out['mean_predicted'] == pytest.approx(p.mean()) and len(out['reliability']) == 10


# --------------------------------------------------------------------------- #
# full-denominator mode (fabricated stage directories)
# --------------------------------------------------------------------------- #


@pytest.fixture
def umask():
    """The stages set umask 077 for their process; give the test process its own back."""
    before = os.umask(0o022)
    os.umask(before)
    yield
    os.umask(before)


def made(directory, files=()):
    """A fabricated complete stage directory: its files, then a receipt."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in files:
        if name.endswith('.parquet'):
            value.to_parquet(directory / name, index=False)
        elif name.endswith('.jsonl'):
            write_jsonl(directory / name, value)
        else:
            write_json(directory / name, value)
    stage.write_receipt(directory, 'fabricated', dict(name=directory.name), [])
    return str(directory)


def test_corpus_default_identity_is_unchanged_and_all_queries_keeps_the_default_draw(tmp_path, umask):
    patients = [dict(id=str(100 + i), text=f'p{i}', boilerplate='', split='val', person=i // 2) for i in range(12)]
    cohorts = [dict(id=f'NCT{i // 2}-ctgov-cohort-{i % 2 + 1}', trial_id=f'NCT{i // 2}', text=f'c{i}', split='val')
               for i in range(6)]
    sources = ['--patients', made(tmp_path / 'patients', [('patients.jsonl', patients)]),
               '--cohorts', made(tmp_path / 'cohorts', [('cohorts.jsonl', cohorts)])]
    rf_prepare.main(['corpus', *sources, '--output', str(tmp_path / 'default'), '--patient-queries', '4',
                     '--cohort-queries', '3'])
    assert stage.read_receipt(tmp_path / 'default')['identity'] == dict(  # what completed runs' receipts hold
        version='retro-fidelity-prepare-v1', stage='corpus', patients=stage.receipt_id(tmp_path / 'patients'),
        cohorts=stage.receipt_id(tmp_path / 'cohorts'), seed=42, patient_queries=4, cohort_queries=3)
    default = json.loads((tmp_path / 'default' / 'sample.json').read_text())
    assert list(default) == ['seed', 'method', 'patient_queries', 'cohort_queries', 'patient_pool', 'cohort_pool']
    rf_prepare.main(['corpus', *sources, '--output', str(tmp_path / 'full'), '--all-queries',
                     '--comparison-patient-queries', '4', '--comparison-cohort-queries', '3'])
    full = json.loads((tmp_path / 'full' / 'sample.json').read_text())
    assert full['patient_queries'] == [r['id'] for r in patients] and full['cohort_queries'] == [r['id'] for r in cohorts]
    assert (full['comparison_patient_queries'], full['comparison_cohort_queries']) == (
        default['patient_queries'], default['cohort_queries'])
    with pytest.raises(SystemExit):  # --all-queries needs the comparison counts
        rf_prepare.main(['corpus', *sources, '--output', str(tmp_path / 'bad'), '--all-queries'])


def test_full_denominator_lists_and_metrics_without_criteria(tmp_path, umask):
    rng, draw = np.random.default_rng(0), random.Random(0)
    patients = [dict(id=f'p{i:02d}', text='key', boilerplate=f'b{i % 3}', split='val', person=i // 2) for i in range(24)]
    cohorts = [dict(id=f'NCT{i // 2}-ctgov-cohort-{i % 2 + 1}', trial_id=f'NCT{i // 2}', text='key',
                    boilerplate='cb' if i % 3 else '', items=[], split='val') for i in range(22)]
    sample = dict(patient_queries=[r['id'] for r in patients], cohort_queries=[r['id'] for r in cohorts],
                  comparison_patient_queries=['p03', 'p11'], comparison_cohort_queries=['NCT4-ctgov-cohort-1'])
    corpus = made(tmp_path / 'corpus', [('patients.jsonl', patients), ('cohorts.jsonl', cohorts), ('sample.json', sample)])
    rows = []  # both embedders rank every query here; (a) and (b) must still be the comparison queries' only
    for workflow, queries, documents in (('patient_centric', patients, cohorts), ('trial_centric', cohorts, patients)):
        for embedder in ('embeddinggemma2', 'trialspace_round2'):
            for q in queries:
                for rank, j in enumerate(rng.permutation(len(documents)), 1):
                    pair = (q['id'], documents[j]['id']) if workflow == 'patient_centric' else (documents[j]['id'], q['id'])
                    rows.append((workflow, q['id'], embedder, rank, *pair, 1.0 / rank))
    rankings = pd.DataFrame(rows, columns=['workflow', 'query_id', 'embedder', 'rank', 'patient_id', 'cohort_id', 'score'])
    retrieval = made(tmp_path / 'retrieval', [('rankings.parquet', rankings)])
    pairs = rankings[rankings.embedder == 'trialspace_round2'][['patient_id', 'cohort_id']].drop_duplicates()
    scores = pairs.assign(**{f'p_{k}': rng.random(len(pairs)) for k in rf_metrics.LABELS})
    decide = made(tmp_path / 'decide_match', [('scores.parquet', scores)])
    common = ['--retrieval', retrieval, '--corpus', corpus, '--criteria-lines', '0']
    rf_lists.main(['ab', *common, '--output', str(tmp_path / 'lists_ab')])
    rf_lists.main(['c', *common, '--decisions', decide, '--lists-ab', str(tmp_path / 'lists_ab'),
                   '--output', str(tmp_path / 'lists_c')])
    rf_lists.main(['d', '--retrieval', retrieval, '--corpus', corpus, '--decisions', decide, '--lists-c',
                   str(tmp_path / 'lists_c'), '--output', str(tmp_path / 'lists_d')])
    lists = pd.read_parquet(tmp_path / 'lists_d' / 'lists.parquet')
    c, d = lists[lists.ranker == 'trialspace_round2_decision'], lists[lists.ranker == 'trialspace_round2_composite']
    ab = lists[lists.ranker.isin(['embeddinggemma2', 'trialspace_round2'])]
    assert set(ab.query_id) == {'p03', 'p11', 'NCT4-ctgov-cohort-1'} and len(ab) == 2 * 3 * 20
    assert c.groupby(['workflow', 'query_id']).size().eq(20).all() and c.query_id.nunique() == 24 + 22
    # (d): the same top K by the app's match_quality_score, ties by TrialSpace rank
    from matchminer_ai.matching.rerank import _scores
    one = scores.iloc[0]
    expected = _scores([{label: one[f'p_{label}'] for label in rf_metrics.LABELS}])[0][0]
    assert rf_lists.composite(scores)[(one.patient_id, one.cohort_id)] == pytest.approx(expected)
    assert expected == pytest.approx(one.p_reasonableness * (1 + one.p_cancer_type + one.p_burden
                                                             + one.p_treatment_history + 2 * one.p_biomarker) / 6)
    first = d[(d.workflow == 'patient_centric') & (d.query_id == 'p00')]
    assert len(first) == 20 and first.score.is_monotonic_decreasing and d.query_id.nunique() == 24 + 22
    keys = lambda name: {(r['patient_id'], r['cohort_id']) for r in rf_lists.read_jsonl(name)}  # noqa: E731
    union_ab = keys(tmp_path / 'lists_ab' / 'union_ab.jsonl')
    extra = keys(tmp_path / 'lists_c' / 'union_c_extra.jsonl')
    assert union_ab == set(zip(ab.patient_id, ab.cohort_id))
    assert extra == set(zip(c.patient_id, c.cohort_id)) - union_ab and keys(tmp_path / 'lists_c' / 'union.jsonl') == union_ab | extra
    extra_d = keys(tmp_path / 'lists_d' / 'union_d_extra.jsonl')
    assert extra_d and extra_d == set(zip(d.patient_id, d.cohort_id)) - union_ab - extra
    assert keys(tmp_path / 'lists_d' / 'union.jsonl') == union_ab | extra | extra_d
    assert not list(tmp_path.glob('lists_*/criteria_*'))
    match, bp, combos, bp_scores = [], [], set(), []
    for name, pair_set in (('match_ab', union_ab), ('match_c', extra), ('match_d', extra_d)):
        rows = []
        for p, k in sorted(pair_set):
            answers = {label: draw.choice(['yes', 'no']) for label in rf_metrics.LABELS}
            rows.append(dict(patient_id=p, cohort_id=k, answers=answers, score=rf_metrics.match_score(answers)))
        match.append(made(tmp_path / name, [('scores.jsonl', rows)]))
    boilerplate = {r['id']: r['boilerplate'] for r in patients + cohorts}
    for p, k in sorted(union_ab | extra | extra_d):
        if boilerplate[k]:
            combination = stage.digest([boilerplate[p], boilerplate[k]])
            bp_scores.append(dict(patient_id=p, cohort_id=k, combination=combination, p_boilerplate=draw.random()))
            if combination not in combos:
                combos.add(combination)
                bp.append(dict(task='boilerplate', boilerplate_combination=combination, excluded=draw.random() < 0.4,
                               patient_id=p))
    out = tmp_path / 'metrics'
    rf_metrics.main(['--corpus', corpus, '--lists', str(tmp_path / 'lists_d'), '--lists-ab', str(tmp_path / 'lists_ab'),
                     '--retrieval', retrieval, '--match-labels', *match,
                     '--boilerplate-labels', made(tmp_path / 'bp', [('labels.jsonl', bp[1:])]),  # one unlabeled
                     '--decide-match', decide, '--decide-boilerplate',
                     made(tmp_path / 'decide_bp', [('scores.parquet', pd.DataFrame(bp_scores))]),
                     '--draws', '20', '--output', str(out)])
    result = json.loads((out / 'metrics.json').read_text())
    assert {k: v['queries'] for k, v in result['retrieval']['table'].items()} == {
        f'{w}/{r}': (2 if w == 'patient_centric' else 1) for w in ('patient_centric', 'trial_centric')
        for r in rf_metrics.RANKERS}  # the ranker comparison: comparison queries only
    full = result['full_pipeline']['table']
    assert (full['patient_centric/trialspace_round2_decision']['queries'],
            full['patient_centric/trialspace_round2_decision']['clusters']) == (24, 12)
    assert (full['trial_centric/trialspace_round2_decision']['queries'],
            full['trial_centric/trialspace_round2_decision']['clusters']) == (22, 11)
    assert result['decision']['pooled']['reasonableness']['n'] == len(union_ab | extra | extra_d)  # every labeled pair
    assert 'patient_centric/precision/trialspace_round2_composite-minus-trialspace_round2_decision' in \
        result['retrieval']['paired_differences']
    assert not any(k.startswith('criterion') for k in result['decision']['pooled']) and 'criteria_lines' not in result['counts']
    assert 'decide_criteria' not in stage.read_receipt(out)['identity']['inputs']
    assert 'Full pipeline over every query' in (out / 'report.md').read_text()
    # each ranker's own lists: its pairs, its boilerplate combinations scored and labeled
    by_ranker = result['decision']['by_ranker']
    assert list(by_ranker) == list(rf_metrics.RANKERS)
    labeled_combos = {row['boilerplate_combination'] for row in bp[1:]}
    for ranker, entry in by_ranker.items():
        mine = lists[lists.ranker == ranker]
        mine_pairs = set(zip(mine.patient_id, mine.cohort_id))
        mine_combos = {stage.digest([boilerplate[p], boilerplate[k]]) for p, k in mine_pairs if boilerplate[k]}
        assert entry['counts'] == dict(list_rows=len(mine), pairs=len(mine_pairs), pairs_with_match_labels=len(mine_pairs),
                                       boilerplate_combinations=len(mine_combos),
                                       boilerplate_combinations_scored=len(mine_combos),
                                       boilerplate_combinations_labeled=len(mine_combos & labeled_combos))
        assert entry['pooled']['reasonableness']['n'] == len(mine_pairs)
        assert entry['pooled']['boilerplate']['n'] == len(mine_combos & labeled_combos)
        for workflow in ('patient_centric', 'trial_centric'):
            rows = mine[mine.workflow == workflow]
            assert entry[workflow]['reasonableness']['n'] == len(set(zip(rows.patient_id, rows.cohort_id)))
            assert entry[workflow]['reasonableness']['clusters'] == rows.query_id.nunique()
    assert '## Decision model by ranker denominator' in (out / 'report.md').read_text()
    assert 'C: TrialSpace top 24 reranked by the decision model' in (out / 'report.md').read_text()  # lists' K
    assert 'D: TrialSpace top 24 reranked by the composite match score' in (out / 'report.md').read_text()


def test_default_run_plan_ranks_and_labels_d_and_reuse_keeps_earlier_labels(tmp_path, monkeypatch):
    monkeypatch.setenv('MMAI_DATA_ROOT', str(tmp_path / 'data'))  # common.EMBEDDERS resolves under it
    from matchminer_ai.evaluation.retro_fidelity import run as rf_run
    root = tmp_path / 'run'
    args = rf_run.parser().parse_args(['--root', str(root), '--trials', 't.csv', '--summaries', 's',
                                       '--denominator', 'd.csv'])
    args.patient_queries, args.cohort_queries, args.training_trials = 500, 200, 'trials.csv'
    jobs = rf_run.plan(args)
    assert jobs['lists_d'].deps == ['lists_c'] and jobs['lists_d'].command[3] == 'd'
    assert {'label_match_d', 'label_boilerplate_d'} <= set(jobs)
    assert jobs['decide_boilerplate'].command[jobs['decide_boilerplate'].command.index('--pairs') + 1] == \
        str(root / 'lists_d' / 'union.jsonl')  # boilerplate over every ranker's pairs
    metrics = jobs['metrics'].command
    assert metrics[metrics.index('--lists') + 1] == str(root / 'lists_d')  # retrieval and decision tables with (d)
    assert str(root / 'labels' / 'match_d') in metrics and str(root / 'labels' / 'boilerplate_d') in metrics
    assert {'label_match_d', 'label_boilerplate_d'} <= set(jobs['metrics'].deps)
    old = tmp_path / 'old'  # --reuse-run --no-new-boilerplate: (d)'s new pairs skip the earlier labels
    write_json(old / 'reused_labels.json', dict(match=['m1'], boilerplate=['b1']))
    args.reuse_run, args.no_new_boilerplate, args.compare_all = str(old), True, False
    reused = rf_run.plan(args)
    assert 'label_boilerplate_d' not in reused and '--reuse-only' in reused['decide_boilerplate'].command
    command = reused['label_match_d'].command
    assert command[command.index('--skip-labeled') + 1] == 'm1'
