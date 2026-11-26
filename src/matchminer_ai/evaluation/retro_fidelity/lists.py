#!/usr/bin/env python3
"""Stages lists-ab, lists-c and lists-d: the four rankers' top-20 lists, their merged union, and the per-criterion
sample.

Rankers (common.RANKERS), per workflow and query:
- (a) ``embeddinggemma2``: base EmbeddingGemma 2's cosine top 20;
- (b) ``trialspace_round2``: TrialSpace round 2's cosine top 20;
- (c) ``trialspace_round2_decision``: TrialSpace's top K (retrieve.py ``--k``) reranked by the decision model's
  reasonableness P(yes), ties by TrialSpace rank (``rerank``);
- (d) ``trialspace_round2_composite``: the same top K reranked by the composite match score the app ranks by
  (``score_match_quality``'s ``match_quality_score``: P(reasonable) x (1 + P(cancer type) + P(burden) + P(treatment
  history) + 2 P(biomarker)) / 6, ``composite``), ties by TrialSpace rank (user, 2026-11-26: "queue the composite
  ranker").

``ab`` needs no GPU: lists (a) and (b), their union ``union_ab.jsonl`` (every pair once, with every list it is on in
``sources``) and the first part of the per-criterion sample. ``c`` follows the decision model's match scores:
list (c), the union so far ``union.jsonl``, the pairs it adds ``union_c_extra.jsonl``, and the sample's last part.
``d`` reads the same scores (every top-K pair has them, so no new decision-model work): list (d), the full union
``union.jsonl`` of the four rankers and the pairs it adds to lists c's ``union_d_extra.jsonl``; each stage's
``lists.parquet`` holds its rankers and the earlier ones. (d) has no per-criterion strata.

Full-denominator corpus (prepare.py corpus --all-queries): lists (a) and (b), and so ``union_ab.jsonl``, are those of
the comparison queries only; list (c) is every query's, so ``union.jsonl`` is union_ab plus every query's (c) pairs
and ``union_c_extra.jsonl`` the (c) pairs not in union_ab.

``--criteria-lines 0``: no per-criterion sample (no criteria_lines.jsonl or criteria_sample.json).

Per-criterion sample (``criteria_sample``): about ``--criteria-lines`` lines in six strata, workflow x ranker. In
each stratum the pairs of that ranker's top-20 lists are taken in a seeded order (sha256 of seed, stratum and
pair) and every checkable line of a pair (decision_model.checkable_items) not already sampled is added, until the
stratum has added ``--criteria-lines / 6`` lines. A line is one patient's line of one set of criteria
(line_key = digest(patient, criteria_digest, number), as training's per-criterion labels key it), so sibling
cohorts with the same criteria and pairs on several lists are labeled once. Strata run in a fixed order; (c)'s
strata run last, skipping lines ``ab`` sampled.
"""
import argparse
import json
from pathlib import Path

from matchminer_ai.evaluation.retro_fidelity.common import (RANKERS, SPLIT, TOP, WORKFLOWS, digest, full_denominator, log, pair_rows, private,  # noqa: E402
                    queries, read_jsonl, run_stage, stage, write_json, write_jsonl)

VERSION = 'retro-fidelity-lists-v1'
STRATA = {'ab': [(w, r) for w in WORKFLOWS for r in RANKERS[:2]], 'c': [(w, RANKERS[2]) for w in WORKFLOWS]}


def top_lists(rankings, top=TOP):
    """Rankers (a) and (b): each embedder's top ``top`` per query, as rows of (workflow, query_id, ranker, rank,
    patient_id, cohort_id, score)."""
    chosen = rankings[rankings['rank'] <= top].rename(columns={'embedder': 'ranker'})
    return chosen[['workflow', 'query_id', 'ranker', 'rank', 'patient_id', 'cohort_id', 'score']].reset_index(drop=True)


def comparison_lists(rankings, sample, top=TOP):
    """Rankers (a) and (b) of the queries they run for: every query, or a full-denominator corpus's comparison
    queries."""
    ab = top_lists(rankings, top)
    if not full_denominator(sample):
        return ab
    keep = None
    for workflow in WORKFLOWS:
        mine = (ab.workflow == workflow) & ab.query_id.isin(queries(sample, workflow, comparison=True))
        keep = mine if keep is None else keep | mine
    return ab[keep].reset_index(drop=True)


def rerank(candidates, probability, top=TOP, ranker=RANKERS[2]):
    """Rankers (c) and (d): each query's candidates (rows with workflow, query_id, rank, patient_id, cohort_id) ordered
    by ``probability[(patient_id, cohort_id)]`` descending, ties by the retriever's rank; the first ``top`` kept."""
    import pandas as pd
    frame = candidates.copy()
    frame['retriever_rank'] = frame['rank']
    frame['score'] = [probability[(p, c)] for p, c in zip(frame.patient_id, frame.cohort_id, strict=True)]
    frame = frame.sort_values(['workflow', 'query_id', 'score', 'retriever_rank'], ascending=[True, True, False, True],
                              kind='mergesort')
    frame['rank'] = frame.groupby(['workflow', 'query_id']).cumcount() + 1
    frame = frame[frame['rank'] <= top].assign(ranker=ranker)
    return pd.DataFrame(frame[['workflow', 'query_id', 'ranker', 'rank', 'patient_id', 'cohort_id', 'score',
                               'retriever_rank']]).reset_index(drop=True)


def composite(scores):
    """Ranker (d)'s score of each pair of a decide.py match ``scores`` frame: the app's ``match_quality_score``
    (matching.rerank's own formula over the five p_<label> columns), keyed by (patient_id, cohort_id)."""
    from matchminer_ai.matching.match_labels import LABELS
    from matchminer_ai.matching.rerank import _scores
    values, _ = _scores([dict(zip(LABELS, row)) for row in zip(*(scores[f'p_{label}'] for label in LABELS))])
    return dict(zip(zip(scores.patient_id, scores.cohort_id), values))


def union(lists, trial_of):
    """Every pair on any list once, with each list position it holds in ``sources``."""
    return list(pair_rows(lists, 'ranker', 'ranker', trial_of))


def criteria_sample(lists, cohorts, strata, per_stratum, seed, taken=()):
    """Line rows of the stratified per-criterion sample (see the module docstring), and per-stratum counts."""
    from matchminer_ai.matching.decision_model import checkable_items
    from matchminer_ai.training.trialspace.candidates import criteria_digest
    taken, rows, counts = set(taken), [], {}
    for workflow, ranker in strata:
        name = f'{workflow}/{ranker}'
        group = lists[(lists.workflow == workflow) & (lists.ranker == ranker)]
        pairs = sorted({(p, c) for p, c in zip(group.patient_id, group.cohort_id)},
                       key=lambda pair: digest([seed, name, *pair]))
        added = used = 0
        for patient, cohort in pairs:
            if added >= per_stratum:
                break
            entity = cohorts[cohort]
            lines = digest_ = None
            for item in checkable_items(entity.get('items') or []):
                digest_ = digest_ or criteria_digest(entity['items'])
                key = digest([patient, digest_, int(item['number'])])
                if key in taken:
                    continue
                taken.add(key)
                lines = (lines or 0) + 1
                rows.append(dict(patient_id=patient, cohort_id=cohort, trial_id=entity['trial_id'], split=SPLIT,
                                 criteria_digest=digest_, number=int(item['number']),
                                 criterion_type=item['criterion_type'], boilerplate=bool(item.get('boilerplate')),
                                 line_key=key, stratum=name))
            if lines:
                added += lines
                used += 1
        counts[name] = dict(pairs=used, lines=added, pairs_available=len(pairs))
    return rows, counts


def k_binding(reranked, k):
    """Where a reranker's picks sit in the retriever's ranking: shares of its top 20 from ranks 1-20, 21-50, 51-80,
    81-k."""
    bins = [(1, 20), (21, 50), (51, 80), (81, k)]
    out = {}
    for workflow, group in reranked.groupby('workflow'):
        ranks = group.retriever_rank
        out[workflow] = {f'{a}-{b}': round(float(((ranks >= a) & (ranks <= b)).mean()), 4) for a, b in bins}
        out[workflow]['queries_with_a_pick_beyond_80'] = int(group[group.retriever_rank > 80].query_id.nunique())
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('ab')
    c = sub.add_parser('c')
    d = sub.add_parser('d')
    for q in (c, d):
        q.add_argument('--decisions', required=True, help='decide.py match stage directory')
    c.add_argument('--lists-ab', required=True)
    d.add_argument('--lists-c', required=True)
    for q in (a, c, d):
        q.add_argument('--retrieval', required=True)
        q.add_argument('--corpus', required=True)
        q.add_argument('--seed', type=int, default=42)
        q.add_argument('--output', required=True)
        stage.add_check_arg(q)
    for q in (a, c):
        q.add_argument('--criteria-lines', type=int, default=10000,
                       help='Target lines over all six strata (0: no per-criterion sample)')
    d.set_defaults(criteria_lines=0)  # no per-criterion strata for (d)
    args = p.parse_args(argv)
    if args.criteria_lines < 0:
        p.error('--criteria-lines must be >= 0')
    private()
    per_stratum = round(args.criteria_lines / 6)
    criteria = args.criteria_lines > 0
    identity = dict(version=VERSION, part=args.command, retrieval=stage.receipt_id(args.retrieval),
                    corpus=stage.receipt_id(args.corpus), top=TOP, criteria_lines=args.criteria_lines, seed=args.seed)
    if args.command == 'c':
        identity.update(decisions=stage.receipt_id(args.decisions), lists_ab=stage.receipt_id(args.lists_ab))
    elif args.command == 'd':
        identity.update(decisions=stage.receipt_id(args.decisions), lists_c=stage.receipt_id(args.lists_c),
                        score='matching.rerank match_quality_score')
    output = Path(args.output)

    def work():
        import pandas as pd
        cohorts = {row['id']: row for row in read_jsonl(Path(args.corpus) / 'cohorts.jsonl')}
        trial_of = {key: row['trial_id'] for key, row in cohorts.items()}
        rankings = pd.read_parquet(Path(args.retrieval) / 'rankings.parquet')
        sample = json.loads((Path(args.corpus) / 'sample.json').read_text())
        ab = comparison_lists(rankings, sample)
        summary = {}
        rows = counts = None
        if args.command == 'd':
            scores = pd.read_parquet(Path(args.decisions) / 'scores.parquet')
            candidates = rankings[rankings.embedder == RANKERS[1]]
            reranked = rerank(candidates, composite(scores), ranker=RANKERS[3])
            reranked.to_parquet(output / 'lists_d.parquet', index=False)
            lists = pd.concat([pd.read_parquet(Path(args.lists_c) / 'lists.parquet'),
                               reranked.drop(columns='retriever_rank')], ignore_index=True)
            lists.to_parquet(output / 'lists.parquet', index=False)
            pairs = union(lists, trial_of)
            write_jsonl(output / 'union.jsonl', pairs)
            before = {(row['patient_id'], row['cohort_id']) for row in read_jsonl(Path(args.lists_c) / 'union.jsonl')}
            extra = [row for row in pairs if (row['patient_id'], row['cohort_id']) not in before]
            write_jsonl(output / 'union_d_extra.jsonl', extra)
            k = int(rankings['rank'].max())
            summary.update(union=len(pairs), union_d_extra=len(extra), k=k, k_binding=k_binding(reranked, k))
        elif args.command == 'ab':
            ab.to_parquet(output / 'lists.parquet', index=False)
            pairs = union(ab, trial_of)
            write_jsonl(output / 'union_ab.jsonl', pairs)
            if criteria:
                rows, counts = criteria_sample(ab, cohorts, STRATA['ab'], per_stratum, args.seed)
            summary.update(union_ab=len(pairs))
        else:
            scores = pd.read_parquet(Path(args.decisions) / 'scores.parquet')
            probability = dict(zip(zip(scores.patient_id, scores.cohort_id), scores.p_reasonableness))
            candidates = rankings[rankings.embedder == RANKERS[1]]
            reranked = rerank(candidates, probability)
            reranked.to_parquet(output / 'lists_c.parquet', index=False)
            lists = pd.concat([ab, reranked.drop(columns='retriever_rank')], ignore_index=True)
            lists.to_parquet(output / 'lists.parquet', index=False)
            pairs = union(lists, trial_of)
            write_jsonl(output / 'union.jsonl', pairs)
            before = {(row['patient_id'], row['cohort_id']) for row in read_jsonl(Path(args.lists_ab) / 'union_ab.jsonl')}
            extra = [row for row in pairs if (row['patient_id'], row['cohort_id']) not in before]
            write_jsonl(output / 'union_c_extra.jsonl', extra)
            if criteria:
                earlier = Path(args.lists_ab) / 'criteria_lines.jsonl'
                if not earlier.exists():
                    raise ValueError('lists ab has no per-criterion sample; give both parts the same --criteria-lines')
                taken = [row['line_key'] for row in read_jsonl(earlier)]
                rows, counts = criteria_sample(lists, cohorts, STRATA['c'], per_stratum, args.seed, taken)
            k = int(rankings['rank'].max())
            summary.update(union=len(pairs), union_c_extra=len(extra), k=k, k_binding=k_binding(reranked, k))
        if criteria:
            write_jsonl(output / 'criteria_lines.jsonl', rows)
            design = dict(target_lines=args.criteria_lines, per_stratum=per_stratum, seed=args.seed,
                          order='sha256([seed, "<workflow>/<ranker>", patient_id, cohort_id])', strata=counts,
                          lines=len(rows), exclusion_lines=sum(r['criterion_type'] == 'exclusion' for r in rows),
                          boilerplate_lines=sum(r['boilerplate'] for r in rows))
            write_json(output / 'criteria_sample.json', design)
            summary.update(criteria_lines=len(rows), criteria_strata=counts)
        write_json(output / 'lists.json', summary)
        log(f'lists {args.command}: {json.dumps(summary)}')
        return summary

    names = {'ab': ['lists.parquet', 'union_ab.jsonl'],
             'c': ['lists.parquet', 'lists_c.parquet', 'union.jsonl', 'union_c_extra.jsonl'],
             'd': ['lists.parquet', 'lists_d.parquet', 'union.jsonl', 'union_d_extra.jsonl']}[args.command]
    run_stage(output, f'retro-lists-{args.command}', identity,
              names + (['criteria_lines.jsonl', 'criteria_sample.json'] if criteria else []) + ['lists.json'], work,
              check=args.check,
              upstream={k: v for k, v in identity.items() if k in ('retrieval', 'corpus', 'decisions', 'lists_ab',
                                                                    'lists_c')})


if __name__ == '__main__':
    main()
