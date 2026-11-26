#!/usr/bin/env python3
"""Stage metrics: retrieval precision against the teacher and the decision model's fidelity to the teacher.

Gold standard: the Nemotron teacher's labels (label.py), one model's judgments.

Retrieval, per workflow and ranker (lists.py): **P@20**, the mean over queries of the share of the query's top-20
list whose teacher reasonableness answer is yes (over the labeled pairs of the list; unlabeled pairs are counted),
and the mean teacher 0-6 score of the list (0 unless reasonable, else 1 + cancer type + burden + treatment history +
2 x biomarker), each with a percentile bootstrap CI over queries; paired differences between rankers resample the
same queries; queries whose list has fewer than 20 pairs are counted.

Decision model, on the merged top-20 (the union of the three rankers' lists, both workflows): for each of the five
match labels, boilerplate (one row per unique combination of the two boilerplate texts; yes = potentially eligible,
the teacher's ``excluded`` inverted) and the per-criterion lines (yes = the patient does not clearly fail the line;
overall and by inclusion/exclusion type): prevalence of yes, AUROC, average precision of the yes class and of the no
class (score 1 - P(yes)) beside their prevalences, Brier score, expected calibration error over equal-mass bins,
calibration intercept (calibration-in-the-large: logistic model of the label with offset logit(P), slope fixed at 1)
and slope (logistic recalibration of the label on logit(P)), mean P(yes) against the observed rate, and a
reliability table. CIs are cluster bootstraps: pooled rows are clustered by patient (person), per-workflow rows by
query. Counts and metrics only: no clinical text or identifiers reach metrics.json or the report.

Full-denominator corpus (prepare.py corpus --all-queries): the retrieval comparison (``retrieval``: (a), (b), (c) and
their paired differences) covers the comparison queries only (every query with ``--compare-all``), with CIs from
bootstrapping the same clusters as ``full_pipeline``; ``full_pipeline`` adds (c)'s P@20 and mean score over
every query, with CIs from bootstrapping clusters of queries (patient-centric: the index events of one person;
trial-centric: the cohorts of one trial); the decision-model tables cover every labeled pair of the merged union, as
always. Without ``--criteria-labels`` and ``--decide-criteria`` (run.py --no-criteria) there are no per-criterion
rows.

By ranker (v4; user, 2026-11-26: the decision model against the teacher "on the A (embedding gemma only) and B
(trialspace only) denominators"): ``decision.by_ranker.<ranker>`` holds the same tables (``pooled``,
``patient_centric``, ``trial_centric``; the five match labels and boilerplate, no per-criterion rows) over one
ranker's own top-20 lists instead of the merged union, with the same clusters, and ``counts``: the ranker's list rows,
unique pairs and pairs with match labels, and its boilerplate combinations (every combination among its pairs whose
cohort has boilerplate criteria), those with a decision score and those with a teacher label (the boilerplate rows'
n). A run without new boilerplate labels (run.py --no-new-boilerplate) covers only the combinations an earlier run
labeled.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from matchminer_ai.evaluation.retro_fidelity.common import (RANKERS, TOP, WORKFLOWS, full_denominator, log, private, queries, read_jsonl,  # noqa: E402
                    run_stage, stage, write_json)

VERSION = 'retro-fidelity-metrics-v4'  # v4: decision-model tables by ranker (v3: clustered comparison CIs)
LABELS = ('reasonableness', 'cancer_type', 'burden', 'treatment_history', 'biomarker')
EPS = 1e-6


def ranker_label(ranker, k=None):
    """The plain name of a ranker's top-``TOP`` denominator (``k``: ranker (c)'s retriever depth, lists_c's K)."""
    return {RANKERS[0]: f'A: EmbeddingGemma 2 top {TOP}', RANKERS[1]: f'B: TrialSpace top {TOP}',
            RANKERS[2]: f'C: TrialSpace top {k or "K"} reranked by the decision model'}[ranker]


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #


def match_score(answers):
    """The 0-6 teacher score of five yes/no answers (matchminer_ai.matching.match_labels.match_score)."""
    if answers['reasonableness'] != 'yes':
        return 0
    return 1 + sum(answers[k] == 'yes' for k in ('cancer_type', 'burden', 'treatment_history')) + 2 * (answers['biomarker'] == 'yes')


def per_query(lists, teacher, top=TOP):
    """Per (workflow, ranker, query): precision (reasonable share of the labeled top-``top`` pairs), mean 0-6 score,
    list length and unlabeled count. ``teacher`` maps (patient_id, cohort_id) to {'answers': ..., 'score': ...}."""
    out = {}
    for (workflow, ranker, query), group in lists.groupby(['workflow', 'ranker', 'query_id'], sort=True):
        group = group[group['rank'] <= top]
        labeled = [teacher[k] for k in zip(group.patient_id, group.cohort_id) if k in teacher]
        out[(workflow, ranker, query)] = dict(
            n=len(group), labeled=len(labeled), unlabeled=len(group) - len(labeled),
            precision=(sum(t['answers']['reasonableness'] == 'yes' for t in labeled) / len(labeled)) if labeled else np.nan,
            mean_score=(sum(t['score'] for t in labeled) / len(labeled)) if labeled else np.nan)
    return out


def bootstrap_mean(values, *, draws=2000, seed=42, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(mean=None, low=None, high=None, n=0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), low=float(np.quantile(means, alpha / 2)),
                high=float(np.quantile(means, 1 - alpha / 2)), n=int(len(values)))


def retrieval_metrics(lists, teacher, *, draws=2000, seed=42, top=TOP, cluster_of=None):
    """P@``top`` and mean score per workflow and ranker, and the rankers' paired differences. CIs bootstrap queries,
    or with ``cluster_of`` (a full-denominator corpus: ``cluster_of[workflow](query_id)``) clusters of queries, as
    the full-pipeline table does."""
    rows = per_query(lists, teacher, top)
    table, differences = {}, {}

    def mean(values, keys, workflow):
        if cluster_of is None:
            return bootstrap_mean(values, draws=draws, seed=seed)
        return cluster_bootstrap_mean(values, [cluster_of[workflow](q) for q in keys], draws=draws, seed=seed)

    for workflow in WORKFLOWS:
        queries = sorted({q for w, _, q in rows if w == workflow})
        for ranker in RANKERS:
            keys = [q for q in queries if (workflow, ranker, q) in rows]
            cells = [rows[(workflow, ranker, q)] for q in keys]
            if not cells:
                continue
            table[f'{workflow}/{ranker}'] = dict(
                queries=len(cells), queries_with_fewer_than_top=sum(c['n'] < top for c in cells),
                queries_without_labels=sum(c['labeled'] == 0 for c in cells),
                unlabeled_pairs=sum(c['unlabeled'] for c in cells),
                precision_at_top=mean([c['precision'] for c in cells], keys, workflow),
                mean_score=mean([c['mean_score'] for c in cells], keys, workflow))
        for first, second in ((RANKERS[1], RANKERS[0]), (RANKERS[2], RANKERS[1]), (RANKERS[2], RANKERS[0])):
            both = [q for q in queries if (workflow, first, q) in rows and (workflow, second, q) in rows]
            for metric in ('precision', 'mean_score'):
                deltas = [rows[(workflow, first, q)][metric] - rows[(workflow, second, q)][metric] for q in both]
                differences[f'{workflow}/{metric}/{first}-minus-{second}'] = mean(deltas, both, workflow)
    return dict(top=top, table=table, paired_differences=differences,
                clusters=None if cluster_of is None else dict(patient_centric='person', trial_centric='trial'))


def cluster_bootstrap_mean(values, clusters, *, draws=2000, seed=42, alpha=0.05):
    """The mean of ``values`` with a percentile CI from resampling whole clusters with replacement (each draw's
    mean is over every value of the drawn clusters); non-finite values are left out."""
    values, clusters = np.asarray(values, dtype=float), np.asarray(clusters)
    keep = np.isfinite(values)
    values, clusters = values[keep], clusters[keep]
    if not len(values):
        return dict(mean=None, low=None, high=None, n=0, clusters=0)
    _, inverse = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for draw in range(draws):
        chosen = rng.integers(0, len(sums), len(sums))
        means[draw] = sums[chosen].sum() / counts[chosen].sum()
    return dict(mean=float(values.mean()), low=float(np.quantile(means, alpha / 2)),
                high=float(np.quantile(means, 1 - alpha / 2)), n=int(len(values)), clusters=int(len(sums)))


def pipeline_metrics(lists, teacher, cluster_of, *, draws=2000, seed=42, top=TOP, ranker=RANKERS[2]):
    """One ranker's P@``top`` and mean 0-6 score over every query of each workflow, with cluster-bootstrap CIs;
    ``cluster_of[workflow](query_id)`` is a query's cluster."""
    rows = per_query(lists[lists.ranker == ranker], teacher, top)
    table = {}
    for workflow in WORKFLOWS:
        keys = sorted(q for w, _, q in rows if w == workflow)
        if not keys:
            continue
        cells = [rows[(workflow, ranker, q)] for q in keys]
        clusters = [cluster_of[workflow](q) for q in keys]
        table[f'{workflow}/{ranker}'] = dict(
            queries=len(cells), clusters=len(set(clusters)), queries_with_fewer_than_top=sum(c['n'] < top for c in cells),
            queries_without_labels=sum(c['labeled'] == 0 for c in cells),
            unlabeled_pairs=sum(c['unlabeled'] for c in cells),
            precision_at_top=cluster_bootstrap_mean([c['precision'] for c in cells], clusters, draws=draws, seed=seed),
            mean_score=cluster_bootstrap_mean([c['mean_score'] for c in cells], clusters, draws=draws, seed=seed))
    return dict(top=top, ranker=ranker, clusters=dict(patient_centric='person', trial_centric='trial'), table=table)


# --------------------------------------------------------------------------- #
# decision model against the teacher
# --------------------------------------------------------------------------- #


def auroc(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        return np.nan
    ranks = rankdata(p)
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y, score):
    """Average precision of the positive class (sklearn's definition: sum over thresholds of precision x recall
    increment, ties grouped)."""
    y = np.asarray(y, dtype=bool)
    if not y.any():
        return np.nan
    order = np.argsort(-np.asarray(score, dtype=float), kind='mergesort')
    s, t = np.asarray(score, dtype=float)[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # the last index of each run of tied scores
    tp = np.cumsum(t)[last]
    precision = tp / (last + 1)
    recall = tp / t.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def logistic_fit(y, x, offset=None, iterations=50):
    """Newton-Raphson logistic regression of y on [1, x] (x None: intercept only), with an optional offset."""
    y = np.asarray(y, dtype=float)
    design = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), x])
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        eta = offset + design @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        step = np.linalg.lstsq(design.T @ (design * w[:, None]), design.T @ (y - mu), rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration(y, p):
    """(intercept, slope): calibration-in-the-large (offset logit(p), slope fixed at 1) and the slope of the
    logistic recalibration of y on logit(p). (0, 1) for a perfectly calibrated model."""
    y = np.asarray(y, dtype=float)
    if y.min() == y.max():
        return np.nan, np.nan
    z = logit(p)
    intercept = float(logistic_fit(y, None, offset=z)[0])
    slope = float(logistic_fit(y, z)[1])
    return intercept, slope


def reliability(y, p, bins=10):
    """Equal-mass bins of P(yes) in sorted order: n, mean predicted and observed rate per bin, and the ECE."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p, kind='mergesort')
    table, ece = [], 0.0
    for part in np.array_split(order, min(bins, len(p))):
        if not len(part):
            continue
        predicted, observed = float(p[part].mean()), float(y[part].mean())
        table.append(dict(n=int(len(part)), p_min=float(p[part].min()), p_max=float(p[part].max()),
                          mean_predicted=predicted, observed=observed))
        ece += len(part) / len(p) * abs(predicted - observed)
    return table, float(ece)


def point_metrics(y, p, bins=10):
    y, p = np.asarray(y, dtype=bool), np.asarray(p, dtype=float)
    intercept, slope = calibration(y, p)
    _, ece = reliability(y, p, bins)
    return dict(auroc=auroc(y, p), auprc_yes=average_precision(y, p), auprc_no=average_precision(~y, 1 - p),
                brier=float(np.mean((p - y) ** 2)), ece=ece, calibration_intercept=intercept, calibration_slope=slope)


def clustered_bootstrap(y, p, clusters, *, draws=1000, seed=42, bins=10, alpha=0.05):
    """Point metrics with percentile CIs from resampling whole clusters with replacement (no rows: n 0 only)."""
    y, p, clusters = np.asarray(y, dtype=bool), np.asarray(p, dtype=float), np.asarray(clusters)
    if not len(y):
        return dict(n=0, clusters=0, bootstrap_draws=draws)
    point = point_metrics(y, p, bins)
    _, inverse = np.unique(clusters, return_inverse=True)
    # each cluster's row indices in ascending order (a stable sort by cluster, split at the cluster boundaries)
    members = np.split(np.argsort(inverse, kind='stable'), np.cumsum(np.bincount(inverse))[:-1]) if len(y) else []
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in point}
    degenerate = 0
    for _ in range(draws):
        chosen = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if y[chosen].all() or not y[chosen].any():
            degenerate += 1
            continue
        for name, value in point_metrics(y[chosen], p[chosen], bins).items():
            samples[name].append(value)
    table, ece = reliability(y, p, bins)
    out = dict(n=int(len(y)), clusters=int(len(members)), prevalence_yes=float(y.mean()) if len(y) else None,
               prevalence_no=float(1 - y.mean()) if len(y) else None, mean_predicted=float(p.mean()) if len(y) else None,
               observed_rate=float(y.mean()) if len(y) else None, bootstrap_draws=draws,
               degenerate_draws=degenerate, ece_bins=f'{bins} equal-mass bins of P(yes)', reliability=table)
    for name, value in point.items():
        values = np.asarray([v for v in samples[name] if np.isfinite(v)])
        out[name] = dict(value=None if not np.isfinite(value) else float(value),
                         low=float(np.quantile(values, alpha / 2)) if len(values) else None,
                         high=float(np.quantile(values, 1 - alpha / 2)) if len(values) else None)
    return out


def boilerplate_yes(excluded):
    """The decision model's boilerplate polarity: yes = potentially eligible = the teacher did not find the patient
    clearly excluded."""
    if type(excluded) is not bool:
        raise ValueError('boilerplate labels need a boolean excluded answer')
    return not excluded


def criterion_yes(answer):
    """Question 7 polarity: yes = the patient does not clearly fail the line."""
    if answer not in ('yes', 'no'):
        raise ValueError('criterion answers are yes or no')
    return answer == 'yes'


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #


def read_labels(directory, fields):
    """Selected fields of a label directory's labels.jsonl (streamed; reasoning and prompts are not kept)."""
    rows = []
    with (Path(directory) / 'labels.jsonl').open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows.append({k: row.get(k) for k in fields})
    return rows


def text_codes(texts):
    """id -> the same integer for identical texts (an exact stand-in for decide.py's digest of the text)."""
    codes = {}
    return {key: codes.setdefault(text, len(codes)) for key, text in texts.items()}


def combination_count(pairs, patient_code, cohort_code):
    """The boilerplate combinations among ``pairs`` (patient_id, cohort_id): decide.py's scoring units, one per
    distinct pair of texts, for the pairs whose cohort has boilerplate criteria (``cohort_code`` holds only those)."""
    import pandas as pd
    codes = pd.DataFrame(dict(patient=pairs.patient_id.map(patient_code).to_numpy(),
                              cohort=pairs.cohort_id.map(cohort_code).to_numpy()))
    return int(len(codes.dropna().drop_duplicates()))


def table_rows(metrics):
    def cell(value):
        if not value or value.get('value') is None:
            return 'n/a'
        return f"{value['value']:.3f} [{value['low']:.3f}, {value['high']:.3f}]" if value['low'] is not None \
            else f"{value['value']:.3f}"
    lines = ['| Label | n | clusters | prevalence yes | AUROC | AUPRC (yes) | AUPRC (no) | Brier | ECE | '
             'cal. intercept | cal. slope | mean P(yes) | observed |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, m in metrics.items():
        if not m.get('n'):
            lines.append(f'| {name} | 0 | | | | | | | | | | | |')
            continue
        lines.append(f"| {name} | {m['n']:,} | {m['clusters']:,} | {m['prevalence_yes']:.3f} | {cell(m['auroc'])} | "
                     f"{cell(m['auprc_yes'])} | {cell(m['auprc_no'])} (base {m['prevalence_no']:.3f}) | {cell(m['brier'])} | "
                     f"{cell(m['ece'])} | {cell(m['calibration_intercept'])} | {cell(m['calibration_slope'])} | "
                     f"{m['mean_predicted']:.3f} | {m['observed_rate']:.3f} |")
    return lines


def report(result, design, k=None):
    """report.md; ``k``: ranker (c)'s retriever depth (lists_c's K), for the by-ranker section's names."""
    names = {RANKERS[0]: '(a) EmbeddingGemma 2 (base)', RANKERS[1]: '(b) TrialSpace round 2',
             RANKERS[2]: '(c) TrialSpace round 2 top-K + decision reasonableness'}
    r = result['retrieval']
    out = ['# Retrospective distillation fidelity', '',  # the Design lines name the patients
           f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `matchminer_ai.evaluation.retro_fidelity.metrics`. "
           'Counts and metrics '
           'only. This directory holds PHI; do not copy it elsewhere.', '', '## Design', '']
    out += [f'- {line}' for line in design]
    titles = {'patient_centric': 'Trials for a patient (patient-centric)',
              'trial_centric': 'Patients for a trial (trial-centric)'}
    note = ('Yes means potentially eligible / not clearly failing; for match questions 2-5, yes means the cohort '
            'specifically requires that dimension and the patient documentedly satisfies it. AUPRC (no) scores '
            '1 - P(yes) against the no class; its chance baseline is the no prevalence shown.')

    def reliability_tables(tables):
        lines = []
        for name, m in tables.items():
            if not m.get('n'):
                continue
            lines += [f'**{name}**', '', '| bin | n | P(yes) range | mean P(yes) | observed yes |',
                      '| --- | --- | --- | --- | --- |']
            for i, b in enumerate(m['reliability'], 1):
                lines.append(f"| {i} | {b['n']:,} | {b['p_min']:.3f}-{b['p_max']:.3f} | {b['mean_predicted']:.3f} | "
                             f"{b['observed']:.3f} |")
            lines.append('')
        return lines

    full = result.get('full_pipeline')
    criteria = any(name.startswith('criterion') for name in result['decision']['pooled'])
    for workflow in WORKFLOWS:
        out += ['', f'## {titles[workflow]}', '', f'### Retrieval: P@{r["top"]} against the teacher\'s reasonableness answer'
                + (' (comparison queries)' if full else ''),
                '', '| Ranker | Queries | P@20 [95% CI] | Mean 0-6 score [95% CI] | Queries < 20 | Unlabeled pairs |',
                '| --- | --- | --- | --- | --- | --- |']
        for key, m in r['table'].items():
            w, ranker = key.split('/')
            if w != workflow:
                continue
            p, s = m['precision_at_top'], m['mean_score']
            out.append(f"| {names[ranker]} | {m['queries']} | {p['mean']:.3f} [{p['low']:.3f}, {p['high']:.3f}] | "
                       f"{s['mean']:.2f} [{s['low']:.2f}, {s['high']:.2f}] | {m['queries_with_fewer_than_top']} | "
                       f"{m['unlabeled_pairs']} |")
        out += ['', 'Paired differences (bootstrap over the same queries):', '', '| Difference | Mean [95% CI] |',
                '| --- | --- |']
        for key, m in r['paired_differences'].items():
            if key.startswith(workflow + '/') and m['mean'] is not None:
                out.append(f"| {key.split('/', 1)[1]} | {m['mean']:+.3f} [{m['low']:+.3f}, {m['high']:+.3f}] |")
        if full:
            cluster = full['clusters'][workflow]
            out += ['', f'### Full pipeline over every query: P@{full["top"]} of {names[full["ranker"]]}', '',
                    f'Every query of this workflow; 95% CIs from bootstrapping {cluster}s (all of a {cluster}\'s queries '
                    'together).', '',
                    f'| Ranker | Queries | {cluster.capitalize()}s | P@20 [95% CI] | Mean 0-6 score [95% CI] | Queries < 20 | '
                    'Unlabeled pairs |', '| --- | --- | --- | --- | --- | --- | --- |']
            m = full['table'].get(f'{workflow}/{full["ranker"]}')
            if m and m['precision_at_top']['mean'] is not None:
                p, s = m['precision_at_top'], m['mean_score']
                out.append(f"| {names[full['ranker']]} | {m['queries']} | {m['clusters']} | {p['mean']:.3f} [{p['low']:.3f}, "
                           f"{p['high']:.3f}] | {s['mean']:.2f} [{s['low']:.2f}, {s['high']:.2f}] | "
                           f"{m['queries_with_fewer_than_top']} | {m['unlabeled_pairs']} |")
        per_line = '; per-criterion: per sampled line of this workflow\'s strata' if criteria else ''
        out += ['', '### Decision model against the teacher (this workflow\'s merged top 20)', '',
                f'One row per unique pair of this workflow (boilerplate: per unique combination of the two boilerplate '
                f'texts{per_line}); 95% CIs from bootstrapping its queries. {note}', '']
        out += table_rows(result['decision'][workflow])
        out += ['', '#### Reliability tables (equal-mass bins of P(yes))', '']
        out += reliability_tables(result['decision'][workflow])
    out += ['## Both workflows pooled', '',
            'Decision model over the union of both workflows\' merged top 20, one row per unique pair; 95% CIs from '
            f'bootstrapping patients (persons). {note}', '']
    out += table_rows(result['decision']['pooled'])
    out += ['', '### Reliability tables (equal-mass bins of P(yes), pooled)', '']
    out += reliability_tables(result['decision']['pooled'])
    by_ranker = result['decision'].get('by_ranker') or {}
    if by_ranker:
        scopes = dict(pooled='Both workflows pooled (95% CIs bootstrap patients)',
                      patient_centric=f'{titles["patient_centric"]} (95% CIs bootstrap queries)',
                      trial_centric=f'{titles["trial_centric"]} (95% CIs bootstrap queries)')
        out += ['## Decision model by ranker denominator', '',
                'The tables above, over one ranker\'s own top-20 lists instead of the merged union: one row per unique '
                'pair of its lists (boilerplate: per unique combination of the two boilerplate texts); no per-criterion '
                f'rows. {note}', '']
        for ranker, entry in by_ranker.items():
            c = entry['counts']
            share = (f" ({c['boilerplate_combinations_labeled'] / c['boilerplate_combinations']:.1%})"
                     if c['boilerplate_combinations'] else '')
            out += [f"### {ranker_label(ranker, k)}", '',
                    f"{c['list_rows']:,} list rows, {c['pairs']:,} unique pairs, {c['pairs_with_match_labels']:,} with "
                    f"teacher match labels. Boilerplate: {c['boilerplate_combinations']:,} combinations among its pairs, "
                    f"{c['boilerplate_combinations_scored']:,} with a decision score, "
                    f"{c['boilerplate_combinations_labeled']:,} with a teacher label{share}.", '']
            for scope, title in scopes.items():
                if entry.get(scope):
                    out += [f'**{title}**', '', *table_rows(entry[scope]), '']
    out += ['## Caveats', '',
            '- The gold standard is one teacher model\'s judgments (Nemotron 3.5 Super), not clinician review; '
            'P@20 measures agreement with the teacher, not true eligibility.',
            '- The trial-side cohorts were extracted by the same teacher model (screen, cohort names, criteria, merge, '
            'per-line boilerplate and inclusion/exclusion tags), so teacher errors in extraction are invisible here.',
            '- The decision model and TrialSpace were distilled from this teacher\'s labels on synthetic data; these '
            'numbers measure fidelity to the teacher on real summaries, not clinical accuracy.',
            '- The validation split is a development set: tuning on it is fine; the test split stays untouched.',
            '- Only pairs on some ranker\'s top-20 list are labeled, so the decision-model metrics are conditional on '
            'retrieval (selection-conditioned)' + (', and the per-criterion lines are a stratified sample (workflow x '
                                                    'ranker) of those pairs, pooled without weights.' if criteria else '.'),
            '- Patient summaries were made by the same teacher family from notes up to 14 days before the plan start; '
            'trial listings are the 2026-10-15 ClinicalTrials.gov export, not the trial as it stood at the index date.',
            '- MatchMiner-AI is a research prioritization tool; scores are screening signals for human review.']
    return '\n'.join(out) + '\n'


def coverage(sample, criteria):
    """Design lines of a full-denominator report: which queries each table covers."""
    counts = {w: (len(queries(sample, w)), len(queries(sample, w, comparison=True))) for w in WORKFLOWS}
    (patients, compared_p), (cohorts, compared_c) = counts['patient_centric'], counts['trial_centric']
    return [f'Queries (full denominator): every index event ({patients:,} patient-centric queries) and every cohort '
            f'({cohorts:,} trial-centric queries); seeded comparison subset: {compared_p:,} index events and '
            f'{compared_c:,} cohorts (corpus/sample.json comparison_patient_queries / comparison_cohort_queries).',
            'Retrieval comparison tables ((a), (b), (c) and their paired differences): the comparison queries only; '
            '95% CIs bootstrap those queries.',
            'Full-pipeline tables ((c) alone): every query; 95% CIs bootstrap persons (patient-centric: all index '
            'events of one patient together) or trials (trial-centric: all cohorts of one trial together).',
            'Decision-model tables: every teacher-labeled pair of the merged union, that is (a) and (b) top 20 of the '
            'comparison queries and (c) top 20 of every query' + (
                '; per-criterion lines: the stratified sample ((a) and (b) strata from the comparison queries, (c) '
                'strata from every query).' if criteria else '; no per-criterion rows (not run).')]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', required=True)
    p.add_argument('--lists', required=True, help='lists.py c stage directory (all three rankers)')
    p.add_argument('--lists-ab', required=True)
    p.add_argument('--retrieval', required=True)
    p.add_argument('--match-labels', nargs='+', required=True)
    p.add_argument('--boilerplate-labels', nargs='+', required=True)
    p.add_argument('--criteria-labels', nargs='+', default=[], help='Per-criterion label directories (none: no '
                   'per-criterion rows; give with --decide-criteria)')
    p.add_argument('--decide-match', required=True)
    p.add_argument('--decide-boilerplate', required=True)
    p.add_argument('--decide-criteria', help='decide.py criteria stage directory (with --criteria-labels)')
    p.add_argument('--draws', type=int, default=1000)
    p.add_argument('--bins', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', required=True)
    stage.add_check_arg(p)
    args = p.parse_args(argv)
    criteria = bool(args.criteria_labels)
    if criteria != bool(args.decide_criteria):
        p.error('give --criteria-labels and --decide-criteria together (or neither: no per-criterion rows)')
    private()
    sources = dict(corpus=args.corpus, lists=args.lists, lists_ab=args.lists_ab, retrieval=args.retrieval,
                   decide_match=args.decide_match, decide_boilerplate=args.decide_boilerplate)
    labels = [('match', args.match_labels), ('boilerplate', args.boilerplate_labels)]
    if criteria:  # the identity has no criteria inputs without them
        sources['decide_criteria'] = args.decide_criteria
        labels.append(('criteria', args.criteria_labels))
    identity = dict(version=VERSION, draws=args.draws, bins=args.bins, seed=args.seed,
                    inputs={k: stage.receipt_id(v) for k, v in sources.items()},
                    labels={kind: [stage.receipt_id(d) for d in dirs] for kind, dirs in labels})
    output = Path(args.output)

    def work():
        import pandas as pd
        patient_rows = read_jsonl(Path(args.corpus) / 'patients.jsonl')
        cohort_rows = read_jsonl(Path(args.corpus) / 'cohorts.jsonl')
        persons = {row['id']: row['person'] for row in patient_rows}
        sample = json.loads((Path(args.corpus) / 'sample.json').read_text())
        full = full_denominator(sample)
        lists = pd.read_parquet(Path(args.lists) / 'lists.parquet')
        teacher = {}
        for directory in args.match_labels:
            for row in read_jsonl(Path(directory) / 'scores.jsonl'):
                if row['score'] != match_score(row['answers']):
                    raise ValueError('A teacher score disagrees with its answers')
                teacher[(row['patient_id'], row['cohort_id'])] = dict(answers=row['answers'], score=row['score'])
        compared = lists
        if full:  # the ranker comparison covers the queries (a) and (b) ran for
            compared = pd.concat([lists[(lists.workflow == w) & lists.query_id.isin(queries(sample, w, comparison=True))]
                                  for w in WORKFLOWS], ignore_index=True)
        clusters = None
        if full:
            trial_of = {row['id']: row['trial_id'] for row in cohort_rows}
            clusters = dict(patient_centric=persons.__getitem__, trial_centric=trial_of.__getitem__)
        result = dict(retrieval=retrieval_metrics(compared, teacher, seed=args.seed, cluster_of=clusters))
        if full:
            result['full_pipeline'] = pipeline_metrics(lists, teacher, clusters, seed=args.seed)
        # --- decision model on the merged top 20
        union = lists[['workflow', 'query_id', 'patient_id', 'cohort_id']].drop_duplicates()
        decide = pd.read_parquet(Path(args.decide_match) / 'scores.parquet').set_index(['patient_id', 'cohort_id'])
        pairs = union[['patient_id', 'cohort_id']].drop_duplicates()
        labeled = pairs[[k in teacher for k in zip(pairs.patient_id, pairs.cohort_id)]]
        bp_teacher = {}
        for directory in args.boilerplate_labels:
            for row in read_labels(directory, ('boilerplate_combination', 'excluded', 'patient_id', 'task')):
                if row['task'] == 'boilerplate':
                    bp_teacher[row['boilerplate_combination']] = dict(yes=boilerplate_yes(row['excluded']),
                                                                      person=persons[row['patient_id']])
        bp = pd.read_parquet(Path(args.decide_boilerplate) / 'scores.parquet')
        lines, crit = [], None
        for directory in args.criteria_labels:
            for row in read_labels(directory, ('patient_id', 'cohort_id', 'number', 'criterion_type', 'criterion', 'task')):
                if row['task'] == 'criteria':
                    lines.append(dict(patient_id=row['patient_id'], cohort_id=row['cohort_id'], number=int(row['number']),
                                      criterion_type=row['criterion_type'], yes=criterion_yes(row['criterion']['answer'])))
        if criteria:
            crit = pd.DataFrame(lines).merge(pd.read_parquet(Path(args.decide_criteria) / 'scores.parquet'),
                                             on=['patient_id', 'cohort_id', 'number'], how='inner', suffixes=('', '_d'),
                                             validate='one_to_one')
            if not crit.criterion_type.eq(crit.criterion_type_d).all():
                raise ValueError('Teacher and decision line types differ')

        def decision_tables(rows, cluster_of):
            out = {}
            keys = list(zip(rows.patient_id, rows.cohort_id))
            for label in LABELS:
                y = [teacher[k]['answers'][label] == 'yes' for k in keys]
                pr = decide.loc[keys, f'p_{label}'].to_numpy()
                out[label] = clustered_bootstrap(y, pr, [cluster_of(k) for k in keys], draws=args.draws,
                                                 seed=args.seed, bins=args.bins)
            return out

        def boilerplate_table(combos, clusters):
            return clustered_bootstrap([bp_teacher[c]['yes'] for c in combos.combination], combos.p_boilerplate.to_numpy(),
                                       clusters, draws=args.draws, seed=args.seed, bins=args.bins)

        def person_boilerplate_table(combos):  # pooled: clustered by the labeled combination's patient
            return boilerplate_table(combos, [bp_teacher[c]['person'] for c in combos.combination])

        def workflow_tables(frame, workflow):
            """One workflow's match and boilerplate tables over the labeled pairs of ``frame`` (the union's or one
            ranker's list rows), clustered by query; and each pair's query."""
            rows = frame[frame.workflow == workflow]
            rows = rows[[k in teacher for k in zip(rows.patient_id, rows.cohort_id)]]
            query_of = dict(zip(zip(rows.patient_id, rows.cohort_id), rows.query_id))
            tables = decision_tables(rows.drop_duplicates(['patient_id', 'cohort_id']), lambda k: query_of[k])
            w_bp = bp.merge(rows[['patient_id', 'cohort_id', 'query_id']], on=['patient_id', 'cohort_id'])
            w_bp = w_bp[w_bp.combination.isin(bp_teacher)].drop_duplicates('combination')
            tables['boilerplate'] = boilerplate_table(w_bp, w_bp.query_id.to_numpy())
            return tables, query_of

        pooled = decision_tables(labeled, lambda k: persons[k[0]])
        combos = bp.drop_duplicates('combination')
        combos = combos[combos.combination.isin(bp_teacher)]
        pooled['boilerplate'] = person_boilerplate_table(combos)
        for name, subset in (() if crit is None else (
                ('criterion (all lines)', crit), ('criterion (inclusion lines)', crit[crit.criterion_type == 'inclusion']),
                ('criterion (exclusion lines)', crit[crit.criterion_type == 'exclusion']))):
            pooled[name] = clustered_bootstrap(subset.yes.to_numpy(), subset.p_yes.to_numpy(),
                                               [persons[p_] for p_ in subset.patient_id], draws=args.draws,
                                               seed=args.seed, bins=args.bins)
        result['decision'] = dict(pooled=pooled)
        for workflow in WORKFLOWS:
            tables, query_of = workflow_tables(union, workflow)
            w_crit = None if crit is None else crit[crit.stratum.str.startswith(workflow + '/')]
            for name, subset in (() if w_crit is None else (
                    ('criterion (all lines)', w_crit),
                    ('criterion (inclusion lines)', w_crit[w_crit.criterion_type == 'inclusion']),
                    ('criterion (exclusion lines)', w_crit[w_crit.criterion_type == 'exclusion']))):
                tables[name] = clustered_bootstrap(
                    subset.yes.to_numpy(), subset.p_yes.to_numpy(),
                    [query_of.get((a, b), a) for a, b in zip(subset.patient_id, subset.cohort_id)],
                    draws=args.draws, seed=args.seed, bins=args.bins)
            result['decision'][workflow] = tables
        # --- the same tables over each ranker's own lists (no per-criterion rows)
        patient_code = text_codes({row['id']: row.get('boilerplate') or '' for row in patient_rows})
        cohort_code = text_codes({row['id']: row['boilerplate'] for row in cohort_rows
                                  if (row.get('boilerplate') or '').strip()})
        by_ranker = result['decision']['by_ranker'] = {}
        for ranker in RANKERS:
            part = lists.loc[lists.ranker == ranker, ['workflow', 'query_id', 'patient_id', 'cohort_id']].drop_duplicates()
            if part.empty:
                continue
            r_pairs = part[['patient_id', 'cohort_id']].drop_duplicates()
            r_labeled = r_pairs[[k in teacher for k in zip(r_pairs.patient_id, r_pairs.cohort_id)]]
            r_scored = bp.merge(r_pairs, on=['patient_id', 'cohort_id']).drop_duplicates('combination')
            r_combos = r_scored[r_scored.combination.isin(bp_teacher)]
            entry = dict(counts=dict(list_rows=int((lists.ranker == ranker).sum()), pairs=int(len(r_pairs)),
                                     pairs_with_match_labels=int(len(r_labeled)),
                                     boilerplate_combinations=combination_count(r_pairs, patient_code, cohort_code),
                                     boilerplate_combinations_scored=int(len(r_scored)),
                                     boilerplate_combinations_labeled=int(len(r_combos))))
            entry['pooled'] = decision_tables(r_labeled, lambda k: persons[k[0]])
            entry['pooled']['boilerplate'] = person_boilerplate_table(r_combos)
            for workflow in WORKFLOWS:
                if (part.workflow == workflow).any():
                    entry[workflow] = workflow_tables(part, workflow)[0]
            by_ranker[ranker] = entry
            log(f'metrics by ranker: {ranker}: {entry["counts"]}')
        counts = dict(union_pairs=int(len(pairs)), union_pairs_with_match_labels=int(len(labeled)),
                      boilerplate_combinations_scored=int(bp.combination.nunique()),
                      boilerplate_combinations_labeled=int(len(combos)))
        if crit is not None:
            counts['criteria_lines'] = int(len(crit))
        counts['union_pairs_without_cohort_boilerplate'] = int(len(pairs) - bp[['patient_id', 'cohort_id']].drop_duplicates().shape[0])
        result['counts'] = counts
        write_json(output / 'metrics.json', result)
        design = json.loads((output.parent / 'design_notes.json').read_text()) if (output.parent / 'design_notes.json').exists() else []
        if full:
            design = design + coverage(sample, criteria)
        depth = ((stage.read_receipt(args.lists) or {}).get('content') or {}).get('k')
        (output / 'report.md').write_text(report(result, design + [f'{name}: {v:,}' for name, v in counts.items()], depth))
        log(f'metrics: {counts}')
        return counts

    run_stage(output, 'retro-metrics', identity, ['metrics.json', 'report.md'], work, check=args.check)


if __name__ == '__main__':
    main()
