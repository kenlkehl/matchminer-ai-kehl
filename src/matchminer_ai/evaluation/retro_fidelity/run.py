#!/usr/bin/env python3
"""Orchestrator: runs every stage of the retrospective distillation-fidelity evaluation in dependency order.

Each job is one stage module (cohorts, prepare, embed, retrieve, lists, label, decide, metrics); a job whose
``--check`` answers 0 is complete and skipped, so rerunning this command resumes the evaluation. A job already
running in another process (same module and output directory) is waited for, not started twice. Teacher-label
jobs run concurrently on the teacher server and an optional second server (through ``teacher_proxy``, which maps
the labeler's model ID to the second server's and caps it at --proxy-concurrency) and are retried up to
--label-attempts times (the labeler resumes); decision-model jobs run one at a time on --gpu once it has room.
Servers come from the site config (``endpoints.teacher.url``, ``endpoints.secondary_teacher.url``) or the
options. When every teacher job has finished, the orchestrator writes ``oa_pause/release`` (the OA waiter then resumes the OA summarization) and
stops the proxy; metrics follow. Status: ``run_status.json``; log: ``logs/run.log``, one log per job in ``logs/``.

A synthetic run (``--synthetic-corpus``, no PHI) takes its patients and ctgov cohorts from the n=5000 run's TrialSpace
corpus instead of the extraction, patients and cohort-table stages, so the Benchmarking tab of the dashboard's
synthetic mode has a run to show. ``--teacher-after`` holds the label jobs until the named files exist (other teacher
work finishes first); ``--embed-device`` encodes on GPUs instead of the CPU.

Rankers (lists.py): (a) EmbeddingGemma 2 and (b) TrialSpace top 20; (c) TrialSpace's top K reranked by the decision
model's P(reasonable); (d) the same top K reranked by the composite match score the app ranks by (lists d, from the
same decision scores). Every ranker's top-20 pairs get teacher labels (labels match_ab, match_c, match_d: each the
pairs the earlier lists did not have), and metrics reports retrieval and the decision model per ranker as well as
over their union.

Full-denominator mode (``--all-queries`` with ``--comparison-patient-queries N --comparison-cohort-queries M``): every
patient and every cohort is a query, the full pipeline (c) and the app's ordering (d) rank and are labeled for all of
them, and rankers (a) and (b) only for the seeded comparison subset (prepare.py corpus). ``--no-criteria`` (or
``--criteria-lines 0``) leaves out the per-criterion sample, its teacher labels and decision scores. ``--extraction-workers N`` passes ``--workers N`` to
the extraction job (cohorts.py; part of its identity with ``--extraction-source``).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from matchminer_ai.evaluation.retro_fidelity.common import EMBEDDERS, hold_gb300, log, private, private_dir, write_json
from matchminer_ai.site import load_site_config
from matchminer_ai.training.paths import child_env

PACKAGE = 'matchminer_ai.evaluation.retro_fidelity'
MODEL = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'
FP8 = 'RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-FP8'
# Per-job ceilings on each server (concurrency control files); the labelers' adaptive controllers work below them.
CAPS = {'match_ab': (512, 6), 'boilerplate_ab': (320, 4), 'criteria_ab': (128, 2),
        'match_c': (256, 2), 'boilerplate_c': (128, 1), 'criteria_c': (64, 1),
        'match_d': (256, 2), 'boilerplate_d': (128, 1)}


class Job:
    def __init__(self, name, command, deps=(), kind='cpu', output=None, attempts=1):
        self.name, self.command, self.deps, self.kind = name, command, list(deps), kind
        self.output, self.attempts = output, attempts
        self.state, self.process, self.tries, self.started, self.finished, self.code = 'pending', None, 0, None, None, None
        self.held = False  # a label job held by --teacher-after (logged once)


def checked(job):
    return subprocess.run(job.command + ['--check'], capture_output=True, text=True, env=child_env()).returncode


def running_elsewhere(job):
    """Another process running the same stage module on the same output directory (not one of ours)."""
    module = job.command[job.command.index('-m') + 1]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode(errors='ignore').split('\0')
        except OSError:
            continue
        if module in args and job.output in args:
            return int(proc.name)
    return None


def plan(args):
    py, root = args.python, Path(args.root)
    e = lambda *parts: str(root.joinpath(*parts))  # noqa: E731
    mod = lambda name: [py, '-m', f'{PACKAGE}.{name}']  # noqa: E731
    if args.synthetic_corpus:
        jobs = [Job('patients', [*mod('prepare'), 'synthetic-patients', '--corpus', args.synthetic_corpus, '--split',
                                 args.synthetic_split, '--output', e('patients')], output=e('patients')),
                Job('cohort_table', [*mod('prepare'), 'synthetic-cohorts', '--corpus', args.synthetic_corpus,
                                     '--output', e('cohort_table')], output=e('cohort_table'))]
    else:
        extraction = [*mod('cohorts'), '--trials', args.trials, '--servers', e('servers', 'extraction_servers.json'),
                      '--output', e('extraction'), '--holders', e('oa_pause', 'holders.json')]
        if args.extraction_source:
            extraction += ['--source', args.extraction_source]
        if args.extraction_workers is not None:
            extraction += ['--workers', str(args.extraction_workers)]
        jobs = [Job('extraction', extraction, kind='gb300', output=e('extraction')),
                Job('patients', [*mod('prepare'), 'patients', '--summaries', args.summaries, '--denominator',
                                 args.denominator, '--output', e('patients')], output=e('patients')),
                Job('cohort_table', [*mod('prepare'), 'cohorts', '--extraction', e('extraction'), '--exclude-trials',
                                     str(args.training_trials), '--output', e('cohort_table')], ['extraction'],
                    output=e('cohort_table'))]
    queries = (['--all-queries', '--compare-all'] if args.compare_all else
               ['--all-queries', '--comparison-patient-queries', str(args.comparison_patient_queries),
                '--comparison-cohort-queries', str(args.comparison_cohort_queries)] if args.all_queries else
               ['--patient-queries', str(args.patient_queries), '--cohort-queries', str(args.cohort_queries)])
    jobs += [Job('corpus', [*mod('prepare'), 'corpus', '--patients', e('patients'), '--cohorts',
                           e('cohort_table'), '--output', e('corpus'), '--seed', str(args.seed), *queries],
                ['patients', 'cohort_table'], output=e('corpus'))]
    embeds, devices = [], args.embed_device.split(',')
    for embedder in EMBEDDERS:
        for side, source, deps in (('patients', e('patients'), ['patients']), ('cohorts', e('corpus'), ['corpus'])):
            name = f'embed_{embedder}_{side}'
            device = devices[len(embeds) % len(devices)]
            embeds.append(name)
            jobs.append(Job(name, [*mod('embed'), '--input', source, '--side', side, '--embedder', embedder,
                                   '--device', device, '--threads', str(args.threads), '--output',
                                   e('embeddings', embedder, side)], deps, output=e('embeddings', embedder, side)))
    jobs += [Job('retrieve', [*mod('retrieve'), '--corpus', e('corpus'), '--embeddings', e('embeddings'),
                              '--k', str(args.k), '--output', e('retrieval')], ['corpus', *embeds], output=e('retrieval')),
             Job('lists_ab', [*mod('lists'), 'ab', '--retrieval', e('retrieval'), '--corpus', e('corpus'),
                              '--criteria-lines', str(args.criteria_lines), '--seed', str(args.seed), '--output',
                              e('lists_ab')], ['retrieve'], output=e('lists_ab'))]
    decide = [*mod('decide')]
    gpu = ['--corpus', e('corpus'), '--gpu', args.gpu, '--min-free-gb', str(args.min_free_gb), '--wait',
           '--token-budget', str(args.token_budget), '--max-gpu-gb', str(args.max_gpu_gb)] + \
        (['--not-before', args.not_before] if args.not_before else [])
    jobs += [Job('decide_match', decide + ['match', '--pairs', e('retrieval', 'decision_pairs.jsonl'), '--output',
                                           e('decide', 'match')] + gpu, ['retrieve'], 'gpu', e('decide', 'match')),
             Job('lists_c', [*mod('lists'), 'c', '--retrieval', e('retrieval'), '--corpus', e('corpus'),
                             '--decisions', e('decide', 'match'), '--lists-ab', e('lists_ab'), '--criteria-lines',
                             str(args.criteria_lines), '--seed', str(args.seed), '--output', e('lists_c')],
                 ['decide_match', 'lists_ab'], output=e('lists_c')),
             # ranker (d): the same top K reranked by the app's composite match score (no new decision scores)
             Job('lists_d', [*mod('lists'), 'd', '--retrieval', e('retrieval'), '--corpus', e('corpus'),
                             '--decisions', e('decide', 'match'), '--lists-c', e('lists_c'), '--seed', str(args.seed),
                             '--output', e('lists_d')], ['lists_c'], output=e('lists_d')),
             Job('decide_boilerplate', decide + ['boilerplate', '--pairs', e('lists_d', 'union.jsonl'), '--output',
                                                 e('decide', 'boilerplate')] + gpu, ['lists_d'], 'gpu',
                 e('decide', 'boilerplate')),
             *([] if args.no_criteria else [
                 Job('decide_criteria', decide + ['criteria', '--lines', e('lists_ab', 'criteria_lines.jsonl'),
                                                  e('lists_c', 'criteria_lines.jsonl'), '--output',
                                                  e('decide', 'criteria')] + gpu,
                     ['lists_c', 'decide_boilerplate'], 'gpu', e('decide', 'criteria'))])]
    label = [*mod('label'), '--corpus', e('corpus'), '--server_urls_file',
             e('servers', 'teacher_servers.json'), '--teacher-model', MODEL, '--teacher-tokenizer', MODEL,
             '--teacher-context', '250000', '--max_concurrent_per_server', '1024', '--initial-concurrent-per-server', '96',
             '--pending-requests', '0', '--max-auto-pending-requests', '32768', '--max_attempts', '16',
             '--request_timeout', '7200', '--scheduler-adaptive-concurrency',
             *(['--cache-dir', args.cache_dir] if args.cache_dir else []),
             '--label-client-processes', '1', '--holders', e('oa_pause', 'holders.json')]
    specs = [('match_ab', 'match', e('lists_ab', 'union_ab.jsonl'), ['lists_ab'], []),
             ('boilerplate_ab', 'boilerplate', e('lists_ab', 'union_ab.jsonl'), ['lists_ab'], []),
             ('criteria_ab', 'criteria', e('lists_ab', 'criteria_lines.jsonl'), ['lists_ab'], []),
             ('match_c', 'match', e('lists_c', 'union_c_extra.jsonl'), ['lists_c'], []),
             ('boilerplate_c', 'boilerplate', e('lists_c', 'union_c_extra.jsonl'), ['lists_c', 'label_boilerplate_ab'],
              ['--skip-labeled', e('labels', 'boilerplate_ab')]),
             ('match_d', 'match', e('lists_d', 'union_d_extra.jsonl'), ['lists_d'], []),
             ('boilerplate_d', 'boilerplate', e('lists_d', 'union_d_extra.jsonl'),
              ['lists_d', 'label_boilerplate_ab', 'label_boilerplate_c'],
              ['--skip-labeled', e('labels', 'boilerplate_ab'), e('labels', 'boilerplate_c')]),
             ('criteria_c', 'criteria', e('lists_c', 'criteria_lines.jsonl'), ['lists_c'], [])]
    if args.no_criteria:
        specs = [spec for spec in specs if spec[1] != 'criteria']
    for name, task, candidates, deps, extra in specs:
        jobs.append(Job(f'label_{name}', label + ['--task', task, '--candidates', candidates, '--output',
                                                  e('labels', name), '--concurrency-control-file',
                                                  e('servers', f'concurrency_{name}.json')] + extra,
                        deps, 'gb300', e('labels', name), attempts=args.label_attempts))
    labels = [f'label_{name}' for name, *_ in specs]
    criteria = [] if args.no_criteria else ['--criteria-labels', e('labels', 'criteria_ab'), e('labels', 'criteria_c')]
    decided = [] if args.no_criteria else ['--decide-criteria', e('decide', 'criteria')]
    jobs.append(Job('metrics', [*mod('metrics'), '--corpus', e('corpus'), '--lists', e('lists_d'),
                                '--lists-ab', e('lists_ab'), '--retrieval', e('retrieval'), '--match-labels',
                                e('labels', 'match_ab'), e('labels', 'match_c'), e('labels', 'match_d'),
                                '--boilerplate-labels', e('labels', 'boilerplate_ab'), e('labels', 'boilerplate_c'),
                                e('labels', 'boilerplate_d'), *criteria,
                                '--decide-match', e('decide', 'match'), '--decide-boilerplate',
                                e('decide', 'boilerplate'), *decided, '--output', e('metrics')],
                    [*labels, *([] if args.no_criteria else ['decide_criteria']), 'decide_boilerplate'],
                    output=e('metrics')))
    jobs = {job.name: job for job in jobs}
    if args.reuse_run:
        reuse(args, jobs)
    return jobs


REUSED_LABELS = 'reused_labels.json'
PARTS = ('ab', 'c', 'd')  # the label jobs' list stages: (a)(b), (c), (d)


def label_dirs(run) -> dict:
    """The label directories a run's metrics read, by task: those it reused (``reused_labels.json``, written by a
    --reuse-run) and its own."""
    run = Path(run)
    if (run / REUSED_LABELS).is_file():
        return json.loads((run / REUSED_LABELS).read_text())
    return {task: [str(run / 'labels' / f'{task}_{part}') for part in PARTS
                   if (run / 'labels' / f'{task}_{part}' / 'stage_receipt.json').is_file()]
            for task in ('match', 'boilerplate')}


def reuse(args, jobs):
    """--reuse-run: the commands that take what an earlier run of the same inputs already has. The corpus keeps its
    comparison samples (unless --compare-all); the decision model scores, and the teacher labels, only pairs (and
    boilerplate combinations) the earlier run has no answer for; metrics read the earlier labels beside the new
    ones. With --no-new-boilerplate no boilerplate is labeled or scored anew."""
    old = Path(args.reuse_run)
    earlier = label_dirs(old)
    if not args.compare_all:
        jobs['corpus'].command += ['--comparison-from', str(old / 'corpus')]
    for task in ('match', 'boilerplate'):
        jobs[f'decide_{task}'].command += ['--reuse', str(old / 'decide' / task / 'scores.parquet')]
        if task == 'boilerplate' and args.no_new_boilerplate:
            jobs['decide_boilerplate'].command.append('--reuse-only')
            for part in PARTS:
                del jobs[f'label_boilerplate_{part}']
            jobs['metrics'].deps = [d for d in jobs['metrics'].deps if not d.startswith('label_boilerplate')]
            command = jobs['metrics'].command
            at = command.index('--boilerplate-labels') + 1
            end = next((i for i in range(at, len(command)) if command[i].startswith('--')), len(command))
            command[at:end] = earlier[task]
            continue
        for part in PARTS:
            command = jobs[f'label_{task}_{part}'].command
            if '--skip-labeled' in command:
                at = command.index('--skip-labeled') + 1
                command[at:at] = earlier[task]
            else:
                command += ['--skip-labeled', *earlier[task]]
        command = jobs['metrics'].command
        at = command.index(f'--{task}-labels') + 1
        command[at:at] = earlier[task]


def write_reused_labels(args, root, jobs):
    """The label directories this run's metrics read (for a later --reuse-run of this run)."""
    command = jobs['metrics'].command
    out = {}
    for task in ('match', 'boilerplate'):
        at = command.index(f'--{task}-labels') + 1
        end = next((i for i in range(at, len(command)) if command[i].startswith('--')), len(command))
        out[task] = command[at:end]
    write_json(root / REUSED_LABELS, out)


def link_reused(args, root, jobs):
    """--reuse-run: link the earlier run's stages whose inputs and settings are this run's (the cohort extraction,
    patients, patient embeddings), and stop unless each one checks complete here, so a stage is never rewritten in
    the earlier run."""
    old = Path(args.reuse_run)
    links = {'extraction': root / 'extraction', 'patients': root / 'patients',
             **{f'embed_{embedder}_patients': root / 'embeddings' / embedder / 'patients' for embedder in EMBEDDERS}}
    for name, path in links.items():
        if not path.exists() and not path.is_symlink():
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.symlink_to(old / path.relative_to(root))
        if path.is_symlink() and checked(jobs[name]) != 0:
            raise SystemExit(f'{name}: the earlier run\'s stage was made with other inputs or settings; not reusing it')


def query_note(args):
    """The design note's account of the queries (after the patients' description)."""
    if not args.all_queries:
        return (f'{args.patient_queries} sampled for the patient-centric workflow and {args.cohort_queries} cohorts for '
                f'the trial-centric one (seed {args.seed}; corpus/sample.json).')
    if args.compare_all:
        return ('every one is a patient-centric query and every cohort a trial-centric one, and all four rankers '
                '(a)-(d) rank and are labeled for every query (full denominators, --compare-all).')
    return ('every one is a patient-centric query and every cohort a trial-centric one (full denominator: rankers (c), '
            'the full pipeline, and (d), the app\'s ordering, rank and are labeled for all of them); rankers (a) and '
            '(b) run only for a seeded '
            f'comparison subset of {args.comparison_patient_queries} of them and {args.comparison_cohort_queries} '
            f'cohorts (seed {args.seed}; corpus/sample.json comparison_patient_queries / comparison_cohort_queries).')


def criteria_note(args):
    if args.no_criteria:
        return 'Per-criterion: not run (--no-criteria): no criteria sample, teacher labels or decision scores.'
    return (f'Per-criterion sample: about {args.criteria_lines:,} criteria in 6 strata (workflow x ranker; '
            'lists_ab/criteria_sample.json, lists_c/criteria_sample.json)'
            + ('; the (c) strata sample every query\'s list.' if args.all_queries else '.'))



SPLIT_ORDER = ('train', 'validation', 'test')


def patients_note(args) -> str:
    """A real run's patients: the denominator's kind of index event and the splits its summarized events come from
    (read from the summary batches' IDs and the denominator's split column; no patient data is written)."""
    import pandas as pd
    from matchminer_ai.evaluation.demographics import EVAL_KINDS
    from matchminer_ai.training.cohort_format import read_clinical_csv
    kind = EVAL_KINDS.get(Path(args.denominator).name, ('other', ''))[0]
    what = {'soc': 'standard-of-care treatment start', 'enrollment': 'trial enrollment (the start of research '
            'treatment on a trial)'}.get(kind, 'index event')
    ids = set()
    for path in sorted(Path(args.summaries).glob('*.parquet')):
        ids.update(pd.read_parquet(path, columns=['patient_id']).patient_id.astype(str))
    frame = read_clinical_csv(args.denominator, usecols=['pseudo_mrn', 'split'], dtype=str).drop_duplicates()
    splits = sorted(set(frame[frame.pseudo_mrn.isin(ids)].split.dropna()),
                    key=lambda s: (SPLIT_ORDER.index(s) if s in SPLIT_ORDER else len(SPLIT_ORDER), s))
    if len(splits) == 1:
        where = f'of the denominator\'s {splits[0]} split'
    elif set(splits) >= set(SPLIT_ORDER):
        where = 'from every split of the denominator (' + ', '.join(splits) + '; the distilled models were trained on ' \
                'synthetic patients only)'
    else:
        where = 'from the denominator\'s ' + ' and '.join(splits) + ' splits'
    return f'Patients: every summarized {what} {where}; {query_note(args)}'


def trials_note(args) -> str:
    """A real run's trials: the trial list and how their cohorts were extracted."""
    import pandas as pd
    from matchminer_ai.evaluation.demographics import EVAL_KINDS
    kind = EVAL_KINDS.get(Path(args.denominator).name, ('other', ''))[0]
    n = pd.read_csv(args.trials, usecols=['nct_id']).nct_id.nunique()
    what = {'soc': f'the v22-style SOC trial sample ({n:,} NCT IDs, training trials excluded)',
            'enrollment': f'the {n:,} trials the denominator\'s patients enrolled on (their ctgov listings)'}.get(
                kind, f'{n:,} trials')
    how = (f'the frozen code snapshot {Path(args.extraction_source).name}' if args.extraction_source
           else "the training run's frozen programs")
    return (f'Trials: {what}; ctgov cohorts extracted with {how} (extraction/). Trials of the synthetic-training sample '
            '(--training-trials) are dropped from the cohort table (cohort_table/cohorts.json excluded_trials), so '
            'none is a candidate or a query.')


def teacher_note(args) -> str:
    where = ('the on-prem teacher server and its FP8 build on a second server' if not args.no_proxy else
             'the servers of servers/teacher_servers.json (the on-prem teacher server, and any replicas of the same '
             'model a server pool adds to it)')
    return (f'Teacher: {MODEL} on {where}, contracts match edbff870, boilerplate f0fc2d3c, per-criterion 7a7eb230; '
            'Nemotron sampling (temperature 1.0, top_p 0.95), thinking on, JSON tasks schema-constrained.')

def write_servers(args, root, proxy_ok):
    rows = [dict(url=args.teacher_url, model=MODEL, kind='teacher (on-prem)')]
    if proxy_ok:
        rows.append(dict(url=f'http://127.0.0.1:{args.proxy_port}/v1', model=MODEL, upstream=args.secondary_teacher_url,
                         upstream_model=FP8, kind='secondary teacher through teacher_proxy '
                         f'(model-ID alias, at most {args.proxy_concurrency} in flight)'))
    write_json(root / 'servers' / 'teacher_servers.json', dict(ready=True, servers=rows))
    for name, (gb300, lan) in CAPS.items():
        per = {args.teacher_url: gb300}
        if proxy_ok:
            per[f'http://127.0.0.1:{args.proxy_port}/v1'] = lan
        write_json(root / 'servers' / f'concurrency_{name}.json',
                   dict(max_concurrent_per_server=max(per.values()), per_server=per))


def start_proxy(args, root):
    config = root / 'servers' / 'proxy_8002.json'
    write_json(config, dict(upstream=args.secondary_teacher_url.removesuffix('/v1'), expected_root=FP8,
                            served_model=MODEL, upstream_model=FP8,
                            concurrency=args.proxy_concurrency, request_timeout=7260))
    log_file = open(root / 'logs' / 'proxy_8002.log', 'a')
    process = subprocess.Popen([args.python, '-m', f'{PACKAGE}.teacher_proxy', '--config', str(config),
                                '--port', str(args.proxy_port)], stdout=log_file, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True, env=child_env())
    for _ in range(30):
        time.sleep(2)
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.proxy_port}/v1/models', timeout=5) as response:
                if json.loads(response.read())['data'][0]['root'] == FP8:
                    return process
        except Exception:
            if process.poll() is not None:
                break
    process.terminate()
    return None


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True, help='The evaluation directory (PHI)')
    p.add_argument('--trials', help='The trial sample CSV (a real run)')
    p.add_argument('--summaries', help='The summary batches (a real run)')
    p.add_argument('--denominator', help='processed_soc_treatments.csv (a real run)')
    p.add_argument('--reuse-run', help='A real run: an earlier run directory of the same patients and trials whose '
                   'patients, patient embeddings, comparison samples, decision scores and teacher labels are reused '
                   '(for a re-run after its cohort table changed; needs --no-criteria)')
    p.add_argument('--compare-all', action='store_true', help='With --all-queries: rankers (a) and (b) run, and are '
                   'labeled, for every query too (no comparison subset)')
    p.add_argument('--no-new-boilerplate', action='store_true', help='With --reuse-run: no boilerplate label or '
                   'decision for pairs the earlier run lacks (the match labels are enough for the ranker comparison)')
    p.add_argument('--training-trials', help='A real run: CSV (nct_id) of the synthetic-training trial sample, whose '
                   'cohorts are never in the corpus (default: <data root>/no_phi/'
                   'training_run_screened_sample5000_20261104/registry_sample_trials.csv)')
    p.add_argument('--synthetic-corpus', help="A synthetic run: the n=5000 run's trialspace_corpus directory")
    p.add_argument('--synthetic-split', default='test', help='A synthetic run: the generator split of its patients')
    p.add_argument('--embed-device', default='cpu', help='Encoding device(s), comma-separated (cpu, cuda:0,cuda:1)')
    p.add_argument('--teacher-after', nargs='*', default=[], help='Files that must exist before label jobs start')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--extraction-source', help='Code snapshot for cohorts.py --source (revised cohort prompts)')
    p.add_argument('--extraction-workers', type=int, help='cohorts.py --workers for the extraction job (default: its '
                   'own; part of its identity with --extraction-source)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patient-queries', type=int, help='Sampled patient-centric queries (default 500)')
    p.add_argument('--cohort-queries', type=int, help='Sampled trial-centric queries (default 200)')
    p.add_argument('--all-queries', action='store_true', help='Full denominator: every patient and every cohort is a '
                   'query of rankers (c) and (d); (a) and (b) only for the comparison queries')
    p.add_argument('--comparison-patient-queries', type=int, help='With --all-queries: seeded patient-centric queries '
                   'of rankers (a) and (b)')
    p.add_argument('--comparison-cohort-queries', type=int, help='With --all-queries: seeded trial-centric queries of '
                   'rankers (a) and (b)')
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--criteria-lines', type=int, default=10000)
    p.add_argument('--no-criteria', action='store_true', help='No per-criterion sample, labels or decision scores')
    p.add_argument('--threads', type=int, default=64)
    p.add_argument('--gpu', default='3')
    p.add_argument('--min-free-gb', type=float, default=30)
    p.add_argument('--not-before', help='ISO time before which decision jobs leave the GPU alone')
    p.add_argument('--token-budget', type=int, default=65536)
    p.add_argument('--max-gpu-gb', type=float, default=24)
    p.add_argument('--proxy-port', type=int, default=18231)
    p.add_argument('--proxy-concurrency', type=int, default=12)
    p.add_argument('--no-proxy', action='store_true', help='Label on the teacher server only')
    p.add_argument('--teacher-url', help="Teacher server (default: the site config's endpoints.teacher.url)")
    p.add_argument('--secondary-teacher-url', help="Second teacher (FP8 build), reached through teacher_proxy "
                   "(default: the site config's endpoints.secondary_teacher.url; none: --no-proxy)")
    p.add_argument('--cache-dir', help="Hugging Face cache for the teacher tokenizer (default: the site config's "
                   "paths.hf_cache)")
    p.add_argument('--label-attempts', type=int, default=3)
    p.add_argument('--poll-seconds', type=int, default=30)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not args.synthetic_corpus and not (args.trials and args.summaries and args.denominator):
        p.error('a real run needs --trials, --summaries and --denominator (or give --synthetic-corpus)')
    comparison = (args.comparison_patient_queries, args.comparison_cohort_queries)
    if args.all_queries:
        if args.patient_queries is not None or args.cohort_queries is not None:
            p.error('--all-queries takes every patient and cohort; give no --patient-queries/--cohort-queries')
        if args.compare_all:
            if comparison != (None, None):
                p.error('--compare-all takes every query; give no --comparison-patient/cohort-queries')
        elif None in comparison or min(comparison) < 0:
            p.error('--all-queries needs --comparison-patient-queries and --comparison-cohort-queries (>= 0), or '
                    '--compare-all')
    elif comparison != (None, None):
        p.error('--comparison-patient-queries/--comparison-cohort-queries need --all-queries')
    args.patient_queries = 500 if args.patient_queries is None else args.patient_queries
    args.cohort_queries = 200 if args.cohort_queries is None else args.cohort_queries
    if args.criteria_lines < 0:
        p.error('--criteria-lines must be >= 0')
    args.no_criteria = args.no_criteria or args.criteria_lines == 0
    if args.no_criteria:
        args.criteria_lines = 0  # lists.py writes no per-criterion sample
    site = load_site_config()
    args.teacher_url = args.teacher_url or site.require('endpoints.teacher.url')
    args.secondary_teacher_url = args.secondary_teacher_url or site.get('endpoints.secondary_teacher.url')
    args.no_proxy = args.no_proxy or not args.secondary_teacher_url
    args.cache_dir = args.cache_dir or site.get('paths.hf_cache')
    if args.reuse_run and (args.synthetic_corpus or not args.no_criteria or not args.all_queries):
        p.error('--reuse-run is for a real full-denominator run with --no-criteria')
    if args.compare_all and not args.all_queries:
        p.error('--compare-all needs --all-queries')
    if args.no_new_boilerplate and not args.reuse_run:
        p.error('--no-new-boilerplate needs --reuse-run')
    if not args.synthetic_corpus:
        from matchminer_ai.evaluation.retro_cohorts.sample_soc_trials import RUN
        from matchminer_ai.training.paths import no_phi
        args.training_trials = args.training_trials or no_phi(RUN, 'registry_sample_trials.csv')
        if not Path(args.training_trials).is_file():
            raise SystemExit(f'No synthetic-training trial list at {args.training_trials} (--training-trials)')
    private()
    root = private_dir(args.root)
    for name in ('logs', 'servers', 'oa_pause', 'labels', 'decide', 'embeddings'):
        private_dir(root / name)
    scratch = private_dir(Path('/tmp/mmai-fidelity-synthetic/labels' if args.synthetic_corpus
                               else '/tmp/mmai-fidelity-phi/labels'))
    os.environ.update(MMAI_LABEL_SCRATCH_DIR=str(scratch), HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      PYTHONUNBUFFERED='1')
    os.environ.setdefault('OPENAI_API_KEY', 'not-needed')
    hold_gb300(root / 'oa_pause' / 'holders.json', 'orchestrator (run.py)')
    jobs = plan(args)
    if args.reuse_run:
        link_reused(args, root, jobs)
        write_reused_labels(args, root, jobs)
    if args.synthetic_corpus:
        write_json(root / 'design_notes.json', [
            f'Synthetic run (no PHI): the n=5000 run\'s {args.synthetic_split}-split synthetic patients with their '
            f'summaries; {query_note(args)}',
            "Trials: the n=5000 run's ctgov cohorts (training's extraction; trialspace_corpus).",
            f'Rankers: (a) base google/embeddinggemma-2 @914f7f89 cosine top 20; (b) TrialSpace round 2 cosine top 20; '
            f'(c) TrialSpace\'s top {args.k} reranked by the decision model\'s reasonableness P(yes), top 20; '
            f'(d) the same top {args.k} reranked by the composite match score (as in the app), top 20. '
            f'Key texts, trialspace prompt profile, 4,096-token cap, encoded on {args.embed_device}.',
            'TrialSpace round 2 was fine-tuned on every synthetic patient (split policy all_synthetic_training), '
            'these included, so its retrieval results here are optimistic; the decision model never trained on the '
            'test split.',
            'Teacher: nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 on the on-prem teacher server, contracts '
            'match edbff870, boilerplate f0fc2d3c, per-criterion 7a7eb230; Nemotron sampling (temperature 1.0, top_p '
            '0.95), thinking on, JSON tasks schema-constrained.',
            'Decision model: the n=5000 final fine-tune of LiquidAI/d1-2B (trialspace/models/decision_model, '
            'published as ksg-dfci/MatchEval-1226).',
            criteria_note(args),
            'A demonstration of the evaluation on synthetic data; the real-patient evaluation is the meaningful one.'])
    else:
        write_json(root / 'design_notes.json', [
            patients_note(args),
            trials_note(args),
            f'Rankers: (a) base google/embeddinggemma-2 @914f7f89 cosine top 20; (b) TrialSpace round 2 cosine top 20; '
            f'(c) TrialSpace top {args.k} reranked by the decision model\'s reasonableness P(yes), top 20; '
            f'(d) the same top {args.k} reranked by the composite match score (as in the app: P(reasonable) x (1 + '
            'P(cancer type) + P(burden) + P(treatment history) + 2 P(biomarker)) / 6), top 20. '
            'Key texts, trialspace prompt profile, 4,096-token cap, '
            + ('CPU float32 encoding.' if args.embed_device == 'cpu' else
               f'encoded on {args.embed_device} (bf16 autocast).'),
            teacher_note(args),
            'Decision model: the n=5000 final fine-tune of LiquidAI/d1-2B (trialspace/models/decision_model).',
            criteria_note(args),
            *([f'Re-run of {Path(args.reuse_run).name}: its cohort extraction, patients and patient embeddings are '
               'reused' + ('' if args.compare_all else ', and its comparison samples (without any dropped cohorts)')
               + ', and so are its decision scores and teacher labels (and those it reused); only pairs and boilerplate '
               'combinations it had no answer for were scored and labeled. Teacher labels are sampled (temperature '
               '1.0), so reused and new labels are draws of the same labeler.'] if args.reuse_run else []),
            *(['No boilerplate was labeled or scored anew (--no-new-boilerplate): the boilerplate tables cover the '
               'combinations the earlier runs labeled, the pairs of every (c) and (d) list and of the earlier '
               'comparison subset.'] if args.no_new_boilerplate else [])])
    proxy, released = None, False
    status_path = root / 'run_status.json'
    log(f'orchestrator started: {len(jobs)} jobs')
    while True:
        for job in jobs.values():
            if job.state == 'pending' and job.process is None:
                if any(jobs[d].state in ('failed', 'blocked') for d in job.deps):
                    job.state = 'blocked'
                    log(f'{job.name}: blocked by a failed dependency')
                    continue
                if not all(jobs[d].state == 'done' for d in job.deps):
                    continue
                code = checked(job)
                if code == 0:
                    job.state = 'done'
                    log(f'{job.name}: complete')
                    continue
                if code == 20:
                    job.state = 'failed'
                    log(f'{job.name}: its directory was begun under other settings (check 20)')
                    continue
                if running_elsewhere(job):
                    continue  # wait for the other process; checked again next round
                if job.name.startswith('label_') and not all(Path(f).exists() for f in args.teacher_after):
                    if not job.held:
                        job.held = True
                        log(f'{job.name}: waiting for other teacher work to finish (--teacher-after)')
                    continue
                if job.kind == 'gpu' and any(j.kind == 'gpu' and j.state == 'running' for j in jobs.values()):
                    continue
                if job.kind == 'gb300' and job.name.startswith('label_') and proxy is None and not args.no_proxy:
                    proxy = start_proxy(args, root)
                    write_servers(args, root, proxy is not None)
                    log('teacher servers: the teacher' + (' + the second through the proxy' if proxy else ' only (proxy failed)'))
                elif job.kind == 'gb300' and args.no_proxy and not (root / 'servers' / 'teacher_servers.json').exists():
                    write_servers(args, root, False)
                stream = open(root / 'logs' / f'{job.name}.log', 'a')
                job.process = subprocess.Popen(job.command, stdout=stream, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, cwd=root, env=child_env())
                job.state, job.tries, job.started = 'running', job.tries + 1, time.time()
                log(f'{job.name}: started (attempt {job.tries})')
            elif job.state == 'running':
                code = job.process.poll()
                if code is None:
                    continue
                job.process, job.code, job.finished = None, code, time.time()
                if code == 0 and checked(job) == 0:
                    job.state = 'done'
                    log(f'{job.name}: done in {(job.finished - job.started) / 60:.1f} min')
                elif job.tries < job.attempts:
                    job.state = 'pending'
                    log(f'{job.name}: exited {code}; retrying')
                else:
                    job.state = 'failed'
                    log(f'{job.name}: failed (exit {code}); see logs/{job.name}.log')
        teacher = [j for j in jobs.values() if j.kind == 'gb300']
        if not released and all(j.state in ('done', 'failed', 'blocked') for j in teacher):
            (root / 'oa_pause' / 'release').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z') + '\n')
            released = True
            log('every teacher job has finished: wrote oa_pause/release (OA resumes)')
            if proxy is not None:
                proxy.terminate()
                proxy = None
        write_json(status_path, dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), released=released, jobs={
            j.name: dict(state=j.state, tries=j.tries, exit=j.code, started=j.started, finished=j.finished)
            for j in jobs.values()}))
        if all(j.state in ('done', 'failed', 'blocked') for j in jobs.values()):
            break
        time.sleep(args.poll_seconds)
    failed = [j.name for j in jobs.values() if j.state != 'done']
    log('orchestrator finished' + (f'; not done: {failed}' if failed else ': every job done'))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
