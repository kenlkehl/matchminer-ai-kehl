"""Helpers shared by the retrospective distillation-fidelity stages (see __init__.py).

Every output of this evaluation describes real patients (PHI): directories are created at mode 700 and files at
600 (``private()`` sets umask 077 for the process and its children), and nothing here prints or logs clinical text
or medical record numbers; status lines carry counts only (``mask`` hides any number of 6 or more digits).
"""
import json
import os
from pathlib import Path
import re
import time

from matchminer_ai.site import load_site_config
from matchminer_ai.training.cohort_format import file_stamp
from matchminer_ai.training.paths import no_phi
from matchminer_ai.training.trialspace import stage
from matchminer_ai.training.trialspace.data import digest, read_jsonl, write_json, write_jsonl

# Every patient and cohort of this evaluation is in the corpus split "val" (the labeler requires a named split;
# the patients are the SOC cohort's validation split, a development set).
SPLIT = 'val'
WORKFLOWS = ('patient_centric', 'trial_centric')
DIRECTION = {'patient_centric': 'patient_to_cohort', 'trial_centric': 'cohort_to_patient'}
EMBEDDER_NAMES = ('embeddinggemma2', 'trialspace_round2')


def __getattr__(name):
    """The n=5000 run's dataset (``L``), its models and the embedders, resolved on first use from the data root."""
    if name == 'L':
        return no_phi('synthetic_dataset_n5000_screened_20261106')
    if name == 'DECISION_MODEL':
        return __getattr__('L') / 'trialspace/models/decision_model'
    if name == 'EMBEDDERS':
        return {
            # (a) base EmbeddingGemma 2, as training's round-1 mining encoded with it (a complete 914f7f89 snapshot
            # in the site config's paths.embeddinggemma2_cache, else the default Hugging Face cache)
            'embeddinggemma2': dict(model='google/embeddinggemma-2', revision='914f7f89142e33e77833254d9c9b90c3cef7303b',
                                    cache_dir=load_site_config().get('paths.embeddinggemma2_cache')),
            # (b) the round-2 TrialSpace fit, the inference default
            'trialspace_round2': dict(model=str(__getattr__('L') / 'trialspace/models/trialspace_round2'),
                                      revision=None, cache_dir=None),
        }
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


# (a), (b): cosine top 20; (c) TrialSpace's top K reranked by P(reasonable); (d) the same top K reranked by the
# composite match_quality_score, as the app ranks (lists.py)
RANKERS = ('embeddinggemma2', 'trialspace_round2', 'trialspace_round2_decision', 'trialspace_round2_composite')
TOP = 20
QUERY_KEYS = {'patient_centric': 'patient_queries', 'trial_centric': 'cohort_queries'}


def full_denominator(sample):
    """Whether a corpus's sample.json is a full-denominator one (prepare.py corpus --all-queries): every patient and
    cohort is a query, and rankers (a) and (b) run only for its seeded comparison subset."""
    return 'comparison_patient_queries' in sample


def queries(sample, workflow, comparison=False):
    """A workflow's query IDs in sample.json; ``comparison``: only those rankers (a) and (b) run for (the comparison
    subset of a full-denominator corpus; every query otherwise)."""
    key = QUERY_KEYS[workflow]
    return sample['comparison_' + key] if comparison and full_denominator(sample) else sample[key]


def pair_rows(frame, column, name, trial_of):
    """Every (patient_id, cohort_id) of a ranking frame once, in sorted order, with each row it holds (in frame
    order) in ``sources`` as dict(direction, workflow, query_id, <name>=frame[column], rank); a generator."""
    frame = frame.reset_index(drop=True)
    groups = frame.groupby(['patient_id', 'cohort_id'], sort=True).indices
    workflow, query = frame.workflow.to_numpy(), frame.query_id.to_numpy()
    value, rank = frame[column].to_numpy(), frame['rank'].to_numpy()
    for patient, cohort in sorted(groups):
        yield dict(patient_id=patient, cohort_id=cohort, trial_id=trial_of[cohort], split=SPLIT,
                   sources=[{'direction': DIRECTION[workflow[i]], 'workflow': workflow[i], 'query_id': query[i],
                             name: value[i], 'rank': int(rank[i])} for i in groups[(patient, cohort)]])


def private():
    """umask 077 for this process and every process it starts: new files 600, new directories 700."""
    os.umask(0o077)


def private_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def mask(value):
    """Text safe to print: any run of 6 or more digits becomes #s."""
    return re.sub(r'\d{6,}', lambda m: '#' * len(m.group()), str(value))


def log(message):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S %Z")}] {mask(message)}', flush=True)


def stamp_inputs(paths):
    return {Path(p).name: file_stamp(p) for p in paths}


def hold_gb300(holders, note):
    """Register this process as holding the teacher pause (a waiter that resumes other teacher work, such as the
    real-time summarization, once no registered process has been alive for its grace period)."""
    if not holders:
        return
    import fcntl
    holders = Path(holders)
    pid = os.getpid()
    ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    with holders.with_name(holders.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(holders.read_text()) if holders.exists() else []
        rows.append(dict(pid=pid, start_ticks=ticks, note=note, at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        temporary = holders.with_name(holders.name + f'.{pid}.tmp')
        temporary.write_text(json.dumps(rows, indent=2) + '\n')
        temporary.replace(holders)


def run_stage(directory, kind, identity, outputs, work, *, check=False, content=None, upstream=None):
    """Stage protocol of the training repository: --check answers 0/10/20; a complete directory is skipped; a
    directory begun under other settings is refused; ``work()`` runs, then the receipt is written last."""
    if check:
        stage.exit_check(directory, kind, identity)
    if stage.require_new_or_same(directory, kind, identity):
        log(f'[skip] {kind} already complete')
        return stage.read_receipt(directory)
    private_dir(directory)
    result = work() or {}
    names = outputs() if callable(outputs) else outputs
    return stage.write_receipt(directory, kind, identity, names, content={**(content or {}), **result},
                               upstream=upstream)


__all__ = ['DECISION_MODEL', 'DIRECTION', 'EMBEDDERS', 'EMBEDDER_NAMES', 'L', 'QUERY_KEYS', 'RANKERS', 'SPLIT', 'TOP',
           'WORKFLOWS', 'digest', 'file_stamp', 'full_denominator', 'hold_gb300', 'log', 'mask', 'pair_rows', 'private',
           'private_dir', 'queries', 'read_jsonl', 'run_stage', 'stage', 'stamp_inputs', 'write_json', 'write_jsonl']
