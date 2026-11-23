"""Demographics of a retrospective fidelity run's patients: Table 1 and fidelity metrics by demographic group.

User, 2026-11-23: the Benchmarking tab's PDF export carries, for each real fidelity run (one eval: enrollments or
SOC), a classic Table 1 of its patients and a v22-style table of the headline metrics by demographic group, both with
small-cell suppression.

**Linkage.** A run's patients are its corpus's index events (``corpus/patients.jsonl``: ``id`` is the event's
pseudo_mrn, ``person`` an opaque per-patient index). The patients stage's receipt names the denominator CSV it read
(``--denominator``, checked against the receipt's size and mtime stamp), which maps each pseudo_mrn to the patient's
``dfci_mrn``, the index date (``trial_start_dt``: the treatment plan's start) and the plan (``epic_tplan_id``). From
the extract's structured data (the site config's ``paths.retro_source_root``/``structured_data``), selected columns
only:

- ``PT_INFO.csv``: birth date and sex (``GENDER_NM``, else ``SEX_AT_BIRTH_NM``; Female, Male or Unknown);
- ``PT_DEMOGRAPHICS.csv``: race (``DERIVED_RACE_NM``; "Unknown ..." values are Unknown, combinations of several races
  are Multiple) and ethnicity (``HISPANIC_IND``: Hispanic, Non-Hispanic or Unknown);
- ``TREATMENT_PLAN.csv``: the index plan's ICD-10 diagnosis codes, grouped into cancer types (``cancer_type``).

These are v22's rules (``real_eval_code/eval_phi_*/demographics_utils.py``) on the current extract, whose tables are
named ``PT_INFO`` and ``PT_DEMOGRAPHICS`` (v22's ``PT_INFO_STATUS_REGISTRATION`` and ``DEMOGRAPHICS_REGISTRATION``),
plus the cancer type. Age is at the treatment start (the index date), in v22's groups.

**Table 1.** Two columns: patients (race, ethnicity and sex; age and cancer type at the patient's first treatment start
in the eval) and treatment starts (every index event). **By group:** the full pipeline's (ranker (c)) P@20 and mean
teacher score over the patient-centric queries of each group, with person-cluster bootstrap intervals
(``retro_fidelity.metrics.cluster_bootstrap_mean``, as its full-pipeline table), and the decision model's
reasonableness AUROC over the labeled pairs of the group's patients (both workflows pooled, the metrics stage's pooled
rows), with a person-cluster bootstrap.

**Suppression** (``suppress``): counts of 1 to 10 show as "<11" with no percentage, zero shows as 0, and totals stay
exact. Within each section of a column, larger counts are also hidden ("*"), smallest first, until at least two are
hidden, they add up to 11 or more and their sum has more than one split, so no hidden count follows from the total. A group's metrics are
hidden ("n<11") when it has fewer than 11 patients.

Real data only: ``run_demographics`` returns aggregates, suppressed for the report. The unsuppressed aggregates are
cached owner-only in the run's ``reports/`` (``demographics.json``), inside the run's own PHI directory. Nothing here
prints or logs values; errors name files and columns, never their contents.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

VERSION = 'fidelity-demographics-v1'
SMALL = 11  # counts 1..SMALL-1 are hidden
CACHE = 'demographics.json'

AGE_EDGES = (40, 50, 60, 70, 80)
AGE_LABELS = ('<40', '40-49', '50-59', '60-69', '70-79', '80+')
UNKNOWN = 'Unknown'
SEXES = ('Female', 'Male')
ETHNICITIES = ('Hispanic', 'Non-Hispanic')
RACES = ('White', 'Black or African American', 'Asian', 'American Indian or Alaskan Native',
         'Native Hawaiian or Other Islander', 'Some Other Race', 'Multiple')
DIMENSIONS = {'age_group': 'Age at treatment start', 'race': 'Race', 'ethnicity': 'Ethnicity', 'sex': 'Sex'}
TABLE1_DIMENSIONS = {**DIMENSIONS, 'cancer_type': 'Cancer type (treatment plan diagnosis)'}

PT_INFO, PT_DEMOGRAPHICS, TREATMENT_PLAN = 'PT_INFO.csv', 'PT_DEMOGRAPHICS.csv', 'TREATMENT_PLAN.csv'
EVAL_KINDS = {'processed_trial_enrollments.csv': ('enrollment', 'Trial enrollments (starts of research treatment)'),
              'processed_soc_treatments.csv': ('soc', 'Standard-of-care treatment starts')}

# ICD-10 three-character categories of a treatment plan's diagnosis, in table order.
CANCER_TYPES = (
    ('Head and neck', ('C00-C14', 'C30-C32')),
    ('Esophagus and stomach', ('C15-C16',)),
    ('Colorectal and anal', ('C18-C21',)),
    ('Liver and biliary', ('C22-C24',)),
    ('Pancreas', ('C25',)),
    ('Other digestive', ('C17', 'C26')),
    ('Lung and other thoracic', ('C33-C34', 'C37-C39', 'C45')),
    ('Sarcoma (bone and soft tissue)', ('C40-C41', 'C46-C49')),
    ('Melanoma and other skin', ('C43-C44', 'D03')),
    ('Breast', ('C50', 'D05')),
    ('Gynecologic', ('C51-C58', 'D06')),
    ('Prostate', ('C61',)),
    ('Testis and other male genital', ('C60', 'C62-C63')),
    ('Kidney and renal pelvis', ('C64-C65',)),
    ('Bladder and other urinary', ('C66-C68',)),
    ('Brain, CNS and eye', ('C69-C72',)),
    ('Thyroid and other endocrine', ('C73-C75',)),
    ('Neuroendocrine', ('C7A',)),
    ('Lymphoma', ('C81-C86', 'C88')),
    ('Myeloma and plasma cell', ('C90',)),
    ('Leukemia', ('C91-C95',)),
    ('MDS, MPN and other hematologic', ('C96', 'D45-D47')),
    ('Unknown primary or secondary site only', ('C76-C80', 'C7B')),
)
OTHER_CODE = 'Other diagnosis code'
CANCER_TYPE_LABELS = (*(label for label, _ in CANCER_TYPES), OTHER_CODE)
SECONDARY = 'Unknown primary or secondary site only'
IN_SITU = ('D03', 'D05', 'D06')


class NoDemographics(Exception):
    """The run's patients have no registration records to link (a synthetic run)."""


class DemographicsError(ValueError):
    """Why a real run's demographics cannot be linked; the message names files, never their contents."""


# --------------------------------------------------------------------------- categories


def _in(category: str, spec: str) -> bool:
    low, _, high = spec.partition('-')
    if not high:
        return category == low
    # C7A/C7B are not numeric: they fall lexically between C76 and C80, so only an exact spec takes them
    return category[1:].isdigit() and low <= category <= high


def code_group(code: str) -> str | None:
    """The cancer type of one ICD-10 code (its first three characters), None for a code outside the table."""
    category = str(code).strip().upper()[:3]
    for label, specs in CANCER_TYPES:
        if any(_in(category, spec) for spec in specs):
            return label
    return None


def cancer_type(codes) -> str:
    """An index event's cancer type from its plans' ICD-10 codes, in order: the first primary cancer (a C code, or
    MDS/MPN), else an in situ code, else unknown primary or a secondary site, else another code; Unknown without
    codes."""
    codes = [c for c in codes if str(c).strip()]
    if not codes:
        return UNKNOWN
    groups = [(str(c).strip().upper()[:3], code_group(c)) for c in codes]
    for rank in (lambda cat, g: g not in (None, SECONDARY) and cat not in IN_SITU,
                 lambda cat, g: cat in IN_SITU, lambda cat, g: g == SECONDARY):
        for category, group in groups:
            if rank(category, group):
                return group
    return OTHER_CODE


def split_codes(value) -> list[str]:
    """TREATMENT_PLAN's TPLAN_ICD_DX_CODES: comma-separated ICD-10 codes."""
    if not isinstance(value, str):
        return []
    return [c.strip() for c in value.split(',') if c.strip()]


def age_group(age):
    import pandas as pd
    groups = pd.cut(pd.Series(age, dtype=float), bins=[-np.inf, *AGE_EDGES, np.inf], labels=AGE_LABELS, right=False)
    return groups.astype(object).where(groups.notna(), UNKNOWN)


def sex_category(gender, sex_at_birth):
    """v22's rule: GENDER_NM, else (when it is missing) SEX_AT_BIRTH_NM; anything but Female or Male is Unknown."""
    def clean(series):
        series = series.astype(object).where(series.notna(), '').astype(str).str.strip().str.title()
        return series.where(series.ne(''))
    sex = clean(gender).fillna(clean(sex_at_birth))
    return sex.where(sex.isin(SEXES), UNKNOWN).astype(object)


def race_category(race):
    race = race.astype(object).where(race.notna(), '').astype(str).str.strip()
    race = race.mask(race.str.startswith('Unknown') | race.eq(''), UNKNOWN)
    return race.mask(race.str.contains('/', regex=False), 'Multiple').astype(object)


def ethnicity_category(hispanic):
    hispanic = hispanic.astype('string').str.strip().str.upper()
    yes, no = (hispanic.eq(v).fillna(False).to_numpy(dtype=bool) for v in ('Y', 'N'))
    return np.select([yes, no], ETHNICITIES, default=UNKNOWN)


# --------------------------------------------------------------------------- the extract


def _mrn(series):
    import pandas as pd
    return pd.to_numeric(series, errors='coerce').round().astype('Int64')


def read_registration(structured: Path, mrns):
    """Birth date, sex, race and ethnicity of the given patients (``dfci_mrn``), from PT_INFO and PT_DEMOGRAPHICS."""
    import pandas as pd
    wanted = pd.Index(pd.array(list(mrns), dtype='Int64')).unique()
    info = pd.read_csv(structured / PT_INFO, usecols=['DFCI_MRN', 'BIRTH_DT', 'GENDER_NM', 'SEX_AT_BIRTH_NM'],
                       dtype=str)
    info['dfci_mrn'] = _mrn(info.pop('DFCI_MRN'))
    info = info[info.dfci_mrn.isin(wanted)].drop_duplicates('dfci_mrn')
    info['birth_dt'] = pd.to_datetime(info.BIRTH_DT, errors='coerce', format='mixed')
    info['sex'] = sex_category(info.GENDER_NM, info.SEX_AT_BIRTH_NM)
    demo = pd.read_csv(structured / PT_DEMOGRAPHICS, usecols=['DFCI_MRN', 'DERIVED_RACE_NM', 'HISPANIC_IND'], dtype=str)
    demo['dfci_mrn'] = _mrn(demo.pop('DFCI_MRN'))
    demo = demo[demo.dfci_mrn.isin(wanted)].drop_duplicates('dfci_mrn')
    demo['race'] = race_category(demo.DERIVED_RACE_NM)
    demo['ethnicity'] = ethnicity_category(demo.HISPANIC_IND)
    return (info[['dfci_mrn', 'birth_dt', 'sex']].assign(in_pt_info=True)
            .merge(demo[['dfci_mrn', 'race', 'ethnicity']].assign(in_demographics=True), on='dfci_mrn', how='outer'))


def read_plan_codes(structured: Path, plan_ids) -> dict:
    """Each given treatment plan's ICD-10 codes (TREATMENT_PLAN's EPIC_TPLAN_ID and TPLAN_ICD_DX_CODES)."""
    import pandas as pd
    wanted = set(map(str, plan_ids))
    plans = pd.read_csv(structured / TREATMENT_PLAN, usecols=['EPIC_TPLAN_ID', 'TPLAN_ICD_DX_CODES'], dtype=str)
    plans = plans[plans.EPIC_TPLAN_ID.isin(wanted)].drop_duplicates('EPIC_TPLAN_ID')
    return {plan: split_codes(codes) for plan, codes in zip(plans.EPIC_TPLAN_ID, plans.TPLAN_ICD_DX_CODES)}


# --------------------------------------------------------------------------- a run's patients


def _receipt(directory: Path) -> dict:
    return json.loads((directory / 'stage_receipt.json').read_text())


def denominator_of(run: Path) -> Path:
    """The denominator CSV the run's patients stage read, unchanged since (its receipt's size and mtime stamp)."""
    from matchminer_ai.training.cohort_format import file_stamp
    receipt = _receipt(run / 'patients')
    identity = receipt.get('identity') or {}
    if identity.get('stage') != 'patients':
        raise NoDemographics('The run\'s patients are synthetic.')
    argv = list((receipt.get('provenance') or {}).get('argv') or [])
    if '--denominator' not in argv or argv.index('--denominator') + 1 >= len(argv):
        raise DemographicsError('The patients stage receipt does not name its denominator file.')
    path = Path(argv[argv.index('--denominator') + 1])
    stamp = (identity.get('inputs') or {}).get(path.name)
    if stamp is None:
        raise DemographicsError('The patients stage receipt does not stamp its denominator file.')
    if not path.is_file():
        raise DemographicsError('The run\'s denominator file is not available.')
    if file_stamp(path) != stamp:
        raise DemographicsError('The run\'s denominator file changed after its patients stage ran.')
    return path


def read_persons(run: Path) -> dict:
    """Each index event's person index (``corpus/patients.jsonl``; the summary text is not kept)."""
    persons = {}
    with (run / 'corpus' / 'patients.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                persons[str(row['id'])] = row['person']
    return persons


def patient_frame(events, registration, plan_codes):
    """One row per index event with its demographic categories: ``events`` holds patient_id, person, dfci_mrn,
    index_date (a timestamp) and plan_ids (a list); ``registration`` is ``read_registration``'s frame and
    ``plan_codes`` ``read_plan_codes``' map."""
    frame = events.assign(dfci_mrn=events.dfci_mrn.astype('Int64')).merge(registration, on='dfci_mrn', how='left')
    age = (frame.index_date - frame.birth_dt).dt.days / 365.25
    frame['age'] = age.where(age >= 0)
    frame['age_group'] = age_group(frame.age).to_numpy()
    for column in ('race', 'ethnicity', 'sex'):
        frame[column] = frame[column].where(frame[column].notna(), UNKNOWN)
    frame['cancer_type'] = [cancer_type([c for plan in plans for c in plan_codes.get(str(plan), [])])
                            for plans in frame.plan_ids]
    frame['linked'] = frame.in_pt_info.eq(True) & frame.in_demographics.eq(True)
    return frame.drop(columns=['in_pt_info', 'in_demographics'])


def run_patients(run: Path, extract_root: Path, progress=None):
    """(the run's patient frame, the eval's kind and label, its splits) from its denominator and the extract."""
    import pandas as pd
    say = progress or (lambda message: None)
    denominator = denominator_of(run)
    kind, label = EVAL_KINDS.get(denominator.name, ('other', 'Retrospective cohort'))
    say('reading the run\'s patients')
    persons = read_persons(run)
    rows = pd.read_csv(denominator, usecols=['pseudo_mrn', 'dfci_mrn', 'trial_start_dt', 'epic_tplan_id', 'split'],
                       dtype=str)
    rows = rows[rows.pseudo_mrn.isin(persons)].sort_values(['pseudo_mrn', 'epic_tplan_id'])
    grouped = rows.groupby('pseudo_mrn', sort=False)
    events = grouped.agg(dfci_mrn=('dfci_mrn', 'first'), index_date=('trial_start_dt', 'first'),
                         plan_ids=('epic_tplan_id', list), split=('split', 'first')).reset_index()
    if grouped.dfci_mrn.nunique().gt(1).any() or grouped.trial_start_dt.nunique().gt(1).any():
        raise DemographicsError('An index event maps to more than one patient or date in the denominator file.')
    missing = sorted(set(persons) - set(events.pseudo_mrn))
    if missing:  # events without a denominator row stay in the table, as Unknown
        events = pd.concat([events, pd.DataFrame(dict(pseudo_mrn=missing, plan_ids=[[] for _ in missing]))],
                           ignore_index=True)
    events = events.rename(columns={'pseudo_mrn': 'patient_id'})
    events['person'] = events.patient_id.map(persons)
    events['dfci_mrn'] = _mrn(events.dfci_mrn)
    events['index_date'] = pd.to_datetime(events.index_date, errors='coerce')
    structured = Path(extract_root) / 'structured_data'
    say('reading registration records')
    registration = read_registration(structured, events.dfci_mrn.dropna())
    say('reading treatment plan diagnoses')
    codes = read_plan_codes(structured, {p for plans in events.plan_ids for p in plans})
    frame = patient_frame(events, registration, codes)
    frame['in_denominator'] = frame.dfci_mrn.notna()
    splits = sorted(set(events.split.dropna()))
    return frame, dict(kind=kind, label=label, splits=splits)


# --------------------------------------------------------------------------- suppression


def hidden_cells(counts) -> list[str | None]:
    """Which of a section's counts to hide, and why: every count of 1 to 10 ('small'), then the smallest shown
    non-zero counts ('complement', each 11 or more) until the hidden cells cannot be worked out from the total: at
    least two are hidden, they add up to 11 or more, and their sum has more than one split (it is neither every hidden
    cell at its least, 1 or 11, nor every small cell at 10)."""
    counts = [int(n) for n in counts]
    hidden = ['small' if 0 < n < SMALL else None for n in counts]
    if not any(hidden):
        return hidden
    while True:
        small = [n for n, h in zip(counts, hidden) if h == 'small']
        complement = [n for n, h in zip(counts, hidden) if h == 'complement']
        total = sum(small) + sum(complement)
        unique = total == len(small) + SMALL * len(complement) or (not complement and total == (SMALL - 1) * len(small))
        if len(small) + len(complement) >= 2 and total >= SMALL and not unique:
            return hidden
        shown = [i for i, n in enumerate(counts) if not hidden[i] and n > 0]
        if not shown:
            return hidden
        hidden[min(shown, key=lambda i: (counts[i], i))] = 'complement'


HIDDEN_TEXT = {'small': f'<{SMALL}', 'complement': '*'}


def suppress(counts, total) -> list[dict]:
    """Display cells of a section's counts: {n, pct} shown, zero as 0; a hidden count has no n or pct and its text is
    '<11' (a count of 1 to 10) or '*' (a larger count hidden so that no small one follows from the total)."""
    cells = []
    for n, why in zip(counts, hidden_cells(counts)):
        if why:
            cells.append(dict(n=None, pct=None, text=HIDDEN_TEXT[why]))
        else:
            cells.append(dict(n=int(n), pct=(100.0 * n / total) if total and n else (0.0 if n == 0 else None),
                              text=None))
    return cells


RULE = (f'Counts of 1 to {SMALL - 1} are shown as <{SMALL}, without a percentage; zero is shown as 0 and totals are '
        'exact. Within each section of a column, larger counts are also hidden (*), smallest first, until no hidden '
        f'count can be worked out from the total (at least two hidden, adding up to {SMALL} or more). A group\'s '
        f'metrics are shown as n<{SMALL} when it has fewer than {SMALL} patients.')


# --------------------------------------------------------------------------- Table 1


def categories(dimension: str, values) -> list[str]:
    """A section's rows: the fixed categories of the dimension, then any other value, then Unknown."""
    fixed = {'age_group': AGE_LABELS, 'race': RACES, 'ethnicity': ETHNICITIES, 'sex': SEXES,
             'cancer_type': CANCER_TYPE_LABELS}[dimension]
    extra = sorted({v for v in values if v not in fixed and v != UNKNOWN})
    return [*fixed, *extra, UNKNOWN]


def age_summary(age) -> dict:
    age = np.asarray(age, dtype=float)
    age = age[np.isfinite(age)]
    if not len(age):
        return dict(median=None, q1=None, q3=None, known=0)
    q1, median, q3 = np.quantile(age, [0.25, 0.5, 0.75])
    return dict(median=float(median), q1=float(q1), q3=float(q3), known=int(len(age)))


def table1_counts(frame) -> dict:
    """Table 1's counts (unsuppressed) of a patient frame (``patient_frame``): a patients column (race, ethnicity and
    sex; age and cancer type at each patient's first treatment start) and a treatment-starts column (every index
    event)."""
    first = frame.sort_values(['index_date', 'patient_id'], na_position='last').drop_duplicates('person')
    columns = [('patients', 'Patients', first), ('events', 'Treatment starts', frame)]
    sections = []
    for dimension, label in TABLE1_DIMENSIONS.items():
        rows = categories(dimension, set(frame[dimension]))
        counts = {key: part[dimension].value_counts() for key, _, part in columns}
        sections.append(dict(key=dimension, label=label, categories=rows,
                             counts={key: [int(counts[key].get(c, 0)) for c in rows] for key, *_ in columns}))
    return dict(columns=[dict(key=k, label=lab, n=int(len(part))) for k, lab, part in columns],
                age={k: age_summary(part.age) for k, _, part in columns}, sections=sections)


def table1(counts: dict) -> dict:
    """Table 1 for the report: ``table1_counts``' counts suppressed per section and column."""
    total = {c['key']: c['n'] for c in counts['columns']}
    sections = []
    for section in counts['sections']:
        cells = {key: suppress(values, total[key]) for key, values in section['counts'].items()}
        sections.append(dict(key=section['key'], label=section['label'],
                             rows=[dict(category=c, cells={k: cells[k][i] for k in cells})
                                   for i, c in enumerate(section['categories'])]))
    return dict(columns=counts['columns'], age=counts['age'], sections=sections)


# --------------------------------------------------------------------------- metrics by group


def cluster_bootstrap_auroc(y, p, clusters, *, draws=1000, seed=42, alpha=0.05) -> dict:
    """AUROC (ties count half, as ``retro_fidelity.metrics.auroc``) with a percentile interval from resampling whole
    clusters: each draw weights every row by its cluster's multiplicity."""
    from matchminer_ai.evaluation.retro_fidelity.metrics import auroc
    y, p = np.asarray(y, dtype=bool), np.asarray(p, dtype=float)
    value = auroc(y, p) if len(y) else np.nan
    if not np.isfinite(value):
        return dict(value=None, low=None, high=None)
    _, cluster = np.unique(np.asarray(clusters), return_inverse=True)
    order = np.argsort(p, kind='mergesort')
    _, tie = np.unique(p[order], return_inverse=True)  # tie groups in ascending P(yes)
    yes, cluster = y[order], cluster[order]
    rng, count = np.random.default_rng(seed), int(cluster.max()) + 1
    values = []
    for _ in range(draws):
        weight = np.bincount(rng.integers(0, count, count), minlength=count)[cluster]
        pos, neg = np.bincount(tie, weights=weight * yes), np.bincount(tie, weights=weight * ~yes)
        if not pos.sum() or not neg.sum():
            continue
        below = np.cumsum(neg) - neg  # negative weight strictly below each tie group
        values.append(float((pos * (below + neg / 2)).sum() / (pos.sum() * neg.sum())))
    if not values:
        return dict(value=float(value), low=None, high=None)
    return dict(value=float(value), low=float(np.quantile(values, alpha / 2)),
                high=float(np.quantile(values, 1 - alpha / 2)))


def read_teacher(run: Path) -> dict:
    """The teacher's match labels of the run: (patient_id, cohort_id) -> dict(answers, score) (scores.jsonl only)."""
    teacher = {}
    for directory in sorted((run / 'labels').glob('match_*')):
        with (directory / 'scores.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    teacher[(row['patient_id'], row['cohort_id'])] = dict(answers=row['answers'], score=row['score'])
    return teacher


def group_metrics(run: Path, frame, progress=None, *, draws=2000, auroc_draws=1000, seed=42) -> dict:
    """The full pipeline's P@20 and mean teacher score per patient-centric query, and the decision model's
    reasonableness AUROC over the labeled pairs, overall and per demographic group (raw, before suppression)."""
    import pandas as pd
    from matchminer_ai.evaluation.retro_fidelity.common import RANKERS
    from matchminer_ai.evaluation.retro_fidelity.metrics import cluster_bootstrap_mean, per_query
    say = progress or (lambda message: None)
    say('reading the teacher labels')
    teacher = read_teacher(run)
    lists = pd.read_parquet(run / 'lists_c' / 'lists.parquet')
    ranker = RANKERS[2]
    rows = per_query(lists[(lists.ranker == ranker) & (lists.workflow == 'patient_centric')], teacher)
    queries = sorted(q for _, _, q in rows)
    cells = [rows[('patient_centric', ranker, q)] for q in queries]
    info = frame.set_index('patient_id')
    person = info.person.to_dict()
    say('reading the decision model\'s scores')
    union = lists[['patient_id', 'cohort_id']].drop_duplicates()
    union = union[[k in teacher for k in zip(union.patient_id, union.cohort_id)]]
    scores = pd.read_parquet(run / 'decide' / 'match' / 'scores.parquet',
                             columns=['patient_id', 'cohort_id', 'p_reasonableness'])
    pairs = union.merge(scores, on=['patient_id', 'cohort_id'], how='inner', validate='one_to_one')
    pairs['yes'] = [teacher[k]['answers']['reasonableness'] == 'yes' for k in zip(pairs.patient_id, pairs.cohort_id)]
    pairs['person'] = pairs.patient_id.map(person)
    query_frame = pd.DataFrame(dict(query=queries, person=[person.get(q) for q in queries],
                                    precision=[c['precision'] for c in cells], score=[c['mean_score'] for c in cells]))

    def one(q_mask, pair_mask):
        q, pr = query_frame[q_mask], pairs[pair_mask]
        return dict(queries=int(len(q)), query_persons=int(q.person.nunique()), pair_persons=int(pr.person.nunique()),
                    precision=cluster_bootstrap_mean(q.precision, q.person, draws=draws, seed=seed),
                    mean_score=cluster_bootstrap_mean(q.score, q.person, draws=draws, seed=seed),
                    auroc=cluster_bootstrap_auroc(pr.yes, pr.p_reasonableness, pr.person, draws=auroc_draws,
                                                  seed=seed))

    say('bootstrapping all patients')
    out = dict(overall=one(np.ones(len(query_frame), bool), np.ones(len(pairs), bool)), dimensions={})
    for dimension in DIMENSIONS:
        say(f'bootstrapping by {DIMENSIONS[dimension].lower()}')
        of_query = query_frame['query'].map(info[dimension]).fillna(UNKNOWN)
        of_pair = pairs.patient_id.map(info[dimension]).fillna(UNKNOWN)
        out['dimensions'][dimension] = {c: one(of_query.eq(c).to_numpy(), of_pair.eq(c).to_numpy())
                                        for c in categories(dimension, set(of_query) | set(of_pair))}
    out['labeled_pairs'] = int(len(pairs))
    return out


def _interval(cell: dict, key: str) -> dict:
    return dict(value=cell.get(key), low=cell.get('low'), high=cell.get('high'))


def suppress_groups(raw: dict) -> dict:
    """``group_metrics``' result for the report: query counts suppressed per section (as Table 1's treatment-starts
    column), and a group's metrics hidden when it has fewer than 11 patients."""
    def row(category, cell, shown):
        small_q, small_p = cell['query_persons'] < SMALL, cell['pair_persons'] < SMALL
        return dict(category=category, queries=shown,
                    precision=None if small_q else _interval(cell['precision'], 'mean'),
                    mean_score=None if small_q else _interval(cell['mean_score'], 'mean'),
                    auroc=None if small_p else _interval(cell['auroc'], 'value'),
                    hidden=dict(precision=small_q and cell['queries'] > 0, auroc=small_p and cell['pair_persons'] > 0))
    overall = raw['overall']
    sections = []
    for dimension, groups in raw['dimensions'].items():
        names = list(groups)
        cells = suppress([groups[c]['queries'] for c in names], overall['queries'])
        sections.append(dict(key=dimension, label=DIMENSIONS[dimension],
                             rows=[row(c, groups[c], cell) for c, cell in zip(names, cells)]))
    return dict(overall=row('All', overall, dict(n=overall['queries'], pct=100.0, text=None)), sections=sections)


# --------------------------------------------------------------------------- the cached result


def identity_of(run: Path, extract_root: Path) -> dict:
    from matchminer_ai.training.cohort_format import file_stamp
    from matchminer_ai.training.trialspace import stage
    structured = Path(extract_root) / 'structured_data'
    denominator = denominator_of(run)
    return dict(version=VERSION, metrics=stage.receipt_id(run / 'metrics'), patients=stage.receipt_id(run / 'patients'),
                denominator=file_stamp(denominator),
                extract={name: file_stamp(structured / name) for name in (PT_INFO, PT_DEMOGRAPHICS, TREATMENT_PLAN)})


def share(missing: int, total: int) -> dict | None:
    """The share of treatment starts a linkage step reached, as a percentage; 'all but <11' when 1 to 10 missed."""
    if not total:
        return None
    if 0 < missing < SMALL:
        return dict(pct=None, text=f'all but <{SMALL}')
    return dict(pct=100.0 * (total - missing) / total, text=None)


def aggregates(run: Path, extract_root: Path, progress=None) -> dict:
    """The run's aggregates, before suppression: the eval, linkage counts, Table 1's counts and the metrics by group."""
    frame, kind = run_patients(run, extract_root, progress)
    missing = dict(denominator=~frame.in_denominator, registration=~frame.linked.astype(bool),
                   birth_date=frame.age.isna(), plan_diagnosis=frame.cancer_type.eq(UNKNOWN))
    linkage = dict(events=int(len(frame)), persons=int(frame.person.nunique()),
                   missing={k: int(v.sum()) for k, v in missing.items()})
    groups = group_metrics(run, frame, progress)
    return dict(eval=kind, linkage=linkage, table1=table1_counts(frame), groups=groups)


def run_demographics(run, extract_root, progress=None, *, cache=True) -> dict:
    """A real run's Table 1 and metrics by group, suppressed for the report. The aggregates are cached in
    ``<run>/reports/`` (owner-only, beside the run's own PHI) and reused while its inputs are unchanged. Raises
    ``NoDemographics`` for a synthetic run."""
    run, extract_root = Path(run), Path(extract_root)
    if not (run / 'metrics' / 'metrics.json').is_file():
        raise DemographicsError('The run has no metrics yet.')
    identity = identity_of(run, extract_root)
    path = run / 'reports' / CACHE
    raw = None
    if cache and path.is_file():
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            raw = None
        if raw is not None and raw.get('identity') != identity:
            raw = None
    if raw is None:
        started = time.time()
        raw = dict(identity=identity, built_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   **aggregates(run, extract_root, progress), seconds=None)
        raw['seconds'] = round(time.time() - started, 1)
        if cache:
            write_private(path, json.dumps(raw, indent=1) + '\n')
    link = raw['linkage']
    return dict(identity=raw['identity'], built_at=raw['built_at'], eval=raw['eval'],
                linkage=dict(events=link['events'], persons=link['persons'],
                             **{k: share(v, link['events']) for k, v in link['missing'].items()}),
                table1=table1(raw['table1']), groups=suppress_groups(raw['groups']),
                labeled_pairs=raw['groups']['labeled_pairs'], rule=RULE)


def write_private(path: Path, text: str | bytes) -> None:
    """Write a file 600 in a directory 700, atomically."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(temporary, flags, 0o600), 'wb') as stream:
        stream.write(text.encode('utf-8') if isinstance(text, str) else text)
    os.chmod(temporary, 0o600)
    temporary.replace(path)
