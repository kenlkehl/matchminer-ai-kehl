"""Demographics of a fidelity run (matchminer_ai.evaluation.demographics), on fabricated data only: the small-cell
rule, and Table 1 and the metrics by group of a fabricated real run linked to a fabricated extract."""
import json
import os
import stat

import pandas as pd
import pytest

from matchminer_ai.evaluation import demographics as dm
from matchminer_ai.labeling.files import write_json, write_jsonl
from matchminer_ai.training.cohort_format import file_stamp
from matchminer_ai.training.trialspace import stage


def test_small_cells_are_hidden_with_a_complement_and_totals_stay_exact():
    small, other = 'small', 'complement'
    assert dm.hidden_cells([0, 5, 100, 200]) == [None, small, other, None]  # 5, then the smallest other count
    assert dm.hidden_cells([5, 7, 100]) == [small, small, None]  # two hidden adding up to 11 or more
    assert dm.hidden_cells([1, 1, 50, 60]) == [small, small, other, None]  # 1 + 1 alone would give both away
    assert dm.hidden_cells([10, 10, 500, 600]) == [small, small, other, None]  # 10 + 10 = 20: both 10
    assert dm.hidden_cells([1, 11, 500, 600]) == [small, other, other, None]  # 1 + 11 = 12: only 1 and 11
    assert dm.hidden_cells([12, 30, 0]) == [None, None, None]
    cells = dm.suppress([0, 5, 100, 200], 305)
    assert cells[0] == dict(n=0, pct=0.0, text=None)  # zero shows as 0
    assert cells[1] == dict(n=None, pct=None, text='<11') and cells[2] == dict(n=None, pct=None, text='*')
    assert cells[3]['n'] == 200 and cells[3]['pct'] == pytest.approx(200 / 305 * 100)


def fabricated_run(tmp_path):
    """A fabricated real run (64 treatment starts of 60 patients) and extract; no real data."""
    run, extract = tmp_path / 'phi' / 'run', tmp_path / 'extract' / 'structured_data'
    extract.mkdir(parents=True)
    starts = 64  # patient 59 has five starts (59..63), everyone else one
    events = [dict(pseudo_mrn=str(i), dfci_mrn=str(1000 + min(i, 59)), trial_start_dt=f'2020-0{1 + i % 9}-15',
                   epic_tplan_id=f'T{i}', split='test') for i in range(starts)]
    for row in events[60:]:
        row['trial_start_dt'] = f'202{int(row["pseudo_mrn"]) - 59}-03-01'
    events.append(dict(events[1], epic_tplan_id='T1b'))  # a second plan starting the same day
    denominator = tmp_path / 'phi' / 'denominators' / 'processed_soc_treatments.csv'
    denominator.parent.mkdir(parents=True)
    pd.DataFrame(events).to_csv(denominator, index=False)
    write_jsonl(run / 'corpus' / 'patients.jsonl',
                [dict(id=str(i), text='synthetic', boilerplate='', split='val', person=min(i, 59)) for i in range(starts)])
    (run / 'patients').mkdir(parents=True)
    stage.write_receipt(run / 'patients', 'retro-patients', dict(version='v', stage='patients', split='val',
                        inputs={denominator.name: file_stamp(denominator)}), [])
    receipt = json.loads((run / 'patients' / 'stage_receipt.json').read_text())
    receipt['provenance']['argv'] = ['prepare.py', 'patients', '--denominator', str(denominator)]
    (run / 'patients' / 'stage_receipt.json').write_text(json.dumps(receipt))
    mrns = [1000 + j for j in range(60)]
    pd.DataFrame(dict(DFCI_MRN=mrns, LAST_NM='x', BIRTH_DT=['1950-06-01 00:00:00'] * 59 + [None],
                      GENDER_NM=['FEMALE'] * 40 + ['MALE'] * 18 + [None, 'UNKNOWN'],
                      SEX_AT_BIRTH_NM=[None] * 58 + ['Male', 'Female'])).to_csv(extract / 'PT_INFO.csv', index=False)
    pd.DataFrame(dict(DFCI_MRN=mrns, DERIVED_RACE_NM=['White'] * 45 + ['Black or African American'] * 12
                      + ['Asian/White', 'Asian', None], HISPANIC_IND=['N'] * 58 + ['Y', None])
                 ).to_csv(extract / 'PT_DEMOGRAPHICS.csv', index=False)
    codes = {f'T{i}': 'C61' if 40 <= i < 52 else 'C50.911' for i in range(starts)}
    codes |= {'T1': 'C78.0', 'T1b': 'Z51.11, C34.12', 'T5': 'C18.7, C78.7'}
    pd.DataFrame(dict(EPIC_TPLAN_ID=list(codes), TPLAN_ICD_DX_CODES=list(codes.values()))).to_csv(
        extract / 'TREATMENT_PLAN.csv', index=False)
    # patient-centric (c) lists of two cohorts each, teacher labels and decision scores
    lists, labels, scores = [], [], []
    for i in range(starts):
        for rank, cohort in enumerate(('NCT1-ctgov-cohort-1', 'NCT2-ctgov-cohort-1'), 1):
            lists.append(dict(workflow='patient_centric', query_id=str(i), ranker='trialspace_round2_decision',
                              rank=rank, patient_id=str(i), cohort_id=cohort, score=1.0))
            yes = (i + rank) % 3 != 0
            answers = dict(reasonableness='yes' if yes else 'no', cancer_type='yes', burden='no',
                           treatment_history='no', biomarker='no')
            labels.append(dict(patient_id=str(i), cohort_id=cohort, answers=answers, score=2 if yes else 0))
            scores.append(dict(patient_id=str(i), cohort_id=cohort, p_reasonableness=0.8 if yes else 0.3))
    (run / 'lists_c').mkdir()
    pd.DataFrame(lists).to_parquet(run / 'lists_c' / 'lists.parquet', index=False)
    write_jsonl(run / 'labels' / 'match_c' / 'scores.jsonl', labels)
    (run / 'decide' / 'match').mkdir(parents=True)
    pd.DataFrame(scores).to_parquet(run / 'decide' / 'match' / 'scores.parquet', index=False)
    write_json(run / 'metrics' / 'metrics.json', {})
    stage.write_receipt(run / 'metrics', 'retro-metrics', dict(version='v'), ['metrics.json'])
    return run, extract.parent


def test_table1_and_groups_of_a_fabricated_real_run(tmp_path, monkeypatch):
    run, extract = fabricated_run(tmp_path)
    result = dm.run_demographics(run, extract)
    assert result['eval']['kind'] == 'soc' and result['eval']['splits'] == ['test']
    link = result['linkage']
    assert (link['events'], link['persons']) == (64, 60) and link['registration']['pct'] == 100.0
    assert link['birth_date'] == dict(pct=None, text='all but <11')  # patient 59's five starts
    t1 = result['table1']
    assert [c['n'] for c in t1['columns']] == [60, 64]  # totals stay exact
    assert 69 < t1['age']['events']['median'] < 71
    section = {s['key']: {r['category']: r['cells'] for r in s['rows']} for s in t1['sections']}
    sex = section['sex']  # GENDER_NM, else (missing) the sex at birth: patient 58 Male, 59 (UNKNOWN) Unknown
    assert sex['Female']['patients'] == dict(n=40, pct=pytest.approx(200 / 3), text=None)
    assert sex['Unknown']['events']['text'] == '<11' and sex['Male']['events']['text'] == '*'  # its complement
    assert section['race']['White']['events']['n'] == 45 and section['race']['Black or African American'][
        'patients']['text'] == '*'  # hidden with Asian, Multiple and Unknown
    cancer = {k: v['patients'] for k, v in section['cancer_type'].items()}  # first start; plan codes in order
    assert cancer['Breast']['n'] == 46 and cancer['Melanoma and other skin'] == dict(n=0, pct=0.0, text=None)
    assert cancer['Lung and other thoracic']['text'] == cancer['Colorectal and anal']['text'] == '<11'
    groups = result['groups']
    assert groups['overall']['queries']['n'] == 64 and groups['overall']['precision']['value'] == pytest.approx(
        sum(((i + 1) % 3 != 0) + ((i + 2) % 3 != 0) for i in range(64)) / 128)
    assert groups['overall']['auroc']['value'] == 1.0
    by_sex = {r['category']: r for r in groups['sections'][3]['rows']}
    assert by_sex['Male']['precision'] is not None and by_sex['Male']['queries']['text'] == '*'
    assert by_sex['Unknown']['precision'] is None and by_sex['Unknown']['hidden']['precision']  # one patient
    cache = run / 'reports' / dm.CACHE
    assert stat.S_IMODE(os.stat(cache).st_mode) == 0o600 and stat.S_IMODE(os.stat(cache.parent).st_mode) == 0o700
    assert '1000' not in cache.read_text() and '1950' not in cache.read_text()  # no MRN, no birth date
    monkeypatch.setattr(dm, 'run_patients', None)  # reused while the inputs stand: the extract is not read again
    assert dm.run_demographics(run, extract) == result
