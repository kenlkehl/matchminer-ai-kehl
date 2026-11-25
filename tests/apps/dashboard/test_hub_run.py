"""Synthetic mode's run built from its published dataset (``apps.dashboard.hub_run``), over a fabricated dataset laid
out as mmai-synthetic-1226 is (zstd JSON-lines corpus shards, parquet tables), with a fake encoder: no download and
no model."""
from __future__ import annotations

import json

import numpy as np
import pytest

from .conftest import COHORTS, PATIENTS, criteria_text, items, unit

REVISION = "1" * 40
TRIALSPACE_REVISION = "7b1cdebf2e9b65ed261113948fa187b0e9da886e"


def write_shard(path, rows):
    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.CompressedOutputStream(pa.OSFile(str(path), "wb"), "zstd") as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())


def published_dataset(root):
    """The dataset files the app reads, in the published layout."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    corpus = root / "trialspace" / "trialspace_corpus"
    write_shard(corpus / "patients-00000.jsonl.zst",
                [{"id": pid, "text": key, "boilerplate": bp, "split": "train", "source_trial_id": trial,
                  "source_split": "train"} for pid, key, bp, _, trial, _ in PATIENTS])
    write_shard(corpus / "cohorts-00000.jsonl.zst",
                [{"id": cid, "trial_id": trial, "source": source, "cohort_name": name,
                  "text": criteria_text(lines, False), "boilerplate": criteria_text(lines, True), "items": items(lines),
                  "split": "train", "source_split": "train"} for cid, trial, source, name, lines in COHORTS])
    write_shard(corpus / "source_trial_pairs-00000.jsonl.zst",
                [{"patient_id": "patient_a", "cohort_id": "NCT00000001-cohort-0"}])
    (root / "trials").mkdir(parents=True)
    pq.write_table(pa.table({"nct_id": ["NCT00000001", "NCT00000002"], "phases": ['["PHASE2"]', '["PHASE3"]'],
                             "overall_status": ["RECRUITING", "COMPLETED"]}),
                   root / "trials/ctgov_sample_trials.parquet")
    pq.write_table(pa.table({"nct_id": ["NCT00000001", "NCT00000002"], "title": ["Lung trial", "Breast trial"],
                             "brief_summary": ["A summary.", ""], "detailed_description": ["", ""]}),
                   root / "trials/ctgov_trials_with_cohorts.parquet")
    (root / "summaries").mkdir()
    pq.write_table(pa.table({"patient_id": [p[0] for p in PATIENTS], "patient_class": [p[3] for p in PATIENTS],
                             "nct_id": [p[4] for p in PATIENTS], "cohort_trial_id": [p[5] for p in PATIENTS]}),
                   root / "summaries/synthetic_patient_metadata.parquet")
    (root / "patients").mkdir()
    pq.write_table(pa.table({"patient_id": ["patient_a"], "date": ["2024-01-01"], "event_type": ["clinic_note"],
                             "synthetic_note": ["A note."]}), root / "patients/all_synthetic_notes.parquet")
    return root


@pytest.fixture
def published(tmp_path, monkeypatch, site_config):
    from matchminer_ai.apps.dashboard import hub_run

    root = published_dataset(tmp_path / "published")
    site_config.write_text(site_config.read_text() + "  synthetic_run: "
                           f"hf://datasets/example-org/example-run@{REVISION}\n")
    def hub_path(ref):
        path = root / ref.path
        if not path.exists():
            raise FileNotFoundError(ref.path)
        return path

    monkeypatch.setattr(hub_run, "hub_path", hub_path)
    monkeypatch.setattr(hub_run, "published_trialspace", lambda: ("ksg-dfci/TrialSpace-1226", TRIALSPACE_REVISION))
    calls = []

    def encode(settings, texts, prompt, log, chunk=8192):
        calls.append((settings.model, prompt, len(texts)))
        return np.stack([unit(len(t) + 7 * i) for i, t in enumerate(texts)])

    monkeypatch.setattr(hub_run, "_encode", encode)
    return {"root": root, "calls": calls}


def test_the_published_run_becomes_a_run_folder_the_app_accepts(published, tmp_path):
    from matchminer_ai.apps.dashboard import hub_run
    from matchminer_ai.apps.dashboard.core.cache import JSONCache
    from matchminer_ai.apps.dashboard.core.corpus import Corpus
    from matchminer_ai.apps.dashboard.core.vectors import encoder_settings, select_round
    from matchminer_ai.apps.dashboard.settings import DashSettings
    from matchminer_ai.apps.dashboard.state import make_builder

    settings = DashSettings(mode="synthetic", store_dir=tmp_path / "data" / "no_phi" / "store", device="cpu",
                            embedding_caches=(tmp_path / "hf",), embedding_python=tmp_path / "missing-python")
    assert settings.synthetic_run == (tmp_path / "data" / "no_phi" / "synthetic_runs"
                                      / f"example-org--example-run--{REVISION}")
    record = hub_run.prepare(settings, log=lambda *_: None)
    assert record["dataset"] == f"hf://datasets/example-org/example-run@{REVISION}"
    assert [c[1:] for c in published["calls"]] == [("patient_summary", len(PATIENTS)), ("this_space", len(COHORTS))]

    # The round checks the app makes: the published TrialSpace (a Hub model) made round 3 from exactly these rows.
    corpus = Corpus(settings.path("corpus"), settings.path("patient_metadata"), JSONCache(tmp_path / "cache"))
    info, skipped = select_round(settings.path("trialspace"), corpus)
    assert skipped == [] and info.number == hub_run.ROUND
    assert (info.model.kind, info.model.model, info.model.revision) == (
        "hub", "ksg-dfci/TrialSpace-1226", TRIALSPACE_REVISION)
    assert encoder_settings(info, cache_dir=None)[1]  # the query encoder rebuilds the same identity
    assert corpus.meta["patient_a"]["cls"] == "positive"
    assert settings.path("notes").resolve() == (published["root"] / "patients/all_synthetic_notes.parquet").resolve()
    store = make_builder(settings).build()
    assert store["counts"]["trials"] == 2 and store["trials"]["NCT00000001"]["short_title"] == "Lung trial"

    # A restart reuses the folder.
    assert hub_run.prepare(settings, log=lambda *_: None)["corpus"] == record["corpus"]
    assert len(published["calls"]) == 2


def test_round_encoder_settings_are_training_minings():
    """The vectors are made as training's mining makes a round's (same encoder identity)."""
    pytest.importorskip("sklearn")
    from matchminer_ai.apps.dashboard.hub_run import round_encoder_settings
    from matchminer_ai.embedding.encoder import encoder_identity
    from matchminer_ai.training.trialspace import mine, mining

    args = mine.parser().parse_args(["--corpus", "corpus", "--output", "out", "--model", "ksg-dfci/TrialSpace-1226",
                                     "--revision", TRIALSPACE_REVISION, "--devices", "cpu"])
    assert encoder_identity(mining.encoder_settings(args)) == encoder_identity(
        round_encoder_settings("ksg-dfci/TrialSpace-1226", TRIALSPACE_REVISION, "cuda:0"))


def test_published_round_vectors_are_linked_not_embedded(published, tmp_path):
    """A dataset that publishes the run's round vectors (with the mining manifest naming these corpus rows) is not
    embedded again."""
    from matchminer_ai.apps.dashboard import hub_run
    from matchminer_ai.apps.dashboard.core.cache import JSONCache
    from matchminer_ai.apps.dashboard.core.corpus import Corpus
    from matchminer_ai.apps.dashboard.core.vectors import select_round
    from matchminer_ai.apps.dashboard.settings import DashSettings
    from matchminer_ai.labeling.corpus import load_corpus
    from matchminer_ai.labeling.files import digest

    from .conftest import DIM, encoder_identity_for, write_jsonl

    staging = tmp_path / "corpus"  # the published rows, to name them in the published manifest
    write_jsonl(staging / "patients.jsonl", [json.loads(line) for line in _lines(published["root"], "patients")])
    write_jsonl(staging / "cohorts.jsonl", [json.loads(line) for line in _lines(published["root"], "cohorts")])
    patients, cohorts = load_corpus(staging)
    mining = published["root"] / "trialspace" / f"trialspace_round{hub_run.ROUND}" / "mining"
    mining.mkdir(parents=True)
    np.save(mining / "patients.npy", np.stack([unit(i, DIM) for i in range(len(patients))]))
    np.save(mining / "cohorts.npy", np.stack([unit(50 + i, DIM) for i in range(len(cohorts))]))
    (mining / "manifest.json").write_text(json.dumps({
        "corpus": "the-run-corpus-receipt", "dimension": DIM, "patients": digest(patients), "cohorts": digest(cohorts),
        "encoder": encoder_identity_for("0" * 64, None)}))  # the fit's receipt id, as training names a fitted model

    settings = DashSettings(mode="synthetic", store_dir=tmp_path / "data" / "no_phi" / "store", device="cpu",
                            embedding_caches=(tmp_path / "hf",), embedding_python=tmp_path / "missing-python")
    hub_run.prepare(settings, log=lambda *_: None)
    assert published["calls"] == []
    corpus = Corpus(settings.path("corpus"), settings.path("patient_metadata"), JSONCache(tmp_path / "cache"))
    info, skipped = select_round(settings.path("trialspace"), corpus)
    assert skipped == [] and info.model.kind == "hub" and info.shapes["patients.npy"] == [len(PATIENTS), DIM]


def _lines(root, name):
    import pyarrow as pa

    shard = root / "trialspace" / "trialspace_corpus" / f"{name}-00000.jsonl.zst"
    with pa.CompressedInputStream(pa.OSFile(str(shard), "rb"), "zstd") as stream:
        return [line for line in stream.read().decode().splitlines() if line.strip()]


def test_another_trial_set_from_a_cohort_table_and_trial_tables(dataset, tmp_path):
    """Synthetic mode's trials from the site config's trial_cohorts (the pipeline's cohort table, text columns only,
    as published) and trial_tables (merged by nct_id), instead of the run's own."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from matchminer_ai.apps.dashboard.settings import DashSettings
    from matchminer_ai.apps.dashboard.state import make_builder

    criteria = "Inclusion criteria:\nMetastatic melanoma.\nExclusion criteria:\nPrior anti-PD-1."
    items = json.dumps([{"number": 1, "heading": "Inclusion criteria:", "text": "Metastatic melanoma.",
                         "criterion_type": "inclusion", "boilerplate": False}])
    rows = [("NCT00000009", 0, 2, "yes", "Arm A"), ("NCT00000009", 1, 2, "yes", "Arm B"),
            ("NCT00000010", 0, 1, "no", "All participants")]
    pq.write_table(pa.table({
        "cohort_trial_id": [f"{t}-ctgov-cohort-{n}" for t, n, *_ in rows], "trial_id": [r[0] for r in rows],
        "cohort_number": [str(r[1]) for r in rows], "cohort_count": [str(r[2]) for r in rows],
        "multiple_cohorts": [r[3] for r in rows], "cohort_name": [r[4] for r in rows],
        "cohort_eligibility_criteria": [criteria] * 3, "cohort_key_criteria": [criteria] * 3,
        "cohort_boilerplate_criteria": [""] * 3, "cohort_criteria_items": [items] * 3,
        "cohort_criteria_source": ["ctgov"] * 3, "cancer_treatment_trial": ["yes"] * 3}), tmp_path / "cohorts.parquet")
    pq.write_table(pa.table({"nct_id": ["NCT00000009", "NCT00000010"], "title": ["Melanoma A", "Melanoma B"],
                             "overall_status": ["RECRUITING", "COMPLETED"], "phases": ['["PHASE2"]', '["PHASE1"]']}),
                   tmp_path / "trials.parquet")
    settings = DashSettings(mode="synthetic", synthetic_run=dataset["root"], store_dir=tmp_path / "store",
                            device="cpu", trial_cohorts=tmp_path / "cohorts.parquet",
                            trial_tables=(tmp_path / "trials.parquet",))
    store = make_builder(settings).build()
    assert sorted(store["trials"]) == ["NCT00000009", "NCT00000010"]
    trial = store["trials"]["NCT00000009"]
    assert (trial["short_title"], trial["status"], trial["is_open"]) == ("Melanoma A", "Recruiting", True)
    assert [store["cohorts"][c]["name"] for c in trial["cohort_ids"]] == ["Arm A", "Arm B"]
    assert store["cohorts"][trial["cohort_ids"][0]]["items"][0]["text"] == "Metastatic melanoma."
