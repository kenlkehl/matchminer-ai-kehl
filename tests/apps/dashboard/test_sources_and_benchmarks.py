"""Each mode's sources (synthetic mode's trial store and patients, real mode without a population cache) and the
Benchmarking tab (run discovery, adjudication through the app, owner-only decisions), on fabricated data."""
from __future__ import annotations

import os
import stat

import pytest

from .conftest import FakeEncoder


def test_synthetic_mode_shows_the_runs_trials_with_the_rounds_cohort_vectors(dataset, fake_model, tmp_path):
    from fastapi.testclient import TestClient

    from matchminer_ai.apps.dashboard.server import create_app
    from matchminer_ai.apps.dashboard.settings import DashSettings
    from matchminer_ai.apps.dashboard.state import DashState

    settings = DashSettings(mode="synthetic", synthetic_run=dataset["root"], store_dir=tmp_path / "store", device="cpu",
                            embedding_caches=(tmp_path / "hf",), embedding_python=tmp_path / "missing-python")
    state = DashState(settings, encoder=FakeEncoder(), log=lambda *_: None)
    assert state.patient_source == "training" and state.store.get("source") == "synthetic"
    corpus_cohorts = {row["id"] for row in state.training_corpus.cohorts}
    assert set(state.store["cohorts"]) == corpus_cohorts  # every corpus cohort, under its own id
    assert not state.cohort_vectors.missing(state.store)  # the round's vectors, not embedded again
    client = TestClient(create_app(state), base_url="http://127.0.0.1")
    meta = client.get("/api/meta").json()
    assert meta["mode"] == "synthetic" and "benchmarking" not in meta["tabs"] and not meta["sign_in"]
    found = client.post("/api/match/trials-for-patient", json={"patient_id": "patient_a", "k": 3}).json()
    assert found["results"] and found["results"][0]["cohort"]["cohort_id"] in corpus_cohorts


def test_real_mode_reads_the_oa_trials_and_always_signs_in(dataset, fake_model, tmp_path):
    from matchminer_ai.apps.dashboard.settings import DashSettings
    from matchminer_ai.apps.dashboard.state import make_builder

    settings = DashSettings(mode="real", synthetic_run=dataset["root"], store_dir=tmp_path / "store", device="cpu")
    assert settings.require_token and settings.trial_data is not None
    assert type(make_builder(settings)).__name__ == "Builder"  # the OA trials


def _judge_run(root):
    from matchminer_ai.evaluation.dual_judge import compare, judging

    items = [dict(item_id=f"b-{n}", task="boilerplate", cohort_id="NCT0-ctgov-cohort-0", number=None,
                  criterion_type=None, stratum="s", weight=1.0, teacher={"boilerplate": "yes"},
                  decision_p_yes={"boilerplate": 0.8}, messages=[{"role": "user", "content": "synthetic task"}])
             for n in (1, 2)]
    judging.write_json(root / "build.json", {"version": "test", "components": {"decision": {"items": 2}}})
    judging.write_jsonl(root / "items" / "decision.jsonl", items)
    judging.write_jsonl(root / "requests" / "decision.jsonl",
                        [dict(request_id=str(n), item_id=f"b-{n}") for n in (1, 2)])
    yes, no = {"reason": "r", "excluded": False}, {"reason": "r", "excluded": True}
    for judge, answers in (("databricks", (yes, yes)), ("vertex", (yes, no))):  # a Databricks judge is compared
        judging.write_jsonl(root / "responses" / judge / "decision.jsonl", [
            dict(request_id=str(n), item_id=f"b-{n}", component="decision", judge=judge, attempt=1, final=True,
                 status="ok", answer=a) for n, a in zip((1, 2), answers)])
    compare.compare(root, audit_fraction=0.0)


def test_benchmarks_find_runs_and_adjudicate_through_the_app(tmp_path):
    from matchminer_ai.apps.dashboard.benchmarking import Benchmarks
    from matchminer_ai.evaluation.dual_judge import judging

    run = tmp_path / "no_phi" / "evals" / "judge_run"
    _judge_run(run)
    assert judging.compared_judges(run) == ("vertex", "databricks")
    with pytest.raises(ValueError, match="phi"):
        Benchmarks("synthetic", roots=[tmp_path / "phi"])
    benchmarks = Benchmarks("synthetic", roots=[tmp_path / "no_phi"], user="reviewer")
    [found] = benchmarks.runs()
    assert found["kind"] == "judge" and found["judges"] == ["databricks", "vertex"] and found["compared"]
    store = benchmarks.store(found["id"])
    assert store.state()["components"]["decision"]["queued"] == 1  # b-2: the judges disagree
    item = store.item("decision", "b-2")
    [unit] = item["units"]
    no = next(j["value"] for j in unit["judges"] if j["value"] != "yes")
    decided = store.decide({"unit_id": unit["unit_id"], "choice": "other", "value": no})
    assert decided["units"][0]["decision"]["value"] == no
    [row] = judging.read_jsonl(run / "adjudications.jsonl")
    assert row["picked_judge"] == "vertex" and row["user"] == "reviewer"  # the judge who gave the chosen value
    assert stat.S_IMODE(os.stat(run / "adjudications.jsonl").st_mode) == 0o600


def test_fidelity_export_builds_and_downloads_a_pdf_without_demographics_in_synthetic_mode(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from matchminer_ai.apps.dashboard.benchmarking import SYNTHETIC_NOTE, Benchmarks
    from matchminer_ai.apps.dashboard.core.errors import InputError
    from matchminer_ai.apps.dashboard.routes.benchmarking import router
    from matchminer_ai.evaluation.dual_judge import judging
    from matchminer_ai.training.trialspace import stage

    run = tmp_path / "no_phi" / "benchmarks" / "fidelity_run"  # a fabricated synthetic run's results
    interval = dict(mean=0.8, low=0.7, high=0.9, n=10)
    value = dict(value=0.9, low=0.85, high=0.95)
    judging.write_json(run / "run_status.json", {"updated_at": "2026-11-23T06:00:00-0500", "released": True,
                                                  "jobs": {"metrics": {"state": "done", "tries": 1}}})
    judging.write_json(run / "design_notes.json", ["Synthetic run (no PHI)."])
    judging.write_json(run / "metrics" / "metrics.json", {
        "retrieval": {"top": 20, "table": {"patient_centric/trialspace_round2_decision": dict(
            queries=10, queries_with_fewer_than_top=0, queries_without_labels=0, unlabeled_pairs=0,
            precision_at_top=interval, mean_score=dict(interval, mean=3.1, low=2.9, high=3.3))},
            "paired_differences": {}},
        "decision": {"pooled": {"reasonableness": dict(
            n=200, clusters=10, prevalence_yes=0.6, prevalence_no=0.4, mean_predicted=0.6, observed_rate=0.6,
            auroc=value, auprc_yes=value, auprc_no=value, brier=value, ece=value, calibration_intercept=value,
            calibration_slope=value, reliability=[dict(n=100, p_min=0.0, p_max=0.5, mean_predicted=0.2, observed=0.25),
                                                  dict(n=100, p_min=0.5, p_max=1.0, mean_predicted=0.8,
                                                       observed=0.75)])}},
        "counts": {"union_pairs": 200}})
    (run / "patients").mkdir()
    stage.write_receipt(run / "patients", "retro-patients", dict(stage="synthetic_patients", source_split="test"), [])
    benchmarks = Benchmarks("synthetic", roots=[tmp_path / "no_phi"], extract_root=tmp_path / "phi" / "extract")
    app = FastAPI()
    app.include_router(router(benchmarks))
    client = TestClient(app)
    [found] = client.get("/api/benchmarks").json()["runs"]
    with pytest.raises(InputError, match="Export the results first"):
        client.get(f"/api/benchmarks/fidelity/{found['id']}/report.pdf")
    started = client.post(f"/api/benchmarks/fidelity/{found['id']}/report")
    assert started.status_code == 202
    job = benchmarks.jobs.get(started.json()["job"]["id"])
    job.future.result(timeout=60)
    assert job.status == "done" and job.result["demographics"] is False and job.result["note"] == SYNTHETIC_NOTE
    pdf = client.get(f"/api/benchmarks/fidelity/{found['id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF") and "fidelity-fidelity_run.pdf" in pdf.headers["content-disposition"]
    assert stat.S_IMODE(os.stat(run / "reports" / "fidelity_report.pdf").st_mode) == 0o600
    assert client.get(f"/api/benchmarks/fidelity/{found['id']}").json()["report"]["bytes"] == len(pdf.content)
