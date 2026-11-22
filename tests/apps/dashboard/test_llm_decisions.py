"""The LLM decision checks (matchminer_ai.apps.dashboard/llm_decisions.py): the LLM alternative to the decision model,
over the fabricated trials of test_dash.py, with fakes for the inference package's three LLM functions. No request
leaves the machine."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from .conftest import wait_job
from .test_dash import dash, dash_client, dash_settings, fake_ncit, trial_data  # noqa: F401  (fixtures)

NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
TEACHER = "http://teacher.example:8000/v1"
COHORT = "NCT90000001-cohort-0"  # has common (boilerplate) criteria
CTGOV = "NCT90000001-ctgov-cohort-0"  # has none


def labels(reasonable: bool, cancer_type: bool = True) -> str:
    def one(yes):
        return {"answer": "yes" if yes else "no", "evidence_status": "supported", "reason": "Fabricated reason."}

    return json.dumps({"reasonableness": one(reasonable), "cancer_type": one(cancer_type), "burden": one(False),
                       "treatment_history": one(False), "biomarker": one(False)})


@pytest.fixture
def fake_llm(monkeypatch):
    """The package's LLM functions, faked: EGFR summaries are reasonable for EGFR cohorts, others not; "ECOG 2"
    is a common exclusion."""
    import matchminer_ai.matching as matching
    from matchminer_ai.matching.match_labels import match_score, parse_match_labels

    calls = {"match": [], "boilerplate": [], "criteria": [], "configs": []}

    def score_match_quality_with_llm(frame, *, config=None, return_metadata=False):
        calls["match"] += frame.to_dict("records")
        calls["configs"].append(config)
        texts = [labels("egfr" in p.lower() and "egfr" in c.lower()) if "garbled" not in c else "not json"
                 for p, c in zip(frame["cancer_history_summary"], frame["cohort_key_criteria"])]
        out = frame[["patient_id", "cohort_trial_id"]].copy()
        out["llm_match_quality_score"] = [match_score(parse_match_labels(t)) if t != "not json" else -1 for t in texts]
        out["llm_match_quality_answer_text"] = texts
        out["llm_match_quality_reasoning_text"] = "Fabricated thinking summary."
        out["llm_match_quality_parse_status"] = ["parsed" if t != "not json" else "parse_failed: JSON" for t in texts]
        return out

    def exclusion_criteria_check_with_llm(frame, *, config=None, return_metadata=False):
        calls["boilerplate"] += frame.to_dict("records")
        out = frame[["patient_id", "cohort_trial_id"]].copy()
        excluded = ["ECOG 2" in text for text in frame["general_exclusion_criteria_evidence"]]
        out["llm_exclusion_criteria_pass"] = [not e for e in excluded]
        out["llm_exclusion_criteria_answer_text"] = ["1. ECOG: fails. Yes!" if e else "1. ECOG: met. No!" for e in excluded]
        out["llm_exclusion_criteria_reasoning_text"] = ""
        out["llm_exclusion_criteria_parse_status"] = "parsed"
        return out

    def check_eligibility_criteria_with_llm(frame, *, config, return_metadata=False, progress=None):
        calls["criteria"].append(frame.iloc[0].to_dict())
        items = json.loads(frame.iloc[0]["cohort_criteria_items"])
        rows = [{"patient_id": frame.iloc[0]["patient_id"], "cohort_trial_id": frame.iloc[0]["cohort_trial_id"],
                 "criterion_number": i["number"], "criterion_type": i["criterion_type"], "boilerplate": i["boilerplate"],
                 "criterion_text": i["text"], "llm_criterion_answer": "no" if i["criterion_type"] == "exclusion" else "yes",
                 "llm_criterion_clearly_fails": i["criterion_type"] == "exclusion", "llm_criterion_status": "met",
                 "llm_criterion_reason": "Fabricated reason.", "llm_criterion_finish_reason": "stop",
                 "llm_criterion_attempts": 1, "llm_criterion_parse_status": "parsed",
                 "llm_criterion_answer_text": '{"answer": "yes"}', "llm_criterion_reasoning_text": "Fabricated."}
                for i in items]
        if progress:
            progress(len(rows), len(rows))
        metadata = {"criteria_contract": "7a7eb230", "model_metadata": {"check_eligibility_criteria_with_llm": {
            "requests": len(rows), "retries": 0, "parse_failures": 0, "unchecked_lines": 0, "structured_outputs": True}}}
        return (pd.DataFrame(rows), metadata) if return_metadata else pd.DataFrame(rows)

    monkeypatch.setattr(matching, "score_match_quality_with_llm", score_match_quality_with_llm)
    monkeypatch.setattr(matching, "exclusion_criteria_check_with_llm", exclusion_criteria_check_with_llm)
    monkeypatch.setattr(matching, "check_eligibility_criteria_with_llm", check_eligibility_criteria_with_llm)
    return calls


def use_teacher(client, model=NEMOTRON):
    """This session's LLM decision checks on an OpenAI-compatible server (Data and Models)."""
    response = client.post("/api/models", json={"llm_backend": "local", "endpoint": TEACHER, "model": model,
                                                "task": "llm_decisions"})
    assert response.status_code == 200, response.text
    return response.json()


def pairs(cohort=COHORT, patients=("patient_a", "patient_b")):
    return {"direction": "patients_for_cohort", "query_id": cohort, "target_ids": list(patients)}


def score(client, body):
    started = client.post("/api/llm-decisions/score", json=body)
    assert started.status_code == 200, started.text
    started = started.json()
    return wait_job(client, started["job"]["id"])["result"]["results"] if "job" in started else started["results"]


def test_the_unified_task_defaults_to_the_app_connection_and_takes_criteria_llm_connections(tmp_path, monkeypatch):
    from matchminer_ai.apps.dashboard.guidelines.settings import LLM_TASKS, Settings, migrate_task_connections
    from matchminer_ai.apps.dashboard.settings import DashSettings, guideline_settings

    assert "llm_decisions" in LLM_TASKS and "criteria_llm" not in LLM_TASKS
    site = tmp_path / "gateway.yaml"
    site.write_text("databricks:\n  host: https://workspace.example.cloud.databricks.com\n  profile: p\n"
                    "  endpoints:\n    haiku: catalog.schema.team-claude-haiku-5-5\n")
    monkeypatch.setenv("MMAI_SITE_CONFIG", str(site))
    tasks = guideline_settings(DashSettings(mode="synthetic", store_dir=tmp_path / "s", device="cpu"))
    unified = tasks.for_task("llm_decisions")
    assert (unified.llm_backend, unified.model) == ("claude_databricks", "catalog.schema.team-claude-haiku-5-5")
    # A connection saved under the per-criterion task's old name serves the unified task.
    old = {"llm_backend": "local", "endpoint": TEACHER, "model": NEMOTRON}
    moved = Settings(task_connections={"criteria_llm": old, "summarize": {"model": "m"}})
    assert moved.task_connections == {"llm_decisions": old, "summarize": {"model": "m"}}
    assert moved.for_task("llm_decisions").endpoint == TEACHER
    assert migrate_task_connections({"criteria_llm": old, "llm_decisions": {"model": "x"}}) == {
        "llm_decisions": {"model": "x"}}


def test_the_toggle_routes_to_the_decision_model_or_the_llm(dash_client, fake_model, fake_llm):
    use_teacher(dash_client)
    decision = dash_client.post("/api/score", json=pairs()).json()
    wait_job(dash_client, decision["job"]["id"])
    assert fake_model["match"] == 2 and fake_model["boilerplate"] == 2 and not fake_llm["match"]
    results = score(dash_client, pairs())
    assert fake_model["match"] == 2 and len(fake_llm["match"]) == 2 and len(fake_llm["boilerplate"]) == 2
    first = results[0]
    assert first["match"]["scorer"] == "llm" and first["boilerplate"]["scorer"] == "llm"
    # The LLM's requests carry the session's connection, its sampling profile and debug mode (raw answers).
    config = fake_llm["configs"][0]
    assert config.remote["server_urls"] == [TEACHER] and config.debug_mode
    assert config.llm_match_quality["remote"]["model_name"] == NEMOTRON
    assert config.llm_match_quality["sampling_profile"] == "auto"


def test_llm_answers_keep_their_scale_raw_output_and_reasoning(dash_client, fake_llm):
    use_teacher(dash_client)
    a, b = score(dash_client, pairs())
    match = a["match"]
    assert match["score"] == 2 and match["max"] == 6 and match["reasonable"] is True  # reasonable + cancer type
    assert match["answers"]["cancer_type"] == {"answer": "yes", "evidence_status": "supported",
                                               "reason": "Fabricated reason."}
    assert json.loads(match["output"])["burden"]["answer"] == "no" and match["parse_status"] == "parsed"
    assert match["reasoning"] == "Fabricated thinking summary." and match["connection"]["model"] == NEMOTRON
    assert "reasoning" in match["reasoning_note"] and match["scale"] == "LLM match score"
    assert b["match"]["score"] == 0 and b["match"]["reasonable"] is False  # patient_b: no EGFR mention
    assert a["boilerplate"] == {**a["boilerplate"], "pass": True, "excluded": False, "answer": "No"}
    assert b["boilerplate"]["excluded"] is True and b["boilerplate"]["output"].endswith("Yes!")
    assert b["boilerplate"]["reasoning"] == ""  # this backend returned none; the page says so
    # A cohort without common criteria asks no boilerplate question.
    skipped = score(dash_client, pairs(CTGOV, ["patient_a"]))[0]
    assert skipped["boilerplate"] == {"skipped": "the cohort has no boilerplate criteria"}
    # The full answer, for the reasoning dialog: this session's only.
    assert dash_client.get(f"/api/llm-decisions/answers/{match['key']}").json()["output"] == match["output"]
    from fastapi.testclient import TestClient

    with TestClient(dash_client.app, base_url="http://127.0.0.1") as other:
        assert other.get(f"/api/llm-decisions/answers/{match['key']}").status_code == 404


def test_answers_are_cached_per_session_model_and_prompt(dash_client, dash, fake_llm):
    use_teacher(dash_client)
    score(dash_client, pairs())
    lookup = dash_client.post("/api/llm-decisions/lookup", json=pairs()).json()
    assert lookup["estimate"]["requests"] == 0 and lookup["estimate"]["cached"] == 4
    assert lookup["results"][0]["match"]["score"] == 2
    again = dash_client.post("/api/llm-decisions/score", json=pairs()).json()
    assert "job" not in again and len(fake_llm["match"]) == 2  # answered from the session's memory
    score(dash_client, {**pairs(), "rerun": True})
    assert len(fake_llm["match"]) == 4  # "Ask again"
    # Another browser session has its own answers; another model is another answer.
    from fastapi.testclient import TestClient

    with TestClient(dash_client.app, base_url="http://127.0.0.1") as other:
        use_teacher(other)
        fresh = other.post("/api/llm-decisions/lookup", json=pairs()).json()
        assert fresh["estimate"]["requests"] == 4 and fresh["results"][0]["match"] is None
    use_teacher(dash_client, "other/model")
    assert dash_client.post("/api/llm-decisions/lookup", json=pairs()).json()["estimate"]["requests"] == 4
    # Nothing reaches the disk caches.
    assert dash.llm.cache.count() == 0


def test_large_runs_need_confirmation_and_errors_are_reduced(dash_client, dash, fake_llm, monkeypatch):
    import matchminer_ai.apps.dashboard.llm_decisions as module
    import matchminer_ai.matching as matching

    use_teacher(dash_client)
    monkeypatch.setattr(module, "LARGE_RUN", 3)
    refused = dash_client.post("/api/llm-decisions/score", json=pairs())
    assert refused.status_code == 400 and "confirm" in refused.json()["error"] and not fake_llm["match"]
    assert len(score(dash_client, {**pairs(), "confirm": True})) == 2

    def down(*args, **kwargs):
        raise ConnectionError("endpoint down: patient text that must not reach the page")

    monkeypatch.setattr(matching, "score_match_quality_with_llm", down)
    started = dash_client.post("/api/llm-decisions/score", json={**pairs(), "rerun": True, "confirm": True}).json()
    job = wait_job(dash_client, started["job"]["id"])
    assert job["status"] == "error" and job["error"].startswith("ConnectionError") and "patient" not in job["error"]


def test_the_per_criterion_check_runs_as_a_session_job(dash_client, fake_llm):
    use_teacher(dash_client)
    body = {"patient_id": "patient_a", "cohort_id": COHORT}
    lookup = dash_client.post("/api/llm-decisions/criteria", json={**body, "lookup": True}).json()
    assert lookup["result"] is None and lookup["estimate"]["requests"] == 3 and not fake_llm["criteria"]
    started = dash_client.post("/api/llm-decisions/criteria", json=body).json()
    result = wait_job(dash_client, started["job"]["id"])["result"]
    assert [line["fails"] for line in result["lines"]] == [False, False, True]
    assert result["counts"]["clearly_fail"] == 1 and result["lines"][0]["output"] == '{"answer": "yes"}'
    assert result["lines"][0]["reasoning"] == "Fabricated." and result["scorer"] == "llm"
    assert fake_llm["criteria"][0]["general_exclusion_criteria_evidence"] == "ECOG 1. Creatinine 0.9 mg/dL."
    cached = dash_client.post("/api/llm-decisions/criteria", json=body).json()
    assert cached["result"]["cached"] and len(fake_llm["criteria"]) == 1
    status = dash_client.get("/api/llm").json()["decisions"]
    assert status["available"] and status["cached"] == 1 and status["scales"]["match"]["range"] == [0, 6]


def test_guideline_spaces_and_populations_are_ranked_by_the_llm(fake_llm):
    from matchminer_ai.apps.dashboard.llm import Connection
    from matchminer_ai.apps.dashboard.llm_decisions import LLMDecisions

    decisions, connection = LLMDecisions(), Connection.make("openai", TEACHER, NEMOTRON)
    spaces = [{"space_trial_id": "S1", "name": "Breast", "rank": 1, "similarity_score": 0.9, "match_quality_score": 0.8,
               "match_quality_pass": True, "cohort_key_criteria": "Breast cancer."},
              {"space_trial_id": "S2", "name": "EGFR lung", "rank": 2, "similarity_score": 0.7,
               "match_quality_score": 0.6, "match_quality_pass": True, "cohort_key_criteria": "EGFR-mutant lung cancer."},
              {"space_trial_id": "S3", "name": "Garbled", "rank": 3, "similarity_score": 0.95,
               "match_quality_score": 0.5, "match_quality_pass": True, "cohort_key_criteria": "garbled"}]
    ranked = decisions.rank_guideline_spaces("session", {"spaces": spaces, "summary": "EGFR lung adenocarcinoma.",
                                                         "report": "# Package report"}, 2, connection)
    assert [s["space_trial_id"] for s in ranked["spaces"]] == ["S2", "S1"]  # by the LLM's score; no valid answer last
    assert ranked["spaces"][0]["rank"] == 1 and ranked["spaces"][0]["decision_model_rank"] == 2
    assert "match_quality_score" not in ranked["spaces"][0] and ranked["spaces"][0]["llm_match"]["score"] == 2
    assert ranked["ranking"]["scorer"] == "llm" and ranked["report"].endswith("# Package report")
    assert "decision model's, which this ranking does not use" in ranked["report"]

    selected = pd.DataFrame({"patient_id": ["p1", "p2", "p3"], "cancer_history_summary": ["Breast.", "Lung EGFR.", "EGFR lung."],
                             "similarity_score": [0.9, 0.5, 0.8], "retrieval_rank": [1, 3, 2],
                             "match_quality_score": [0.9, 0.1, 0.5], "match_quality_reasonableness": [0.9, 0.1, 0.5],
                             "match_quality_pass": [True, False, True], "rank": [1, 3, 2]})
    space = {"space_trial_id": "S2", "cohort_key_criteria": "EGFR-mutant lung cancer."}
    chosen, metadata = decisions.select_guideline_patients("session", selected, {"space": space, "candidates": []},
                                                           connection)
    assert chosen["patient_id"].tolist() == ["p3", "p2"]  # reasonable only; same score, so by similarity
    assert "match_quality_score" not in chosen and chosen["llm_match_score"].tolist() == [2, 2]
    assert metadata["selected_count"] == 2 and metadata["scorer"]["kind"] == "llm"
    assert [c["patient_id"] for c in metadata["candidates"]] == ["p3", "p2", "p1"]
    assert "cancer_history_summary" not in metadata["candidates"][0]
