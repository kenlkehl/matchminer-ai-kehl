"""Each LLM task's own connection: routing, copies to every task, backups and their snapshots, all per session."""
from threading import Event

import httpx
import pandas as pd
import pytest

from matchminer_ai.apps.dashboard.guidelines.services import Inference
from matchminer_ai.apps.dashboard.guidelines.settings import LLM_TASKS, vertex_profiles

from .conftest import SOURCE, WorkupInference, assert_config, catalog, job, peer, selection, settings_for, wait

CONNECTION_FIELDS = ("llm_backend", "endpoint", "model", "tokenizer", "reasoning")


def shared(task="workup", **overrides):
    return {"task": task, "apply_to_all": True, "llm_backend": "local", "endpoint": "shared:8000",
            "model": "Qwen/Qwen3.5-4B", "tokenizer": "matching-tokenizer", "reasoning": True, **overrides}


def connection(task):
    return {key: task[key] for key in CONNECTION_FIELDS}


@pytest.mark.parametrize("thinking", [False, True])
def test_each_task_routes_to_its_endpoint(tmp_path, monkeypatch, make_client, thinking):
    from matchminer_ai import patients

    catalog(tmp_path)
    calls = []
    originals = [n["note_text"] for n in SOURCE["notes"]]

    def summarize(frame, **kwargs):
        assert_config(kwargs["config"], "summarize", thinking)
        assert frame.note_text.tolist() == originals
        calls.append("summarize")
        return pd.DataFrame([{"cancer_history_summary": "Fabricated summary"}])

    def review(frame, items, **kwargs):
        assert_config(kwargs["config"], "workup", thinking)
        assert frame.note_text.tolist() == originals
        if "thinking" in kwargs:
            assert kwargs["thinking"] is None  # inherit the selected task mode
        calls.append("workup")
        return {"assessments": [], "metadata": {}, "notice": "Fabricated review."}

    monkeypatch.setattr(patients, "summarize_patients", summarize)
    monkeypatch.setattr(patients, "review_patient_workup", review)
    monkeypatch.setattr(patients, "review_patient_workup_with_note_search", review)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)
    client = make_client(settings_for(tmp_path))
    for task in ("workup", "summarize"):
        updated = client.post("/api/models", json=selection(task, thinking)).json()
        assert updated["tasks"][task]["reasoning"] is thinking
        assert updated["tasks"][task]["sampling"]["temperature"] == (1.0 if thinking else 0.7)
        if task == "workup":
            assert updated["tasks"]["summarize"]["model"] == "default-model"
    job(client, "/api/patient/summarize", SOURCE)
    retrieval = job(client, "/api/guidelines/retrieve", {"summary": "Fabricated summary"})["id"]
    for method in ("full_notes", "agentic"):
        job(client, "/api/guidelines/workup", {**SOURCE, "method": method, "retrieval_job_id": retrieval,
                                               "space_index": 0})
    assert calls == ["summarize", "workup", "workup"]


def test_task_settings_are_private_and_running_job_keeps_submitted_endpoint_and_thinking(
        tmp_path, monkeypatch, make_client):
    from matchminer_ai import patients

    entered, release = Event(), Event()
    configs = []

    def summarize(frame, **kwargs):
        configs.append(kwargs["config"])
        entered.set()
        assert release.wait(5)
        return pd.DataFrame([{"cancer_history_summary": "Fabricated summary"}])

    monkeypatch.setattr(patients, "summarize_patients", summarize)
    a = make_client(settings_for(tmp_path))
    b = peer(a)
    defaults = b.get("/api/models").json()["tasks"]
    assert set(defaults) == set(LLM_TASKS) and all(task["reasoning"] is False for task in defaults.values())
    a.post("/api/models", json=selection("summarize", True))
    try:
        key = a.post("/api/patient/summarize", json=SOURCE).json()["job"]["id"]
        assert entered.wait(5)
        a.post("/api/models", json={**selection("summarize"), "endpoint": "changed.invalid"})
        a.post("/api/models", json=selection("workup", True))
        # A copy to every task does not redirect the running job either.
        assert a.post("/api/models", json=shared()).status_code == 200
    finally:
        release.set()
    assert wait(a, key)["status"] == "done"
    assert_config(configs[0], "summarize", True)
    assert b.get("/api/models").json()["tasks"] == defaults
    assert a.post("/api/models", json={**selection("summarize"), "task": "invalid"}).status_code == 422


def test_discovery_uses_selected_task_without_modifying_other_tasks(tmp_path, monkeypatch, make_client):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.5-4B"}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    client = make_client(settings_for(tmp_path))
    client.post("/api/models", json=selection("summarize"))
    before = client.get("/api/models").json()
    assert client.post("/api/models/check", json={"task": "summarize"}).json()["available"] is True
    assert client.get("/api/models").json() == before
    assert len(seen) == 1 and str(seen[0].url) == "http://summarize.invalid/v1/models"
    assert seen[0].method == "GET" and not seen[0].content


# ---------------------------------------------------------------------------------------------- every task at once


@pytest.mark.parametrize("source", list(LLM_TASKS))
def test_all_tasks_copy_source_connection_and_later_edits_remain_independent(tmp_path, make_client, source):
    client = make_client(settings_for(tmp_path))
    other = peer(client)
    defaults = other.get("/api/models").json()
    for task in LLM_TASKS:
        values = {**selection(task), "model": f"previous-{task}", "backup_enabled": task == "workup_full_backup",
                  "failure_threshold": 7 if task == "workup_agent_backup" else 4}
        assert client.post("/api/models", json=values).status_code == 200
    prior = client.get("/api/models").json()["tasks"]
    response = client.post("/api/models", json=shared(source))
    assert response.status_code == 200
    applied = response.json()
    assert applied["model"] == "Qwen/Qwen3.5-4B"
    expected = {"llm_backend": "local", "endpoint": "http://shared:8000/v1", "model": "Qwen/Qwen3.5-4B",
                "tokenizer": "matching-tokenizer", "reasoning": True}
    for task, saved in applied["tasks"].items():
        assert connection(saved) == expected
        assert saved["sampling"]["temperature"] == 1.0 and saved["sampling"]["top_p"] == 0.95
        if task != source:
            assert saved["backup_enabled"] == prior[task]["backup_enabled"]
            assert saved["failure_threshold"] == prior[task]["failure_threshold"]
    assert client.get("/api/models").json() == applied
    assert other.get("/api/models").json() == defaults
    edited = client.post("/api/models", json={**selection("summarize"), "model": "one-task-model"}).json()["tasks"]
    assert edited["summarize"]["model"] == "one-task-model"
    assert all(edited[t] == applied["tasks"][t] for t in LLM_TASKS if t != "summarize")


def test_copy_from_backup_saves_only_its_own_backup_controls(tmp_path, make_client):
    client = make_client(settings_for(tmp_path))
    client.post("/api/models", json={**selection("workup_full_backup"), "backup_enabled": True,
                                     "failure_threshold": 4})
    result = client.post("/api/models", json=shared("workup_agent_backup", backup_enabled=True,
                                                    failure_threshold=9)).json()["tasks"]
    assert result["workup_agent_backup"]["backup_enabled"]
    assert result["workup_agent_backup"]["failure_threshold"] == 9
    assert result["workup_full_backup"]["backup_enabled"]
    assert result["workup_full_backup"]["failure_threshold"] == 4
    assert not any(result[t]["backup_enabled"] for t in ("summarize", "workup"))


@pytest.mark.parametrize("provider", ["gemini4_vertex", "gemini_vertex", "gemma_vertex"])
def test_bulk_cloud_selection_uses_server_owned_destination(tmp_path, make_client, provider):
    client = make_client(settings_for(tmp_path))
    response = client.post("/api/models", json=shared(llm_backend=provider, endpoint="http://untrusted.invalid/v1",
                                                      model="untrusted"))
    assert response.status_code == 200
    profile = vertex_profiles()[provider]
    for saved in response.json()["tasks"].values():
        assert all(saved[k] == profile[k] for k in ("endpoint", "model", "tokenizer"))


@pytest.mark.parametrize("overrides", [
    {"task": None},
    {"endpoint": "file:///tmp/model"},
    {"failure_threshold": 0},
    {"task": "workup", "apply_to_all": False, "backup_enabled": True},
    {"task": "workup_full_backup", "apply_to_all": False, "failure_threshold": 0},
    {"task": "workup_agent_backup", "apply_to_all": False, "failure_threshold": 11},
])
def test_invalid_connection_request_changes_no_task(tmp_path, make_client, overrides):
    client = make_client(settings_for(tmp_path))
    before = client.get("/api/models").json()
    assert client.post("/api/models", json=shared(**overrides)).status_code == 422
    assert client.get("/api/models").json() == before


# ---------------------------------------------------------------------------------------------- backups


@pytest.mark.parametrize("method,task", [("full_notes", "workup_full_backup"), ("agentic", "workup_agent_backup")])
def test_only_selected_method_receives_enabled_backup(tmp_path, monkeypatch, make_client, method, task):
    from matchminer_ai import patients

    catalog(tmp_path)
    calls = []

    def review(frame, items, **kwargs):
        calls.append(kwargs)
        assert_config(kwargs["config"], "workup", False)
        assert frame.note_text.tolist() == [n["note_text"] for n in SOURCE["notes"]]
        return {"assessments": [], "metadata": {}, "notice": "Fabricated review."}

    monkeypatch.setattr(patients, "review_patient_workup", review)
    monkeypatch.setattr(patients, "review_patient_workup_with_note_search", review)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)
    client = make_client(settings_for(tmp_path))
    initial = client.get("/api/models").json()["tasks"]
    assert not initial["workup_full_backup"]["backup_enabled"]
    assert not initial["workup_agent_backup"]["backup_enabled"]
    assert initial[task]["failure_threshold"] == 3
    client.post("/api/models", json=selection("workup"))
    body = {**selection(task, True), "backup_enabled": True, "failure_threshold": 4}
    assert client.post("/api/models", json=body).status_code == 200
    retrieval = job(client, "/api/guidelines/retrieve", {"summary": "Fabricated summary"})["id"]
    for chosen in ("full_notes", "agentic"):
        job(client, "/api/guidelines/workup", {**SOURCE, "method": chosen, "retrieval_job_id": retrieval,
                                               "space_index": 0})
        options = calls[-1]
        if chosen == method:
            assert_config(options["backup_config"], task, True)
            assert options["max_consecutive_failures"] == 4
            assert options["backup_config"].remote["request_timeout"] == 7200
        else:
            assert "backup_config" not in options
    client.post("/api/models", json={**body, "backup_enabled": False})
    job(client, "/api/guidelines/workup", {**SOURCE, "method": method, "retrieval_job_id": retrieval,
                                           "space_index": 0})
    assert "backup_config" not in calls[-1]


def test_backup_snapshot_and_session_isolation(tmp_path, monkeypatch, make_client):
    from matchminer_ai import patients

    catalog(tmp_path)
    entered, release = Event(), Event()
    captured = []

    def review(frame, items, **kwargs):
        captured.append(kwargs)
        entered.set()
        assert release.wait(5)
        return {"assessments": [], "metadata": {}, "notice": "Fabricated review."}

    monkeypatch.setattr(patients, "review_patient_workup", review)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)
    a = make_client(settings_for(tmp_path))
    b = peer(a)
    body = {**selection("workup_full_backup"), "backup_enabled": True}
    a.post("/api/models", json=body)
    assert b.get("/api/models").json()["tasks"]["workup_full_backup"]["backup_enabled"] is False
    retrieval = job(a, "/api/guidelines/retrieve", {"summary": "Fabricated summary"})["id"]
    try:
        started = a.post("/api/guidelines/workup", json={**SOURCE, "retrieval_job_id": retrieval,
                                                         "space_index": 0}).json()["job"]["id"]
        assert entered.wait(5)
        a.post("/api/models", json={**body, "endpoint": "http://changed.invalid/v1", "model": "new-model"})
    finally:
        release.set()
    assert wait(a, started)["status"] == "done"
    assert captured[0]["backup_config"].remote["server_urls"] == ["http://workup_full_backup.invalid/v1"]
