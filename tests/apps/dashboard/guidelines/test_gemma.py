"""Selectable Gemma MaaS uses a fixed Vertex destination and shared inference defaults."""
import pytest

from matchminer_ai.apps.dashboard.guidelines.services import Inference
from matchminer_ai.apps.dashboard.guidelines.settings import LLM_TASKS, Settings, vertex_profiles, vertex_project

from .conftest import catalog, peer, settings_for, wait


def gemma():
    return vertex_profiles()["gemma_vertex"]


def selection():
    return {"llm_backend": "gemma_vertex", "endpoint": "http://untrusted.test/v1", "model": "untrusted-model",
            "tokenizer": "untrusted-tokenizer"}


def assert_gemma(config):
    from matchminer_ai.llm.backends import build_llm_runtime_config
    from matchminer_ai.llm.structured import resolve_structured_config

    runtime = build_llm_runtime_config("patient", config.patient, config=config)
    resolved, _ = resolve_structured_config(runtime, cache_dir=None)
    assert resolved.provider == "google_agent_platform"
    assert resolved.google_project_id == vertex_project() == "example-project"
    assert resolved.base_url == gemma()["endpoint"]
    assert resolved.model == gemma()["model"]
    assert resolved.context_window == 262144 and resolved.max_tokens == 65536
    assert resolved.tokenizer_mode == "bytes"
    assert resolved.thinking == "off" and resolved.top_k == 64
    assert resolved.temperature == 1.0 and resolved.top_p == 0.95
    assert "reasoning_effort" not in resolved.request_params
    assert resolved.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert resolved.request_start_interval_seconds == 2.0
    assert resolved.capacity_retry_initial_seconds == 15.0
    assert resolved.capacity_retry_max_seconds == 60.0


def test_gemma_selection_is_fixed_and_session_scoped(tmp_path, make_client):
    a = make_client(settings_for(tmp_path))
    b = peer(a)
    result = a.post("/api/models", json=selection())
    assert result.status_code == 200
    assert all(result.json()[key] == value for key, value in gemma().items())
    assert a.get("/api/models").json()["llm_backend"] == "gemma_vertex"
    assert b.get("/api/models").json()["llm_backend"] == "local"
    assert result.json()["vertex_profiles"]["gemma_vertex"]["model"] == gemma()["model"]


@pytest.mark.parametrize("method,api_name", [
    ("full_notes", "review_patient_workup"),
    ("agentic", "review_patient_workup_with_note_search"),
])
def test_gemma_workup_uses_selected_model(tmp_path, monkeypatch, make_client, method, api_name):
    import matchminer_ai.patients

    catalog(tmp_path)
    configs = []
    monkeypatch.setattr(Inference, "retrieve", lambda *args: {
        "summary": "Synthetic summary",
        "spaces": [{"clinical_space_summary": "Population", "diagnostic_workup": [{"name": "Synthetic workup"}]}],
    })

    def review(notes, recommendations, **kwargs):
        assert notes == "Raw fabricated notes"
        configs.append(kwargs["config"])
        return {"assessments": []}

    monkeypatch.setattr(matchminer_ai.patients, api_name, review)
    client = make_client(settings_for(tmp_path))
    client.post("/api/models", json=selection())
    key = client.post("/api/guidelines/retrieve", json={"summary": "Synthetic summary"}).json()["job"]["id"]
    wait(client, key)
    key = client.post("/api/guidelines/workup", json={"text": "Raw fabricated notes", "retrieval_job_id": key,
                                                      "space_index": 0, "method": method}).json()["job"]["id"]
    assert wait(client, key)["status"] == "done"
    assert_gemma(configs[0])


def test_gemma_connection_ignores_browser_destination(tmp_path, monkeypatch, make_client):
    calls = []
    monkeypatch.setattr("matchminer_ai.llm.vllm_server.check_openai_endpoint",
                        lambda endpoint, **kwargs: calls.append((endpoint, kwargs)) or [(endpoint, 200, "OK")])
    client = make_client(settings_for(tmp_path))
    result = client.post("/api/models/check", json={"llm_backend": "gemma_vertex", "endpoint": "http://untrusted.test",
                                                    "model": "bad"}).json()
    assert result["available"] and result["models"] == [gemma()["model"]]
    assert client.get("/api/models").json()["llm_backend"] == "local"
    assert calls == [(gemma()["endpoint"], {"provider": "google_agent_platform", "google_project_id": "example-project",
                                            "model_name": gemma()["model"], "timeout": 30})]


def test_gemma_default_applies_to_every_task_and_matches_display(tmp_path, make_client):
    # Gemma was the default until 2026-11-14; Gemini 4 Hydrogen is now (see test_gemini).
    settings = Settings(device="cpu", catalog_root=tmp_path, llm_backend="gemma_vertex")
    for task in LLM_TASKS:
        selected = settings.for_task(task)
        assert_gemma(selected.inference_config())
        assert selected.backup_enabled is False
    displayed = make_client(settings).get("/api/models").json()
    assert set(displayed["tasks"]) == set(LLM_TASKS)
    for connection in (displayed, *displayed["tasks"].values()):
        assert connection["llm_backend"] == "gemma_vertex"
        assert all(connection[key] == value for key, value in gemma().items())
        assert connection["reasoning"] is False
        assert connection["backup_enabled"] is False
