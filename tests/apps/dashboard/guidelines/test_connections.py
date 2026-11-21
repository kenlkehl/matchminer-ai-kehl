"""Model connections: endpoint rules, per-session connections that a submitted job keeps, and model discovery that
saves nothing and sends no notes."""
import threading

import httpx
import pandas as pd
import pytest

from matchminer_ai.apps.dashboard.guidelines.services import Inference
from matchminer_ai.apps.dashboard.guidelines.settings import normalize_endpoint

from .conftest import catalog, peer, settings_for, wait


@pytest.mark.parametrize("address,expected", [
    ("gpu-server:8060", "http://gpu-server:8060/v1"),
    (" http://localhost:8000/ ", "http://localhost:8000/v1"),
    ("https://example.test/prefix/v1/", "https://example.test/prefix/v1"),
    ("http://[::1]:8000", "http://[::1]:8000/v1"),
])
def test_endpoint_normalization(address, expected):
    assert normalize_endpoint(address) == expected


@pytest.mark.parametrize("address", [
    "", "file:///etc/passwd", "http://", "http://host:99999",
    "http://user:secret@host/v1", "http://host/v1?key=secret",
    "http://host/#fragment", "http://some host/v1",
])
def test_invalid_endpoint_rejected(address):
    with pytest.raises(ValueError):
        normalize_endpoint(address)


def test_settings_are_session_scoped_and_jobs_keep_their_snapshot(tmp_path, monkeypatch, make_client):
    import matchminer_ai.patients

    entered, release = threading.Event(), threading.Event()
    configs = []

    def summarize(frame, *, config):
        configs.append(config)
        entered.set()
        assert release.wait(5)
        return pd.DataFrame([{"cancer_history_summary": "Synthetic summary", "general_exclusion_criteria_evidence": ""}])

    monkeypatch.setattr(matchminer_ai.patients, "summarize_patients", summarize)
    a = make_client(settings_for(tmp_path))
    b = peer(a)
    assert a.post("/api/models", json={"endpoint": "gpu-server:8060", "model": "first-model",
                                       "tokenizer": "matching-tokenizer"}).status_code == 200
    assert a.get("/api/models").json()["model"] == "first-model"
    assert b.get("/api/models").json()["model"] == "default-model"
    try:
        first = a.post("/api/patient/summarize", json={"text": "Fabricated note"}).json()["job"]["id"]
        assert entered.wait(5)
        a.post("/api/models", json={"endpoint": "other:8000", "model": "second-model"})
        second = b.post("/api/patient/summarize", json={"text": "Other fabricated note"}).json()["job"]["id"]
    finally:
        release.set()
    assert wait(a, first)["status"] == "done"
    assert wait(b, second)["status"] == "done"
    assert configs[0].remote["server_urls"] == ["http://gpu-server:8060/v1"]
    assert configs[0].patient["remote"]["model_name"] == "first-model"
    assert configs[0].patient["remote"]["tokenizer_name"] == "matching-tokenizer"
    assert configs[0].patient["reasoning_effort"] == "xhigh"
    assert configs[0].patient["sampling_profile"] == "auto"
    assert configs[1].remote["server_urls"] == ["http://default:8000/v1"]
    assert a.get("/api/models").json()["tokenizer"] == "second-model"
    assert b.get("/api/jobs/" + first).status_code == 404


def test_workup_uses_saved_connection_with_raw_notes(tmp_path, monkeypatch, make_client):
    import matchminer_ai.patients

    catalog(tmp_path)
    captured = []
    monkeypatch.setattr(Inference, "retrieve", lambda *args: {
        "summary": "Synthetic summary", "spaces": [{"clinical_space_summary": "Population",
                                                    "diagnostic_workup": [{"name": "Synthetic workup"}]}],
    })

    def review(notes, recommendations, **kwargs):
        captured.append((notes, recommendations, kwargs["config"]))
        return {"assessments": []}

    monkeypatch.setattr(matchminer_ai.patients, "review_patient_workup", review)
    client = make_client(settings_for(tmp_path))
    client.post("/api/models", json={"endpoint": "gpu-server:8060", "model": "served-model"})
    key = client.post("/api/guidelines/retrieve", json={"summary": "Synthetic summary"}).json()["job"]["id"]
    wait(client, key)
    job = client.post("/api/guidelines/workup", json={"text": "Raw fabricated notes", "retrieval_job_id": key,
                                                      "space_index": 0}).json()["job"]["id"]
    assert wait(client, job)["status"] == "done"
    assert captured[0][0] == "Raw fabricated notes"
    assert captured[0][2].remote["server_urls"] == ["http://gpu-server:8060/v1"]
    remote = captured[0][2].patient["remote"]
    assert remote["model_name"] == "served-model"
    assert remote["tokenizer_name"] == "served-model"


def mock_models(monkeypatch, handle):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))


@pytest.mark.parametrize("payload,expected", [
    ({"data": [{"id": "model-a"}, {"id": "model-b"}, {"id": "model-a"}]}, ["model-a", "model-b"]),
    ({"data": []}, []),
    ({"data": [{"id": None}, "bad", {"id": ""}]}, []),
    ({"data": None}, []),
])
def test_discovery_uses_entered_endpoint_without_saving_or_sending_notes(tmp_path, monkeypatch, make_client,
                                                                       payload, expected):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=payload)

    mock_models(monkeypatch, handle)
    client = make_client(settings_for(tmp_path))
    result = client.post("/api/models/check", json={"endpoint": "gpu-server:8060", "model": "model-b"}).json()
    assert result["models"] == expected
    assert result["model"] == ("model-b" if expected else None)
    assert result["available"] == bool(expected)
    assert client.get("/api/models").json()["model"] == "default-model"
    if expected:
        assert client.post("/api/models/check", json={"endpoint": "gpu-server:8060"}).json()["model"] == "model-a"
    assert all(str(r.url) == "http://gpu-server:8060/v1/models" for r in calls)
    assert all(r.method == "GET" and not r.content for r in calls)


def test_discovery_failure_does_not_echo_provider_details(tmp_path, monkeypatch, make_client):
    def fail(request):
        raise httpx.ConnectError("INTERNAL PROVIDER DETAIL", request=request)

    mock_models(monkeypatch, fail)
    client = make_client(settings_for(tmp_path))
    response = client.post("/api/models/check", json={})
    assert response.json()["models"] == []
    assert "INTERNAL PROVIDER DETAIL" not in response.text


def test_connection_rejects_invalid_fields(tmp_path, make_client):
    client = make_client(settings_for(tmp_path))
    body = {"endpoint": "gpu-server:8060", "model": "new-model"}
    assert client.post("/api/models", json={**body, "model": "   "}).status_code == 422
    assert client.post("/api/models", json={**body, "api_key": "secret"}).status_code == 422
    assert client.post("/api/models", json={**body, "endpoint": "file:///tmp"}).status_code == 422
    assert client.get("/api/models").json()["model"] == "default-model"
