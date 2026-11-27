"""OpenRouter on Data and models: a session's own key stays in the app's memory and never comes back, real mode never
offers OpenRouter (the routes and the package's connection builders refuse it), and with --local-server-only it is
accepted only with --allow-openrouter. No request reaches OpenRouter."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from matchminer_ai.apps.dashboard.llm import Connection
from matchminer_ai.llm import connections, openrouter

from .conftest import SESSION_COOKIE, peer

KEY = "sk-or-v1-" + "fedcba9876543210" * 4  # fabricated
MODEL = "nvidia/nemotron-3.5-super-vl"
ROUTED = {"llm_backend": "openrouter", "endpoint": openrouter.ENDPOINT, "model": MODEL}
ELSEWHERE = {"llm_backend": "local", "endpoint": "http://127.0.0.1:9/v1", "model": "example/fake"}


def test_the_key_stays_in_memory_and_never_comes_back(make_client, caplog):
    caplog.set_level(logging.DEBUG)
    client = make_client()
    assert client.get("/api/models").json()["openrouter"]["key_set"] is False
    refused = client.post("/api/models/openrouter-key", json={"key": "not a key"})
    assert refused.status_code == 400 and "not a key" not in refused.text
    answers = [client.post("/api/models/openrouter-key", json={"key": KEY}),
               client.post("/api/models", json={**ROUTED, "task": "summarize", "apply_to_all": True}),
               client.get("/api/models"), client.get("/api/llm")]
    assert all(a.status_code == 200 and KEY not in a.text for a in answers)
    assert answers[0].json()["openrouter"]["key_hint"] == "…" + KEY[-4:]
    assert not peer(client).get("/api/models").json()["openrouter"]["key_set"]  # another session has none
    g, owner = client.app.state.mmai.guidelines, client.cookies.get(SESSION_COOKIE)
    selected = g.task_settings(owner, "llm_decisions")
    connection = Connection.for_task(selected)
    assert connection.label == f"{MODEL} (OpenRouter)" and connection.credentials_error() is None
    assert KEY not in repr(selected) + repr(connection) + str(connection.describe())
    assert openrouter.api_key(selected.api_key_ref) == KEY
    config = connection.configure(client.app.state.mmai.decisions.config, ("llm_match_quality",))
    assert config.remote["provider"] == "openrouter" and KEY not in str(config.remote)
    assert client.delete("/api/models/openrouter-key").status_code == 200
    assert not openrouter.has_key(selected.api_key_ref)
    assert Connection.for_task(g.task_settings(owner, "llm_decisions")).credentials_error() == openrouter.NO_KEY
    assert KEY not in caplog.text


def test_real_mode_never_offers_openrouter(make_client):
    client = make_client(mode="real", access_token="t" * 40)
    assert client.get("/api/models").json()["openrouter"] == {"offered": False}
    for response in (client.post("/api/models/openrouter-key", json={"key": KEY}),
                     client.post("/api/models", json={**ROUTED, "task": "summarize"}),
                     client.post("/api/models/check", json={**ROUTED, "task": "summarize"})):
        assert response.status_code == 400 and "synthetic mode only" in response.json()["error"]
    with pytest.raises(connections.ConnectionRefused):  # the package's builders, wherever the app reaches them
        connections.connect_remote(SimpleNamespace(remote={}), connections.Connection("openrouter", MODEL))
    with pytest.raises(ValueError, match="synthetic mode only"):
        make_client(mode="real", access_token="t" * 40, allow_openrouter=True)


def test_local_server_only_accepts_openrouter_only_when_allowed(make_client):
    client = make_client(local_server_only=True)
    assert client.get("/api/models").json()["openrouter"] == {"offered": False}
    assert client.post("/api/models", json={**ROUTED, "task": "summarize"}).status_code == 400
    client = make_client(local_server_only=True, allow_openrouter=True)
    assert client.get("/api/models").json()["openrouter"]["offered"]
    assert client.post("/api/models/openrouter-key", json={"key": KEY}).status_code == 200
    assert client.post("/api/models", json={**ROUTED, "task": "summarize", "apply_to_all": True}).status_code == 200
    assert client.post("/api/models", json={**ELSEWHERE, "task": "summarize"}).status_code == 400
    connections.connect_remote(SimpleNamespace(remote={}), connections.Connection("openrouter", MODEL))
    with pytest.raises(connections.ConnectionRefused):
        connections.connect_remote(SimpleNamespace(remote={}), connections.Connection("local", "m", ELSEWHERE["endpoint"]))


def test_with_no_local_server_openrouter_is_the_only_connection_and_the_default(make_client):
    """The small-GPU Space: --local-server-only, --allow-openrouter and no local server. Every task defaults to
    OpenRouter's default model, a session without a key is told to paste one, and nothing else is accepted."""
    client = make_client(local_server_only=True, allow_openrouter=True, local_server_enabled=False)
    status = client.get("/api/local-server").json()
    assert status["enabled"] is False and "OpenRouter" in status["disabled_reason"]
    assert client.post("/api/local-server/start", json={"spec": {"engine": "llama_cpp", "model": "m"}}).status_code == 400
    view = client.get("/api/models").json()
    default = openrouter.presets()["default"]
    assert {(t["llm_backend"], t["model"]) for t in view["tasks"].values()} == {("openrouter", default)}
    assert view["local_server"]["enabled"] is False and view["openrouter"]["offered"]
    response = client.post("/api/patient/summarize", json={"text": "Fabricated note."})
    assert response.status_code == 400 and response.json()["error"] == openrouter.NO_KEY
    for body in (ELSEWHERE, {**ELSEWHERE, "endpoint": "http://127.0.0.1:8000/v1"}):
        assert client.post("/api/models", json={**body, "task": "summarize"}).status_code == 400
    assert client.post("/api/models/openrouter-key", json={"key": KEY}).status_code == 200
    g, owner = client.app.state.mmai.guidelines, client.cookies.get(SESSION_COOKIE)
    assert Connection.for_task(g.task_settings(owner, "llm_decisions")).credentials_error() is None
    with pytest.raises(connections.ConnectionRefused):
        connections.connect_remote(SimpleNamespace(remote={}), connections.Connection("local", "m", ELSEWHERE["endpoint"]))
    with pytest.raises(ValueError, match="allow-openrouter"):
        make_client(local_server_only=True, local_server_enabled=False)


HOSTED_KEY = "sk-or-v1-" + "0" * 60 + "Zq9X"  # fabricated
LITE = {"local_server_only": True, "allow_openrouter": True, "local_server_enabled": False}


def test_the_hosted_key_serves_sessions_without_their_own(make_client, monkeypatch, caplog):
    """The lite Space: the hosted block without the secret offers nothing; with it, the hosted model is every task's
    default for sessions without a key and the only model they may choose; the key leaves the environment and never
    comes back; a session's own key serves instead, for any model. Real mode refuses the block."""
    import os

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(openrouter, "_fetch_key_record", lambda ref, timeout=10.0: (
        200, {"data": {"free_model_daily_requests": {"used": 100, "limit": 1000, "remaining": 900}}}))
    monkeypatch.delenv(openrouter.HOSTED_KEY_ENV, raising=False)
    view = make_client(**LITE, hosted_openrouter={}).get("/api/models").json()
    assert view["openrouter"]["hosted"] is None
    assert view["tasks"]["summarize"]["model"] == openrouter.presets()["default"]

    monkeypatch.setenv(openrouter.HOSTED_KEY_ENV, HOSTED_KEY)
    client = make_client(**LITE, hosted_openrouter={})
    assert openrouter.HOSTED_KEY_ENV not in os.environ
    hosted_model = openrouter.HOSTED_MODEL
    answers = [client.get("/api/models"), client.get("/api/llm"), client.get("/api/local-server"),
               client.post("/api/models", json={**ROUTED, "model": hosted_model, "task": "summarize",
                                                "apply_to_all": True})]
    assert all(a.status_code == 200 and HOSTED_KEY[-4:] not in a.text for a in answers)
    view = answers[0].json()
    hosted = view["openrouter"]["hosted"]
    assert hosted["model"] == hosted_model and hosted["label"].startswith("Hosted: ") and not hosted["own_key"]
    assert hosted["today"]["available"] == 880 and hosted["session"] == {"used": 0, "share": 150, "remaining": 150}
    assert view["openrouter"]["key_set"] is False and view["openrouter"]["key_hint"] is None
    assert {(t["llm_backend"], t["model"]) for t in view["tasks"].values()} == {("openrouter", hosted_model)}
    g, owner = client.app.state.mmai.guidelines, client.cookies.get(SESSION_COOKIE)
    selected = g.task_settings(owner, "llm_decisions")
    assert openrouter.is_hosted(selected.api_key_ref) and Connection.for_task(selected).credentials_error() is None
    locked = openrouter.MODEL_LOCKED.format(model=hosted_model)
    refused = client.post("/api/models", json={**ROUTED, "task": "summarize"})  # another model, no own key
    assert refused.status_code == 400 and refused.json()["error"] == locked
    # A session's own key serves instead, for any model.
    assert client.post("/api/models/openrouter-key", json={"key": KEY}).status_code == 200
    saved = client.post("/api/models", json={**ROUTED, "task": "summarize"})
    assert saved.status_code == 200 and saved.json()["openrouter"]["hosted"]["own_key"]
    assert openrouter.api_key(g.task_settings(owner, "summarize").api_key_ref) == KEY
    # Without it again, that connection is refused before any request.
    assert client.delete("/api/models/openrouter-key").status_code == 200
    response = client.post("/api/patient/summarize", json={"text": "Fabricated note."})
    assert response.status_code == 400 and response.json()["error"] == locked
    assert HOSTED_KEY not in caplog.text and HOSTED_KEY not in repr(selected)
    with pytest.raises(ValueError, match="synthetic mode only"):
        make_client(mode="real", access_token="t" * 40, hosted_openrouter={})
