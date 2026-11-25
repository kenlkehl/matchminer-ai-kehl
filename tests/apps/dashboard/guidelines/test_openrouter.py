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
