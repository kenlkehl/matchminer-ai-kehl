"""Data and models' local model server, in every dashboard: its routes, a session's use of it, the default switch
and ``--local-server-only``, with a fake engine (no GPU, model download or network)."""
from __future__ import annotations

import sys
import time

import pytest

from types import SimpleNamespace

from matchminer_ai.apps.dashboard.llm import Connection
from matchminer_ai.llm import connections

from .conftest import SESSION_COOKIE, peer

FAKE_ENGINE = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
name, port = args[args.index("--alias") + 1], int(args[args.index("--port") + 1])
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": name}]}).encode())
    def log_message(self, *a):
        pass
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''
SPEC = {"engine": "llama_cpp", "model": "example/fake-GGUF", "served_name": "example/fake", "gpus": []}
HAIKU = "example_gateway.team_project.team_project-claude-haiku-5-5"
GATEWAY = {"llm_backend": "claude_databricks", "endpoint": "https://gateway.invalid", "model": HAIKU}


@pytest.fixture
def gateway(site_config):
    """The site config names a gateway endpoint for Haiku, so Claude would otherwise be every task's default."""
    site_config.write_text(site_config.read_text() + f"  endpoints:\n    haiku: {HAIKU}\n")


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    """llama.cpp is "installed" as the fake engine; vLLM is not."""
    script = tmp_path / "fake_engine.py"
    script.write_text(FAKE_ENGINE)
    found = {"vllm": {"label": "vLLM", "available": False, "version": None, "command": None, "reason": "not here"},
             "llama_cpp": {"label": "llama.cpp", "available": True, "version": "test",
                           "command": [sys.executable, str(script)], "reason": None}}
    monkeypatch.setattr("matchminer_ai.llm.local_server.engines", lambda llama_server=None: found)


def server_client(make_client, **settings):
    client = make_client(**settings)
    control = client.app.state.mmai.local_server
    control.download, control.discover = False, lambda: []
    return client


def started(client):
    response = client.post("/api/local-server/start", json={"spec": SPEC})
    assert response.status_code == 200, response.text
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        server = client.get("/api/local-server").json()["server"]
        if server["state"] != "starting":
            assert server["state"] == "ready", server
            return server
        time.sleep(0.05)
    pytest.fail("the fake server did not become ready")


def test_every_dashboard_has_the_local_server(make_client):
    client = make_client()  # no flag
    assert client.get("/api/local-server").json()["only"] is False
    assert client.get("/api/local-server/engines").status_code == 200
    assert client.get("/api/local-server/gpus").status_code == 200
    assert client.get("/api/models").json()["local_server"]["only"] is False


def test_start_refuses_a_second_server_and_a_session_uses_it(make_client, fake_engine):
    client = server_client(make_client)
    server = started(client)
    assert server["endpoint"].startswith("http://127.0.0.1:")
    assert client.post("/api/local-server/start", json={"spec": SPEC}).status_code == 409
    view = client.post("/api/local-server/use", json={"apply_to_all": True}).json()
    assert {(t["llm_backend"], t["endpoint"], t["model"]) for t in view["tasks"].values()} == {
        ("local", server["endpoint"], "example/fake")}
    assert view["local_server"]["state"] == "ready"
    # Another session keeps the app's default until it chooses the server too.
    assert peer(client).get("/api/models").json()["endpoint"] != server["endpoint"]
    stopped = client.post("/api/local-server/stop").json()["server"]
    assert stopped["state"] == "stopped"
    assert client.post("/api/local-server/use", json={"apply_to_all": True}).status_code == 400


def test_the_default_switch_rebinds_the_worker_and_is_undone_on_stop(make_client, fake_engine):
    client = server_client(make_client, local_server_default=True)
    g = client.app.state.mmai.guidelines
    before = g.settings.endpoint
    server = started(client)
    # Sessions that chose nothing get the server for every task, through the default worker bound to it.
    assert g.settings.endpoint == server["endpoint"] and g.settings.task_connections == {}
    assert g.inference.settings is g.settings
    assert g.patient_inference("someone", "summarize") is g.inference
    assert peer(client).get("/api/models").json()["tasks"]["workup_sql"]["endpoint"] == server["endpoint"]
    client.post("/api/local-server/stop")
    assert g.settings.endpoint == before and g.inference.settings is g.settings


def test_only_mode_has_no_hosted_default_before_the_server_is_ready(make_client, gateway):
    assert make_client().get("/api/models").json()["llm_backend"] == "claude_databricks"  # the site's default
    client = make_client(local_server_only=True)
    view = client.get("/api/models").json()
    assert {(t["llm_backend"], t["endpoint"]) for t in view["tasks"].values()} == {("local", "")}
    assert view["claude_profiles"] == {} and view["vertex_profiles"] == {} and view["local_server"]["only"]
    response = client.post("/api/patient/summarize", json={"text": "Fabricated note."})
    assert response.status_code == 400 and "local model server" in response.json()["error"]
    assert not client.app.state.mmai.llm.connection.configured


def test_only_mode_refuses_other_connections_on_update_and_on_use(make_client, fake_engine, gateway):
    client = server_client(make_client, local_server_only=True)
    refused = client.post("/api/models", json={**GATEWAY, "task": "summarize"})
    assert refused.status_code == 400 and "--local-server-only" in refused.json()["error"]
    server = started(client)
    other_loopback = {"llm_backend": "local", "endpoint": "http://127.0.0.1:9/v1", "model": "example/fake"}
    for body in (GATEWAY, other_loopback):
        assert client.post("/api/models", json={**body, "task": "summarize", "apply_to_all": True}).status_code == 400
        assert client.post("/api/models/check", json=body).status_code == 400
    # The server itself is accepted, and is every task's default.
    mine = {"llm_backend": "local", "endpoint": server["endpoint"], "model": "example/fake", "task": "assistant"}
    assert client.post("/api/models", json=mine).status_code == 200
    assert client.post("/api/local-server/use", json={"apply_to_all": True}).status_code == 200
    # On use: a connection saved past the checks (or for a server since restarted elsewhere) is not used.
    g, owner = client.app.state.mmai.guidelines, client.cookies.get(SESSION_COOKIE)
    g.workspace_settings[owner]["values"]["task_connections"]["llm_decisions"] = dict(GATEWAY)
    assert Connection.for_task(g.task_settings(owner, "llm_decisions")).endpoint == server["endpoint"]
    # The package's builders refuse every other destination too, wherever the app reaches them.
    with pytest.raises(connections.ConnectionRefused):
        connections.connect_remote(SimpleNamespace(remote={}), connections.Connection("claude_databricks", HAIKU))
    with pytest.raises(connections.ConnectionRefused):
        connections.claude_connection("haiku")
    connections.connect_remote(SimpleNamespace(remote={}), g.settings.llm_connection)  # the server itself
