"""Data and models' local model server (``--local-server``): its routes, a session's use of it and the default
switch, with a fake engine (no GPU, model download or network)."""
from __future__ import annotations

import sys
import time

import pytest

from .conftest import peer

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
    client = make_client(local_server=True, **settings)
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


def test_without_the_flag_there_is_no_local_server(make_client):
    client = make_client()
    assert client.get("/api/local-server").status_code == 404
    assert client.post("/api/local-server/start", json={"spec": SPEC}).status_code in (404, 405)  # no such route
    assert client.get("/api/models").json()["local_server"] == {"enabled": False}


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
