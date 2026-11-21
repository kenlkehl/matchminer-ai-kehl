"""The dashboard's guideline features (``guidelines/``, ``routes/guidelines.py``, ``routes/population.py``) over the
fabricated run, trials and site config of ``tests/apps/dashboard``. Fabricated notes, records and tokens only: no
network, GPU or real-model call."""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from ..conftest import FakeEncoder
from ..test_dash import fake_ncit, trial_data  # noqa: F401  (fixtures)

SESSION_COOKIE = "mmai_dash_session"
BASE_URL = "http://127.0.0.1"  # a loopback Host, accepted in both modes
# The Databricks gateway is a site value: a placeholder workspace.
DATABRICKS_HOST = "https://workspace.example.cloud.databricks.com"


@pytest.fixture(autouse=True)
def site_config(site_config):
    """The dashboard tests' fabricated site config plus a placeholder Databricks workspace."""
    site_config.write_text(site_config.read_text() + f"databricks:\n  host: {DATABRICKS_HOST}\n")
    return site_config


@pytest.fixture(autouse=True)
def vertex_credentials(monkeypatch):
    """This server's Google credentials, faked as present (a Vertex check never asks Google for a token)."""
    monkeypatch.setattr("matchminer_ai.llm.connections.vertex_credentials_error", lambda max_age=60.0: None)


def catalog(root, name="fictional", complete=True):
    """A finalized guideline catalog's marker files (its spaces come from a faked loader or inference)."""
    path = root / "run" / name
    path.mkdir(parents=True)
    (path / "validation.json").write_text(json.dumps({"status": "passed", "paradigms": 2}))
    (path / "status.json").write_text(
        json.dumps({"status": "complete" if complete else "running", "updated_utc": "2026-09-01"}))
    (path / "sources.json").write_text(json.dumps({"disease": name, "title": name.title(), "version": "1.2026"}))
    (path / "paradigms.jsonl").write_text("{}")
    return path


def peer(client):
    """Another browser session (its own session cookie) on the same app."""
    return TestClient(client.app, base_url=str(client.base_url))


def wait(client, job_id, timeout=15.0):
    """The job once it is done, failed or stopped."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200, response.text
        found = response.json()
        if found["status"] in {"done", "error", "cancelled"}:
            return found
        time.sleep(0.02)
    pytest.fail("Job did not finish")


def job(client, route, body, status="done", timeout=15.0):
    """Submit a job and wait for it to end with ``status``."""
    response = client.post(route, json=body)
    assert response.status_code == 202, response.text
    done = wait(client, response.json()["job"]["id"], timeout)
    assert done["status"] == status, done
    return done


@pytest.fixture
def make_client(dataset, trial_data, fake_model, fake_ncit, tmp_path):  # noqa: F811
    """``make_client(settings=None, inference=None, access_token=None, mode=None)``: a client of the whole app.

    ``settings`` are the guideline features' settings (``guidelines.settings.Settings``); the app's mode follows
    them (``settings.mode``), else ``mode`` (default synthetic), and without them the app derives its own.
    ``inference`` replaces the guideline inference adapter. With ``access_token`` the client is signed in.
    """
    from matchminer_ai.apps.dashboard.server import create_app
    from matchminer_ai.apps.dashboard.settings import DashSettings
    from matchminer_ai.apps.dashboard.state import DashState

    states, clients = [], []

    def make(settings=None, inference=None, access_token=None, mode=None):
        mode = settings.mode if settings is not None else (mode or "synthetic")
        dash = DashSettings(mode=mode, synthetic_run=dataset["root"], store_dir=tmp_path / f"store-{len(states)}",
                            trial_data=trial_data["root"], oa_db=trial_data["db"],
                            good_options_catalog=trial_data["catalog"], embedding_caches=(tmp_path / "hf",),
                            embedding_python=tmp_path / "missing-python", device="cpu")
        state = DashState(dash, encoder=FakeEncoder(), guideline_settings=settings, guideline_inference=inference,
                          patient_source="training", log=lambda *_: None)
        states.append(state)
        client = TestClient(create_app(state, access_token=access_token), base_url=BASE_URL)
        if access_token:
            client.headers["Authorization"] = f"Bearer {access_token}"
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()
    for state in states:
        state.registry.shutdown()


# ---------------------------------------------------------------------------------------------- shared fakes

# A fabricated record of one patient: dated, typed and undated notes.
SOURCE = {
    "text": "Fabricated longitudinal record",
    "notes": [
        {"note_text": "Computed tomography completed.", "note_date": "2026-01-01", "note_type": "Radiology"},
        {"note_text": "No fever documented.", "note_date": None},
    ],
}


class FakeInference:
    """Stands in for ``guidelines.services.Inference``: records its calls."""

    def __init__(self):
        self.calls = []

    def retrieve(self, summary, paths, top_n, candidate_k, progress):
        self.calls.append((summary, paths, top_n, candidate_k))
        progress("Scoring with the decision model")
        return {"spaces": [], "summary": summary, "report": "Synthetic report"}

    def summarize(self, text, notes, progress):
        self.calls.append((text, notes))
        return {"summary": "Synthetic cancer history", "exclusion_evidence": "Unknown"}


class WorkupInference(FakeInference):
    """One guideline space with one workup item."""

    def retrieve(self, summary, paths, top_n, candidate_k, progress):
        return {"summary": summary, "spaces": [{"name": "Synthetic space",
                "diagnostic_workup": [{"name": "Imaging", "conditions": "If indicated"}]}]}

    def review_workup(self, text, notes, space, progress, *, method="full_notes", patient_summary=None):
        self.calls.append((text, notes, space, method))
        progress("Reviewing raw-note chunk 1/1")
        return {"assessments": [], "notice": "Synthetic review"}


def settings_for(tmp_path, **values):
    """Synthetic-mode settings with a local default connection and the catalogs under ``tmp_path``."""
    from matchminer_ai.apps.dashboard.guidelines.settings import Settings

    return Settings(**{"llm_backend": "local", "catalog_root": tmp_path, "device": "cpu",
                       "endpoint": "http://default:8000/v1", "model": "default-model",
                       "tokenizer": "default-tokenizer", **values})


def selection(task, thinking=False):
    """A task's own connection, as Data and Models posts it."""
    return {"task": task, "endpoint": f"http://{task}.invalid/v1", "model": "Qwen/Qwen3.5-4B", "reasoning": thinking}


def assert_config(config, task, thinking):
    """The inference configuration a job received names the task's endpoint and Qwen's sampling."""
    from matchminer_ai.llm.backends import build_llm_runtime_config
    from matchminer_ai.llm.remote_inference import build_remote_request_config

    assert config.remote["server_urls"] == [f"http://{task}.invalid/v1"]
    runtime = build_llm_runtime_config("patient", config.patient, config=config)
    params, extra = build_remote_request_config(runtime)
    assert params["temperature"] == (1.0 if thinking else 0.7)
    assert params["top_p"] == (0.95 if thinking else 0.8)
    assert params["presence_penalty"] == 1.5
    assert extra["chat_template_kwargs"] == {"enable_thinking": thinking}
    assert extra["top_k"] == 20 and extra["min_p"] == 0.0
    assert extra["repetition_penalty"] == 1.0
    assert "reasoning_effort" not in params and "reasoning_effort" not in extra
