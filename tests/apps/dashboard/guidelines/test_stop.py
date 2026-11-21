"""Stopping a guideline job: only its own session can, a refreshed page sees it stopped, and an agentic review's
note REPLs all close with its inputs kept."""

import json
from threading import Event, Lock

import pytest

from matchminer_ai.apps.dashboard.guidelines.services import Inference
from matchminer_ai.apps.dashboard.guidelines.settings import Settings

from .conftest import FakeInference, catalog, peer, wait


def stopped(client, key):
    found = wait(client, key, timeout=3)
    assert found["status"] == "cancelled", "Stop did not release the inference worker"
    return found


def test_stop_is_session_owned_idempotent_and_refreshable(tmp_path, make_client):
    from matchminer_ai.cancellation import cancel_sleep

    catalog(tmp_path)
    inference = FakeInference()
    started = Event()

    def slow(*args):
        started.set()
        cancel_sleep(30)
        pytest.fail("Cancelled summarizer continued")

    inference.summarize = slow
    client = make_client(Settings(catalog_root=tmp_path, device="cpu"), inference)
    other = peer(client)
    key = client.post("/api/patient/summarize", json={"text": "Synthetic notes"}).json()["job"]["id"]
    assert started.wait(1)
    assert client.get("/api/patient/session").json()["job"]["id"] == key
    assert other.post(f"/api/jobs/{key}/stop").status_code == 404
    assert client.post(f"/api/jobs/{key}/stop").status_code == 200
    assert stopped(client, key)["result"] is None
    assert client.post(f"/api/jobs/{key}/stop").json()["status"] == "cancelled"
    assert client.get("/api/patient/session").json()["job"]["status"] == "cancelled"
    inference.summarize = FakeInference().summarize
    next_key = client.post("/api/patient/summarize", json={"text": "Synthetic notes"}).json()["job"]["id"]
    assert wait(client, next_key)["status"] == "done"


def test_agentic_stop_closes_all_question_repls_and_preserves_inputs(tmp_path, monkeypatch, make_client):
    from matchminer_ai.llm.structured import StructuredClient
    from matchminer_ai.patients import _note_repl

    catalog(tmp_path)
    settings = Settings(
        llm_backend="local",
        catalog_root=tmp_path,
        device="cpu",
        endpoint="http://synthetic.invalid/v1",
        model="synthetic",
    )
    original_config = Settings.inference_config

    def config(self):
        runtime = original_config(self)
        runtime.patient.update(tokenizer_mode="bytes", context_window=32768)
        runtime.patient["remote"].setdefault("request_params", {})["max_tokens"] = 1024
        return runtime

    monkeypatch.setattr(Settings, "inference_config", config)
    retrieval = {
        "summary": "Synthetic summary",
        "spaces": [
            {
                "name": "Synthetic population",
                "diagnostic_workup": [{"name": name} for name in ["CT", "ECG", "LDH"]],
            }
        ],
    }
    monkeypatch.setattr(Inference, "retrieve", lambda *args: retrieval)
    all_waiting, release = Event(), Event()
    lock, processes, calls = Lock(), [], []
    original_popen = _note_repl.subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(_note_repl.subprocess, "Popen", popen)

    def http(self, endpoint, body=None, **kwargs):
        payload = json.loads(body["messages"][1]["content"].rsplit("Request:\n", 1)[-1])
        if "last_cell_result" not in payload:
            value = {"terms": ["scan"]}
        elif payload["search_only"]:
            value = dict(
                action="python",
                code=["notes"],
                memory="",
                status="unknown",
                answer={},
                limitations=[],
                needs_review=False,
            )
        else:
            with lock:
                calls.append(1)
                if len(calls) == 3:
                    all_waiting.set()
            assert release.wait(10)
            value = {}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(StructuredClient, "_http_request", http)
    notes = [
        {"note_text": "Fabricated CT ordered.", "note_date": "2026-01-01", "note_type": "Oncology"}
    ]
    client = make_client(settings)
    key = client.post("/api/guidelines/retrieve", json={"summary": "Synthetic summary"}).json()["job"]["id"]
    wait(client, key)
    key2 = client.post(
        "/api/guidelines/workup",
        json={
            "text": "Fabricated CT ordered.",
            "notes": notes,
            "retrieval_job_id": key,
            "space_index": 0,
            "method": "agentic",
        },
    ).json()["job"]["id"]
    try:
        assert all_waiting.wait(6)
        assert client.post(f"/api/jobs/{key2}/stop").status_code == 200
        job = stopped(client, key2)
        assert job["result"] is None
        assert job["restore"]["notes_input"]["notes"] == notes
        assert job["restore"]["retrieval"] == retrieval
        assert all(p.poll() is not None for p in processes)
        assert len(calls) == 3
        assert client.get("/api/patient/session").json()["job"]["status"] == "cancelled"
    finally:
        release.set()
