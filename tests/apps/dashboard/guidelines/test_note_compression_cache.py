"""A session's saved note compression: reuse with the same record and connection, invalidation, privacy and
expiry, and its use by summaries and both raw-note reviews."""

from copy import deepcopy

import pandas as pd
import pytest

from matchminer_ai.apps.dashboard.guidelines.note_compressions import NoteCompressions
from matchminer_ai.apps.dashboard.guidelines.services import Inference
from matchminer_ai.apps.dashboard.guidelines.settings import Settings

from .conftest import SESSION_COOKIE, SOURCE, WorkupInference, catalog, job


@pytest.fixture
def compressed_app(tmp_path, monkeypatch, make_client, request):
    """A client whose compression, summary and reviews are faked; synthetic mode unless parametrized (real: True)."""
    from matchminer_ai import patients

    catalog(tmp_path)
    settings = Settings(
        phi=getattr(request, "param", False),
        catalog_root=tmp_path,
        llm_backend="local",
        device="cpu",
        endpoint="http://fabricated.invalid/v1",
        model="fabricated-model",
    )
    calls = {"compression": [], "summary": [], "review": []}

    def compress(frame, **kwargs):
        calls["compression"].append(frame.to_dict("records"))
        kwargs["progress_callback"](len(frame), len(frame))
        return frame.assign(
            compressed_note_text=[f"Compressed {n + 1}." for n in range(len(frame))]
        )

    def summarize(frame, **kwargs):
        calls["summary"].append(frame.note_text.tolist())
        return pd.DataFrame([{"cancer_history_summary": "Fabricated summary"}])

    def review(frame, items, **kwargs):
        calls["review"].append(
            frame.note_text.tolist() if isinstance(frame, pd.DataFrame) else frame
        )
        return {
            "assessments": [],
            "notice": "Fabricated review.",
            "metadata": {"note_count": len(frame), "requests": 1},
        }

    monkeypatch.setattr(patients, "compress_patient_notes", compress)
    monkeypatch.setattr(patients, "summarize_patients", summarize)
    monkeypatch.setattr(patients, "review_patient_workup", review)
    monkeypatch.setattr(patients, "review_patient_workup_with_note_search", review)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)
    return make_client(settings), calls, settings


@pytest.mark.parametrize("dated", [False, True])
def test_standalone_compression_survives_retrieval_and_is_optional_for_both_reviews(
    compressed_app, dated
):
    client, calls, _ = compressed_app
    source = SOURCE if dated else {"text": "Computed tomography completed.", "notes": None}
    compressed = ["Compressed 1.", "Compressed 2."] if dated else ["Compressed 1."]
    result = job(client, "/api/patient/compress", source)["result"]
    assert result["compression"]["reused"] is False
    assert calls["summary"] == [] and calls["review"] == []
    job(client, "/api/patient/summarize", {**source, "note_source": "compressed"})
    saved = client.get("/api/patient/note-compression").json()["note_compression"]
    assert saved["notes_input"] == source
    assert [row["compressed_note_text"] for row in saved["notes"]] == compressed
    assert saved["model"] == "fabricated-model"
    if dated:
        assert [
            {k: v for k, v in row.items() if k != "compressed_note_text"}
            for row in saved["notes"]
        ] == SOURCE["notes"]
    retrieval = job(client, "/api/guidelines/retrieve", {"summary": "Edited fabricated summary"})["id"]
    assert client.get("/api/patient/session").json()["note_compression"] == saved
    for method in ("full_notes", "agentic"):
        payload = {**source, "retrieval_job_id": retrieval, "space_index": 0, "method": method}
        reused = job(client, "/api/guidelines/workup", {**payload, "note_source": "compressed"})["result"]
        assert reused["review"]["metadata"]["compression"]["reused"] is True
        assert calls["review"][-1] == compressed
        assert reused["notes_input"] == source
        raw = job(client, "/api/guidelines/workup", {**payload, "note_source": "original"})["result"]
        assert "compression" not in raw["review"]["metadata"]
        assert calls["review"][-1] == (
            [row["note_text"] for row in SOURCE["notes"]] if dated else source["text"]
        )
        assert client.get("/api/patient/note-compression").json()["note_compression"] == saved
    repeated = job(client, "/api/patient/summarize", {**source, "note_source": "compressed"})["result"]
    assert repeated["compression"]["reused"] is True
    assert calls["summary"] == [compressed, compressed]
    assert len(calls["compression"]) == 1


@pytest.mark.parametrize("changed", ["text", "note_text", "note_date", "note_type", "order"])
def test_changed_patient_record_invalidates_and_recompresses(compressed_app, changed):
    client, calls, _ = compressed_app
    source = deepcopy(SOURCE)
    if changed == "text":
        source["text"] += " Another record."
    elif changed == "order":
        source["notes"].reverse()
    else:
        source["notes"][0][changed] += "-changed" if changed != "note_date" else "T12:00:00"
    job(client, "/api/patient/compress", SOURCE)
    failed = job(
        client, "/api/patient/summarize", {**source, "note_source": "compressed"}, status="error"
    )
    assert "Run Compress notes first" in failed["error"]
    assert calls["summary"] == [] and len(calls["compression"]) == 1
    job(client, "/api/patient/summarize", {**source, "note_source": "original"})
    assert client.get("/api/patient/note-compression").json()["note_compression"] is None
    fresh = job(client, "/api/patient/compress", source)["result"]
    assert fresh["compression"]["reused"] is False
    assert len(calls["compression"]) == 2
    assert client.get("/api/patient/session").json()["note_compression"]["notes_input"] == source


@pytest.mark.parametrize("compressed_app", [False, True], ids=["synthetic", "real"], indirect=True)
def test_cache_is_session_private_and_clear_and_connection_changes_invalidate(compressed_app):
    client, calls, _ = compressed_app
    job(client, "/api/patient/compress", SOURCE)
    cookie = client.cookies.get(SESSION_COOKIE)
    assert cookie
    client.cookies.clear()
    assert client.get("/api/patient/note-compression").json()["note_compression"] is None
    assert client.delete("/api/patient/note-compression").status_code == 204
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE, cookie)
    assert client.get("/api/patient/note-compression").json()["note_compression"] is not None
    assert (
        client.post(
            "/api/models",
            json={
                "llm_backend": "local",
                "endpoint": "http://fabricated.invalid/v1",
                "model": "new-model",
            },
        ).status_code
        == 200
    )
    assert client.get("/api/patient/note-compression").json()["note_compression"] is None
    job(client, "/api/patient/compress", SOURCE)
    assert (
        client.get("/api/patient/note-compression").json()["note_compression"]["model"] == "new-model"
    )
    assert client.delete("/api/patient/session").status_code == 204
    assert client.get("/api/patient/session").json() == {"job": None, "note_compression": None}
    job(client, "/api/patient/compress", SOURCE)
    assert len(calls["compression"]) == 3


def test_complete_compression_is_available_after_downstream_failure(compressed_app, monkeypatch):
    from matchminer_ai import patients

    api, calls, _ = compressed_app

    def fail(*args, **kwargs):
        raise RuntimeError("Fabricated downstream failure")

    monkeypatch.setattr(patients, "summarize_patients", fail)
    client = api
    job(client, "/api/patient/compress", SOURCE)
    job(client, "/api/patient/summarize", {**SOURCE, "note_source": "compressed"}, status="error")
    assert client.get("/api/patient/session").json()["note_compression"] is not None
    retrieval = job(client, "/api/guidelines/retrieve", {"summary": "Fabricated summary"})["id"]
    job(
        client,
        "/api/guidelines/workup",
        {
            **SOURCE,
            "note_source": "compressed",
            "retrieval_job_id": retrieval,
            "space_index": 0,
            "method": "agentic",
        },
    )
    assert len(calls["compression"]) == 1


def test_failed_compression_does_not_cache_partial_text(compressed_app, monkeypatch):
    from matchminer_ai import patients

    api, calls, _ = compressed_app

    def fail(*args, **kwargs):
        raise RuntimeError("Fabricated compression failure")

    monkeypatch.setattr(patients, "compress_patient_notes", fail)
    client = api
    job(client, "/api/patient/compress", SOURCE, status="error")
    assert client.get("/api/patient/session").json()["note_compression"] is None
    assert calls["summary"] == []


@pytest.mark.parametrize("workflow", ["summarize", "full_notes", "agentic"])
@pytest.mark.parametrize("stale", [False, True])
def test_unavailable_compressed_notes_never_start_compression_or_clinical_inference(
    compressed_app, workflow, stale
):
    client, calls, _ = compressed_app
    if stale:
        job(client, "/api/patient/compress", SOURCE)
        client.delete("/api/patient/note-compression")
    before = deepcopy(calls)
    payload = {**SOURCE, "note_source": "compressed"}
    route = "/api/patient/summarize"
    if workflow != "summarize":
        retrieval = job(client, "/api/guidelines/retrieve", {"summary": "Fabricated summary"})["id"]
        payload.update(retrieval_job_id=retrieval, space_index=0, method=workflow)
        route = "/api/guidelines/workup"
    failed = job(client, route, payload, status="error")
    assert "Run Compress notes first" in failed["error"]
    assert calls == before
    assert client.get("/api/patient/note-compression").json()["note_compression"] is None


def test_compression_job_preserves_current_summary_and_owned_retrieval(compressed_app):
    client, calls, _ = compressed_app
    retrieval = job(client, "/api/guidelines/retrieve", {"summary": "Fabricated summary"})
    payload = {
        **SOURCE,
        "summary": "Edited fabricated summary",
        "retrieval_job_id": retrieval["id"],
    }
    compression = job(client, "/api/patient/compress", payload)
    assert compression["kind"] == "compress"
    assert compression["restore"]["summary"] == payload["summary"]
    assert compression["result"]["retrieval"] == retrieval["result"]
    assert compression["result"]["retrieval_job_id"] == retrieval["id"]
    assert client.get("/api/patient/session").json()["job"] == compression
    assert calls["summary"] == [] and calls["review"] == []
    again = job(client, "/api/patient/compress", payload)
    assert again["result"]["compression"]["reused"] is True
    assert len(calls["compression"]) == 1
    client.cookies.clear()
    # Another browser session cannot reach this session's search: it compresses without it (as after an expired
    # search) and gets none of its results.
    other = client.post("/api/patient/compress", json=payload)
    assert other.status_code == 202 and "retrieval" not in (other.json()["job"]["restore"] or {})


def test_old_implicit_compression_argument_is_rejected(compressed_app):
    client, calls, _ = compressed_app
    assert (
        client.post("/api/patient/summarize", json={**SOURCE, "compress_notes": True}).status_code
        == 422
    )
    assert (
        client.post("/api/patient/summarize", json={**SOURCE, "note_source": "unknown"}).status_code
        == 422
    )
    assert calls == {"compression": [], "summary": [], "review": []}


def test_independent_compression_keeps_matching_summary_provenance_and_workup(compressed_app):
    client, _, _ = compressed_app
    summary = job(client, "/api/patient/summarize", SOURCE)["result"]
    compression = job(client, "/api/patient/compress", {**SOURCE, "summary": summary["summary"]})
    assert compression["restore"]["summary_result"] == summary
    retrieval = job(client, "/api/guidelines/retrieve", {"summary": summary["summary"]})["id"]
    workup = job(
        client,
        "/api/guidelines/workup",
        {
            **SOURCE,
            "retrieval_job_id": retrieval,
            "space_index": 0,
            "note_source": "compressed",
        },
    )["result"]
    compression = job(
        client,
        "/api/patient/compress",
        {
            **SOURCE,
            "summary": summary["summary"],
            "retrieval_job_id": retrieval,
            "note_source": "original",
        },
    )
    assert compression["restore"]["workup_result"] == workup
    assert compression["restore"]["note_source"] == "original"


def test_cache_expiry_and_late_producer_cannot_restore_cleared_patient(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("matchminer_ai.apps.dashboard.guidelines.note_compressions.time.monotonic", lambda: clock[0])
    config = Settings(device="cpu").inference_config()
    store = NoteCompressions(ttl=10)
    bound = store.bind("owner")
    bound.save(**SOURCE, config=config, compressed_notes=[], metadata={})
    assert store.latest("owner") is not None
    clock[0] += 11
    assert store.latest("owner") is None
    old = store.bind("owner")
    store.clear("owner")
    new = store.bind("owner")
    old.save(**SOURCE, config=config, compressed_notes=[], metadata={})
    assert new.lookup(**SOURCE, config=config) is None
    assert store.latest("owner") is None


def test_configuration_identity_and_snapshot_copies(compressed_app):
    _, _, settings = compressed_app
    config = settings.inference_config()
    store = NoteCompressions()
    bound = store.bind("owner")
    rows = [{**SOURCE["notes"][0], "compressed_note_text": "CT done."}]
    bound.save(**SOURCE, config=config, compressed_notes=rows, metadata={})
    rows[0]["compressed_note_text"] = "Changed outside cache"
    snapshot = bound.lookup(**SOURCE, config=config)
    assert snapshot["notes"][0]["compressed_note_text"] == "CT done."
    snapshot["notes"][0]["compressed_note_text"] = "Changed returned copy"
    assert store.latest("owner")["notes"][0]["compressed_note_text"] == "CT done."
    config.remote["server_urls"] = ["http://different.invalid/v1"]
    assert bound.lookup(**SOURCE, config=config) is None
    assert store.latest("owner") is None
