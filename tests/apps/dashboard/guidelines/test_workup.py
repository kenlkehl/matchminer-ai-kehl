"""Workup reviews for one patient: the session's own guideline search and raw notes, synthetic example notes and the
inference adapters."""
import pytest

from matchminer_ai.apps.dashboard.guidelines.services import Inference
from matchminer_ai.apps.dashboard.guidelines.settings import Settings

from .conftest import SESSION_COOKIE, FakeInference, WorkupInference, catalog, wait


@pytest.mark.parametrize("phi", [False, True], ids=["synthetic", "real"])
@pytest.mark.parametrize("method", [None, "full_notes", "agentic"])
def test_workup_uses_owned_retrieval_and_raw_notes(tmp_path, make_client, method, phi):
    catalog(tmp_path)
    inference = WorkupInference()
    client = make_client(Settings(phi=phi, catalog_root=tmp_path, device="cpu"), inference)
    key = client.post("/api/guidelines/retrieve", json={"summary": "Synthetic summary"}).json()["job"]["id"]
    wait(client, key)
    body = dict(text="Raw synthetic notes", notes=[dict(note_text="Imaging planned", note_date="2026-01-01")],
                retrieval_job_id=key, space_index=0)
    if method:
        body["method"] = method
    response = client.post("/api/guidelines/workup", json=body)
    assert response.status_code == 202
    done = wait(client, response.json()["job"]["id"])
    assert done["status"] == "done"
    assert inference.calls[0][0] == body["text"]
    assert inference.calls[0][1] == body["notes"]
    assert inference.calls[0][2]["diagnostic_workup"][0]["name"] == "Imaging"
    assert inference.calls[0][3] == (method or "full_notes")
    assert done["result"]["method"] == (method or "full_notes")
    assert client.get("/api/patient/session").json()["job"]["result"]["method"] == (method or "full_notes")
    for invalid in ("invalid", "colbert"):  # ColBERT retrieval was removed
        assert client.post("/api/guidelines/workup", json={**body, "method": invalid}).status_code == 422
    assert done["result"]["notes_input"] == {"text": body["text"], "notes": body["notes"]}
    assert client.post("/api/guidelines/workup", json={**body, "space_index": 1}).status_code == 400
    # Another browser session cannot use this session's search.
    cookie = client.cookies.get(SESSION_COOKIE)
    assert cookie
    client.cookies.clear()
    assert client.post("/api/guidelines/workup", json=body).status_code == 400
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE, cookie)
    assert client.get(f"/api/jobs/{key}").status_code == 200
    client.delete("/api/patient/session")
    assert client.get(f"/api/jobs/{key}").status_code == 404


def test_synthetic_notes_route_supports_gradio_csv_and_dates(tmp_path, make_client):
    sample = tmp_path / "synthetic.csv"
    sample.write_text("pseudo_mrn,date,text\na,2026-02-01,Later synthetic note\na,2026-01-01,Earlier synthetic note\n")
    client = make_client(Settings(catalog_root=tmp_path, device="cpu", synthetic_notes=str(sample)), FakeInference())
    result = client.get("/api/patient/synthetic-notes")
    assert result.status_code == 200
    notes = result.json()["notes"]
    assert notes[0]["note_text"] == "Earlier synthetic note"
    assert notes[0]["note_date"].startswith("2026-01-01")
    assert "Synthetic patient" in result.json()["notice"]


def test_biliary_sample_keeps_raw_notes_and_dates_separate_from_demo(tmp_path, make_client):
    import csv

    sample = tmp_path / "biliary.csv"
    raw = 'Synthetic biopsy report.\nResult: "adenocarcinoma", liver specimen.'
    with sample.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "note_date", "note_text"])
        writer.writerow(["biliary", "2026-02-01", raw])
        writer.writerow(["biliary", "2026-01-01", "Synthetic imaging report."])
    demo = tmp_path / "demo.csv"
    demo.write_text("patient_id,note_text\ndemo,Separate synthetic demo\n")
    settings = Settings(catalog_root=tmp_path, device="cpu", synthetic_notes=str(demo), biliary_synthetic_notes=sample)
    client = make_client(settings, FakeInference())
    sources = client.get("/api/models").json()["synthetic_note_sources"]
    assert [item["id"] for item in sources] == ["biliary", "demo"]
    assert all(set(item) == {"id", "label"} for item in sources)
    response = client.get("/api/patient/synthetic-notes", params={"source": "biliary"})
    assert response.status_code == 200
    result = response.json()
    assert result["source"] == "biliary"
    assert "cholangiocarcinoma" in result["notice"]
    assert result["notes"] == [
        {"note_text": "Synthetic imaging report.", "note_date": "2026-01-01T00:00:00"},
        {"note_text": raw, "note_date": "2026-02-01T00:00:00"},
    ]
    assert raw in result["text"]
    assert client.get("/api/patient/synthetic-notes").json()["notes"] == [
        {"note_text": "Separate synthetic demo", "note_date": None},
    ]


def test_unavailable_or_unknown_synthetic_source_is_rejected(tmp_path, make_client):
    settings = Settings(catalog_root=tmp_path, device="cpu", biliary_synthetic_notes=tmp_path / "missing.csv")
    client = make_client(settings, FakeInference())
    sources = client.get("/api/models").json()["synthetic_note_sources"]
    assert [item["id"] for item in sources] == ["demo"]
    for source in ["biliary", "unknown", "../../notes.csv"]:
        response = client.get("/api/patient/synthetic-notes", params={"source": source})
        assert response.status_code == 400
        assert "unavailable" in response.json()["error"]


def test_workup_adapter_sends_no_summary(monkeypatch):
    import matchminer_ai.patients

    captured = []
    monkeypatch.setattr(matchminer_ai.patients, "review_patient_workup",
                        lambda notes, recommendations, **kw: captured.append((notes, recommendations)))
    settings = Settings(device="cpu")
    monkeypatch.setattr(settings, "inference_config", lambda: None)
    Inference(settings).review_workup("Raw notes only", None,
                                      {"diagnostic_workup": [{"name": "Scan"}], "summary": "MUST NOT ENTER"},
                                      lambda _: None)
    assert captured == [("Raw notes only", [{"name": "Scan"}])]


def test_agentic_adapter_uses_selected_config_and_dated_notes(monkeypatch):
    import matchminer_ai.patients

    captured = []
    monkeypatch.setattr(matchminer_ai.patients, "review_patient_workup_with_note_search",
                        lambda notes, recommendations, **kw: captured.append((notes, recommendations, kw)))
    settings = Settings(llm_backend="local", endpoint="http://selected.invalid/v1", model="Qwen/Qwen3.8-27B-FP8",
                        device="cpu")
    notes = [{"note_text": "Fabricated imaging completed", "note_date": "2026-01-01"}]
    items = [{"name": f"Scan {i}"} for i in range(6)]
    Inference(settings).review_workup(
        "Not the dated source", notes,
        {"diagnostic_workup": items, "clinical_space_summary": "One-line population",
         "cohort_key_criteria": "Population", "summary": "MUST NOT ENTER"}, lambda _: None, method="agentic")
    frame, recommendations, options = captured[0]
    assert frame.to_dict(orient="records") == notes
    assert recommendations == items
    assert options["population_context"] == "Population"
    assert options["config"].remote["server_urls"] == ["http://selected.invalid/v1"]
    assert options["config"].remote["max_concurrent_requests"] == 6
    assert settings.inference_config().remote["max_concurrent_requests"] == 1
