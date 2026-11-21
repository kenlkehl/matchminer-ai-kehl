"""The Patient tab's notes and guideline search: catalogs, uploads, summaries, sessions and job errors."""
import threading

import pytest

from matchminer_ai.apps.dashboard.guidelines.services import Catalogs, InputError, display_considerations, parse_upload
from matchminer_ai.apps.dashboard.guidelines.settings import Settings

from .conftest import FakeInference, catalog, peer, wait


@pytest.fixture
def setup(tmp_path, make_client):
    catalog(tmp_path)
    inference = FakeInference()
    return make_client(Settings(catalog_root=tmp_path, device="cpu"), inference), inference


def test_only_completed_catalogs_are_exposed(tmp_path):
    catalog(tmp_path, "ready")
    catalog(tmp_path, "partial", False)
    assert [r["disease"] for r in Catalogs(tmp_path).list()] == ["ready"]


def test_category_display_clarifies_absence_without_rewriting_evidence():
    import pandas as pd

    original = pd.DataFrame([{
        "diagnostic_workup": [{"category": "Not specified in supplied evidence"}],
        "treatment_options": [
            {"category": "Not specified in the supplied evidence.",
             "evidence": [{"quote": "Not specified in the supplied evidence"}]},
            {"category": "Category 1"},
        ],
    }])
    displayed = display_considerations(original).iloc[0]
    assert displayed.diagnostic_workup[0]["category"] == "NCCN recommendation category not specified"
    assert displayed.treatment_options[0]["category"] == "NCCN recommendation category not specified"
    assert displayed.treatment_options[1]["category"] == "Category 1"
    assert displayed.treatment_options[0]["evidence"] == original.iloc[0].treatment_options[0]["evidence"]
    assert original.iloc[0].diagnostic_workup[0]["category"] == "Not specified in supplied evidence"


def test_retrieval_defaults_parameters_and_session_isolation(setup):
    a, inference = setup
    b = peer(a)
    # The default search reranks five candidates and returns one space.
    default = a.post("/api/guidelines/retrieve", json={"summary": "Synthetic"}).json()["job"]
    assert wait(a, default["id"])["result"]["search"] == {"catalogs": [], "top_n": 1, "candidate_k": 5}
    assert inference.calls[0][2:] == (1, 5)
    info = a.get("/api/guidelines/catalogs").json()
    assert "path" not in info["catalogs"][0]
    request = a.post("/api/guidelines/retrieve", json={
        "summary": "Fabricated history", "catalogs": [info["catalogs"][0]["id"]], "top_n": 3, "candidate_k": 12})
    assert request.status_code == 202
    key = request.json()["job"]["id"]
    assert wait(a, key)["result"]["summary"] == "Fabricated history"
    assert inference.calls[1][2:] == (3, 12)
    assert b.get(f"/api/jobs/{key}").status_code == 404
    assert b.get("/api/patient/session").json()["job"] is None
    assert a.get("/api/patient/session").json()["job"]["id"] == key
    assert a.get(f"/api/jobs/{key}").headers["cache-control"] == "no-store"
    assert a.delete("/api/patient/session").status_code == 204
    assert a.get(f"/api/jobs/{key}").status_code == 404


def test_rejects_invalid_search_and_unrecognized_catalog(setup):
    client = setup[0]
    assert client.post("/api/guidelines/retrieve", json={"summary": " ", "top_n": 5}).status_code == 422
    assert client.post("/api/guidelines/retrieve",
                       json={"summary": "x", "top_n": 5, "candidate_k": 2}).status_code == 422
    unknown = client.post("/api/guidelines/retrieve", json={"summary": "x", "catalogs": ["../../data"]})
    assert unknown.status_code == 400 and "no longer available" in unknown.json()["error"]


def test_upload_parsing_dates_aliases_types_and_single_patient():
    data = b"patient_id,note_text,note_date\na,Later,2026-02-01\na,Earlier,2026-01-01\na,Undated,\n"
    result = parse_upload("notes.csv", data)
    assert [r["note_text"] for r in result["notes"]] == ["Earlier", "Later", "Undated"]
    assert result["notes"][-1]["note_date"] is None
    assert "1900" not in result["text"]
    for multiple in (b"patient_id,note_text\na,one\nb,two", b"pseudo_mrn,text\na,one\nb,two"):
        with pytest.raises(InputError, match="exactly one"):
            parse_upload("notes.csv", multiple)
    with pytest.raises(InputError, match="invalid"):
        parse_upload("notes.csv", b"note_text,note_date\none,not a date")
    typed = parse_upload("notes.csv", b"text,date,type\nCT done,2026-02-01,Radiology\nCT ordered,2026-01-01,Oncology\n"
                                      b"Undated note,,\n")
    assert typed["notes"] == [
        {"note_text": "CT ordered", "note_date": "2026-01-01T00:00:00", "note_type": "Oncology"},
        {"note_text": "CT done", "note_date": "2026-02-01T00:00:00", "note_type": "Radiology"},
        {"note_text": "Undated note", "note_date": None},
    ]


def test_unreadable_pdf_does_not_silently_skip_pages():
    import io

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument.new()
    document.new_page(100, 100)  # blank: no text layer, nothing for OCR
    buffer = io.BytesIO()
    document.save(buffer)
    with pytest.raises(InputError, match="page 1"):
        parse_upload("scan.pdf", buffer.getvalue())


def test_upload_and_summary_adapter(setup):
    client, inference = setup
    data = client.post("/api/patient/upload",
                       files={"file": ("notes.csv", b"note_text,note_date\nSynthetic note,2026-01-01")}).json()
    response = client.post("/api/patient/summarize", json={"text": data["text"], "notes": data["notes"]})
    assert wait(client, response.json()["job"]["id"])["result"]["summary"] == "Synthetic cancer history"
    assert inference.calls[0][1][0]["note_date"].startswith("2026-01-01")
    assert client.post("/api/patient/upload", files={"file": ("bad.exe", b"invalid")}).status_code == 400


def test_inference_errors_do_not_echo_patient_data(setup):
    client, inference = setup

    def fail(*args):
        raise RuntimeError("CONFIDENTIAL CONTENT")

    inference.summarize = fail
    key = client.post("/api/patient/summarize", json={"text": "Synthetic"}).json()["job"]["id"]
    wait(client, key)
    response = client.get(f"/api/jobs/{key}")
    assert response.json()["status"] == "error" and response.json()["error"].startswith("RuntimeError")
    assert "CONFIDENTIAL CONTENT" not in response.text


def test_busy_jobs_survive_refresh_and_cannot_be_cleared(setup):
    client, inference = setup
    release = threading.Event()

    def slow(*args):
        release.wait(5)
        return {"summary": "Synthetic", "exclusion_evidence": ""}

    inference.summarize = slow
    try:
        key = client.post("/api/patient/summarize", json={"text": "Synthetic"}).json()["job"]["id"]
        assert client.get("/api/patient/session").json()["job"]["id"] == key
        assert client.delete("/api/patient/session").status_code == 409
        assert client.post("/api/patient/summarize", json={"text": "Synthetic"}).status_code == 409
    finally:
        release.set()
    assert wait(client, key)["status"] == "done"
