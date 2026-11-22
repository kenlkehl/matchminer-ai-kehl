"""The Patient tab's structured records and criteria screening routes, over fabricated notes and records only."""
from __future__ import annotations

import copy

from matchminer_ai.matching import structured_criteria
from matchminer_ai.matching.structured_criteria import CRITERION_EXAMPLES

from .conftest import job

ECOG = CRITERION_EXAMPLES[0][2]
NOTES = {"text": "Fabricated record", "notes": [
    {"note_text": "ECOG 3 today.", "note_date": "2026-01-05", "note_type": "Clinic"},
    {"note_text": "Follow-up visit.", "note_date": "2026-01-20", "note_type": "Clinic"}]}


def fabricated_extraction(frame, levels, config, progress):
    from matchminer_ai.patients.extraction.notes import note_ids

    first = note_ids(frame)[0]
    item = {"item_type": "performance_status", "scale": "ecog", "score": 3,
            "evidence": [{"note_id": first, "note_date": "2026-01-05", "quote": "ECOG 3", "start": 0, "end": 6}]}
    items = {key: [] for key in ("imaging_studies", "procedures", "diagnoses", "biomarkers", "treatments",
                                 "disease_assessments", "performance_status", "lab_tests", "counseling")}
    items["performance_status"] = [item]
    items["disease_assessments"] = [{"item_type": "disease_assessment", "disease_status": "progression",
                                     "disease_sites": ["liver"], "site_groups": ["liver"], "cancer_ids": [],
                                     "evidence": [{"note_id": first, "note_date": "2026-01-05", "quote": "ECOG",
                                                   "start": 0, "end": 4}]}]
    record = {"schema_version": "mmai-clinical-extraction-v2", "level": "document", "patient_id": "workspace-patient",
              "note_id": first, "note_date": "2026-01-05", "note_type": "Clinic", "document_type": "clinic_note",
              "document_type_text": "", "extensions": [], "items": items}
    return {"documents": [{"patient_id": "workspace-patient", "note_id": first, "note_date": "2026-01-05",
                           "note_type": "Clinic", "document_type": "clinic_note", "extraction_status": "complete",
                           "error": None, "item_count": 1, "record": record}],
            "patients": None, "metadata": {"levels": ["document"], "requests": 2, "counts": {"complete": 2},
                                           "schema_version": "mmai-clinical-extraction-v2", "model": "fabricated"}}


def test_records_extract_view_and_criteria_screen(make_client, monkeypatch):
    client = make_client()
    assert client.get("/api/records/status").json()["store"]["available"] is False
    body = {**NOTES, "levels": "document"}
    before = client.post("/api/patient/records", json=body).json()
    assert before["source"] is None and before["estimate"]["document_requests"] == 2
    monkeypatch.setattr("matchminer_ai.apps.dashboard.guidelines.records.extract_on_demand", fabricated_extraction)
    done = job(client, "/api/patient/records/extract", body)["result"]
    assert done["source"]["kind"] == "session" and done["view"]["counts"]["document_items"] == 2
    course = done["view"]["disease_course"]
    tags = {r["type"]: (r["tag"], r["source"], r["cancer_ids"]) for r in course["rows"]}
    assert course["cancers"] == [] and tags == {"disease_assessment": ("progression", "note", []),
                                               "performance_status": ("ECOG 3", "note", [])}
    assert done["view"]["documents"][0]["record"]["items"]["disease_assessments"]
    item = next(i for i in done["view"]["documents"][0]["items"] if i["type"] == "performance_status")
    assert item["label"].startswith("ECOG") and item["evidence"][0]["note_number"] == 1 and item["raw"]["score"] == 3
    # The session's extraction is the patient's records now (and the Guidelines review's).
    assert client.post("/api/patient/records", json=body).json()["source"]["kind"] == "session"

    def compile_fake(lines, *, cohort_context, config, cache, return_metadata, progress_callback):
        entries = [{"line": {k: line.get(k) for k in ("number", "text", "criterion_type", "boilerplate")},
                    "status": "compiled" if structured_criteria.checkable(line) else "unchecked",
                    "query": copy.deepcopy(ECOG) if structured_criteria.checkable(line) else None,
                    "cache_key": "k"} for line in lines]
        return entries, {"model": "fabricated", "requests": 0, "cached": 0, "compiled": len(entries)}

    monkeypatch.setattr(structured_criteria, "compile_criteria_queries", compile_fake)
    cohorts = client.app.state.mmai.store["cohorts"]
    cohort_id = next(cid for cid, c in cohorts.items()
                     if any(structured_criteria.checkable(line) for line in c.get("items") or []))
    result = job(client, "/api/criteria-records", {**body, "cohort_id": cohort_id})["result"]
    checked = [line for line in result["lines"] if line["status"] != "query_unavailable"]
    assert checked and all(line["clearly_fails"] for line in checked)
    assert checked[0]["evidence"][0]["note_number"] == 1 and result["source"]["kind"] == "session"
    # Another session has no records of these notes, and Find patients needs a store.
    other = client.__class__(client.app, base_url=str(client.base_url))
    assert other.post("/api/criteria-records", json={**body, "cohort_id": cohort_id}).status_code == 400
    response = client.post(f"/api/cohorts/{cohort_id}/screen-records", json={"patient_ids": ["patient_a"]})
    assert response.status_code == 400 and "store" in response.json()["error"]


def test_patient_list_marks_and_filters_stored_patients(make_client, monkeypatch):
    client = make_client()
    monkeypatch.setattr(client.app.state.mmai.guidelines.records, "patient_ids", lambda: frozenset({"patient_a"}))
    every = client.get("/api/patients").json()
    assert every["total"] > 1 and every["with_records"] == 1
    assert {r["id"] for r in every["rows"] if r["has_records"]} == {"patient_a"}
    assert [r["id"] for r in client.get("/api/patients?records=true").json()["rows"]] == ["patient_a"]
