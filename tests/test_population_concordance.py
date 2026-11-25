"""Anchored guideline-concordance review: anchors, status mapping, aggregates and prepare -> assess.

Every patient, note, provider, record and guideline space here is fabricated.
"""

import pandas as pd
import pytest

from matchminer_ai.patients.workup_queries import language as lang
from matchminer_ai.population import report
from matchminer_ai.population.anchors import choose_anchor_notes, notes_through_anchor
from matchminer_ai.population.concordance import assess_concordance, concordance_category
from matchminer_ai.population.prepare import check_output_location, prepare_anchored_population

PATIENTS = pd.DataFrame([
    {"patient_id": "p1", "provider_id": "A1", "provider_name": "Ann Alpha", "department_name": "Thoracic"},
    {"patient_id": "p2", "provider_id": "", "provider_name": "Dr  B", "department_name": ""},
    {"patient_id": "p3", "provider_id": "A1", "provider_name": "", "department_name": ""},
    {"patient_id": "p4", "provider_id": "", "provider_name": "dr b", "department_name": ""},
    {"patient_id": "p5", "provider_id": "A1", "provider_name": "", "department_name": ""},
])


def note(pid, date, text, provider_id="", provider_name="", department=""):
    return {"patient_id": pid, "note_date": date, "note_text": text, "note_type": "Progress note",
            "provider_id": provider_id, "provider_name": provider_name, "department_name": department}


NOTES = pd.DataFrame([
    note("p1", "2024-01-05", "Fabricated visit one.", "A1", "Other Name", "thoracic"),
    note("p1", "2024-02-01", "Fabricated later visit.", "Z9", "Someone Else", "thoracic"),
    note("p2", "2024-03-01", "Fabricated visit two.", "", "DR B", ""),
    note("p3", "2024-03-02", "Fabricated visit three.", "A1", "", ""),
    note("p4", "2024-03-03", "Fabricated visit four.", "", "Dr B", ""),
    note("p5", "2024-03-04", "Fabricated visit five.", "Z9", "", ""),
])


def test_anchor_choice_window_department_order_and_through_anchor_notes():
    patients = pd.DataFrame([
        {"patient_id": "id", "provider_id": "D1", "provider_name": "Wrong Name"},
        {"patient_id": "name", "provider_name": "  dr   SMITH "},
        {"patient_id": "dept", "provider_id": "D1", "department_id": "GI"},
        {"patient_id": "window", "provider_id": "D1"},
        {"patient_id": "undated", "provider_id": "D1"},
        {"patient_id": "nobody", "provider_id": "D1"},
        {"patient_id": "silent", "provider_id": "D1"},
    ])
    notes = pd.DataFrame([
        {"patient_id": "id", "note_date": "2024-01-01", "note_text": "a", "provider_id": "D1",
         "provider_name": "Dr Smith", "department_id": "TH"},
        {"patient_id": "id", "note_date": "2024-12-31 18:00", "note_text": "b", "provider_id": "D1",
         "provider_name": "Dr Smith", "department_id": "TH"},
        {"patient_id": "id", "note_date": "2023-12-31", "note_text": "outside", "provider_id": "D1",
         "provider_name": "Dr Smith", "department_id": "TH"},
        {"patient_id": "name", "note_date": "2024-05-01", "note_text": "c", "provider_id": "X",
         "provider_name": "Dr Smith", "department_id": "TH"},
        {"patient_id": "dept", "note_date": "2024-05-01", "note_text": "d", "provider_id": "D1",
         "provider_name": "", "department_id": "TH"},
        {"patient_id": "window", "note_date": "2025-01-01", "note_text": "e", "provider_id": "D1",
         "provider_name": "", "department_id": "TH"},
        {"patient_id": "undated", "note_date": "", "note_text": "f", "provider_id": "D1",
         "provider_name": "", "department_id": "TH"},
        {"patient_id": "nobody", "note_date": "2024-05-01", "note_text": "g", "provider_id": "D2",
         "provider_name": "", "department_id": "TH"},
    ])
    window = dict(start_date="2024-01-01", end_date="2024-12-31", seed=5)
    anchors, exclusions = choose_anchor_notes(patients, notes, **window)
    assert dict(zip(exclusions.patient_id, exclusions.reason)) == {
        "dept": "no_department_note", "window": "no_note_in_window", "undated": "no_dated_note",
        "nobody": "no_provider_note", "silent": "no_notes"}
    rows = anchors.set_index("patient_id")
    # ID matching ignores the name; both inclusive bounds count (the end day's 18:00 note too).
    assert rows.loc["id", "candidate_notes"] == 2 and rows.loc["name", "candidate_notes"] == 1
    # The choice is seeded per patient, not by row order.
    again, _ = choose_anchor_notes(patients.iloc[::-1], notes.iloc[::-1], **window)
    assert again.set_index("patient_id").anchor_note_id.to_dict() == rows.anchor_note_id.to_dict()
    # Through the anchor: the same day is included, later notes are not, by any author.
    notes = pd.DataFrame([
        {"patient_id": "t", "note_date": "2024-03-01 09:00", "note_text": "anchor", "provider_id": "D1"},
        {"patient_id": "t", "note_date": "2024-03-01", "note_text": "same day", "provider_id": "D2"},
        {"patient_id": "t", "note_date": "2024-03-01 10:00", "note_text": "later that day", "provider_id": "D2"},
        {"patient_id": "t", "note_date": "2024-02-01", "note_text": "earlier", "provider_id": "D2"},
        {"patient_id": "t", "note_date": "2024-03-02", "note_text": "next day", "provider_id": "D2"},
    ])
    anchors, _ = choose_anchor_notes(pd.DataFrame([{"patient_id": "t", "provider_id": "D1"}]), notes)
    assert anchors.iloc[0].anchor_time == "2024-03-01T09:00:00"
    assert notes_through_anchor(notes, anchors).note_text.tolist() == ["earlier", "same day", "anchor"]


def test_status_mapping_and_patient_counts():
    assert concordance_category("met", "applies") == ("met", None)
    assert concordance_category("completed", "applies") == ("met", None)
    assert concordance_category("done_not_recommended", "applies") == ("not_met", None)
    assert concordance_category("not_documented", "applies") == ("not_met", None)
    assert concordance_category("not_met", "applies", recommends_against=True) == \
        ("met", "not_recommended_not_done")
    assert concordance_category("unknown", "applies") == ("unknown", None)
    assert concordance_category("error", "uncertain") == ("unknown", None)
    assert concordance_category("met", "uncertain") == ("unknown", "applicability_uncertain")
    assert concordance_category("not_met", "not_applicable") == ("not_applicable", None)
    with pytest.raises(ValueError):
        concordance_category("maybe", "applies")
    items = pd.DataFrame([
        {"patient_id": "a", "category": c, "space_trial_id": "s", "space_name": "S", "disease": "d",
         **dict.fromkeys(("provider_id", "provider_name", "department_id", "department_name"))}
        for c in ("met", "not_met", "not_met", "unknown", "not_applicable")
    ] + [{"patient_id": "b", "category": "unknown", "space_trial_id": "s", "space_name": "S", "disease": "d",
          **dict.fromkeys(("provider_id", "provider_name", "department_id", "department_name"))}])
    matches = pd.DataFrame({"patient_id": ["a", "b"], "match_quality_reasonableness": [0.9, 0.8]})
    anchors = pd.DataFrame({"patient_id": ["a", "b"], "anchor_date": ["2024-01-01", "2024-01-02"]})
    rows = report.patient_results(items, matches, anchors).set_index("patient_id")
    assert rows.loc["a", ["items", "applicable", "met", "not_met", "unknown", "not_applicable"]].tolist() == \
        [5, 3, 1, 2, 1, 1]
    assert rows.loc["a", "concordance"] == pytest.approx(1 / 3) and pd.isna(rows.loc["b", "concordance"])


def test_prepare_then_assess_with_models_and_llms_replaced(tmp_path, monkeypatch):
    import matchminer_ai.matching as matching
    import matchminer_ai.patients as patients
    import matchminer_ai.trials as trials

    def summarize(notes, **_):
        ids = sorted(notes.patient_id.unique())
        return (pd.DataFrame({"patient_id": ids, "cancer_history_summary": [f"Fabricated {p}" for p in ids],
                              "general_exclusion_criteria_evidence": "Not documented."}), {},
                pd.DataFrame({"metric": ["fabricated"], "count": [0]}))

    def ct(note_id, note_date):
        evidence = [{"note_id": note_id, "note_date": note_date, "quote": "CT CHEST", "start": 0, "end": 8}]
        return {"item_type": "imaging_study", "relation": "this_report", "source_of_document": True,
                "status": "performed", "modality": "ct", "modality_text": "CT CHEST", "body_regions": ["chest"],
                "body_site_text": "", "contrast": "with_contrast", "tracer": "",
                "study_date": {"value": note_date, "precision": "day", "source": "note_date"},
                "evidence": evidence}

    def extract(notes, **_):
        rows = []
        for r in notes.to_dict("records"):
            items = [ct(r["note_id"], r["note_date"])] if r["patient_id"] == "p1" else []
            failed = r["patient_id"] == "p3"
            lists = {k: [] for k in ("imaging_studies", "procedures", "diagnoses", "biomarkers", "treatments",
                                     "disease_assessments", "performance_status")}
            lists["imaging_studies"] = items
            rows.append({"patient_id": r["patient_id"], "note_id": r["note_id"], "note_date": r["note_date"],
                         "note_type": r["note_type"], "document_type": "clinic_note",
                         "extraction_status": "failed" if failed else "complete", "error": None,
                         "item_count": len(items),
                         "record": None if failed else {
                             "schema_version": "mmai-clinical-extraction-v1", "level": "document",
                             "patient_id": r["patient_id"], "note_id": r["note_id"], "note_date": r["note_date"],
                             "note_type": None, "document_type": "clinic_note", "document_type_text": "",
                             "extensions": [], "items": lists}})
        return pd.DataFrame(rows), {"schema_version": "fabricated"}

    workup = [{"name": "CT chest", "conditions": "", "category": "2A"},
              {"name": "Pulmonary function tests", "conditions": "", "category": "2A"},
              {"name": "History and physical examination", "conditions": "", "category": "2A"}]

    def retrieve(summaries, catalog, **_):
        return pd.DataFrame([{
            "patient_id": pid, "space_trial_id": "space-1", "name": "Fabricated space", "trial_id": "g:1",
            "source": {"disease": "fab", "title": "Fabricated", "version": "1.2026"}, "retrieval_rank": 1,
            "rank": 1, "similarity_score": 0.5, "match_quality_score": 0.4,
            "match_quality_reasonableness": 0.2 if pid == "p4" else 0.9, "match_quality_pass": pid != "p4",
            "cohort_key_criteria": "Fabricated criteria.", "clinical_space_summary": "Fabricated.",
            "diagnostic_workup": workup} for pid in summaries.patient_id]), {"model_metadata": {}}

    compiles = []

    def compile_queries(items, **_):
        compiles.append([i["name"] for i in items])
        ct_query = {"summary": "CT chest.", "expressible": True, "not_expressible_reason": "",
                    "approximations": [], "conditional": False, "condition_not_expressible_reason": "",
                    "criteria": [{"id": "ct", "label": "CT", "item_type": "imaging_study", "status": "performed",
                                  "filters": [{"field": "modality", "op": "in", "values": ["ct"],
                                               "if_not_stated": "no_match"}],
                                  "window": {"anchor": "any_time", "anchor_criterion": "", "days_before": -1,
                                             "days_after": -1, "if_no_anchor": "unknown"}, "min_count": 1}],
                    "applies_when": {"all_of": []}, "not_applicable_when": {"all_of": []},
                    "requirement": {"all_of": [{"any_of": [{"criterion": "ct", "negated": False}]}]}}
        queries = [ct_query, dict(lang.EXAMPLES)["Pulmonary function tests"]]
        return [{"status": "compiled", "query": q, "cache_key": f"k{n}", "requirement": i}
                for n, (q, i) in enumerate(zip(queries, items, strict=True))], {"distinct": 2, "requests": 2}

    monkeypatch.setattr(patients, "summarize_patients", summarize)
    monkeypatch.setattr(patients, "extract_document_records", extract)
    monkeypatch.setattr(matching, "retrieve_guideline_considerations", retrieve)
    monkeypatch.setattr(patients, "compile_workup_queries", compile_queries)
    monkeypatch.setattr(trials, "load_guideline_catalog", lambda paths, **_: (None, {
        "catalog_sha256": "fabricated", "sources": [{"disease": "fab", "title": "Fabricated", "version": "1"}]}))

    with pytest.raises(ValueError, match="PHI folder"):
        check_output_location(tmp_path / "real", synthetic=False)
    prepared_dir, out = tmp_path / "prepared", tmp_path / "assessment"
    prepared = prepare_anchored_population(PATIENTS, NOTES, output_dir=prepared_dir, end_date="2024-12-31",
                                           levels="document", synthetic=True)
    assert prepared.exclusions.set_index("patient_id").reason.to_dict() == {"p5": "no_provider_note"}
    assert prepare_anchored_population(PATIENTS, NOTES, output_dir=prepared_dir, end_date="2024-12-31",
                                       levels="document", synthetic=True, check=True) == {
        "anchors": (0, "complete"), "summaries": (0, "complete"), "extractions": (0, "complete")}
    tables = assess_concordance(prepared_dir, catalogs=[tmp_path / "catalog"], output_dir=out, synthetic=True)
    assert tables["flow"] == {"given": 5, "anchored": 4, "summarized": 4, "extracted": 3, "matched": 2,
                              "assessed": 2, "excluded": {
                                  "anchor": {"no_provider_note": 1}, "summary": {},
                                  "extraction": {"no_extraction_records": 1},
                                  "match": {"no_reasonable_guideline_match": 1}, "assessment": {}}}
    assert compiles == [["CT chest", "Pulmonary function tests"]]  # history and physical left out
    results = tables["patient_results"].set_index("patient_id")
    assert results.loc["p1", ["items", "met", "unknown", "concordance"]].tolist() == [2, 1, 1, 1.0]
    assert results.loc["p2", ["met", "not_met", "concordance"]].tolist() == [0, 1, 0.0]
    matches = pd.read_parquet(out / "matches" / "guideline_matches.parquet").set_index("patient_id")
    assert matches.loc["p4", "excluded"] and matches.loc["p4", "match_quality_reasonableness"] == 0.2
    providers = tables["provider_summary"].fillna("").set_index("provider_id")
    assert providers.loc["A1", ["patients", "included", "excluded_no_extraction_records",
                                "excluded_no_provider_note", "mean_concordance"]].tolist() == [3, 1, 1, 1, 1.0]
    by_name = tables["provider_summary"].set_index("provider_name")
    assert by_name.loc["Dr  B", "pooled_concordance"] == 0.0 and by_name.loc["Dr  B", "small_group"]
    departments = tables["department_summary"].fillna("").set_index("department_name")
    assert departments.loc["Thoracic", "included"] == 1 and departments.loc["", "patients"] == 4
    assert "Concordance is measured from documentation" in (out / "report.md").read_text()
    assert oct((out / "item_results.parquet").stat().st_mode & 0o777) == "0o600"
    # The precomputed path: the same folder again is complete and not recomputed.
    assert assess_concordance(prepared_dir, catalogs=[tmp_path / "catalog"], output_dir=out, synthetic=True,
                              check=True) == {"matches": (0, "complete"), "assessment": (0, "complete")}
