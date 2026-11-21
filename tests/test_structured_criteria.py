"""Criteria-level screening from extraction records: the criterion query kind, its evaluator and compile loop.

Every record, note and criteria line here is fabricated; no clinical data is used.
"""

import copy
import json

import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.matching.structured_criteria import (
    CRITERION_EXAMPLES,
    check_criteria_with_extractions,
    compile_criteria_queries,
    criteria_query_cache_key,
    validate_criterion_query,
)

EXAMPLES = {text: query for _, text, query in CRITERION_EXAMPLES}
ECOG = EXAMPLES["ECOG performance status of 0 or 1."]
ICI = EXAMPLES["Prior treatment with an anti-PD-1, anti-PD-L1 or anti-CTLA-4 antibody."]
SURGERY = EXAMPLES["For patients with prior surgery, at least 4 weeks must have passed since the operation."]
KEYS = {"performance_status": "performance_status", "treatment": "treatments", "biomarker": "biomarkers"}


def quote(note_id, note_date, text):
    return {"note_id": note_id, "note_date": note_date, "quote": text, "start": 0, "end": len(text)}


def ecog(note_id, note_date, score):
    return {"item_type": "performance_status", "scale": "ecog", "score": score,
            "evidence": [quote(note_id, note_date, f"ECOG {score}")]}


def treatment(note_id, note_date, kind="systemic_therapy", agents=("carboplatin",), start=None):
    return {"item_type": "treatment", "treatment_type": kind, "regimen": "", "agents": list(agents), "site": "",
            "setting": "not_stated", "line_of_therapy": 0, "treatment_status": "completed",
            "start_date": {"value": start, "precision": "day" if start else "unknown",
                           "source": "document_text" if start else "unknown"},
            "end_date": {"value": None, "precision": "unknown", "source": "unknown"},
            "best_response": "not_stated", "end_reason": "", "toxicities": [],
            "evidence": [quote(note_id, note_date, " ".join(agents) or kind)]}


def document(patient, note_id, note_date, items=()):
    lists = {key: [] for key in ("imaging_studies", "procedures", "diagnoses", "biomarkers", "treatments",
                                 "disease_assessments", "performance_status", "lab_tests", "counseling")}
    for item in items:
        lists[KEYS[item["item_type"]]].append(item)
    record = {"schema_version": "mmai-clinical-extraction-v2", "level": "document", "patient_id": patient,
              "note_id": note_id, "note_date": note_date, "note_type": None, "document_type": "clinic_note",
              "document_type_text": "", "extensions": [], "items": lists}
    return {"patient_id": patient, "note_id": note_id, "note_date": note_date, "note_type": None,
            "document_type": "clinic_note", "extraction_status": "complete", "error": None,
            "item_count": len(items), "record": record}


def compiled(*pairs):
    return [{"line": {"number": n, "text": text, "criterion_type": kind, "boilerplate": False},
             "status": "compiled", "query": copy.deepcopy(query), "cache_key": f"k{n}"}
            for n, (kind, text, query) in enumerate(pairs, 1)]


def test_examples_validate_and_absence_never_fails_a_line():
    for _, text, query in CRITERION_EXAMPLES:
        assert validate_criterion_query(query, []) == [], text
    absent = copy.deepcopy(ICI)
    absent["fails_when"]["all_of"][0]["any_of"] = [{"criterion": "ici_agent", "negated": True},
                                                    {"criterion": "ici_regimen", "negated": True}]
    assert any("fails_when: needs a documented fact" in p for p in validate_criterion_query(absent, []))
    empty = copy.deepcopy(ICI)
    empty["fails_when"] = {"all_of": []}
    problems = validate_criterion_query(empty, [])
    assert any("an expressible line needs at least one" in p for p in problems)


def test_statuses_follow_the_decision_models_polarity():
    rows = [
        document("ok", "n1", "2025-03-01", [ecog("n1", "2025-03-01", 1)]),
        document("poor", "n2", "2025-03-01", [ecog("n2", "2025-03-01", 3),
                                              treatment("n2", "2025-03-01", agents=("pembrolizumab",))]),
        document("both", "n3", "2025-03-01", [ecog("n3", "2025-03-01", 1), ecog("n3", "2025-03-01", 2)]),
        document("old", "n4", "2024-01-01", [ecog("n4", "2024-01-01", 3)]),
        document("old", "n5", "2025-03-01"),
        document("cut", "n6", "2025-03-01", [treatment("n6", "2025-03-01", "surgery", (), "2025-02-20")]),
    ]
    lines = compiled(("inclusion", "ECOG 0-1", ECOG), ("exclusion", "Prior ICI", ICI),
                     ("inclusion", "4 weeks after surgery", SURGERY))
    result = check_criteria_with_extractions(rows, lines, patient_ids=["ok", "poor", "both", "old", "cut", "none"])
    status = {p["patient_id"]: [line["status"] for line in p["lines"]] for p in result["patients"]}
    assert status["ok"] == ["meets", "not_documented", "not_documented"]
    # Documented contrary facts fail; an exclusion's condition documented fails it.
    assert status["poor"] == ["clearly_fails", "clearly_fails", "not_documented"]
    assert status["both"][0] == "conflicting"
    # A score outside the 60-day window says nothing about the present.
    assert status["old"][0] == "not_documented"
    assert status["cut"][2] == "clearly_fails"
    assert status["none"] == ["no_records"] * 3
    poor = next(p for p in result["patients"] if p["patient_id"] == "poor")
    assert poor["counts"]["clearly_failed_lines"] == [1, 2]
    assert poor["lines"][0]["evidence"][0]["side"] == "fails"
    assert [row["clearly_fail"] for row in result["aggregate"]] == [1, 1, 1]


class FakeBackend:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
        self.calls.append((prompt_list, llm_config))
        texts = [json.dumps(self.answers.pop(0)) for _ in prompt_list]
        return LLMGenerationResult(final_outputs=texts, finish_reasons=["stop"] * len(texts),
                                   reasoning_outputs=[""] * len(texts), raw_outputs=[],
                                   model_metadata={"model_sha": "test"})


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    return config


def test_compile_shares_boilerplate_lines_across_cohorts_and_skips_untyped_lines(config, monkeypatch):
    backend = FakeBackend([ECOG, ICI])
    monkeypatch.setattr("matchminer_ai.patients.workup_queries.compile.get_llm_backend", lambda _c: backend)
    lines = [{"number": 1, "text": "ECOG 0-1", "criterion_type": "inclusion", "boilerplate": False},
             {"number": 2, "text": "Prior ICI", "criterion_type": "exclusion", "boilerplate": True},
             {"number": 3, "text": "Inclusion criteria:", "criterion_type": None, "boilerplate": False}]
    cache: dict = {}
    entries = compile_criteria_queries(lines, cohort_context="Cohort A {line}", config=config, cache=cache)
    assert [e["status"] for e in entries] == ["compiled", "compiled", "unchecked"]
    prompt = backend.calls[0][0][0].messages[0]["content"]
    assert "Type: inclusion\nText: ECOG 0-1" in prompt and "Cohort A {line}" in prompt
    # A boilerplate line has no context, so another cohort reuses its query without a request.
    assert criteria_query_cache_key(lines[1], "Cohort A") == criteria_query_cache_key(lines[1], "Cohort B")
    assert criteria_query_cache_key(lines[0], "Cohort A") != criteria_query_cache_key(lines[0], "Cohort B")
    again = compile_criteria_queries([dict(lines[1], number=7)], cohort_context="Cohort B", config=config,
                                     cache=cache)
    assert again[0]["cached"] and again[0]["line"]["number"] == 7 and len(backend.calls) == 1
