"""The eligibility-criterion query: the workup query language's matchers, asked whether a line is satisfied or
clearly failed.

A criterion query (``kind="criterion"`` of :mod:`matchminer_ai.patients.workup_queries.language`) has the workup
query's ``criteria`` (named item matchers with filters and date windows), ``conditional``, ``applies_when`` and
``not_applicable_when``, and two expressions in place of ``requirement``, both relative to eligibility:

* ``meets_when``: documented facts showing the patient satisfies the line (an inclusion line met, or an exclusion
  line's condition documented absent);
* ``fails_when``: documented facts showing the patient clearly fails it (an inclusion line contradicted, such as a
  negative result for a required marker, or an exclusion line's condition documented present).

Each nonempty expression needs a criterion that is not negated: a missing item never shows that a patient satisfies
or fails a line. A line no item type can check (age, organ-function thresholds, comorbidities) is not expressible.
"""
from __future__ import annotations

import json

from matchminer_ai.patients.workup_queries.language import (
    _all,
    _criterion,
    _NONE,
    check_query,
    criterion_query_response_schema,
    validate_criterion_query,
)

#: The criterion query's own version (its matchers are the workup language's, mmai-workup-query-v2).
CRITERION_QUERY_VERSION = "mmai-criterion-query-v1"
LINE_TYPES = ("inclusion", "exclusion")


def _recent(days: int) -> dict:
    return {"anchor": "reference_date", "anchor_criterion": "", "days_before": days, "days_after": 0,
            "if_no_anchor": "unknown"}


def _query(summary, criteria, *, meets=_NONE, fails=_NONE, approximations=(), applies=_NONE,
           not_applicable=_NONE, conditional=False, condition_reason=""):
    return {
        "summary": summary, "expressible": True, "not_expressible_reason": "",
        "approximations": list(approximations), "conditional": conditional,
        "condition_not_expressible_reason": condition_reason, "criteria": criteria,
        "applies_when": applies, "not_applicable_when": not_applicable,
        "meets_when": meets, "fails_when": fails,
    }


_ICI = ["pembrolizumab", "nivolumab", "cemiplimab", "dostarlimab", "atezolizumab", "durvalumab", "avelumab",
        "ipilimumab", "tremelimumab", "tislelizumab", "toripalimab", "retifanlimab", "anti-PD-1", "anti-PD-L1",
        "anti-CTLA-4", "checkpoint inhibitor", "immunotherapy"]

#: (line type, line text, query): the compile prompt's examples, each valid (checked in the tests).
CRITERION_EXAMPLES: list[tuple[str, str, dict]] = [
    ("inclusion", "ECOG performance status of 0 or 1.", _query(
        "An ECOG score of 0-1 documented in the 60 days before screening; a score of 2 or more fails.",
        [_criterion("ecog_ok", "ECOG 0-1", "performance_status",
                    [("scale", "in", ["ecog"], "no_match"), ("score", "number_at_most", ["1"], "no_match")],
                    status="any", window=_recent(60)),
         _criterion("ecog_poor", "ECOG 2 or more", "performance_status",
                    [("scale", "in", ["ecog"], "no_match"), ("score", "number_at_least", ["2"], "no_match")],
                    status="any", window=_recent(60))],
        meets=_all([("ecog_ok", False)]), fails=_all([("ecog_poor", False)]),
        approximations=["Only ECOG scores documented in the 60 days before the screening date count; Karnofsky "
                        "scores are not converted."])),
    ("inclusion", "Documented activating EGFR mutation (exon 19 deletion or L858R).", _query(
        "A positive EGFR result naming exon 19 deletion or L858R; a negative EGFR result with no positive one "
        "fails.",
        [_criterion("egfr_activating", "EGFR exon 19 deletion or L858R", "biomarker",
                    [("biomarker", "text_contains_any", ["EGFR"], "no_match"),
                     ("result", "in", ["positive"], "no_match"),
                     ("alteration", "text_contains_any",
                      ["exon 19", "19del", "E746_A750del", "L858R"], "unknown")], status="any"),
         _criterion("egfr_positive", "EGFR positive", "biomarker",
                    [("biomarker", "text_contains_any", ["EGFR"], "no_match"),
                     ("result", "in", ["positive"], "no_match")], status="any"),
         _criterion("egfr_negative", "EGFR negative", "biomarker",
                    [("biomarker", "text_contains_any", ["EGFR"], "no_match"),
                     ("result", "in", ["negative"], "no_match")], status="any")],
        meets=_all([("egfr_activating", False)]),
        fails=_all([("egfr_negative", False)], [("egfr_positive", True)]))),
    ("exclusion", "Prior treatment with an anti-PD-1, anti-PD-L1 or anti-CTLA-4 antibody.", _query(
        "A received immune checkpoint inhibitor, named among a treatment's agents or in its regimen.",
        [_criterion("ici_agent", "checkpoint inhibitor (agents)", "treatment",
                    [("agents", "text_contains_any", _ICI, "no_match")]),
         _criterion("ici_regimen", "checkpoint inhibitor (regimen)", "treatment",
                    [("regimen", "text_contains_any", _ICI, "no_match")])],
        fails=_all([("ici_agent", False), ("ici_regimen", False)]))),
    ("inclusion", "No prior systemic therapy for metastatic disease.", _query(
        "A received systemic therapy given in the advanced or metastatic setting fails; its absence is not "
        "evidence.",
        [_criterion("metastatic_therapy", "systemic therapy for advanced disease", "treatment",
                    [("treatment_type", "in", ["systemic_therapy"], "no_match"),
                     ("setting", "in", ["advanced_or_metastatic"], "unknown")])],
        fails=_all([("metastatic_therapy", False)]),
        approximations=["A therapy whose setting is not stated is uncertain, not failing."])),
    ("exclusion", "Radiotherapy within 14 days before the first dose of study treatment.", _query(
        "Radiation therapy started in the 14 days before screening fails.",
        [_criterion("recent_rt", "radiation in the last 14 days", "treatment",
                    [("treatment_type", "in", ["radiation_therapy"], "no_match")], window=_recent(14))],
        fails=_all([("recent_rt", False)]),
        approximations=["The window tests the radiation start date; end dates are not checked, and the first "
                        "dose is read as the screening date."])),
    ("inclusion", "For patients with prior surgery, at least 4 weeks must have passed since the operation.", _query(
        "For a patient with documented surgery, a surgery in the 28 days before screening fails.",
        [_criterion("any_surgery", "prior surgery", "treatment",
                    [("treatment_type", "in", ["surgery"], "no_match")]),
         _criterion("recent_surgery", "surgery in the last 28 days", "treatment",
                    [("treatment_type", "in", ["surgery"], "no_match")], window=_recent(28))],
        conditional=True, applies=_all([("any_surgery", False)]),
        fails=_all([("recent_surgery", False)]))),
    ("inclusion", "Adequate organ function: ANC >= 1.5 x 10^9/L, platelets >= 100 x 10^9/L, total bilirubin "
     "<= 1.5 x ULN.", {
        "summary": "Laboratory thresholds cannot be checked from the extracted items.", "expressible": False,
        "not_expressible_reason": "Lab items record which test was done and its result as free text, not numeric "
        "values to compare with thresholds.", "approximations": [], "conditional": False,
        "condition_not_expressible_reason": "", "criteria": [], "applies_when": _NONE,
        "not_applicable_when": _NONE, "meets_when": _NONE, "fails_when": _NONE}),
]


def render_criterion_examples() -> str:
    return "\n\n".join(
        f"{line_type.capitalize()} line: {text}\n{json.dumps(query, ensure_ascii=False)}"
        for line_type, text, query in CRITERION_EXAMPLES
    )


def check_criterion_query(query, extensions=()) -> dict:
    """A criterion query in canonical form; raises ``QueryError`` when it is invalid."""
    return check_query(query, extensions, kind="criterion")


__all__ = [
    "CRITERION_EXAMPLES",
    "CRITERION_QUERY_VERSION",
    "LINE_TYPES",
    "check_criterion_query",
    "criterion_query_response_schema",
    "render_criterion_examples",
    "validate_criterion_query",
]
