"""Run compiled criterion queries on patients' extraction records: one result per patient and line, in code.

Results use the decision model's polarity: a line is ``clearly_fails`` only when documented facts contradict
eligibility (``fails_when`` holds, ``meets_when`` does not, and the line applies); every other status is "not
clearly failing". Statuses:

* ``clearly_fails``: an inclusion line contradicted, or an exclusion line's condition documented;
* ``meets``: documented facts show the patient satisfies the line;
* ``not_documented``: no extracted item shows either (not evidence of failing);
* ``uncertain``: a candidate item's date or details are not certain, or a failing fact under a condition that may
  not apply;
* ``conflicting``: documented facts both satisfy and fail the line;
* ``not_applicable``: documented facts show the line does not apply;
* ``not_checkable``: no item type can check the line (age, lab thresholds, comorbidities);
* ``no_records``: the patient has no extraction records;
* ``query_unavailable``: no valid query could be compiled (or the line has no type).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from matchminer_ai.patients.workup_queries.evaluate import (
    MAYBE,
    NO,
    YES,
    PatientData,
    _all_extensions,
    _criterion_entry,
    _evidence,
    _explain,
    _incomplete,
    evaluate_criterion,
    evaluate_expression,
    patient_data,
    query_applicability,
    query_context,
)

from .language import check_criterion_query

STATUSES = ("clearly_fails", "meets", "not_documented", "uncertain", "conflicting", "not_applicable",
            "not_checkable", "no_records", "query_unavailable")
#: Statuses that say something about the line from the patient's items (the others: nothing could be checked).
CHECKED = frozenset({"clearly_fails", "meets", "not_documented", "uncertain", "conflicting", "not_applicable"})
NOTICE = (
    "Structured-record criteria screen for human review, not an eligibility determination. Queries compiled by an "
    "LLM from each criteria line ran in code on items an LLM extracted from the notes; check every item against its "
    "source note and the current protocol. Not documented means no extracted item showed the line satisfied or "
    "failed, not that the patient is eligible. Lines no extracted item type can check are marked not checkable."
)


def _empty(expression: dict) -> bool:
    return not expression["all_of"]


def _result(status, reason, text, *, line_type, applicability="uncertain", query=None, criteria=None,
            evidence=None, limitations=None, reference_date=None) -> dict:
    return {
        "status": status,
        "clearly_fails": status == "clearly_fails",
        "reason": reason,
        "applicability": applicability,
        "criterion_type": line_type,
        "bottom_line": text,
        "criteria": criteria or [],
        "evidence": evidence or [],
        "limitations": limitations if limitations is not None else (
            list(query.get("approximations", [])) if query else []),
        "reference_date": reference_date,
        "query_summary": query.get("summary") if query else None,
    }


def _tagged(evidence: list[dict], side: str) -> list[dict]:
    return [{**e, "side": side} for e in evidence]


def _uncertain_reason(ctx, groups) -> str:
    found = {r for state, literals in groups if state == MAYBE for lit, s in literals
             if s == MAYBE and not lit["negated"] for r in evaluate_criterion(ctx, lit["criterion"]).reasons}
    if "anchor_unavailable" in found:
        return "anchor_unavailable"
    if any("date" in r or "mention" in r or "plan" in r for r in found):
        return "uncertain_dates"
    return "fields_not_stated"


def evaluate_criterion_query(
    query: dict | None,
    patient: PatientData,
    *,
    criterion_type: str,
    reference_date=None,
    anchors: Mapping[str, object] | None = None,
    extensions: Iterable[str] | None = None,
) -> dict:
    """Evaluate one compiled criterion query on one patient's items (see the module docstring for statuses).

    ``reference_date`` (YYYY-MM-DD, the screening date) dates ``reference_date`` windows; it defaults to the
    patient's latest note date. Returns ``status``, ``clearly_fails``, ``reason``, ``applicability``, a machine-made
    ``bottom_line``, each criterion's matches with dates and evidence, the ``evidence`` quotes (each with ``side``:
    ``fails``, ``meets`` or ``not_applicable``), ``limitations`` and the ``query_summary``.
    """
    extensions = _all_extensions() if extensions is None else list(extensions)
    if query is None:
        return _result("query_unavailable", "query_unavailable",
                       "Not checked: no valid query could be compiled for this line.", line_type=criterion_type)
    query = check_criterion_query(query, extensions)
    if not query["expressible"]:
        return _result("not_checkable", "not_expressible",
                       "Not checkable from extracted items (" + query["not_expressible_reason"].strip().rstrip(".")
                       + ").", line_type=criterion_type, query=query)
    if not patient.has_records:
        return _result("no_records", "no_extraction", "Not checked: no extraction records for this patient.",
                       line_type=criterion_type, query=query)
    ctx = query_context(query, patient, reference_date=reference_date, anchors=anchors, extensions=extensions)
    applicability, notes, excluded_groups = query_applicability(ctx, query)
    meets, meet_groups = (NO, []) if _empty(query["meets_when"]) else evaluate_expression(ctx, query["meets_when"])
    fails, fail_groups = (NO, []) if _empty(query["fails_when"]) else evaluate_expression(ctx, query["fails_when"])
    exclusion = criterion_type == "exclusion"
    failed_words = "Excluding condition documented" if exclusion else "Clearly not met"
    met_words = "Excluding condition documented absent" if exclusion else "Met"
    evidence: list[dict] = []
    if applicability == "not_applicable":
        status, reason = "not_applicable", "documented_not_applicable"
        text = "Not applicable: " + "; ".join(_explain(ctx, excluded_groups, YES)) + "."
        evidence = _tagged(_evidence(ctx, excluded_groups, (YES,)), "not_applicable")
    elif fails == YES and meets == YES:
        status, reason = "conflicting", "documented_both"
        text = (f"Conflicting: {met_words.lower()} ({'; '.join(_explain(ctx, meet_groups, YES))}) and "
                f"{failed_words.lower()} ({'; '.join(_explain(ctx, fail_groups, YES))}).")
        evidence = (_tagged(_evidence(ctx, fail_groups, (YES,)), "fails")
                    + _tagged(_evidence(ctx, meet_groups, (YES,)), "meets"))
    elif fails == YES and applicability == "applies":
        status, reason = "clearly_fails", "documented_failing"
        text = f"{failed_words}: " + "; ".join(_explain(ctx, fail_groups, YES)) + "."
        evidence = _tagged(_evidence(ctx, fail_groups, (YES,)), "fails")
    elif fails == YES:
        status, reason = "uncertain", "applicability_uncertain"
        text = ("Possibly failing, if the line applies: " + "; ".join(_explain(ctx, fail_groups, YES)) + ".")
        evidence = _tagged(_evidence(ctx, fail_groups, (YES,)), "fails")
    elif meets == YES:
        status, reason = "meets", "documented_meeting"
        text = f"{met_words}: " + "; ".join(_explain(ctx, meet_groups, YES)) + "."
        evidence = _tagged(_evidence(ctx, meet_groups, (YES,)), "meets")
    elif MAYBE in (fails, meets):
        status = "uncertain"
        groups = [g for value, gs in ((fails, fail_groups), (meets, meet_groups)) if value == MAYBE for g in gs]
        reason = _uncertain_reason(ctx, groups)
        parts = []
        if fails == MAYBE:
            parts.append("possibly failing: " + "; ".join(_explain(ctx, fail_groups, MAYBE)))
            evidence += _tagged(_evidence(ctx, fail_groups, (MAYBE,)), "fails")
        if meets == MAYBE:
            parts.append("possibly met: " + "; ".join(_explain(ctx, meet_groups, MAYBE)))
            evidence += _tagged(_evidence(ctx, meet_groups, (MAYBE,)), "meets")
        text = "Uncertain: " + "; ".join(parts) + "."
    else:
        status, reason = "not_documented", "not_documented"
        text = ("Not documented: no extracted item shows this line satisfied or failed. A missing item does not "
                "mean the patient satisfies or fails the line.")
    if notes and status not in ("not_applicable",):
        text += " " + " ".join(notes)
    criteria = [_criterion_entry(evaluate_criterion(ctx, c["id"])) for c in query["criteria"]]
    limitations = list(query["approximations"])
    if patient.incomplete:
        limitations.append(_incomplete(patient).capitalize() + ".")
    return _result(status, reason, text, line_type=criterion_type, applicability=applicability, query=query,
                   criteria=criteria, evidence=evidence, limitations=limitations,
                   reference_date=ctx.reference_text)


def check_criteria_with_extractions(
    records,
    compiled: Sequence[dict],
    *,
    patient_ids: Sequence[str] | None = None,
    reference_dates: Mapping[str, object] | str | None = None,
    anchors: Mapping[str, Mapping[str, object]] | None = None,
    coverage: Mapping[str, Mapping] | None = None,
) -> dict:
    """Check every compiled criteria line on every patient's extraction records.

    ``records`` are as for :func:`~matchminer_ai.patients.workup_queries.check_workup_with_extractions`;
    ``compiled`` is :func:`~.compile.compile_criteria_queries` output (one entry per line, in order).
    ``patient_ids`` adds patients with no records and fixes the order; ``reference_dates`` is one screening date
    for everyone or dates by patient. Returns ``{"patients": [{"patient_id", "lines", "counts"}], "aggregate",
    "notice"}``. Pure computation: no I/O, no model call.
    """
    data = patient_data(records, coverage=coverage)
    ids = [str(p) for p in patient_ids] if patient_ids is not None else sorted(data)
    if len(set(ids)) != len(ids):
        raise ValueError("patient_ids repeat")
    extensions = _all_extensions()
    queries = [check_criterion_query(e["query"], extensions) if e.get("status") == "compiled"
               and e.get("query") is not None else None for e in compiled]
    patients = []
    for pid in ids:
        patient = data.get(pid) or PatientData(pid)
        reference = reference_dates.get(pid) if isinstance(reference_dates, Mapping) else reference_dates
        lines = []
        for entry, query in zip(compiled, queries, strict=True):
            line = entry["line"]
            result = evaluate_criterion_query(query, patient, criterion_type=line.get("criterion_type"),
                                              reference_date=reference, anchors=(anchors or {}).get(pid),
                                              extensions=extensions)
            lines.append({"number": line.get("number"), "text": line.get("text"),
                          "boilerplate": bool(line.get("boilerplate")), **result,
                          "query_key": entry.get("cache_key")})
        patients.append({"patient_id": pid, "lines": lines, "counts": line_counts(lines),
                         "coverage": patient.coverage()})
    return {"patients": patients, "aggregate": aggregate_structured_criteria(patients, compiled),
            "notice": NOTICE}


def line_counts(lines: Sequence[dict]) -> dict:
    """One patient's lines by status, with how many were checked and clearly failed."""
    counts = {s: 0 for s in STATUSES}
    for line in lines:
        counts[line["status"]] += 1
    return {**counts, "lines": len(lines), "checked": sum(counts[s] for s in CHECKED),
            "clearly_failed_lines": [line["number"] for line in lines if line["clearly_fails"]]}


def aggregate_structured_criteria(patients: Sequence[dict], compiled: Sequence[dict]) -> list[dict]:
    """Per line: statuses across patients, and how many patients clearly fail it."""
    rows = []
    for number, entry in enumerate(compiled):
        statuses = {s: 0 for s in STATUSES}
        for patient in patients:
            if len(patient["lines"]) != len(compiled):
                raise ValueError("Incomplete structured criteria matrix.")
            statuses[patient["lines"][number]["status"]] += 1
        rows.append({"index": number, "number": entry["line"].get("number"), "text": entry["line"].get("text"),
                     "criterion_type": entry["line"].get("criterion_type"),
                     "boilerplate": bool(entry["line"].get("boilerplate")),
                     "compile_status": entry.get("status"), "patient_count": len(patients),
                     "statuses": statuses, "clearly_fail": statuses["clearly_fails"],
                     "checked": sum(statuses[s] for s in CHECKED)})
    return rows


__all__ = [
    "CHECKED",
    "NOTICE",
    "STATUSES",
    "aggregate_structured_criteria",
    "check_criteria_with_extractions",
    "evaluate_criterion_query",
    "line_counts",
]
