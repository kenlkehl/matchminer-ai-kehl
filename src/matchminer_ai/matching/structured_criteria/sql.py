"""Criteria lines checked by the SQL agent: one accepted cohort-level query per line, run once over a cohort.

The agent (:mod:`matchminer_ai.patients.workup_sql.agent`, the same loop, isolated read-only sandbox, profile and
search as for workup checks) explores an exploration database built from a whole extraction store and writes, per
checkable criteria line, one SQL query returning each patient's outcome: meets, fails, conflicting, uncertain,
not_documented or not_applicable. Code then runs each accepted query once over a cohort's records, with no model
call, and reports the statuses of :mod:`.evaluate`. As for workup, exploration results are patient-derived extracted
values (no patient ids, note text or quotes) and reach the configured endpoint.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path

from matchminer_ai._storage import digest
from matchminer_ai.cancellation import check_cancelled
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.structured import StructuredConfig
from matchminer_ai.patients.workup_queries.evaluate import PatientData, _incomplete
from matchminer_ai.patients.workup_sql.agent import (
    SQLTask,
    WorkupSQLLimits,
    _database_facts,
    agent_contract,
    agent_limits,
    compile_sql_queries,
    run_final_query,
)
from matchminer_ai.patients.workup_sql.database import ExtractionDatabase, build_extraction_database
from matchminer_ai.patients.workup_sql.evaluate import _evidence
from matchminer_ai.patients.workup_sql.sandbox import SQLSandbox, SQLSandboxError

from .compile import checkable, line_context, line_text
from .evaluate import NOTICE, aggregate_structured_criteria, line_counts

METHOD = "criteria_sql"
CRITERIA_SQL_REASONS = {
    "meets": ("documented_meeting",),
    "fails": ("documented_failing",),
    "conflicting": ("documented_both",),
    "uncertain": ("uncertain_dates", "fields_not_stated", "anchor_unavailable", "applicability_uncertain"),
    "not_documented": ("not_documented",),
    "not_applicable": ("documented_not_applicable",),
}
#: The agent's outcomes as the statuses of :mod:`.evaluate`.
STATUS_OF = {"meets": "meets", "fails": "clearly_fails", "conflicting": "conflicting", "uncertain": "uncertain",
             "not_documented": "not_documented", "not_applicable": "not_applicable"}


def _rules(row, flag, recommends_against) -> None:
    if recommends_against:
        flag("recommends_against must be false for a criteria line", "recommends_against for a line")
    if row["status"] == "fails" and row["applicability"] != "applies":
        flag("status fails needs applicability applies (give uncertain / applicability_uncertain when the line "
             "may not apply)", "fails without applies")


CRITERIA_SQL_TASK = SQLTask(
    name="criteria", prompt_file="patient.criteria_sql.system.txt", agent_version="mmai-criteria-sql-agent-v1",
    reasons=CRITERIA_SQL_REASONS, subject="line", subject_text=line_text,
    label=lambda line: line["text"][:80], context_key="trial_cohort", context_sha_key="context_sha256",
    row_rules=_rules, context_for=line_context,
)


def criteria_sql_contract(config: MMAIConfig | None = None, extensions=()) -> dict:
    return agent_contract(config, extensions, task=CRITERIA_SQL_TASK)


def criteria_sql_cache_key(line: dict, cohort_context: str | None = None, *, database_fingerprint: str,
                           contract: dict) -> str:
    """The cache key of one line's accepted query (a boilerplate line's is shared across trials)."""
    values = line_text(line)
    return digest({"line": values["text"], "criterion_type": values["criterion_type"],
                   "context": line_context(line, cohort_context), "database": database_fingerprint, **contract})


def compile_criteria_sql(
    lines: Sequence[dict],
    *,
    database: str | Path | ExtractionDatabase,
    config: MMAIConfig | None = None,
    cohort_context: str | None = None,
    cache: MutableMapping[str, dict] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    return_metadata: bool = False,
    llm: StructuredConfig | None = None,
    activity_callback: Callable[[list[dict]], None] | None = None,
):
    """One accepted cohort-level SQL query per checkable criteria line (agent runs as
    :func:`~matchminer_ai.patients.workup_sql.compile_workup_sql` documents; the agent uses
    ``config.workup_sql_agent``). Returns one entry per line, in order, with the ``line`` (and its ``number``);
    a line without an inclusion or exclusion type is ``unchecked``."""
    config = config or load_default_preset()
    fingerprint, extensions = _database_facts(database)[:2]
    contract = criteria_sql_contract(config, extensions)
    numbered = [(n, line) for n, line in enumerate(lines) if checkable(line)]
    subjects = [line_text(line) for _, line in numbered]
    keys = [criteria_sql_cache_key(line, cohort_context, database_fingerprint=fingerprint, contract=contract)
            for _, line in numbered]
    result = compile_sql_queries(subjects, keys, task=CRITERIA_SQL_TASK, database=database, config=config,
                                 context=cohort_context, contract=contract, cache=cache, progress_callback=progress_callback,
                                 return_metadata=return_metadata, llm=llm, activity_callback=activity_callback) \
        if numbered else ([], {"lines": 0, "distinct": 0, "cached": 0, "compiled": 0, "failed": 0,
                               "requests": 0, "contract": contract}) if return_metadata else []
    entries_in, metadata = result if return_metadata else (result, None)
    by_position = {n: entry for (n, _), entry in zip(numbered, entries_in, strict=True)}
    entries = []
    for n, line in enumerate(lines):
        if n in by_position:
            entry = by_position[n]
            entries.append({**entry, "line": {**entry["line"], "number": line.get("number")}})
        else:
            entries.append({"line": {"text": (line or {}).get("text"), "number": (line or {}).get("number"),
                                     "criterion_type": (line or {}).get("criterion_type"),
                                     "boilerplate": bool((line or {}).get("boilerplate"))},
                            "status": "unchecked", "sql": None, "cache_key": None,
                            "error": "the criterion has no inclusion or exclusion type", "contract": contract})
    if not return_metadata:
        return entries
    metadata["unchecked_lines"] = len(lines) - len(numbered)
    return entries, metadata


def _assess(row, entry, patient: PatientData, database: ExtractionDatabase, *, failure=None, reference=None):
    line = entry["line"]
    base = {"criterion_type": line.get("criterion_type"), "criteria": [], "reference_date": reference,
            "query_summary": entry.get("summary") or None,
            "limitations": list(entry.get("approximations") or [])}

    def result(status, reason, text, applicability="uncertain", evidence=None):
        return {**base, "status": status, "clearly_fails": status == "clearly_fails", "reason": reason,
                "applicability": applicability, "bottom_line": text, "evidence": evidence or []}

    if entry.get("status") != "compiled":
        return result("query_unavailable", "query_unavailable", "Not checked: no SQL query could be accepted "
                      "for this criterion.")
    if entry.get("expressible") is False:
        reason = (entry.get("not_expressible_reason") or "").strip().rstrip(".")
        return result("not_checkable", "not_expressible", "Not checkable from extracted items"
                      + (f" ({reason})" if reason else "") + ".")
    if failure is not None:
        return result("query_unavailable", "query_unavailable", f"Not checked: {failure}")
    if not patient.has_records:
        return result("no_records", "no_extraction", "Not checked: no extraction records for this patient.")
    status = STATUS_OF[row["status"]]
    views = [(k, database.views[k]) for k in row["evidence_item_keys"] if k in database.views]
    side = {"clearly_fails": "fails", "meets": "meets", "not_applicable": "not_applicable"}.get(status, "fails")
    evidence = [{**e, "side": side} for e in _evidence(views, "unknown" if status == "uncertain" else "yes")]
    detail = row["detail"].rstrip(".")
    lead = {"clearly_fails": ("Excluding condition documented" if line.get("criterion_type") == "exclusion"
                              else "Clearly not met"),
            "meets": "Met", "conflicting": "Conflicting", "uncertain": "Uncertain",
            "not_documented": "Not documented", "not_applicable": "Not applicable"}[status]
    text = f"{lead}: {detail}." if detail else f"{lead}."
    if status == "not_documented":
        text += " A missing item does not mean the patient satisfies or fails the criterion."
    out = result(status, row["reason"], text, row["applicability"], evidence)
    missing = database.uncovered(patient.patient_id, entry.get("sql") or "")
    if missing:
        out["limitations"].append(f"This patient's records were extracted without {', '.join(missing)} (an older "
                                  "extraction schema).")
    if patient.incomplete:
        out["limitations"].append(_incomplete(patient).capitalize() + ".")
    return out


def check_criteria_with_sql(
    records,
    compiled: Sequence[dict],
    *,
    patient_ids: Sequence[str] | None = None,
    reference_dates: Mapping[str, object] | str | None = None,
    coverage: Mapping[str, Mapping] | None = None,
    config: MMAIConfig | None = None,
    limits: WorkupSQLLimits | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run every accepted criteria query once over the cohort's records: one result per patient and line, as
    :func:`~.evaluate.check_criteria_with_extractions` returns. No model call."""
    limits = limits or agent_limits(config or load_default_preset())
    database = build_extraction_database(records, coverage=coverage, patient_ids=patient_ids,
                                         reference_dates=reference_dates)
    ids = database.patient_ids
    progress = progress_callback or (lambda done, total: None)
    runnable = [n for n, e in enumerate(compiled) if e.get("status") == "compiled"
                and e.get("expressible") is not False and isinstance(e.get("sql"), str)]
    rows: dict[int, dict] = {}
    failures: dict[int, str] = {}
    progress(0, len(compiled))
    if runnable and ids:
        with SQLSandbox(database=database, limits=limits.sandbox()) as box:
            for done, number in enumerate(runnable, 1):
                check_cancelled()
                try:
                    check = run_final_query(box, compiled[number]["sql"], len(ids), limits,
                                            task=CRITERIA_SQL_TASK)
                except SQLSandboxError:
                    failures[number] = "the isolated SQL worker failed while running this query."
                    break
                if check.problems:
                    failures[number] = ("the SQL query failed on this cohort's records."
                                        if check.problems[0].startswith("The query failed")
                                        else "the SQL query's output broke its contract on this cohort's records.")
                else:
                    rows[number] = check.rows
                progress(done, len(compiled))
    for number in runnable:
        if number not in rows and number not in failures:
            failures[number] = "the isolated SQL worker failed before this query ran."
    reference = {pid: row["reference_date"] for pid, row in
                 zip(ids, database.tables["patients"].to_pylist(), strict=True)}
    patients = []
    for key, pid in enumerate(ids, 1):
        patient = database.data.get(pid) or PatientData(pid)
        day = reference[pid].isoformat() if reference[pid] else None
        lines = []
        for number, entry in enumerate(compiled):
            result = _assess((rows.get(number) or {}).get(key), entry, patient, database,
                             failure=failures.get(number), reference=day)
            lines.append({"number": entry["line"].get("number"), "text": entry["line"].get("text"),
                          "boilerplate": bool(entry["line"].get("boilerplate")), **result,
                          "query_key": entry.get("cache_key"), "sql": entry.get("sql")})
        patients.append({"patient_id": pid, "lines": lines, "counts": line_counts(lines),
                         "coverage": patient.coverage()})
    progress(len(compiled), len(compiled))
    return {"patients": patients, "aggregate": aggregate_structured_criteria(patients, compiled),
            "notice": NOTICE, "method": METHOD}


__all__ = [
    "CRITERIA_SQL_REASONS",
    "CRITERIA_SQL_TASK",
    "METHOD",
    "check_criteria_with_sql",
    "compile_criteria_sql",
    "criteria_sql_cache_key",
    "criteria_sql_contract",
]
