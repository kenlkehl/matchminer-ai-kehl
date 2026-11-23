"""Structured extraction records in the dashboard: one patient's records, and criteria screening on them.

* **Records** of the patient in view come from the mode's extraction store when it holds the patient
  (``dashboard.<mode>.extraction_records``, built offline by ``preprocess_extractions``), else from the session's
  on-demand extraction of the workspace notes (the same records the Guidelines extraction review uses).
* **Criteria screening** compiles each criteria line of a cohort once (compiled criterion queries or the SQL agent,
  :mod:`matchminer_ai.matching.structured_criteria`), cached per writing model, then runs it in code on patients'
  records: one patient (the Patient tab, a Clinical trial management patient dialog) or every candidate with stored records
  (Clinical trial management Find patients).
"""
from __future__ import annotations

import copy
import hashlib
import threading
from pathlib import Path

import pandas as pd

from ..core.errors import InputError
from .extraction_workup import (
    LEVELS,
    QueryCache,
    SessionExtractions,
    estimate_cost,
    exploration_database,
    extract_on_demand,
    notes_frame,
    open_store,
)

WORKSPACE = "workspace-patient"
ENGINES = ("queries", "sql")
#: Item types in display order, with their labels.
TYPE_LABELS = {
    "diagnosis": "Diagnoses", "biomarker": "Biomarkers", "treatment": "Treatments", "procedure": "Procedures",
    "imaging_study": "Imaging", "lab_test": "Labs", "performance_status": "Performance status",
    "disease_assessment": "Disease assessments", "event": "Events", "counseling": "Counseling",
}
_SKIP = {"evidence", "item_type", "item_id"}
#: Item types extracted note by note only (schema v4): the Patient level view merges their note items in code.
NOTE_LEVEL_TYPES = ("imaging_study", "lab_test", "counseling")
#: Item types that are time points: the disease course has a row for each note's (the others, one per fact).
POINT_TYPES = ("disease_assessment", "performance_status")
#: The disease course's tag of each item type: its categorical field (performance status: scale and score).
COURSE_TAGS = {"diagnosis": "extent_at_diagnosis", "biomarker": "result", "treatment": "treatment_type",
               "procedure": "specimen_result", "event": "event_type", "disease_assessment": "disease_status",
               "imaging_study": "modality", "lab_test": "result_flag", "counseling": "counseling_kind"}


# ---------------------------------------------------------------------------
# Stores and caches
# ---------------------------------------------------------------------------


class StoreHandle:
    """The mode's extraction store, opened on first use and reopened when its manifest changes."""

    def __init__(self, settings):
        self.settings, self.lock = settings, threading.Lock()
        self.store, self.stamp, self.error = None, None, None
        self._ids: tuple | None = None

    def get(self):
        path = Path(self.settings.extraction_records) / "manifest.json"
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        with self.lock:
            if stamp is not None and stamp == self.stamp:
                return self.store
            try:
                self.store, self.error = open_store(self.settings), None
            except InputError as exc:
                self.store, self.error = None, str(exc)
            self.stamp = stamp
            return self.store

    def patient_ids(self) -> frozenset[str]:
        """The patients the store holds (read once per manifest)."""
        import pyarrow.dataset as ds

        from .extraction_workup import DOCUMENT_FILE, PATIENT_FILE

        store = self.get()
        if store is None:
            return frozenset()
        with self.lock:
            if self._ids is not None and self._ids[0] == self.stamp:
                return self._ids[1]
        ids: set[str] = set()
        for name in (DOCUMENT_FILE, PATIENT_FILE):
            file = store.path / name
            if file.is_file():
                column = ds.dataset(file, format="parquet").to_table(columns=["patient_id"]).column(0)
                ids.update(str(v) for v in column.unique().to_pylist())
        with self.lock:
            self._ids = (self.stamp, frozenset(ids))
        return self._ids[1]

    def describe(self) -> dict:
        store = self.get()
        if store is None:
            return {"available": False, "error": self.error or "No extraction record store is configured."}
        manifest = store.manifest
        extraction = manifest.get("extraction") or {}
        return {"available": True, "name": store.path.name, "created_utc": manifest.get("created_utc"),
                "patients": (manifest.get("counts") or {}).get("patients", 0), "levels": manifest.get("levels"),
                "schema_version": extraction.get("schema_version"), "model": model_label(extraction.get("model"))}


def criteria_query_cache(settings, config):
    """Compiled criterion queries, across sessions, apart for each writing model (``criteria_query_compile``)."""
    model = str((config.criteria_query_compile.get("remote") or {}).get("model_name") or "")
    return QueryCache(Path(settings.cache) / "criteria_queries"
                      / f"writer-{hashlib.sha256(model.encode()).hexdigest()[:16]}")


def criteria_sql_cache(settings):
    """Accepted criteria SQL queries (their keys include the agent's model)."""
    return QueryCache(Path(settings.cache) / "criteria_sql")


# ---------------------------------------------------------------------------
# A patient's records, for display
# ---------------------------------------------------------------------------


def corpus_frame(pid: str, notes: list[dict]) -> pd.DataFrame:
    """A corpus patient's notes ({date, type, text}, the order the app shows) as an extraction note table, blank
    notes left out (as the Patient tab and the store builds leave them out)."""
    return pd.DataFrame([{"patient_id": pid, "note_text": n.get("text") or "", "note_date": n.get("date"),
                          "note_type": n.get("type")} for n in notes if (n.get("text") or "").strip()],
                        columns=["patient_id", "note_text", "note_date", "note_type"])


def note_numbers(frame: pd.DataFrame) -> dict:
    """{note_id: 1-based position}: extraction note ids of ``frame``'s notes, in its order."""
    from matchminer_ai.patients.extraction.notes import note_ids

    if frame.empty:
        return {}
    numbers = {}
    for number, note in enumerate(note_ids(frame), 1):
        numbers.setdefault(note, number)
    return numbers


def _plain(value):
    if isinstance(value, dict) and {"value", "precision", "source"} <= set(value):
        return value["value"]
    return value


def _item(item, *, level, note_id=None, note_date=None, document_type=None, numbers):
    from matchminer_ai.patients.workup_queries.evaluate import _date_text, describe_item, item_view

    view = item_view(item, level=level, note_id=note_id, note_date=note_date, document_type=document_type)
    return {"item_id": item.get("item_id"), "type": item["item_type"], "label": describe_item(view),
            "date": view.date_value, "date_bound": view.bound, "date_text": _date_text(view), "status": view.status,
            "fields": {k: _plain(v) for k, v in item.items() if k not in _SKIP}, "raw": item,
            "evidence": [{k: e.get(k) for k in ("note_id", "note_date", "quote", "start", "end")}
                         | {"note_number": numbers.get(e.get("note_id"))} for e in item.get("evidence") or []]}


def _records(frame):
    if frame is None or not len(frame):
        return []
    return [r for r in frame.to_dict("records")]


def records_view(documents, patients, numbers: dict) -> dict:
    """Patient-level items by type and document records by note, with evidence mapped to note numbers."""
    by_type: dict[str, list] = {}
    patient_rows = _records(patients)
    for row in patient_rows:
        record = row.get("record")
        if not isinstance(record, dict):
            continue
        for items in record["items"].values():
            for item in items:
                by_type.setdefault(item["item_type"], []).append(_item(item, level="patient", numbers=numbers))
    notes = []
    for row in sorted(_records(documents), key=lambda r: (str(r.get("note_date") or ""), str(r.get("note_id")))):
        record = row.get("record")
        items = []
        if isinstance(record, dict):
            for values in record["items"].values():
                items.extend(_item(item, level="document", note_id=row.get("note_id"),
                                   note_date=row.get("note_date"), document_type=record.get("document_type"),
                                   numbers=numbers) for item in values)
        notes.append({"note_id": row.get("note_id"), "note_number": numbers.get(row.get("note_id")),
                      "note_date": None if row.get("note_date") is None else str(row.get("note_date"))[:10],
                      "note_type": row.get("note_type"),
                      "document_type": (record or {}).get("document_type") if isinstance(record, dict) else None,
                      "status": row.get("extraction_status"), "items": items,
                      "record": record if isinstance(record, dict) else None})
    rows = _records(documents)
    merged = merged_note_items(rows, numbers, lambda t: t in NOTE_LEVEL_TYPES and t not in by_type) \
        if patient_rows else {}
    order = list(TYPE_LABELS)
    types = [{"type": t, "label": TYPE_LABELS.get(t, t.replace("_", " ").capitalize()),
              "items": by_type.get(t) or merged[t], "merged": t in merged}
             for t in sorted({*by_type, *merged}, key=lambda t: (order.index(t) if t in order else len(order), t))]
    statuses = [r.get("extraction_status") for r in rows]
    patient_record = patient_rows[0].get("record") if patient_rows else None
    return {
        "patient": {"available": bool(patient_rows), "types": types,
                    "status": patient_rows[0].get("extraction_status") if patient_rows else None,
                    "record": patient_record if isinstance(patient_record, dict) else None},
        "documents": notes,
        # Facts once each: the Patient level view's items, else (no patient-level record) the notes' merged.
        "disease_course": disease_course(
            notes, {**by_type, **merged} if patient_rows else
            merged_note_items(rows, numbers, lambda t: t not in POINT_TYPES),
            linked=bool(patient_rows), merged=set(merged) if patient_rows else None),
        "counts": {"notes": len(notes), "patient_items": sum(len(t["items"]) for t in types),
                   "document_items": sum(len(n["items"]) for n in notes),
                   "failed_notes": statuses.count("failed"), "partial_notes": statuses.count("partial")},
    }


def merged_note_items(rows, numbers: dict, keep) -> dict[str, list]:
    """The notes' items of the types ``keep`` accepts, one entry per distinct study, test or fact (the same
    clustering the workup and criteria queries use): the Patient level view's imaging, labs and counseling, and
    the disease course's facts without a patient-level record. No model call."""
    from matchminer_ai.patients.workup_queries.evaluate import cluster_items, item_view

    views: dict[str, list] = {}
    for row in rows:
        record = row.get("record")
        if not isinstance(record, dict):
            continue
        for values in record["items"].values():
            for item in values:
                if keep(item["item_type"]):
                    views.setdefault(item["item_type"], []).append(item_view(
                        item, level="document", note_id=row.get("note_id"), note_date=row.get("note_date"),
                        document_type=record.get("document_type")))
    merged = {}
    for kind, found in views.items():
        entries = []
        for cluster in cluster_items(found):
            rep = cluster.rep
            entry = _item(rep.item, level="document", note_id=rep.note_id, note_date=rep.note_date,
                          document_type=rep.document_type, numbers=numbers)
            seen, evidence = set(), []
            for member in cluster.members:
                for e in member.item.get("evidence") or []:
                    mark = (e.get("note_id"), e.get("start"), e.get("end"))
                    if mark not in seen:
                        seen.add(mark)
                        evidence.append({k: e.get(k) for k in ("note_id", "note_date", "quote", "start", "end")}
                                        | {"note_number": numbers.get(e.get("note_id"))})
            entry.update(evidence=evidence[:8], mentions=len(cluster.members),
                         raw=[m.item for m in cluster.members])
            entries.append(entry)
        entries.sort(key=lambda e: (e["date"] or "", e["label"]))
        merged[kind] = entries
    return merged


def _tag(item: dict):
    fields = item["fields"]
    if item["type"] == "performance_status":
        scale = {"ecog": "ECOG", "karnofsky": "KPS"}.get(fields.get("scale"), fields.get("scale") or "")
        return f"{scale} {fields['score']}".strip() if fields.get("score") is not None else None
    value = fields.get(COURSE_TAGS.get(item["type"], ""))
    return None if value in (None, "", "not_stated") else value


def disease_course(notes: list[dict], facts: dict[str, list], *, linked: bool, merged: set | None) -> dict:
    """The patient's course as one timeline of every item type (no model call): each fact once (``facts``, by
    type, the patient-level record's items or the notes' merged), and each note's time points (disease
    assessments, performance status). Each row has its type's tag (``COURSE_TAGS``) and the patient-level
    diagnoses it concerns (``cancer_ids``; a patient-level diagnosis is its own). ``merged``: the fact types merged
    from the notes (None: all of them). Rows are in date order, undated last."""
    diagnoses = facts.get("diagnosis", []) if linked else []
    cancers = [{"id": d["item_id"], "label": d["label"]} for d in diagnoses if d.get("item_id")]
    rows = []
    for kind, items in facts.items():
        if kind in POINT_TYPES:
            continue
        source = "notes" if merged is None or kind in merged else "patient"
        for item in items:
            own = kind == "diagnosis" and source == "patient" and item.get("item_id")
            rows.append({**item, "tag": _tag(item), "source": source, "note": None,
                         "cancer_ids": [item["item_id"]] if own else
                         list(item["fields"].get("cancer_ids") or []) if linked else []})
    for note in notes:
        for item in note["items"]:
            if item["type"] in POINT_TYPES:
                rows.append({**item, "tag": _tag(item), "source": "note",
                             "note": {k: note[k] for k in ("note_number", "note_type", "document_type")},
                             "cancer_ids": list(item["fields"].get("cancer_ids") or []) if linked else []})
    order = list(TYPE_LABELS)
    rows.sort(key=lambda r: (r["date"] is None, r["date"] or "", (r["note"] or {}).get("note_number") or 0,
                             order.index(r["type"]) if r["type"] in order else len(order), r["label"]))
    return {"cancers": cancers, "rows": rows}


# ---------------------------------------------------------------------------
# Where a patient's records come from
# ---------------------------------------------------------------------------


def store_records(handle: StoreHandle, pid: str | None):
    """``(documents, patients, store)`` of a corpus patient the store holds, else None."""
    store = handle.get() if pid else None
    if store is None:
        return None
    documents, patients = store.records([pid])
    if (documents is None or documents.empty) and (patients is None or patients.empty):
        return None
    return documents, patients, store


def source_of(store) -> dict:
    extraction = store.manifest.get("extraction") or {}
    return {"kind": "store", "name": store.path.name, "created_utc": store.manifest.get("created_utc"),
            "model": model_label(extraction.get("model")), "schema_version": extraction.get("schema_version"),
            "levels": store.manifest.get("levels")}


def workspace_frame(text, notes):
    if not notes and not (text or "").strip():
        return None
    return notes_frame(text, notes)


def session_records(sessions: SessionExtractions, owner, frame, levels, config):
    """The session's on-demand extraction of these notes with at least ``levels`` (the Structured data tab's or a
    Guidelines review's), or None."""
    if frame is None or sessions is None:
        return None
    return sessions.get(owner, SessionExtractions.key(frame, config), levels)


def extraction_frames(extraction):
    documents = pd.DataFrame(extraction["documents"])
    patients = pd.DataFrame(extraction["patients"]) if extraction.get("patients") else None
    return documents, patients


def extract_workspace(frame, levels, config, sessions, owner, progress):
    """Extract the workspace notes now (one request per note, plus patient-level chunks) and keep them in the
    session."""
    if levels not in LEVELS:
        raise InputError("Choose document-level or document and patient-level extraction.")
    extraction = extract_on_demand(frame, levels, config, progress)
    sessions.save(owner, SessionExtractions.key(frame, config), levels, extraction)
    return extraction


def model_label(name):
    """A model's display name (Claude endpoint names as "Claude Haiku 5.5")."""
    from ..llm import claude_name

    return ", ".join(claude_name(part.strip()) for part in str(name).split(",")) if name else name


def extraction_estimate(frame, levels, config) -> dict:
    cost = estimate_cost(frame, levels, compile_requests=0, cached_queries=0, items=0)
    cost.pop("typical_reasoning_tokens", None)
    return {**cost, "model": model_label(config.document_extraction["remote"]["model_name"])}


# ---------------------------------------------------------------------------
# Criteria screening
# ---------------------------------------------------------------------------


def cohort_lines(cohort: dict) -> list[dict]:
    lines = [dict(line) for line in cohort.get("items") or []]
    if not any(line.get("criterion_type") in ("inclusion", "exclusion") for line in lines):
        raise InputError("This cohort has no criteria typed inclusion or exclusion.")
    return lines


def compile_cohort_criteria(cohort, engine, *, settings, config, store, progress):
    """``(compiled, summary)``: one query per checkable line of the cohort (cached across sessions)."""
    from matchminer_ai.matching.structured_criteria import compile_criteria_queries, compile_criteria_sql

    if engine not in ENGINES:
        raise InputError("Choose compiled queries or the SQL agent.")
    lines = cohort_lines(cohort)
    if engine == "queries":
        compiled, metadata = compile_criteria_queries(
            lines, cohort_context=cohort.get("key") or "", config=config,
            cache=criteria_query_cache(settings, config), return_metadata=True,
            progress_callback=lambda done, total: progress(f"Compiling criteria queries · {done}/{total} ready"))
    else:
        if store is None:
            raise InputError("The SQL agent explores an extraction record store, and none is available.")
        database = exploration_database(settings, store, progress)
        compiled, metadata = compile_criteria_sql(
            lines, database=database, config=config, cohort_context=cohort.get("key") or "",
            cache=criteria_sql_cache(settings), return_metadata=True,
            progress_callback=lambda done, total: progress(f"SQL agent · {done}/{total} criteria ready"))
    summary = {k: metadata.get(k) for k in ("model", "compiled_by", "requests", "cached", "compiled", "distinct",
                                            "failed", "unchecked_lines")}
    summary["engine"] = engine
    summary["model"] = model_label(summary.get("model"))
    return compiled, summary


def run_criteria(records, compiled, engine, patient_ids, config, progress=lambda _: None):
    from matchminer_ai.matching.structured_criteria import (
        check_criteria_with_extractions,
        check_criteria_with_sql,
    )

    n = len(compiled)
    progress(f"Checking {n} {'criterion' if n == 1 else 'criteria'} on {len(patient_ids)} patients' records")
    if engine == "sql":
        return check_criteria_with_sql(records, compiled, patient_ids=patient_ids, config=config)
    return check_criteria_with_extractions(records, compiled, patient_ids=patient_ids)


def map_evidence(lines, numbers):
    for line in lines:
        for evidence in line.get("evidence") or []:
            evidence["note_number"] = numbers.get(evidence.get("note_id"))
    return lines


def compiled_view(compiled):
    """What the UI shows of each line's query (no patient data)."""
    keys = ("status", "cached", "model", "error", "expressible", "not_expressible_reason", "summary", "sql",
            "approximations")
    out = []
    for entry in compiled:
        view = {k: copy.deepcopy(entry.get(k)) for k in keys if k in entry}
        if entry.get("query") is not None:
            view["query"] = copy.deepcopy(entry["query"])
            view["summary"] = entry["query"].get("summary")
        view["number"] = entry["line"].get("number")
        out.append(view)
    return out


__all__ = [
    "ENGINES",
    "StoreHandle",
    "WORKSPACE",
    "compile_cohort_criteria",
    "corpus_frame",
    "criteria_query_cache",
    "criteria_sql_cache",
    "extract_workspace",
    "extraction_estimate",
    "note_numbers",
    "records_view",
    "run_criteria",
    "session_records",
    "store_records",
]
