"""Structured extraction records (the Patient tab's Structured data) and criteria screening on them.

``POST /api/patient/records`` returns the records of the patient in view: the mode's store when it holds the
corpus patient, else the session's on-demand extraction of the workspace notes, else an estimate of what
extracting them would send. ``POST /api/patient/records/extract`` extracts the workspace notes (a job).
``POST /api/criteria-records`` checks one cohort's criteria lines on one patient's records, and
``POST /api/cohorts/{cohort_id}/screen-records`` on every candidate the store holds (jobs; each line's query is
compiled once and cached per writing model).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..core.errors import InputError
from ..guidelines import records as rec
from ..guidelines.settings import SQL_TASK
from ..state import DashState
from .common import owner as session_owner
from .guidelines import Note

Levels = Literal["document", "document_and_patient"]
Engine = Literal["queries", "sql"]


class RecordsRequest(BaseModel):
    #: A corpus patient (its stored records); else the workspace notes (an on-demand extraction).
    patient_id: str | None = Field(default=None, max_length=200)
    text: str | None = Field(default=None, max_length=2_000_000)
    notes: list[Note] | None = Field(default=None, max_length=5000)
    levels: Levels = "document_and_patient"


class CriteriaRecordsRequest(RecordsRequest):
    cohort_id: str = Field(min_length=1, max_length=200)
    engine: Engine = "queries"


class ScreenRequest(BaseModel):
    patient_ids: list[str] = Field(min_length=1, max_length=1000)
    engine: Engine = "queries"


def router(state: DashState) -> APIRouter:
    api = APIRouter()
    g = state.guidelines

    def configs(owner):
        """(extraction config, writer config): the session's concordance and query-writing connections."""
        return (g.task_settings(owner, "workup").extraction_config(),
                g.task_settings(owner, SQL_TASK).extraction_config())

    def notes_of(body):
        return [n.model_dump() for n in body.notes] if body.notes else None

    def corpus_numbers(pid):
        entity = state.patient_entity(pid)
        if entity["custom"]:
            return {}
        return rec.note_numbers(rec.corpus_frame(pid, state.notes.notes(pid)))

    def locate(body, owner, config):
        """``(documents, patients, source, numbers, patient_key)`` of the patient in view, or None."""
        if body.patient_id and not body.patient_id.startswith("entered-"):
            stored = rec.store_records(g.records, body.patient_id)
            if stored is not None:
                documents, patients, store = stored
                return documents, patients, rec.source_of(store), corpus_numbers(body.patient_id), body.patient_id
        frame = rec.workspace_frame(body.text, notes_of(body))
        # The levels asked for first, then the other (a Guidelines review may have extracted documents only).
        saved = None
        for levels in dict.fromkeys((body.levels, "document_and_patient", "document")):
            saved = saved or rec.session_records(g.session_extractions, owner, frame, levels, config)
        if saved is not None:
            documents, patients = rec.extraction_frames(saved)
            source = {"kind": "session", **saved["metadata"], "model": rec.model_label(saved["metadata"].get("model"))}
            return documents, patients, source, rec.note_numbers(frame), rec.WORKSPACE
        return None

    @api.get("/api/records/status")
    def status():
        return {"store": g.records.describe(), "engines": list(rec.ENGINES)}

    @api.post("/api/patient/records")
    def patient_records(body: RecordsRequest, request: Request):
        owner = session_owner(request)
        config, _ = configs(owner)
        found = locate(body, owner, config)
        if found is not None:
            documents, patients, source, numbers, _ = found
            return {"source": source, "view": rec.records_view(documents, patients, numbers), "estimate": None}
        frame = rec.workspace_frame(body.text, notes_of(body))
        return {"source": None, "view": None,
                "estimate": rec.extraction_estimate(frame, body.levels, config) if frame is not None else None,
                "store": g.records.describe()}

    @api.post("/api/patient/records/extract", status_code=202)
    def extract(body: RecordsRequest, request: Request):
        owner = session_owner(request)
        config, _ = configs(owner)
        frame = rec.workspace_frame(body.text, notes_of(body))
        if frame is None:
            raise InputError("Load or paste this patient's clinical notes first.")

        def run(progress):
            extraction = rec.extract_workspace(frame, body.levels, config, g.session_extractions, owner, progress)
            documents, patients = rec.extraction_frames(extraction)
            return {"source": {"kind": "session", **extraction["metadata"],
                               "model": rec.model_label(extraction["metadata"].get("model"))},
                    "view": rec.records_view(documents, patients, rec.note_numbers(frame))}

        job = g.jobs.submit("records", f"structured records of {len(frame)} notes", run, owner=owner,
                            device=g.settings.device)
        return {"job": job.describe(with_result=False)}

    @api.post("/api/criteria-records", status_code=202)
    def criteria_records(body: CriteriaRecordsRequest, request: Request):
        owner = session_owner(request)
        config, writer = configs(owner)
        cohort = state.cohort(body.cohort_id)
        rec.cohort_lines(cohort)
        found = locate(body, owner, config)
        if found is None:
            raise InputError("This patient has no structured records yet: extract them on the Patient tab's "
                             "Structured data view first.")
        documents, patients, source, numbers, key = found
        store = g.records.get()

        def run(progress):
            compiled, summary = rec.compile_cohort_criteria(cohort, body.engine, settings=g.settings, config=writer,
                                                            store=store, progress=progress)
            frames = [f for f in (documents, patients) if f is not None and len(f)]
            report = rec.run_criteria(frames, compiled, body.engine, [key], writer, progress)
            patient = report["patients"][0]
            return {"cohort_id": cohort["id"], "engine": body.engine, "source": source,
                    "lines": rec.map_evidence(patient["lines"], numbers), "counts": patient["counts"],
                    "coverage": patient["coverage"], "queries": rec.compiled_view(compiled), "compile": summary,
                    "notice": report["notice"]}

        job = g.jobs.submit("criteria_records", f"{cohort['id']} structured criteria", run, owner=owner,
                            device=g.settings.device)
        return {"job": job.describe(with_result=False)}

    @api.post("/api/cohorts/{cohort_id}/screen-records", status_code=202)
    def screen(cohort_id: str, body: ScreenRequest, request: Request):
        owner = session_owner(request)
        _, writer = configs(owner)
        cohort = state.cohort(cohort_id)
        rec.cohort_lines(cohort)
        store = g.records.get()
        if store is None:
            raise InputError("No extraction record store is available in this mode: "
                             + (g.records.error or "build one with preprocess_extractions."))
        ids = list(dict.fromkeys(str(p) for p in body.patient_ids))

        def run(progress):
            progress(f"Reading stored records of {len(ids)} candidates")
            documents, patients = store.records(ids)
            held = sorted(set(documents.patient_id if documents is not None else [])
                          | set(patients.patient_id if patients is not None else []))
            compiled, summary = rec.compile_cohort_criteria(cohort, body.engine, settings=g.settings, config=writer,
                                                            store=store, progress=progress)
            frames = [f for f in (documents, patients) if f is not None and len(f)]
            report = rec.run_criteria(frames, compiled, body.engine, held, writer, progress) if held else \
                {"patients": [], "aggregate": [], "notice": ""}
            by_patient = {p["patient_id"]: {"counts": p["counts"],
                                            "failed": [{"number": line["number"], "text": line["text"],
                                                        "bottom_line": line["bottom_line"]}
                                                       for line in p["lines"] if line["clearly_fails"]]}
                          for p in report["patients"]}
            return {"cohort_id": cohort["id"], "engine": body.engine, "store": rec.source_of(store),
                    "patients": by_patient, "without_records": [p for p in ids if p not in by_patient],
                    "aggregate": report["aggregate"], "queries": rec.compiled_view(compiled), "compile": summary,
                    "notice": report["notice"]}

        job = g.jobs.submit("screen_records", f"{cohort['id']} · {len(ids)} candidates", run, owner=owner,
                            device=g.settings.device)
        return {"job": job.describe(with_result=False)}

    return api
