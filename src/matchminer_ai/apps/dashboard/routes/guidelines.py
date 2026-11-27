"""The Patient tab's notes and guideline workup, and the model connections of every LLM task.

- ``/api/patient/*``: a session's patient notes (upload, synthetic examples in synthetic mode), summarization and
  the saved workspace.
- ``/api/guidelines/*``: the guideline library, guideline-space retrieval for a summary, and workup reviews
  (with an extraction-cost estimate that makes no model calls).
- ``/api/models*``: the session's connection per LLM task (``guidelines.settings.LLM_TASKS``) and connection
  checks. Vertex, Claude and OpenRouter destinations are built on the server, never from browser input; Claude
  through the Databricks gateway serves every task (since 2026-11-21). With ``--local-server-only`` both refuse
  every connection but the app's ready local model server (and OpenRouter with ``--allow-openrouter``;
  ``GuidelineState.refusal``).
- ``/api/models/openrouter-key``: a session's own OpenRouter key (synthetic mode only), kept in the app's memory;
  no response ever contains it (``GuidelineState.set_openrouter_key``). Without one, a session's OpenRouter
  connections use the hosted key when the app has one (``apps.dashboard.hosted_openrouter``), for its model only.

Long work runs as session jobs (``guidelines.state.GuidelineState.jobs``) polled through ``/api/jobs/{id}``.
"""
from __future__ import annotations

import asyncio
import os
from typing import Literal

import httpx
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from matchminer_ai.llm import connections, openrouter

from ..core.errors import InputError
from ..guidelines.services import parse_upload, synthetic_notes
from ..guidelines.settings import (
    CLAUDE_BACKENDS, CLAUDE_ENDPOINT_NAME, VERTEX_BACKENDS, VERTEX_LABELS, claude_backends,
    databricks_profile, normalize_endpoint, vertex_profiles, vertex_project,
)
from ..guidelines.state import GuidelineState
from ..llm_decisions import search_connection
from .common import owner as session_owner

LLMBackend = Literal["local", "gemini4_vertex", "gemini_vertex", "gemma_vertex", "claude_databricks", "openrouter"]
LLMTask = Literal["summarize", "workup", "workup_sql", "workup_full_backup", "workup_agent_backup", "good_option",
                  "llm_decisions", "assistant", "criteria_labels"]
EXTRACTION_METHODS = {"extractions", "extraction_sql"}


def claude_endpoint_name(value):
    if not CLAUDE_ENDPOINT_NAME.fullmatch(value or ""):
        raise ValueError("Enter the full three-part Databricks endpoint name, such as "
                         "ai_gateway_model_services_prod.<team>_<project>.<team>_<project>-claude-sonnet-5-5.")
    return value


class LLMSettings(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    llm_backend: LLMBackend = "local"
    endpoint: str = Field(min_length=1, max_length=2048)
    model: str = Field(min_length=1, max_length=512)
    tokenizer: str = Field(default="", max_length=512)
    task: LLMTask | None = None
    apply_to_all: bool = False
    reasoning: bool = False
    backup_enabled: bool = False
    failure_threshold: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def backup_task(self):
        if self.apply_to_all and self.task is None:
            raise ValueError("Choose a source task when applying a model to all tasks.")
        if self.backup_enabled and self.task not in {"workup_full_backup", "workup_agent_backup"}:
            raise ValueError("Enable backup only for a concordance backup task.")
        if self.llm_backend in CLAUDE_BACKENDS:
            claude_endpoint_name(self.model)
        if self.llm_backend == "openrouter":
            openrouter.valid_model(self.model)
        return self

    @field_validator("endpoint")
    @classmethod
    def endpoint_url(cls, value):
        return normalize_endpoint(value)


class EndpointCheck(BaseModel):
    task: LLMTask | None = None
    llm_backend: LLMBackend | None = None
    endpoint: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=512)

    @field_validator("endpoint")
    @classmethod
    def endpoint_url(cls, value):
        return normalize_endpoint(value) if value is not None else None


class Note(BaseModel):
    note_text: str = Field(min_length=1, max_length=2_000_000)
    note_date: str | None = None
    note_type: str | None = Field(default=None, max_length=1000)


class NotesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2_000_000)
    notes: list[Note] | None = Field(default=None, max_length=5000)


class SearchRequest(BaseModel):
    summary: str = Field(min_length=1, max_length=100_000)
    catalogs: list[str] = Field(default_factory=list, max_length=100)
    top_n: int = Field(default=1, ge=1, le=20)
    candidate_k: int = Field(default=5, ge=1, le=100)
    # Who ranks TrialSpace's candidates: the decision model, or the LLM decision checks' connection.
    scorer: Literal["decision_model", "llm"] = "decision_model"
    confirm: bool = False  # the user confirmed a large LLM run's estimate

    @model_validator(mode="after")
    def check(self):
        if not self.summary.strip() or self.candidate_k < self.top_n:
            raise ValueError("Supply a summary and candidate_k >= top_n")
        return self


class WorkupRequest(NotesRequest):
    retrieval_job_id: str = Field(min_length=1, max_length=100)
    space_index: int = Field(ge=0, le=19)
    # extractions: compiled extraction queries; extraction_sql: the SQL agent's queries.
    method: Literal["full_notes", "agentic", "extractions", "extraction_sql"] = "full_notes"
    # Extraction methods only: extract each note, or also build a patient-level record.
    extraction_levels: Literal["document", "document_and_patient"] = "document"
    # Extraction methods only: the corpus patient whose notes these are, so the mode's extraction store's records
    # serve the review.
    patient_id: str | None = Field(default=None, max_length=200)


class ExtractionCostRequest(NotesRequest):
    retrieval_job_id: str = Field(min_length=1, max_length=100)
    space_index: int = Field(ge=0, le=19)
    extraction_levels: Literal["document", "document_and_patient"] = "document"
    method: Literal["extractions", "extraction_sql"] = "extractions"
    patient_id: str | None = Field(default=None, max_length=200)


def router(g: GuidelineState) -> APIRouter:
    api = APIRouter()
    settings, jobs, catalogs = g.settings, g.jobs, g.catalogs

    def submit(owner, kind, label, run, restore=None):
        job = jobs.submit(kind, label, run, owner=owner, restore=restore, device=settings.device)
        return {"job": job.describe(with_result=False)}

    def finished(owner, job_id, kind):
        job = jobs.get(job_id, owner) if job_id else None
        return job if job is not None and job.kind == kind and job.status == "done" else None

    # ------------------------------------------------------------------ model connections

    @api.get("/api/models")
    def models(request: Request):
        owner = session_owner(request)
        return g.config_view(g.current_settings(owner), owner)

    @api.post("/api/models")
    def update_models(body: LLMSettings, request: Request):
        values = body.model_dump(exclude={"task", "apply_to_all"})
        values["tokenizer"] = body.tokenizer or body.model
        if body.llm_backend in VERTEX_BACKENDS:
            # The Google destination is built on the server; ADC never goes to a browser-supplied endpoint.
            values.update(vertex_profiles()[body.llm_backend])
        elif body.llm_backend in CLAUDE_BACKENDS:
            # The same for Databricks tokens: the server fixes the gateway URL; the browser picks only the
            # gateway's endpoint name.
            profile = claude_backends()[body.llm_backend]
            values.update(endpoint=profile["endpoint"], tokenizer=profile["tokenizer"])
        elif body.llm_backend == "openrouter":
            # The server fixes OpenRouter's endpoint; the session's key goes nowhere else.
            values.update(endpoint=openrouter.ENDPOINT,
                          tokenizer=body.tokenizer or openrouter.model_settings(body.model)["tokenizer"])
        owner = session_owner(request)
        updated = g.update(owner, values, task=body.task, apply_to_all=body.apply_to_all)
        return g.config_view(updated, owner)

    @api.post("/api/models/check")
    async def check_model(body: EndpointCheck, request: Request):
        selected = g.current_settings(session_owner(request))
        if body.task:
            selected = selected.for_task(body.task)
        backend = body.llm_backend or selected.llm_backend
        # --local-server-only: no request goes to any endpoint but the app's own local server.
        reason = g.refusal({"llm_backend": backend, "endpoint": body.endpoint or selected.endpoint,
                            "model": body.model if body.model is not None else selected.model})
        if reason:
            raise InputError(reason)
        if backend == "openrouter":
            model = body.model if body.model is not None else selected.model
            try:
                model = openrouter.valid_model(model)
            except ValueError as exc:
                return {"available": False, "endpoint": openrouter.ENDPOINT, "models": [], "model": None,
                        "error": str(exc)}
            # The key's own record and the public model list; no prompt (the hosted key: its model only, and its
            # quota's reading of the key's record). Errors are fixed texts.
            result = await asyncio.to_thread(openrouter.check, selected.api_key_ref, model)
            return {"available": result["available"], "endpoint": openrouter.ENDPOINT,
                    "models": [model] if result["available"] else [],
                    "model": model if result["available"] else None, "error": result["error"]}
        if backend in CLAUDE_BACKENDS:
            from matchminer_ai.llm.anthropic_messages import check_gateway

            profile = claude_backends()[backend]
            model = body.model if body.model is not None else selected.model
            try:
                claude_endpoint_name(model)
            except ValueError as exc:
                return {"available": False, "endpoint": profile["endpoint"], "models": [], "model": None,
                        "error": str(exc)}
            # A short request with no patient data; errors are fixed texts.
            result = await asyncio.to_thread(check_gateway, profile["endpoint"], model, profile=databricks_profile(),
                                             timeout=60)
            return {"available": result["available"], "endpoint": profile["endpoint"],
                    "models": [model] if result["available"] else [],
                    "model": model if result["available"] else None, "error": result["error"]}
        if backend in VERTEX_BACKENDS:
            profile = vertex_profiles()[backend]
            missing = connections.vertex_credentials_error()  # reported at once, not waited on
            if missing:
                return {"available": False, "endpoint": profile["endpoint"], "models": [], "model": None,
                        "error": missing}
            available = await connections.vertex_available(backend, timeout=30)
            return {"available": available, "endpoint": profile["endpoint"],
                    "models": [profile["model"]] if available else [],
                    "model": profile["model"] if available else None,
                    "error": None if available else
                    f"Could not connect to {VERTEX_LABELS[backend]}. Check this server's Google Application Default "
                    f"Credentials and model access in project {vertex_project()}."}
        endpoint = body.endpoint or selected.endpoint
        model = body.model if body.model is not None else selected.model
        try:
            names = [m["id"] for m in await connections.served_models(endpoint, timeout=10,
                                                                      api_key=os.environ.get("OPENAI_API_KEY"))]
            return {"available": model in names, "models": names, "endpoint": endpoint,
                    "model": model if model in names else next(iter(names), None),
                    "error": None if names else "The endpoint returned no served models."}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {"available": False, "models": [], "endpoint": endpoint, "model": None,
                    "error": "Could not read the endpoint's model list. Check the address and server."}

    @api.post("/api/models/openrouter-key")
    async def set_openrouter_key(request: Request):
        # Read by hand: a validation error would echo the body (the key) back.
        try:
            payload = await request.json()
        except ValueError:
            payload = None
        key = payload.get("key") if isinstance(payload, dict) else None
        if not isinstance(key, str):
            raise InputError(openrouter.BAD_KEY)
        owner = session_owner(request)
        g.set_openrouter_key(owner, key)  # fixed texts only
        return g.config_view(g.current_settings(owner), owner)

    @api.delete("/api/models/openrouter-key")
    def forget_openrouter_key(request: Request):
        owner = session_owner(request)
        g.forget_openrouter_key(owner)
        return g.config_view(g.current_settings(owner), owner)

    # ------------------------------------------------------------------ the patient's notes

    @api.get("/api/patient/synthetic-notes")
    def example_notes(source: str = "demo"):
        try:
            return synthetic_notes(settings, source)
        except InputError:
            raise
        except Exception:
            raise InputError("Could not load the synthetic CSV. Check the sample path or HF connection.") from None

    @api.post("/api/patient/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(settings.max_upload + 1)
        await file.close()
        if len(data) > settings.max_upload:
            raise InputError(f"File exceeds the {settings.max_upload // 1024**2} MB upload limit.")
        if not data:
            raise InputError("The uploaded file is empty.")
        try:
            # A PDF may need OCR: off the event loop.
            result = await asyncio.to_thread(parse_upload, file.filename or "", data)
        except InputError:
            raise
        except Exception:
            raise InputError("This file could not be read. Check its format and try again.") from None
        if len(result["text"]) > 2_000_000:
            raise InputError("Extracted notes exceed the 2 million character limit. Split the record.")
        return result

    def source_notes(body):
        if not body.text.strip():
            raise InputError("Enter patient notes first.")
        notes = ([n.model_dump(exclude={"note_type"} if n.note_type is None else set()) for n in body.notes]
                 if body.notes else None)
        if notes and sum(len(n["note_text"]) for n in notes) > 2_000_000:
            raise InputError("Too much note text.")
        return notes

    @api.post("/api/patient/summarize", status_code=202)
    def summarize(body: NotesRequest, request: Request):
        owner = session_owner(request)
        notes = source_notes(body)
        worker = g.patient_inference(owner, "summarize")

        def summary_job(progress):
            result = worker.summarize(body.text, notes, progress)
            result.update(notes_input={"text": body.text, "notes": notes})
            return result

        return submit(owner, "summarize", "patient summary", summary_job,
                      restore={"notes_input": {"text": body.text, "notes": notes}})

    @api.get("/api/patient/session")
    def workspace(request: Request):
        latest = jobs.latest(session_owner(request), exclude_prefix="population-")
        return {"job": latest.describe() if latest else None}

    @api.delete("/api/patient/session", status_code=204)
    def clear(request: Request):
        g.clear(session_owner(request))
        return Response(status_code=204)

    # ------------------------------------------------------------------ guideline spaces and workup

    @api.get("/api/guidelines/catalogs")
    def library(refresh: bool = False):
        rows = catalogs.list(refresh)
        return {"catalogs": [{k: v for k, v in row.items() if k != "path"} for row in rows],
                "spaces": sum(row["spaces"] for row in rows)}

    @api.post("/api/guidelines/retrieve", status_code=202)
    def retrieve(body: SearchRequest, request: Request):
        owner = session_owner(request)
        paths = catalogs.resolve(body.catalogs)
        worker = g.inference
        # LLM ranking: every TrialSpace candidate is asked about, then the top N kept (llm_decisions).
        llm = search_connection(g, owner, body.candidate_k, body.confirm) if body.scorer == "llm" else None
        search = {"catalogs": body.catalogs, "top_n": body.top_n, "candidate_k": body.candidate_k,
                  "scorer": body.scorer}

        def retrieve_job(progress):
            result = worker.retrieve(body.summary, paths, body.candidate_k if llm else body.top_n, body.candidate_k,
                                     progress)
            if llm:
                result = g.llm_decisions.rank_guideline_spaces(owner, result, body.top_n, llm, progress)
            result["search"] = search
            return result

        return submit(owner, "retrieve", "guideline spaces", retrieve_job,
                      restore={"summary": body.summary, "search": search})

    def workup_space(owner, retrieval_job_id, space_index):
        retrieval = finished(owner, retrieval_job_id, "retrieve")
        if retrieval is None:
            raise InputError("Find guideline spaces in this browser session first.")
        spaces = retrieval.result["spaces"]
        if space_index >= len(spaces):
            raise InputError("The selected guideline space is unavailable.")
        space = spaces[space_index]
        if not space["diagnostic_workup"]:
            raise InputError("This space has no cataloged diagnostic/workup items.")
        return retrieval, space

    @api.post("/api/guidelines/workup/extraction-cost")
    def extraction_cost(body: ExtractionCostRequest, request: Request):
        """What an extraction-based review would send; makes no model calls."""
        owner = session_owner(request)
        notes = source_notes(body)
        _, space = workup_space(owner, body.retrieval_job_id, body.space_index)
        worker = g.patient_inference(owner, "workup")
        return worker.extraction_cost(body.text, notes, space, body.extraction_levels, method=body.method,
                                      patient_id=body.patient_id)

    @api.post("/api/guidelines/workup", status_code=202)
    def workup(body: WorkupRequest, request: Request):
        owner = session_owner(request)
        notes = source_notes(body)
        retrieval, space = workup_space(owner, body.retrieval_job_id, body.space_index)
        worker = g.patient_inference(owner, "workup")
        result = retrieval.result

        def review_job(progress):
            options = {"patient_summary": result.get("summary")} if body.method == "agentic" else {}
            if body.method in EXTRACTION_METHODS:
                options.update(extraction_levels=body.extraction_levels, patient_id=body.patient_id)
            review = worker.review_workup(body.text, notes, space, progress, method=body.method, **options)
            return {"review": review, "space_index": body.space_index, "method": body.method,
                    "extraction_levels": body.extraction_levels,
                    "retrieval_job_id": body.retrieval_job_id, "retrieval": result,
                    "notes_input": {"text": body.text, "notes": notes}}

        return submit(owner, "workup", "workup review", review_job, restore={
            "retrieval": result, "retrieval_job_id": body.retrieval_job_id,
            "notes_input": {"text": body.text, "notes": notes}, "method": body.method,
            "space_index": body.space_index})

    return api
