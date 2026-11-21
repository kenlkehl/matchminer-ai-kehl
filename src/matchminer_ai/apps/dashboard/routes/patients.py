"""Patients and matching: the patient corpus, both matching directions, decision-model scoring and the LLM checks."""
from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..llm import Connection
from ..state import DashState
from .common import cohort_detail, cohort_row, owner, type_labels

DIRECTIONS = ("patients_for_cohort", "trials_for_patient")


class PatientsForCohort(BaseModel):
    cohort_id: str
    k: int | None = None
    cancer_type: str = ""
    sex: str = ""


class TrialsForPatient(BaseModel):
    patient_id: str | None = None
    summary: str | None = Field(default=None, max_length=100_000)  # an entered cancer summary (embedded with the round's encoder)
    boilerplate: str | None = Field(default=None, max_length=100_000)  # its common-exclusion evidence ("Other relevant history")
    k: int | None = None
    scope: str = "dashboard"  # "dashboard": the cohorts the dashboards show; "all": both versions of every trial
    per_trial: bool = True


class ScoreRequest(BaseModel):
    direction: str
    query_id: str
    target_ids: list[str] = Field(default_factory=list)


class PairRequest(BaseModel):
    patient_id: str
    cohort_id: str


class LLMCheckRequest(BaseModel):
    patient_id: str
    cohort_id: str | None = None  # the per-criterion check
    trial_id: str | None = None  # the GoodOption check
    rerun: bool = False


def router(state: DashState) -> APIRouter:
    api = APIRouter()

    @api.get("/api/patients")
    def patients(q: str = "", cancer_type: str = "", sex: str = "", offset: int = 0, limit: int = 50):
        mask = state.patient_mask(cancer_type, "", "")
        out = state.corpus.search_patients(q, "", sex, max(0, offset), max(1, min(limit, 500)), mask, "text")
        out["rows"] = [state.patient_summary(r["id"]) for r in out["rows"]]
        return out

    @api.get("/api/patients/{pid}")
    def patient(pid: str):
        entity = state.patient_entity(pid)
        return {**state.patient_summary(pid), "key": entity["key"], "boilerplate": entity["boilerplate"],
                "cancer_type_labels": type_labels(state.patient_summary(pid).get("cancer_types") or [])}

    @api.get("/api/patients/{pid}/notes")
    def patient_notes(pid: str):
        entity = state.patient_entity(pid)
        if entity["custom"]:
            return {"state": "none", "notes": [], "message": "An entered summary has no notes."}
        current = state.notes.ensure()
        if current != "ready":
            return {"state": current, "notes": [], "message": state.notes.error or "Indexing the notes (about 30 s, once)."}
        return {"state": "ready", "notes": state.notes.notes(pid)}

    def require_round():
        if state.vectors is None:
            raise HTTPException(503, "no valid TrialSpace round")

    def require_patients():
        require_round()
        if state.patient_vectors is None:
            raise HTTPException(503, state.patients_error or "no patient vectors")

    @api.post("/api/match/patients-for-cohort")
    def patients_for_cohort(request: PatientsForCohort):
        require_patients()
        started = time.monotonic()
        with state.round_lock:
            cohort = state.cohort(request.cohort_id)
            vector = state.cohort_vectors.vector(state.store, cohort["id"])
            if vector is None:
                job = state.embed_cohorts()
                return JSONResponse({"error": "This cohort's TrialSpace vector is still being made; try again shortly.",
                                     "pending": True, "job": job.describe(False) if job else None}, status_code=409)
            k = max(1, min(int(request.k or state.settings.default_k), state.settings.max_k))
            mask = state.patient_mask(request.cancer_type, request.sex, "")
            hits = state.patient_vectors.search(vector, "patients", k, mask)
            searched = len(state.corpus.patients) if mask is None else int(mask.sum())
        rows = [{"rank": rank, "patient": state.patient_summary(state.corpus.patients[i]["id"]), "cosine": round(c, 5)}
                for rank, (i, c) in enumerate(hits, 1)]
        pairs = [state.pair(r["patient"]["id"], cohort["id"]) for r in rows]
        for row, cached in zip(rows, state.decisions.cached_scores(pairs)):
            row["scores"] = cached
        return {"direction": "patients_for_cohort", "query": cohort_detail(state, cohort), "k": k, "searched": searched,
                "of": len(state.corpus.patients), "round": state.round.number,
                "search_ms": round((time.monotonic() - started) * 1000, 1), "results": rows}

    @api.post("/api/match/trials-for-patient")
    def trials_for_patient(request: TrialsForPatient):
        require_round()
        from matchminer_ai.matching.retrieval import stable_top_k

        started = time.monotonic()
        with state.round_lock:
            if request.patient_id:
                require_patients()
                entity = state.patient_entity(request.patient_id)
            elif request.summary and request.summary.strip():
                entity = state.add_custom_patient(request.summary, request.boilerplate or "")
            else:
                raise ValueError("choose a patient or enter a cancer summary")
            store = state.store
            scope = [c["id"] for c in store["cohorts"].values() if request.scope == "all" or c["primary"]]
            ids, matrix = state.cohort_vectors.matrix(store, scope)
            if not ids:
                raise HTTPException(409, "The trial cohorts' TrialSpace vectors are still being made; try again shortly.")
            scores = (matrix @ np.asarray(state.patient_vector(entity), dtype=np.float32)).astype(np.float64)
        k = max(1, min(int(request.k or state.settings.default_k), state.settings.max_k))
        order, seen = [], set()
        for i in stable_top_k(scores, len(ids)):
            trial_id = store["cohorts"][ids[i]]["trial_id"]
            if request.per_trial and trial_id in seen:
                continue
            seen.add(trial_id)
            order.append(i)
            if len(order) == k:
                break
        rows = [{"rank": rank, "cohort": cohort_row(state, store["cohorts"][ids[i]]), "cosine": round(float(scores[i]), 5)}
                for rank, i in enumerate(order, 1)]
        pairs = [state.pair(entity["id"], r["cohort"]["cohort_id"]) for r in rows]
        for row, cached in zip(rows, state.decisions.cached_scores(pairs)):
            row["scores"] = cached
        return {"direction": "trials_for_patient", "patient": {**state.patient_summary(entity["id"]), "key": entity["key"],
                                                                "boilerplate": entity["boilerplate"]},
                "k": k, "searched": len(ids), "of": len(scope), "round": state.round.number,
                "search_ms": round((time.monotonic() - started) * 1000, 1), "results": rows}

    @api.post("/api/score")
    def score(request: ScoreRequest):
        if request.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if not request.target_ids:
            raise ValueError("give target_ids")
        if len(request.target_ids) > state.settings.max_score_n:
            raise ValueError(f"score at most {state.settings.max_score_n} pairs at a time")
        if request.direction == "patients_for_cohort":
            pairs = [state.pair(t, request.query_id) for t in request.target_ids]
        else:
            pairs = [state.pair(request.query_id, t) for t in request.target_ids]
        job = state.jobs.submit("score", f"{len(pairs)} pairs",
                                lambda progress: {"scores": state.decisions.score(pairs, ("match", "boilerplate"), progress)},
                                device=state.decisions.device)
        return {"job": job.describe(with_result=False)}

    @api.post("/api/criteria")
    def criteria(request: PairRequest):
        pair = state.pair(request.patient_id, request.cohort_id)
        if not pair.items:
            raise ValueError("this cohort has no tagged criteria lines")
        job = state.jobs.submit("criteria", f"{request.patient_id} x {request.cohort_id}",
                                lambda progress: state.decisions.criteria(pair, progress), device=state.decisions.device)
        return {"job": job.describe(with_result=False)}

    # LLM checks (the configured endpoint)

    def connection(request: Request, task: str) -> Connection:
        """The session's connection for the check (Data and Models)."""
        return Connection.for_task(state.guidelines.task_settings(owner(request), task))

    @api.get("/api/llm")
    def llm(request: Request):
        return state.llm.describe({task: connection(request, task) for task in ("good_option", "criteria_llm")})

    @api.post("/api/good-option")
    def good_option(body: LLMCheckRequest, request: Request):
        if not body.trial_id:
            raise ValueError("give trial_id")
        trial = state.trial(body.trial_id)
        entity = state.patient_entity(body.patient_id)
        return state.llm.good_option({"id": entity["id"], "key": entity["key"]}, trial["id"], trial.get("drugs") or [],
                                     body.rerun, connection=connection(request, "good_option"))

    @api.post("/api/criteria-llm")
    def criteria_llm(body: LLMCheckRequest, request: Request):
        if not body.cohort_id:
            raise ValueError("give cohort_id")
        cohort = state.cohort(body.cohort_id)
        return state.llm.criteria(state.pair(body.patient_id, cohort["id"]), body.rerun,
                                  connection=connection(request, "criteria_llm"))

    return api
