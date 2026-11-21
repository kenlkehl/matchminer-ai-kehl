"""The app's description (mode, tabs, counts), Data and Models, and the jobs of every tab."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import pipeline
from ..core import cancer_types
from ..llm import Connection, discover_models
from ..state import DashState
from .common import owner


class ConnectionRequest(BaseModel):
    backend: str = "openai"
    endpoint: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=512)
    tokenizer: str | None = Field(default=None, max_length=512)


def router(state: DashState) -> APIRouter:
    api = APIRouter()

    @api.get("/api/meta")
    def meta():
        s = state.store
        settings = state.settings
        return {"mode": settings.mode, "tabs": state.tabs(),
                "built_at": s["built_at"], "counts": s["counts"], "higher_threshold_groups": s["higher_threshold_groups"],
                "centers": [{"id": c["id"], "name": c["name"], "short_name": c["short_name"],
                             "trials": len(c["trial_ids"]), "cohorts": sum(len(g["cohort_ids"]) for g in c["groups"]),
                             "pending": len(c["pending_trial_ids"])} for c in s["centers"]],
                "cancer_types": cancer_types.describe(), "auto_score_n": settings.auto_score_n,
                "default_k": settings.default_k, "max_k": settings.max_k,
                "round": state.round.number if state.round else None, "site": pipeline.branding(),
                "vectors": state.cohort_vectors.describe(), "patients": len(state.corpus.patients),
                "sign_in": bool(settings.require_token),
                # The GoodOption catalog's state (None: a store built before the drugs), so a page can say why a
                # trial shows no drugs.
                "good_options": ({k: s["good_options"].get(k) for k in ("state", "message")}
                                 if s.get("good_options") else None)}

    @api.get("/api/status")
    def status(request: Request):
        return {**state.describe(), "jobs": state.registry.recent(10, owner(request))}

    @api.post("/api/rebuild")
    def rebuild():
        job = state.background.submit("rebuild", "dashboard store", lambda progress: {"rebuilt": state.check_store()})
        return {"job": job.describe(with_result=False)}

    @api.get("/api/jobs/{job_id}")
    def job(job_id: str, request: Request):
        found = state.registry.get(job_id, owner(request))
        if found is None:
            raise HTTPException(404, "This job has expired or belongs to another browser session.")
        return found.describe()

    @api.post("/api/jobs/{job_id}/stop")
    def stop(job_id: str, request: Request):
        found = state.registry.cancel(job_id, owner(request))
        if found is None:
            raise HTTPException(404, "This job has expired, belongs to another browser session or cannot be stopped.")
        return found.describe(with_result=False)

    # The LLM checks' connection (replaced by the per-task models in routes.models).

    @api.get("/api/llm")
    def llm():
        return state.llm.describe()

    @api.post("/api/llm")
    def set_llm(request: ConnectionRequest):
        state.llm.set_connection(Connection.make(request.backend, request.endpoint, request.model, request.tokenizer))
        return state.llm.describe()

    @api.post("/api/llm/models")
    async def llm_models(request: ConnectionRequest):
        connection = (Connection.make(request.backend) if request.backend != "openai"
                      else state.llm.connection if not request.endpoint
                      else Connection("openai", "", "", ""))
        return await discover_models(connection, request.endpoint)

    return api
