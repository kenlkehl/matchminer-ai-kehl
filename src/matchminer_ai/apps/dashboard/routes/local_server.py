"""Data and models' local model server (``apps.dashboard.local_server``), in every dashboard.

- ``GET /api/local-server``: the server's spec, command, state, phase and download progress, endpoint and its log's
  last lines.
- ``GET /api/local-server/gpus`` and ``/engines``; ``GET /api/local-server/recipe?model=&engine=&gpus=``: the form's
  starting point (a known model's settings, else generic ones), the repository's files and GGUF choices, fit
  warnings and the command; ``POST /api/local-server/preview``: the command and warnings of an edited spec.
- ``POST /api/local-server/start`` (409 while one runs) and ``/stop``.
- ``POST /api/local-server/use``: the session's connection for a task, or every task, becomes the ready server.

Anyone using the app may start, stop and configure the server (user, 2026-11-24).
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from ..local_server import LocalServerControl
from .common import owner
from .guidelines import LLMTask


class SpecBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: dict = Field(default_factory=dict)


class UseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: LLMTask | None = None
    apply_to_all: bool = False
    reasoning: bool = False


def gpu_indices(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None if value is None else []
    return [int(v) for v in value.split(",") if v.strip()]


def router(control: LocalServerControl) -> APIRouter:
    api = APIRouter()

    @api.get("/api/local-server")
    def describe(log_lines: int = 200):
        return control.describe(max(0, min(log_lines, 1000)))

    @api.get("/api/local-server/gpus")
    def gpus():
        return control.gpus()

    @api.get("/api/local-server/engines")
    def engines():
        return {name: {k: v for k, v in info.items() if k != "command"} for name, info in control.engines().items()}

    @api.get("/api/local-server/recipe")
    def recipe(model: str = "", engine: str | None = None, gpus: str | None = None):
        return control.recipe(model, engine, gpu_indices(gpus))

    @api.post("/api/local-server/preview")
    def preview(body: SpecBody):
        return control.preview(body.spec)

    @api.post("/api/local-server/start")
    def start(body: SpecBody):
        return control.start(body.spec)

    @api.post("/api/local-server/stop")
    def stop():
        return control.stop()

    @api.post("/api/local-server/use")
    def use(body: UseBody, request: Request):
        updated = control.use(owner(request), task=body.task, apply_to_all=body.apply_to_all,
                              reasoning=body.reasoning)
        return control.guidelines.config_view(updated)

    return api
