"""The Benchmarking tab's routes (``benchmarking``): runs, a fidelity run and its PDF export, a judge run,
recomputing a judge run's comparison and metrics, and adjudicating its queued units."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ..benchmarking import Benchmarks
from ..core.errors import InputError


class Recompute(BaseModel):
    steps: list[Literal["compare", "metrics"]] = Field(default_factory=lambda: ["compare", "metrics"], min_length=1)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str = Field(min_length=1, max_length=2000)
    choice: Literal["judge1", "judge2", "other", "skip"]
    value: str | None = Field(default=None, max_length=200)
    note: str = Field(default="", max_length=4000)
    show_reference: bool = False


def router(benchmarks: Benchmarks) -> APIRouter:
    api = APIRouter()

    @api.get("/api/benchmarks")
    def runs():
        return {"runs": benchmarks.runs(), "roots": len(benchmarks.roots)}

    @api.get("/api/benchmarks/fidelity/{run_id}")
    def fidelity(run_id: str):
        return benchmarks.fidelity(run_id)

    @api.get("/api/benchmarks/fidelity/{run_id}/table1")
    def table1(run_id: str):
        return benchmarks.table1(run_id)

    @api.post("/api/benchmarks/fidelity/{run_id}/report", status_code=202)
    def export(run_id: str):
        return {"job": benchmarks.export(run_id).describe(with_result=False)}

    @api.get("/api/benchmarks/fidelity/{run_id}/report.pdf")
    def report(run_id: str):
        path, name = benchmarks.report_file(run_id)
        return FileResponse(path, media_type="application/pdf", filename=name,
                            headers={"Cache-Control": "no-store"})

    @api.get("/api/benchmarks/judge/{run_id}")
    def judge(run_id: str):
        return benchmarks.judge(run_id)

    @api.post("/api/benchmarks/judge/{run_id}/recompute", status_code=202)
    def recompute(run_id: str, body: Recompute):
        return {"job": benchmarks.recompute(run_id, tuple(dict.fromkeys(body.steps))).describe(with_result=False)}

    @api.get("/api/benchmarks/judge/{run_id}/adjudication")
    def adjudication(run_id: str):
        return benchmarks.store(run_id).state()

    @api.get("/api/benchmarks/judge/{run_id}/adjudication/item")
    def item(run_id: str, component: str, item_id: str, reference: bool = False):
        store = benchmarks.store(run_id)
        if item_id not in store.items.get(component, {}):
            raise InputError("That item is not in this run.")
        return store.item(component, item_id, reference)

    @api.post("/api/benchmarks/judge/{run_id}/adjudication/decide")
    def decide(run_id: str, body: Decision):
        try:
            return benchmarks.store(run_id).decide(body.model_dump())
        except KeyError:
            raise InputError("That unit is not in this run's comparison (rerun compare?).") from None

    return api
