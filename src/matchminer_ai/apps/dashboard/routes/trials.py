"""The Trials tab: the store's disease centers (or cancer-type groups), trials, cohorts and their drugs."""
from __future__ import annotations

from fastapi import APIRouter

from ..core import cancer_types
from ..state import DashState, NotFound
from .common import cohort_detail, cohort_row, trial_summary


def router(state: DashState) -> APIRouter:
    api = APIRouter()

    @api.get("/api/centers/{center_id}")
    def center(center_id: str):
        found = next((c for c in state.store["centers"] if c["id"] == center_id), None)
        if found is None:
            raise NotFound(center_id)
        cohorts = state.store["cohorts"]
        groups = [{"id": g["id"], "name": g["name"], "trial_count": len(g["trial_ids"]),
                   "rows": [cohort_row(state, cohorts[cid]) for cid in g["cohort_ids"]]} for g in found["groups"]]
        pending = [trial_summary(state.store["trials"][t]) | {
            "versions": {v: {k: x[k] for k in ("state", "excluded_reason")}
                         for v, x in state.store["trials"][t]["versions"].items()}}
            for t in found["pending_trial_ids"]]
        return {"id": found["id"], "name": found["name"], "short_name": found["short_name"],
                "built_at": state.store["built_at"], "trial_count": len(found["trial_ids"]), "groups": groups,
                "pending": pending, "cancer_type_labels": cancer_types.TYPE_LABELS}

    @api.get("/api/trials/{trial_id}")
    def trial(trial_id: str):
        t = state.trial(trial_id)
        versions = {}
        for version, info in t["versions"].items():
            versions[version] = {**{k: v for k, v in info.items() if k != "cohort_ids"},
                                 "cohorts": [cohort_detail(state, state.store["cohorts"][c]) for c in info["cohort_ids"]]}
        return {**{k: v for k, v in t.items() if k not in ("versions", "cohort_ids")}, "drugs": t.get("drugs") or [],
                "versions": versions}

    @api.get("/api/cohorts/{cohort_id}")
    def cohort(cohort_id: str):
        return cohort_detail(state, state.cohort(cohort_id))

    # GoodOption catalog evidence (read on demand)

    @api.get("/api/drugs/{drug_id}")
    def drug_evidence(drug_id: str):
        return state.evidence.drug(drug_id, state.store)

    @api.get("/api/classes/{class_id}")
    def class_evidence(class_id: str):
        return state.evidence.klass(class_id, state.store)

    return api

