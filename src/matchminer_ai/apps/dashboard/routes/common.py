"""What the route modules share: the request's browser session and the dashboard's row shapes."""
from __future__ import annotations

from fastapi import Request

from ..core import cancer_types
from ..state import DashState


def owner(request: Request) -> str:
    """The request's browser session (``server`` sets it from the session cookie)."""
    return request.state.owner


DRUG_BRIEF = ("drug_id", "name", "listed_name", "role", "mechanism_types", "all_classes", "class_ids", "payload",
              "payload_classes")


def trial_summary(trial: dict) -> dict:
    """A trial's fields on every cohort row; its drugs without their arms and mechanism sentence (those are in the
    trial's details)."""
    keys = ("id", "nct_id", "protocol_no", "short_title", "phase", "status", "is_open", "center", "center_id",
            "center_short_name", "site_pi", "overall_pi", "contact", "sites", "primary_version", "state")
    drugs = [{k: d.get(k) for k in DRUG_BRIEF} for d in trial.get("drugs") or []]  # a store built before drugs: none
    return {**{k: trial.get(k) for k in keys}, "drugs": drugs}


def lines(items: list[dict], kind: str, boilerplate: bool | None = False) -> list[str]:
    return [i["text"] for i in items if i.get("criterion_type") == kind
            and (boilerplate is None or bool(i.get("boilerplate")) == boilerplate)]


def cohort_row(state: DashState, cohort: dict) -> dict:
    """A dashboard row: the cohort (``state.cohort``: its current criteria) with its trial's summary fields and its
    management state: what kind of unit it is (merged, original, subcohort), its parent, whether its criteria were
    edited or their base changed, and its slots."""
    trial = state.store["trials"][cohort["trial_id"]]
    items = cohort["items"]
    version = trial["versions"][cohort["version"]]
    managed = cohort.get("management") or {}
    return {"cohort_id": cohort["id"], "trial": trial_summary(trial), "name": cohort["name"], "number": cohort["number"],
            "version": cohort["version"], "merged_from": cohort["merged_from"], "group": cohort.get("group"),
            "group_reason": cohort.get("group_reason"), "cancer_types": cohort["cancer_types"],
            "key_inclusion": lines(items, "inclusion"), "key_exclusion": lines(items, "exclusion"),
            "common_count": sum(1 for i in items if i.get("boilerplate")), "line_count": len(items),
            "completed_at": version["completed_at"], "stale": version["state"] == "stale",
            "unit": {"kind": managed.get("kind") or "merged", "parent": managed.get("parent"),
                     "parent_name": parent_name(state, managed.get("parent")), "edited": bool(managed.get("edited")),
                     "base_changed": bool(managed.get("stale")), "unclassified": managed.get("unclassified") or 0},
            "slots": managed.get("slots")}


def parent_name(state: DashState, parent: str | None) -> str | None:
    if not parent:
        return None
    cohort = state.management.effective(parent)
    return cohort["name"] if cohort else None


def cohort_detail(state: DashState, cohort: dict) -> dict:
    return {**cohort_row(state, cohort), "key": cohort["key"], "boilerplate": cohort["boilerplate"],
            "items": cohort["items"]}


def type_labels(keys: list[str]) -> list[str]:
    return [cancer_types.TYPE_LABELS[k] for k in keys if k in cancer_types.TYPE_LABELS]
