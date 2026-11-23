"""Clinical trial management's Trial management sidebar: slots, edited criteria, subcohorts, groups and the trial
management file (``trial_management``).

Every change is written to the mode's file at once and shared by everyone using the dashboard. A change of criteria
then classifies its new and changed lines (a job of the browser session, on its ``criteria_labels`` connection;
skipped without one, leaving them unclassified) and embeds the cohort's key criteria with TrialSpace, so Find
patients, the decision model, the LLM checks and structured-data screening read the current criteria.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..core.errors import InputError
from ..criteria_labels import TASK, unavailable
from ..llm import Connection
from ..state import DashState
from .common import owner as session_owner

MAX_IMPORT = 50 * 1024 * 1024


class Slots(BaseModel):
    available: int | None = None
    consented: int | None = None
    enrolled: int | None = None
    #: The unit's revision the page showed (a change by someone else since is refused).
    revision: int | None = None


class Line(BaseModel):
    heading: str = Field(default="", max_length=40)
    text: str = Field(max_length=20_000)
    criterion_type: str | None = None
    boilerplate: bool | None = None
    labeled_by: str | None = None


class Criteria(BaseModel):
    items: list[Line] = Field(max_length=2000)
    revision: int | None = None
    #: The cohort has subcohorts: whether its new criteria replace theirs (they are kept for reference).
    propagate: bool | None = None


class Revert(BaseModel):
    revision: int | None = None
    propagate: bool | None = None


class Revision(BaseModel):
    revision: int | None = None


class NewSubcohort(BaseModel):
    parent_id: str = Field(min_length=1, max_length=200)
    name: str = Field(max_length=400)


class Name(BaseModel):
    name: str = Field(max_length=400)
    revision: int | None = None


class TrialRevision(BaseModel):
    #: The trial's version the page showed (``trial_revision``); a reset of a trial changed since is refused.
    revision: str | None = Field(default=None, max_length=64)


class ClearAll(BaseModel):
    #: The word the user typed to confirm.
    confirm: str = Field(default="", max_length=40)


class Pin(BaseModel):
    #: A group of the cohort's center; null: back to its rule group (an original or a subcohort: no group).
    group_id: str | None = Field(default=None, max_length=200)


def router(state: DashState) -> APIRouter:
    api = APIRouter()
    m = state.management

    def after(request: Request, unit_ids: list[str]) -> dict:
        """Classify the changed units' unclassified lines, then embed them; without a connection, only embed."""
        todo = m.unlabeled(unit_ids)
        if not todo:
            state.embed_cohorts()
            return {"classifying": False, "skipped": None}
        connection = Connection.for_task(state.guidelines.task_settings(session_owner(request), TASK))
        reason = unavailable(connection)
        if reason:
            for unit_id in todo:
                m.errors[unit_id] = reason
            state.embed_cohorts()
            return {"classifying": False, "skipped": reason}
        job = classify(request, todo, connection)
        return {"classifying": True, "skipped": None, "job": job.describe(with_result=False)}

    def classify(request: Request, unit_ids: list[str], connection: Connection):
        from ..core.jobs import public_error
        from .. import criteria_labels

        inputs = m.classification_input(unit_ids)
        count = sum(1 for c in inputs for i in c["items"] if not i["criterion_type"])
        noun = "criterion" if count == 1 else "criteria"

        def run(progress):
            progress(0, count, f"Classifying {count} {noun} with {connection.label}")
            try:
                config = criteria_labels.configure(connection, state.settings.cache_dir / "llm_model_metadata")
                results = criteria_labels.classify(inputs, config)
            except Exception as exc:
                for unit_id in unit_ids:
                    m.errors[unit_id] = f"Classification failed ({public_error(exc)})"
                state.embed_cohorts()
                raise
            labeled = m.apply_labels(inputs, results)
            failed = count - min(labeled, count)
            for unit_id in unit_ids:
                if failed:
                    m.errors[unit_id] = (
                        f"{failed} criterion could not be classified; label it by hand or try again." if failed == 1
                        else f"{failed} criteria could not be classified; label them by hand or try again.")
                else:
                    m.errors.pop(unit_id, None)
            state.embed_cohorts()
            progress(count, count, "classified")
            return {"classified": count - failed, "unclassified": failed}

        job = state.llm.jobs.submit("criteria_labels", f"criteria classification: {count} {noun}", run,
                                    owner=session_owner(request), device="endpoint")
        for unit_id in unit_ids:
            m.classifying[unit_id] = job
            m.errors.pop(unit_id, None)
        return job

    @api.get("/api/management")
    def status(request: Request):
        connection = Connection.for_task(state.guidelines.task_settings(session_owner(request), TASK))
        return {**m.describe(), "classifier": {"connection": connection.describe(), "unavailable": unavailable(connection)}}

    @api.get("/api/management/centers/{center_id}/summary")
    def center_summary(center_id: str):
        """The Trial management list's badges (the file only; no model is asked)."""
        return m.center_summary(center_id)

    @api.get("/api/management/trials/{trial_id}")
    def trial(trial_id: str):
        return m.trial_view(trial_id)

    @api.post("/api/management/trials/{trial_id}/open")
    def open_trial(trial_id: str, request: Request):
        """A trial opened on the Trial management list: its original cohorts' lines that no label could be reused
        for are classified (once per version of their lines; Classify retries)."""
        state.trial(trial_id)
        todo = m.originals_to_classify(trial_id)
        if not todo:
            return {"classifying": False, "skipped": None, "units": []}
        result = after(request, todo)
        if result["classifying"]:
            m.mark_auto_tried(todo)
        return {**result, "units": todo}

    @api.put("/api/management/units/{unit_id}/slots")
    def slots(unit_id: str, body: Slots):
        m.set_slots(unit_id, body.model_dump(exclude={"revision"}), body.revision)
        state.embed_cohorts()  # a cohort with slots is searched on its own criteria (open-slots search)
        return {"ok": True}

    @api.put("/api/management/units/{unit_id}/criteria")
    def criteria(unit_id: str, body: Criteria, request: Request):
        changed = m.set_criteria(unit_id, [line.model_dump() for line in body.items], revision=body.revision,
                                 propagate=body.propagate)
        return {"changed": changed, **after(request, changed)}

    @api.post("/api/management/units/{unit_id}/revert")
    def revert(unit_id: str, body: Revert, request: Request):
        changed = m.revert(unit_id, revision=body.revision, propagate=body.propagate)
        return {"changed": changed, **after(request, changed)}

    @api.post("/api/management/units/{unit_id}/keep")
    def keep(unit_id: str, body: Revision):
        m.keep(unit_id, revision=body.revision)
        return {"ok": True}

    @api.post("/api/management/units/{unit_id}/reset-to-parent")
    def reset(unit_id: str, body: Revision, request: Request):
        m.reset_to_parent(unit_id, revision=body.revision)
        return {"changed": [unit_id], **after(request, [unit_id])}

    @api.post("/api/management/units/{unit_id}/dismiss-previous")
    def dismiss(unit_id: str):
        m.dismiss_previous(unit_id)
        return {"ok": True}

    @api.post("/api/management/units/{unit_id}/classify")
    def classify_unit(unit_id: str, request: Request):
        if not m.unlabeled([unit_id]):
            raise InputError("This cohort has no unclassified criteria.")
        return after(request, [unit_id])

    @api.delete("/api/management/units/{unit_id}")
    def delete(unit_id: str):
        m.delete_unit(unit_id)
        state.embed_cohorts()
        return {"ok": True}

    @api.post("/api/management/subcohorts")
    def create(body: NewSubcohort, request: Request):
        unit_id = m.create_subcohort(body.parent_id, body.name)
        return {"id": unit_id, **after(request, [unit_id])}

    @api.patch("/api/management/subcohorts/{unit_id}")
    def rename(unit_id: str, body: Name):
        m.rename_subcohort(unit_id, body.name, body.revision)
        return {"ok": True}

    @api.post("/api/management/centers/{center_id}/groups")
    def create_group(center_id: str, body: Name):
        return {"id": m.create_group(center_id, body.name)}

    @api.patch("/api/management/centers/{center_id}/groups/{group_id}")
    def rename_group(center_id: str, group_id: str, body: Name):
        m.rename_group(center_id, group_id, body.name)
        return {"ok": True}

    @api.delete("/api/management/centers/{center_id}/groups/{group_id}")
    def delete_group(center_id: str, group_id: str):
        m.delete_group(center_id, group_id)
        return {"ok": True}

    @api.put("/api/management/pins/{unit_id}")
    def pin(unit_id: str, body: Pin):
        m.pin(unit_id, body.group_id)
        return {"ok": True}

    @api.post("/api/management/trials/{trial_id}/reset")
    def reset_trial(trial_id: str, body: TrialRevision):
        """Reset this trial: its slots, edits, subcohorts and placements go (the file is kept beside it first)."""
        result = m.reset_trial(trial_id, body.revision)
        state.embed_cohorts()
        return result

    @api.post("/api/management/clear")
    def clear_all(body: ClearAll):
        """Clear all: everything users entered goes (the file is kept beside it first)."""
        if body.confirm.strip().lower() != "clear":
            raise InputError('Type "clear" to confirm.')
        return m.clear_all()

    @api.get("/api/management/export")
    def export():
        name = f"trial_management_{state.settings.mode}.json"
        return Response(m.export(), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @api.post("/api/management/import")
    async def import_file(file: UploadFile = File(...), confirm: str = Form("")):
        """Replace the file with an exported one (the page asks first; the replaced file is kept beside it)."""
        if confirm != "replace":
            raise InputError("Confirm that the imported file replaces the current one.")
        data, size = bytearray(), 0
        try:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > MAX_IMPORT:
                    raise InputError("The file is larger than a trial management file can be.")
                data.extend(block)
        finally:
            await file.close()
        described = m.import_file(bytes(data))
        state.embed_cohorts()
        return described

    return api
