"""What the app holds for its guideline features and model connections, per browser session.

- **Model connections:** the app's default connection (``Settings``) and, per session, the connection the user
  chose for each LLM task (``LLM_TASKS``). A job takes a snapshot of its task's connection when it is submitted,
  so a later change cannot redirect a running or queued job's patient text to another endpoint.
- **The patient workspace:** a session's guideline jobs (summaries, guideline searches, workup reviews), its saved
  note compression and its extraction records.
- **Populations:** a session's population and its guideline-space cohort (``population.PopulationSessions``).

Everything is in memory and expires ``settings.ttl`` seconds after its last use.
"""
from __future__ import annotations

from dataclasses import replace
import threading
import time

from ..core.jobs import JobManager
from .extraction_workup import SessionExtractions
from .note_compressions import NoteCompressions
from .population import PopulationSessions
from .services import Catalogs, Inference, synthetic_note_sources
from .settings import (
    CLAUDE_LABELS, GEMINI_BACKENDS, LLM_TASKS, SQL_ONLY_BACKENDS, SQL_TASK, VERTEX_BACKENDS, VERTEX_LABELS,
    Settings, claude_backends, databricks_profile, vertex_profiles, vertex_project,
)


class GuidelineState:
    def __init__(self, settings: Settings, inference=None):
        self.settings = settings
        self.inference = inference or Inference(settings)
        self.catalogs = Catalogs(settings.catalog_root)
        # Guideline inference: one job at a time, one active job per session.
        self.jobs = JobManager(ttl=settings.ttl, exclusive=True, name="mmai-guidelines")
        self.note_compressions = NoteCompressions(settings.ttl)
        self.session_extractions = SessionExtractions(settings.ttl)
        self.population_sessions = PopulationSessions(settings.ttl)
        self.workspace_settings: dict[str, dict] = {}
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ expiry

    def clean(self, active_owners: set[str] | None = None) -> None:
        """Forget what sessions left (the app's watch loop calls this)."""
        with self.lock:
            expired = [owner for owner, entry in self.workspace_settings.items()
                       if time.monotonic() - entry["updated"] > self.settings.ttl]
            for owner in expired:
                del self.workspace_settings[owner]
        self.jobs.clean()
        self.note_compressions.clean()
        self.session_extractions.clean()
        self.population_sessions.clean(active_owners if active_owners is not None else self.jobs.active_owners())

    def clear(self, owner: str) -> None:
        """Clear a session's patient workspace and population (refused while one of its jobs runs)."""
        self.jobs.clear(owner)
        self.note_compressions.clear(owner)
        self.session_extractions.clear(owner)
        self.population_sessions.clear(owner)

    # ------------------------------------------------------------------ connections

    def current_settings(self, owner: str) -> Settings:
        with self.lock:
            entry = self.workspace_settings.get(owner)
            if entry is None:
                return self.settings
            entry["updated"] = time.monotonic()
            return replace(self.settings, **entry["values"])

    def task_settings(self, owner: str, task: str) -> Settings:
        return self.current_settings(owner).for_task(task)

    def patient_inference(self, owner: str, task: str):
        """The task's inference worker, bound now to the session's connections."""
        workspace = self.current_settings(owner)
        selected = workspace.for_task(task)
        worker = self.inference if selected is self.settings else self.inference.with_settings(selected)
        if isinstance(worker, Inference):
            worker = worker.with_compression_cache(self.note_compressions.bind(owner),
                                                   settings=workspace.for_task("compress"))
            if task == "workup":
                worker = worker.with_workup_backups(
                    full=workspace.for_task("workup_full_backup"), agent=workspace.for_task("workup_agent_backup"),
                ).with_extractions(self.session_extractions, owner)
        return worker

    def update(self, owner: str, values: dict, *, task: str | None, apply_to_all: bool) -> Settings:
        """Save a session's connection for ``task`` (or, with ``apply_to_all``, for every task)."""
        with self.lock:
            previous = self.current_settings(owner)
            saved = (self.workspace_settings.get(owner) or {}).get("values", {})
            if apply_to_all:
                connection = {k: v for k, v in values.items() if k not in {"backup_enabled", "failure_threshold"}}
                tasks = {}
                for name in LLM_TASKS:
                    prior = previous.for_task(name)
                    tasks[name] = {**connection,
                                   "backup_enabled": prior.backup_enabled if name.endswith("_backup") else False,
                                   "failure_threshold": prior.failure_threshold}
                # The source task's own backup controls are saved without copying them onto the other tasks.
                tasks[task] = values
                values = {**saved, **connection, "task_connections": tasks}
            elif task is not None:
                values = {**saved, "task_connections": {**previous.task_connections, task: values}}
            else:
                values = {**values, "task_connections": {}}
            updated = replace(self.settings, **values)
            self.workspace_settings[owner] = {"values": values, "updated": time.monotonic()}
            if connection_view(previous.for_task("compress")) != connection_view(updated.for_task("compress")):
                self.note_compressions.clear(owner)
            return updated

    def config_view(self, selected: Settings) -> dict:
        settings = self.settings
        project = vertex_project()
        return dict(
            **connection_view(selected),
            tasks={task: {"label": label, **connection_view(selected.for_task(task))}
                   for task, label in LLM_TASKS.items()},
            note_compression_available=True,
            vertex_profiles={key: {**profile, "label": VERTEX_LABELS[key], "project": project}
                             for key, profile in vertex_profiles().items()},
            claude_profiles={key: {**profile, "label": CLAUDE_LABELS[key], "databricks_profile": databricks_profile(),
                                   "task": SQL_TASK} for key, profile in claude_backends().items()},
            local_defaults={"endpoint": settings.endpoint, "model": settings.model, "tokenizer": settings.tokenizer},
            device=settings.device,
            ttl_minutes=settings.ttl // 60,
            max_upload_mb=settings.max_upload // 1024**2,
            synthetic_note_sources=synthetic_note_sources(settings),
            notice="Patient text is sent to the selected LLM endpoint; use an endpoint authorized for the data.",
        )


def connection_view(selected: Settings) -> dict:
    from matchminer_ai.llm.sampling import vendor_defaults

    if selected.llm_backend in VERTEX_BACKENDS:
        connection = vertex_profiles()[selected.llm_backend]
    elif selected.llm_backend in SQL_ONLY_BACKENDS:
        connection = {**claude_backends()[selected.llm_backend], "model": selected.model}
    else:
        connection = {"endpoint": selected.endpoint, "model": selected.model, "tokenizer": selected.tokenizer}
    managed = selected.llm_backend in GEMINI_BACKENDS or selected.llm_backend in SQL_ONLY_BACKENDS
    sampling, _ = (vendor_defaults(connection["model"], template={"enable_thinking": selected.reasoning})
                   if not managed else ({}, {}))
    return {"llm_backend": selected.llm_backend, **connection, "reasoning": selected.reasoning,
            "thinking_supported": selected.thinking_supported, "sampling": sampling,
            "backup_enabled": selected.backup_enabled, "failure_threshold": selected.failure_threshold}
