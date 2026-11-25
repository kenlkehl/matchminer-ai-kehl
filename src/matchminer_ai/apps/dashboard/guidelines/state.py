"""What the app holds for its guideline features and model connections, per browser session.

- **Model connections:** the app's default connection (``Settings``) and, per session, the connection the user
  chose for each LLM task (``LLM_TASKS``). A job takes a snapshot of its task's connection when it is submitted,
  so a later change cannot redirect a running or queued job's patient text to another endpoint.
- **The patient workspace:** a session's guideline jobs (summaries, guideline searches, workup reviews) and its
  extraction records.
- **Populations:** a session's population and its guideline-space cohort (``population.PopulationSessions``).

Everything is in memory and expires ``settings.ttl`` seconds after its last use.
"""
from __future__ import annotations

from dataclasses import replace
import threading
import time

from ..core.errors import InputError
from ..core.jobs import JobManager
from .extraction_workup import SessionExtractions
from .population import PopulationSessions
from .services import Catalogs, Inference, synthetic_note_sources
from .settings import (
    CLAUDE_BACKENDS, CLAUDE_LABELS, GEMINI_BACKENDS, LLM_TASKS, VERTEX_BACKENDS, VERTEX_LABELS,
    Settings, claude_backends, databricks_profile, vertex_profiles, vertex_project,
)


# The values of a saved connection that name where its requests go.
CONNECTION_KEYS = ("llm_backend", "endpoint", "model", "tokenizer")


class GuidelineState:
    def __init__(self, settings: Settings, inference=None):
        self.settings = settings
        self.inference = inference or Inference(settings)
        self.catalogs = Catalogs(settings.catalog_root)
        # Guideline inference: one job at a time, one active job per session.
        self.jobs = JobManager(ttl=settings.ttl, exclusive=True, name="mmai-guidelines")
        self.session_extractions = SessionExtractions(settings.ttl)
        from .records import StoreHandle

        # The mode's extraction record store (the Patient tab's Structured data, criteria screening).
        self.records = StoreHandle(settings)
        self.population_sessions = PopulationSessions(settings.ttl)
        # The LLM decision checks (``apps.dashboard.llm_decisions``), for the guideline searches' LLM rankings; the
        # app sets them.
        self.llm_decisions = None
        # The local model server's control (``apps.dashboard.local_server``); the app sets it. With
        # --local-server-only its ``refusal`` decides which connections are allowed.
        self.local_server = None
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
        self.session_extractions.clean()
        self.population_sessions.clean(active_owners if active_owners is not None else self.jobs.active_owners())

    def clear(self, owner: str) -> None:
        """Clear a session's patient workspace and population (refused while one of its jobs runs)."""
        self.jobs.clear(owner)
        self.session_extractions.clear(owner)
        self.population_sessions.clear(owner)

    # ------------------------------------------------------------------ connections

    def current_settings(self, owner: str) -> Settings:
        with self.lock:
            entry = self.workspace_settings.get(owner)
            if entry is None:
                return self.settings
            entry["updated"] = time.monotonic()
            return replace(self.settings, **self.allowed(entry["values"]))

    def refusal(self, values: dict) -> str | None:
        """Why a connection is refused (``--local-server-only``: anything but the app's ready local server)."""
        return self.local_server.refusal(values) if self.local_server is not None else None

    def allowed(self, values: dict) -> dict:
        """A session's saved values without the connections now refused (a local server since stopped or restarted
        on another port), so those tasks fall back to the app's default."""
        if self.local_server is None or not self.local_server.only:
            return values
        kept = dict(values)
        if any(k in kept for k in CONNECTION_KEYS) and self.refusal(kept):
            for key in CONNECTION_KEYS:
                kept.pop(key, None)
        kept["task_connections"] = {task: v for task, v in (kept.get("task_connections") or {}).items()
                                    if not self.refusal(v)}
        return kept

    def task_settings(self, owner: str, task: str) -> Settings:
        return self.current_settings(owner).for_task(task)

    def patient_inference(self, owner: str, task: str):
        """The task's inference worker, bound now to the session's connections."""
        with self.lock:  # the base settings and their worker as one pair (set_default_connection)
            workspace = self.current_settings(owner)
            base, inference = self.settings, self.inference
        selected = workspace.for_task(task)
        selected.require_connection()
        worker = inference if selected is base else inference.with_settings(selected)
        if isinstance(worker, Inference) and task == "workup":
            worker = worker.with_workup_backups(
                full=workspace.for_task("workup_full_backup"), agent=workspace.for_task("workup_agent_backup"),
            ).with_extractions(self.session_extractions, owner, self.records)
        return worker

    def set_default_connection(self, values: dict, task_connections: dict | None = None) -> tuple[Settings, object]:
        """Make ``values`` (``llm_backend``, ``endpoint``, ``model``, ``tokenizer``, ...) the app's default connection
        for every task (``task_connections``: the per-task defaults, none by default), in sessions that chose none.

        The base settings and the default inference worker change together, under the lock:
        ``patient_inference`` uses the bound worker only while a session's settings are the base ones. Returns the
        previous ``(settings, inference)`` for ``restore_default_connection``."""
        with self.lock:
            previous = (self.settings, self.inference)
            settings = replace(self.settings, **values, task_connections=dict(task_connections or {}))
            rebind = getattr(self.inference, "with_settings", None)
            self.settings, self.inference = settings, rebind(settings) if rebind else self.inference
            return previous

    def restore_default_connection(self, previous: tuple[Settings, object]) -> None:
        with self.lock:
            self.settings, self.inference = previous

    def update(self, owner: str, values: dict, *, task: str | None, apply_to_all: bool) -> Settings:
        """Save a session's connection for ``task`` (or, with ``apply_to_all``, for every task). A connection the
        app refuses (``refusal``) raises ``InputError``."""
        reason = self.refusal(values)
        if reason:
            raise InputError(reason)
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
            return updated

    def config_view(self, selected: Settings) -> dict:
        settings = self.settings
        only = self.local_server is not None and self.local_server.only
        project = vertex_project() if not only else ""
        return dict(
            **connection_view(selected),
            tasks={task: {"label": label, **connection_view(selected.for_task(task))}
                   for task, label in LLM_TASKS.items()},
            # --local-server-only: no hosted profiles to offer (nor site values to read).
            vertex_profiles={} if only else {key: {**profile, "label": VERTEX_LABELS[key], "project": project}
                                             for key, profile in vertex_profiles().items()},
            claude_profiles={} if only else {
                key: {**profile, "label": CLAUDE_LABELS[key], "databricks_profile": databricks_profile()}
                for key, profile in claude_backends().items()},
            local_defaults={"endpoint": settings.endpoint, "model": settings.model, "tokenizer": settings.tokenizer},
            local_server=(self.local_server.connection_view() if self.local_server is not None else
                          {"only": False, "state": None, "endpoint": None, "model": None, "tokenizer": None}),
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
    elif selected.llm_backend in CLAUDE_BACKENDS:
        connection = {**claude_backends()[selected.llm_backend], "model": selected.model}
    else:
        connection = {"endpoint": selected.endpoint, "model": selected.model, "tokenizer": selected.tokenizer}
    managed = selected.llm_backend in GEMINI_BACKENDS or selected.llm_backend in CLAUDE_BACKENDS
    sampling, _ = (vendor_defaults(connection["model"], template={"enable_thinking": selected.reasoning})
                   if not managed else ({}, {}))
    return {"llm_backend": selected.llm_backend, **connection, "reasoning": selected.reasoning,
            "thinking_supported": selected.thinking_supported, "sampling": sampling,
            "backup_enabled": selected.backup_enabled, "failure_threshold": selected.failure_threshold}
