"""What the app holds for its guideline features and model connections, per browser session.

- **Model connections:** the app's default connection (``Settings``) and, per session, the connection the user
  chose for each LLM task (``LLM_TASKS``). A job takes a snapshot of its task's connection when it is submitted,
  so a later change cannot redirect a running or queued job's patient text to another endpoint.
- **OpenRouter keys** (synthetic mode only): a session's pasted key, kept by ``matchminer_ai.llm.openrouter`` in this
  process's memory; the session holds only its reference (``Settings.api_key_ref``). It is never written, logged or
  sent back to the browser (``config_view`` shows only that a key is set and its last four characters), and it is
  forgotten when the session expires or the user removes it.
- **The hosted OpenRouter key** (synthetic mode, ``apps.dashboard.hosted_openrouter``): serves every OpenRouter
  connection of a session without its own key, through the session's own reference to it (``key_ref``), for one
  model only (``update`` refuses another without the session's own key). Nothing of it is ever shown.
- **The patient workspace:** a session's guideline jobs (summaries, guideline searches, workup reviews) and its
  extraction records.
- **Populations:** a session's population and its guideline-space cohort (``population.PopulationSessions``).

Everything is in memory and expires ``settings.ttl`` seconds after its last use.
"""
from __future__ import annotations

from dataclasses import replace
import threading
import time

from matchminer_ai.llm import openrouter

from ..core.errors import InputError
from ..core.jobs import JobManager
from .extraction_workup import SessionExtractions
from .population import PopulationSessions
from .services import Catalogs, Inference, synthetic_note_sources
from .settings import (
    CLAUDE_BACKENDS, CLAUDE_LABELS, GEMINI_BACKENDS, LLM_TASKS, OPENROUTER, VERTEX_BACKENDS, VERTEX_LABELS,
    Settings, claude_backends, databricks_profile, vertex_profiles, vertex_project,
)


# The values of a saved connection that name where its requests go.
CONNECTION_KEYS = ("llm_backend", "endpoint", "model", "tokenizer")
OPENROUTER_REAL = "OpenRouter is offered in synthetic mode only; real mode never sends patient data to it."
OPENROUTER_NOTICE = ("At your own risk: the key and the synthetic data the tasks send go to OpenRouter (openrouter.ai) "
                     "under your own account, which is billed for them. The app keeps the key only in its memory for "
                     "this browser session, never on disk, and never shows it again.")


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
        # Each session's OpenRouter key reference ({"ref", "updated"}); the key is in llm.openrouter's memory.
        self.openrouter_keys: dict[str, dict] = {}
        # The hosted OpenRouter connection's limits (``llm.openrouter.HostedQuota``) and the name of its model; the
        # app sets them when it has one.
        self.hosted: openrouter.HostedQuota | None = None
        self.hosted_label = ""
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ expiry

    def clean(self, active_owners: set[str] | None = None) -> None:
        """Forget what sessions left (the app's watch loop calls this)."""
        with self.lock:
            expired = [owner for owner, entry in self.workspace_settings.items()
                       if time.monotonic() - entry["updated"] > self.settings.ttl]
            for owner in expired:
                del self.workspace_settings[owner]
            for owner in [o for o, e in self.openrouter_keys.items()
                          if time.monotonic() - e["updated"] > self.settings.ttl]:
                openrouter.forget_key(self.openrouter_keys.pop(owner)["ref"])
        self.jobs.clean()
        self.session_extractions.clean()
        self.population_sessions.clean(active_owners if active_owners is not None else self.jobs.active_owners())

    def clear(self, owner: str) -> None:
        """Clear a session's patient workspace and population (refused while one of its jobs runs)."""
        self.jobs.clear(owner)
        self.session_extractions.clear(owner)
        self.population_sessions.clear(owner)

    # ------------------------------------------------------------------ connections

    def key_ref(self, owner: str) -> str:
        """The OpenRouter key a session's OpenRouter connections use: its own, else its reference to the hosted key
        (when the app has one), else none ("")."""
        with self.lock:
            key = self.openrouter_keys.get(owner)
            if key is not None:
                key["updated"] = time.monotonic()
                return str(key["ref"])
        return self.hosted.session_ref(owner) if self.hosted is not None else ""

    def current_settings(self, owner: str) -> Settings:
        with self.lock:
            entry = self.workspace_settings.get(owner)
            ref = self.key_ref(owner)
            if entry is None:
                return replace(self.settings, api_key_ref=ref) if ref else self.settings
            entry["updated"] = time.monotonic()
            return replace(self.settings, **self.allowed(entry["values"]), api_key_ref=ref)

    def refusal(self, values: dict) -> str | None:
        """Why a connection is refused (``--local-server-only``: anything but the app's ready local server, and
        OpenRouter unless allowed; real mode: OpenRouter)."""
        if self.local_server is not None:
            return self.local_server.refusal(values)
        return OPENROUTER_REAL if values.get("llm_backend") == OPENROUTER and self.settings.phi else None

    def openrouter_refusal(self) -> str | None:
        """Why this app does not offer OpenRouter (None when it does)."""
        return self.refusal({"llm_backend": OPENROUTER})

    def hosted_refusal(self, owner: str, values: dict) -> str | None:
        """Why a session without its own key may not save an OpenRouter connection: the hosted key serves its one
        model only."""
        if (self.hosted is None or values.get("llm_backend") != OPENROUTER
                or values.get("model") == self.hosted.model):
            return None
        with self.lock:
            own = owner in self.openrouter_keys
        return None if own else openrouter.MODEL_LOCKED.format(model=self.hosted.model)

    def set_openrouter_key(self, owner: str, key: str) -> None:
        """Keep a session's OpenRouter key in memory (``ValueError`` with a fixed text when it is not a key)."""
        reason = self.openrouter_refusal()
        if reason:
            raise InputError(reason)
        ref = openrouter.register_key(key)
        with self.lock:
            previous = self.openrouter_keys.get(owner)
            self.openrouter_keys[owner] = {"ref": ref, "updated": time.monotonic()}
        if previous:
            openrouter.forget_key(previous["ref"])

    def forget_openrouter_key(self, owner: str) -> None:
        with self.lock:
            previous = self.openrouter_keys.pop(owner, None)
        if previous:
            openrouter.forget_key(previous["ref"])

    def allowed(self, values: dict) -> dict:
        """A session's saved values without the connections now refused (a local server since stopped or restarted
        on another port), so those tasks fall back to the app's default."""
        if self.local_server is None or not self.local_server.restricts:
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
        reason = self.refusal(values) or self.hosted_refusal(owner, values)
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
            updated = replace(self.settings, **values, api_key_ref=previous.api_key_ref)
            self.workspace_settings[owner] = {"values": values, "updated": time.monotonic()}
            return updated

    def config_view(self, selected: Settings, owner: str | None = None) -> dict:
        """``/api/models``: the session's connections (``selected``: its settings) and what the app offers. ``owner``:
        the session, for its share of the hosted key's quota."""
        settings = self.settings
        only = self.local_server is not None and self.local_server.only
        project = vertex_project() if not only else ""
        offered = self.openrouter_refusal() is None
        # The session's own key only: the hosted key is never "set" for it, and nothing of it is shown.
        own_key = openrouter.has_key(selected.api_key_ref) and not openrouter.is_hosted(selected.api_key_ref)
        hosted = None
        if offered and self.hosted is not None:
            from ..hosted_openrouter import view

            ref = self.hosted.session_ref(owner) if owner is not None else (
                selected.api_key_ref if openrouter.is_hosted(selected.api_key_ref) else "")
            hosted = view(self.hosted, ref, self.hosted_label, own_key=own_key)
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
            # The key itself is never sent back: only whether one is set and its last four characters. The hosted key:
            # nothing of it, only its model, limits and what is left today (``hosted``).
            openrouter=({"offered": True, "endpoint": openrouter.ENDPOINT, "notice": OPENROUTER_NOTICE,
                         "key_set": own_key,
                         "key_hint": openrouter.key_hint(selected.api_key_ref) if own_key else None,
                         "default_model": openrouter.presets()["default"], "hosted": hosted,
                         "models": [{"id": key, "label": entry.get("label") or key,
                                     "reasoning": bool(entry.get("reasoning", False)),
                                     "claude": openrouter.is_claude(key)}
                                    for key, entry in openrouter.presets()["models"].items()]}
                        if offered else {"offered": False}),
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
    elif selected.llm_backend == OPENROUTER:
        connection = {"endpoint": openrouter.ENDPOINT, "model": selected.model, "tokenizer": selected.tokenizer}
    else:
        connection = {"endpoint": selected.endpoint, "model": selected.model, "tokenizer": selected.tokenizer}
    managed = (selected.llm_backend in GEMINI_BACKENDS or selected.llm_backend in CLAUDE_BACKENDS
               or (selected.llm_backend == OPENROUTER and openrouter.is_claude(selected.model)))
    sampling, _ = (vendor_defaults(connection["model"], template={"enable_thinking": selected.reasoning})
                   if not managed else ({}, {}))
    return {"llm_backend": selected.llm_backend, **connection, "reasoning": selected.reasoning,
            "thinking_supported": selected.thinking_supported, "sampling": sampling,
            "backup_enabled": selected.backup_enabled, "failure_threshold": selected.failure_threshold}
