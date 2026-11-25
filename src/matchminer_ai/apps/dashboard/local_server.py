"""The local model server of Data and models: one vLLM or llama.cpp server on this machine at a time, for the app's
LLM tasks (``matchminer_ai.llm.local_server``).

- **Who:** every dashboard has it, in both modes, and anyone using the app may start, stop and configure it (user,
  2026-11-24).
- **Only the local server** (``--local-server-only``): no other connection is accepted (``refusal``), wherever one
  is chosen (``GuidelineState.update``, the connection check) or read back (``GuidelineState.current_settings``
  drops a saved one that is not the ready server); the ready server is every task's default, and until it is ready
  LLM tasks have no connection and say so. ``--allow-openrouter`` (synthetic mode) also accepts OpenRouter with a
  session's own key, e.g. while the server loads or on a machine without a GPU (a Hugging Face Space).
- **OpenRouter** is refused in real mode, always (``refusal``), by the same policy the package's connection builders
  apply (``connection_policy``).
- **GPUs:** discovered with ``llm.gpus``; each GPU's memory is split into this app (its process and its children),
  the local server (its process group) and other processes. Process IDs are not sent to the page.
- **Use:** once the server is ready, a session's tasks can use it (``use``), as an OpenAI-compatible connection with
  the server's endpoint, model and tokenizer. With ``--local-server-default`` it also becomes the default connection
  of every task in sessions that chose none, until it stops.
- **Lifecycle:** the server runs in its own thread and process group, not in a job (it outlives jobs); the app stops
  it when it exits (``shutdown``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import threading
import time

from matchminer_ai.llm import local_server as ls
from matchminer_ai.llm.gpus import GPU, discover_gpus

from matchminer_ai.llm.connections import normalize_endpoint

from .core.errors import ConflictError, InputError
from .llm import Connection

ENGINE_CACHE_SECONDS = 60
ONLY = ("This dashboard uses only its own local model server (--local-server-only); other model connections are "
        "refused. Use the local server on Data and models.")
ONLY_OR_OPENROUTER = ("This dashboard uses only its own local model server (--local-server-only) or OpenRouter with "
                      "your own key; other model connections are refused.")
OPENROUTER_BACKEND = "openrouter"
OPENROUTER_REAL = "OpenRouter is offered in synthetic mode only; real mode never sends patient data to it."
NOT_READY = ("This dashboard uses only its own local model server (--local-server-only), which is not ready: start it "
             "on Data and models.")


def load_spec(value: str) -> dict:
    """An autostart spec: JSON text, or a JSON or YAML file."""
    text = value.strip()
    if not text.startswith("{"):
        path = Path(text).expanduser()
        if not path.is_file():
            raise ValueError(f"--local-server-autostart: {path} is neither JSON nor a file")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            loaded = yaml.safe_load(text)
            if not isinstance(loaded, dict):
                raise ValueError("--local-server-autostart: the file must hold one mapping")
            return loaded
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("--local-server-autostart: the spec must be a JSON object")
    return loaded


def ancestors(pid: int, limit: int = 64) -> set[int]:
    """``pid``'s parent chain (Linux ``/proc``; empty elsewhere)."""
    chain: set[int] = set()
    for _ in range(limit):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
        if pid <= 1 or pid in chain:
            break
        chain.add(pid)
    return chain


def connection_of(server) -> dict:
    """A server as a model connection: an OpenAI-compatible server (backend ``local``)."""
    return {"llm_backend": "local", "endpoint": server.endpoint, "model": server.spec.name,
            "tokenizer": server.spec.tokenizer_repo}


class LocalServerControl:
    def __init__(self, settings, guidelines, llm_checks=None, log=print, gpus=discover_gpus):
        self.only = bool(getattr(settings, "local_server_only", False))
        self.make_default = self.only or bool(getattr(settings, "local_server_default", False))
        self.real = str(getattr(settings, "mode", "synthetic")) == "real"
        # With --local-server-only: also OpenRouter with a session's own key (never in real mode).
        self.allow_openrouter = bool(getattr(settings, "allow_openrouter", False)) and not self.real
        self.llama_server = getattr(settings, "llama_server", None)
        self.allow_downloads = bool(getattr(settings, "allow_downloads", False))
        self.device = str(getattr(settings, "device", "") or "")
        self.log_dir = Path(settings.cache_dir) / "local_server"
        self.guidelines, self.llm_checks, self.log, self.discover = guidelines, llm_checks, log, gpus
        self.server: ls.ManagedServer | None = None
        self.download = True  # tests start fake engines without the download step
        self.lock = threading.RLock()
        self._previous: tuple | None = None
        self._engines: tuple[float, dict] | None = None

    # ------------------------------------------------------------------ what the page shows

    def engines(self) -> dict:
        with self.lock:
            if self._engines is None or time.monotonic() - self._engines[0] > ENGINE_CACHE_SECONDS:
                self._engines = (time.monotonic(), ls.engines(self.llama_server))
            return self._engines[1]

    def gpu_list(self) -> list[GPU]:
        return list(self.discover())

    def gpus(self) -> dict:
        """Each GPU with its memory split into this app, the local server and other processes."""
        found = self.gpu_list()
        me = os.getpid()
        server = self.server
        group = server.process.pid if server and server.process and server.active else None
        decision = int(self.device.split(":")[1]) if self.device.startswith("cuda:") and \
            self.device.split(":")[1].isdigit() else (0 if self.device == "cuda" else None)
        rows = []
        for g in found:
            parts = {"dashboard": 0, "local_server": 0}
            for p in g.processes:
                try:
                    pgid = os.getpgid(p.pid)
                except OSError:
                    pgid = None
                owner = ("local_server" if group is not None and pgid == group else
                         "dashboard" if p.pid == me or me in ancestors(p.pid) else None)
                if owner and p.used:
                    parts[owner] += p.used
            used = g.memory_used if g.memory_used is not None else None
            rows.append({"index": g.index, "name": g.name, "uuid": g.uuid, "compute_capability": g.compute_capability,
                         "memory_total": g.memory_total, "memory_used": used, "memory_free": g.memory_free,
                         "utilization": g.utilization, "processes": len(g.processes), "source": g.source,
                         "dashboard": parts["dashboard"], "local_server": parts["local_server"],
                         "other": max(0, used - parts["dashboard"] - parts["local_server"]) if used is not None
                         else None, "decision_model": decision == g.index,
                         "server_gpu": bool(server and server.active and g.index in server.spec.gpus)})
        return {"gpus": rows, "checked_at": time.time(),
                "note": None if any(r["processes"] for r in rows) or not rows else
                "The driver lists no processes here (a container), so all used memory shows as other."}

    def chosen(self, indices) -> list[GPU]:
        found = {g.index: g for g in self.gpu_list()}
        missing = [i for i in indices if i not in found]
        if missing:
            raise InputError(f"GPU {missing[0]} is not visible to this app.")
        return [found[i] for i in indices]

    def recipe(self, model: str, engine: str | None, gpus: list[int] | None) -> dict:
        model = (model or ls.recipes().get("default") or "").strip()
        if not ls.REPO_ID.match(model):
            raise InputError("Enter the model as a Hugging Face repository, such as org/model.")
        if engine not in (None, "", *ls.ENGINES):
            raise InputError("Unknown engine.")
        found = self.gpu_list()
        if gpus is None:  # the GPU with the most free memory
            best = max(found, key=lambda g: g.memory_free or 0, default=None)
            chosen = [best] if best else []
        else:
            chosen = self.chosen(gpus)
        return ls.recommend(model, engine or None, chosen, self.engines())

    def preview(self, values: dict) -> dict:
        """The command and fit warnings of a spec as the form holds it (nothing is started)."""
        try:
            spec = ls.ServerSpec.from_dict(values)
        except (TypeError, ValueError) as exc:
            return {"valid": False, "error": str(exc), "command": None, "warnings": []}
        gpus = self.gpu_list()
        try:
            downloads = ls.plan_downloads(spec)
            error = None
        except (LookupError, ValueError) as exc:
            downloads, error = None, str(exc)
        command, env = ls.build_command(spec, 0, gpus=gpus, parent_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                                        engine_command=self.engines()[spec.engine]["command"])
        prefix = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
        return {"valid": error is None, "error": error, "warnings": ls.fit_warnings(spec, gpus, downloads),
                "command": f"{prefix} {shlex.join(command)}".strip().replace("--port 0", "--port <free port>"),
                "download_bytes": sum(d.size for d in downloads) if downloads else None}

    def describe(self, log_lines: int = 200) -> dict:
        server = self.server
        return {"only": self.only, "allow_openrouter": self.allow_openrouter,
                "server": server.describe(log_lines) if server else None,
                "default_on_ready": self.make_default, "allow_downloads": self.allow_downloads,
                "hub_cache": str(ls.hub_cache()), "default_model": ls.recipes().get("default"),
                "known_models": [{"model": key, "label": entry.get("label") or key}
                                 for key, entry in ls.recipes()["models"].items()]}

    def connection(self) -> dict | None:
        """The ready server as a model connection (``GuidelineState.update`` values), else None."""
        server = self.server
        return connection_of(server) if server is not None and server.state == "ready" else None

    def connection_view(self) -> dict:
        """``/api/models``'s ``local_server``: whether it is the only connection allowed, its state and, once
        ready, its connection."""
        server = self.server
        connection = self.connection() or {}
        return {"only": self.only, "state": server.state if server else None,
                "endpoint": connection.get("endpoint"), "model": connection.get("model"),
                "tokenizer": connection.get("tokenizer"), "allow_openrouter": self.allow_openrouter}

    @property
    def restricts(self) -> bool:
        """Whether ``refusal`` refuses anything: with --local-server-only, and in real mode (OpenRouter)."""
        return self.only or self.real

    def connection_policy(self, connection) -> str | None:
        """``llm.connections``' policy with ``--local-server-only`` (``set_connection_policy``): every builder refuses
        a destination other than the ready local server. A connection with no endpoint (no destination: the
        default before the server is ready) passes; the LLM tasks refuse it themselves."""
        if connection.backend == "local" and not connection.endpoint:
            return None
        return self.refusal({"llm_backend": connection.backend, "endpoint": connection.endpoint,
                             "model": connection.model})

    def refusal(self, values) -> str | None:
        """Why a connection (``llm_backend``, ``endpoint``, ``model``) is refused: with ``--local-server-only``,
        anything but this app's own ready local server (by endpoint and served name) and, with --allow-openrouter,
        OpenRouter; in real mode, OpenRouter. None when it is allowed."""
        if (values.get("llm_backend") or "local") == OPENROUTER_BACKEND:
            if self.real:
                return OPENROUTER_REAL
            return ONLY if self.only and not self.allow_openrouter else None
        if not self.only:
            return None
        server = self.server
        refused = ONLY_OR_OPENROUTER if self.allow_openrouter else ONLY
        if server is None or server.state != "ready":
            return NOT_READY + (" Or use OpenRouter with your own key." if self.allow_openrouter else "")
        try:
            endpoint = normalize_endpoint(str(values.get("endpoint") or ""))
        except ValueError:
            return refused
        if (values.get("llm_backend") or "local") != "local" or endpoint != server.endpoint \
                or values.get("model") != server.spec.name:
            return refused
        return None

    # ------------------------------------------------------------------ control

    def complete(self, values: dict) -> dict:
        """``values`` over the matching recipe's settings, as the form would start them: a known model's engine
        settings and its variant for the chosen GPUs' free memory (``llm.local_server.recommend``), else generic
        ones. Engine, model and GPUs are enough (no engine: the proposed one; no GPUs: the GPU with the most free
        memory). The model becomes the engine's repository (a known model's GGUF repository for llama.cpp)."""
        unknown = sorted(set(values) - set(ls.ServerSpec().to_dict()))
        if unknown:
            raise InputError(f"Unknown server settings: {', '.join(unknown)}.")
        gpus = values.get("gpus")
        recipe = self.recipe(str(values.get("model") or ""), values.get("engine") or None,
                             list(gpus) if gpus is not None else None)
        return {**recipe["spec"], **{k: v for k, v in values.items() if k != "model"}}

    def start(self, values: dict, *, fill: bool = False) -> dict:
        """Start a server (``fill``: complete ``values`` from the recipe first, as autostart does)."""
        if fill:
            values = self.complete(values)
        try:
            spec = ls.ServerSpec.from_dict(values)
        except (TypeError, ValueError) as exc:
            raise InputError(str(exc)) from None
        engine = self.engines()[spec.engine]
        if not engine["available"]:
            raise InputError(engine["reason"] or f"{ls.ENGINES[spec.engine]} is not available.")
        gpus = self.gpu_list()
        visible = {g.index for g in gpus}
        if any(i not in visible for i in spec.gpus):
            raise InputError("A chosen GPU is not visible to this app.")
        with self.lock:
            if self.server is not None and self.server.active:
                raise ConflictError("A local server is already running; stop it first.")
            self.server = ls.ManagedServer(spec, self.log_dir, engine_command=engine["command"], gpus=gpus,
                                           download=self.download, on_ready=self._ready,
                                           on_end=self._ended).start()
        self.log(f"Local model server: starting {spec.name} with {ls.ENGINES[spec.engine]} on GPUs "
                 f"{spec.gpus or 'none'}")
        return self.describe()

    def stop(self) -> dict:
        server = self.server
        if server is not None and server.active:
            self.log("Local model server: stopping")
            server.stop()
        return self.describe()

    def shutdown(self) -> None:
        """Stop the server when the app exits (``__main__``'s ``finally`` and ``atexit``)."""
        server = self.server
        if server is not None and server.active:
            server.stop()

    def use(self, owner: str, *, task: str | None, apply_to_all: bool, reasoning: bool):
        """Save the ready server as the session's connection for ``task`` (or every task)."""
        connection = self.connection()
        if connection is None:
            raise InputError("The local server is not ready.")
        values = {**connection, "reasoning": bool(reasoning)}
        if apply_to_all:
            task = task or "summarize"
        elif task is None:
            raise InputError("Choose a task, or use the server for every task.")
        return self.guidelines.update(owner, values, task=task, apply_to_all=apply_to_all)

    # ------------------------------------------------------------------ the default connection

    def _ready(self, server) -> None:
        if not self.make_default or server is not self.server:
            return
        connection = connection_of(server)
        _, recipe = ls.find_recipe(server.spec.model)
        values = {**connection, "reasoning": bool((recipe or {}).get("reasoning", False))}
        previous = self.guidelines.set_default_connection(values)
        previous_llm = None
        if self.llm_checks is not None:
            previous_llm = self.llm_checks.connection
            self.llm_checks.set_connection(Connection.make("openai", connection["endpoint"], connection["model"],
                                                           connection["tokenizer"]))
        with self.lock:
            self._previous = (previous, previous_llm, connection["endpoint"])
        self.log(f"Local model server: ready; the default connection is now {connection['model']} at "
                 f"{connection['endpoint']}")

    def _ended(self, server) -> None:
        """Give the default connection back once the server that took it is gone."""
        with self.lock:
            previous, self._previous = self._previous, None
        if previous is None:
            return
        base, llm, endpoint = previous  # base: the previous (settings, inference) pair
        if self.guidelines.settings.endpoint == endpoint:
            self.guidelines.restore_default_connection(base)
        if llm is not None and self.llm_checks is not None and self.llm_checks.connection.endpoint == endpoint:
            self.llm_checks.set_connection(llm)
        self.log(f"Local model server: {server.state}; the default connection is restored")
