"""Connection profiles shared by the apps: an OpenAI-compatible server, the Vertex (Agent Platform) profiles,
Claude through a Databricks AI Gateway and OpenRouter with a user's own key.

A connection names where a preset section's requests go. Four kinds:

- **an OpenAI-compatible server** (vLLM): an address normalized by ``normalize_endpoint`` and a model chosen from
  the server's ``/v1/models`` (``served_models``);
- **a Vertex profile** (``VERTEX_MODELS``): a hosted Gemini or Gemma model in the site config's
  ``vertex.project``, reached with this server's Application Default Credentials. The Google destination is built
  here from the profile (``vertex_profile``), never from browser input. Hosted Gemini manages its own sampling and
  thinking (``GEMINI_BACKENDS``); Gemma MaaS takes a thinking switch and needs request pacing (``GEMMA_PACING``);
- **Claude through a Databricks AI Gateway** (``claude_gateway_endpoint``): the gateway of the site config's
  ``databricks.host``, its Anthropic Messages route, a three-part endpoint name (``CLAUDE_ENDPOINT_NAME``) and the
  Databricks CLI login profile;
- **OpenRouter** (``matchminer_ai.llm.openrouter``): any OpenRouter model ID (``presets/openrouter.yaml`` lists
  some), always at ``openrouter.ENDPOINT``, with the key a user pasted, held in this process's memory and named by
  the connection's ``api_key_ref``. Claude models there are treated as through the gateway (JSON-schema answers,
  the same efforts, no sampling settings); other models get their model profile's sampling.

The ``vertex_*`` helpers give the parts of a preset section that a Vertex profile fixes (request parameters, extra
body, chat-template switches, context window, sampling profile). ``connect_section`` and ``connect_patient_section``
apply a ``Connection`` of any kind to preset sections, for the apps and the package's command lines alike.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from . import openrouter
from .remote_auth import GOOGLE_AGENT_PLATFORM_PROVIDER, OPENROUTER_PROVIDER
from .remote_inference import normalize_openai_base_url

# ---------------------------------------------------------------------------------------- OpenAI-compatible


def normalize_endpoint(value: str) -> str:
    """Accept a server address or an OpenAI-compatible API base URL (``host:port`` gains ``http://`` and ``/v1``).

    Raises ``ValueError`` for anything but an HTTP(S) URL without credentials, query or fragment.
    """
    value = (value or "").strip()
    if not value or any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise ValueError("Enter an HTTP or HTTPS LLM server address.")
    candidate = value if "://" in value else "http://" + value
    try:
        parsed = urlsplit(candidate)
        valid = bool(
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
        )
        parsed.port  # malformed or out-of-range ports raise here, before any request
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Use an HTTP(S) server URL without credentials, query, or fragment.")
    return normalize_openai_base_url(f"{parsed.scheme}://{parsed.netloc}{parsed.path}")


async def served_models(endpoint: str, *, timeout: float = 10.0, api_key: str | None = None) -> list[dict]:
    """The models an OpenAI-compatible endpoint serves (``/models``): ``[{"id", "max_model_len"}]``, de-duplicated.

    Raises ``httpx.HTTPError``, ``ValueError``, ``KeyError`` or ``TypeError`` when the list cannot be read.
    """
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(endpoint.rstrip("/") + "/models", headers=headers)
        response.raise_for_status()
        data = response.json()["data"]
    if not isinstance(data, list):
        raise ValueError("expected a model list")
    models = [
        {"id": m["id"].strip(), "max_model_len": m.get("max_model_len")}
        for m in data
        if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"].strip() and len(m["id"]) <= 512
    ]
    return list({m["id"]: m for m in models}.values())


# ---------------------------------------------------------------------------------------- Vertex profiles

# Local chunk boundaries only; not Gemini's tokenizer or weights.
VERTEX_TOKENIZER = "Qwen/Qwen3-0.6B"
VERTEX_MODELS = {
    # Gemini 4 Hydrogen (GA 2026-11-12): 1,048,576-token window shared by input and output, up to 65,536 output.
    "gemini4_vertex": "google/gemini-4-hydrogen",
    "gemini_vertex": "google/gemini-3.8-flash",
    "gemma_vertex": "google/gemma-4-26b-a4b-it-maas",
}
VERTEX_LABELS = {
    "gemini4_vertex": "Gemini 4 Hydrogen",
    "gemini_vertex": "Gemini 3.8 Flash",
    "gemma_vertex": "Gemma 4 26B MaaS",
}
# Hosted Gemini models manage their own sampling and thinking (reasoning_effort, no chat-template switches).
GEMINI_BACKENDS = frozenset({"gemini4_vertex", "gemini_vertex"})
VERTEX_MAX_OUTPUT_TOKENS = 65536
# The Gemma MaaS endpoint's request pacing (shared remote settings).
GEMMA_PACING = {
    "request_start_interval_seconds": 2.0,
    "capacity_retry_initial_seconds": 15.0,
    "capacity_retry_max_seconds": 60.0,
}


def vertex_project() -> str:
    """The Google Cloud project of the Vertex profiles: the site config's ``vertex.project`` ("" when unset)."""
    from matchminer_ai.site import load_site_config

    return str(load_site_config().get("vertex.project") or "")


def vertex_endpoint(project: str | None = None) -> str:
    project = vertex_project() if project is None else project
    return f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi"


def vertex_profile(backend: str) -> dict:
    """``{"endpoint", "model", "tokenizer"}`` of a Vertex profile (``KeyError`` for another backend)."""
    return {"endpoint": vertex_endpoint(), "model": VERTEX_MODELS[backend], "tokenizer": VERTEX_TOKENIZER}


def vertex_profiles() -> dict[str, dict]:
    return {backend: vertex_profile(backend) for backend in VERTEX_MODELS}


def vertex_context_tokens(backend: str) -> int:
    return 1048576 if backend in GEMINI_BACKENDS else 262144


def vertex_request_params(backend: str, max_tokens: int = 32768) -> dict:
    """Request parameters a Vertex profile fixes: the output cap and, for Gemini, high reasoning effort."""
    params: dict = {"max_tokens": min(VERTEX_MAX_OUTPUT_TOKENS, int(max_tokens))}
    if backend in GEMINI_BACKENDS:
        params["reasoning_effort"] = "high"
    return params


def vertex_chat_template(backend: str, thinking: bool) -> dict:
    """Chat-template switches: none for Gemini, Gemma's thinking switch."""
    return {} if backend in GEMINI_BACKENDS else {"enable_thinking": thinking}


def vertex_extra_body(backend: str, thinking: bool) -> dict:
    template = vertex_chat_template(backend, thinking)
    return {"chat_template_kwargs": template} if template else {}


def vertex_sampling_profile(backend: str) -> str:
    """Gemini samples on its own ("none"); Gemma takes the model profile's sampling ("auto")."""
    return "none" if backend in GEMINI_BACKENDS else "auto"


def vertex_remote(backend: str) -> dict:
    """The shared ``remote`` settings of a Vertex profile: provider, project, endpoint and Gemma's pacing."""
    values = {
        "provider": GOOGLE_AGENT_PLATFORM_PROVIDER,
        "google_project_id": vertex_project(),
        "server_urls": [vertex_endpoint()],
    }
    if backend not in GEMINI_BACKENDS:
        values.update(GEMMA_PACING)
    return values


async def vertex_available(backend: str, *, timeout: float = 30) -> bool:
    """Whether this server's credentials reach a Vertex profile's model (a short request with no patient data)."""
    import asyncio

    from .vllm_server import check_openai_endpoint

    profile = vertex_profile(backend)
    try:
        checks = await asyncio.to_thread(
            check_openai_endpoint,
            profile["endpoint"],
            provider=GOOGLE_AGENT_PLATFORM_PROVIDER,
            google_project_id=vertex_project(),
            model_name=profile["model"],
            timeout=timeout,
        )
        return bool(checks and checks[0][1] == 200)
    except Exception:
        return False


CREDENTIALS_TIMEOUT = 15.0  # seconds to get a Google token before calling the credentials unavailable
_credentials: dict = {"checked": None, "error": None}
_credentials_lock = threading.Lock()


def vertex_credentials_error(max_age: float = 60.0) -> str | None:
    """None when this server's Application Default Credentials give a Google token, else why not. Checked at most
    once per ``max_age`` seconds and never for longer than ``CREDENTIALS_TIMEOUT``, so missing credentials are
    reported, not waited on."""
    with _credentials_lock:
        checked = _credentials["checked"]
        if checked is not None and time.monotonic() - checked < max_age:
            return _credentials["error"]

    def token():
        import google.auth
        from google.auth.transport.requests import Request

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        if not credentials.token:
            raise ValueError("no token")

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vertex-credentials")
    error: str | None
    try:
        pool.submit(token).result(timeout=CREDENTIALS_TIMEOUT)
        error = None
    except FutureTimeout:
        error = (
            f"Vertex credentials not available: no Google token within {CREDENTIALS_TIMEOUT:.0f} s from this "
            f"server's Application Default Credentials (project {vertex_project()})."
        )
    except Exception as exc:
        error = (
            f"Vertex credentials not available on this server ({type(exc).__name__}): set up Google Application "
            f"Default Credentials with access to project {vertex_project()}, or choose an OpenAI-compatible "
            f"endpoint."
        )
    finally:
        pool.shutdown(wait=False)
    with _credentials_lock:
        _credentials.update(checked=time.monotonic(), error=error)
    return error


# ---------------------------------------------------------------------------------------- Claude via Databricks

# Gateway endpoint names have three dot-separated parts, the last ending in the Claude model ID.
CLAUDE_ENDPOINT_NAME = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*claude-[a-z0-9-]+")
CLAUDE_CONTEXT_TOKENS = 1_000_000
DEFAULT_DATABRICKS_PROFILE = "unity-gateway"


def databricks_host() -> str:
    """The Databricks workspace whose AI Gateway serves Claude: the site config's ``databricks.host``."""
    from matchminer_ai.site import load_site_config

    return str(load_site_config().get("databricks.host") or "").rstrip("/")


def databricks_profile() -> str:
    """The Databricks CLI login profile (``databricks auth login --profile``): site ``databricks.profile``."""
    from matchminer_ai.site import load_site_config

    return str(load_site_config().get("databricks.profile") or DEFAULT_DATABRICKS_PROFILE)


def claude_gateway_endpoint(host: str | None = None) -> str:
    """The gateway's Anthropic Messages route."""
    return (databricks_host() if host is None else host.rstrip("/")) + "/ai-gateway/anthropic"


def claude_models() -> dict[str, str]:
    """The gateway's Claude endpoint names by model (site config ``databricks.endpoints``: haiku, sonnet, opus)."""
    from matchminer_ai.site import load_site_config

    endpoints = load_site_config().get("databricks.endpoints") or {}
    if not isinstance(endpoints, dict):
        return {}
    return {str(k): str(v).strip() for k, v in endpoints.items() if str(v or "").strip()}


def claude_profile() -> dict:
    """``{"endpoint", "model", "models", "tokenizer"}``: the gateway route of the site config's ``databricks.host``,
    Haiku's endpoint name first (else the first the site names), and every endpoint name by model."""
    models = claude_models()
    return {
        "endpoint": claude_gateway_endpoint(databricks_host()),
        "model": models.get("haiku") or next(iter(models.values()), ""),
        "models": models,
        # Unused: hosted prompt budgets use conservative UTF-8 bytes.
        "tokenizer": VERTEX_TOKENIZER,
    }


# ---------------------------------------------------------------------------------------- Preset sections

LOCAL_BACKEND = "local"  # an OpenAI-compatible server (vLLM)
CLAUDE_BACKEND = "claude_databricks"  # Claude through the site config's Databricks gateway
OPENROUTER_BACKEND = "openrouter"  # OpenRouter with a user's own key (matchminer_ai.llm.openrouter)
#: The sections that write queries: the compile steps of workup and trial-criteria queries, and the SQL agent.
QUERY_WRITERS = ("workup_query_compile", "criteria_query_compile", "workup_sql_agent")
# The query writers through Claude: Sonnet 5.5's default effort. On two colon items (2026-11-19) medium took the same
# time (~25 s per item) but skipped reference-date bounds and planned-only reasons that high effort checked.
CLAUDE_QUERY_WRITER_EFFORT = "high"
# Every other task: Haiku 5.5's default effort (the OA summarization test of 2026-11-20 ran at it).
CLAUDE_EFFORT = "medium"
#: Patient-level extraction prompts through Claude: their budget and Claude's tokens per chunk-tokenizer token.
CLAUDE_PATIENT_PROMPT_TOKENS = 85_000
CLAUDE_TOKEN_RATIO = 1.6


@dataclass(frozen=True)
class Connection:
    """Where one task's requests go, applied to preset sections by ``connect_section`` and
    ``connect_patient_section``.

    - ``backend``: ``local`` (an OpenAI-compatible server at ``endpoint``), ``claude_databricks`` (Claude through the
      site config's gateway, always its own route) or a Vertex profile (``VERTEX_MODELS``).
    - ``model``: the served model id, or the gateway's endpoint name (``CLAUDE_ENDPOINT_NAME``); a Vertex profile
      names its own.
    - ``tokenizer``: the local tokenizer for prompt budgets (default: ``model``).
    - ``thinking``: the chat-template thinking switch of a local model or Gemma MaaS that takes one (for OpenRouter,
      its reasoning switch); None sends none, so the model keeps its own default.
    - ``api_key_ref``: OpenRouter only, the reference of the key in this process's memory
      (``openrouter.register_key``).
    """

    backend: str = LOCAL_BACKEND
    model: str = ""
    endpoint: str = ""
    tokenizer: str = ""
    thinking: bool | None = None
    api_key_ref: str = field(default="", repr=False)


class ConnectionRefused(ValueError):
    """A connection this process's policy refuses (``set_connection_policy``); the message says why."""


# A check every builder below applies to its connection (None: none). The dashboard's --local-server-only sets one
# that accepts only the app's own local model server.
_connection_policy: Callable[[Connection], str | None] | None = None


def set_connection_policy(policy: Callable[[Connection], str | None] | None) -> None:
    """Make ``connect_section``, ``connect_remote``, ``connect_patient_section`` and ``claude_connection`` refuse
    (``ConnectionRefused``) every connection ``policy`` returns a reason for; None removes the policy."""
    global _connection_policy
    _connection_policy = policy


def check_connection(connection: Connection) -> Connection:
    """``connection``, unless the process's policy refuses it (``ConnectionRefused``)."""
    policy = _connection_policy
    reason = policy(connection) if policy is not None else None
    if reason:
        raise ConnectionRefused(reason)
    return connection


def claude_connection(name: str = "haiku") -> Connection:
    """Claude through the gateway: ``name`` is a model the site config's ``databricks.endpoints`` names (haiku,
    sonnet, opus) or a full endpoint name. Raises ``ValueError`` when the site config names no gateway or the name
    is not an endpoint."""
    profile = claude_profile()
    model = profile["models"].get(name, name)
    if not databricks_host():
        raise ValueError("The site config names no Databricks gateway (databricks.host).")
    if not CLAUDE_ENDPOINT_NAME.fullmatch(model or ""):
        raise ValueError(f"{name!r} is neither a model of the site config's databricks.endpoints nor an endpoint name.")
    return check_connection(Connection(CLAUDE_BACKEND, model, profile["endpoint"], profile["tokenizer"]))


def connect_section(section: dict, name: str, connection: Connection, *, own_connection: bool = False) -> None:
    """Point the preset section ``name`` (such as ``document_extraction``) at ``connection``, in place.

    ``own_connection``: the section names its endpoint and provider itself, so it can differ from the shared
    ``remote`` the other sections use. Claude answers in the section's JSON schema with no sampling settings, at
    ``CLAUDE_QUERY_WRITER_EFFORT`` for the query writers and ``CLAUDE_EFFORT`` otherwise; its patient-level
    extraction prompts stay under ``CLAUDE_PATIENT_PROMPT_TOKENS``. Hosted Vertex models answer free JSON that the
    package validates and retries (the SQL agent's small action schema keeps JSON-schema responses).
    """
    check_connection(connection)
    remote = dict(section["remote"])
    local = dict(section.get("local", {}))
    if connection.backend == CLAUDE_BACKEND:
        profile = claude_profile()
        effort = CLAUDE_QUERY_WRITER_EFFORT if name in QUERY_WRITERS else CLAUDE_EFFORT
        remote.update(
            provider="databricks_anthropic", server_urls=[profile["endpoint"]],
            databricks_profile=databricks_profile(), model_name=connection.model,
            tokenizer_name=profile["tokenizer"], extra_body={},
            request_params={"max_tokens": min(
                65536, remote.get("request_params", {}).get("max_tokens", 32768)),
                "reasoning_effort": effort},
        )
        local["engine"] = {**local.get("engine", {}), "max_model_len": CLAUDE_CONTEXT_TOKENS}
        local["chat_template_kwargs"] = {}
        # Claude answers in the section's JSON schema; no sampling settings (Haiku 5.5 rejects them).
        section.update(structured_outputs=True, reasoning_effort=effort, sampling_profile="none")
        if name == "patient_extraction":
            # Haiku 5.5 bills a prompt over 100,000 tokens at five times the rate (user, 2026-11-21: keep
            # patient-level prompts under it): 85,000 for the notes and record, leaving room for the response
            # schema (input too, about 8,000), at about 1.6 Claude tokens per chunk-tokenizer token (measured).
            section.update(prompt_tokens=CLAUDE_PATIENT_PROMPT_TOKENS, prompt_token_ratio=CLAUDE_TOKEN_RATIO)
    elif connection.backend in VERTEX_MODELS:
        backend = connection.backend
        profile = vertex_profile(backend)
        thinking = bool(connection.thinking)
        remote.update(
            model_name=profile["model"], tokenizer_name=profile["tokenizer"],
            request_params=vertex_request_params(backend, remote.get("request_params", {}).get("max_tokens", 32768)),
            extra_body=vertex_extra_body(backend, thinking),
        )
        if own_connection:
            remote.update({k: v for k, v in vertex_remote(backend).items()
                           if k in ("provider", "google_project_id", "server_urls")})
        local["chat_template_kwargs"] = vertex_chat_template(backend, thinking)
        local["engine"] = {**local.get("engine", {}), "max_model_len": vertex_context_tokens(backend)}
        section.update(structured_outputs=name == "workup_sql_agent", reasoning_effort="high",
                       sampling_profile=vertex_sampling_profile(backend))
    elif connection.backend == OPENROUTER_BACKEND:
        settings = openrouter.model_settings(connection.model)
        remote.update(model_name=connection.model, tokenizer_name=connection.tokenizer or settings["tokenizer"])
        if own_connection:
            remote.update(openrouter_remote(connection))
        local["engine"] = {**local.get("engine", {}), "max_model_len": settings["context"]}
        params = dict(remote.get("request_params") or {})
        params["max_tokens"] = min(openrouter.MAX_OUTPUT_TOKENS, params.get("max_tokens", 32768))
        if openrouter.is_claude(connection.model):
            # As through the gateway: JSON-schema answers at the same efforts, no sampling settings.
            effort = CLAUDE_QUERY_WRITER_EFFORT if name in QUERY_WRITERS else CLAUDE_EFFORT
            remote.update(extra_body={}, request_params={"max_tokens": params["max_tokens"], "reasoning_effort": effort})
            local["chat_template_kwargs"] = {}
            section.update(structured_outputs=True, reasoning_effort=effort, sampling_profile="none")
            if name == "patient_extraction":
                section.update(prompt_tokens=CLAUDE_PATIENT_PROMPT_TOKENS, prompt_token_ratio=CLAUDE_TOKEN_RATIO)
        else:
            remote["request_params"] = params
            # JSON-schema answers (OpenRouter's require_parameters keeps to providers that honour them) and the
            # model profile's sampling. The request carries a reasoning switch, not an effort, so the section's
            # effort goes too (the profile's default then applies; Qwen 3.8's profile refuses "high").
            section.update(structured_outputs=True, sampling_profile="auto")
            section.pop("reasoning_effort", None)
            if connection.thinking is not None:
                template = {"enable_thinking": connection.thinking}
                remote["extra_body"] = {**remote.get("extra_body", {}), "chat_template_kwargs": template.copy()}
                local["chat_template_kwargs"] = template.copy()
    elif connection.backend == LOCAL_BACKEND:
        remote.update(model_name=connection.model, tokenizer_name=connection.tokenizer or connection.model)
        if own_connection:
            remote.update(provider="openai", server_urls=[connection.endpoint])
        if name == "workup_sql_agent":
            # Read the served context from /v1/models rather than assume the preset's.
            local["engine"] = {k: v for k, v in local.get("engine", {}).items() if k != "max_model_len"}
        if connection.thinking is not None:
            template = {"enable_thinking": connection.thinking}
            remote["extra_body"] = {**remote.get("extra_body", {}), "chat_template_kwargs": template.copy()}
            local["chat_template_kwargs"] = template.copy()
    else:
        raise ValueError("Unsupported LLM backend")
    section.update(remote=remote, local=local)


def openrouter_remote(connection: Connection) -> dict:
    """The ``remote`` settings of an OpenRouter connection: provider, the fixed endpoint and the key's reference."""
    return {"provider": OPENROUTER_PROVIDER, "server_urls": [openrouter.ENDPOINT],
            "api_key_ref": connection.api_key_ref}


def connect_remote(config, connection: Connection) -> None:
    """Point the shared ``remote`` (the endpoint, provider and credentials scope every section uses but the SQL
    agent's own) at ``connection`` and enable it, in place. Concurrency and timeouts stay the caller's."""
    check_connection(connection)
    config.remote["enabled"] = True
    if connection.backend == LOCAL_BACKEND:
        config.remote["server_urls"] = [connection.endpoint]
    elif connection.backend in VERTEX_MODELS:
        # Provider, project, endpoint and, for Gemma MaaS, its request pacing.
        config.remote.update(vertex_remote(connection.backend))
    elif connection.backend == CLAUDE_BACKEND:
        # The gateway's Messages route with this server's Databricks CLI profile.
        config.remote.update(provider="databricks_anthropic", server_urls=[claude_profile()["endpoint"]],
                             databricks_profile=databricks_profile())
    elif connection.backend == OPENROUTER_BACKEND:
        config.remote.update(openrouter_remote(connection))
        # One fixed endpoint: no endpoint pool or adaptive concurrency (those are for vLLM servers).
        config.remote.setdefault("adaptive_concurrency", {})["enabled"] = False
    else:
        raise ValueError("Unsupported LLM backend")


def connect_patient_section(config, connection: Connection) -> None:
    """Point the ``patient`` section (serial summaries, full-note and note-search workup reviews) and the shared
    ``remote`` (``connect_remote``) at ``connection``, in place. Concurrency, timeouts and prompt-building workers
    stay the caller's."""
    check_connection(connection)
    connect_remote(config, connection)
    patient = config.patient
    patient["remote"].update(model_name=connection.model, tokenizer_name=connection.tokenizer or connection.model)
    patient["sampling_profile"] = "auto"
    patient["reasoning_effort"] = "xhigh"
    template = {} if connection.thinking is None else {"enable_thinking": connection.thinking}
    patient.setdefault("local", {})["chat_template_kwargs"] = template.copy()
    patient["remote"]["extra_body"] = {"chat_template_kwargs": template.copy()} if template else {}
    if connection.backend in VERTEX_MODELS:
        backend = connection.backend
        profile = vertex_profile(backend)
        thinking = bool(connection.thinking)
        context_window = vertex_context_tokens(backend)
        patient.update(sampling_profile=vertex_sampling_profile(backend), reasoning_effort="high",
                       context_window=context_window, tokenizer_mode="bytes")
        patient["local"]["engine"]["max_model_len"] = context_window
        patient["local"]["chat_template_kwargs"] = vertex_chat_template(backend, thinking)
        patient["remote"].update(
            model_name=profile["model"],
            tokenizer_name=profile["tokenizer"],
            request_params=vertex_request_params(backend, VERTEX_MAX_OUTPUT_TOKENS),
            extra_body=vertex_extra_body(backend, thinking),
        )
    elif connection.backend == OPENROUTER_BACKEND:
        settings = openrouter.model_settings(connection.model)
        claude = openrouter.is_claude(connection.model)
        # Hosted prompt budgets count conservative UTF-8 bytes; Claude gets no sampling settings.
        patient.update(sampling_profile="none" if claude else "auto", context_window=settings["context"],
                       tokenizer_mode="bytes", **({"reasoning_effort": CLAUDE_EFFORT} if claude else {}))
        patient["local"]["engine"]["max_model_len"] = settings["context"]
        params = {**patient["remote"].get("request_params", {}), "max_tokens": openrouter.MAX_OUTPUT_TOKENS}
        if claude:
            patient["local"]["chat_template_kwargs"] = {}
            params = {"max_tokens": openrouter.MAX_OUTPUT_TOKENS, "reasoning_effort": CLAUDE_EFFORT}
        patient["remote"].update(model_name=connection.model,
                                 tokenizer_name=connection.tokenizer or settings["tokenizer"],
                                 request_params=params, **({"extra_body": {}} if claude else {}))
    elif connection.backend == CLAUDE_BACKEND:
        profile = claude_profile()
        # Hosted prompt budgets count conservative UTF-8 bytes; no sampling settings.
        patient.update(sampling_profile="none", reasoning_effort=CLAUDE_EFFORT, context_window=CLAUDE_CONTEXT_TOKENS,
                       tokenizer_mode="bytes")
        patient["local"]["engine"]["max_model_len"] = CLAUDE_CONTEXT_TOKENS
        patient["local"]["chat_template_kwargs"] = {}
        patient["remote"].update(
            model_name=connection.model, tokenizer_name=profile["tokenizer"], extra_body={},
            request_params={"max_tokens": VERTEX_MAX_OUTPUT_TOKENS, "reasoning_effort": CLAUDE_EFFORT},
        )
