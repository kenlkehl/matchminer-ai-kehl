"""The LLM endpoint the dashboard's LLM checks use (the GoodOption check and the LLM decision checks: match
scores, common exclusions and per-criterion checks, ``llm_decisions``).

The connection profiles are the package's (``matchminer_ai.llm.connections``, shared with the guideline features):
``normalize_endpoint``, the Vertex profiles and the parts of a section they fix, model discovery from
``/v1/models`` and the Vertex credentials check. Two kinds of connection:

- a Vertex profile, by default Gemini 4 Hydrogen (the fetcher's ``gemini4_vertex``; user, 2026-11-17): the site
  config's ``vertex.project``, this server's Application Default Credentials, hosted Gemini managing its own sampling
  and thinking.
  The Google destination is built here from the profile, never from browser input. The credentials are checked
  before a request (at most once a minute, bounded by a timeout), so their absence is reported, not waited on;
- an OpenAI-compatible server (vLLM): a URL, by default the site config's ``endpoints.teacher.url`` (an on-prem
  Nemotron 3.5 Super server), and a model chosen from the server's ``/v1/models``.

Sampling comes from the inference package's model profiles: each check's section gets ``sampling_profile: auto`` and
loses the section's own fixed sampling keys (the GoodOption preset's greedy decoding), so the served model's profile
applies (Nemotron 3.5 Super: NVIDIA's). Hosted Gemini gets ``sampling_profile: none`` and no sampling keys. The default is
set at startup (``--llm-backend``, ``--llm-endpoint``, ``--llm-model``); each browser session can choose its own per check
on Data and Models (``Connection.for_task`` of the session's ``good_option`` and ``llm_decisions`` tasks); nothing is
written to disk.

Claude through the gateway returns its thinking only as summaries, and only when asked: ``configure(...,
thinking_summaries=True)`` asks for them (the LLM decision checks show them; billing is the same).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field

from matchminer_ai.llm import connections
from matchminer_ai.llm.connections import normalize_endpoint, vertex_project

NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
OPENAI = "openai"  # an OpenAI-compatible server (vLLM)
CLAUDE = "claude_databricks"  # Claude through the Databricks gateway (guidelines.settings.claude_backends)
VERTEX_MODELS = connections.VERTEX_MODELS
VERTEX_LABELS = {key: f"{label} (Vertex)" for key, label in connections.VERTEX_LABELS.items()}
GEMINI_BACKENDS = connections.GEMINI_BACKENDS  # manage their own sampling and thinking
BACKENDS = (OPENAI, *VERTEX_MODELS, CLAUDE)


def claude_name(endpoint: str) -> str:
    """"Claude Haiku 5.5" from a gateway endpoint name ending in claude-haiku-5-5 (else the name)."""
    import re

    match = re.search(r"claude-([a-z]+)-(\d+)-(\d+)$", endpoint or "")
    return f"Claude {match.group(1).title()} {match.group(2)}.{match.group(3)}" if match else endpoint


def default_backend() -> str:
    """Claude Haiku 5.5 through the gateway when the site config names it (user, 2026-11-21), else Gemini 4 Hydrogen."""
    from .guidelines.settings import default_backend as guideline_default

    return guideline_default()


def default_model() -> str:
    """The default backend's model: the gateway's Haiku endpoint, else the on-prem teacher's."""
    from .guidelines.settings import claude_backends

    return claude_backends()[CLAUDE]["model"] if default_backend() == CLAUDE else NEMOTRON


def default_endpoint() -> str:
    """The OpenAI-compatible server offered first: the site config's ``endpoints.teacher.url`` ("" for none)."""
    from matchminer_ai.site import load_site_config

    return load_site_config().get("endpoints.teacher.url") or ""


NOTICE = "The patient summary is sent to the selected LLM endpoint; use an endpoint authorized for the data."
# The LLM decision checks also send the cohort's criteria (protocol text, for a protocol cohort).
CLAUDE_THINKING_SUMMARIES = {"type": "adaptive", "display": "summarized"}
# Sampling keys a section may fix; removed so the served model's profile supplies them.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
                 "frequency_penalty")
MAX_IN_FLIGHT = 16  # requests the dashboard keeps in flight on one endpoint (a shared, busy server)


@dataclass(frozen=True)
class Connection:
    backend: str = OPENAI
    endpoint: str = field(default_factory=default_endpoint)  # "" for none (the LLM checks are then disabled)
    model: str = NEMOTRON
    tokenizer: str = ""  # default: the model

    @classmethod
    def make(cls, backend: str = OPENAI, endpoint: str | None = None, model: str | None = None,
             tokenizer: str | None = None) -> "Connection":
        """A validated connection; a Vertex profile fixes its own endpoint, model and tokenizer."""
        backend = (backend or OPENAI).strip()
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {', '.join(BACKENDS)}")
        if backend in VERTEX_MODELS:
            profile = connections.vertex_profile(backend)
            return cls(backend, profile["endpoint"], profile["model"], profile["tokenizer"])
        if backend == CLAUDE:  # the server fixes the gateway; the model is one of its endpoint names
            from .guidelines.settings import claude_backends

            profile = claude_backends()[CLAUDE]
            return cls(CLAUDE, profile["endpoint"], (model or "").strip() or profile["model"], profile["tokenizer"])
        endpoint = (endpoint or "").strip()
        if endpoint.lower() in ("", "none", "off"):
            return cls(OPENAI, "", (model or "").strip(), (tokenizer or "").strip())
        model = (model or "").strip()
        if not model or len(model) > 512:
            raise ValueError("Choose the model the endpoint serves (check the endpoint for its models).")
        return cls(OPENAI, normalize_endpoint(endpoint), model, (tokenizer or "").strip())

    @classmethod
    def for_task(cls, settings) -> "Connection":
        """A session's connection for a check: its task's ``guidelines.settings.Settings`` (backend ``local`` is an
        OpenAI-compatible server)."""
        backend = OPENAI if settings.llm_backend == "local" else settings.llm_backend  # Vertex and Claude keep theirs
        return cls.make(backend, settings.endpoint, settings.model, settings.tokenizer)

    @property
    def configured(self) -> bool:
        return bool(self.endpoint and self.model)

    @property
    def vertex(self) -> bool:
        return self.backend in VERTEX_MODELS

    @property
    def gemini(self) -> bool:
        return self.backend in GEMINI_BACKENDS

    @property
    def claude(self) -> bool:
        return self.backend == CLAUDE

    @property
    def label(self) -> str:
        if self.claude:
            return f"{claude_name(self.model)} (Databricks gateway)"
        return VERTEX_LABELS[self.backend] if self.vertex else f"{self.model} at {self.endpoint}"

    def identity(self) -> dict:
        """What the cached answers are keyed on."""
        return {"backend": self.backend, "endpoint": self.endpoint, "model": self.model}

    def describe(self) -> dict:
        return {**asdict(self), "configured": self.configured, "label": self.label, "vertex": self.vertex,
                "project": vertex_project() if self.vertex else None}

    # ------------------------------------------------------------------ the inference package's configuration

    def configure(self, config, sections: tuple[str, ...], *, thinking_summaries: bool = False):
        """A copy of ``config`` sending ``sections`` (``llm_good_option``, ``llm_match_quality``,
        ``llm_exclusion_criteria``, ``llm_criteria_check``) to this connection, with the served model's sampling
        profile and debug mode (answers, reasoning, parse details). ``thinking_summaries``: Claude returns summaries
        of its thinking (the 5.5 models omit them by default)."""
        if not self.configured:
            raise ValueError("No LLM endpoint is configured (Data and models).")
        c = deepcopy(config)
        c.debug_mode = True
        c.remote.update(enabled=True, server_urls=[self.endpoint], max_concurrent_requests=MAX_IN_FLIGHT,
                        request_timeout=3600, provider="openai")
        c.remote.setdefault("adaptive_concurrency", {}).update(enabled=False)
        if self.vertex:  # provider and project and, for Gemma MaaS, its pacing (the fetcher's)
            c.remote.update({k: v for k, v in connections.vertex_remote(self.backend).items() if k != "server_urls"})
        if self.claude:
            from .guidelines.settings import CLAUDE_EFFORT, databricks_profile

            c.remote.update(provider="databricks_anthropic", databricks_profile=databricks_profile())
        for name in sections:
            section = getattr(c, name) if hasattr(c, name) else c.raw.get(name)
            if section is None:
                raise ValueError(f"the inference package's preset has no {name} section")
            remote = deepcopy(section.get("remote") or {})
            local = deepcopy(section.get("local") or {})
            params = {k: v for k, v in (remote.get("request_params") or {}).items() if k not in SAMPLING_KEYS}
            extra = {k: v for k, v in (remote.get("extra_body") or {}).items() if k not in SAMPLING_KEYS}
            local["generation"] = {k: v for k, v in (local.get("generation") or {}).items() if k not in SAMPLING_KEYS}
            remote.update(model_name=self.model, tokenizer_name=self.tokenizer or self.model)
            if self.claude:
                # Claude answers in the section's JSON schema; no sampling settings (Haiku 5.5 rejects them).
                params = {"max_tokens": min(65536, params.get("max_tokens", 32768)), "reasoning_effort": CLAUDE_EFFORT}
                if thinking_summaries:
                    params["thinking"] = dict(CLAUDE_THINKING_SUMMARIES)
                extra = {}
                local["chat_template_kwargs"] = {}
                local["engine"] = {**(local.get("engine") or {}), "max_model_len": connections.CLAUDE_CONTEXT_TOKENS}
                section["sampling_profile"] = "none"
                section["structured_outputs"] = True
            elif self.vertex:
                params.update(connections.vertex_request_params(self.backend, params.get("max_tokens", 32768)))
                extra = connections.vertex_extra_body(self.backend, True)
                if self.gemini:
                    local["chat_template_kwargs"] = {}
                local["engine"] = {**(local.get("engine") or {}),
                                   "max_model_len": connections.vertex_context_tokens(self.backend)}
                section["sampling_profile"] = connections.vertex_sampling_profile(self.backend)
                section["structured_outputs"] = False  # hosted models answer free JSON that the package validates
            else:
                section["sampling_profile"] = "auto"
            remote.update(request_params=params, extra_body=extra)
            section.update(remote=remote, local=local)
            if name == "llm_criteria_check":
                section.update(chunk_size=MAX_IN_FLIGHT, chunks_in_flight=1)
        return c
