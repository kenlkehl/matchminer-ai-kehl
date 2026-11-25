"""The assistant's model connection: the session's ``assistant`` task connection as a structured-JSON client.

Every backend the app offers works, as for the other LLM tasks (``guidelines.settings``):

- Claude through the Databricks gateway (provider ``databricks_anthropic``: the official anthropic SDK on the gateway's
  Messages route, with the server's Databricks CLI profile), the default with the site config's Opus endpoint. The
  conversation so far is cached (``cache_control``), so each step of a turn reads the earlier steps from the cache;
- a Vertex profile (the server's Application Default Credentials);
- an OpenAI-compatible server (its context window read from ``/v1/models``).

Every answer is one JSON object in the action schema (``actions``), constrained by the endpoint's JSON-schema
response format and validated in code. Requests and answers stay in memory (``StructuredClient`` without a
checkpoint folder): nothing of a conversation is written to disk or logged; failures keep only their HTTP status.
"""
from __future__ import annotations

from dataclasses import replace

from matchminer_ai.llm import connections
from matchminer_ai.llm.remote_auth import DATABRICKS_ANTHROPIC_PROVIDER
from matchminer_ai.llm.structured import StructuredConfig, resolve_structured_config
from matchminer_ai.patients.note_search_qa import _MeasuredClient

from ..guidelines.settings import (
    CLAUDE_ASSISTANT_EFFORT, CLAUDE_BACKENDS, CLAUDE_CONTEXT, VERTEX_BACKENDS, claude_backends, databricks_profile,
)

#: Output tokens per step: the answer's Markdown and, for Claude, its thinking.
MAX_OUTPUT_TOKENS = 16000
#: Seconds a step's request may take (a busy endpoint's 429/503 is waited out within the attempts).
REQUEST_TIMEOUT = 600.0
ATTEMPTS = 3


def structured_config(selected) -> StructuredConfig:
    """The request configuration of a task connection (``guidelines.settings.Settings.for_task("assistant")``)."""
    selected.require_connection()
    backend = selected.llm_backend
    runtime = {"max_retries": ATTEMPTS, "request_timeout": REQUEST_TIMEOUT, "max_concurrent_requests": 8,
               "response_format": "json_schema", "tokenizer_mode": "bytes", "stream": True,
               "capacity_retry_initial_seconds": 10.0, "capacity_retry_max_seconds": 60.0}
    if backend in CLAUDE_BACKENDS:
        # The server fixes the gateway URL; the session chose only the gateway's endpoint name.
        profile = claude_backends()[backend]
        runtime.update(provider=DATABRICKS_ANTHROPIC_PROVIDER, server_urls=[profile["endpoint"]],
                       databricks_profile=databricks_profile(), model_name=selected.model,
                       context_window=CLAUDE_CONTEXT, sampling_profile="none",
                       remote={"request_params": {"max_tokens": MAX_OUTPUT_TOKENS,
                                                  "reasoning_effort": CLAUDE_ASSISTANT_EFFORT},
                               "extra_body": {}})
    elif backend in VERTEX_BACKENDS:
        profile = connections.vertex_profile(backend)
        runtime.update(connections.vertex_remote(backend))
        runtime.update(model_name=profile["model"], context_window=connections.vertex_context_tokens(backend),
                       sampling_profile=connections.vertex_sampling_profile(backend),
                       remote={"request_params": connections.vertex_request_params(backend, MAX_OUTPUT_TOKENS),
                               "extra_body": connections.vertex_extra_body(backend, selected.reasoning)})
    else:  # an OpenAI-compatible server (vLLM)
        template = ({"chat_template_kwargs": {"enable_thinking": bool(selected.reasoning)}}
                    if selected.thinking_supported else {})
        runtime.update(provider="openai", server_urls=[selected.endpoint], model_name=selected.model,
                       sampling_profile="auto",
                       remote={"request_params": {"max_tokens": MAX_OUTPUT_TOKENS}, "extra_body": template})
    config, _ = resolve_structured_config(runtime, cache_dir=None)
    if backend in CLAUDE_BACKENDS:
        config = replace(config, request_params={**config.request_params, "cache_control": {"type": "ephemeral"}})
    return config


class MeasuredClient(_MeasuredClient):
    """The SQL agent's measured in-memory client, also counting Claude's cache reads and writes."""

    def __init__(self, config: StructuredConfig):
        super().__init__(config)
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0

    @property
    def provider(self) -> str:
        return self.config.provider

    def _http(self, endpoint, body=None, **kwargs):
        result = super()._http(endpoint, body, **kwargs)
        if endpoint == "/chat/completions":
            usage = result.get("usage") or {}
            self.cache_read_tokens += int(usage.get("cache_read_input_tokens") or 0)
            self.cache_write_tokens += int(usage.get("cache_creation_input_tokens") or 0)
        return result

    def usage(self) -> dict:
        return {"requests": self.requests, "input_tokens": self.prompt_tokens, "output_tokens": self.completion_tokens,
                "cache_read_tokens": self.cache_read_tokens, "cache_write_tokens": self.cache_write_tokens}


def make_client(selected) -> MeasuredClient:
    return MeasuredClient(structured_config(selected))


def connection_label(selected) -> str:
    """The connection in a few words (as Data and models shows it)."""
    from ..llm import claude_name

    if selected.llm_backend in CLAUDE_BACKENDS:
        return f"{claude_name(selected.model)} (Databricks gateway)"
    if selected.llm_backend in VERTEX_BACKENDS:
        return f"{connections.VERTEX_LABELS[selected.llm_backend]} (Google Vertex)"
    if not selected.endpoint:
        return "No connection"
    return f"{selected.model} at {selected.endpoint}"
