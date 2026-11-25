"""OpenRouter (https://openrouter.ai): many hosted models behind one OpenAI-compatible API, with a user's own key.

- **Destination:** always ``ENDPOINT``; a key goes nowhere else (``destination_error``).
- **Key:** held only in this process's memory. ``register_key`` keeps it and returns an opaque reference
  (``api_key_ref``) that configurations carry instead of the key; ``forget_key`` drops it. The key is never written
  to disk, logged, put in a configuration, an error or a response; ``key_hint`` gives its last four characters.
- **Requests** (``request_body``): JSON-schema response formats are kept, with ``provider.require_parameters`` so
  only providers that honour every parameter serve a request. Reasoning goes through OpenRouter's ``reasoning``
  object: Claude's effort as ``reasoning.effort`` and ``verbosity`` (Claude's output effort), other models' thinking
  switch as ``reasoning.enabled``. Claude gets no sampling parameters (the current models reject them); other models
  keep the package's model-profile sampling, minus neutral values. vLLM-only fields (``chat_template_kwargs``,
  ``structured_outputs`` and the like) are dropped.
- **Errors:** fixed texts by HTTP status (``status_message``); response bodies are never kept.
- **Models offered:** ``presets/openrouter.yaml`` (``presets``); any other model ID may be typed.
"""
from __future__ import annotations

import re
import secrets
import threading
from functools import lru_cache
from importlib import resources

ENDPOINT = "https://openrouter.ai/api/v1"
PROVIDER = "openrouter"
#: OpenRouter's app attribution headers: the app's public page and name, never a user or host of the site.
APP_URL = "https://huggingface.co/spaces/ksg-dfci/matchminer-ai-1226"
APP_TITLE = "MatchMiner-AI"
#: The prompt budget for a typed model the presets do not list.
DEFAULT_CONTEXT = 131072
#: Local chunk boundaries only (as for the other hosted models), unless a preset names the model's tokenizer.
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
#: The output cap of every OpenRouter request (OpenRouter reserves credit for the whole output budget).
MAX_OUTPUT_TOKENS = 65536
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:-]*")
_KEY = re.compile(r"[A-Za-z0-9_.-]{20,256}")

NO_KEY = ("No OpenRouter API key for this browser session: paste one on Data and models (it is kept only in the "
          "app's memory for this session).")
BAD_KEY = "That is not an OpenRouter API key (sk-or-...)."
STATUS = {
    400: "OpenRouter rejected the request (HTTP 400): the prompt may exceed the model's context, or the model "
         "does not accept a parameter.",
    401: "OpenRouter did not accept the API key (HTTP 401). Paste a valid key on Data and models.",
    402: "The OpenRouter account has too few credits for this request (HTTP 402). Add credits on openrouter.ai, or "
         "choose a free model.",
    403: "OpenRouter refused the request (HTTP 403): the key may not have access to this model, or the input was "
         "flagged by moderation.",
    404: "OpenRouter has no provider for this model that supports every parameter the task needs (JSON-schema "
         "output, reasoning), or the model ID is unknown (HTTP 404). Choose another model.",
    408: "OpenRouter timed out (HTTP 408). Try again.",
    429: "OpenRouter's rate limit for this model or account was reached (HTTP 429). Try again later, or choose "
         "another model.",
}
UNAVAILABLE = "OpenRouter or the model's provider is unavailable (HTTP {status}). Try again later."
UNREACHABLE = "Could not reach OpenRouter. Check this server's internet connection."
#: Statuses worth retrying: timeouts, rate limits and provider outages.
RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})


class OpenRouterError(RuntimeError):
    """An OpenRouter failure with a fixed message (no key, prompt or response text)."""

    def __init__(self, message: str, *, http_status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.http_status = http_status
        self.retryable = retryable


def status_message(status: int) -> str:
    return STATUS.get(status) or (UNAVAILABLE.format(status=status) if status >= 500 else
                                  f"OpenRouter refused the request (HTTP {status}).")


def error_for_status(status: int) -> OpenRouterError:
    return OpenRouterError(status_message(status), http_status=status, retryable=status in RETRYABLE)


# ---------------------------------------------------------------------------------------- the key, in memory only

_keys: dict[str, str] = {}
_lock = threading.Lock()


def register_key(key: str) -> str:
    """Keep ``key`` in memory; returns its reference (``ValueError`` with a fixed text when it is not a key)."""
    key = (key or "").strip()
    if not _KEY.fullmatch(key):
        raise ValueError(BAD_KEY)
    ref = "openrouter-" + secrets.token_urlsafe(18)
    with _lock:
        _keys[ref] = key
    return ref


def forget_key(ref: str | None) -> None:
    with _lock:
        _keys.pop(ref or "", None)


def has_key(ref: str | None) -> bool:
    with _lock:
        return bool(ref) and ref in _keys


def key_hint(ref: str | None) -> str | None:
    """The key's last four characters ("…abcd"), or None without a key."""
    with _lock:
        key = _keys.get(ref or "")
    return "…" + key[-4:] if key else None


def api_key(ref: str | None) -> str:
    """The key of ``ref`` (``OpenRouterError`` with ``NO_KEY`` when there is none)."""
    with _lock:
        key = _keys.get(ref or "")
    if not key:
        raise OpenRouterError(NO_KEY, http_status=401)
    return key


def destination_error(base_url: str) -> str | None:
    """Why ``base_url`` may not receive an OpenRouter key (None for ``ENDPOINT``)."""
    if (base_url or "").rstrip("/") != ENDPOINT:
        return f"An OpenRouter key is sent only to {ENDPOINT}."
    return None


def app_headers() -> dict[str, str]:
    return {"HTTP-Referer": APP_URL, "X-Title": APP_TITLE}


def headers(ref: str | None, base_url: str = ENDPOINT) -> dict[str, str]:
    """The request headers with the key of ``ref``, for ``ENDPOINT`` only."""
    error = destination_error(base_url)
    if error:
        raise ValueError(error)
    return {"Authorization": "Bearer " + api_key(ref), **app_headers()}


# ---------------------------------------------------------------------------------------- models

@lru_cache(maxsize=1)
def presets() -> dict:
    """``presets/openrouter.yaml``: ``{"default": model ID, "models": {ID: {label, context, tokenizer, reasoning}}}``."""
    import yaml

    text = resources.files("matchminer_ai.presets").joinpath("openrouter.yaml").read_text(encoding="utf-8")
    loaded: dict = yaml.safe_load(text)
    return loaded


def model_settings(model: str) -> dict:
    """``{"context", "tokenizer", "reasoning"}`` of a model: its preset's, else the defaults."""
    entry = presets()["models"].get(model) or {}
    return {"context": int(entry.get("context") or DEFAULT_CONTEXT),
            "tokenizer": str(entry.get("tokenizer") or DEFAULT_TOKENIZER),
            "reasoning": bool(entry.get("reasoning", False))}


def valid_model(model: str) -> str:
    model = (model or "").strip()
    if len(model) > 200 or not MODEL_ID.fullmatch(model):
        raise ValueError("Enter an OpenRouter model ID such as nvidia/nemotron-3.5-super-vl.")
    return model


def is_claude(model: str) -> bool:
    return str(model or "").startswith("anthropic/")


# ---------------------------------------------------------------------------------------- requests

#: Chat-completions fields OpenRouter receives; anything else (vLLM's extensions) is dropped.
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty")
_KEPT = frozenset({"model", "messages", "max_tokens", "max_completion_tokens", "stream", "stream_options", "stop",
                   "seed", "response_format", "tools", "tool_choice", "reasoning", "verbosity", "provider",
                   *SAMPLING})
# Values that change nothing (vLLM's way of saying "off"); a provider without the parameter can then still serve.
_NEUTRAL = {"top_k": (-1, 0), "min_p": (0,), "repetition_penalty": (1,), "presence_penalty": (0,),
            "frequency_penalty": (0,)}
#: OpenRouter's top-level request fields; the OpenAI SDK sends the rest of a body as its extra body.
SDK_PARAMS = frozenset({"max_tokens", "max_completion_tokens", "temperature", "top_p", "presence_penalty",
                        "frequency_penalty", "seed", "stop", "response_format", "tools", "tool_choice", "stream",
                        "stream_options"})
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def request_body(body: dict) -> dict:
    """``body`` (a chat-completions request, every field at the top level) as OpenRouter takes it.

    Idempotent: a ``reasoning`` object already decided is kept, except that an explicit thinking switch off
    (``chat_template_kwargs.enable_thinking`` false) turns reasoning off."""
    model = str(body.get("model") or "")
    template = body.get("chat_template_kwargs") or {}
    thinking = template.get("enable_thinking") if isinstance(template, dict) else None
    effort = body.get("reasoning_effort") or (template.get("reasoning_effort") if isinstance(template, dict) else None)
    out = {key: value for key, value in body.items() if key in _KEPT}
    for name, neutral in _NEUTRAL.items():
        if name in out and out[name] in neutral:
            out.pop(name)
    for name in ("max_tokens", "max_completion_tokens"):
        if isinstance(out.get(name), int):
            out[name] = min(out[name], MAX_OUTPUT_TOKENS)
    reasoning = dict(out.get("reasoning") or {})
    if is_claude(model):
        for name in SAMPLING:
            out.pop(name, None)
        effort = reasoning.get("effort") or out.get("verbosity") or effort or "medium"
        effort = "low" if effort == "minimal" else effort if effort in CLAUDE_EFFORTS else "high"
        # OpenRouter maps both to Claude's output effort; the newer models read verbosity.
        out["reasoning"] = {"effort": effort}
        out["verbosity"] = effort
    else:
        out.pop("verbosity", None)
        if thinking is False:
            reasoning = {"enabled": False}
        elif thinking is True and not reasoning:
            reasoning = {"enabled": True}
        if reasoning:
            out["reasoning"] = reasoning
    out["provider"] = {**(out.get("provider") or {}), "require_parameters": True}
    return out


def split_request(params: dict, extra: dict, model: str) -> tuple[dict, dict]:
    """``request_body`` over a request's parameters and extra body for ``model``, split back into the two (the
    OpenAI SDK's arguments and its extra body). The model and messages are not part of either."""
    body = request_body({**params, **extra, "model": model})
    body.pop("model", None)
    body.pop("messages", None)
    return ({k: v for k, v in body.items() if k in SDK_PARAMS},
            {k: v for k, v in body.items() if k not in SDK_PARAMS})


class ChatClient:
    """An ``AsyncOpenAI`` client for ``ENDPOINT`` whose failures carry fixed texts (``OpenRouterError``)."""

    def __init__(self, ref: str | None, *, timeout: float, max_connections: int = 16):
        import httpx
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=ENDPOINT, api_key=api_key(ref), timeout=timeout, max_retries=0, default_headers=app_headers(),
            http_client=httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(
                max_connections=max(1, int(max_connections)), max_keepalive_connections=0)),
        )
        self.chat = self
        self.completions = self

    async def create(self, **request):
        import openai

        for name in ("max_tokens", "max_completion_tokens"):  # a prompt's own budget may be larger
            if isinstance(request.get(name), int):
                request[name] = min(request[name], MAX_OUTPUT_TOKENS)
        try:
            return await self._client.chat.completions.create(**request)
        except openai.APIStatusError as exc:
            raise error_for_status(exc.status_code) from None
        except openai.APIConnectionError:
            raise OpenRouterError(UNREACHABLE, retryable=True) from None

    async def close(self):
        await self._client.close()


# ---------------------------------------------------------------------------------------- checks

def check(ref: str | None, model: str, *, timeout: float = 20.0) -> dict:
    """Whether the key works and OpenRouter lists ``model`` with JSON-schema output: ``{"available", "error"}``
    (fixed texts). Reads only the key's own record and the public model list; sends no prompt."""
    import httpx

    try:
        key_headers = headers(ref)
    except OpenRouterError as exc:
        return {"available": False, "error": str(exc)}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(ENDPOINT + "/key", headers=key_headers)
            if response.status_code != 200:
                return {"available": False, "error": status_message(response.status_code)}
            listed = client.get(ENDPOINT + "/models", headers=app_headers())
            if listed.status_code != 200:
                return {"available": False, "error": status_message(listed.status_code)}
            models = {m.get("id"): m for m in listed.json().get("data", []) if isinstance(m, dict)}
    except (httpx.HTTPError, ValueError):
        return {"available": False, "error": UNREACHABLE}
    found = models.get(model)
    if found is None:
        return {"available": False, "error": f"OpenRouter lists no model {model!r}."}
    supported = set(found.get("supported_parameters") or ())
    if "response_format" not in supported:
        return {"available": False,
                "error": f"OpenRouter lists no JSON-schema output (response_format) for {model!r}; the app's tasks "
                         "need it."}
    return {"available": True, "error": None}


__all__ = [
    "APP_TITLE", "APP_URL", "ChatClient", "DEFAULT_CONTEXT", "ENDPOINT", "MAX_OUTPUT_TOKENS", "NO_KEY",
    "OpenRouterError", "PROVIDER", "api_key", "app_headers", "check", "destination_error", "error_for_status",
    "forget_key", "has_key", "headers", "is_claude", "key_hint", "model_settings", "presets", "register_key",
    "request_body", "split_request", "status_message", "valid_model",
]
