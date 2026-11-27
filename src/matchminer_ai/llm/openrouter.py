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
- **The hosted key** (``register_hosted``): a key the app's owner supplies in ``HOSTED_KEY_ENV`` (a Space secret)
  for users without their own. It is read once and removed from the environment (``take_hosted_key``), so no child
  process inherits it, and kept like a user's key, but ``key_hint`` never shows it. Its references serve one model
  only: a request through one for another model is refused before anything is sent (``admit``, ``model_error``).
  Every request through them passes ``HostedQuota``'s limits (per minute, per session and UTC day, and the
  account's free-model requests left today), and a 429 is not retried (``error_for_status``).
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
import os
import re
import secrets
import threading
import time
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
#: Statuses worth retrying: timeouts, rate limits and provider outages (a hosted key's 429 excepted).
RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})

#: The environment variable of the hosted key (``register_hosted``): read once, then removed from the environment.
HOSTED_KEY_ENV = "MMAI_OPENROUTER_HOSTED_KEY"
#: The hosted key's default model: free on OpenRouter, within the account's free-model limits.
HOSTED_MODEL = "nvidia/nemotron-3.5-super-vl:free"
#: How long a reading of the account's free-model counter (``GET /key``) is used before it is read again.
QUOTA_CACHE_SECONDS = 60.0
#: How far OpenRouter's counter may lag: requests sent this long before a reading count against it again, so the
#: estimate of what is left errs low.
COUNTER_LAG_SECONDS = 90.0
WINDOW_SECONDS = 60.0
#: After a 429 through the hosted key, how long every request through it is refused here, before anything is sent,
#: so no retry above this module (a summary's quality retries, a structured request's attempts) spends the account's
#: daily requests on it.
PAUSE_SECONDS = 60.0
MODEL_LOCKED = "This Space's key serves only {model}; paste your own key for other models."
HOSTED_DAILY_STOP = ("This Space's shared OpenRouter quota for today is used up (it resets at 00:00 UTC). Paste your "
                     "own OpenRouter key on Data and models to continue.")
HOSTED_SESSION_STOP = ("This browser session has used its share of this Space's OpenRouter quota for today ({share} "
                       "requests; it resets at 00:00 UTC). Paste your own OpenRouter key on Data and models to "
                       "continue.")
HOSTED_PAUSED = ("OpenRouter's free-model limit for this Space's key was reached (HTTP 429); requests through it pause "
                 "for a minute. Try again later, or paste your own OpenRouter key on Data and models.")
HOSTED_STATUS = {
    401: "OpenRouter did not accept this Space's key (HTTP 401). Paste your own OpenRouter key on Data and models.",
    402: "This Space's OpenRouter account cannot pay for this request (HTTP 402). Paste your own OpenRouter key on "
         "Data and models.",
    429: "OpenRouter's free-model limit for this Space's key was reached (HTTP 429). The request is not retried: try "
         "again later, or paste your own OpenRouter key on Data and models.",
}


class OpenRouterError(RuntimeError):
    """An OpenRouter failure with a fixed message (no key, prompt or response text)."""

    def __init__(self, message: str, *, http_status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.http_status = http_status
        self.retryable = retryable


def status_message(status: int, *, hosted: bool = False) -> str:
    """The fixed text of an HTTP failure (``hosted``: through the hosted key)."""
    if hosted and status in HOSTED_STATUS:
        return HOSTED_STATUS[status]
    return STATUS.get(status) or (UNAVAILABLE.format(status=status) if status >= 500 else
                                  f"OpenRouter refused the request (HTTP {status}).")


def error_for_status(status: int, *, hosted: bool = False) -> OpenRouterError:
    """The failure of an HTTP status. Through the hosted key a 429 is final: a retry would spend the account's daily
    free-model requests (OpenRouter counts failed ones too)."""
    return OpenRouterError(status_message(status, hosted=hosted), http_status=status,
                           retryable=status in RETRYABLE and not (hosted and status == 429))


# ---------------------------------------------------------------------------------------- the key, in memory only

_keys: dict[str, str] = {}
# A reference's only model (the hosted key's), and the hosted references' limits.
_models: dict[str, str] = {}
_hosted: dict[str, "HostedQuota"] = {}
_lock = threading.Lock()


def _new_ref() -> str:
    return "openrouter-" + secrets.token_urlsafe(18)


def register_key(key: str, *, model: str | None = None) -> str:
    """Keep ``key`` in memory; returns its reference (``ValueError`` with a fixed text when it is not a key). With
    ``model``, the reference serves that model only (``model_error``)."""
    key = (key or "").strip()
    if not _KEY.fullmatch(key):
        raise ValueError(BAD_KEY)
    ref = _new_ref()
    with _lock:
        _keys[ref] = key
        if model:
            _models[ref] = model
    return ref


def forget_key(ref: str | None) -> None:
    with _lock:
        _keys.pop(ref or "", None)
        _models.pop(ref or "", None)
        _hosted.pop(ref or "", None)


def has_key(ref: str | None) -> bool:
    with _lock:
        return bool(ref) and ref in _keys


def is_hosted(ref: str | None) -> bool:
    """Whether ``ref`` names the hosted key (a session's reference to it)."""
    with _lock:
        return bool(ref) and ref in _hosted


def key_hint(ref: str | None) -> str | None:
    """The key's last four characters ("…abcd"), or None without a key; never any part of the hosted key."""
    with _lock:
        key = None if (ref or "") in _hosted else _keys.get(ref or "")
    return "…" + key[-4:] if key else None


def model_error(ref: str | None, model: str | None) -> str | None:
    """Why ``ref`` may not serve ``model`` (a reference restricted to another model), else None."""
    with _lock:
        allowed = _models.get(ref or "")
    return MODEL_LOCKED.format(model=allowed) if allowed and model != allowed else None


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


# ---------------------------------------------------------------------------------------- the hosted key

_hosted_secret: str | None = None


def take_hosted_key() -> None:
    """Move the hosted key from the environment (``HOSTED_KEY_ENV``) into this process's memory, so no child process
    inherits it. Call it at startup, before any child process starts; ``register_hosted`` calls it too."""
    global _hosted_secret
    key = os.environ.pop(HOSTED_KEY_ENV, None)
    if key is not None:
        with _lock:
            _hosted_secret = key.strip() or None


def register_hosted(*, model: str = HOSTED_MODEL, requests_per_minute: int = 16, session_requests: int = 150,
                    reserve: int = 20, fetch=None, clock=time.monotonic, wall=time.time) -> "HostedQuota | None":
    """The hosted key's limits (``HostedQuota``), replacing any earlier ones; None without a hosted key
    (``ValueError`` with a fixed text when it is not a key). The key then lives only in the key store."""
    global _hosted_secret
    take_hosted_key()
    with _lock:
        key, _hosted_secret = _hosted_secret, None
    discard_hosted()
    if key is None:
        return None
    ref = register_key(key, model=model)  # ValueError(BAD_KEY): never the key itself
    quota = HostedQuota(ref, model, requests_per_minute=requests_per_minute, session_requests=session_requests,
                        reserve=reserve, fetch=fetch, clock=clock, wall=wall)
    with _lock:
        _hosted[ref] = quota
    return quota


def discard_hosted() -> None:
    """Forget the hosted key and every reference to it (and a hosted key still in the environment)."""
    global _hosted_secret
    take_hosted_key()
    with _lock:
        _hosted_secret = None
        refs = list(_hosted)
    for ref in refs:
        forget_key(ref)


def hosted_quota(ref: str | None) -> "HostedQuota | None":
    with _lock:
        return _hosted.get(ref or "")


def hosted_stop(ref: str | None) -> str | None:
    """Why requests through a hosted reference are refused now (a fixed text), else None (also for any other key)."""
    quota = hosted_quota(ref)
    return quota.stopped(ref or "") if quota is not None else None


def rate_limited(ref: str | None) -> None:
    """OpenRouter answered a request through ``ref`` with 429: a hosted reference's key pauses (``PAUSE_SECONDS``)."""
    quota = hosted_quota(ref)
    if quota is not None:
        quota.rate_limited()


def admit(ref: str | None, model: str | None, *, sleep=time.sleep) -> None:
    """Before a chat request through ``ref`` (nothing is sent but, for the hosted key, OpenRouter's ``GET /key``):
    refuse a model the reference does not serve, and hold a hosted reference's request to its limits, waiting for a
    slot of the per-minute limit with ``sleep``. Raises ``OpenRouterError`` (not retryable) with a fixed text."""
    error = model_error(ref, model)
    if error:
        raise OpenRouterError(error, retryable=False)
    quota = hosted_quota(ref)
    if quota is not None:
        quota.admit(ref or "", sleep=sleep)


async def admit_async(ref: str | None, model: str | None) -> None:
    """``admit`` for an event loop: the wait is ``asyncio.sleep`` and the counter's reading runs in a thread."""
    error = model_error(ref, model)
    if error:
        raise OpenRouterError(error, retryable=False)
    quota = hosted_quota(ref)
    if quota is not None:
        await quota.admit_async(ref or "")


def daily_requests(record) -> dict | None:
    """``{"used", "limit", "remaining"}`` of the account's free-model requests today, from ``GET /key``'s
    ``free_model_daily_requests`` (in its ``data`` object, or at the top level), else None."""
    if not isinstance(record, dict):
        return None
    data = record["data"] if isinstance(record.get("data"), dict) else record
    daily = data.get("free_model_daily_requests")
    if not isinstance(daily, dict):
        return None

    def count(name):
        value = daily.get(name)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None

    used, limit, remaining = count("used"), count("limit"), count("remaining")
    if remaining is None and limit is not None and used is not None:
        remaining = max(0, limit - used)
    return None if remaining is None else {"used": used, "limit": limit, "remaining": remaining}


def _fetch_key_record(ref: str, timeout: float = 10.0) -> tuple[int | None, dict | None]:
    """``GET /key`` with ``ref``'s key: ``(HTTP status, JSON)``; ``(None, None)`` when OpenRouter is unreachable. It
    reads the key's own record and uses no model request."""
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(ENDPOINT + "/key", headers=headers(ref))
        if response.status_code != 200:
            return response.status_code, None
        return 200, response.json()
    except (httpx.HTTPError, ValueError, OpenRouterError):
        return None, None


class HostedQuota:
    """The hosted key's limits, for every session's reference to it (one per process; ``register_hosted``).

    - **Per minute:** at most ``requests_per_minute`` requests start in any minute; a request waits for a slot.
    - **Per day:** the account's free-model requests left today, from OpenRouter's ``GET /key``
      (``free_model_daily_requests``), read at most once a minute and lowered by the requests sent since (and in the
      counter's lag before). At ``reserve`` left, requests are refused (``HOSTED_DAILY_STOP``). When OpenRouter
      reports no counter, the per-minute and per-session limits still hold.
    - **Per session:** each session's reference (``session_ref``) may send ``session_requests`` requests a UTC day.

    - **After a 429:** every request pauses for ``PAUSE_SECONDS`` (refused here, nothing sent), and the counter is read
      again before the next one.

    Every attempt counts (OpenRouter counts failed requests too); nothing here retries."""

    def __init__(self, ref: str, model: str, *, requests_per_minute: int, session_requests: int, reserve: int,
                 fetch=None, clock=time.monotonic, wall=time.time):
        self.ref, self.model = ref, model
        self.requests_per_minute = max(1, int(requests_per_minute))
        self.session_requests = max(1, int(session_requests))
        self.reserve = max(0, int(reserve))
        self._fetch, self._clock, self._wall = fetch, clock, wall
        self._lock = threading.Lock()
        self._reading_lock = threading.Lock()
        self._booked: deque[float] = deque()  # the per-minute limit's slots (clock times)
        self._sent: deque[float] = deque()  # requests let through (clock times), for the daily estimate
        self._used: dict[str, int] = {}  # requests per session reference today
        self._day = self._today()
        self._reading: dict | None = None  # {"at", "day", "status", "daily"}
        self._sessions: dict[str, str] = {}  # a session (the app's owner ID) -> its reference
        self._seen: dict[str, float] = {}  # reference -> last use (wall clock)
        self._paused_until = float("-inf")  # after a 429 (clock time)

    def _today(self) -> str:
        return datetime.fromtimestamp(self._wall(), timezone.utc).date().isoformat()

    # ------------------------------------------------------------------ sessions

    def session_ref(self, session: str) -> str:
        """``session``'s own reference to the hosted key (the same one all day), for its daily share."""
        now = self._wall()
        with _lock:
            if self.ref not in _keys:  # discarded (register_hosted replaced it): no key
                return ""
            ref = self._sessions.get(session)
            if ref is None or ref not in _keys:
                ref = _new_ref()
                _keys[ref], _models[ref], _hosted[ref] = _keys[self.ref], self.model, self
                self._sessions[session] = ref
            self._seen[ref] = now
        return ref

    def _roll(self) -> None:
        """A new UTC day: the sessions' counts start again, and references unused for a day are forgotten."""
        today = self._today()
        if today == self._day:
            return
        self._day, self._used = today, {}
        cutoff = self._wall() - 86400
        with _lock:
            stale = {session: ref for session, ref in self._sessions.items() if self._seen.get(ref, 0) < cutoff}
            for session, ref in stale.items():
                del self._sessions[session]
                self._seen.pop(ref, None)
                for store in (_keys, _models, _hosted):
                    store.pop(ref, None)

    # ------------------------------------------------------------------ the account's counter

    def _refresh(self) -> None:
        """Read ``GET /key`` when the last reading is a minute old or from another UTC day (one reader at a time)."""
        with self._reading_lock:
            with self._lock:
                reading = self._reading
                if (reading is not None and reading["day"] == self._today()
                        and self._clock() - reading["at"] < QUOTA_CACHE_SECONDS):
                    return
            at, day = self._clock(), self._today()
            status, record = (self._fetch or _fetch_key_record)(self.ref)
            with self._lock:
                self._reading = {"at": at, "day": day, "status": status,
                                 "daily": daily_requests(record) if status == 200 else None}

    def _remaining(self) -> int | None:
        """The account's free-model requests left today (an estimate that errs low), or None when unknown."""
        reading = self._reading
        if reading is None or reading["daily"] is None or reading["day"] != self._today():
            return None
        since = reading["at"] - COUNTER_LAG_SECONDS
        return max(0, int(reading["daily"]["remaining"]) - sum(1 for t in self._sent if t >= since))

    # ------------------------------------------------------------------ admission

    def rate_limited(self) -> None:
        with self._lock:
            self._paused_until = self._clock() + PAUSE_SECONDS
            self._reading = None  # the counter is read again before the next request

    def _check(self, ref: str, *, count: bool = False) -> None:
        with self._lock:
            self._roll()
            if self._clock() < self._paused_until:
                raise OpenRouterError(HOSTED_PAUSED, retryable=False)
            used = self._used.get(ref, 0)
            if used >= self.session_requests:
                raise OpenRouterError(HOSTED_SESSION_STOP.format(share=self.session_requests), retryable=False)
            remaining = self._remaining()
            if remaining is not None and remaining <= self.reserve:
                raise OpenRouterError(HOSTED_DAILY_STOP, retryable=False)
            if count:
                now = self._clock()
                self._used[ref] = used + 1
                self._sent.append(now)
                while self._sent and self._sent[0] < now - 600:
                    self._sent.popleft()

    def _book(self) -> float:
        """Book the next slot of the per-minute limit; returns the seconds to wait for it."""
        with self._lock:
            now = self._clock()
            while self._booked and self._booked[0] <= now - WINDOW_SECONDS:
                self._booked.popleft()
            at = now
            if len(self._booked) >= self.requests_per_minute:
                at = max(at, self._booked[-self.requests_per_minute] + WINDOW_SECONDS)
            if self._booked:
                at = max(at, self._booked[-1])
            self._booked.append(at)
            return float(at - now)

    def admit(self, ref: str, *, sleep=time.sleep) -> None:
        """Let one request through ``ref`` go, once it is within every limit (``OpenRouterError`` otherwise)."""
        self._refresh()
        self._check(ref)
        delay = self._book()
        if delay > 0:
            sleep(delay)
        self._refresh()
        self._check(ref, count=True)

    async def admit_async(self, ref: str) -> None:
        await asyncio.to_thread(self._refresh)
        self._check(ref)
        delay = self._book()
        if delay > 0:
            await asyncio.sleep(delay)
        await asyncio.to_thread(self._refresh)
        self._check(ref, count=True)

    # ------------------------------------------------------------------ what a page shows

    def _stopped(self, ref: str) -> str | None:
        remaining = self._remaining()
        if remaining is not None and remaining <= self.reserve:
            return HOSTED_DAILY_STOP
        if self._used.get(ref, 0) >= self.session_requests:
            return HOSTED_SESSION_STOP.format(share=self.session_requests)
        return HOSTED_PAUSED if self._clock() < self._paused_until else None

    def stopped(self, ref: str) -> str | None:
        """Why requests through ``ref`` are refused now, from what is known (no reading of the counter)."""
        with self._lock:
            self._roll()
            return self._stopped(ref)

    def status(self, ref: str | None = None) -> dict:
        """The limits and what is left: today's account counter (``known`` False when OpenRouter reported none),
        ``ref``'s session share, and why requests stop now (``stopped``) or the counter's reading failed (``error``);
        fixed texts only."""
        self._refresh()
        with self._lock:
            self._roll()
            remaining = self._remaining()
            reading = self._reading or {}
            daily = reading.get("daily") or {}
            used = self._used.get(ref or "", 0)
            stopped = self._stopped(ref or "")
            status = reading.get("status")
            error = (None if status == 200 else UNREACHABLE if status is None else status_message(status, hosted=True))
            return {"model": self.model, "requests_per_minute": self.requests_per_minute, "reserve": self.reserve,
                    "today": {"known": remaining is not None, "remaining": remaining,
                              "available": None if remaining is None else max(0, remaining - self.reserve),
                              "limit": daily.get("limit") if remaining is not None else None},
                    "session": {"used": used, "share": self.session_requests,
                                "remaining": max(0, self.session_requests - used)},
                    "stopped": stopped, "error": error}


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

        self._ref = ref
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
        # The reference's model and, for the hosted key, its limits: before anything is sent.
        await admit_async(self._ref, request.get("model"))
        try:
            return await self._client.chat.completions.create(**request)
        except openai.APIStatusError as exc:
            if exc.status_code == 429:
                rate_limited(self._ref)
            raise error_for_status(exc.status_code, hosted=is_hosted(self._ref)) from None
        except openai.APIConnectionError:
            raise OpenRouterError(UNREACHABLE, retryable=True) from None

    async def close(self):
        await self._client.close()


# ---------------------------------------------------------------------------------------- checks

def check(ref: str | None, model: str, *, timeout: float = 20.0) -> dict:
    """Whether the key works and OpenRouter lists ``model`` with JSON-schema output: ``{"available", "error"}``
    (fixed texts). Reads only the key's own record and the public model list; sends no prompt. The hosted key: its
    model only, and its quota's reading of the key's record (at most one a minute)."""
    import httpx

    try:
        key_headers = headers(ref)
    except OpenRouterError as exc:
        return {"available": False, "error": str(exc)}
    error = model_error(ref, model)
    if error:
        return {"available": False, "error": error}
    quota = hosted_quota(ref)
    if quota is not None:
        state = quota.status(ref)
        if state["error"] or state["stopped"]:
            return {"available": False, "error": state["error"] or state["stopped"]}
    try:
        with httpx.Client(timeout=timeout) as client:
            if quota is None:
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
    "APP_TITLE", "APP_URL", "ChatClient", "DEFAULT_CONTEXT", "ENDPOINT", "HOSTED_KEY_ENV", "HOSTED_MODEL",
    "HostedQuota", "MAX_OUTPUT_TOKENS", "NO_KEY", "OpenRouterError", "PROVIDER", "admit", "admit_async", "api_key",
    "app_headers", "check", "daily_requests", "destination_error", "discard_hosted", "error_for_status", "forget_key",
    "has_key", "headers", "hosted_quota", "hosted_stop", "is_claude", "is_hosted", "key_hint", "model_error",
    "model_settings", "presets", "rate_limited", "register_hosted", "register_key", "request_body", "split_request",
    "status_message", "take_hosted_key", "valid_model",
]
