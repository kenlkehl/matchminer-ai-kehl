"""The hosted OpenRouter connection (synthetic mode only): an OpenRouter key the app's owner supplies as a secret
(``MMAI_OPENROUTER_HOSTED_KEY``, e.g. a Hugging Face Space secret) serves the sessions that pasted no key of their own,
for one model only.

- **When:** the site config's ``dashboard.synthetic.hosted_openrouter`` block is present (``DashSettings``) *and* the
  secret is set, so one site config serves a Space with the secret and one without. Real mode refuses the block
  (``DashSettings``) and drops the secret.
- **The key:** read once at startup and removed from the environment (``llm.openrouter.take_hosted_key``), then kept
  only in the package's in-memory key store. It is never written, logged, put in a configuration, an error, an export
  or a response; pages say "this Space's key".
- **One model:** the key's references serve ``model`` only; a request for another is refused in
  ``llm.openrouter`` before anything is sent, and the key's guardrail on OpenRouter is the second lock.
- **Limits** (``llm.openrouter.HostedQuota``): ``requests_per_minute`` for the whole app (requests wait for a slot),
  ``session_daily_requests`` per browser session and UTC day, and a stop at ``daily_reserve`` of the account's
  free-model requests left today (OpenRouter's ``GET /key``). A 429 is not retried.
- **Who uses it:** every OpenRouter connection of a session without its own key (``GuidelineState.key_ref``); with
  OpenRouter the only connection (the lite Space), its model is every task's default. A session's own key serves
  instead once pasted, for any model.
"""
from __future__ import annotations

from matchminer_ai.llm import openrouter

DEFAULTS = {"model": openrouter.HOSTED_MODEL, "label": "Nemotron 3.5 Super", "requests_per_minute": 16,
            "session_daily_requests": 150, "daily_reserve": 20}
# OpenRouter's free-model limit is 20 requests a minute per account.
MAX_REQUESTS_PER_MINUTE = 20
REAL = "The hosted OpenRouter connection is for synthetic mode only (hosted_openrouter); real mode never offers it."
NOTICE = ("Requests go to OpenRouter (openrouter.ai) under this Space's account, with this Space's key, for {model} "
          "only. Free models' providers may log prompts and outputs. Synthetic data only: do not enter real patient "
          "information.")


def normalize(block) -> dict:
    """The site config's ``hosted_openrouter`` block (a mapping; ``{}`` or ``true``: the defaults) with its defaults,
    checked (``ValueError``)."""
    if block is True:
        block = {}
    if not isinstance(block, dict):
        raise ValueError("hosted_openrouter is a mapping (model, label, requests_per_minute, session_daily_requests, "
                         "daily_reserve), or {} for the defaults.")
    unknown = sorted(set(block) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown hosted_openrouter settings: {', '.join(map(str, unknown))}.")
    values = {**DEFAULTS, **block}
    values["model"] = openrouter.valid_model(str(values["model"]))
    values["label"] = str(values["label"]).strip() or DEFAULTS["label"]
    for name, low, high in (("requests_per_minute", 1, MAX_REQUESTS_PER_MINUTE),
                            ("session_daily_requests", 1, 100_000), ("daily_reserve", 0, 100_000)):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"hosted_openrouter.{name} is a whole number from {low} to {high}.")
    return values


def setup(settings, log=print) -> openrouter.HostedQuota | None:
    """The hosted connection's limits, when ``settings`` (``DashSettings``) has the block and the secret is set; else
    None (the secret, if any, is dropped)."""
    block = getattr(settings, "hosted_openrouter", None)
    if getattr(settings, "real", False) or not block:
        openrouter.discard_hosted()
        return None
    try:
        quota = openrouter.register_hosted(model=block["model"], requests_per_minute=block["requests_per_minute"],
                                           session_requests=block["session_daily_requests"],
                                           reserve=block["daily_reserve"])
    except ValueError:  # a fixed text; never the value
        log(f"Hosted OpenRouter connection: {openrouter.HOSTED_KEY_ENV} is not an OpenRouter key; none is offered.")
        return None
    if quota is None:
        log(f"Hosted OpenRouter connection: no {openrouter.HOSTED_KEY_ENV} secret; none is offered.")
    else:
        log(f"Hosted OpenRouter connection: {quota.model} with this app's key, at most "
            f"{quota.requests_per_minute} requests a minute and {quota.session_requests} per session a day, "
            f"stopping at {quota.reserve} of the account's daily free-model requests left.")
    return quota


def option_label(label: str, model: str) -> str:
    return f"Hosted: {label} ({'free, ' if model.endswith(':free') else ''}shared quota)"


def view(quota: openrouter.HostedQuota, ref: str, label: str, *, own_key: bool) -> dict:
    """``/api/models``' ``openrouter.hosted``: the option, its notice and what is left (``HostedQuota.status``) for
    the session whose reference to the key is ``ref``; ``own_key``: the session's own key serves instead."""
    return {"label": option_label(label, quota.model), "notice": NOTICE.format(model=quota.model),
            "own_key": own_key, **quota.status(ref)}
