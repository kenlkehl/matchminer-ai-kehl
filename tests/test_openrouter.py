"""OpenRouter requests (``matchminer_ai.llm.openrouter``): the key goes only to OpenRouter, in a header; requests
carry JSON-schema output with ``provider.require_parameters``; Claude gets its effort and no sampling settings;
failures are fixed texts. Mocked HTTP only: nothing reaches OpenRouter."""

import io
import json
from dataclasses import replace
from urllib.error import HTTPError

import pytest

from matchminer_ai.llm import openrouter
from matchminer_ai.llm.remote_inference import build_remote_request_config
from matchminer_ai.llm.structured import EndpointError, StructuredClient, resolve_structured_config

KEY = "sk-or-v1-" + "0123456789abcdef" * 4  # fabricated
SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"],
          "additionalProperties": False}


def runtime(ref, model, thinking=True):
    return {"provider": "openrouter", "server_urls": [openrouter.ENDPOINT], "api_key_ref": ref, "model_name": model,
            "max_retries": 1, "context_window": 131072, "tokenizer_mode": "bytes", "response_format": "json_schema",
            "stream": False, "sampling_profile": "none" if openrouter.is_claude(model) else "auto",
            "remote": {"request_params": {"max_tokens": 1000, "reasoning_effort": "medium"},
                       "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}}}


def test_requests_carry_the_schema_and_the_key_only_in_a_header(monkeypatch):
    ref = openrouter.register_key(KEY)
    sent, status = [], {"code": 200}

    def respond(request, timeout):
        sent.append(request)
        if status["code"] != 200:  # a body that echoes the key, which must not surface
            raise HTTPError(request.full_url, status["code"], "refused", {}, io.BytesIO(KEY.encode()))
        return io.BytesIO(b'{"choices":[{"message":{"content":"{\\"a\\": \\"x\\"}"},"finish_reason":"stop"}]}')

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    monkeypatch.setenv("OPENAI_API_KEY", "an-unrelated-key")
    messages = [{"role": "user", "content": "Fabricated synthetic note."}]
    for model in ("nvidia/nemotron-3.5-super-vl", "anthropic/claude-haiku-5.5"):
        config, _ = resolve_structured_config(runtime(ref, model), cache_dir=None)
        assert ref not in repr(config) and "api_key_ref" not in config.public_dict()
        assert StructuredClient(config, None).complete("job", messages, SCHEMA, lambda value: value) == {"a": "x"}
        request = sent[-1]
        body = json.loads(request.data)
        assert request.full_url == openrouter.ENDPOINT + "/chat/completions"
        assert request.get_header("Authorization") == "Bearer " + KEY and KEY not in request.data.decode()
        assert body["response_format"]["json_schema"]["schema"] == SCHEMA
        assert body["provider"] == {"require_parameters": True}
        assert not {"chat_template_kwargs", "reasoning_effort", "top_k", "min_p"} & body.keys()
        if openrouter.is_claude(model):  # Claude: its effort, no sampling settings
            assert not {"temperature", "top_p"} & body.keys()
            assert body["reasoning"] == {"effort": "medium"} and body["verbosity"] == "medium"
        else:  # the model profile's sampling and the reasoning switch
            assert body["temperature"] == 1.0 and body["top_p"] == 0.95 and body["reasoning"] == {"enabled": True}
    for code, words in ((401, "API key"), (402, "credits"), (429, "rate limit")):
        status["code"] = code
        with pytest.raises(EndpointError) as failed:
            StructuredClient(config, None).complete("job", messages, SCHEMA, lambda value: value)
        assert words in str(failed.value) and KEY not in str(failed.value)
    with pytest.raises(ValueError, match="only to"):  # the key goes nowhere else
        StructuredClient(replace(config, base_url="https://example.invalid/v1"), None)
    openrouter.forget_key(ref)
    with pytest.raises(openrouter.OpenRouterError, match="No OpenRouter API key"):
        StructuredClient(config, None).complete("job", messages, SCHEMA, lambda value: value)


def test_the_remote_path_maps_the_same_parameters_and_switches_reasoning_off():
    params, extra = build_remote_request_config(runtime("ref", "nvidia/nemotron-3.5-super-vl", thinking=False))
    assert params == {"max_tokens": 1000, "temperature": 1.0, "top_p": 0.95}
    assert extra == {"reasoning": {"enabled": False}, "provider": {"require_parameters": True}}
    params, extra = build_remote_request_config(runtime("ref", "anthropic/claude-opus-5.5"))
    assert params == {"max_tokens": 1000}
    assert extra == {"reasoning": {"effort": "medium"}, "verbosity": "medium", "provider": {"require_parameters": True}}


HOSTED_KEY = "sk-or-v1-" + "0" * 60 + "Zq9X"  # fabricated
WALL = 1_800_000_000.0  # a fixed UTC day


def daily(remaining):
    return lambda ref: (200, {"data": {"free_model_daily_requests": {"used": 1000 - remaining, "limit": 1000,
                                                                     "remaining": remaining}}})


def test_the_hosted_key_serves_one_model_within_its_limits(monkeypatch):
    """The hosted key leaves the environment and is never hinted; its references refuse another model before anything
    is sent; requests wait for a per-minute slot; the session share and the daily reserve stop requests with fixed
    texts; a 429 is not retried, on either request path, and pauses the key so no retry above reaches OpenRouter."""
    import asyncio
    import os

    import httpx
    import openai

    from matchminer_ai.llm.remote_inference import single_inference_request
    from matchminer_ai.llm.structured import CredentialsError

    clock, waits = {"now": 1000.0}, []

    def sleep(seconds):
        waits.append(seconds)
        clock["now"] += seconds

    def register(**limits):
        monkeypatch.setenv(openrouter.HOSTED_KEY_ENV, HOSTED_KEY)
        quota = openrouter.register_hosted(clock=lambda: clock["now"], wall=lambda: WALL, **limits)
        assert openrouter.HOSTED_KEY_ENV not in os.environ  # no child process inherits it
        return quota

    quota = register(requests_per_minute=2, fetch=lambda ref: (200, {"data": {}}))  # no counter reported
    ref = quota.session_ref("session-a")
    assert quota.session_ref("session-a") == ref and openrouter.key_hint(ref) is None
    assert openrouter.api_key(ref) == HOSTED_KEY and openrouter.is_hosted(ref)

    sent = []

    def respond(request, timeout):
        sent.append(request)
        raise HTTPError(request.full_url, 429, "limited", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    messages = [{"role": "user", "content": "Fabricated synthetic note."}]
    other, _ = resolve_structured_config(runtime(ref, "anthropic/claude-haiku-5.5"), cache_dir=None)
    with pytest.raises(CredentialsError, match="serves only nvidia/nemotron-3.5-super-vl:free"):
        StructuredClient(other, None).complete("job", messages, SCHEMA, lambda value: value)
    assert sent == []  # refused before anything was sent
    config, _ = resolve_structured_config({**runtime(ref, quota.model), "max_retries": 3}, cache_dir=None)
    with pytest.raises(CredentialsError, match="not retried") as failed:
        StructuredClient(config, None).complete("job", messages, SCHEMA, lambda value: value)
    assert len(sent) == 1 and HOSTED_KEY not in str(failed.value)
    with pytest.raises(CredentialsError, match="pause for a minute"):  # a retry from above: refused, nothing sent
        StructuredClient(config, None).complete("job", messages, SCHEMA, lambda value: value)
    assert len(sent) == 1

    clock["now"] += 60
    calls = []

    async def limited(**request):
        calls.append(request)
        response = httpx.Response(429, request=httpx.Request("POST", openrouter.ENDPOINT + "/chat/completions"))
        raise openai.RateLimitError("limited", response=response, body=None)

    client = openrouter.ChatClient(ref, timeout=5)
    client._client.chat.completions.create = limited
    result = asyncio.run(single_inference_request(client, 0, messages, quota.model, {}, {}, max_retries=6))
    assert len(calls) == 1 and result.finish_reason == "error" and "not retried" in result.summary
    assert openrouter.error_for_status(429).retryable and not openrouter.error_for_status(429, hosted=True).retryable

    clock["now"] += 120  # the two requests above have left the per-minute window
    for _ in range(3):  # the third waits for the next minute's slot
        openrouter.admit(ref, quota.model, sleep=sleep)
    assert waits == [60.0]

    readings = []

    def counter(ref):
        readings.append(ref)
        return daily(28)(ref)

    quota = register(session_requests=2, reserve=25, fetch=counter)
    assert not openrouter.has_key(ref)  # the earlier registration's references are gone
    first, second = quota.session_ref("session-a"), quota.session_ref("session-b")
    for _ in range(2):  # 28 left today, then 26
        openrouter.admit(first, quota.model, sleep=sleep)
    with pytest.raises(openrouter.OpenRouterError, match="This browser session has used its share") as stop:
        openrouter.admit(first, quota.model, sleep=sleep)
    assert stop.value.retryable is False
    openrouter.admit(second, quota.model, sleep=sleep)  # 26 left: one more, then the reserve
    with pytest.raises(openrouter.OpenRouterError, match="quota for today is used up"):
        openrouter.admit(second, quota.model, sleep=sleep)
    status = quota.status(second)
    assert status["today"] == {"known": True, "remaining": 25, "available": 0, "limit": 1000}
    assert status["session"] == {"used": 1, "share": 2, "remaining": 1} and "used up" in status["stopped"]
    assert len(readings) == 1  # the account's counter is read at most once a minute
    openrouter.discard_hosted()
    assert not openrouter.has_key(first)
