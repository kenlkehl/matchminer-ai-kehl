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
