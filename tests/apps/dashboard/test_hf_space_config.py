"""The Hugging Face Space's site config (deploy/hf_space/space/site.yaml) loads: its Hub references parse, and its
local model server spec is a valid vLLM spec that serves Nemotron 3.5 Super with the package's Super profile."""
from __future__ import annotations

from pathlib import Path

import yaml

SITE = Path(__file__).resolve().parents[3] / "deploy" / "hf_space" / "space" / "site.yaml"


def test_the_space_site_config_and_its_vllm_server(monkeypatch):
    from matchminer_ai.llm import local_server as ls
    from matchminer_ai.site import expand_env, is_hub_ref, parse_hub_ref

    monkeypatch.delenv("MMAI_DATASET_REVISION", raising=False)
    values = expand_env(yaml.safe_load(SITE.read_text()))
    references = []

    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif is_hub_ref(value):
            references.append(parse_hub_ref(value))

    walk(values)
    assert references and {r.revision for r in references} == {"main"}  # MMAI_DATASET_REVISION unset: main
    spec = ls.ServerSpec.from_dict(values["dashboard"]["synthetic"]["local_server"]["autostart"])
    argv, _ = ls.build_command(spec, 8000, engine_command=["vllm", "serve"], parent_visible="0")
    joined = " ".join(argv)
    for flag in ("--reasoning-parser nemotron_v3", "--tool-call-parser qwen3_coder", "--language-model-only",
                 "--trust-remote-code", "--enable-prefix-caching", '"method":"mtp"'):
        assert flag in joined
    assert spec.engine == "vllm" and spec.gpu_memory_utilization <= 0.85 and spec.max_model_len >= 200_000


def test_the_spaces_tier_by_gpu_memory():
    """tier.py: Super with vLLM from 90 GB (site.yaml's autostart), Bolt with llama.cpp from 40 GB, else no local
    server (a T4, or no GPU: OpenRouter only)."""
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("space_tier", SITE.parent / "tier.py")
    tier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tier)
    assert tier.tier(97_887) == ("super", {})  # an RTX PRO 6000
    name, env = tier.tier(46_068)  # an L40S
    assert name == "bolt" and json.loads(env["MMAI_DASH_LOCAL_SERVER_AUTOSTART"])["engine"] == "llama_cpp"
    for memory in (15_360, 0):  # a T4; no GPU
        assert tier.tier(memory) == ("openrouter", {"MMAI_DASH_LOCAL_SERVER_ENABLED": "false"})
