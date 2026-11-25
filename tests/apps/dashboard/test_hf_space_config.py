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
