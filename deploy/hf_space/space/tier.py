"""The Space's tier, by its GPU's memory: which local model server it runs, if any. entrypoint.sh evaluates the shell
exports this prints (one image serves every Space, whatever its hardware).

- ``super``, 90 GB or more (an RTX PRO 6000): site.yaml's vLLM server with Nemotron 3.5 Super NVFP4.
- ``bolt``, 40 GB or more (an L40S): Nemotron 3.5 Bolt's GGUF with llama.cpp (its 24 GB file and KV cache, beside
  the decision model and TrialSpace).
- ``openrouter``, less or no GPU (a T4): no local model server. With site.yaml's ``only`` and ``allow_openrouter``,
  the LLM tasks then use OpenRouter with each visitor's own API key, and nothing else.
"""
from __future__ import annotations

import json
import shlex
import sys

SUPER_MIB = 90_000
BOLT_MIB = 40_000
BOLT = {"engine": "llama_cpp", "model": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "gpus": [0]}


def tier(gpu_mib: int) -> tuple[str, dict[str, str]]:
    """``(name, environment)`` for a GPU of ``gpu_mib`` MiB (0: none)."""
    if gpu_mib >= SUPER_MIB:
        return "super", {}
    if gpu_mib >= BOLT_MIB:
        return "bolt", {"MMAI_DASH_LOCAL_SERVER_AUTOSTART": json.dumps(BOLT, separators=(",", ":"))}
    return "openrouter", {"MMAI_DASH_LOCAL_SERVER_ENABLED": "false"}


def gpu_mib() -> int:
    """The first GPU's memory in MiB as torch sees it (0 without one). Torch 2.14 is a CUDA 13 build: host driver 580
    or later."""
    try:
        import torch

        available = torch.cuda.is_available()
        print("torch", torch.__version__, "CUDA", torch.version.cuda, "available:", available,
              torch.cuda.get_device_name(0) if available else "", file=sys.stderr)
        return int(torch.cuda.get_device_properties(0).total_memory // 2**20) if available else 0
    except Exception as exc:  # no usable GPU: the smallest tier
        print(f"tier: no GPU ({type(exc).__name__})", file=sys.stderr)
        return 0


def main() -> int:
    memory = gpu_mib()
    name, environment = tier(memory)
    print(f"tier: {name} (GPU memory {memory} MiB)", file=sys.stderr)
    for key, value in {"MMAI_SPACE_TIER": name, **environment}.items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
