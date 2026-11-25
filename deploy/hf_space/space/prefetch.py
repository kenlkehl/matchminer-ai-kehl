"""Download the local model server's weights into the Hugging Face cache while the app starts.

entrypoint.sh runs this in the background. The app starts the server only after the decision model has loaded, and
the server then finds its files in the cache (or waits on the files still being written): the same files, those the
server itself downloads (``matchminer_ai.llm.local_server.plan_downloads``). Only a vLLM spec is prefetched; a
llama.cpp recipe chooses its GGUF by the GPU's free memory when it starts.
"""
from __future__ import annotations

import json
import os
import sys
import time


def autostart_spec() -> dict | None:
    """The spec the app will start: ``MMAI_DASH_LOCAL_SERVER_AUTOSTART``, else the site config's."""
    from matchminer_ai.site import load_site_config

    value = os.environ.get("MMAI_DASH_LOCAL_SERVER_AUTOSTART") or (
        load_site_config().get("dashboard.synthetic.local_server") or {}).get("autostart")
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else None


def main() -> int:
    from huggingface_hub import snapshot_download

    from matchminer_ai.llm.local_server import ServerSpec, plan_downloads

    values = autostart_spec()
    if not values or values.get("engine") != "vllm":
        return 0
    spec = ServerSpec.from_dict(values)
    started = time.monotonic()
    for item in plan_downloads(spec):
        print(f"prefetch: {item.repo} ({item.size / 1e9:.1f} GB) ...", flush=True)
        snapshot_download(item.repo, revision=item.revision, allow_patterns=item.files, max_workers=16)
        print(f"prefetch: {item.repo} done after {round(time.monotonic() - started)} s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # the server downloads what is missing itself
        print(f"prefetch failed ({type(exc).__name__}: {exc}); the local server will download its files",
              flush=True)
        sys.exit(0)
