#!/bin/sh
# The Space's start command: the dashboard in synthetic mode on the Space's port. Everything else (the published
# data and models, the local model server, framing by the Space page) is in site.yaml.
set -eu
# Persistent storage, when the Space has a storage bucket mounted at /data: the downloads and the prepared synthetic
# run then survive restarts.
if [ -d /data ] && [ -w /data ]; then
    export MMAI_DATA_ROOT=/data/mmai HF_HOME=/data/huggingface
fi
mkdir -p "$MMAI_DATA_ROOT"
echo "MatchMiner-AI $(cat /home/user/space/MMAI_COMMIT 2>/dev/null || echo unknown); data in $MMAI_DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi: none"
# The local model server by the GPU's memory (tier.py): Nemotron 3.5 Super with vLLM from 90 GB, Bolt with llama.cpp
# from 40 GB, else none (OpenRouter with each visitor's own key only).
# The hosted OpenRouter key (the Space secret MMAI_OPENROUTER_HOSTED_KEY, the lite Space's only) goes to the app
# alone, which removes it from its environment at startup; these helpers run without it.
eval "$(env -u MMAI_OPENROUTER_HOSTED_KEY python /home/user/space/tier.py)"
if [ "${MMAI_SPACE_TIER:-}" != openrouter ]; then
    # The local model server's weights download while the app starts (prefetch.py).
    env -u MMAI_OPENROUTER_HOSTED_KEY python /home/user/space/prefetch.py &
fi
exec mmai-dashboard --mode synthetic --port 7860 --allow-downloads serve
