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
# torch's view of the GPU (torch 2.14 is a CUDA 13 build: host driver 580 or later), and its memory in MiB.
GPU_MIB=$(python - <<'EOF' || echo 0
import sys
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "CUDA", torch.version.cuda, "available:", ok,
      torch.cuda.get_device_name(0) if ok else "", file=sys.stderr)
print(torch.cuda.get_device_properties(0).total_memory // 2**20 if ok else 0)
EOF
)
if [ "${GPU_MIB:-0}" -lt 90000 ]; then
    # Too small for Nemotron 3.5 Super (site.yaml's vLLM server), e.g. an L40S: Bolt's GGUF with llama.cpp.
    echo "GPU memory ${GPU_MIB:-0} MiB: the local model server runs Nemotron 3.5 Bolt with llama.cpp"
    export MMAI_DASH_LOCAL_SERVER_AUTOSTART='{"engine":"llama_cpp","model":"nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB","gpus":[0]}'
fi
# The local model server's weights download while the app starts (prefetch.py).
python /home/user/space/prefetch.py &
exec mmai-dashboard --mode synthetic --port 7860 --allow-downloads serve
