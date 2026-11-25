#!/bin/sh
# The Space's start command: the dashboard in synthetic mode on the Space's port. Everything else (the published
# data and models, the local model server, framing by the Space page) is in site.yaml.
set -eu
# Persistent storage, when the Space has a storage bucket mounted at /data: the downloads and the prepared synthetic
# run (with its TrialSpace vectors) then survive restarts.
if [ -d /data ] && [ -w /data ]; then
    export MMAI_DATA_ROOT=/data/mmai HF_HOME=/data/huggingface
fi
mkdir -p "$MMAI_DATA_ROOT"
echo "MatchMiner-AI $(cat /home/user/space/MMAI_COMMIT 2>/dev/null || echo unknown); data in $MMAI_DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU visible"
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available:', torch.cuda.is_available())" \
    || true
exec mmai-dashboard --mode synthetic --port 7860 --allow-downloads serve
