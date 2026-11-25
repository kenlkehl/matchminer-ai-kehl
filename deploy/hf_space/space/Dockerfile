# syntax=docker/dockerfile:1
# The MatchMiner-AI dashboard in synthetic mode, as a Hugging Face Docker Space. These files are deploy/hf_space/space/
# of github.com/kenlkehl/matchminer-ai-kehl; deploy/hf_space/publish.py uploads them with MMAI_COMMIT, the commit
# whose code this image installs. Stages: the source at that commit, its React frontend, llama.cpp's llama-server
# for the Space's GPU, and the Python runtime. Layers that do not depend on the commit (llama.cpp, the pinned Python
# dependencies) come first, so a code update rebuilds only the last layers.
ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu24.04

# ---------------------------------------------------------------------------------------------- llama-server
# Built for the L40S (Ada, compute capability 8.9). The local model server (Data and models) runs Nemotron 3.5 Bolt's
# GGUF with it; its card asks for a recent llama.cpp (DFlash speculative decoding). Pinned to a commit known to run
# the app's llama.cpp recipes.
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS llama
ARG LLAMA_CPP_COMMIT=33aeb17f85869279e3821f746bc645357139bc2c
ARG CUDA_ARCHITECTURES=89
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git \
    && rm -rf /var/lib/apt/lists/*
RUN git init /llama.cpp && cd /llama.cpp \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch --depth 1 origin "${LLAMA_CPP_COMMIT}" && git checkout --quiet FETCH_HEAD
# Static llama libraries in one binary; the CUDA libraries come from the runtime image, libcuda from the host driver
# (linked against the toolkit's stub, as llama.cpp's own CUDA image does).
RUN cd /llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --target llama-server -j"$(nproc)"

# ---------------------------------------------------------------------------------------------- the source
FROM alpine:3 AS source
ARG REPOSITORY=kenlkehl/matchminer-ai-kehl
RUN apk add --no-cache curl tar
COPY MMAI_COMMIT /MMAI_COMMIT
RUN commit="$(tr -d '[:space:]' < /MMAI_COMMIT)" \
    && echo "$commit" | grep -Eq '^[0-9a-f]{40}$' \
    && mkdir /src \
    && curl -fsSL "https://codeload.github.com/${REPOSITORY}/tar.gz/${commit}" | tar -xz --strip-components=1 -C /src

# ---------------------------------------------------------------------------------------------- the frontend
FROM node:26-slim AS frontend
RUN npm install -g pnpm@12.21.0
COPY --from=source /src /src
WORKDIR /src/frontend/dashboard
# The build writes src/matchminer_ai/apps/dashboard/static/ (with the How this works tab's pages).
RUN pnpm install --frozen-lockfile && pnpm build && rm -rf node_modules

# ---------------------------------------------------------------------------------------------- the runtime
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /usr/local/bin/uv
# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/venv/bin:/home/user/bin:$PATH \
    UV_PYTHON_INSTALL_DIR=/home/user/.python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
WORKDIR /home/user
RUN uv venv --python 3.13 /home/user/venv
COPY --chown=user constraints.txt /home/user/space/constraints.txt
RUN uv pip install --python /home/user/venv/bin/python -r /home/user/space/constraints.txt
COPY --from=llama --chown=user /llama.cpp/build/bin/llama-server /home/user/bin/llama-server
# The package at MMAI_COMMIT, with its built frontend.
COPY --from=frontend --chown=user /src /home/user/matchminer-ai-kehl
RUN uv pip install --python /home/user/venv/bin/python -c /home/user/space/constraints.txt \
        "/home/user/matchminer-ai-kehl[apps,sql]"
COPY --chown=user MMAI_COMMIT entrypoint.sh site.yaml /home/user/space/
ENV MMAI_SITE_CONFIG=/home/user/space/site.yaml \
    MMAI_DATA_ROOT=/home/user/mmai-data \
    MMAI_LLAMA_SERVER=/home/user/bin/llama-server \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["/bin/sh", "/home/user/space/entrypoint.sh"]
