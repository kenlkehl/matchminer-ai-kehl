# syntax=docker/dockerfile:1
# The MatchMiner-AI dashboard in synthetic mode, as a Hugging Face Docker Space. These files are deploy/hf_space/space/
# of github.com/kenlkehl/matchminer-ai-kehl; deploy/hf_space/publish.py uploads them with MMAI_COMMIT, the commit
# whose code this image installs. Stages: the source at that commit, its React frontend, llama.cpp's llama-server
# for the Space's GPU, and the Python runtime. Layers that do not depend on the commit (llama.cpp, the pinned Python
# dependencies) come first, so a code update rebuilds only the last layers.
ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu24.04

# ---------------------------------------------------------------------------------------------- llama-server
# The local model server's second engine: Nemotron 3.5 Bolt's GGUF, which entrypoint.sh starts on a GPU too small
# for Nemotron 3.5 Super (vLLM, site.yaml), such as an L40S. Built for Ada (8.9) and Blackwell RTX PRO (12.0); its
# card asks for a recent llama.cpp (DFlash speculative decoding). Pinned to a commit known to run the app's recipes.
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS llama
ARG LLAMA_CPP_COMMIT=33aeb17f85869279e3821f746bc645357139bc2c
ARG CUDA_ARCHITECTURES="89;120"
ARG GCC_VERSION=14
ENV DEBIAN_FRONTEND=noninteractive
# GCC 14, as llama.cpp's own CUDA image at this commit builds with (CUDA 12.8 accepts it as nvcc's host compiler).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git \
        gcc-${GCC_VERSION} g++-${GCC_VERSION} \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}
RUN git init /llama.cpp && cd /llama.cpp \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch --depth 1 origin "${LLAMA_CPP_COMMIT}" && git checkout --quiet FETCH_HEAD
# Static llama libraries in one binary; the CUDA libraries come from the runtime image, libcuda from the host driver
# (linked against the toolkit's stub, as llama.cpp's own CUDA image does). One GPU: no NCCL. No embedded chat UI
# (the app uses only the API), so nothing is downloaded at build time. At most 8 compile jobs: each nvcc job needs
# gigabytes of memory.
RUN cd /llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DGGML_CUDA_NCCL=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && jobs="$(nproc)" && if [ "$jobs" -gt 8 ]; then jobs=8; fi \
    && cmake --build build --target llama-server -j"$jobs" \
    && ! ldd build/bin/llama-server | grep -E "libllama|libggml|libmtmd|libnccl"

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
# The runtime: the CUDA 12.8 runtime image (llama-server's libraries); torch and vLLM bring
# their own CUDA 13 libraries and nvcc (pip, pinned to torch's 13.0 in constraints.txt), which the local model server
# finds for vLLM's kernel builds. A C/C++ compiler is needed at run time too: Triton, torch.compile and any kernel
# FlashInfer's precompiled ones lack are compiled when vLLM starts.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.11 /uv /usr/local/bin/uv
# Spaces run the container as user 1000; the Ubuntu 24.04 base images already have one (ubuntu), so it is replaced.
RUN (userdel -r ubuntu 2>/dev/null || true) && useradd -m -u 1000 -s /bin/bash user
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
# FlashInfer's precompiled kernels for flashinfer-python 0.7.2.post1 and CUDA 13.0, so vLLM loads them instead of
# compiling FlashInfer's kernels when it starts: the cubins, and the AOT modules for SM 12.0 (the RTX PRO 6000), the
# jit-cache shim with that one provider (FlashInfer's own "minimal" install). Other GPUs run llama.cpp (entrypoint.sh).
RUN uv pip install --python /home/user/venv/bin/python --no-deps --index-url https://flashinfer.ai/whl/cu130 \
        "flashinfer-jit-cache==0.7.2.post1+cu130" "flashinfer-jit-cache-sm120f==0.7.2.post1+cu130" \
    && uv pip install --python /home/user/venv/bin/python --no-deps --index-url https://flashinfer.ai/whl \
        "flashinfer-cubin==0.7.2.post1"
COPY --from=llama --chown=user /llama.cpp/build/bin/llama-server /home/user/bin/llama-server
# The package at MMAI_COMMIT, with its built frontend.
COPY --from=frontend --chown=user /src /home/user/matchminer-ai-kehl
RUN uv pip install --python /home/user/venv/bin/python -c /home/user/space/constraints.txt \
        "/home/user/matchminer-ai-kehl[apps,sql]"
COPY --chown=user MMAI_COMMIT entrypoint.sh prefetch.py site.yaml /home/user/space/
ENV MMAI_SITE_CONFIG=/home/user/space/site.yaml \
    MMAI_DATA_ROOT=/home/user/mmai-data \
    MMAI_LLAMA_SERVER=/home/user/bin/llama-server \
    HF_HOME=/home/user/.cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["/bin/sh", "/home/user/space/entrypoint.sh"]
