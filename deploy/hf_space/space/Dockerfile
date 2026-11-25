# syntax=docker/dockerfile:1
# The MatchMiner-AI dashboard in synthetic mode, as a Hugging Face Docker Space. These files are deploy/hf_space/space/
# of github.com/kenlkehl/matchminer-ai-kehl; deploy/hf_space/publish.py uploads them with MMAI_COMMIT, the commit
# whose code this image installs. Stages: the source at that commit, its React frontend, llama.cpp's llama-server
# for the Space's GPU, and the Python runtime. Layers that do not depend on the commit (llama.cpp, the pinned Python
# dependencies) come first, so a code update rebuilds only the last layers.
#
# Two variants (MMAI_VARIANT; publish.py sets it per Space, and a Space variable of that name would too):
# - full: the app and the local model server's engines (vLLM with FlashInfer's kernels, llama.cpp), on the CUDA
#   runtime image;
# - lite: the app only, for a Space that hosts no LLM (OpenRouter only, tier.py): no llama.cpp stage, no vLLM or
#   FlashInfer, no CUDA compiler, on plain Ubuntu (torch brings its CUDA libraries). BuildKit builds only the stages
#   the chosen variant uses.
ARG CUDA_VERSION=12.8.1
ARG UBUNTU=ubuntu24.04
ARG MMAI_VARIANT=full

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
# The toolkit's driver stub, which the runtime's pip CUDA toolkit lacks (see the runtime stage).
RUN cp /usr/local/cuda/lib64/stubs/libcuda.so /libcuda-stub.so

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
# The full variant's base: the CUDA 12.8 runtime image (llama-server's libraries); torch and vLLM bring their own
# CUDA 13 libraries and nvcc (pip, pinned to torch's 13.0 in constraints-engines.txt), which the local model server
# finds for vLLM's kernel builds.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS base-full
# The lite variant's: plain Ubuntu; torch's wheels bring the CUDA libraries, the container runtime the driver.
FROM ubuntu:24.04 AS base-lite
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# The Python environment with the pinned dependencies both variants share. A C/C++ compiler is needed at run time:
# Triton (the decision model's convolution kernel), and in the full variant torch.compile and the kernels vLLM
# compiles when it starts.
FROM base-${MMAI_VARIANT} AS base
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

# The lite variant adds nothing; installing the package leaves out vLLM (its only engine dependency).
FROM base AS engines-lite
COPY --chown=user no-engines.txt /home/user/space/no-engines.txt
ENV MMAI_PACKAGE_PINS="--overrides /home/user/space/no-engines.txt"

# The full variant: vLLM and its run-time kernel builds, and llama-server.
FROM base AS engines-full
COPY --chown=user constraints-engines.txt /home/user/space/constraints-engines.txt
RUN uv pip install --python /home/user/venv/bin/python -c /home/user/space/constraints.txt \
        -r /home/user/space/constraints-engines.txt
ENV MMAI_PACKAGE_PINS="-c /home/user/space/constraints-engines.txt"
# FlashInfer's precompiled kernels for flashinfer-python 0.7.2.post1 and CUDA 13.0, so vLLM loads them instead of
# compiling FlashInfer's kernels when it starts: the cubins, and the AOT modules for SM 12.0 (the RTX PRO 6000), the
# jit-cache shim with that one provider (FlashInfer's own "minimal" install). Other GPUs run llama.cpp (tier.py).
RUN uv pip install --python /home/user/venv/bin/python --no-deps --index-url https://flashinfer.ai/whl/cu130 \
        "flashinfer-jit-cache==0.7.2.post1+cu130" "flashinfer-jit-cache-sm120f==0.7.2.post1+cu130" \
    && uv pip install --python /home/user/venv/bin/python --no-deps --index-url https://flashinfer.ai/whl \
        "flashinfer-cubin==0.7.2.post1"
# The pip CUDA toolkit as kernels built at run time link against it. FlashInfer links with -L$CUDA_HOME/lib64
# -L$CUDA_HOME/lib64/stubs -lcudart -lcuda (some modules also -lnvrtc, -lcublas, -lcublasLt); nvcc's own links, as
# tilelang's, search the same two folders. The wheels ship only lib/, with versioned names. So: lib64 -> lib, an
# unversioned name beside each versioned library, and the CUDA toolkit's driver stub as lib/stubs/libcuda.so. The stub
# is only linked against (its SONAME is libcuda.so.1); at run time the host driver's libcuda.so.1 is loaded.
ARG CUDA_PIP=/home/user/venv/lib/python3.13/site-packages/nvidia/cu13
COPY --from=llama --chown=user /libcuda-stub.so ${CUDA_PIP}/lib/stubs/libcuda.so
RUN cd "${CUDA_PIP}" && ln -s lib lib64 \
    && for f in lib/lib*.so.*; do b="${f%%.so.*}.so"; [ -e "$b" ] || ln -s "${f#lib/}" "$b"; done \
    && for l in cudart nvrtc cublas cublasLt; do test -e "lib64/lib$l.so"; done && test -e lib64/stubs/libcuda.so
# The FlashInfer kernel vLLM compiles at start that the jit-cache cannot hold, being built per model: the Mamba state
# update for Nemotron 3.5 Super with site.yaml's flashinfer Mamba backend (fp16 state cache, Philox-5 stochastic
# rounding; head dim 64, state 128, 6 tokens per step: MTP's 5 and 1). Built here (nvcc needs no GPU) into FlashInfer's
# cache with what vLLM's run-time build uses (the CUDA_HOME the local server sets, SM 12.0), so a cold start loads it
# instead of compiling it. Any other such kernel is still compiled when first needed.
RUN CUDA_HOME="${CUDA_PIP}" FLASHINFER_CUDA_ARCH_LIST=12.0f MAX_JOBS=4 python -c "import torch; \
from flashinfer.jit.mamba import gen_selective_state_update_sm100_module as gen; \
gen(torch.float16, torch.bfloat16, torch.bfloat16, torch.float32, torch.int32, None, 64, 128, 6, torch.int32, \
torch.int32, philox_rounds=5).build()"
COPY --from=llama --chown=user /llama.cpp/build/bin/llama-server /home/user/bin/llama-server
ENV MMAI_LLAMA_SERVER=/home/user/bin/llama-server

# The variant's environment, then the package at MMAI_COMMIT with its built frontend.
FROM engines-${MMAI_VARIANT} AS runtime
ARG MMAI_VARIANT
COPY --from=frontend --chown=user /src /home/user/matchminer-ai-kehl
RUN uv pip install --python /home/user/venv/bin/python -c /home/user/space/constraints.txt $MMAI_PACKAGE_PINS \
        "/home/user/matchminer-ai-kehl[apps,sql]"
COPY --chown=user MMAI_COMMIT entrypoint.sh prefetch.py site.yaml tier.py /home/user/space/
ENV MMAI_VARIANT=${MMAI_VARIANT} \
    MMAI_SITE_CONFIG=/home/user/space/site.yaml \
    MMAI_DATA_ROOT=/home/user/mmai-data \
    HF_HOME=/home/user/.cache/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["/bin/sh", "/home/user/space/entrypoint.sh"]
