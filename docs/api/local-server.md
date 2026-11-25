# Local model server

`matchminer_ai.llm.local_server` starts one OpenAI-compatible model server on this machine, with vLLM or llama.cpp's
`llama-server`, and `matchminer_ai.llm.gpus` describes the machine's GPUs. The dashboard's Data and models tab uses
them ([Dashboard: local model server](../apps/dashboard.md#local-model-server)); they also work on their own.

```python
from matchminer_ai.llm import gpus, local_server as ls

found = gpus.discover_gpus()
plan = ls.recommend("Qwen/Qwen3-0.6B", None, found[:1], ls.engines())
server = ls.ManagedServer(ls.ServerSpec.from_dict(plan["spec"]), "logs").start()
server.wait()                      # "ready", or "exited" / "failed" with exit_code / error
print(server.endpoint, server.spec.name)  # an OpenAI-compatible /v1 base URL and its model
server.stop()
```

## GPUs

`discover_gpus()` lists each GPU's name, compute capability, total, used and free memory, utilization and the
processes using it. It asks NVML (`pynvml`, from nvidia-ml-py) first, then `nvidia-smi`, then PyTorch, which gives no
process list or free memory (measuring it would create a CUDA context on every GPU).

- **Indices:** `GPU.index` is the index this process sees. With `CUDA_VISIBLE_DEVICES` set (indices or UUIDs), the
  listed GPUs are numbered from 0 in its order; `GPU.physical_index` is NVML's. `physical_gpu(i)` finds a GPU by its
  NVML index whatever `CUDA_VISIBLE_DEVICES` says (the retrospective fidelity evaluation's GPU wait uses it).
- **Containers:** NVML may list no processes in a container; their memory still counts as used.

## The server spec

`ServerSpec.from_dict` validates a spec and refuses unknown fields.

- **Common:** `engine` (`vllm` or `llama_cpp`), `model` (a Hugging Face repository: weights for vLLM, GGUF files
  for llama.cpp), `revision`, `served_name` (the API's model name; default the model), `tokenizer` (a repository with
  the tokenizer, for the app's prompt budgets), `gpus` (indices as this process sees them), `extra_args` and `env`.
- **vLLM:** `tensor_parallel_size` (default: the number of GPUs), `pipeline_parallel_size`, `max_model_len`,
  `gpu_memory_utilization`, `max_num_seqs`, `dtype`, `quantization`, `kv_cache_dtype`, `enable_prefix_caching`,
  `trust_remote_code`, `enforce_eager`, `reasoning_parser` (`auto` takes the model's profile and gives none for a
  model no profile knows, rather than an error), `reasoning_parser_plugin` (a file in the model repository that
  registers the parser, as Nemotron 3.5 Bolt's `ultra_v3` needs), `tool_call_parser` and `speculative_config`
  (`--speculative-config`; its draft model is downloaded first).
- **llama.cpp:** `gguf` (a file in the repository, or a quantization such as `Q8_0`; a split GGUF's parts are found),
  `ctx_size` (none: llama.cpp fits the context to free memory), `parallel`, `kv_unified` (one KV buffer for every
  slot; otherwise llama.cpp splits the context across them), `n_gpu_layers` (999: all), `flash_attn`,
  `cache_type_k` and `cache_type_v`, `batch_size` and `ubatch_size`, `split_mode`, `tensor_split`,
  `reasoning_format` (`deepseek`: the thinking in `reasoning_content`) and a speculative drafter (`draft_model`,
  `draft_gguf`, `draft_type` such as `draft-dflash`, `draft_tokens`).
- **Refused:** extra flags that would move the server off loopback or rename it (`--host`, `--port`, `--api-key`,
  `--served-model-name`, `--alias`, model and media-path flags), and environment variables other than engine and
  library tuning (`VLLM_`, `GGML_`, `NCCL_`, `PYTORCH_`, `TORCH_`, `FLASHINFER_`, `CUDA_`, not
  `CUDA_VISIBLE_DEVICES`).

## Commands

`build_command(spec, port)` returns the engine's command line and the environment variables it adds.

- **vLLM:** `vllm serve` beside this interpreter, else `python -m vllm.entrypoints.openai.api_server`, with `--host
  127.0.0.1 --port <port> --served-model-name <name>`, the spec's settings (the flag mapping of
  `llm.vllm_server`), the reasoning and tool-call parsers, and request logging off.
- **llama.cpp:** `llama-server --model <downloaded GGUF> --alias <name> --host 127.0.0.1 --port <port>`, with
  `--n-gpu-layers`, `--jinja`, `--reasoning-format`, `--kv-unified` and `--no-webui`. The app downloads the GGUF itself
  (rather than `--hf-repo`), so there is one cache, the download's progress is known and the engine never downloads.
  The binary is `$MMAI_LLAMA_SERVER`, else `llama-server` on `PATH`.
- **GPUs:** the child sees the chosen GPUs by their physical indices in PCI order (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `CUDA_VISIBLE_DEVICES`), whatever this process's own `CUDA_VISIBLE_DEVICES` is. vLLM also gets the pip CUDA toolkit
  as `CUDA_HOME` on a driver-only image (`cuda_environment`, which training's teacher servers use too).

`engines()` reports which engines are installed and their versions (vLLM from its package metadata, llama-server from
`--version`), with the reason when one is missing.

## Recipes

`presets/local_servers.yaml` holds known models' settings per engine, copied from their model cards: the engine's
repository (a GGUF repository for llama.cpp), the GPUs its recipe runs on (`compute_capabilities`, with `why`),
settings and variants by free memory. `recommend(model, engine, gpus, engines())` gives the form's starting point:
the proposed engine and why, the spec (the first variant whose memory fits the chosen GPUs' free memory), the
repository's files and GGUF choices, the fit warnings and the command. An unknown model gets generic settings, and a
repository of GGUF files only is proposed for llama.cpp.

`fit_warnings` compares the weights with the chosen GPUs' free memory, vLLM's share with each GPU's free memory
(`suggested_utilization`: (free - 1 GB) / total of the tightest GPU) and the download with the cache's free disk, and
reports a recipe's hardware limits. They are warnings, never refusals.

The default model is Nemotron 3.5 Bolt; see [the dashboard page](../apps/dashboard.md#local-model-server) for its
settings and what was left out of the model cards (image and video inputs, `--allowed-local-media-path`). Its request
profile (`llm.model_profiles.NEMOTRON_3_5_BOLT`) gives NVIDIA's sampling, the `ultra_v3` reasoning parser and the
`qwen3_coder` tool parser.

## The managed process

`ManagedServer(spec, log_dir).start()` returns at once; a monitor thread does the work:

1. **Download:** `plan_downloads` lists the repositories and files the engine needs (vLLM: configs, tokenizer,
   code, chat template and safetensors weights; llama.cpp: the GGUF, the drafter and the tokenizer's files). A free
   disk check comes first; the download runs in a child process (so a stop ends it), and `progress` compares the bytes
   in the cache with the files' sizes. Offline (`HF_HUB_OFFLINE`), only what the cache holds is used.
2. **Launch:** the engine starts in its own session (process group) with `HF_HUB_OFFLINE=1`; its log begins with the
   command line.
3. **Readiness:** the server is ready once `GET /v1/models` lists the served name; until then `phase` follows the
   log (loading weights, allocating the KV cache, capturing CUDA graphs).
4. **End:** an engine that exits on its own leaves `state` "exited" with `exit_code`; `stop()` sends SIGTERM to the
   whole process group, then SIGKILL after 45 seconds, so no engine worker keeps a GPU. On Linux the engine is
   started through a small exec wrapper that sets the parent-death signal (`with_parent_watch`), so it also gets
   SIGTERM if the process that started it dies.

::: matchminer_ai.llm.local_server
    options:
      members:
        - ServerSpec
        - ManagedServer
        - build_command
        - engines
        - recommend
        - fit_warnings
        - plan_downloads

::: matchminer_ai.llm.gpus
    options:
      members:
        - GPU
        - discover_gpus
        - physical_gpu
        - visible
