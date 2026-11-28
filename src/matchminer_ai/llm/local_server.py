"""A local OpenAI-compatible model server on this machine: vLLM or llama.cpp's ``llama-server``.

The dashboard's Data and models tab starts one (``apps.dashboard.local_server``), also for machines that cannot
reach a hosted endpoint. Both engines serve the same OpenAI-compatible API on loopback, so every LLM task can use the
server as an ordinary OpenAI-compatible connection.

- **Spec:** ``ServerSpec`` holds the engine, the Hugging Face model, the GPUs and the engine's parameters; unknown
  fields are refused (``ServerSpec.from_dict``).
- **Command:** ``build_command`` turns a spec into the engine's command line and environment: loopback only, the
  chosen GPUs in ``CUDA_VISIBLE_DEVICES``, request logging off.
- **Recipes:** ``presets/local_servers.yaml`` gives known models' settings per engine and their hardware limits;
  ``recommend`` fills a spec from them (or from generic defaults) and proposes an engine for the chosen GPUs.
- **Download:** before launch, the files the engine needs are downloaded into the Hugging Face cache in a child
  process (so a stop ends it), after a free-space check. An app that does not download (``HF_HUB_OFFLINE``) uses
  only what the cache holds.
- **Process:** ``ManagedServer`` starts the engine in its own session (process group), follows its log for the
  phase, calls it ready once ``GET /v1/models`` lists the served name, notices when it exits, and stops the whole
  group: SIGTERM, then SIGKILL after ``STOP_GRACE_SECONDS``. On Linux the engine also gets SIGTERM if the app dies
  (``with_parent_watch``).

The server's log has the command line and the engine's own messages; request logging is off, so prompts are not
written to it.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest

ENGINES = {"vllm": "vLLM", "llama_cpp": "llama.cpp (llama-server)"}
HOST = "127.0.0.1"
GB = 1024**3
STOP_GRACE_SECONDS = 45
LLAMA_SERVER_ENV = "MMAI_LLAMA_SERVER"  # the llama-server binary (default: llama-server on PATH)
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,96}$")
NAME = re.compile(r"^[\w.:/@+-]{1,200}$")
TOKEN = re.compile(r"^[\w.+-]{1,64}$")
# Flags the server sets itself: the address stays loopback and the names are the app's.
RESERVED_FLAGS = frozenset({
    "--host", "--port", "--uds", "--api-key", "--api-key-file", "--served-model-name", "--alias", "-a", "--model",
    "-m", "--hf-repo", "-hf", "-hfr", "--hf-file", "-hff", "--allowed-local-media-path", "--media-path",
    "--ssl-keyfile", "--ssl-certfile", "--ssl-key-file", "--ssl-cert-file", "--reasoning-parser-plugin",
    "--model-draft", "-md", "--spec-draft-model", "--connect",
})
# Environment variables a spec may set: engine and library tuning only.
ENV_PREFIXES = ("VLLM_", "GGML_", "NCCL_", "PYTORCH_", "TORCH_", "FLASHINFER_", "CUDA_")
ENV_RESERVED = frozenset({"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_HOST_IP", "VLLM_API_KEY"})
CACHE_TYPES = frozenset({"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"})


# ---------------------------------------------------------------------------------------------- the spec


@dataclass
class ServerSpec:
    """One server: the engine, model, GPUs and engine parameters (None: the engine's own default)."""

    engine: str = "llama_cpp"
    model: str = ""  # a Hugging Face repo: weights for vLLM, GGUF files for llama.cpp
    revision: str | None = None
    served_name: str = ""  # the API's model name (default: the model)
    tokenizer: str = ""  # a Hub repo with the model's tokenizer, for the app's prompt budgets (default: the model)
    gpus: list[int] = field(default_factory=list)  # indices as the app sees them (CUDA_VISIBLE_DEVICES)
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # ---- vLLM
    tensor_parallel_size: int | None = None  # default: the number of GPUs
    pipeline_parallel_size: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    dtype: str | None = None
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    enable_prefix_caching: bool | None = None
    trust_remote_code: bool = False
    enforce_eager: bool = False
    reasoning_parser: str = "auto"  # auto (the model's profile; none when unknown), none, or a parser name
    reasoning_parser_plugin: str | None = None  # a file in the model repo that registers the parser
    tool_call_parser: str = "auto"
    speculative_config: dict | None = None  # vLLM's --speculative-config; its "model" is downloaded first
    # ---- llama.cpp
    gguf: str | None = None  # a file in the repo, or a quantization such as Q8_0
    ctx_size: int | None = None  # None: llama.cpp fits the context to free memory
    parallel: int | None = None
    kv_unified: bool = True  # one KV buffer for every slot (else the context is split across them)
    n_gpu_layers: int = 999
    flash_attn: str = "auto"
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    split_mode: str | None = None
    tensor_split: list[float] | None = None
    reasoning_format: str = "deepseek"
    draft_model: str | None = None  # speculative decoding: a GGUF repo
    draft_gguf: str | None = None
    draft_type: str | None = None  # llama.cpp's --spec-type, e.g. draft-dflash
    draft_tokens: int | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ServerSpec":
        """A validated spec (``ValueError`` with the reason; unknown fields are refused)."""
        defaults = {f.name: f.default for f in fields(cls)}
        unknown = sorted(set(values) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown server settings: {', '.join(unknown)}.")
        # None leaves a setting at its default (the engine's own, for the optional ones).
        spec = cls(**{k: v for k, v in values.items() if v is not None or defaults[k] is None})
        spec.validate()
        return spec

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {', '.join(ENGINES)}.")
        if not REPO_ID.match(self.model or ""):
            raise ValueError("Enter the model as a Hugging Face repository, such as org/model.")
        for name in ("revision", "served_name", "tokenizer"):
            value = getattr(self, name)
            if value and not NAME.match(value):
                raise ValueError(f"{name} has characters a model name cannot have.")
        if self.tokenizer and not REPO_ID.match(self.tokenizer):
            raise ValueError("Enter the tokenizer as a Hugging Face repository.")
        if not isinstance(self.gpus, list) or not all(isinstance(g, int) and g >= 0 for g in self.gpus) \
                or len(set(self.gpus)) != len(self.gpus):
            raise ValueError("gpus must be distinct GPU indices.")
        if self.engine == "vllm" and not self.gpus:
            raise ValueError("vLLM needs at least one GPU.")
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "max_model_len", "max_num_seqs", "ctx_size",
                     "parallel", "batch_size", "ubatch_size", "draft_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive whole number.")
        if not isinstance(self.n_gpu_layers, int) or self.n_gpu_layers < 0:
            raise ValueError("n_gpu_layers must be 0 or more.")
        if self.gpu_memory_utilization is not None and not 0 < float(self.gpu_memory_utilization) <= 1:
            raise ValueError("gpu_memory_utilization must be above 0 and at most 1.")
        for name in ("dtype", "quantization", "kv_cache_dtype", "reasoning_parser", "tool_call_parser",
                     "reasoning_format", "draft_type"):
            value = getattr(self, name)
            if value and not TOKEN.match(str(value)):
                raise ValueError(f"{name} must be a single name.")
        if self.flash_attn not in ("on", "off", "auto"):
            raise ValueError("flash_attn must be on, off or auto.")
        if self.split_mode not in (None, "none", "layer", "row", "tensor"):
            raise ValueError("split_mode must be none, layer, row or tensor.")
        for name in ("cache_type_k", "cache_type_v"):
            if getattr(self, name) not in (None, *CACHE_TYPES):
                raise ValueError(f"{name} must be one of {', '.join(sorted(CACHE_TYPES))}.")
        if self.tensor_split is not None and (not isinstance(self.tensor_split, list) or not all(
                isinstance(v, (int, float)) and v >= 0 for v in self.tensor_split)):
            raise ValueError("tensor_split must be a list of non-negative numbers.")
        for name in ("gguf", "draft_gguf", "reasoning_parser_plugin"):
            value = getattr(self, name)
            if value and (not re.match(r"^[\w.+-][\w./+-]{0,250}$", value) or ".." in value.split("/")):
                raise ValueError(f"{name} must be a file name in the repository.")
        if self.draft_model and not REPO_ID.match(self.draft_model):
            raise ValueError("draft_model must be a Hugging Face repository.")
        if self.speculative_config is not None:
            if not isinstance(self.speculative_config, dict) or len(json.dumps(self.speculative_config)) > 4000:
                raise ValueError("speculative_config must be a small JSON object.")
            draft = self.speculative_config.get("model")
            if draft is not None and not REPO_ID.match(str(draft)):
                raise ValueError("speculative_config.model must be a Hugging Face repository.")
        if not isinstance(self.extra_args, list) or len(self.extra_args) > 100:
            raise ValueError("extra_args must be a list of command-line arguments.")
        for arg in self.extra_args:
            if not isinstance(arg, str) or len(arg) > 4000 or any(ord(c) < 32 for c in arg):
                raise ValueError("Each extra argument must be one line of text.")
            if arg.split("=", 1)[0] in RESERVED_FLAGS:
                raise ValueError(f"{arg.split('=', 1)[0]} is set by the app.")
        if not isinstance(self.env, dict) or len(self.env) > 50:
            raise ValueError("env must be a mapping of environment variables.")
        for key, value in self.env.items():
            if not re.match(r"^[A-Z][A-Z0-9_]{0,99}$", str(key)) or not str(key).startswith(ENV_PREFIXES) \
                    or key in ENV_RESERVED:
                raise ValueError(f"{key} cannot be set here (allowed: {', '.join(ENV_PREFIXES)} variables).")
            if not isinstance(value, str) or len(value) > 1000 or any(ord(c) < 32 for c in value):
                raise ValueError(f"{key} must be one line of text.")

    @property
    def name(self) -> str:
        """The API's model name."""
        return self.served_name or self.model

    @property
    def tokenizer_repo(self) -> str:
        return self.tokenizer or self.model


# ---------------------------------------------------------------------------------------------- engines

_llama_versions: dict[str, tuple[str | None, str | None]] = {}


def llama_server_path(explicit: str | None = None) -> str | None:
    """The llama-server binary: ``explicit``, else ``$MMAI_LLAMA_SERVER``, else ``llama-server`` on PATH."""
    candidate = explicit or os.environ.get(LLAMA_SERVER_ENV) or "llama-server"
    return shutil.which(candidate) or (candidate if Path(candidate).is_file() and os.access(candidate, os.X_OK)
                                       else None)


def vllm_command() -> list[str] | None:
    """``vllm serve`` next to this interpreter, else its API server module (None when vLLM is not installed)."""
    binary = Path(sys.executable).with_name("vllm")
    if binary.is_file():
        return [str(binary), "serve"]
    import importlib.util

    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] if importlib.util.find_spec("vllm") else None


def engines(llama_server: str | None = None) -> dict[str, dict]:
    """Which engines are installed: ``{engine: {label, available, version, command, reason}}``."""
    from importlib import metadata

    result: dict[str, dict] = {}
    command = vllm_command()
    version: str | None
    try:
        version = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        version = None
    result["vllm"] = {"label": ENGINES["vllm"], "available": bool(command), "version": version, "command": command,
                      "reason": None if command else "vLLM is not installed in this Python environment."}
    path = llama_server_path(llama_server)
    problem: str | None = f"llama-server was not found (put it on PATH or set {LLAMA_SERVER_ENV})."
    version = None
    if path:
        if path not in _llama_versions:
            try:
                out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
                text = (out.stdout + out.stderr).strip()
                found = re.search(r"version:\s*(.+)", text)
                _llama_versions[path] = (found.group(1).strip() if found else text.splitlines()[0][:100]
                                         if text else "unknown", None)
            except (OSError, subprocess.SubprocessError) as exc:
                _llama_versions[path] = (None, f"llama-server did not run ({type(exc).__name__}).")
        version, problem = _llama_versions[path]
    result["llama_cpp"] = {"label": ENGINES["llama_cpp"], "available": bool(path and not problem),
                           "version": version, "command": [path] if path else None, "reason": problem}
    return result


def cuda_environment(executable: str, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``env`` (default: this process's) with ``executable``'s folder first on PATH and, when no CUDA toolkit is
    visible (no ``CUDA_HOME``, no ``nvcc``), the pip CUDA toolkit of the interpreter beside it as ``CUDA_HOME``,
    for a driver-only accelerator image (vLLM compiles kernels at startup)."""
    env = dict(os.environ if env is None else env)
    path = Path(shutil.which(executable) or executable).expanduser()
    env["PATH"] = str(path.parent) + os.pathsep + env.get("PATH", "")
    if not env.get("CUDA_HOME") and not shutil.which("nvcc", path=env["PATH"]):
        python = path if path.name.startswith("python") else path.parent / "python"
        probe = ("import sysconfig; from pathlib import Path; "
                 "root=Path(sysconfig.get_paths()['purelib'])/'nvidia'; "
                 "print(next((str(p) for p in sorted(root.glob('cu*'), reverse=True) "
                 "if (p/'bin/nvcc').is_file()), ''))")
        try:
            cuda_home = subprocess.check_output([str(python), "-c", probe], text=True, timeout=60).strip()
        except (OSError, subprocess.SubprocessError):
            cuda_home = ""
        if cuda_home:
            env["CUDA_HOME"] = cuda_home
            env["PATH"] = str(Path(cuda_home) / "bin") + os.pathsep + env["PATH"]
    return env


def visible_devices(spec: ServerSpec, gpus: Sequence | None = None,
                    parent: str | None = None) -> dict[str, str]:
    """The child's GPU variables for ``spec.gpus`` (indices as this process sees them).

    With the GPUs' descriptions from NVML or nvidia-smi (``llm.gpus.GPU``), the child gets their physical indices
    in PCI order (``CUDA_DEVICE_ORDER=PCI_BUS_ID``), so the mapping holds whatever this process's own
    ``CUDA_VISIBLE_DEVICES`` is. Otherwise each index is mapped through ``parent`` (this process's
    ``CUDA_VISIBLE_DEVICES``) when it is set. No GPUs: none visible."""
    if not spec.gpus:
        return {"CUDA_VISIBLE_DEVICES": ""}
    known = {g.index: g for g in gpus or ()}
    if all(i in known and getattr(known[i], "source", "torch") != "torch" for i in spec.gpus):
        return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": ",".join(str(known[i].physical_index) for i in spec.gpus)}
    entries = [e.strip() for e in parent.split(",")] if parent else None
    if entries is not None and any(i >= len(entries) for i in spec.gpus):
        raise ValueError("A chosen GPU is not visible to this app.")
    return {"CUDA_VISIBLE_DEVICES": ",".join(entries[i] if entries is not None else str(i) for i in spec.gpus)}


def resolve_parser(spec: ServerSpec) -> str | None:
    """vLLM's reasoning parser: "auto" takes the model's profile and, unlike the package's own resolution, gives
    none for a model it does not know (a launch should not fail on it)."""
    from .reasoning import resolve_reasoning_parser

    for name in (spec.name, spec.model):
        try:
            parser = resolve_reasoning_parser(name, spec.reasoning_parser)
        except ValueError:
            continue
        return parser
    return None


def resolve_tool_parser(spec: ServerSpec) -> str | None:
    from .reasoning import resolve_tool_call_parser

    return resolve_tool_call_parser(spec.name, spec.tool_call_parser) or (
        resolve_tool_call_parser(spec.model, spec.tool_call_parser) if spec.name != spec.model else None)


def build_command(spec: ServerSpec, port: int, *, files: Mapping[str, str] | None = None,
                  engine_command: Sequence[str] | None = None, gpus: Sequence | None = None,
                  parent_visible: str | None = None) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)``: the engine's command line and the environment variables it adds.

    ``files``: downloaded paths (``gguf``, ``draft``, ``reasoning_parser_plugin``); without them the command names
    the repository's files (a preview). ``engine_command``: ``vllm serve``, the API server module or the
    llama-server binary (default: what ``engines`` finds). ``gpus`` and ``parent_visible``: see
    ``visible_devices``."""
    files = dict(files or {})
    env = {**spec.env, **visible_devices(spec, gpus, parent_visible)}
    if spec.engine == "vllm":
        from .vllm_server import _engine_flags

        base = list(engine_command or vllm_command() or ["vllm", "serve"])
        argv = [*base, spec.model] if base[-1] == "serve" else [*base, "--model", spec.model]
        argv += ["--host", HOST, "--port", str(port), "--served-model-name", spec.name]
        if spec.revision:
            argv += ["--revision", spec.revision]
        argv += ["--tensor-parallel-size", str(spec.tensor_parallel_size or max(1, len(spec.gpus)))]
        for flag, value in (("--pipeline-parallel-size", spec.pipeline_parallel_size),
                            ("--max-model-len", spec.max_model_len),
                            ("--gpu-memory-utilization", spec.gpu_memory_utilization),
                            ("--max-num-seqs", spec.max_num_seqs), ("--dtype", spec.dtype),
                            ("--quantization", spec.quantization), ("--kv-cache-dtype", spec.kv_cache_dtype)):
            if value is not None:
                argv += [flag, str(value)]
        engine = {"trust_remote_code": True} if spec.trust_remote_code else {}
        if spec.enable_prefix_caching is not None:
            engine["enable_prefix_caching"] = spec.enable_prefix_caching
        argv += _engine_flags(engine)
        if spec.enforce_eager:
            argv.append("--enforce-eager")
        parser = resolve_parser(spec)
        if parser:
            if spec.reasoning_parser_plugin:
                argv += ["--reasoning-parser-plugin",
                         files.get("reasoning_parser_plugin") or f"<{spec.model}>/{spec.reasoning_parser_plugin}"]
            argv += ["--reasoning-parser", parser]
        tool = resolve_tool_parser(spec)
        if tool:
            argv += ["--enable-auto-tool-choice", "--tool-call-parser", tool]
        if spec.speculative_config:
            argv += ["--speculative-config", json.dumps(spec.speculative_config, separators=(",", ":"))]
        argv += ["--no-enable-log-requests", "--disable-uvicorn-access-log"]
    else:
        base = list(engine_command or [llama_server_path() or "llama-server"])
        gguf = files.get("gguf") or f"<{spec.model}>/{spec.gguf or '*.gguf'}"
        argv = [*base, "--model", gguf, "--alias", spec.name, "--host", HOST, "--port", str(port),
                "--n-gpu-layers", str(spec.n_gpu_layers if spec.gpus else 0)]
        for flag, value in (("--ctx-size", spec.ctx_size), ("--parallel", spec.parallel),
                            ("--batch-size", spec.batch_size), ("--ubatch-size", spec.ubatch_size),
                            ("--cache-type-k", spec.cache_type_k), ("--cache-type-v", spec.cache_type_v),
                            ("--split-mode", spec.split_mode)):
            if value is not None:
                argv += [flag, str(value)]
        if spec.tensor_split:
            argv += ["--tensor-split", ",".join(f"{v:g}" for v in spec.tensor_split)]
        argv += ["--kv-unified" if spec.kv_unified else "--no-kv-unified", "--flash-attn", spec.flash_attn,
                 "--jinja", "--reasoning-format", spec.reasoning_format, "--no-webui"]
        if spec.draft_model and spec.draft_gguf:
            argv += ["--spec-type", spec.draft_type or "draft-simple",
                     "--spec-draft-model", files.get("draft") or f"<{spec.draft_model}>/{spec.draft_gguf}",
                     "--spec-draft-ngl", str(spec.n_gpu_layers if spec.gpus else 0)]
            if spec.draft_tokens:
                argv += ["--spec-draft-n-max", str(spec.draft_tokens)]
    argv += [str(a) for a in spec.extra_args]
    return argv, env


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------------------------- repositories

_repo_cache: dict[tuple[str, str | None], tuple[float, list[dict]]] = {}
REPO_CACHE_SECONDS = 600
WEIGHT_SUFFIXES = (".safetensors",)
SKIP_DIRS = ("original/", "onnx/", "openvino/", "gguf/", "metal/", "coreml/")
TOKENIZER_FILES = re.compile(r"^(tokenizer.*|.*\.model|special_tokens_map\.json|config\.json|generation_config\.json|"
                             r"chat_template.*|vocab.*|merges\.txt|.*\.tiktoken)$")
QUANT_PREFERENCE = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q4_K_S", "IQ4_XS", "Q4_0", "BF16", "F16")


def offline() -> bool:
    try:
        from huggingface_hub import constants

        return bool(constants.HF_HUB_OFFLINE)
    except ImportError:
        return True


def hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def repo_files(repo: str, revision: str | None = None) -> list[dict]:
    """``[{"name", "size"}]`` of a model repository: from the Hub, or from the local cache when offline or the Hub
    cannot be reached (``LookupError`` when neither has it). Kept for ``REPO_CACHE_SECONDS``."""
    key = (repo, revision)
    hit = _repo_cache.get(key)
    if hit and time.monotonic() - hit[0] < REPO_CACHE_SECONDS:
        return hit[1]
    found: list[dict] | None = None
    if not offline():
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(repo, revision=revision, files_metadata=True, timeout=30)
            found = [{"name": s.rfilename, "size": int(s.size or 0)} for s in info.siblings or ()]
        except Exception:
            found = None
    if found is None:
        found = cached_repo_files(repo, revision)
    if found is None:
        raise LookupError(f"{repo} could not be read from Hugging Face{' (this app is offline)' if offline() else ''}"
                          " and is not in the local cache.")
    _repo_cache[key] = (time.monotonic(), found)
    return found


def cached_repo_files(repo: str, revision: str | None = None) -> list[dict] | None:
    """The files of a cached snapshot of ``repo`` (the revision's, else the newest), or None."""
    folder = hub_cache() / f"models--{repo.replace('/', '--')}"
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return None
    commit = None
    ref = folder / "refs" / (revision or "main")
    if ref.is_file():
        commit = ref.read_text().strip()
    elif revision and (snapshots / revision).is_dir():
        commit = revision
    candidates = [snapshots / commit] if commit else sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime,
                                                            reverse=True)
    for snapshot in candidates:
        if snapshot.is_dir():
            return [{"name": str(p.relative_to(snapshot)), "size": p.stat().st_size}
                    for p in sorted(snapshot.rglob("*")) if p.is_file()]
    return None


def gguf_choices(files: Sequence[Mapping]) -> list[dict]:
    """The model GGUFs of a repo (projectors and later split parts left out), with each one's total size."""
    names = {f["name"]: int(f.get("size") or 0) for f in files}
    choices = []
    for name in sorted(names):
        base = name.rsplit("/", 1)[-1].lower()
        if not base.endswith(".gguf") or base.startswith("mmproj") or re.search(r"-0000[2-9]-of-|-000[1-9]\d-of-",
                                                                                base):
            continue
        parts = split_parts(name, names)
        choices.append({"name": name, "size": sum(names[p] for p in parts), "parts": len(parts)})
    return choices


def split_parts(name: str, names: Mapping[str, int] | Sequence[str]) -> list[str]:
    """A split GGUF's parts (``x-00001-of-00003.gguf`` ...), else the file itself."""
    match = re.match(r"^(.*)-00001-of-(\d{5})\.gguf$", name)
    if not match:
        return [name]
    total = int(match.group(2))
    parts = [f"{match.group(1)}-{i:05d}-of-{match.group(2)}.gguf" for i in range(1, total + 1)]
    return [p for p in parts if p in names]


def choose_gguf(choices: Sequence[Mapping], wanted: str | None) -> str | None:
    """A file name or quantization tag (``Q8_0``) among ``choices``; None for no match. Without ``wanted``: the
    first of ``QUANT_PREFERENCE`` present, else the first file."""
    names = [str(c["name"]) for c in choices]
    if wanted:
        if wanted in names:
            return wanted
        tagged = [n for n in names if wanted.lower() in n.rsplit("/", 1)[-1].lower()]
        return tagged[0] if tagged else None
    for tag in QUANT_PREFERENCE:
        tagged = [n for n in names if tag.lower() in n.lower()]
        if tagged:
            return tagged[0]
    return names[0] if names else None


def vllm_files(files: Sequence[Mapping]) -> list[str]:
    """What vLLM loads from a repo: configs, tokenizer, code, chat template and the weights (safetensors, else
    ``.bin``), not GGUFs, ONNX exports, the original checkpoint, documentation or images."""
    names = [f["name"] for f in files if not f["name"].startswith(SKIP_DIRS)]
    weights = [n for n in names if n.endswith(WEIGHT_SUFFIXES)] or [n for n in names if n.endswith(".bin")]
    keep = {n for n in names if n.endswith((".json", ".py", ".jinja", ".model", ".tiktoken", ".txt"))
            and not n.lower().startswith(("readme", "license"))}
    return sorted(keep | set(weights))


@dataclass
class Download:
    repo: str
    revision: str | None
    files: list[str]
    size: int  # bytes of ``files``
    role: str  # model, draft, tokenizer


def plan_downloads(spec: ServerSpec) -> list[Download]:
    """The repositories and files the server needs (``LookupError``/``ValueError`` when one is missing)."""
    files = repo_files(spec.model, spec.revision)
    sizes = {f["name"]: int(f.get("size") or 0) for f in files}
    plan: list[Download] = []
    if spec.engine == "vllm":
        wanted = vllm_files(files)
        if not any(n.endswith((".safetensors", ".bin")) for n in wanted):
            raise ValueError(f"{spec.model} has no safetensors weights for vLLM (a GGUF repository is for llama.cpp).")
        if spec.reasoning_parser_plugin and spec.reasoning_parser_plugin not in sizes:
            raise ValueError(f"{spec.model} has no {spec.reasoning_parser_plugin}.")
        plan.append(Download(spec.model, spec.revision, wanted, sum(sizes[n] for n in wanted), "model"))
        draft = (spec.speculative_config or {}).get("model")
        if draft:
            draft_files = repo_files(str(draft))
            names = vllm_files(draft_files)
            plan.append(Download(str(draft), None, names,
                                 sum(int(f.get("size") or 0) for f in draft_files if f["name"] in names), "draft"))
    else:
        gguf = choose_gguf(gguf_choices(files), spec.gguf)
        if gguf is None:
            raise ValueError(f"{spec.model} has no GGUF file{' matching ' + spec.gguf if spec.gguf else ''}.")
        parts = split_parts(gguf, sizes)
        plan.append(Download(spec.model, spec.revision, parts, sum(sizes[p] for p in parts), "model"))
        if spec.draft_model and spec.draft_gguf:
            draft_sizes = {f["name"]: int(f.get("size") or 0) for f in repo_files(spec.draft_model)}
            if spec.draft_gguf not in draft_sizes:
                raise ValueError(f"{spec.draft_model} has no {spec.draft_gguf}.")
            plan.append(Download(spec.draft_model, None, [spec.draft_gguf], draft_sizes[spec.draft_gguf], "draft"))
    tokenizer = spec.tokenizer_repo
    if tokenizer != spec.model or spec.engine == "llama_cpp":
        try:
            tokenizer_files = repo_files(tokenizer)
        except LookupError:
            tokenizer_files = []  # the app then counts prompt tokens without it
        names = [f["name"] for f in tokenizer_files if TOKENIZER_FILES.match(f["name"])]
        if names:
            plan.append(Download(tokenizer, None, names,
                                 sum(int(f.get("size") or 0) for f in tokenizer_files if f["name"] in names),
                                 "tokenizer"))
    return plan


def cached_bytes(downloads: Sequence[Download]) -> int:
    """Bytes of the planned files already in the cache, plus partial downloads in their repos' blob folders."""
    from huggingface_hub import try_to_load_from_cache

    done = 0
    for item in downloads:
        for name in item.files:
            path = try_to_load_from_cache(item.repo, name, revision=item.revision)
            if isinstance(path, str) and os.path.exists(path):
                done += os.path.getsize(path)
        blobs = hub_cache() / f"models--{item.repo.replace('/', '--')}" / "blobs"
        if blobs.is_dir():
            done += sum(p.stat().st_size for p in blobs.glob("*.incomplete"))
    return min(done, sum(d.size for d in downloads))


def weights_bytes(spec: ServerSpec, downloads: Sequence[Download] | None = None) -> int:
    """The size of the model's (and drafter's) weights: what must fit on the GPUs before any KV cache."""
    downloads = plan_downloads(spec) if downloads is None else downloads
    return sum(d.size for d in downloads if d.role in ("model", "draft"))


# ---------------------------------------------------------------------------------------------- recipes


def recipes() -> dict:
    """The known models (``presets/local_servers.yaml``)."""
    import yaml

    text = resources.files("matchminer_ai.presets").joinpath("local_servers.yaml").read_text(encoding="utf-8")
    loaded: dict = yaml.safe_load(text)
    return loaded


def find_recipe(model: str) -> tuple[str | None, dict | None]:
    """``(key, recipe)`` of a known model, by its own ID or an engine's repository (a GGUF repo)."""
    known = recipes()["models"]
    for key, entry in known.items():
        repos = {key, *(e.get("model") for e in (entry.get("engines") or {}).values())}
        if model in repos:
            return key, entry
    return None, None


def compute_ok(capability: str | None, allowed: Sequence[str] | None) -> bool:
    """Whether a GPU's compute capability is among ``allowed`` ("12.0"; "12.x" for a major version)."""
    if not allowed:
        return True
    if not capability:
        return False
    major = capability.split(".")[0]
    return any(capability == a or (a.endswith(".x") and a.split(".")[0] == major) for a in allowed)


def propose_engine(model: str, gpus: Sequence, installed: Mapping[str, Mapping]) -> dict:
    """``{"engine", "reason"}``: the engine to propose for ``model`` on ``gpus`` (the chosen ``llm.gpus.GPU``s)."""
    _, recipe = find_recipe(model)
    available = [e for e in ENGINES if installed.get(e, {}).get("available")]
    options = list((recipe or {}).get("engines") or {}) or list(ENGINES)
    reasons = []
    for engine in options:
        limits = ((recipe or {}).get("engines") or {}).get(engine) or {}
        bad = [g for g in gpus if not compute_ok(g.compute_capability, limits.get("compute_capabilities"))]
        if bad:
            g = bad[0]
            reasons.append(f"{limits.get('why') or ENGINES[engine] + ' does not support this GPU'}; "
                           f"{g.name} is SM {g.compute_capability or '?'}")
            continue
        if engine not in available:
            reasons.append(installed.get(engine, {}).get("reason") or f"{ENGINES[engine]} is not installed")
            continue
        other = "; ".join(reasons)
        return {"engine": engine, "reason": (f"{other}, so this uses {ENGINES[engine]}." if other else
                                             f"{ENGINES[engine]} serves this model on the chosen GPUs.")}
    return {"engine": options[0] if options else "llama_cpp",
            "reason": "; ".join(reasons) + "." if reasons else "No engine is installed."}


def suggested_utilization(gpus: Sequence, reserve: int = GB) -> float | None:
    """vLLM's ``gpu_memory_utilization`` for the chosen GPUs: (free - ``reserve``) / total of the tightest one,
    since vLLM counts memory other processes (the app's own models) already hold against its share."""
    values = [(g.memory_free - reserve) / g.memory_total for g in gpus if g.memory_free is not None
              and g.memory_total]
    if not values:
        return None
    return max(0.05, int(min(values) * 100) / 100)


def recommend(model: str, engine: str | None, gpus: Sequence, installed: Mapping[str, Mapping],
              *, with_files: bool = True) -> dict:
    """The form's starting point for ``model`` on ``gpus``: the proposed engine and why, the spec (the recipe's
    settings, else generic ones), the recipe variant used, the repository's GGUF choices and sizes, fit warnings
    and the command preview."""
    key, recipe = find_recipe(model)
    proposal = propose_engine(key or model, gpus, installed)
    if recipe is None and with_files and not engine:  # an unknown repository of GGUF files only: llama.cpp
        try:
            files = repo_files(model)
        except LookupError:
            files = []
        if gguf_choices(files) and not any(f["name"].endswith(WEIGHT_SUFFIXES) for f in files):
            proposal = {"engine": "llama_cpp", "reason": "This repository holds GGUF files, which llama.cpp serves."}
    engine = engine or proposal["engine"]
    entry = ((recipe or {}).get("engines") or {}).get(engine) or {}
    free = sum(g.memory_free or 0 for g in gpus)
    values: dict = {"engine": engine, "model": entry.get("model") or key or model, "gpus": [g.index for g in gpus]}
    if recipe:
        values.update(served_name=recipe.get("served_name") or key, tokenizer=recipe.get("tokenizer") or "")
    values.update(entry.get("spec") or {})
    variant = None
    for option in entry.get("variants") or ():  # the first whose memory fits
        variant = option
        if free >= option.get("memory_gb", 0) * GB:
            break
    if variant:
        values.update(variant.get("spec") or {})
    if engine == "llama_cpp" and recipe is None:
        values.update(parallel=4, kv_unified=True)
    if engine == "vllm":
        if values.get("gpu_memory_utilization") is None:
            values["gpu_memory_utilization"] = suggested_utilization(gpus)
    spec = ServerSpec.from_dict(values)
    result: dict = {"known": recipe is not None, "key": key, "label": (recipe or {}).get("label") or model,
                    "reasoning": bool((recipe or {}).get("reasoning", False)), "proposal": proposal,
                    "engine": engine, "spec": spec.to_dict(), "variant": (variant or {}).get("name"),
                    "variants": [{"name": v.get("name"), "memory_gb": v.get("memory_gb"), "note": v.get("note")}
                                 for v in entry.get("variants") or ()],
                    "note": entry.get("note") or (recipe or {}).get("note"), "source": (recipe or {}).get("source"),
                    "files": [], "gguf_choices": [], "warnings": [], "error": None}
    if with_files:
        try:
            files = repo_files(spec.model, spec.revision)
            result["files"] = files
            result["gguf_choices"] = gguf_choices(files)
            if engine == "llama_cpp" and not spec.gguf:
                spec.gguf = choose_gguf(result["gguf_choices"], None)
                result["spec"] = spec.to_dict()
            result["warnings"] = fit_warnings(spec, gpus, plan_downloads(spec))
        except (LookupError, ValueError) as exc:
            result["error"] = str(exc)
    command = build_command(spec, 0, gpus=gpus, engine_command=(installed.get(engine) or {}).get("command"))[0]
    result["command"] = shlex.join(command).replace("--port 0", "--port <free port>")
    return result


def fit_warnings(spec: ServerSpec, gpus: Sequence, downloads: Sequence[Download] | None = None) -> list[str]:
    """Warnings (never refusals) about whether the spec fits: weights against the chosen GPUs' free memory, vLLM's
    share against each GPU's free memory, the recipe's hardware limits and the cache's free disk."""
    warnings = []
    chosen = [g for g in gpus if g.index in spec.gpus]
    _, recipe = find_recipe(spec.model)
    limits = ((recipe or {}).get("engines") or {}).get(spec.engine) or {}
    for g in chosen:
        if not compute_ok(g.compute_capability, limits.get("compute_capabilities")):
            warnings.append(f"{limits.get('why') or 'This engine does not support this GPU'}; GPU {g.index} "
                            f"({g.name}) is SM {g.compute_capability or '?'}.")
    free = sum(g.memory_free or 0 for g in chosen)
    if downloads is not None:
        weights = sum(d.size for d in downloads if d.role in ("model", "draft"))
        if chosen and all(g.memory_free is not None for g in chosen) and weights > free:
            warnings.append(f"The weights ({weights / GB:.1f} GB) are more than the chosen GPUs' free memory "
                            f"({free / GB:.1f} GB).")
        try:
            needed = sum(d.size for d in downloads) - cached_bytes(downloads)
            disk = shutil.disk_usage(hub_cache() if hub_cache().exists() else Path.home()).free
            if needed > disk:
                warnings.append(f"The download ({needed / GB:.1f} GB) is more than the Hugging Face cache's free "
                                f"disk ({disk / GB:.1f} GB).")
            elif needed > 0 and offline():
                warnings.append(f"{needed / GB:.1f} GB is not in the Hugging Face cache, and this app does not "
                                "download (start it with --allow-downloads).")
        except Exception:
            pass
    if spec.engine == "vllm" and spec.gpu_memory_utilization:
        for g in chosen:
            if g.memory_free is not None and spec.gpu_memory_utilization * g.memory_total > g.memory_free:
                suggestion = suggested_utilization([g])
                warnings.append(f"vLLM would take {spec.gpu_memory_utilization * g.memory_total / GB:.1f} GB of GPU "
                                f"{g.index}, which has {g.memory_free / GB:.1f} GB free; suggested "
                                f"gpu_memory_utilization: {suggestion}.")
    if not chosen and spec.engine == "llama_cpp":
        warnings.append("No GPU is chosen: llama.cpp runs on the CPU, slowly.")
    return warnings


# ---------------------------------------------------------------------------------------------- the process

ACTIVE = frozenset({"downloading", "starting", "ready", "stopping"})
# (pattern, phase): the last log line matching one gives the phase while the server starts. Only the engines' progress
# lines, not their configuration dumps (vLLM's names kv_cache_dtype and cudagraph_mode). vLLM 0.30 captures CUDA
# graphs twice: once to profile their memory, then after allocating the KV cache.
PHASES = (
    (re.compile(r"Starting to load model|Loading weights|Loading safetensors|load_tensors:|llama_model_load"
                r"|main: loading model"), "Loading weights"),
    (re.compile(r"Compiling a graph|torch\.compile took"), "Compiling (torch.compile)"),
    (re.compile(r"Capturing CUDA graphs|Graph capturing"), "Capturing CUDA graphs"),
    (re.compile(r"Available KV cache memory|GPU KV cache size|llama_kv_cache|llama_context:"),
     "Allocating the KV cache"),
    (re.compile(r"Application startup complete|server is listening|Uvicorn running|all slots are idle"),
     "Starting the API"),
)


class ManagedServer:
    """One engine process and its lifecycle: download, launch, readiness, exit and stop.

    ``start()`` returns at once; a monitor thread does the work. ``state``: downloading, starting, ready, stopping,
    stopped, exited (the process ended on its own; ``exit_code``) or failed (``error``). The monitor thread calls
    ``on_ready(server)`` just before the state becomes ready and ``on_end(server)`` once it has ended."""

    def __init__(self, spec: ServerSpec, log_dir: Path, *, port: int | None = None,
                 engine_command: Sequence[str] | None = None, gpus: Sequence | None = None,
                 ready_timeout: float = 4 * 3600, poll_seconds: float = 2.0, download: bool = True,
                 on_ready: Callable[["ManagedServer"], None] | None = None,
                 on_end: Callable[["ManagedServer"], None] | None = None,
                 environment: Mapping[str, str] | None = None):
        spec.validate()
        self.spec, self.port = spec, port or free_port()
        self.engine_command = list(engine_command) if engine_command else None
        self.gpus = list(gpus) if gpus is not None else None
        self.ready_timeout, self.poll_seconds, self.download = ready_timeout, poll_seconds, download
        self.on_ready, self.on_end = on_ready, on_end
        self.environment = dict(os.environ if environment is None else environment)
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        self.log_path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.started_at))}-{self.port}.log"
        self.state, self.phase, self.error, self.exit_code = "starting", "Preparing", None, None
        self.ready_at: float | None = None
        self.ended_at: float | None = None
        self.progress: dict | None = None
        self.command: list[str] | None = None
        self.process: subprocess.Popen | None = None
        self.helper: subprocess.Popen | None = None  # the download process
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None
        self._end_called = False

    # ------------------------------------------------------------------ public

    @property
    def endpoint(self) -> str:
        return f"http://{HOST}:{self.port}/v1"

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    def start(self) -> "ManagedServer":
        self.thread = threading.Thread(target=self._run, name="local-model-server", daemon=True)
        self.thread.start()
        return self

    def stop(self, grace: float = STOP_GRACE_SECONDS) -> None:
        """Stop the download or the engine's whole process group (SIGTERM, then SIGKILL after ``grace`` s)."""
        with self.lock:
            if not self.active:
                return
            self.stopping.set()
            self.state, self.phase = "stopping", "Stopping"
        for process in (self.helper, self.process):
            stop_group(process, grace)
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=grace + 15)
        with self.lock:
            if self.state == "stopping":
                self.state, self.phase, self.ended_at = "stopped", "Stopped", time.time()
        self._ended()

    def wait(self, states: Sequence[str] = ("ready", "stopped", "exited", "failed"), timeout: float = 60) -> str:
        deadline = time.monotonic() + timeout
        while self.state not in states and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.state

    def log_tail(self, lines: int = 200) -> list[str]:
        try:
            with self.log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 256 * 1024))
                text = handle.read().decode("utf-8", "replace")
        except OSError:
            return []
        rows = text.replace("\r", "\n").splitlines()
        return [r[:2000] for r in rows if r.strip()][-lines:]

    def describe(self, log_lines: int = 200) -> dict:
        end = self.ended_at or time.time()
        return {"spec": self.spec.to_dict(), "engine": self.spec.engine, "state": self.state, "phase": self.phase,
                "error": self.error, "exit_code": self.exit_code, "progress": self.progress,
                "endpoint": self.endpoint, "port": self.port, "model": self.spec.name,
                "tokenizer": self.spec.tokenizer_repo, "pid": self.process.pid if self.process else None,
                "command": shlex.join(self.command) if self.command else None, "started_at": self.started_at,
                "ready_at": self.ready_at, "elapsed_seconds": round(end - self.started_at, 1),
                "log_path": str(self.log_path), "log": self.log_tail(log_lines) if log_lines else []}

    # ------------------------------------------------------------------ the monitor thread

    def _log(self, text: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip("\n") + "\n")

    def _set(self, **values) -> bool:
        """Update the state unless a stop has begun (False then)."""
        with self.lock:
            if self.stopping.is_set():
                return False
            for key, value in values.items():
                setattr(self, key, value)
            return True

    def _run(self) -> None:
        try:
            files = self._download() if self.download else {}
            if files is None:
                return
            argv, extra = build_command(self.spec, self.port, files=files, engine_command=self.engine_command,
                                        gpus=self.gpus, parent_visible=self.environment.get("CUDA_VISIBLE_DEVICES"))
            env = cuda_environment(argv[0], self.environment) if self.spec.engine == "vllm" else dict(self.environment)
            env.update(extra)
            if self.download:  # every file is in the cache now: the engine never reaches the Hub
                env["HF_HUB_OFFLINE"] = "1"
            self.command = argv
            self._log(f"$ {' '.join(f'{k}={shlex.quote(v)}' for k, v in sorted(extra.items()))} {shlex.join(argv)}")
            with self.lock:
                if self.stopping.is_set():
                    return
                with self.log_path.open("ab") as handle:
                    self.process = subprocess.Popen(with_parent_watch(argv), env=env, stdout=handle,
                                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                    start_new_session=True)
                self.state, self.phase = "starting", "Starting"
            self._follow()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:2000]
            if not self.stopping.is_set():  # the log first: a reader that sees the state finds the reason
                self._log(f"[local server] failed: {error}")
            self._set(state="failed", phase="Failed", error=error, ended_at=time.time())
            stop_group(self.process, STOP_GRACE_SECONDS)
        finally:
            if self.state in ("exited", "failed"):
                self._ended()

    def _ended(self) -> None:
        """``on_end``, once."""
        with self.lock:
            if self._end_called or self.on_end is None:
                return
            self._end_called = True
        self._callback(self.on_end)

    def _callback(self, function) -> None:
        try:
            function(self)
        except Exception as exc:
            self._log(f"[local server] callback failed: {type(exc).__name__}")

    def _follow(self) -> None:
        deadline = time.monotonic() + self.ready_timeout
        assert self.process is not None
        while True:
            code = self.process.poll()
            if code is not None:
                if not self.stopping.is_set():
                    self._log(f"[local server] the engine exited with code {code}")
                self._set(state="exited", phase=f"Exited with code {code}", exit_code=code, ended_at=time.time())
                return
            if self.stopping.is_set():
                return
            if self.state == "starting":
                if self._ready():
                    self._log(f"[local server] ready at {self.endpoint} as {self.spec.name}")
                    if self.on_ready and not self.stopping.is_set():
                        self._callback(self.on_ready)  # before the state: whoever sees "ready" sees its effects
                    self._set(state="ready", phase="Ready", ready_at=time.time())
                    continue
                phase = self._phase()
                if phase:
                    self._set(phase=phase)
                if time.monotonic() > deadline:
                    raise TimeoutError(f"the server was not ready within {self.ready_timeout / 60:.0f} minutes")
            self.stopping.wait(self.poll_seconds if self.state == "starting" else 1.0)

    def _ready(self) -> bool:
        try:  # loopback: never through a proxy
            opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
            with opener.open(urlrequest.Request(self.endpoint + "/models"), timeout=5) as response:
                data = json.load(response).get("data") or []
            return self.spec.name in {str(m.get("id")) for m in data if isinstance(m, dict)}
        except (OSError, urlerror.URLError, ValueError, AttributeError):
            return False

    def _phase(self) -> str | None:
        phase = None
        for line in self.log_tail(400):
            for pattern, name in PHASES:
                if pattern.search(line):
                    phase = name
        return phase

    def _download(self) -> dict | None:
        """Download what the engine needs; the files it names on its command line (None once stopped)."""
        if not self._set(state="downloading", phase="Checking the model's files"):
            return None
        plan = plan_downloads(self.spec)
        total = sum(d.size for d in plan)
        done = cached_bytes(plan)
        cache = hub_cache()
        cache.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(cache).free
        if total - done > free:
            raise OSError(f"the download needs {(total - done) / GB:.1f} GB and the Hugging Face cache has "
                          f"{free / GB:.1f} GB free")
        self._set(progress={"done": done, "total": total},
                  phase="Downloading" if done < total else "Checking the cached files")
        result = self.log_path.with_suffix(".download.json")
        request = [{"repo": d.repo, "revision": d.revision, "files": d.files} for d in plan]
        env = {**self.environment, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        self._log(f"[local server] downloading {len(plan)} repositories ({total / GB:.2f} GB; "
                  f"{done / GB:.2f} GB cached)")
        with self.lock:
            if self.stopping.is_set():
                return None
            with self.log_path.open("ab") as handle:
                self.helper = subprocess.Popen([sys.executable, "-m", "matchminer_ai.llm.local_server", "download",
                                                json.dumps(request), str(result)], env=env, stdout=handle,
                                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                               start_new_session=True)
        while self.helper.poll() is None:
            if self.stopping.wait(1.0):
                return None
            self._set(progress={"done": cached_bytes(plan), "total": total})
        if self.stopping.is_set():
            return None
        if self.helper.returncode != 0 or not result.is_file():
            offline_hint = " (this app does not download: start it with --allow-downloads)" if offline() else ""
            raise RuntimeError(f"the download failed{offline_hint}; see the log")
        folders = json.loads(result.read_text())
        result.unlink(missing_ok=True)
        self._set(progress={"done": total, "total": total})
        files: dict[str, str] = {}
        for index, item in enumerate(plan):
            folder = Path(folders[str(index)])
            if item.role == "model" and self.spec.engine == "llama_cpp":
                files["gguf"] = str(folder / item.files[0])
            elif item.role == "draft" and self.spec.engine == "llama_cpp":
                files["draft"] = str(folder / item.files[0])
            elif item.role == "model" and self.spec.reasoning_parser_plugin:
                files["reasoning_parser_plugin"] = str(folder / self.spec.reasoning_parser_plugin)
        return files


# Linux: the engine gets SIGTERM if the thread that started it dies (the app killed or crashed), set in the child
# before it execs the engine (prctl PR_SET_PDEATHSIG), so not even a killed app leaves a server on the GPU.
_PARENT_WATCH = (
    "import ctypes, os, signal, sys\n"
    "ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)\n"
    "if os.getppid() != int(sys.argv[1]):\n"
    "    sys.exit(1)\n"
    "os.execvp(sys.argv[2], sys.argv[2:])\n"
)


def with_parent_watch(argv: Sequence[str]) -> list[str]:
    """``argv`` started through a small Python exec wrapper that sets the parent-death signal (Linux only)."""
    if not sys.platform.startswith("linux"):
        return list(argv)
    return [sys.executable, "-c", _PARENT_WATCH, str(os.getpid()), *argv]


def stop_group(process: subprocess.Popen | None, grace: float = STOP_GRACE_SECONDS) -> None:
    """SIGTERM to ``process``'s process group, then SIGKILL after ``grace`` seconds (its engine workers too)."""
    if process is None or process.poll() is not None:
        if process is not None:
            _kill_group(process.pid, signal.SIGKILL)  # a leader gone with workers left behind
        return
    _kill_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    _kill_group(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def _kill_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def group_alive(pgid: int) -> bool:
    """Whether any process of the group is left."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


# ---------------------------------------------------------------------------------------------- the download helper


def _download_main(request: str, result: str) -> int:
    from huggingface_hub import snapshot_download

    folders = {}
    for index, item in enumerate(json.loads(request)):
        print(f"[local server] {item['repo']}: {len(item['files'])} files", flush=True)
        folders[str(index)] = snapshot_download(item["repo"], revision=item.get("revision"),
                                                allow_patterns=item["files"])
    Path(result).write_text(json.dumps(folders))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 3 and args[0] == "download":
        try:
            return _download_main(args[1], args[2])
        except Exception as exc:  # the parent reports it; the type and message go to the server's log
            print(f"[local server] download failed: {type(exc).__name__}: {exc}", flush=True)
            return 1
    print("usage: python -m matchminer_ai.llm.local_server download <request-json> <result-path>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ENGINES", "ManagedServer", "ServerSpec", "build_command", "cuda_environment", "engines", "fit_warnings",
    "plan_downloads", "propose_engine", "recipes", "recommend", "repo_files", "suggested_utilization",
    "visible_devices",
]
