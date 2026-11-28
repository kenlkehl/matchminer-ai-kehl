"""Where the app reads its data and models, where it keeps its store, and how it serves, by mode.

The app starts in one of two modes (``--mode``, on the command line only):

- **real:** the OA trials being extracted in ``trial_data`` with the OA DuckDB copy's trial table
  (``pipeline.trial_table``), and real patients (the population cache; guideline populations from the OA database).
  It serves on loopback only and always asks for the access token. Its store, vectors and caches hold confidential
  protocol-derived text and patient data, so they live beside the trial data under the data root's ``phi/`` side
  (owner only), never in a ``no_phi`` folder.
- **synthetic:** the synthetic run's trials and patients only. It may serve beyond loopback (``0.0.0.0`` by
  default, any Host name unless ``--allowed-hosts`` names them) and asks for a token only with
  ``--require-token``. It never reads or writes a ``phi`` folder.

The TrialSpace rounds, the encoder and the decision model come from the synthetic n=5000 run in both modes
(``core.settings``): the run's final decision model, else the published MatchEval-1226 (only a final model scores;
user, 2026-11-21). The model runs on ``cuda:0`` when CUDA is available. Each trial's drugs and mechanisms come from the GoodOption catalog
(``good_options_catalog``), read in place.

A mode's own data paths come from the site config's ``dashboard.<mode>`` section, shared values from its
``dashboard`` keys, else folders under the data root (``matchminer_ai.site``); ``--site-config`` and ``--data-root`` name them.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import re

from matchminer_ai.apps.common.security import LOOPBACK_HOSTS
from matchminer_ai.site import (DATA_ROOT_ENV, SITE_CONFIG_ENV, allow_hub_downloads, data_root, is_hub_ref,
                               load_site_config)

from .core.settings import Settings
from .llm import NEMOTRON, default_backend, default_endpoint, default_model

# The bundled trial-group rules: a small example. A site's own rules: the site config's dashboard.grouping.
DEFAULT_GROUPING = Path(__file__).with_name("grouping.yaml")


MODES = ("real", "synthetic")
FRAME_ORIGIN = re.compile(r"^https://(?:\*\.)?[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?::\d{1,5})?$")
# Where a mode's own folders sit under the data root when the site config names none.
MODE_FOLDERS = {"real": Path("phi") / "real_time", "synthetic": Path("no_phi")}
MODE_PORTS = {"real": 8781, "synthetic": 8034}
ANY_HOST = "0.0.0.0"


def site_default(mode: str, key: str, relative: str | None, *, writable: bool = False) -> Path | None:
    """A mode's own data path: the site config's ``dashboard.<mode>.<key>``, else ``relative`` under the mode's
    folder of the data root (None when the mode has no default). Shared values (``dashboard.grouping``,
    ``dashboard.synthetic_run``, the branding and trial table) stay directly under ``dashboard``. A Hub reference
    (``hf://``, read-only) is refused where the app writes (``writable``)."""
    value = load_site_config().get(f"dashboard.{mode}.{key}", resolve=not writable)
    if writable and is_hub_ref(value):
        raise ValueError(f"dashboard.{mode}.{key} is written by the app: name a local folder, not {value}")
    if value:
        return Path(value).expanduser()
    return data_root() / MODE_FOLDERS[mode] / relative if relative else None


def mode_side_error(mode: str, path) -> str | None:
    """Why ``path`` is on the other mode's side of the data root (None when it is not). Real-mode data and caches
    never sit in a ``no_phi`` folder, and synthetic mode never reads or writes a ``phi`` folder."""
    parts = Path(path).expanduser().resolve().parts
    if mode == "real" and "no_phi" in parts:
        return "Real mode keeps patient and protocol data out of no_phi folders."
    if mode == "synthetic" and "phi" in parts:
        return "Synthetic mode never uses a phi folder."
    return None


def default_grouping() -> Path:
    value = load_site_config().get("dashboard.grouping")
    return Path(value).expanduser() if value else DEFAULT_GROUPING
ENV_PREFIX = "MMAI_DASH_"
ACCESS_TOKEN_ENV = "MMAI_DASH_ACCESS_TOKEN"
ACCESS_TOKEN_FILE = "~/.config/mmai-dashboard/access_token"
DEFAULT_DEVICE = "cuda:0"  # the decision model's default when CUDA is available (user, 2026-11-12); else cpu
AUTO_DEVICE = "auto"


@dataclass
class DashSettings(Settings):
    # "real" or "synthetic" (see the module docstring); the command line requires it.
    mode: str = "real"
    host: str | None = None  # default: 127.0.0.1 (real), 0.0.0.0 (synthetic)
    port: int | None = None  # default: 8781 (real), 8034 (synthetic)
    cache_dir: Path | None = None  # default: <store_dir>/cache
    # Real mode: the OA trials' extraction runs and the newest OA snapshot (a glob).
    trial_data: Path | None = None
    oa_db: Path | None = None
    store_dir: Path | None = None
    grouping: Path = field(default_factory=default_grouping)
    # Trials' drugs (public trial and drug information only, built from NCT IDs and drug names, read in place);
    # a missing catalog: no drugs.
    good_options_catalog: Path | None = None
    # Slots, edited criteria, subcohorts and group pins (trial_management): one file shared by the mode's users.
    trial_management: Path | None = None
    # The default model connection of every LLM task (each browser session can change it per task on Data and Models):
    # Claude Haiku 5.5 through the Databricks gateway when the site config names it (user, 2026-11-21), else the
    # Vertex profile Gemini 4 Hydrogen; with llm_backend "openai", an OpenAI-compatible server (llm_endpoint, default
    # the site config's endpoints.teacher.url).
    llm_backend: str = field(default_factory=default_backend)
    llm_endpoint: str = field(default_factory=default_endpoint)
    llm_model: str = field(default_factory=default_model)
    llm_tokenizer: str = ""
    rebuild_poll_seconds: float = 30  # how often new extraction results are looked for; 0: never
    auto_score_n: int = 10  # pairs the result pages score with the decision model as they open
    default_k: int = 50
    device: str = AUTO_DEVICE  # the decision model: cuda:0 when CUDA is available, else cpu (device_notice says so)
    # Host headers accepted. Default: loopback names (real mode; a forwarded port's Host is localhost or
    # 127.0.0.1), any (synthetic mode).
    allowed_hosts: tuple[str, ...] | None = None
    # Require an access token (MMAI_DASH_ACCESS_TOKEN, else a private token file): always in real mode, to keep out
    # other accounts on this machine, which can reach a loopback port; on request in synthetic mode.
    require_token: bool = False
    device_notice: str | None = None
    # The local model server (Data and models; ``local_server``), in every dashboard: anyone using the app may start,
    # stop and configure it (user, 2026-11-24).
    # A server spec (JSON, or a JSON/YAML file) started at boot, after the decision model has loaded.
    local_server_autostart: str | None = None
    # Once ready, the server becomes the default connection of every task in sessions that chose none.
    local_server_default: bool = False
    # Every LLM task uses this app's own local model server, and no other connection is accepted (no gateway, Vertex
    # or other OpenAI-compatible endpoint); implies local_server_default.
    local_server_only: bool = False
    # Synthetic mode, with local_server_only: also accept OpenRouter connections with a browser session's own key (a
    # Space whose local server is loading or which has no GPU). Without local_server_only, synthetic mode always
    # offers OpenRouter; real mode never does.
    allow_openrouter: bool = False
    # False on a machine that cannot host an LLM (no GPU, or one too small): no local model server is offered, and
    # no autostart runs. With local_server_only and allow_openrouter, OpenRouter with a session's own key is then the
    # only connection and every task's default (``openrouter_only``).
    local_server_enabled: bool = True
    # The llama-server binary (default: $MMAI_LLAMA_SERVER, else llama-server on PATH).
    llama_server: str | None = None
    # Synthetic mode only: another trial set than the synthetic run's (site config dashboard.synthetic.trial_cohorts
    # and .trial_tables; ``synthetic_trials``): its cohorts (corpus JSON lines or a cohort table) and trial tables.
    trial_cohorts: Path | None = None
    trial_tables: tuple[Path, ...] = ()
    # Synthetic mode only: origins whose pages may show the app in a frame (CSP frame-ancestors), e.g. a Hugging Face
    # Space page (https://huggingface.co). Its session cookie is then a partitioned third-party cookie (SameSite=None,
    # Secure), so a framed app keeps its session. Default: no framing.
    frame_ancestors: tuple[str, ...] = ()
    # Synthetic mode only: the hosted OpenRouter connection (site config dashboard.synthetic.hosted_openrouter;
    # ``hosted_openrouter``): the app owner's key from MMAI_OPENROUTER_HOSTED_KEY serves sessions without their own,
    # for one model, within its limits. Offered only when the secret is set too. None: no hosted connection.
    hosted_openrouter: dict | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {self.mode!r}")
        real = self.mode == "real"
        if self.host is None:
            self.host = "127.0.0.1" if real else ANY_HOST
        if self.port is None:
            self.port = MODE_PORTS[self.mode]
        if self.allowed_hosts is None:
            self.allowed_hosts = tuple(LOOPBACK_HOSTS) if real else ("*",)
        if real:
            self.require_token = True
        self.frame_ancestors = tuple(self.frame_ancestors or ())
        if self.frame_ancestors and real:
            raise ValueError("Real mode is never shown in another site's frame (frame ancestors: synthetic mode only).")
        for origin in self.frame_ancestors:
            if not FRAME_ORIGIN.match(origin):
                raise ValueError(f"A frame ancestor is an https origin such as https://huggingface.co, not {origin!r}")
        if not self.local_server_enabled:
            self.local_server_autostart, self.local_server_default = None, False
            if self.local_server_only and not self.allow_openrouter:
                raise ValueError("With no local model server, --local-server-only needs --allow-openrouter: no other "
                                 "connection would be accepted.")
        elif self.local_server_only:
            self.local_server_default = True
        if self.allow_openrouter and real:
            raise ValueError("OpenRouter is for synthetic mode only (allow_openrouter); real mode never offers it.")
        if self.hosted_openrouter is not None:
            from .hosted_openrouter import REAL, normalize

            if real:
                raise ValueError(REAL)
            self.hosted_openrouter = normalize(self.hosted_openrouter)
        if self.llm_backend == "openrouter":
            raise ValueError("OpenRouter needs each browser session's own key, so it cannot be the default connection; "
                             "choose it on Data and models.")
        if self.device in (None, "", AUTO_DEVICE):
            self.device, self.device_notice = default_device()
        if self.trial_data is None and real:
            self.trial_data = site_default("real", "trial_data", "trial_data")
        if self.oa_db is None and real:
            self.oa_db = site_default("real", "oa_db", "oa_db_*.duckdb")
        if self.store_dir is None:
            self.store_dir = site_default(self.mode, "store_dir", "dashboard" if real else "dashboard_synthetic",
                                          writable=True)
        if self.good_options_catalog is None:
            self.good_options_catalog = site_default(self.mode, "good_options_catalog",
                                                     "good_options_catalog/catalog" if real else None)
        if not real:
            site = load_site_config()
            if self.trial_cohorts is None and site.get("dashboard.synthetic.trial_cohorts"):
                self.trial_cohorts = Path(site.get("dashboard.synthetic.trial_cohorts")).expanduser()
            if not self.trial_tables:
                tables = site.get("dashboard.synthetic.trial_tables") or ()
                self.trial_tables = tuple(Path(t).expanduser() for t in ([tables] if isinstance(tables, str)
                                                                          else tables))
        if self.trial_management is None:
            # Beside the store, not in it: the store folder is rebuildable, the users' edits are not.
            self.trial_management = site_default(self.mode, "trial_management",
                                                  "dashboard_trial_management/trial_management.json", writable=True)
        self.trial_management = Path(self.trial_management).expanduser()
        self.trial_data = Path(self.trial_data).expanduser() if self.trial_data else None
        self.oa_db = Path(self.oa_db).expanduser() if self.oa_db else None
        self.store_dir = Path(self.store_dir).expanduser()
        self.grouping = Path(self.grouping).expanduser()
        self.good_options_catalog = Path(self.good_options_catalog).expanduser() if self.good_options_catalog else None
        if self.cache_dir is None:
            self.cache_dir = self.store_dir / "cache"
        super().__post_init__()
        for name in ("store_dir", "cache_dir", "trial_management"):
            error = mode_side_error(self.mode, getattr(self, name))
            if error:
                raise ValueError(f"{error} Choose another --{name.replace('_', '-')}.")
        if not real:
            for name in ("trial_data", "oa_db", "good_options_catalog", "trial_cohorts"):
                value = getattr(self, name)
                if value is not None and mode_side_error(self.mode, value):
                    raise ValueError(f"Synthetic mode never reads a phi folder: {name}.")

    @property
    def real(self) -> bool:
        return self.mode == "real"

    @property
    def openrouter_only(self) -> bool:
        """OpenRouter is the only LLM connection, with a session's own key or the hosted one: no local server here, and
        nothing else accepted (--local-server-only with --allow-openrouter)."""
        return self.local_server_only and self.allow_openrouter and not self.local_server_enabled

    @property
    def store_path(self) -> Path:
        return self.store_dir / "dashboard.json"

def guideline_settings(dash: "DashSettings", decision_model: dict | None = None, *, hosted_model: str | None = None):
    """The guideline features' settings (``guidelines.settings.Settings``) for the app: real mode reads real
    patients; the default model connection is the app's (``--llm-*``); TrialSpace and the decision model run on
    the app's device with the app's decision model (``decision_model``: its preset section).

    With ``--local-server-only`` there is no default connection until the local model server is ready (an empty
    endpoint, which LLM tasks refuse), whatever the ``--llm-*`` options and the site config name. With OpenRouter as
    the only connection (``openrouter_only``) the default is the hosted connection's model (``hosted_model``, served
    by the hosted key to sessions without their own) or else OpenRouter's default model, with reasoning off, which
    each session's own key then serves (until a session pastes one, LLM tasks say to)."""
    from .guidelines.settings import LOCAL, OPENROUTER, Settings as GuidelineSettings, default_task_connections

    if dash.openrouter_only:
        from matchminer_ai.llm import openrouter

        model = hosted_model or openrouter.presets()["default"]
        return GuidelineSettings(phi=dash.real, llm_backend=OPENROUTER, endpoint=openrouter.ENDPOINT, model=model,
                                 tokenizer=openrouter.model_settings(model)["tokenizer"], reasoning=False,
                                 device=dash.device, decision_model=dict(decision_model or {}), task_connections={})
    if dash.local_server_only:
        return GuidelineSettings(phi=dash.real, llm_backend=LOCAL, endpoint="", model="", tokenizer="",
                                 device=dash.device, decision_model=dict(decision_model or {}), task_connections={})
    backend = LOCAL if dash.llm_backend == "openai" else dash.llm_backend
    return GuidelineSettings(phi=dash.real, llm_backend=backend, endpoint=dash.llm_endpoint or "http://localhost:8000/v1",
                             model=dash.llm_model, tokenizer=dash.llm_tokenizer or dash.llm_model, device=dash.device,
                             decision_model=dict(decision_model or {}), task_connections=default_task_connections())


def _flag(name: str) -> bool:
    return (_env(name) or "").lower() in ("1", "true", "yes")


def local_server_settings(args, mode: str) -> dict:
    """The local model server's settings: the command line, then MMAI_DASH_* variables (LOCAL_SERVER_ONLY,
    LOCAL_SERVER_AUTOSTART, LOCAL_SERVER_DEFAULT, LLAMA_SERVER, ALLOW_OPENROUTER, LOCAL_SERVER_ENABLED), then the site
    config's ``dashboard.<mode>.local_server`` (``only``, ``autostart``, ``default``, ``llama_server``,
    ``allow_openrouter``, ``enabled``)."""
    import json

    site = load_site_config().get(f"dashboard.{mode}.local_server") or {}
    if not isinstance(site, dict):
        site = {}
    autostart = args.local_server_autostart or _env("local_server_autostart") or site.get("autostart")
    if isinstance(autostart, dict):
        autostart = json.dumps(autostart)
    only = bool(args.local_server_only or _flag("local_server_only") or site.get("only"))
    return {"local_server_only": only, "local_server_autostart": autostart or None,
            "local_server_default": bool(only or args.local_server_default or _flag("local_server_default")
                                         or site.get("default")),
            "llama_server": args.llama_server or _env("llama_server") or site.get("llama_server") or None,
            "allow_openrouter": bool(getattr(args, "allow_openrouter", False) or _flag("allow_openrouter")
                                     or site.get("allow_openrouter")),
            "local_server_enabled": not (getattr(args, "no_local_server", False)
                                         or (_env("local_server_enabled") or "").lower() in ("0", "false", "no")
                                         or site.get("enabled") is False)}


def default_device() -> tuple[str, str | None]:
    """(device, notice): cuda:0 when CUDA is available, else cpu and why."""
    try:
        available = cuda_available()
    except Exception as exc:  # a broken torch or driver: the model still runs on CPU
        return "cpu", (f"Could not check for CUDA ({type(exc).__name__}): the decision model runs on cpu, "
                       f"not {DEFAULT_DEVICE}")
    if available:
        return DEFAULT_DEVICE, None
    return "cpu", f"CUDA is not available: the decision model runs on cpu, not {DEFAULT_DEVICE}"


def cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _env(name: str):
    return os.environ.get(ENV_PREFIX + name.upper())


def parse_settings(argv: list[str] | None = None) -> DashSettings:
    """Settings from the command line, then MMAI_DASH_* variables, then the defaults."""
    parser = argparse.ArgumentParser(prog="mmai-dashboard",
                                     description="MatchMiner-AI dashboard: trials, populations, patients and "
                                                 "benchmarks (research use)")
    parser.add_argument("command", nargs="?", default="serve", choices=("serve", "build", "prepare"),
                        help="serve (default), build the dashboard store once and exit, or prepare the synthetic run "
                             "from its published dataset (a site config naming hf://datasets/...) and exit")
    parser.add_argument("--mode", required=True, choices=MODES,
                        help="real: the OA trials and real patients, loopback only, with sign-in; synthetic: the "
                             "synthetic run's trials and patients only, may serve beyond loopback")
    parser.add_argument("--site-config", help=f"site config YAML (default ${SITE_CONFIG_ENV})")
    parser.add_argument("--data-root", help=f"data root holding phi/ and no_phi/ (default ${DATA_ROOT_ENV} or the "
                        "site config's paths.data_root)")
    parser.add_argument("--trial-data", type=Path, help="process_oa_trials.py output (default: site config "
                        "dashboard.trial_data, else phi/real_time/trial_data under the data root)")
    parser.add_argument("--oa-db", type=Path, help="OA DuckDB copy, or a glob for its newest (default: dashboard.oa_db, "
                        "else phi/real_time/oa_db_*.duckdb)")
    parser.add_argument("--store-dir", type=Path, help="dashboard store, vectors and caches (default: "
                        "dashboard.store_dir, else phi/real_time/dashboard)")
    parser.add_argument("--grouping", type=Path, help="trial-group rules (default: the site config's dashboard.grouping, else the bundled rules)")
    parser.add_argument("--good-options-catalog", type=Path, help="GoodOption catalog directory with the trials' drugs "
                        "(default: dashboard.good_options_catalog; missing: no drugs)")
    parser.add_argument("--trial-management", type=Path, help="the trial management file: slots, edited criteria, "
                        "subcohorts and group pins (default: dashboard.<mode>.trial_management, else "
                        "dashboard_trial_management/trial_management.json under the mode's folder of the data root)")
    parser.add_argument("--rebuild-poll-seconds", type=float, help="how often to look for new extraction results (default 30)")
    parser.add_argument("--llm-backend", help="the default model connection: claude_databricks (default when the site "
                        "config names the Databricks gateway and a Haiku endpoint), a Vertex profile (gemini4_vertex, the "
                        "default otherwise; also "
                        "gemini_vertex, gemma_vertex) or openai (an OpenAI-compatible URL, --llm-endpoint)")
    parser.add_argument("--llm-endpoint", help="with --llm-backend openai: the endpoint (default: the site config's "
                        "endpoints.teacher.url; none: the LLM checks are off)")
    parser.add_argument("--llm-model", help=f"the model: a gateway endpoint name for Claude (default: the site's "
                        f"databricks.endpoints.haiku), else the endpoint's model (default {NEMOTRON})")
    parser.add_argument("--llm-tokenizer", help="tokenizer for local prompt budgets (default: the model)")
    parser.add_argument("--synthetic-run", type=Path, help="synthetic run (patients, TrialSpace rounds, decision model; "
                        "default: dashboard.synthetic_run, else no_phi/synthetic_run under the data root)")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--host", help="real mode: a loopback address only (default 127.0.0.1); synthetic mode: "
                        f"default {ANY_HOST}")
    parser.add_argument("--allowed-hosts", help="comma-separated Host header names to accept (default: "
                        f"{','.join(LOOPBACK_HOSTS)} in real mode, any in synthetic mode)")
    parser.add_argument("--require-token", action="store_true", help="synthetic mode: require an access token "
                        f"(MMAI_DASH_ACCESS_TOKEN, else {ACCESS_TOKEN_FILE}, created on first use); always on in "
                        "real mode")
    parser.add_argument("--port", type=int, help=f"default {MODE_PORTS['real']} (real), {MODE_PORTS['synthetic']} "
                        "(synthetic)")
    parser.add_argument("--decision-model", help="checkpoint directory or Hub id (default: the run's trained decision "
                        "model once complete, else its newest interim export, else fit 2, else base d1-2B)")
    parser.add_argument("--decision-revision")
    parser.add_argument("--device", help="decision-model device, e.g. cpu or cuda:1 (default cuda:0 when CUDA is "
                                         "available, else cpu)")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), help="the decision model's "
                        "dtype (default auto: bfloat16 where the GPU computes in it, from Ampere on, and on the CPU; "
                        "matching.decision_model.PRE_AMPERE_DTYPE on an older GPU such as a T4)")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--embedding-python", type=Path)
    parser.add_argument("--embedding-cache", type=Path, action="append")
    parser.add_argument("--embedding-device", help="TrialSpace encoder device for the trial cohorts (default cpu)")
    parser.add_argument("--auto-score-n", type=int, help="pairs scored automatically per search (default 10)")
    parser.add_argument("--round-poll-seconds", type=float)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--local-server-only", action="store_true", help="every LLM task uses this app's local model "
                        "server (Data and models), and no other connection is accepted: no gateway, Vertex or other "
                        "OpenAI-compatible endpoint (implies --local-server-default)")
    parser.add_argument("--local-server-autostart", metavar="JSON|PATH", help="start this local model server spec "
                        "(JSON, or a JSON or YAML file) once the decision model has loaded; engine, model and GPUs "
                        "are enough")
    parser.add_argument("--local-server-default", action="store_true", help="once the local model server is ready, "
                        "make it the default connection of every task in sessions that chose none")
    parser.add_argument("--allow-openrouter", action="store_true", help="synthetic mode, with --local-server-only: "
                        "also accept OpenRouter connections with each browser session's own key (synthetic mode "
                        "offers OpenRouter anyway without --local-server-only; real mode never does)")
    parser.add_argument("--no-local-server", action="store_true", help="offer no local model server (a machine that "
                        "cannot host an LLM); with --local-server-only and --allow-openrouter, OpenRouter with each "
                        "session's own key is the only LLM connection (also MMAI_DASH_LOCAL_SERVER_ENABLED=false)")
    parser.add_argument("--llama-server", help="the llama-server binary (default: $MMAI_LLAMA_SERVER, else "
                        "llama-server on PATH)")
    parser.add_argument("--frame-ancestors", help="synthetic mode: comma-separated https origins whose pages may show "
                        "the app in a frame, e.g. https://huggingface.co for a Space page (default: none; site config "
                        "dashboard.synthetic.frame_ancestors)")
    args = parser.parse_args(argv)
    # Before any default reads them.
    if args.site_config:
        os.environ[SITE_CONFIG_ENV] = args.site_config
    if args.data_root:
        os.environ[DATA_ROOT_ENV] = args.data_root
    names = {"trial_data": Path, "oa_db": Path, "store_dir": Path, "grouping": Path, "good_options_catalog": Path,
             "trial_management": Path,
             "rebuild_poll_seconds": float, "llm_backend": str, "llm_endpoint": str, "llm_model": str,
             "llm_tokenizer": str,
             "synthetic_run": Path, "cache_dir": Path, "host": str, "port": int, "decision_model": str,
             "decision_revision": str, "device": str, "dtype": str, "threads": int, "embedding_python": Path,
             "embedding_device": str, "auto_score_n": int, "round_poll_seconds": float}
    values: dict = {}
    for name, kind in names.items():
        raw = getattr(args, name)
        if raw is None and _env(name) is not None:
            raw = kind(_env(name))
        if raw is not None:
            values[name] = raw
    if args.embedding_cache:
        values["embedding_caches"] = tuple(args.embedding_cache)
    values["allow_downloads"] = bool(args.allow_downloads or (_env("allow_downloads") or "").lower() in ("1", "true"))
    values["require_token"] = bool(args.require_token or (_env("require_token") or "").lower() in ("1", "true"))
    values.update(local_server_settings(args, args.mode))
    hosted = load_site_config().get(f"dashboard.{args.mode}.hosted_openrouter", resolve=False)
    if hosted is not None and hosted is not False:  # real mode refuses it (DashSettings)
        values["hosted_openrouter"] = hosted
    frames = args.frame_ancestors or _env("frame_ancestors") or load_site_config().get(
        f"dashboard.{args.mode}.frame_ancestors", resolve=False)
    if frames:
        listed = frames.split(",") if isinstance(frames, str) else frames
        values["frame_ancestors"] = tuple(str(f).strip() for f in listed if str(f).strip())
    # Hub references in the site config (hf://) download only with --allow-downloads; set before any default reads one.
    allow_hub_downloads(values["allow_downloads"])
    hosts = args.allowed_hosts or _env("allowed_hosts")
    if hosts:
        values["allowed_hosts"] = tuple(h.strip() for h in hosts.split(",") if h.strip())
    settings = DashSettings(mode=args.mode, **values)
    settings.command = args.command
    return settings
