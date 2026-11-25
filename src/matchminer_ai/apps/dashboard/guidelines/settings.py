"""Settings of the app's guideline features (the Patient tab's Guidelines, the Population tab) and its model
connections, one per LLM task.

``phi`` follows the app's mode (``DashSettings.mode == "real"``): real patient records come from an operator-selected
DuckDB file or a preprocessed population cache, and the caches stay under the data root's ``phi/`` side; synthetic
mode reads the synthetic population and never a ``phi`` folder.

Paths default to folders under the data root (``matchminer_ai.site.data_root``). Institution-specific values come
from the site config, read when a value is needed (never at import, so ``--site-config`` applies):
``dashboard.<mode>.<key>`` for a mode's data (``real`` or ``synthetic``), ``dashboard.guideline_catalogs``, and
``vertex`` and ``databricks`` for the connection profiles; see the site-config reference in the docs.
"""

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from matchminer_ai.llm import connections
from matchminer_ai.site import data_root, load_site_config


def data_path(relative):
    """``relative`` under the data root (which has no default; see ``matchminer_ai.site``)."""
    return data_root() / relative


def site_path(phi, key, relative):
    """The site config's ``dashboard.<mode>.<key>``, else ``relative`` under the data root (None without one)."""
    value = load_site_config().get(f"dashboard.{'real' if phi else 'synthetic'}.{key}")
    if value:
        return Path(value).expanduser()
    return data_path(relative) if relative else None


# Each mode keeps model metadata and guideline embeddings on its own side of the data root.
NO_PHI_CACHE = "no_phi/guideline_fetcher_cache"
PHI_CACHE = "phi/guideline_fetcher_phi_cache"
POPULATION_ARCHIVE_LIMIT = 8 * 1024**3


def default_catalog_root():
    """The finalized NCCN catalogs both modes use (site config ``dashboard.guideline_catalogs``)."""
    from matchminer_ai.trials.guideline_catalog import default_guideline_catalog_root

    return default_guideline_catalog_root()


def default_population_duckdb():
    """Real mode's patient database copy (site config ``dashboard.real.population_duckdb``)."""
    return site_path(True, "population_duckdb", "phi/guideline_fetcher_population.duckdb")


def default_population_cache():
    """Built by the preprocess_population command; contains PHI, so it stays beside the database (site config
    ``dashboard.real.population_cache``)."""
    configured = site_path(True, "population_cache", None)
    if configured:
        return configured
    duckdb = default_population_duckdb()
    return duckdb.parent / "guideline_population_cache" / duckdb.stem


def default_duckdb_roots():
    """Folders real mode may browse for DuckDB files (site config ``dashboard.real.duckdb_roots``)."""
    roots = load_site_config().get("dashboard.real.duckdb_roots")
    return [Path(r).expanduser() for r in roots] if roots else [data_root()]


def default_extraction_records(phi):
    """Structured extraction records built offline by the preprocess_extractions command. Real mode's sit beside its
    population cache; synthetic mode's hold synthetic notes only."""
    configured = site_path(phi, "extraction_records", None)
    if configured:
        return configured
    if phi:
        duckdb = default_population_duckdb()
        return duckdb.parent / "guideline_extraction_records" / duckdb.stem
    return data_path("no_phi/guideline_fetcher_extractions/synthetic")


# The connection profiles (matchminer_ai.llm.connections): the Vertex project is the site config's vertex.project.
# Gemini 4 Hydrogen (GA 2026-11-12) is the default since 2026-11-14 (user: the Gemma 4 26B MaaS endpoint "is very
# janky").
VERTEX_LABELS = connections.VERTEX_LABELS
GEMINI_BACKENDS = connections.GEMINI_BACKENDS
VERTEX_BACKENDS = frozenset(connections.VERTEX_MODELS)  # Gemini and Gemma MaaS
DEFAULT_BACKEND = "gemini4_vertex"
LOCAL = connections.LOCAL_BACKEND  # an OpenAI-compatible server (vLLM)


def vertex_project():
    return connections.vertex_project()


def vertex_profiles():
    return connections.vertex_profiles()


# Claude through the Databricks Unity AI Gateway's Anthropic Messages route (provider ``databricks_anthropic``), for
# every LLM task since 2026-11-21 (user instruction; before, the SQL agent only). The server fixes the gateway URL (the
# site config's databricks.host). The model is the full three-part endpoint name from the site config's
# databricks.endpoints (Haiku 5.5 is the app's default, user 2026-11-21). Access tokens come from the Databricks CLI
# login profile (`databricks auth login --host <host> --profile <profile>`). How a connection sets a preset section
# (efforts, prompt budgets) is the package's (matchminer_ai.llm.connections.connect_section).
CLAUDE_LABELS = {connections.CLAUDE_BACKEND: "Claude · Databricks AI Gateway"}
CLAUDE_EFFORT = connections.CLAUDE_EFFORT
# The chat assistant (Opus 5.5 by default): Opus 5.5's default effort, so a chat turn's several requests stay quick.
CLAUDE_ASSISTANT_EFFORT = "medium"
CLAUDE_CONTEXT = connections.CLAUDE_CONTEXT_TOKENS
CLAUDE_ENDPOINT_NAME = connections.CLAUDE_ENDPOINT_NAME
CLAUDE_BACKENDS = frozenset(CLAUDE_LABELS)
# The task whose connection writes the workup queries: extraction queries' compile step and the SQL agent.
SQL_TASK = "workup_sql"
# The Assistant tab's chat agent (``apps.dashboard.assistant``).
ASSISTANT_TASK = "assistant"
QUERY_WRITERS = connections.QUERY_WRITERS


def databricks_profile():
    return connections.databricks_profile()


claude_models = connections.claude_models


def claude_backends():
    """{backend: {endpoint, model, models, tokenizer}}: the gateway URL from the site config; Haiku's endpoint name
    offered first, and every endpoint the site names."""
    return {connections.CLAUDE_BACKEND: connections.claude_profile()}


def default_task_connections() -> dict:
    """The app's per-task defaults besides the default connection, when the site config names the gateway and the
    endpoint: query writing with Claude Sonnet 5.5 (user, 2026-11-21) and the chat assistant with Claude Opus 5.5
    (user, 2026-11-22), both through the gateway."""
    if not connections.databricks_host():
        return {}
    profile = claude_backends()["claude_databricks"]
    chosen = {SQL_TASK: profile["models"].get("sonnet"), ASSISTANT_TASK: profile["models"].get("opus")}
    return {task: {"llm_backend": "claude_databricks", "endpoint": profile["endpoint"], "model": model,
                   "tokenizer": profile["tokenizer"]} for task, model in chosen.items() if model}


def default_backend() -> str:
    """Claude Haiku 5.5 through the gateway when the site config names the gateway and a Haiku endpoint (user,
    2026-11-21), else Gemini 4 Hydrogen on Vertex."""
    return "claude_databricks" if connections.databricks_host() and claude_models().get("haiku") else DEFAULT_BACKEND


# The LLM alternative to the decision model wherever the dashboard scores a pair (``apps.dashboard.llm_decisions``):
# match scores, common exclusions and per-criterion checks. It replaced the per-criterion-only ``criteria_llm`` task
# (2026-11-22, user instruction); a connection saved under the old name is moved to it (``migrate_task_connections``).
LLM_DECISIONS_TASK = "llm_decisions"
CRITERIA_LABELS_TASK = "criteria_labels"
RENAMED_TASKS = {"criteria_llm": LLM_DECISIONS_TASK}

# Every LLM task of the app, each with its own connection (until set: see Settings.for_task).
LLM_TASKS = {
    "summarize": "Patient summarization",
    "workup": "Guideline concordance checks",
    SQL_TASK: "Query writing: guideline workup and trial criteria (compiled queries and SQL agent)",
    "workup_full_backup": "Guideline concordance checks: Full note backup",
    "workup_agent_backup": "Guideline concordance checks: Agent backup",
    "good_option": "GoodOption checks",
    LLM_DECISIONS_TASK: "LLM decision checks: match scores, common exclusions and per-criterion checks",
    ASSISTANT_TASK: "Chat assistant",
    # Edited and new criteria lines of Clinical trial management (apps.dashboard.criteria_labels): inclusion or
    # exclusion, key or common, with the cohort pipeline's own classification step (user, 2026-11-23).
    CRITERIA_LABELS_TASK: "Criteria classification: edited cohort criteria (inclusion or exclusion, key or common)",
}


normalize_endpoint = connections.normalize_endpoint


def migrate_task_connections(values: dict) -> dict:
    """Task connections under their current names: a renamed task's connection (``criteria_llm``) moves to its new
    task (``llm_decisions``) unless that one has its own."""
    moved = dict(values or {})
    for old, new in RENAMED_TASKS.items():
        if old in moved:
            connection = moved.pop(old)
            moved.setdefault(new, connection)
    return moved


def mode_side_error(phi, path):
    """Why ``path`` is on the other mode's side of data/ (None when it is not).

    Real-mode data and caches never sit in a no_phi folder, and synthetic mode never reads or writes a phi folder.
    """
    parts = Path(path).expanduser().resolve().parts
    if phi and "no_phi" in parts:
        return "Real mode keeps patient records out of no_phi folders."
    if not phi and "phi" in parts:
        return "Synthetic mode never uses a phi folder."
    return None


def default_device():
    """CUDA when PyTorch can reach it, else the CPU (the app passes its own device)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Settings:
    # Real patient records (the app's real mode); see the module docstring.
    phi: bool = False
    # The default connection; a task's own connection overrides it (task_connections, for_task).
    llm_backend: str = field(default_factory=default_backend)
    # Both modes use the same finalized NCCN catalogs.
    catalog_root: Path = field(default_factory=default_catalog_root)
    endpoint: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3.8-27B-FP8"
    tokenizer: str = "Qwen/Qwen3.8-27B-FP8"
    # TrialSpace and the decision model run in the app's process, on this device.
    device: str = field(default_factory=default_device)
    # The app's decision-model section (model, revision, dtype, budgets), so guideline features score with the
    # model the other tabs use and the package holds one loaded copy. Empty: the preset's.
    decision_model: dict = field(default_factory=dict)
    # None: NO_PHI_CACHE or, in real mode, PHI_CACHE.
    cache: Path | None = None
    # Synthetic mode's population: summaries and notes, sampled from one split (site config
    # dashboard.synthetic.population_split; None: every patient).
    population_summaries: Path | None = None
    population_notes: Path | None = None
    population_split: str | None = field(
        default_factory=lambda: load_site_config().get("dashboard.synthetic.population_split"))
    # The synthetic tables' patient key (site config dashboard.synthetic.population_key): pseudo_mrn, or patient_id
    # for the v23 run, whose patient_id is the patient's own (older tables' is a generation-prompt ID).
    population_key: str = field(
        default_factory=lambda: load_site_config().get("dashboard.synthetic.population_key") or "pseudo_mrn")
    # Real mode only: the patient database, its preprocessed cache and browsable folders.
    population_duckdb: Path = field(default_factory=default_population_duckdb)
    population_cache: Path = field(default_factory=default_population_cache)
    duckdb_roots: list[Path] = field(default_factory=default_duckdb_roots)
    # None: the mode's default (see __post_init__).
    extraction_records: Path | None = None
    population_sample_size: int = 1000
    population_seed: int = 23
    population_upload_limit: int = 512 * 1024 * 1024
    population_archive_limit: int = POPULATION_ARCHIVE_LIMIT
    ttl: int = 3600
    # Synthetic mode's example notes: a CSV in place of the Hub sample, and the biliary case.
    synthetic_notes: str | None = None
    biliary_synthetic_notes: Path | None = None
    max_upload: int = 20 * 1024 * 1024
    reasoning: bool = False
    backup_enabled: bool = False
    failure_threshold: int = 3
    task_connections: dict = field(default_factory=dict)

    def __post_init__(self):
        self.task_connections = migrate_task_connections(self.task_connections)
        if self.cache is None:
            self.cache = data_path(PHI_CACHE if self.phi else NO_PHI_CACHE)
        self.cache = Path(self.cache)
        # A cache override must stay on the mode's own side of data/.
        side_error = mode_side_error(self.phi, self.cache)
        if side_error:
            raise ValueError(side_error + " Choose another guideline cache.")
        if self.population_summaries is None:
            self.population_summaries = site_path(False, "population_summaries", "no_phi/patient_summaries.parquet")
        if self.population_notes is None:
            self.population_notes = site_path(False, "population_notes", "no_phi/all_synthetic_notes.parquet")
        if self.biliary_synthetic_notes is None:
            self.biliary_synthetic_notes = data_path("no_phi/guideline_fetcher_examples/biliary_patient_notes.csv")
        self.duckdb_roots = [Path(p) for p in self.duckdb_roots]
        if self.extraction_records is None:
            self.extraction_records = default_extraction_records(self.phi)
        self.extraction_records = Path(self.extraction_records)
        side_error = mode_side_error(self.phi, self.extraction_records)
        if side_error:
            raise ValueError(side_error + " Choose other extraction records.")

    @property
    def mode(self):
        return "real" if self.phi else "synthetic"

    def extraction_config(self):
        """The inference configuration for extraction-based workup checks.

        This connection serves on-demand document and patient extraction (their
        preset sections, with its endpoint, model and thinking switch). The query
        writers, the compile steps of workup and criteria queries and the SQL agent,
        use the query-writing task's connection (``SQL_TASK``), which is this one
        until set (the app sets Claude Sonnet 5.5 through the gateway; user,
        2026-11-21).
        Hosted Vertex models answer free JSON that the package validates and
        retries, instead of vLLM structured outputs; the SQL agent's small action
        schema keeps JSON-schema responses there too.
        """
        c = self.inference_config()
        # On-demand extraction sends one request per note; let several run at once.
        c.remote["max_concurrent_requests"] = 8
        for name in ("document_extraction", "patient_extraction"):
            self._connect_section(getattr(c, name), name)
        writer = self.for_task(SQL_TASK)
        for name in QUERY_WRITERS:
            writer._connect_section(getattr(c, name), name, own_connection=True)
        return c

    @property
    def llm_connection(self):
        """This connection for the package's section builders (``matchminer_ai.llm.connections``)."""
        return connections.Connection(self.llm_backend, self.model, self.endpoint, self.tokenizer,
                                      self.reasoning if self.thinking_supported else None)

    def _connect_section(self, section, name, *, own_connection=False):
        """Point one preset section at this connection (``connections.connect_section``).

        ``own_connection``: the section names its endpoint and provider itself, so it
        can differ from the shared ``remote`` the other sections use.
        """
        connections.connect_section(section, name, self.llm_connection, own_connection=own_connection)

    def for_task(self, task):
        if task not in LLM_TASKS:
            raise ValueError("Unsupported LLM task")
        values = self.task_connections.get(task)
        if values:
            return replace(self, **values)
        if task == SQL_TASK:
            # Until set, the SQL agent uses the concordance checks' connection.
            return self.for_task("workup")
        return replace(self, backup_enabled=False) if task.endswith("_backup") else self

    @property
    def thinking_supported(self):
        return self.llm_backend == "gemma_vertex" or (
            self.llm_backend == LOCAL
            and bool(re.search(r"(?:qwen[-_ ]?3|gemma[-_ ]?4)", self.model, re.IGNORECASE))
        )

    def inference_config(self):
        from matchminer_ai import load_default_preset

        c = load_default_preset()
        c.debug_mode = False
        c.model_metadata_cache_dir = str(self.cache)
        # TrialSpace and the decision model (match-quality scoring) run locally.
        c.embedding["device"] = self.device
        c.raw["decision_model"].update(self.decision_model)
        c.raw["decision_model"]["device"] = self.decision_model.get("device", self.device)
        c.remote.update(max_concurrent_requests=1, request_timeout=7200)
        connections.connect_patient_section(c, self.llm_connection)
        c.patient["prompt_build_workers"] = 1
        return c
