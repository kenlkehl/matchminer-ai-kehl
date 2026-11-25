"""Site configuration and the data root: values that belong to one installation, kept outside this repository.

A site config is a YAML file named by ``--site-config`` or ``MMAI_SITE_CONFIG``. It holds what differs between
institutions and machines and is never committed here: endpoint URLs, cloud projects and gateways, database schema
maps, site and group names, intranet links and local paths. Code reads it with dotted keys::

    site = load_site_config()
    site.get("endpoints.teacher.url")       # None when the site does not set it
    site.require("endpoints.teacher.url")   # an error naming the key, the file and how to set it

The data root (``--data-root``, ``MMAI_DATA_ROOT`` or the site config's ``paths.data_root``) holds ``phi/`` and
``no_phi/``. It has no default, and a data path inside this package's source checkout is refused, so clinical data
cannot land in the repository.

**Hugging Face Hub files.** A value may name a file or folder of a Hub repository at a pinned revision, in the Hub's
own path form: ``hf://datasets/<org>/<name>@<revision>/<path>`` (also ``hf://models/<org>/<name>@<revision>``).
``get`` returns its local path in the Hugging Face cache (``HF_HUB_CACHE``), downloading only that file or folder on
first use (``hub_path``). Downloads follow ``allow_hub_downloads`` (default: huggingface_hub's ``HF_HUB_OFFLINE``);
without them only cached files resolve. A dataset reference must name a file or folder, so a whole dataset is never
downloaded; ``get(key, resolve=False)`` returns the reference itself (the dashboard's ``synthetic_run`` names a whole
published run that way: ``apps.dashboard.hub_run``). Hub files are read-only: never name one where an app writes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Any

SITE_CONFIG_ENV = "MMAI_SITE_CONFIG"
DATA_ROOT_ENV = "MMAI_DATA_ROOT"
HUB_SCHEME = "hf://"
_MISSING = object()


class SiteConfig:
    """Read-only view of a site config (empty when none is configured)."""

    def __init__(self, values: dict | None = None, path: Path | None = None):
        self.values = values or {}
        self.path = path

    def __bool__(self) -> bool:
        return bool(self.values)

    def get(self, key: str, default: Any = None, *, resolve: bool = True) -> Any:
        """The value at ``key``; a Hub reference (or a list of them) as its local path, unless ``resolve`` is
        False (see the module docstring)."""
        value: Any = self.values
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return resolve_hub_value(value) if resolve else value

    def require(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING or value is None:
            where = f"site config {self.path}" if self.path else f"a site config (--site-config or {SITE_CONFIG_ENV})"
            raise KeyError(f"{key!r} is not set; add it to {where}.")
        return value

    def section(self, key: str) -> dict:
        value = self.get(key, {}, resolve=False)
        return dict(value) if isinstance(value, dict) else {}


_CACHE: dict[tuple[str, int, int], SiteConfig] = {}


def load_site_config(path: str | os.PathLike | None = None) -> SiteConfig:
    """The site config at ``path``, else at ``$MMAI_SITE_CONFIG``, else an empty one."""
    chosen = path or os.environ.get(SITE_CONFIG_ENV)
    if not chosen:
        return SiteConfig()
    resolved = Path(chosen).expanduser().resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_mtime_ns, stat.st_size)  # an edit within the same second still shows in the size
    if key not in _CACHE:
        import yaml

        values = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if not isinstance(values, dict):
            raise ValueError(f"site config {resolved} must be a YAML mapping")
        _CACHE[key] = SiteConfig(values, resolved)
    return _CACHE[key]


# ---------------------------------------------------------------------------------------------- Hub references

_REPO_TYPES = {"datasets": "dataset", "models": "model"}
_NAME = re.compile(r"^[A-Za-z0-9][\w.-]*$")
_RESOLVED: dict[str, Path] = {}
_hub_downloads: bool | None = None  # None: huggingface_hub's HF_HUB_OFFLINE decides


@dataclass(frozen=True)
class HubRef:
    """A file or folder (``path``; empty: the whole repository) of a Hub repository at ``revision``."""
    repo_type: str  # "dataset" or "model"
    repo_id: str
    revision: str
    path: str = ""

    def __str__(self) -> str:
        kind = next(k for k, v in _REPO_TYPES.items() if v == self.repo_type)
        return f"{HUB_SCHEME}{kind}/{self.repo_id}@{self.revision}" + (f"/{self.path}" if self.path else "")


def is_hub_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(HUB_SCHEME)


def parse_hub_ref(value: str) -> HubRef:
    """``hf://datasets/<org>/<name>@<revision>[/<path>]`` (or ``models/``) as a ``HubRef``. The revision is
    required: a commit hash (recommended), branch or tag without "/"."""
    if not is_hub_ref(value):
        raise ValueError(f"not a Hub reference: {value!r}")
    kind, _, rest = value[len(HUB_SCHEME):].partition("/")
    org, _, rest = rest.partition("/")
    name_revision, _, path = rest.partition("/")
    name, at, revision = name_revision.partition("@")
    if kind not in _REPO_TYPES or not (_NAME.match(org) and _NAME.match(name)) or not at or not revision:
        raise ValueError(f"{value!r} is not hf://datasets|models/<org>/<name>@<revision>[/<path>] (the revision "
                         "is required)")
    path = path.strip("/")
    if ".." in path.split("/"):
        raise ValueError(f"{value!r}: the path may not contain '..'")
    return HubRef(_REPO_TYPES[kind], f"{org}/{name}", revision, path)


def allow_hub_downloads(allowed: bool | None) -> None:
    """Whether ``hub_path`` may download (None: huggingface_hub's ``HF_HUB_OFFLINE`` decides). The dashboard sets
    it from ``--allow-downloads`` before reading its settings."""
    global _hub_downloads
    _hub_downloads = allowed


def _downloads_allowed() -> bool:
    if _hub_downloads is not None:
        return _hub_downloads
    from huggingface_hub import constants

    return not constants.HF_HUB_OFFLINE


def hub_path(value: str | HubRef) -> Path:
    """The local path of a Hub file or folder, downloaded into the Hugging Face cache when it is not there yet
    (only that file or folder: ``snapshot_download`` with ``allow_patterns``). Resolved once per process."""
    ref = value if isinstance(value, HubRef) else parse_hub_ref(value)
    key = str(ref)
    if key in _RESOLVED:
        return _RESOLVED[key]
    if ref.repo_type == "dataset" and not ref.path:
        raise ValueError(f"{key}: name a file or folder of the dataset (a whole dataset is never downloaded)")
    from huggingface_hub import snapshot_download

    patterns = [ref.path, ref.path + "/*"] if ref.path else None
    options = dict(repo_type=ref.repo_type, revision=ref.revision, allow_patterns=patterns)
    if _downloads_allowed():
        # Files already cached at a pinned commit are not downloaded again; a folder whose download was cut short
        # is completed.
        print(f"Hub: {key}", file=sys.stderr, flush=True)
        target = Path(snapshot_download(ref.repo_id, **options)) / ref.path
    else:
        try:
            target = Path(snapshot_download(ref.repo_id, local_files_only=True, **options)) / ref.path
        except Exception as exc:
            raise FileNotFoundError(f"{key} is not in the Hugging Face cache and downloads are off "
                                    f"(--allow-downloads): {type(exc).__name__}") from None
    if not target.exists():
        raise FileNotFoundError(f"{key}: {ref.path} is not in {ref.repo_id}@{ref.revision}"
                                + ("" if _downloads_allowed() else " (or not cached; downloads are off)"))
    _RESOLVED[key] = target
    return target


def resolve_hub_value(value: Any) -> Any:
    """``value`` with Hub references (a string, or strings in a list) replaced by their local paths."""
    if is_hub_ref(value):
        return str(hub_path(value))
    if isinstance(value, list):
        return [str(hub_path(v)) if is_hub_ref(v) else v for v in value]
    return value


# ---------------------------------------------------------------------------------------------- the data root

def source_checkout() -> Path | None:
    """The repository root when this package runs from a source checkout (src layout), else None."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "matchminer_ai").is_dir() else None


def outside_checkout(path: str | os.PathLike, what: str = "data path") -> Path:
    """``path`` resolved, refusing one inside the source checkout."""
    resolved = Path(path).expanduser().resolve()
    root = source_checkout()
    if root is not None and resolved.is_relative_to(root):
        raise ValueError(f"{what} {resolved} is inside the source checkout {root}; keep data outside the repository.")
    return resolved


def data_root(explicit: str | os.PathLike | None = None, *, site: SiteConfig | None = None) -> Path:
    """The data root from ``explicit``, ``$MMAI_DATA_ROOT`` or the site config, outside the source checkout."""
    chosen = explicit or os.environ.get(DATA_ROOT_ENV) or (site or load_site_config()).get("paths.data_root",
                                                                                            resolve=False)
    if not chosen:
        raise ValueError(f"No data root: pass --data-root, set {DATA_ROOT_ENV} or the site config's paths.data_root.")
    if is_hub_ref(chosen):
        raise ValueError("paths.data_root must be a local, writable folder, not a Hub reference.")
    return outside_checkout(chosen, "data root")
