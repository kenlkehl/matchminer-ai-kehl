import os

import pytest

from matchminer_ai import site


def test_site_config_dotted_keys_and_require(tmp_path, monkeypatch):
    path = tmp_path / "site.yaml"
    path.write_text("endpoints:\n  teacher:\n    url: http://teacher.example:8000/v1\n")
    monkeypatch.setenv(site.SITE_CONFIG_ENV, str(path))
    config = site.load_site_config()
    assert config.get("endpoints.teacher.url") == "http://teacher.example:8000/v1"
    assert config.get("endpoints.judge.url") is None
    with pytest.raises(KeyError, match="endpoints.judge.url"):
        config.require("endpoints.judge.url")


def test_no_site_config_is_empty(monkeypatch):
    monkeypatch.delenv(site.SITE_CONFIG_ENV, raising=False)
    assert not site.load_site_config()


def test_data_root_has_no_default_and_refuses_the_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv(site.DATA_ROOT_ENV, raising=False)
    monkeypatch.delenv(site.SITE_CONFIG_ENV, raising=False)
    with pytest.raises(ValueError, match="No data root"):
        site.data_root()
    assert site.data_root(tmp_path) == tmp_path.resolve()
    checkout = site.source_checkout()
    if checkout is not None:
        with pytest.raises(ValueError, match="inside the source checkout"):
            site.data_root(checkout / "data")


def test_hub_references_resolve_to_files_in_the_cache(tmp_path, monkeypatch):
    """hf:// values: parsed in the Hub's path form, downloaded one file or folder at a time (snapshot_download with
    allow_patterns), offline only from the cache, never a whole dataset, and never as the data root."""
    import huggingface_hub

    revision = "0" * 40
    calls = []

    def snapshot_download(repo_id, *, repo_type, revision, allow_patterns, local_files_only=False):
        calls.append((repo_id, repo_type, revision, tuple(allow_patterns or ()), local_files_only))
        root = tmp_path / "hub" / repo_id / revision
        (root / "records" / "store").mkdir(parents=True, exist_ok=True)
        (root / "records" / "store" / "manifest.json").write_text("{}")
        return str(root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(site, "_RESOLVED", {})
    ref = f"hf://datasets/example-org/example-data@{revision}/records/store"
    parsed = site.parse_hub_ref(ref)
    assert (parsed.repo_type, parsed.repo_id, parsed.revision, parsed.path) == (
        "dataset", "example-org/example-data", revision, "records/store")
    assert str(parsed) == ref
    with pytest.raises(ValueError, match="revision is required"):
        site.parse_hub_ref("hf://datasets/example-org/example-data/records")
    with pytest.raises(ValueError, match="whole dataset"):
        site.hub_path(f"hf://datasets/example-org/example-data@{revision}")

    path = tmp_path / "site.yaml"
    path.write_text(f"paths:\n  data_root: {ref}\ndashboard:\n  store: {ref}\n  roots: [{ref}, /local/folder]\n")
    monkeypatch.setenv(site.SITE_CONFIG_ENV, str(path))
    monkeypatch.delenv(site.DATA_ROOT_ENV, raising=False)
    config = site.load_site_config()
    site.allow_hub_downloads(False)
    try:
        expected = str(tmp_path / "hub" / "example-org/example-data" / revision / "records" / "store")
        assert config.get("dashboard.store") == expected
        assert calls == [("example-org/example-data", "dataset", revision, ("records/store", "records/store/*"), True)]
        assert config.get("dashboard.roots") == [expected, "/local/folder"]
        assert config.get("dashboard.store", resolve=False) == ref
        assert len(calls) == 1  # resolved once per process
        with pytest.raises(ValueError, match="writable folder"):
            site.data_root()
    finally:
        site.allow_hub_downloads(None)


def test_environment_variables_and_missing_hub_paths(tmp_path, monkeypatch):
    """${NAME:-default} picks a revision (a Space variable, else the default); a file or folder the revision does
    not have reads as a missing path (the reader finds nothing), not an error, except where a caller requires it."""
    import huggingface_hub

    def snapshot_download(repo_id, *, repo_type, revision, allow_patterns, local_files_only=False):
        return str(tmp_path / "hub" / repo_id / revision)  # nothing of the requested folder

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(site, "_RESOLVED", {})
    monkeypatch.setattr(site, "_ABSENT", {})
    path = tmp_path / "site.yaml"
    path.write_text("dashboard:\n"
                    "  catalogs: hf://datasets/example-org/example-data@${EXAMPLE_REVISION:-main}/catalogs\n"
                    "  root: ${EXAMPLE_ROOT}\n")
    monkeypatch.setenv(site.SITE_CONFIG_ENV, str(path))
    monkeypatch.delenv("EXAMPLE_REVISION", raising=False)
    monkeypatch.delenv("EXAMPLE_ROOT", raising=False)
    config = site.load_site_config()
    site.allow_hub_downloads(True)
    try:
        assert config.get("dashboard.catalogs", resolve=False).endswith("@main/catalogs")
        missing = config.get("dashboard.catalogs")
        assert missing.endswith("/main/catalogs") and not os.path.exists(missing)
        with pytest.raises(FileNotFoundError):
            site.hub_path("hf://datasets/example-org/example-data@main/catalogs")
        monkeypatch.setenv("EXAMPLE_REVISION", "1" * 40)
        assert config.get("dashboard.catalogs", resolve=False).endswith("@" + "1" * 40 + "/catalogs")
        with pytest.raises(KeyError, match="EXAMPLE_ROOT"):
            config.get("dashboard.root")
    finally:
        site.allow_hub_downloads(None)
