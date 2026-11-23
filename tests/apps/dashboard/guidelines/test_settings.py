"""The guideline settings' defaults and mode sides, and the app's mode reaching them."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from matchminer_ai.apps.dashboard.guidelines.settings import Settings


@pytest.mark.parametrize("available, expected", [(True, "cuda"), (False, "cpu")])
def test_automatic_device_reaches_both_local_models(monkeypatch, available, expected):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available)))
    config = SimpleNamespace(embedding={}, raw={"decision_model": {}}, remote={}, patient={"remote": {}})
    monkeypatch.setitem(sys.modules, "matchminer_ai", SimpleNamespace(load_default_preset=lambda: config))
    settings = Settings(llm_backend="local")
    resolved = settings.inference_config()
    assert settings.device == expected
    assert resolved.embedding["device"] == expected
    assert resolved.raw["decision_model"]["device"] == expected


@pytest.mark.parametrize("requested", ["cpu", "cuda:1"])
def test_explicit_device_does_not_probe_or_override_choice(monkeypatch, requested):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert Settings(device=requested).device == requested


def test_each_mode_keeps_its_own_cache_and_extraction_records(tmp_path):
    from matchminer_ai.apps.dashboard.guidelines import settings

    no_phi_cache, phi_cache = settings.data_path(settings.NO_PHI_CACHE), settings.data_path(settings.PHI_CACHE)
    synthetic, real = Settings(device="cpu"), Settings(phi=True, device="cpu")
    assert synthetic.mode == "synthetic" and real.mode == "real"
    assert synthetic.cache == no_phi_cache and "no_phi" in no_phi_cache.parts
    assert real.cache == phi_cache and "phi" in phi_cache.parts
    assert real.inference_config().model_metadata_cache_dir == str(phi_cache)
    assert "no_phi" in synthetic.extraction_records.parts and "phi" in real.extraction_records.parts
    assert Settings(cache=tmp_path / "data/no_phi/cache").cache == tmp_path / "data/no_phi/cache"
    with pytest.raises(ValueError, match="no_phi"):
        Settings(phi=True, cache=tmp_path / "data/no_phi/cache")
    assert Settings(phi=True, cache=tmp_path / "data/phi/cache").cache == tmp_path / "data/phi/cache"
    with pytest.raises(ValueError, match="phi folder"):
        Settings(cache=tmp_path / "data/phi/cache")
    records = tmp_path / "data/no_phi/records"
    assert Settings(extraction_records=records).extraction_records == records
    with pytest.raises(ValueError, match="no_phi"):
        Settings(phi=True, cache=tmp_path / "phi/cache", extraction_records=records)
    with pytest.raises(ValueError, match="phi folder"):
        Settings(extraction_records=tmp_path / "data/phi/records")


@pytest.mark.parametrize("mode", ["synthetic", "real"])
def test_the_apps_mode_sets_the_guideline_side(make_client, mode):
    client = make_client(mode=mode)
    guidelines = client.app.state.mmai.guidelines
    assert guidelines.settings.mode == mode and guidelines.settings.device == "cpu"
    config = client.get("/api/models").json()
    # Synthetic example notes are offered in synthetic mode only.
    assert bool(config["synthetic_note_sources"]) is (mode == "synthetic")
    if mode == "real":
        assert client.get("/api/patient/synthetic-notes").status_code == 400
    # Real patient databases and population caches exist in real mode only.
    defaults = client.get("/api/population").json()["defaults"]
    assert set(defaults) == {"size", "seed", "extraction_records",
                             *(("database", "cache") if mode == "real" else ("split",))}
    for path in ("/api/population/databases", "/api/population/database", "/api/population/cache"):
        assert (client.get(path).status_code == 404) is (mode == "synthetic"), path
    if mode == "synthetic":
        assert "phi" not in Path(defaults["extraction_records"]).parts
        rejected = client.post("/api/population/sample", json={"size": 1, "seed": 1, "database": "/fabricated.duckdb"})
        assert rejected.status_code == 422
