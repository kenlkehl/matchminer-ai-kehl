"""The app's two modes (bind, sign-in, data side) and the shared job manager (sessions, stopping, errors)."""
from __future__ import annotations

import threading
import time

import pytest


def test_modes_set_bind_sign_in_and_data_side(tmp_path):
    from matchminer_ai.apps.dashboard.settings import DashSettings, parse_settings

    real = DashSettings(mode="real", store_dir=tmp_path / "store", device="cpu")
    assert (real.host, real.port, real.require_token) == ("127.0.0.1", 8781, True)
    assert set(real.allowed_hosts) == {"127.0.0.1", "localhost"}
    synthetic = DashSettings(mode="synthetic", store_dir=tmp_path / "store", device="cpu")
    assert (synthetic.host, synthetic.port, synthetic.require_token, synthetic.allowed_hosts) == (
        "0.0.0.0", 8034, False, ("*",))
    with pytest.raises(ValueError, match="never uses a phi folder"):
        DashSettings(mode="synthetic", store_dir=tmp_path / "phi" / "store", device="cpu")
    with pytest.raises(ValueError, match="out of no_phi"):
        DashSettings(mode="real", store_dir=tmp_path / "no_phi" / "store", device="cpu")
    with pytest.raises(SystemExit):  # the mode is required on the command line
        parse_settings(["--store-dir", str(tmp_path / "store")])


def test_real_mode_refuses_a_non_loopback_host(tmp_path, monkeypatch):
    from matchminer_ai.apps.dashboard import __main__ as entry

    with pytest.raises(SystemExit, match="127.0.0.1 only"):
        entry.main(["--mode", "real", "--host", "0.0.0.0", "--store-dir", str(tmp_path / "store"), "--device", "cpu"])


def wait(job, seconds=5):
    deadline = time.time() + seconds
    while job.active and time.time() < deadline:
        time.sleep(0.01)
    return job


def test_jobs_belong_to_their_session_stop_and_hide_error_text():
    from matchminer_ai.apps.dashboard.core.errors import BusyError, InputError
    from matchminer_ai.apps.dashboard.core.jobs import JobManager, JobRegistry

    jobs = JobManager(exclusive=True)
    registry = JobRegistry(guidelines=jobs)
    release = threading.Event()

    def slow(progress):
        while not release.wait(0.01):
            progress("working")

    running = jobs.submit("workup", "one patient", slow, owner="session-a")
    assert registry.get(running.id, "session-b") is None  # another browser session cannot see it
    with pytest.raises(BusyError):
        jobs.submit("workup", "again", slow, owner="session-a")
    assert registry.cancel(running.id, "session-a").status in ("cancelling", "cancelled")
    assert wait(running).status == "cancelled"

    def leaks(progress):
        raise RuntimeError("note text of a patient")

    failed = wait(jobs.submit("summarize", "x", leaks, owner="session-a"))
    assert failed.status == "error" and failed.error.startswith("RuntimeError") and "patient" not in failed.error

    def explains(progress):
        raise InputError("Enter patient notes first.")

    assert wait(jobs.submit("summarize", "x", explains, owner="session-a")).error == "Enter patient notes first."
    system = wait(jobs.submit("rebuild", "store", lambda progress: {"ok": True}))
    assert registry.get(system.id, "session-b").result == {"ok": True}  # system jobs are visible to every session
    assert registry.cancel(system.id, "session-b") is None  # and not stopped from a page
    jobs.shutdown()


def test_haiku_through_the_gateway_is_the_default_when_the_site_names_it(tmp_path, monkeypatch):
    from matchminer_ai.apps.dashboard.settings import DashSettings, guideline_settings

    assert DashSettings(mode="synthetic", store_dir=tmp_path / "a", device="cpu").llm_backend == "gemini4_vertex"
    site = tmp_path / "gateway.yaml"
    site.write_text("databricks:\n  host: https://workspace.example.cloud.databricks.com\n  profile: p\n"
                    "  endpoints:\n    haiku: catalog.schema.team-claude-haiku-5-5\n")
    monkeypatch.setenv("MMAI_SITE_CONFIG", str(site))
    settings = DashSettings(mode="synthetic", store_dir=tmp_path / "b", device="cpu")
    assert (settings.llm_backend, settings.llm_model) == ("claude_databricks", "catalog.schema.team-claude-haiku-5-5")
    tasks = guideline_settings(settings)
    assert {tasks.for_task(t).llm_backend for t in ("summarize", "workup_sql", "criteria_llm")} == {"claude_databricks"}
