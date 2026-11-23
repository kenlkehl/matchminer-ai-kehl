"""Clinical trial management's overlay (``trial_management``): edited criteria reaching every matching input,
originals and subcohorts, propagation, revert, group pins, the stale-base flag, classification with the cohort
pipeline's own step, and the file's round trip, export and import. Fabricated trials only (``test_dash``)."""
from __future__ import annotations

import json
import time

import pytest

from .conftest import FakeEncoder, wait_job, write_json
from .test_dash import THORACIC, entry, fake_ncit, trial_data, write_trial  # noqa: F401 (fixtures)

MERGED = "NCT90000001-cohort-0"
ORIGINAL_A2 = "NCT90000001-original-1"
A1 = ("Inclusion criteria:\nMetastatic non-small cell lung cancer with an EGFR exon 19 deletion.\nECOG 0-1.\n"
      "Exclusion criteria:\nPrior osimertinib.")
A2 = ("Inclusion criteria:\nMetastatic non-small cell lung cancer with an EGFR exon 19 deletion after osimertinib.\n"
      "ECOG 0-1.\nExclusion criteria:\nPrior amivantamab.")
B = "Inclusion criteria:\nPreviously treated metastatic non-small cell lung cancer, EGFR wild type.\nAdequate organ function."
NAMES = ["Cohort A1 (exon 19, first line)", "Cohort A2 (exon 19, after osimertinib)", "Cohort B (NSCLC, EGFR wild type)"]


@pytest.fixture
def managed_data(trial_data):
    """NCT90000001's protocol extraction: three extracted cohorts, the first two merged into cohort 0."""
    folder = trial_data["root"] / "protocol" / "trials" / "NCT90000001"
    write_json(folder / "extraction.json", {"trial_id": "NCT90000001", "completed_at": "2026-11-11T09:00:00+00:00",
                                            "decision": {"cancer_treatment_trial": "yes", "cohorts": [
                                                {"cohort_name": n, "eligibility_criteria": c}
                                                for n, c in zip(NAMES, (A1, A2, B))]}})
    write_json(folder / "merge.json", {"trial_id": "NCT90000001", "groups": [[0, 1], [2]], "cohort_names": NAMES})
    return trial_data


def make_state(dataset, trial_data, tmp_path, path=None):
    from matchminer_ai.apps.dashboard.settings import DashSettings
    from matchminer_ai.apps.dashboard.state import DashState

    settings = DashSettings(synthetic_run=dataset["root"], store_dir=tmp_path / "store", trial_data=trial_data["root"],
                            oa_db=trial_data["db"], good_options_catalog=trial_data["catalog"],
                            embedding_caches=(tmp_path / "hf",), embedding_python=tmp_path / "missing-python",
                            trial_management=path or tmp_path / "phi" / "management" / "trial_management.json")
    state = DashState(settings, encoder=FakeEncoder(), patient_source="training", log=lambda *_: None)
    finish(state.embed_cohorts())
    return state


def finish(job):
    while job is not None and job.status not in ("done", "error"):
        time.sleep(0.01)
    return job


@pytest.fixture
def managed(dataset, managed_data, tmp_path, fake_model, fake_ncit):  # noqa: F811
    return make_state(dataset, managed_data, tmp_path)


@pytest.fixture(autouse=True)
def no_classifier(monkeypatch):
    """No session connection classifies lines unless a test sets one (never a real endpoint)."""
    from matchminer_ai.apps.dashboard.routes import management

    monkeypatch.setattr(management, "unavailable", lambda c: None if c.backend == "openai" and c.configured and
                        "teacher.example" in c.endpoint else "No connection in tests.")


@pytest.fixture
def client(managed):
    from fastapi.testclient import TestClient

    from matchminer_ai.apps.dashboard.server import create_app

    with TestClient(create_app(managed), base_url="http://127.0.0.1") as test_client:
        yield test_client


def lines_of(view, unit_id):
    return next(u for u in view["units"] if u["id"] == unit_id)


def edited(lines, text, *, add=None):
    """The lines with the first one's text replaced, and ``add`` appended under the exclusion heading."""
    out = [dict(line) for line in lines]
    out[0] = {**out[0], "text": text, "labeled_by": None}
    if add:
        out.append({"heading": "Exclusion criteria:", "text": add, "criterion_type": "exclusion", "boilerplate": False,
                    "labeled_by": "manual"})
    return out


def wait_embedded(state, text):
    deadline = time.monotonic() + 10
    while state.cohort_vectors.vector(text) is None and time.monotonic() < deadline:
        time.sleep(0.02)
    return state.cohort_vectors.vector(text)


def test_edited_criteria_reach_every_matching_input(client, managed):
    view = client.get("/api/management/trials/NCT90000001").json()
    merged = lines_of(view, MERGED)
    assert merged["originals"] == ["NCT90000001-original-0", ORIGINAL_A2]  # cohort B stands alone: no originals
    items = edited(merged["lines"], "Metastatic NSCLC with an EGFR exon 19 deletion, untreated.",
                   add="Prior EGFR antibody therapy.")
    saved = client.put(f"/api/management/units/{MERGED}/criteria", json={"items": items}).json()
    # The changed line has no connection to classify it (none configured): saved, unclassified, and counted as key.
    assert saved["changed"] == [MERGED] and saved["classifying"] is False and saved["skipped"]
    cohort = managed.cohort(MERGED)
    assert "untreated" in cohort["key"] and "Prior EGFR antibody therapy." in cohort["key"]
    assert [i["criterion_type"] for i in cohort["items"]] == [None, "inclusion", "exclusion", "exclusion"]
    # The decision model's and the LLM's pairs, and the per-criterion check, read the edited lines ...
    pair = managed.pair("patient_a", MERGED)
    assert pair.cohort_key == cohort["key"] and pair.items == cohort["items"]
    job = client.post("/api/criteria", json={"patient_id": "patient_a", "cohort_id": MERGED}).json()["job"]
    checked = wait_job(client, job["id"])["result"]
    assert [line["text"] for line in checked["lines"]][-1] == "Prior EGFR antibody therapy."
    assert checked["unchecked_lines"] == 1  # the unclassified line
    # ... structured-data screening compiles them ...
    from matchminer_ai.apps.dashboard.guidelines import records

    assert records.cohort_lines(cohort)[3]["text"] == "Prior EGFR antibody therapy."
    # ... and Find patients retrieves with the edited key criteria's own TrialSpace vector.
    assert wait_embedded(managed, cohort["key"]) is not None
    found = client.post("/api/match/patients-for-cohort", json={"cohort_id": MERGED, "k": 2}).json()
    assert found["query"]["key"] == cohort["key"] and found["query"]["unit"]["edited"] is True
    row = next(r for g in client.get(f"/api/centers/{managed.store['trials']['NCT90000001']['center_id']}").json()["groups"]
               for r in g["rows"] if r["cohort_id"] == MERGED)
    assert "Prior EGFR antibody therapy." in row["key_exclusion"]
    # An original cohort and a subcohort can be searched too.
    original = managed.cohort(ORIGINAL_A2)
    assert [i["criterion_type"] for i in original["items"]] == [None, "inclusion", None]  # its merged cohort's labels
    sub = client.post("/api/management/subcohorts", json={"parent_id": ORIGINAL_A2, "name": "Sponsor: ex-US"}).json()
    assert managed.cohort(sub["id"])["key"] == original["key"]
    assert wait_embedded(managed, original["key"]) is not None
    assert client.post("/api/match/patients-for-cohort", json={"cohort_id": sub["id"], "k": 2}).status_code == 200


def test_classification_uses_the_pipelines_step_on_changed_lines_only(client, managed, cohort_backend):
    connection = {"task": "criteria_labels", "llm_backend": "local", "endpoint": "http://teacher.example:8000/v1",
                  "model": "fabricated-model", "tokenizer": "fabricated-model"}
    assert client.post("/api/models", json=connection).status_code == 200
    cohort_backend.script["Cohort A"] = {"deboilerplate": {"Prior EGFR antibody therapy.": [{"boilerplate": True}]}}
    merged = lines_of(client.get("/api/management/trials/NCT90000001").json(), MERGED)
    items = [*merged["lines"], {"heading": "Exclusion criteria:", "text": "Prior EGFR antibody therapy."}]
    saved = client.put(f"/api/management/units/{MERGED}/criteria", json={"items": items}).json()
    assert saved["classifying"] is True
    assert wait_job(client, saved["job"]["id"])["status"] == "done"
    assert len(cohort_backend.calls) == 1 and len(cohort_backend.calls[0][0]) == 1  # the new line only
    line = lines_of(client.get("/api/management/trials/NCT90000001").json(), MERGED)["lines"][-1]
    assert (line["criterion_type"], line["boilerplate"], line["labeled_by"]) == ("exclusion", True, "classifier")
    assert "Prior EGFR antibody therapy." in managed.cohort(MERGED)["boilerplate"]


def test_propagation_and_revert(client, managed):
    sub = client.post("/api/management/subcohorts", json={"parent_id": MERGED, "name": "Sponsor cap"}).json()["id"]
    view = client.get("/api/management/trials/NCT90000001").json()
    own = [*lines_of(view, sub)["lines"], {"heading": "Inclusion criteria:", "text": "Enrolled at a US site.",
                                           "criterion_type": "inclusion", "boilerplate": False, "labeled_by": "manual"}]
    assert client.put(f"/api/management/units/{sub}/criteria", json={"items": own}).status_code == 200
    parent = edited(lines_of(view, MERGED)["lines"], "Metastatic NSCLC, any EGFR mutation.")
    # A parent with subcohorts needs the choice.
    assert client.put(f"/api/management/units/{MERGED}/criteria", json={"items": parent}).status_code == 400
    assert client.put(f"/api/management/units/{MERGED}/criteria",
                      json={"items": parent, "propagate": True}).json()["changed"] == [MERGED, sub]
    child = lines_of(client.get("/api/management/trials/NCT90000001").json(), sub)
    assert [l["text"] for l in child["lines"]] == [l["text"] for l in parent]
    assert "Enrolled at a US site." in [l["text"] for l in child["previous_lines"]]  # shown while re-entering
    assert client.put(f"/api/management/units/{sub}/criteria", json={"items": own}).status_code == 200
    # Without propagation the subcohort keeps its own, and is not flagged.
    again = edited(parent, "Metastatic NSCLC with an EGFR mutation.")
    client.put(f"/api/management/units/{MERGED}/criteria", json={"items": again, "propagate": False})
    child = lines_of(client.get("/api/management/trials/NCT90000001").json(), sub)
    assert "Enrolled at a US site." in [l["text"] for l in child["lines"]] and child["stale"] is None
    # Revert: the AI-extracted criteria again (the edit is an overlay; the AI version was never changed).
    assert client.post(f"/api/management/units/{MERGED}/revert", json={"propagate": False}).status_code == 200
    merged = lines_of(client.get("/api/management/trials/NCT90000001").json(), MERGED)
    assert merged["edited"] is False and merged["lines"] == merged["ai_lines"]
    assert managed.cohort(MERGED)["key"] == managed.store["cohorts"][MERGED]["key"]


def test_group_pins_and_user_groups(client, managed):
    center = managed.store["trials"]["NCT90000001"]["center_id"]
    group = client.post(f"/api/management/centers/{center}/groups", json={"name": "Sponsor slots open"}).json()["id"]
    assert client.put(f"/api/management/pins/{MERGED}", json={"group_id": group}).status_code == 200
    assert client.put(f"/api/management/pins/{ORIGINAL_A2}", json={"group_id": group}).status_code == 200
    client.put(f"/api/management/units/{ORIGINAL_A2}/slots", json={"available": 2, "consented": 5, "enrolled": 3})
    groups = {g["name"]: g for g in client.get(f"/api/centers/{center}").json()["groups"]}
    rows = {r["cohort_id"]: r for r in groups["Sponsor slots open"]["rows"]}
    assert set(rows) == {MERGED, ORIGINAL_A2} and groups["Sponsor slots open"]["custom"] is True
    assert MERGED not in {r["cohort_id"] for r in groups["EGFR"]["rows"]}  # out of its rule group
    assert rows[ORIGINAL_A2]["unit"]["kind"] == "original" and rows[ORIGINAL_A2]["unit"]["parent"] == MERGED
    assert {k: rows[ORIGINAL_A2]["slots"][k] for k in ("available", "consented", "enrolled")} == {
        "available": 2, "consented": 5, "enrolled": 3}
    client.patch(f"/api/management/centers/{center}/groups/{group}", json={"name": "Open slots"})
    client.put(f"/api/management/pins/{MERGED}", json={"group_id": None})
    groups = {g["name"]: g for g in client.get(f"/api/centers/{center}").json()["groups"]}
    assert [r["cohort_id"] for r in groups["Open slots"]["rows"]] == [ORIGINAL_A2]
    assert MERGED in {r["cohort_id"] for r in groups["EGFR"]["rows"]}  # back to its rule group
    client.delete(f"/api/management/centers/{center}/groups/{group}")
    assert "Open slots" not in {g["name"] for g in client.get(f"/api/centers/{center}").json()["groups"]}


def test_a_changed_ai_version_flags_the_edit(client, managed, managed_data):
    sub = client.post("/api/management/subcohorts", json={"parent_id": "NCT90000001-cohort-1", "name": "Wild type"}).json()
    view = client.get("/api/management/trials/NCT90000001").json()
    items = edited(lines_of(view, MERGED)["lines"], "Edited first line.")
    client.put(f"/api/management/units/{MERGED}/criteria", json={"items": items})
    # The trial is re-extracted: both cohorts' AI versions change under the edit and the subcohort.
    write_trial(managed_data["root"], "protocol", "NCT90000001", [
        entry(0, "Cohort A (NSCLC, EGFR exon 19 deletion)", [("Re-extracted EGFR line.", "inclusion", False)]),
        entry(1, "Cohort B (NSCLC, EGFR wild type)", [("Re-extracted wild-type line.", "inclusion", False)])],
          at="2026-11-12T10:00:00+00:00")
    assert managed.check_store()
    view = client.get("/api/management/trials/NCT90000001").json()
    merged = lines_of(view, MERGED)
    assert merged["stale"]["reason"] == "ai_changed" and merged["lines"][0]["text"] == "Edited first line."
    assert managed.cohort(MERGED)["items"][0]["text"] == "Edited first line."  # kept in effect, and flagged
    assert lines_of(view, sub["id"])["stale"]["reason"] == "parent_changed"
    assert client.post(f"/api/management/units/{MERGED}/keep", json={}).status_code == 200
    assert lines_of(client.get("/api/management/trials/NCT90000001").json(), MERGED)["stale"] is None


def test_the_file_round_trip_export_and_import(client, managed, dataset, managed_data, tmp_path):
    path = managed.management.file.path
    client.put(f"/api/management/units/{MERGED}/slots", json={"available": 4, "consented": 1, "enrolled": 0})
    sub = client.post("/api/management/subcohorts", json={"parent_id": MERGED, "name": "Sponsor cap"}).json()["id"]
    assert oct(path.stat().st_mode & 0o777) == "0o600" and oct(path.parent.stat().st_mode & 0o777) == "0o700"
    # A second process (or a restart) reads the same file.
    again = make_state(dataset, managed_data, tmp_path / "second", path)
    assert again.cohort(MERGED)["management"]["slots"]["available"] == 4 and again.cohort(sub)["name"] == "Sponsor cap"
    exported = client.get("/api/management/export")
    assert exported.headers["content-disposition"].startswith("attachment")
    doc = json.loads(exported.content)
    assert doc["format"] == "mmai-dash-trial-management-v1" and [e["action"] for e in doc["log"]] == [
        "slots", "subcohort created"]
    client.delete(f"/api/management/units/{sub}")
    files = {"file": ("tm.json", exported.content, "application/json")}
    assert client.post("/api/management/import", files=files).status_code == 400  # not confirmed
    assert client.post("/api/management/import", files=files, data={"confirm": "replace"}).status_code == 200
    assert managed.cohort(sub)["name"] == "Sponsor cap"
    assert path.with_name("trial_management.before-import.json").exists()
    other = {**doc, "mode": "synthetic"}
    refused = client.post("/api/management/import", data={"confirm": "replace"},
                          files={"file": ("tm.json", json.dumps(other).encode(), "application/json")})
    assert refused.status_code == 400 and "synthetic" in refused.json()["error"]
    # A change based on an old revision is refused rather than overwriting someone else's.
    unit = lines_of(client.get("/api/management/trials/NCT90000001").json(), MERGED)
    client.put(f"/api/management/units/{MERGED}/slots", json={"available": 5, "revision": unit["revision"]})
    stale = client.put(f"/api/management/units/{MERGED}/slots", json={"available": 6, "revision": unit["revision"]})
    assert stale.status_code == 409


def test_synthetic_mode_never_uses_a_phi_folder(dataset, tmp_path):
    from matchminer_ai.apps.dashboard.settings import DashSettings

    with pytest.raises(ValueError, match="phi"):
        DashSettings(mode="synthetic", synthetic_run=dataset["root"], store_dir=tmp_path / "no_phi" / "store",
                     trial_management=tmp_path / "phi" / "trial_management.json")
