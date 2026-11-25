"""Labels for edited and new criteria lines: inclusion or exclusion, key or common (boilerplate).

The cohort pipeline's own classification step labels them (``matchminer_ai.trials.remove_boilerplate_criteria``:
its prompt, one request per line showing the cohort's whole numbered criteria, its validation and retries), so they
match the AI-extracted cohorts' labels. Lines whose labels are known are passed as ``known_items`` and not asked.
The requests go to the browser session's connection for the ``criteria_labels`` task (Data and models; Claude
Haiku 5.5 through the gateway at medium effort by default), so the cohorts' criteria reach that endpoint.
"""
from __future__ import annotations

from .llm import Connection

TASK = "criteria_labels"
SECTION = "trial"


def unavailable(connection: Connection) -> str | None:
    if not connection.configured:
        return ("No LLM connection is ready for criteria classification: start the local model server or choose a "
                "connection on Data and models.")
    if connection.vertex:
        from matchminer_ai.llm.connections import vertex_credentials_error

        return vertex_credentials_error()
    return None


def configure(connection: Connection, metadata_dir) -> object:
    """The package preset with its ``trial`` section on ``connection``."""
    from matchminer_ai import load_default_preset

    config = connection.configure(load_default_preset(), (SECTION,))
    config.debug_mode = False
    config.model_metadata_cache_dir = str(metadata_dir)
    section = config.trial
    # The step reads its structured-output switch from the section's cohort settings; the connection sets the
    # section's (off for hosted Vertex models, which answer free JSON the package validates).
    if "structured_outputs" in section:
        section["cohorts"] = {**section.get("cohorts", {}), "structured_outputs": section["structured_outputs"]}
    return config


def classify(cohorts: list[dict], config) -> list[list[dict | None]]:
    """For each cohort (``trial_id``, ``name``, ``items``: lines with ``heading``, ``text`` and, when known,
    ``criterion_type`` and ``boilerplate``), every line's labels in order: ``{"criterion_type", "boilerplate"}``, or
    None for a line whose answer never validated."""
    import pandas as pd

    from matchminer_ai.trials.cohort_boilerplate import ITEMS_COLUMN, load_criteria_items, remove_boilerplate_criteria
    from matchminer_ai.trials.cohort_run import criteria_from_items

    rows, known = [], []
    for number, cohort in enumerate(cohorts):
        lines = [{"heading": i["heading"], "text": i["text"], "boilerplate": False} for i in cohort["items"]]
        rows.append({"trial_id": cohort["trial_id"], "cohort_number": number, "cohort_name": cohort["name"] or "Cohort",
                     "cohort_trial_id": f"{cohort['trial_id']}-edited-{number}",
                     "cohort_eligibility_criteria": criteria_from_items(lines)})
        known.append({n: {"criterion_type": i["criterion_type"], "boilerplate": bool(i["boilerplate"])}
                      for n, i in enumerate(cohort["items"], 1) if i.get("criterion_type") in ("inclusion", "exclusion")})
    result = remove_boilerplate_criteria(pd.DataFrame(rows), config=config, section=SECTION, known_items=known)
    out = []
    for cohort, value in zip(cohorts, result[ITEMS_COLUMN]):
        tagged = load_criteria_items(value)
        if [(t["heading"], t["text"]) for t in tagged] != [(i["heading"], i["text"]) for i in cohort["items"]]:
            raise RuntimeError("the classified lines do not match the cohort's lines")
        out.append([{"criterion_type": t["criterion_type"], "boilerplate": bool(t["boilerplate"])}
                    if t.get("criterion_type") in ("inclusion", "exclusion") else None for t in tagged])
    return out
