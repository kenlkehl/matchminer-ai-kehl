"""The extracted ("original") cohorts behind each merged dashboard cohort, read-only.

Cohort extraction lists a trial's cohorts with their complete criteria; the merge step makes the cohorts with
essentially the same criteria one row (keeping the first cohort's criteria unchanged), and the boilerplate step tags
that row's lines, so a dashboard cohort may stand for several extracted cohorts. Where each mode keeps them:

- **real** (``trial_data``): a protocol's ``protocol/trials/<NCT>/extraction.json`` (``decision.cohorts``) and
  ``merge.json`` (``groups``, zero-based, one per merged cohort); a ClinicalTrials.gov listing's
  ``ctgov/trials/<NCT>/merge.json`` and the listing run's ``ctgov/run_metadata.json`` (the run's extraction
  metadata, ``extraction.trial_decisions``: listings keep no per-trial extraction file). A trial whose merge does not
  name its extraction's current cohorts (a newer extraction on its way) has none until it is merged.
- **synthetic** (the run's ``trials`` folder): ``synthetic_protocols/cohort_extraction/cohorts_unmerged.parquet``
  with ``cohorts.parquet`` (``merged_from``), and ``ctgov_cohorts/trial_cohort_lineitems_unmerged.csv`` with
  ``trial_cohort_lineitems.csv``. Only the cohorts of merged groups of two or more are kept in memory.

Files are read when first needed and again only when their stamps change (never hashed).
"""
from __future__ import annotations

import json
from pathlib import Path
import threading

from .core.cache import file_stamp

PROTOCOL_TABLES = ("synthetic_protocols/cohort_extraction/cohorts_unmerged.parquet",
                   "synthetic_protocols/cohort_extraction/cohorts.parquet")
CTGOV_TABLES = ("ctgov_cohorts/trial_cohort_lineitems_unmerged.csv", "ctgov_cohorts/trial_cohort_lineitems.csv")


def _members(value, number: int) -> list[int]:
    """A merged row's ``merged_from`` (a JSON list, or missing for a cohort that was not merged)."""
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return [number]
        return [int(n) for n in parsed] if isinstance(parsed, list) and parsed else [number]
    if isinstance(value, (list, tuple)) and value:
        return [int(n) for n in value]
    return [number]


class OriginalCohorts:
    """``groups(trial_id, version)``: ``({merged number: [extracted indexes]}, {index: {"name", "criteria"}})`` for the
    merged cohorts standing for two or more extracted cohorts, or a string saying why there are none."""

    def __init__(self, settings):
        self.settings = settings
        self.lock = threading.Lock()
        self._files: dict[Path, tuple[str, object]] = {}

    def _cached(self, path: Path, read, *also: Path):
        stamps = [file_stamp(p) for p in (path, *also)]
        if None in stamps:
            return None
        stamp = "|".join(stamps)
        with self.lock:
            cached = self._files.get(path)
            if cached and cached[0] == stamp:
                return cached[1]
        try:
            value = read(path)
        except (OSError, ValueError, KeyError, TypeError):  # being written: tried again next time
            return cached[1] if cached else None
        with self.lock:
            self._files[path] = (stamp, value)
        return value

    def groups(self, trial_id: str, version: str):
        if self.settings.real:
            return self._real(trial_id, version)
        return self._synthetic(trial_id, version)

    # ------------------------------------------------------------------ real mode

    def _real(self, trial_id: str, version: str):
        root = Path(self.settings.trial_data)
        folder = root / version / "trials" / trial_id
        merge = self._cached(folder / "merge.json", lambda p: json.loads(p.read_text(encoding="utf-8")))
        if version == "protocol":
            extraction = self._cached(folder / "extraction.json", lambda p: json.loads(p.read_text(encoding="utf-8")))
            cohorts = ((extraction or {}).get("decision") or {}).get("cohorts") if isinstance(extraction, dict) else None
        else:
            decisions = self._cached(root / "ctgov" / "run_metadata.json", _listing_decisions) or {}
            cohorts = decisions.get(trial_id)
        if not cohorts:
            return "The extraction's cohorts are not available for this trial."
        if not isinstance(merge, dict) or not merge.get("groups"):
            return {}, {}  # one cohort, or none merged
        if merge.get("cohort_names") != [c.get("cohort_name") for c in cohorts]:
            return "The trial's newer extraction has not been merged yet."
        groups = {n: [int(i) for i in g] for n, g in enumerate(merge["groups"]) if len(g) > 1}
        members = {i: {"name": cohorts[i].get("cohort_name") or f"Cohort {i + 1}",
                       "criteria": cohorts[i].get("eligibility_criteria") or ""}
                   for g in groups.values() for i in g if i < len(cohorts)}
        return groups, members

    # ------------------------------------------------------------------ synthetic mode

    def _synthetic(self, trial_id: str, version: str):
        root = Path(self.settings.path("trials"))
        unmerged, merged = PROTOCOL_TABLES if version == "protocol" else CTGOV_TABLES
        tables = self._cached(root / unmerged, lambda _: _synthetic_tables(root / unmerged, root / merged), root / merged)
        if tables is None:
            return "The synthetic run's extracted cohorts are not available."
        groups = tables.get(trial_id, ({}, {}))
        return groups


def _listing_decisions(path: Path) -> dict:
    """The listing run's extracted cohorts by trial (its other metadata is dropped)."""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    decisions = (metadata.get("extraction") or {}).get("trial_decisions") or {}
    return {t: [{"cohort_name": c.get("cohort_name"), "eligibility_criteria": c.get("eligibility_criteria")}
                for c in d.get("cohorts") or []] for t, d in decisions.items() if isinstance(d, dict)}


def _synthetic_tables(unmerged: Path, merged: Path) -> dict:
    """{trial: ({merged number: [indexes]}, {index: {"name", "criteria"}})}, for merged groups of two or more."""
    import pandas as pd

    def read(path: Path, columns: list[str]) -> pd.DataFrame:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            return pq.read_table(path, columns=columns).to_pandas()
        return pd.read_csv(path, usecols=columns, dtype={"trial_id": str}, keep_default_na=True)

    rows = read(merged, ["trial_id", "cohort_number", "merged_from"])
    out: dict[str, tuple[dict, dict]] = {}
    for trial_id, number, value in zip(rows.trial_id, rows.cohort_number, rows.merged_from):
        members = _members(value, int(number))
        if len(members) > 1:
            out.setdefault(str(trial_id), ({}, {}))[0][int(number)] = members
    wanted = {(t, i) for t, (groups, _) in out.items() for g in groups.values() for i in g}
    originals = read(unmerged, ["trial_id", "cohort_number", "cohort_name", "cohort_eligibility_criteria"])
    for trial_id, number, name, criteria in zip(originals.trial_id, originals.cohort_number, originals.cohort_name,
                                                originals.cohort_eligibility_criteria):
        key = (str(trial_id), int(number))
        if key in wanted:
            out[key[0]][1][key[1]] = {"name": str(name) if isinstance(name, str) else f"Cohort {key[1] + 1}",
                                      "criteria": criteria if isinstance(criteria, str) else ""}
    return out
