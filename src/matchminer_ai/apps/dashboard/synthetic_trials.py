"""Synthetic mode's trial store: the synthetic run's trials and their cohorts, in the store format of ``pipeline``.

Sources (all read-only, under the synthetic run):

- the TrialSpace corpus's ``cohorts.jsonl``: every cohort of the run's trials, from the synthetic protocol and
  from the ClinicalTrials.gov listing (``source``), with its key criteria (``text``), boilerplate criteria and
  tagged criteria lines;
- ``trials/ctgov_sample_trials.csv``: each trial's phases and recruitment status;
- ``trials/ctgov_cohorts/trials_with_cohorts.csv``: its title and summaries.

The run's trials have no disease centers, so a trial's "center" is the cancer type of its shown cohorts (the
first specific type, else "Multiple tumor types"); ``grouping`` then groups each center's cohorts as it does for a
center without rules. Each trial shows its protocol cohorts when it has them, else its ct.gov cohorts, as in real
mode. There are no drugs unless the mode's site config names a GoodOption catalog.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

from matchminer_ai.site import load_site_config

from . import drugs as drug_catalog
from . import grouping
from .core import cancer_types
from .core.cache import file_stamp
from .core.corpus import cohort_number
from .pipeline import FORMAT, VERSIONS, center_id, counts, dashboard_state, phase_label

PIPELINE_VERSION = "synthetic-2026-11-21.1"
SAMPLE = "ctgov_sample_trials.csv"
LISTINGS = "ctgov_cohorts/trials_with_cohorts.csv"


def _phase(value) -> str:
    """``["PHASE1", "PHASE2"]`` as "Phase 1/2"."""
    try:
        phases = json.loads(value) if isinstance(value, str) else list(value or [])
    except ValueError:
        phases = [value]
    numbers = [str(p).upper().replace("PHASE", "").replace("EARLY_", "Early ").strip() for p in phases if p]
    return phase_label("/".join(n for n in numbers if n)) if numbers else ""


def _text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


@dataclass
class SyntheticSources:
    corpus: Path  # the TrialSpace corpus directory (cohorts.jsonl)
    trials: Path  # the run's trial folder (ctgov_sample_trials.csv, ctgov_cohorts/)
    grouping: Path
    good_options: Path | None = None

    def signature(self) -> dict:
        """What the store was built from: file stamps (size and mtime), never content hashes of data files."""
        stamps = {"cohorts": file_stamp(self.corpus / "cohorts.jsonl"), "sample": file_stamp(self.trials / SAMPLE),
                  "listings": file_stamp(self.trials / LISTINGS), "grouping": file_stamp(self.grouping),
                  "pipeline": PIPELINE_VERSION}
        site = load_site_config()
        if site.path is not None:
            stamps["site_config"] = file_stamp(site.path)
        if self.good_options is not None:
            stamps["good_options"] = str(self.good_options)
            for name, stamp in drug_catalog.catalog_stamps(self.good_options).items():
                stamps[f"good_options/{name}"] = stamp
        return stamps


@dataclass
class SyntheticBuilder:
    sources: SyntheticSources
    _catalog: tuple | None = field(default=None, repr=False)

    def catalog(self) -> tuple[dict[str, list[dict]], dict]:
        directory = self.sources.good_options
        key = (str(directory), drug_catalog.catalog_stamps(directory) if directory is not None else None)
        if self._catalog is None or self._catalog[0] != key:
            self._catalog = (key, drug_catalog.read_catalog(directory))
        return self._catalog[1]

    def trial_rows(self) -> dict[str, dict]:
        import pandas as pd

        rows: dict[str, dict] = {}
        sample = self.sources.trials / SAMPLE
        if sample.is_file():
            for row in pd.read_csv(sample, dtype=str, keep_default_na=False).to_dict("records"):
                rows.setdefault(row["nct_id"], {}).update(row)
        listings = self.sources.trials / LISTINGS
        if listings.is_file():
            frame = pd.read_csv(listings, dtype=str, keep_default_na=False,
                                usecols=lambda c: c in ("nct_id", "title", "brief_summary", "detailed_description"))
            for row in frame.to_dict("records"):
                rows.setdefault(row["nct_id"], {}).update(row)
        return rows

    def build(self) -> dict:
        from matchminer_ai.labeling.corpus import load_corpus

        started = time.monotonic()
        signature = self.sources.signature()
        rules = grouping.load_rules(self.sources.grouping)
        _, corpus_cohorts = load_corpus(self.sources.corpus)
        meta = self.trial_rows()
        trial_drugs, catalog = self.catalog()
        by_trial: dict[str, dict[str, list[dict]]] = {}
        for row in corpus_cohorts:
            version = row.get("source") if row.get("source") in VERSIONS else "ctgov"
            by_trial.setdefault(row["trial_id"], {}).setdefault(version, []).append(row)
        trials, cohorts, centers = {}, {}, {}
        for trial_id in sorted(by_trial):
            info = meta.get(trial_id, {})
            status = _text(info.get("overall_status"))
            primary = next((v for v in VERSIONS if by_trial[trial_id].get(v)), None)
            trial = {
                "id": trial_id, "nct_id": trial_id, "protocol_no": "", "short_title": _text(info.get("title")),
                "long_title": "", "official_title": "", "phase": _phase(info.get("phases")),
                "status": status.replace("_", " ").title(), "is_open": status in ("RECRUITING", "NOT_YET_RECRUITING"),
                "site_pi": "", "overall_pi": "", "contact": {"name": "", "email": "", "role": ""}, "sites": [],
                "brief_summary": _text(info.get("brief_summary")),
                "detailed_description": _text(info.get("detailed_description")),
                "primary_version": primary, "versions": {}, "cohort_ids": [], "drugs": trial_drugs.get(trial_id, []),
            }
            for version in VERSIONS:
                ids = []
                rows = sorted(by_trial[trial_id].get(version, []), key=lambda r: cohort_number(r["id"]) or 0)
                for row in rows:
                    cohort = {"id": row["id"], "trial_id": trial_id, "version": version, "primary": version == primary,
                              "number": cohort_number(row["id"]) or 0, "name": row.get("cohort_name") or "",
                              "merged_from": [], "key": row.get("text") or "", "boilerplate": row.get("boilerplate") or "",
                              "items": row.get("items") or []}
                    cohort["cancer_types"] = grouping.cohort_cancer_types(cohort, trial)
                    cohorts[row["id"]] = cohort
                    ids.append(row["id"])
                trial["versions"][version] = {"state": "ready" if ids else "not_run", "completed_at": None,
                                              "screen": None, "extracted_cohorts": len(ids), "excluded_reason": None,
                                              "cohort_ids": ids}
            trial["cohort_ids"] = trial["versions"][primary]["cohort_ids"] if primary else []
            trial["state"] = dashboard_state(trial)
            types = [t for c in trial["cohort_ids"] for t in cohorts[c]["cancer_types"]]
            specific = [t for t in types if t in grouping.specific_types(types)]
            center = cancer_types.TYPE_LABELS.get(specific[0], "Multiple tumor types") if specific else "Multiple tumor types"
            trial.update(center=center, center_id=center_id(center), center_short_name=center)
            trials[trial_id] = trial
            centers.setdefault(trial["center_id"], {"id": trial["center_id"], "name": center, "short_name": center,
                                                    "trial_ids": []})["trial_ids"].append(trial_id)
        for center in centers.values():
            grouping.assign_groups(center, trials, cohorts, rules)
        ordered = sorted(centers.values(), key=lambda c: (-len(c["trial_ids"]), c["name"]))
        return {"format": FORMAT, "built_at": time.time(), "build_seconds": round(time.monotonic() - started, 2),
                "signature": signature, "oa_db": None, "source": "synthetic",
                "rules": {"path": str(self.sources.grouping), "centers": rules.describe()},
                "centers": ordered, "trials": trials, "cohorts": cohorts, "higher_threshold_groups": [],
                "good_options": {**catalog, "shown": drug_catalog.shown_counts(trials)},
                "counts": counts(trials, cohorts)}
