"""Real mode's patient corpus: a preprocessed population cache (``guidelines.population_cache``) as the patients the
Trials tab searches and the Patient tab picks from.

``PopulationCorpus`` gives the cache's summaries the patient side of ``Corpus`` (rows with ``id``, ``text``: the
cancer history section, ``boilerplate``: the other relevant history; parsed headers; regex search), ``CacheVectors``
its TrialSpace vectors in the round's search, and ``CacheNotes`` each patient's source notes, read on demand. The cache
holds PHI; nothing here writes it anywhere.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .corpus import Corpus, parse_header
from .vectors import VectorStore


class PopulationCorpus(Corpus):
    """The patient side of ``Corpus`` over a population cache (no cohorts, no generation metadata)."""

    def __init__(self, cache=None, error: str | None = None):  # noqa: D107 (Corpus loads files; this takes rows)
        started = time.monotonic()
        self.population = cache
        self.error = error
        self.directory = cache.path if cache is not None else None
        self.receipt = None
        frame = cache.summaries if cache is not None else None
        self.patients = [] if frame is None else [
            {"id": str(pid), "text": str(text or ""), "boilerplate": str(other or "")}
            for pid, text, other in zip(frame.patient_id, frame.cancer_history_summary,
                                        frame.general_exclusion_criteria_evidence)]
        self.info = ({"source": "population_cache", "name": cache.path.name, "created_utc": cache.manifest.get("created_utc"),
                      "patients": len(self.patients)} if cache is not None else {"source": None, "error": error})
        self.cohorts = []
        self.patient_index = {row["id"]: i for i, row in enumerate(self.patients)}
        self.cohort_index = {}
        self.headers = [parse_header(row["text"]) for row in self.patients]
        self.meta = {}
        self.trial_cohorts = {}
        self.from_cohort = {}
        self.digests = {}
        self.load_seconds = round(time.monotonic() - started, 1)
        self._search_lock = threading.Lock()
        self._text_indexes = {}
        self._patient_columns = None

    def warm_search(self, titles: dict | None = None) -> None:
        self._text_index("patients", "text")

    def describe(self) -> dict:
        return {"directory": str(self.directory) if self.directory else None, "patients": len(self.patients),
                "cohorts": 0, "source": "population_cache", "error": self.error, "info": self.info,
                "load_seconds": self.load_seconds}


class CacheVectors(VectorStore):
    """The cache's patient vectors, searched as the round's are (cohorts come from ``vectors.CohortVectors``)."""

    def __init__(self, patients: np.ndarray):  # noqa: D107
        self.info = None
        self.patients = np.asarray(patients, dtype=np.float32)
        self.cohorts = np.zeros((0, self.patients.shape[1] if self.patients.ndim == 2 else 0), dtype=np.float32)


class CacheNotes:
    """A cached patient's source notes, read on demand (the cache sorts them by patient in small row groups)."""

    def __init__(self, cache=None):
        self.cache = cache
        self.state = "ready" if cache is not None else "none"
        self.error = None

    def ensure(self) -> str:
        return self.state

    def notes(self, pid: str) -> list[dict]:
        if self.cache is None:
            return []
        frame = self.cache.notes([pid])
        rows = frame.to_dict("records")
        rows.sort(key=lambda r: str(r.get("note_date") or ""))
        return [{"date": None if r.get("note_date") is None else str(r.get("note_date")), "type": r.get("note_type"),
                 "text": r.get("note_text") or ""} for r in rows]

    def describe(self) -> dict:
        return {"path": str(self.cache.path / "notes.parquet") if self.cache is not None else None,
                "state": self.state, "error": self.error, "patients": None}
