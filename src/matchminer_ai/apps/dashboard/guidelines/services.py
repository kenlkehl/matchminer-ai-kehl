"""Thin adapters over the public MatchMiner-AI APIs."""

import copy
import hashlib
import io
import json
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from matchminer_ai.patients.workup import is_history_and_physical
from matchminer_ai.trials.guideline_catalog import latest_guideline_catalogs, source_label

from ..core.errors import InputError  # noqa: F401  (raised here; imported from here by the routes)


class Catalogs:
    """Discover complete editions; validate their export hashes in the public loader."""

    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.rows = []
        self.expires = 0

    def list(self, refresh=False):
        with self.lock:
            if not refresh and time.monotonic() < self.expires:
                return list(self.rows)
            # One completed catalog per disease, the latest run (the package's discovery rule).
            self.rows = [
                {"id": hashlib.sha256(str(Path(row["path"]).relative_to(self.root)).encode()).hexdigest()[:20],
                 **row}
                for row in latest_guideline_catalogs(self.root)
            ]
            self.expires = time.monotonic() + 60
            return list(self.rows)

    def resolve(self, selected):
        rows = self.list()
        available = {r["id"]: r for r in rows}
        if selected and any(key not in available for key in selected):
            raise InputError("A selected catalog is no longer available. Refresh the library.")
        paths = (
            [available[key]["path"] for key in selected] if selected else [r["path"] for r in rows]
        )
        if not paths:
            raise InputError("No completed catalogs are available yet.")
        return paths


def pdf_pages(content):
    """Each page's text, through the package's PDF reader (``matchminer_ai.documents.ocr_pdf``: the embedded text
    layer, else local RapidOCR; nothing leaves this machine). The upload is written to a private temporary folder
    that is removed afterwards."""
    from tempfile import TemporaryDirectory

    from matchminer_ai.documents import PDFOCRError, PDFOCRNoTextError, ocr_pdf
    from matchminer_ai.documents.ocr import _PAGE_SEPARATOR

    with TemporaryDirectory(prefix="guideline-fetcher-pdf-") as folder:
        source = Path(folder) / "upload.pdf"
        source.write_bytes(content)
        try:
            text = ocr_pdf(source, Path(folder) / "upload.txt").read_text(encoding="utf-8")
        except PDFOCRNoTextError:
            raise InputError("PDF page 1 has no readable text, even with OCR; no pages were imported.") from None
        except PDFOCRError:
            raise InputError("This PDF could not be opened. If it is password protected, upload an unlocked "
                             "copy.") from None
    return [page.strip() for page in text.rstrip("\n").split(_PAGE_SEPARATOR)]


def parse_upload(name, content):
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        pages = pdf_pages(content)
        for i, text in enumerate(pages, 1):
            if not text:
                raise InputError(
                    f"PDF page {i} has no readable text, even with OCR; no pages were imported."
                )
        blocks = [f"[PDF page {i}; note date unavailable]\n{text}" for i, text in enumerate(pages, 1)]
        return {
            "text": "\n\n".join(blocks),
            "notes": None,
            "notice": "PDF order preserved. Dates were not inferred from file metadata. Pages without a text "
                      "layer were read with local OCR; verify tables, layout and OCR text.",
        }
    if suffix in {".txt", ".md"}:
        return {
            "text": content.decode("utf-8-sig"),
            "notes": None,
            "notice": "Free text imported. Structured note dates are unavailable.",
        }
    if suffix != ".csv":
        raise InputError("Choose a PDF, CSV, TXT, or Markdown file.")
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    frame = frame.rename(columns={
        "text": "note_text", "date": "note_date", "type": "note_type", "pseudo_mrn": "patient_id"
    })
    if frame.columns.duplicated().any():
        raise InputError(
            "CSV columns are ambiguous; provide one note_text and one note_date column."
        )
    if "note_text" not in frame:
        raise InputError("CSV needs a note_text column; note_date and patient_id are optional.")
    if "patient_id" in frame and frame.patient_id.nunique() != 1:
        raise InputError("Upload notes for exactly one patient.")
    if "note_date" not in frame:
        frame["note_date"] = ""
    supplied_dates = frame.note_date.str.strip()
    dates = pd.to_datetime(supplied_dates, errors="coerce", format="mixed")
    if (dates.isna() & supplied_dates.ne("")).any():
        raise InputError(
            "Some note dates are invalid. Use YYYY-MM-DD, or leave unknown dates blank."
        )
    frame["note_date"] = dates
    frame = frame.loc[frame.note_text.str.strip().ne("")].sort_values("note_date", kind="stable")
    if frame.empty:
        raise InputError("The file contains no nonempty notes.")
    records = [
        {
            "note_text": r.note_text,
            "note_date": None if pd.isna(r.note_date) else r.note_date.isoformat(),
            **({"note_type": r.note_type} if "note_type" in frame and r.note_type.strip() else {}),
        }
        for r in frame.itertuples()
    ]
    text = "\n\n".join(
        f"[Note date: {r['note_date'] or 'unavailable'}]\n{r['note_text']}" for r in records
    )
    return {
        "text": text,
        "notes": records,
        "notice": f"{len(records)} notes imported in date order. Undated notes follow dated notes.",
    }


def display_considerations(frame):
    """Clarify absent recommendation categories in the UI/report, preserving source data, and name each space's
    source as the views show it (``source.label``: ``guideline_catalog.source_label``).

    History-and-physical workup items are left out (``is_history_and_physical``).
    """
    rows = copy.deepcopy(frame.to_dict("records"))
    for row in rows:
        if isinstance(row.get("source"), dict):  # what the views show as the source ("Derived from PDQ®: ...")
            row["source"]["label"] = source_label(row["source"])
        row["diagnostic_workup"] = [item for item in row["diagnostic_workup"]
                                    if not is_history_and_physical(item)]
        for menu in ("diagnostic_workup", "treatment_options"):
            for option in row[menu]:
                category = " ".join(option["category"].casefold().rstrip(".").split())
                if category in {
                    "not specified in supplied evidence",
                    "not specified in the supplied evidence",
                }:
                    option["category"] = "Recommendation category not specified"
    return pd.DataFrame(rows, columns=frame.columns)


def synthetic_note_sources(settings):
    """Expose named synthetic samples, without accepting client-supplied file paths. Synthetic mode only: real mode
    works on real records."""
    if settings.phi:
        return []
    sources = []
    if settings.biliary_synthetic_notes.is_file():
        sources.append({"id": "biliary", "label": "Biliary cancer · cholangiocarcinoma"})
    sources.append({
        "id": "demo",
        "label": "Configured synthetic case" if settings.synthetic_notes
                 else "Small-cell lung cancer · HF demo",
    })
    return sources


def synthetic_notes(settings, source="demo"):
    """Load a named external synthetic CSV, retaining raw notes and their dates."""
    sources = {item["id"]: item for item in synthetic_note_sources(settings)}
    if source not in sources:
        raise InputError("That synthetic case is unavailable. Refresh the page and choose a case.")
    if source == "biliary":
        path = settings.biliary_synthetic_notes
    elif settings.synthetic_notes:
        path = Path(settings.synthetic_notes)
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(
            repo_id="ksg-dfci/mmai-synthetic", repo_type="dataset",
            filename="sample_patient_notes.csv",
            revision="b1845aaed70b8106868e0369c4c013ad24b48e72",
        ))
    if path.stat().st_size > settings.max_upload:
        raise InputError("Synthetic sample exceeds the upload limit.")
    result = parse_upload("synthetic.csv", path.read_bytes())
    result["source"] = source
    result["label"] = sources[source]["label"]
    result["notice"] = f"Synthetic patient: {result['label']}. " + result["notice"]
    return result


class Inference:
    def __init__(self, settings, *, workup_backups=None, extractions=None):
        self.settings = settings
        self.workup_backups = workup_backups or {}
        # (SessionExtractions, owner, StoreHandle): this session's on-demand extraction records and the mode's
        # extraction store.
        self.extractions = extractions

    def _copy(self, **changes):
        values = dict(settings=self.settings, workup_backups=self.workup_backups, extractions=self.extractions)
        values.update(changes)
        settings = values.pop("settings")
        return type(self)(settings, **values)

    def with_settings(self, settings):
        """Bind a job to an immutable choice of workspace settings."""
        return self._copy(settings=settings)

    def with_workup_backups(self, *, full, agent):
        return self._copy(workup_backups={"full_notes": full, "agentic": agent})

    def with_extractions(self, sessions, owner, records=None):
        return self._copy(extractions=(sessions, owner, records))

    def extraction_cost(self, text, notes, space, levels, *, method="extractions", patient_id=None):
        """What an extraction-based review of this patient would send (no model calls)."""
        from .extraction_workup import (
            LEVELS, SessionExtractions, _stored_records, cached_query_count, cached_sql_count, estimate_cost,
            exploration_store, notes_frame,
        )

        if levels not in LEVELS:
            raise InputError("Choose document-level or document and patient-level extraction.")
        config = self.settings.extraction_config()
        frame = notes_frame(text, notes)
        sql = {}
        if method == "extraction_sql":
            from matchminer_ai.patients.workup_sql import agent_limits

            store = exploration_store(self.settings)
            distinct, cached, built = cached_sql_count(space, self.settings, config, store)
            actions = agent_limits(config).max_actions
            compile_requests = (distinct - cached) * actions
            sql = {"sql_agent": {
                "store": store.path.name,
                "patients": store.manifest["counts"].get("patients", 0),
                "database_built": built,
                "items_to_explore": distinct - cached,
                "max_actions_per_item": actions,
                "model": config.workup_sql_agent["remote"]["model_name"],
            }}
        else:
            distinct, cached = cached_query_count(space, self.settings, config)
            compile_requests = distinct - cached
        sessions, owner, records = self.extractions or (None, None, None)
        stored = _stored_records(records, patient_id, text, notes)
        saved = stored is not None or (sessions is not None and sessions.get(
            owner, SessionExtractions.key(frame, config), levels) is not None)
        cost = estimate_cost(frame, levels, compile_requests=compile_requests,
                             cached_queries=cached, items=len(space["diagnostic_workup"]))
        if saved:
            cost.update(document_requests=0, patient_requests=0,
                        requests=compile_requests, reused_extraction=True)
        return {**cost, **sql, "method": method, "reused_extraction": saved,
                "records_source": "store" if stored is not None else "session" if saved else None,
                "model": config.document_extraction["remote"]["model_name"]}

    def summarize(self, text, notes, progress):
        from matchminer_ai.patients import summarize_patients

        progress("Preparing notes for serial summarization")
        config = self.settings.inference_config()
        frame = pd.DataFrame(notes or [{"note_text": text, "note_date": None}])
        frame["patient_id"] = "workspace-patient"
        # NaT stays unavailable; never substitute a made-up clinical date.
        frame["note_date"] = pd.to_datetime(frame["note_date"], errors="coerce", format="mixed")
        progress("Summarizing the cancer history · this may take several minutes")
        result = summarize_patients(frame, config=config)
        if len(result) != 1 or not str(result.iloc[0].cancer_history_summary).strip():
            from matchminer_ai.llm.openrouter import hosted_stop

            # The hosted OpenRouter key's quota (its own fixed text, which says what to do), else the generic text.
            raise InputError(hosted_stop(self.settings.api_key_ref)
                             or "The summarizer did not return one usable patient summary.")
        return {
            "summary": str(result.iloc[0].cancer_history_summary),
            "exclusion_evidence": str(
                result.iloc[0].get("general_exclusion_criteria_evidence", "")
            ),
        }

    def review_workup(
        self, text, notes, space, progress, *, method="full_notes",
        patient_summary=None, extraction_levels="document", patient_id=None,
    ):
        from matchminer_ai.patients import (
            review_patient_workup,
            review_patient_workup_with_note_search,
        )

        if method in {"extractions", "extraction_sql"}:
            from .extraction_workup import patient_review

            sessions, owner, records = self.extractions or (None, None, None)
            return patient_review(
                text, notes, space, settings=self.settings,
                config=self.settings.extraction_config(), sessions=sessions, owner=owner,
                levels=extraction_levels, progress=progress, records=records, patient_id=patient_id,
                method="extraction_sql" if method == "extraction_sql" else "extraction_queries",
            )
        config = self.settings.inference_config()
        options = {}
        if method not in {"full_notes", "agentic"}:
            raise InputError("Choose full-note review, agentic search, extraction queries or the "
                             "SQL agent.")
        raw = pd.DataFrame(notes) if notes else text
        if method == "agentic":
            review = review_patient_workup_with_note_search
            # All workup questions are eligible to run at once in this job.
            # Summarization and full-note jobs retain their one-request limit.
            config.remote["max_concurrent_requests"] = len(space["diagnostic_workup"])
            options.update(
                history=text if notes else None,
                patient_summary=patient_summary,
                # Inherit this task's known-model switch; avoid the public
                # agent's off default overriding the user's thinking choice.
                thinking=None,
            )
        else:
            review = review_patient_workup
        backup = self.workup_backups.get(method)
        if backup is not None and backup.backup_enabled:
            backup_config = backup.inference_config()
            if method == "agentic":
                backup_config.remote["max_concurrent_requests"] = len(space["diagnostic_workup"])
            options.update(backup_config=backup_config,
                           max_consecutive_failures=backup.failure_threshold)
        return review(
            raw, space["diagnostic_workup"], config=config,
            progress_callback=progress,
            population_context=space.get("cohort_key_criteria") or space.get("clinical_space_summary"),
            **options,
        )

    def retrieve(self, summary, paths, top_n, candidate_k, progress):
        import torch
        from matchminer_ai.matching import (
            retrieve_guideline_considerations,
            write_guideline_considerations_report,
        )

        torch.set_num_threads(8)
        patients = pd.DataFrame(
            [{"patient_id": "workspace-patient", "cancer_history_summary": summary}]
        )
        result, metadata = retrieve_guideline_considerations(
            patients,
            paths,
            top_n=top_n,
            candidate_k=candidate_k,
            config=self.settings.inference_config(),
            embedding_cache_dir=self.settings.cache / "guideline_embeddings",
            return_metadata=True,
            progress_callback=progress,
        )
        result = display_considerations(result)
        with TemporaryDirectory(prefix="guideline-fetcher-report-") as tmp:
            report = write_guideline_considerations_report(
                result,
                patients,
                Path(tmp) / "considerations.md",
                metadata=metadata,
            ).read_text()
        return {
            "spaces": json.loads(result.to_json(orient="records", force_ascii=False)),
            "summary": summary,
            "report": report,
            "provenance": {
                "catalog": metadata.get("catalog", {}),
                "retrieval": metadata.get("retrieval", {}),
                "models": metadata.get("model_metadata", {}),
                "guideline_embedding_cache": metadata.get("guideline_embedding_cache", {}),
            },
        }
