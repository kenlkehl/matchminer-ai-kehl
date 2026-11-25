"""Extraction record stores: a folder of document and patient extraction records with a manifest.

The dashboard's extraction-based workup checks and criteria screening read stores, and its
``preprocess_extractions`` command and ``matchminer_ai.population.prepare`` write them. A store
holds patient-derived text: write it privately (``private=True``: folder 0700, files 0600)
wherever the notes may be kept. Layout::

  manifest.json             format, version, mode, source, levels, counts, extraction, files
  document_records.parquet  patient_id, note_id, note_date, note_type,
                            extraction_status, error, item_count, record_json
  patient_records.parquet   patient_id, extraction_status, error, failed_chunks_json,
                            item_count, note_ids_json, record_json

``mode`` is ``real`` or ``synthetic``: the dashboard opens only the stores of its own mode.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "guideline-fetcher extraction records"
VERSION = 1
MODES = ("real", "synthetic")
# Stores built by the former guideline fetcher record its mode names.
FORMER_MODES = {"phi": "real", "standard": "synthetic"}
DOCUMENT_FILE = "document_records.parquet"
PATIENT_FILE = "patient_records.parquet"
ROW_GROUP = 2000


def _clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _record_json(record):
    return json.dumps(record, ensure_ascii=False) if isinstance(record, dict) else None


def write_extraction_store(output, documents=None, patients=None, *, mode, source, extraction,
                           private=False, overwrite=False):
    """Write a store atomically from extraction output DataFrames; returns its manifest.

    ``documents`` is ``extract_document_records`` output and ``patients``
    ``extract_patient_records`` output, which may carry ``note_ids`` (the ids of the
    notes each patient record was built from). ``source`` and ``extraction``
    (:func:`extraction_metadata`) go into the manifest as they are. The store is
    written beside ``output`` and renamed into place; an existing store is replaced
    only with ``overwrite``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    output = Path(output).expanduser().resolve()
    if output.exists() and not overwrite:
        raise ValueError(f"{output} exists. Pass --overwrite to replace it.")
    if documents is None and patients is None:
        raise ValueError("No extraction records to write.")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial = output.parent / f".{output.name}.partial-{secrets.token_hex(4)}"
    partial.mkdir(mode=0o700)
    try:
        files, counts, levels = {}, {}, []
        if documents is not None:
            levels.append("document")
            frame = documents.sort_values(["patient_id", "note_date", "note_id"],
                                          na_position="last", kind="stable")
            table = pa.table({
                "patient_id": [str(p) for p in frame.patient_id],
                "note_id": [str(n) for n in frame.note_id],
                "note_date": [_clean(d) for d in frame.note_date],
                "note_type": [_clean(t) for t in frame.note_type],
                "extraction_status": list(frame.extraction_status),
                "error": [_clean(e) for e in frame.error],
                "item_count": [int(c) for c in frame.item_count],
                "record_json": [_record_json(r) for r in frame.record],
            })
            pq.write_table(table, partial / DOCUMENT_FILE, row_group_size=ROW_GROUP,
                           compression="zstd")
            counts["documents"] = {s: int((frame.extraction_status == s).sum())
                                   for s in ("complete", "partial", "failed")}
        if patients is not None:
            levels.append("patient")
            frame = patients.sort_values("patient_id", kind="stable")
            note_ids = frame["note_ids"] if "note_ids" in frame else [[]] * len(frame)
            table = pa.table({
                "patient_id": [str(p) for p in frame.patient_id],
                "extraction_status": list(frame.extraction_status),
                "error": [_clean(e) for e in frame.error],
                "failed_chunks_json": [json.dumps(list(c or [])) for c in frame.failed_chunks],
                "item_count": [int(c) for c in frame.item_count],
                "note_ids_json": [json.dumps(list(n)) for n in note_ids],
                "record_json": [_record_json(r) for r in frame.record],
            })
            pq.write_table(table, partial / PATIENT_FILE, row_group_size=ROW_GROUP,
                           compression="zstd")
            counts["patient_records"] = {s: int((frame.extraction_status == s).sum())
                                         for s in ("complete", "partial", "failed")}
        ids = set()
        for frame in (documents, patients):
            if frame is not None:
                ids |= set(map(str, frame.patient_id))
        counts["patients"] = len(ids)
        for name in (DOCUMENT_FILE, PATIENT_FILE):
            if (partial / name).exists():
                files[name] = (partial / name).stat().st_size
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "mode": mode,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "levels": levels,
            "counts": counts,
            "extraction": extraction,
            "files": files,
        }
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        if private:
            for name in (*files, "manifest.json"):
                os.chmod(partial / name, 0o600)
        if output.exists():
            retired = output.parent / f".{output.name}.retired-{secrets.token_hex(4)}"
            output.rename(retired)
            partial.rename(output)
            shutil.rmtree(retired)
        else:
            partial.rename(output)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return manifest


def read_store_manifest(path, *, mode=None):
    """A store's manifest, checked: format and version, ``mode`` when given, and file sizes.

    Raises ``ValueError`` naming the problem; no records are read.
    """
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError):
        raise ValueError("This extraction record manifest cannot be read.") from None
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ValueError("Unsupported extraction record format. Rebuild the records.")
    if mode is not None and FORMER_MODES.get(manifest.get("mode"), manifest.get("mode")) != mode:
        raise ValueError("These extraction records were built for the other app mode.")
    for name, size in manifest["files"].items():
        if not (path / name).is_file() or (path / name).stat().st_size != size:
            raise ValueError("These extraction records are incomplete or changed. Rebuild them.")
    return manifest


def read_store_records(path, patient_ids=None):
    """(document records, patient records) of a store, as the extractors return them.

    ``record`` holds each record dict (None when failed), and patient records'
    ``failed_chunks`` and ``note_ids`` are lists. ``patient_ids`` reads only those
    patients' rows. A level the store lacks is None. Check the store with
    :func:`read_store_manifest` first.
    """
    import pyarrow.dataset as ds

    frames = []
    for name in (DOCUMENT_FILE, PATIENT_FILE):
        file = Path(path) / name
        if not file.is_file():
            frames.append(None)
            continue
        dataset = ds.dataset(file, format="parquet")
        table = dataset.to_table() if patient_ids is None else dataset.to_table(
            filter=ds.field("patient_id").isin([str(p) for p in patient_ids]))
        frame = table.to_pandas()
        frame["record"] = [json.loads(r) if isinstance(r, str) else None
                           for r in frame.pop("record_json")]
        if "failed_chunks_json" in frame:
            frame["failed_chunks"] = [json.loads(v) for v in frame.pop("failed_chunks_json")]
            frame["note_ids"] = [json.loads(v) for v in frame.pop("note_ids_json")]
        frames.append(frame)
    return frames[0], frames[1]


def extraction_metadata(document_metadata=None, patient_metadata=None, model=None, *,
                        servers=None) -> dict:
    """The contract facts of a build for a store's manifest, without per-note attempts or patient text.

    ``document_metadata`` and ``patient_metadata`` are the extractors' metadata
    (``return_metadata=True``); ``servers``: the build's servers and the model each
    served, when it used several.
    """
    def part(metadata):
        if metadata is None:
            return None
        return {k: metadata.get(k) for k in ("schema_version", "contract_sha256", "prompt_sha256",
                                             "response_schema_sha256", "extensions",
                                             "structured_outputs", "linking", "counts")}

    first = document_metadata or patient_metadata or {}
    return {
        "schema_version": first.get("schema_version"),
        "model": model,
        "document": part(document_metadata),
        "patient": part(patient_metadata),
        "package": first.get("package"),
        **({"servers": [dict(s) for s in servers]} if servers else {}),
    }


__all__ = ["DOCUMENT_FILE", "FORMAT", "PATIENT_FILE", "VERSION", "extraction_metadata",
           "read_store_manifest", "read_store_records", "write_extraction_store"]
