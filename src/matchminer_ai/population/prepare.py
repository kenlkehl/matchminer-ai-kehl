"""Part 1 of the anchored concordance review: anchors, then summaries and extraction records through each anchor.

``prepare_anchored_population`` runs three resumable stages into one folder:

- ``anchors/``: ``anchors.parquet``, ``exclusions.parquet`` and the notes through each anchor
  (``notes.parquet``, the only notes the later stages read);
- ``summaries/``: ``summarize_patients`` on those notes (``summaries.parquet``, ``qc.parquet``);
- ``extractions/``: ``extract_patient_records`` and then ``extract_document_records`` linked to them, written as
  an extraction record store (``extractions/store/``, ``matchminer_ai.patients.extraction.store``).

Each stage writes a path-free ``stage_receipt.json`` last (``matchminer_ai.training.trialspace.stage``): its
identity holds the input file stamps, the window, the seed, the models and the prompt contracts, never file hashes.
A complete stage is skipped, an interrupted one resumes from its checkpoints, and a folder begun under other
settings is refused. Every file holds patient data: the process umask is 077 while a stage runs, and a real run
writes only under the site's PHI roots (``check_output_location``).
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .anchors import GROUP_FIELDS, choose_anchor_notes, note_table, notes_through_anchor

VERSION = "mmai-population-concordance-v1"
LEVELS = ("document", "document_and_patient")
SUMMARY_COLUMNS = ("patient_id", *GROUP_FIELDS, "anchor_date", "cancer_history_summary",
                   "general_exclusion_criteria_evidence", "note_count")
ANCHORS, SUMMARIES, EXTRACTIONS = "anchors", "summaries", "extractions"
GATEWAY_LOG = "gateway_requests.jsonl"


@dataclass(frozen=True)
class PreparedPopulation:
    """Part 1's outputs: the anchors and exclusions, the notes through each anchor, the summaries and the
    extraction records (``documents``, ``patient_records``; None for a level not extracted)."""

    anchors: pd.DataFrame
    exclusions: pd.DataFrame
    notes: pd.DataFrame
    summaries: pd.DataFrame
    documents: pd.DataFrame | None = None
    patient_records: pd.DataFrame | None = None
    provenance: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- privacy


def check_output_location(path, *, synthetic):
    """Refuse an output folder on the wrong side of the data: a real run writes notes, summaries and records only
    under the data root's ``phi/`` or a site config ``paths.phi_roots`` folder, never in a ``no_phi`` folder; a
    synthetic run never writes in a ``phi`` folder."""
    from matchminer_ai.site import data_root, load_site_config

    resolved = Path(path).expanduser().resolve()
    parts = resolved.parts
    if synthetic:
        if "phi" in parts:
            raise ValueError("A synthetic run never writes in a phi folder.")
        return resolved
    if "no_phi" in parts:
        raise ValueError("A real run keeps patient data out of no_phi folders.")
    roots = [Path(r).expanduser().resolve() for r in load_site_config().get("paths.phi_roots") or []]
    with contextlib.suppress(ValueError):
        roots.append((data_root() / "phi").resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError("Refusing to write patient data outside a PHI folder (<data root>/phi or the site "
                         "config's paths.phi_roots); pass synthetic=True only for synthetic notes.")
    return resolved


@contextlib.contextmanager
def private_files():
    """umask 077 while the block runs: new files 0600, new folders 0700."""
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


@contextlib.contextmanager
def gateway_log(path):
    """Log every gateway request of the block (status and token counts, never content) to ``path``."""
    previous = os.environ.get("MMAI_GATEWAY_REQUEST_LOG")
    os.environ["MMAI_GATEWAY_REQUEST_LOG"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MMAI_GATEWAY_REQUEST_LOG", None)
        else:
            os.environ["MMAI_GATEWAY_REQUEST_LOG"] = previous


def gateway_usage(path):
    """Token totals of a stage's gateway log (``matchminer_ai.llm.gateway_usage``), or None without requests."""
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        return None
    from matchminer_ai.llm.gateway_usage import summarize

    usage = summarize(path)
    usage.pop("log", None)
    (usage.get("concurrency") or {}).pop("trajectory", None)
    return usage


# --------------------------------------------------------------------------- stages


def stage_status(directory, kind, identity):
    """``(code, reason)``: 0 complete, 10 not yet (absent or resumable), 20 settings or outputs differ."""
    from matchminer_ai.training.trialspace import stage

    return stage.check(Path(directory), kind, identity)


def run_stage(directory, kind, identity, work, *, upstream=None):
    """The stage protocol: a complete stage is skipped; a folder begun under other settings is refused;
    ``work()`` returns ``(output names, content)`` and the receipt is written last. Returns the receipt."""
    from matchminer_ai.training.trialspace import stage

    directory = Path(directory)
    if stage.require_new_or_same(directory, kind, identity):
        return stage.read_receipt(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    outputs, content = work()
    return stage.write_receipt(directory, kind, identity, outputs, content=content, upstream=upstream)


def completed_receipt(directory, kind):
    """The receipt of a complete stage folder (its outputs present with their recorded sizes)."""
    from matchminer_ai.training.trialspace import stage

    receipt = stage.read_receipt(directory)
    if receipt is None or receipt.get("format") != stage.FORMAT or receipt.get("kind") != kind:
        raise ValueError(f"{Path(directory).name}: not a complete {kind} stage; run it first.")
    for name, size in receipt["outputs"].items():
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"{Path(directory).name}: output missing or changed ({name}); run the stage again.")
    return receipt


def _write_parquet(frame, path):
    frame.to_parquet(path, index=False)
    os.chmod(path, 0o600)


def _input(value, what, reader):
    """(DataFrame, stamp) of a table given as a DataFrame (no stamp) or a CSV or Parquet path."""
    if isinstance(value, pd.DataFrame):
        return value, None
    from matchminer_ai.labeling.files import file_stamp

    return reader(value), file_stamp(value)


def read_table(path):
    """A CSV (every column as text) or Parquet table."""
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() in (".csv", ".tsv"):
        return pd.read_csv(path, dtype=str, keep_default_na=False, sep="\t" if path.suffix.lower() == ".tsv"
                           else ",")
    raise ValueError(f"{path.name}: give a CSV or Parquet file.")


def _section(config, name, keys):
    values = getattr(config, name)
    remote = values.get("remote") or {}
    return {**{k: values.get(k) for k in keys}, "model": remote.get("model_name"),
            "provider": config.remote.get("provider")}


def summary_identity(config):
    """The parts of the summary settings that change summaries: model, provider, chunking, effort and prompts."""
    from matchminer_ai._storage import digest
    from matchminer_ai.patients.prompt_builder import load_prompt_text

    patient = config.patient
    prompts = {k: digest(load_prompt_text(str(v))) for k, v in sorted((patient.get("prompt_files") or {}).items())}
    return {**_section(config, "patient", ("chunk_size", "chunk_overlap", "context_window", "reasoning_effort",
                                           "sampling_profile", "summary_quality_max_attempts")),
            "prompts": prompts}


def extraction_identity(config, levels):
    """The extraction models and contracts (schema, prompts and guides) of each level."""
    from matchminer_ai.patients.extraction import documents, patient_records

    linking = levels == "document_and_patient"
    identity = {"levels": levels,
                "document": {**_section(config, "document_extraction", ("reasoning_effort",)),
                             "contract": documents._resolve_settings(config, linking)[-1]}}
    if linking:
        identity["patient"] = {**_section(config, "patient_extraction", ("chunk_tokens", "prompt_tokens",
                                                                          "reasoning_effort")),
                               "contract": patient_records._resolve_settings(config)[-1]}
    return identity


def _seed_document_checkpoint(source, target, note_ids):
    """Copy ``source``'s checkpointed document records of these notes into a new checkpoint at ``target``; the
    extractor reuses each one only when its input hash and contract match (its own rule)."""
    from matchminer_ai.patients.extraction.documents import CHECKPOINT_FILE

    target_file = Path(target) / CHECKPOINT_FILE
    if target_file.exists():
        return 0
    source_file = Path(source) / CHECKPOINT_FILE
    if not source_file.is_file():
        raise ValueError(f"{Path(source).name} has no {CHECKPOINT_FILE} to reuse.")
    wanted, copied = set(note_ids), 0
    Path(target).mkdir(mode=0o700, parents=True, exist_ok=True)
    with source_file.open(encoding="utf-8") as reader, target_file.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            if json.loads(line).get("note_id") in wanted:
                writer.write(line if line.endswith("\n") else line + "\n")
                copied += 1
    return copied


def prepare_anchored_population(patients, notes, *, output_dir, start_date=None, end_date=None, seed=23,
                                columns=None, summary_config=None, extraction_config=None,
                                levels="document_and_patient", reuse_documents=None, synthetic=False,
                                check=False, progress_callback=None):
    """Choose anchors, then summarize and extract each patient's notes through the anchor; see the module.

    ``patients`` and ``notes`` are DataFrames or CSV/Parquet paths, as for
    :func:`~matchminer_ai.population.anchors.choose_anchor_notes` (with ``columns``,
    ``start_date``, ``end_date`` and ``seed``). ``summary_config`` sets the ``patient``
    section and ``extraction_config`` the ``document_extraction`` and
    ``patient_extraction`` sections (``levels``: ``document_and_patient``, the default,
    or ``document``). ``reuse_documents``: an extraction checkpoint folder whose
    document records are reused when their input hash and contract match.
    ``synthetic``: the notes are synthetic, so the output may be outside the PHI roots
    (never in a ``phi`` folder). Every note through an anchor is sent to the configured
    endpoints, which must be authorized for the notes. Gateway token logs go to each
    stage's ``gateway_requests.jsonl``.

    Returns the :class:`PreparedPopulation`, or with ``check`` each stage's
    ``(code, reason)`` without running anything.
    """
    from matchminer_ai import load_default_preset
    from matchminer_ai._storage import digest

    if levels not in LEVELS:
        raise ValueError(f"levels must be one of {LEVELS}")
    output = check_output_location(output_dir, synthetic=synthetic)
    progress = progress_callback or (lambda _: None)
    summary_config = summary_config or load_default_preset()
    extraction_config = extraction_config or load_default_preset()
    patient_frame, patient_stamp = _input(patients, "patients", read_table)
    note_frame, note_stamp = _input(notes, "notes", read_table)
    with private_files():
        progress("Choosing anchor notes")
        table = note_table(note_frame, columns)
        anchors, exclusions = choose_anchor_notes(patient_frame, table, start_date=start_date,
                                                  end_date=end_date, seed=seed, columns=columns)
        through = notes_through_anchor(table, anchors)
        del table
        anchor_identity = {
            "version": VERSION,
            "inputs": {"patients": patient_stamp, "notes": note_stamp},
            "start_date": None if start_date is None else str(start_date),
            "end_date": None if end_date is None else str(end_date),
            "seed": seed,
            "columns": dict(sorted((columns or {}).items())),
            "selection_sha256": digest({
                "anchors": anchors[["patient_id", "anchor_note_id", "anchor_date", "anchor_time"]].values.tolist(),
                "exclusions": exclusions[["patient_id", "reason"]].values.tolist(),
                "notes": through[["patient_id", "note_id"]].values.tolist(),
            }),
        }
        statuses = {}
        anchors_dir = output / ANCHORS

        def write_anchors():
            _write_parquet(anchors, anchors_dir / "anchors.parquet")
            _write_parquet(exclusions, anchors_dir / "exclusions.parquet")
            _write_parquet(through, anchors_dir / "notes.parquet")
            return (["anchors.parquet", "exclusions.parquet", "notes.parquet"],
                    {"patients": len(patient_frame), "anchored": len(anchors), "notes": len(through),
                     "exclusions": exclusions.reason.value_counts().to_dict()})

        if check:
            statuses[ANCHORS] = stage_status(anchors_dir, "population-anchors", anchor_identity)
            anchors_id = completed_receipt(anchors_dir, "population-anchors")["receipt_id"] \
                if statuses[ANCHORS][0] == 0 else None
            # A later stage cannot be complete before the anchors are: it waits (10) or must restart (20).
            upstream_status = (statuses[ANCHORS][0], "the anchors stage is not complete"
                               if statuses[ANCHORS][0] == 10 else "the anchors stage differs")
        else:
            anchors_id = run_stage(anchors_dir, "population-anchors", anchor_identity, write_anchors)["receipt_id"]
            progress(f"Anchors: {len(anchors)} of {len(patient_frame)} patients, {len(through)} notes")

        # Summaries -----------------------------------------------------------------------------------------
        summaries_dir = output / SUMMARIES
        identity = {"version": VERSION, "anchors": anchors_id, "summary": summary_identity(summary_config)}

        def summarize():
            from matchminer_ai.patients import summarize_patients

            checkpoint = summaries_dir / "checkpoints" / digest(identity)[:16]
            log = summaries_dir / GATEWAY_LOG
            progress(f"Summarizing {through.patient_id.nunique()} patients through their anchors")
            with gateway_log(log):
                result, _, qc = summarize_patients(
                    through[["patient_id", "note_text", "note_date", "note_type"]], config=summary_config,
                    checkpoint_dir=checkpoint, return_metadata=True, return_qc=True)
            counts = through.groupby("patient_id").size()
            frame = anchors[["patient_id", *GROUP_FIELDS, "anchor_date"]].merge(
                result[["patient_id", "cancer_history_summary", "general_exclusion_criteria_evidence"]],
                on="patient_id", how="inner")
            frame["note_count"] = [int(counts.get(p, 0)) for p in frame.patient_id]
            _write_parquet(frame[list(SUMMARY_COLUMNS)], summaries_dir / "summaries.parquet")
            _write_parquet(qc.astype(str), summaries_dir / "qc.parquet")
            return ["summaries.parquet", "qc.parquet"], {
                "anchored": len(anchors), "summarized": len(frame), "gateway_usage": gateway_usage(log)}

        if check:
            statuses[SUMMARIES] = stage_status(summaries_dir, "population-summaries", identity) \
                if anchors_id else upstream_status
        else:
            run_stage(summaries_dir, "population-summaries", identity, summarize, upstream={ANCHORS: anchors_id})

        # Extraction records --------------------------------------------------------------------------------
        extractions_dir = output / EXTRACTIONS
        identity = {"version": VERSION, "anchors": anchors_id,
                    "extraction": extraction_identity(extraction_config, levels)}

        def extract():
            import matchminer_ai.patients as package
            from matchminer_ai.patients.extraction.store import extraction_metadata, write_extraction_store

            checkpoint = extractions_dir / "checkpoints"
            checkpoint.mkdir(mode=0o700, parents=True, exist_ok=True)
            reused = 0
            if reuse_documents is not None:
                check_output_location(reuse_documents, synthetic=synthetic)
                reused = _seed_document_checkpoint(reuse_documents, checkpoint, through.note_id)
            log = extractions_dir / GATEWAY_LOG
            patient_records = patient_meta = None
            with gateway_log(log):
                if levels == "document_and_patient":
                    progress(f"Extracting patient records for {through.patient_id.nunique()} patients")
                    patient_records, patient_meta = package.extract_patient_records(
                        through, config=extraction_config, checkpoint_dir=checkpoint, return_metadata=True,
                        progress_callback=lambda done, total: progress(f"Patient records: {done}/{total} chunks"))
                    ids = through.groupby("patient_id").note_id.apply(lambda v: sorted(set(v)))
                    patient_records["note_ids"] = [ids.get(p, []) for p in patient_records.patient_id]
                progress(f"Extracting document records for {len(through)} notes")
                documents, document_meta = package.extract_document_records(
                    through, config=extraction_config, patient_records=patient_records,
                    checkpoint_dir=checkpoint, return_metadata=True,
                    progress_callback=lambda done, total: progress(f"Document records: {done}/{total} notes"))
            model = extraction_config.document_extraction["remote"].get("model_name")
            extraction = extraction_metadata(document_meta, patient_meta, model)
            extraction["gateway_usage"] = gateway_usage(log)
            source = {"kind": "anchored_population", "patients": int(through.patient_id.nunique()),
                      "notes": len(through), "reused_document_checkpoints": reused}
            write_extraction_store(extractions_dir / "store", documents, patient_records,
                                   mode="synthetic" if synthetic else "real", source=source,
                                   extraction=extraction, private=True, overwrite=True)
            names = ["store/manifest.json", "store/document_records.parquet"]
            if patient_records is not None:
                names.append("store/patient_records.parquet")
            statuses_ = {"documents": documents.extraction_status.value_counts().to_dict()}
            if patient_records is not None:
                statuses_["patient_records"] = patient_records.extraction_status.value_counts().to_dict()
            return names, {"statuses": statuses_, "gateway_usage": extraction["gateway_usage"]}

        if check:
            statuses[EXTRACTIONS] = stage_status(extractions_dir, "population-extractions", identity) \
                if anchors_id else upstream_status
            return statuses
        run_stage(extractions_dir, "population-extractions", identity, extract, upstream={ANCHORS: anchors_id})
    return load_prepared_population(output)


def load_prepared_population(folder):
    """Part 1's outputs from a folder whose three stages are complete."""
    from matchminer_ai.patients.extraction.store import read_store_manifest, read_store_records

    folder = Path(folder)
    receipts = {name: completed_receipt(folder / name, f"population-{name}")
                for name in (ANCHORS, SUMMARIES, EXTRACTIONS)}
    store = folder / EXTRACTIONS / "store"
    manifest = read_store_manifest(store)
    extraction = manifest.get("extraction") or {}
    documents, patient_records = read_store_records(store)
    anchors = pd.read_parquet(folder / ANCHORS / "anchors.parquet")
    exclusions = pd.read_parquet(folder / ANCHORS / "exclusions.parquet")
    return PreparedPopulation(
        anchors=anchors, exclusions=exclusions,
        notes=pd.read_parquet(folder / ANCHORS / "notes.parquet"),
        summaries=pd.read_parquet(folder / SUMMARIES / "summaries.parquet"),
        documents=documents, patient_records=patient_records,
        provenance={"receipts": {k: r["receipt_id"] for k, r in receipts.items()},
                    "identity": {k: r["identity"] for k, r in receipts.items()},
                    "content": {k: r.get("content", {}) for k, r in receipts.items()},
                    "extraction_store": {k: manifest.get(k) for k in ("mode", "levels", "counts")},
                    "extraction_model": extraction.get("model"),
                    # The extensions the records were extracted with, which compiled queries must cover.
                    "extensions": list(dict.fromkeys(
                        name for level in ("document", "patient") for name in
                        ((extraction.get(level) or {}).get("extensions") or [])))},
    )


__all__ = ["PreparedPopulation", "check_output_location", "load_prepared_population",
           "prepare_anchored_population", "read_table"]
