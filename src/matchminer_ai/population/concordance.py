"""Part 2 of the anchored concordance review: each patient's guideline space, then that space's workup items.

``assess_concordance`` reads part 1's outputs (a folder, or a :class:`~.prepare.PreparedPopulation`):

1. **Match.** ``retrieve_guideline_considerations`` ranks every space of the catalogs for each patient summary
   (TrialSpace top ``candidate_k``, then the decision model's match questions) and keeps the top space. A patient
   whose top space's P(reasonable) is below ``min_reasonableness`` is excluded ("no reasonable guideline
   match"), keeping that candidate and its scores.
2. **Assess.** Patients are grouped by their space, whose workup items (without history-and-physical items) are
   checked by one method: ``extraction_queries`` (default: compiled queries over the extraction records, compiled
   once per item and cached), ``extraction_sql`` (the SQL agent's queries over a database of the run's records),
   ``full_notes`` or ``note_search`` (an LLM reads the notes through the anchor). Each patient's evaluation date
   is the anchor date.
3. **Concordance.** Each result maps to met, not met, unknown or not applicable (``concordance_category``);
   concordance is met / (met + not met).

Concordance here is measured from documentation in the notes: a step done but not documented counts as not met.
"""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .anchors import GROUP_FIELDS
from .prepare import (
    VERSION, check_output_location, completed_receipt, gateway_log, gateway_usage, load_prepared_population,
    private_files, run_stage, stage_status,
)

METHODS = ("extraction_queries", "extraction_sql", "full_notes", "note_search")
CATEGORIES = ("met", "not_met", "unknown", "not_applicable")
#: Raw statuses by category: compiled queries and the SQL agent, then full-note and note-search reviews (plus
#: ``error`` for a failed review).
STATUS_CATEGORIES = {
    "met": "met", "completed": "met",
    "not_met": "not_met", "done_not_recommended": "not_met", "partially_completed": "not_met",
    "planned": "not_met", "not_done": "not_met", "not_documented": "not_met",
    "unknown": "unknown", "unclear": "unknown", "error": "unknown",
    "not_applicable": "not_applicable",
}
NO_SUMMARY = "no_usable_summary"
NO_RECORDS = "no_extraction_records"
NO_MATCH = "no_reasonable_guideline_match"
NO_WORKUP = "no_workup_items"
CAVEAT = (
    "Concordance is measured from documentation in the notes through each patient's anchor note: a step done but "
    "not documented there counts as not met, and a step the records cannot settle is unknown and left out of the "
    "fraction. Guideline spaces are chosen by TrialSpace and the decision model, and workup results come from LLM "
    "summaries, extraction and compiled queries: review them against the source notes. Research use only; not a "
    "measure of the quality of care."
)
MATCHES = "matches"


def concordance_category(status, applicability, *, recommends_against=False):
    """``(category, note)`` of one workup result.

    - **Not applicable:** status ``not_applicable`` or applicability ``not_applicable``.
    - **Unknown:** ``unknown``, ``unclear`` or ``error``; also a met or not-met result whose applicability is
      uncertain (note ``applicability_uncertain``), which stays out of the fraction as the documentation fractions
      do.
    - **Met:** ``met`` or ``completed``; for a step the guideline advises against, a ``not_met`` result (the step
      is not documented as done; note ``not_recommended_not_done``).
    - **Not met:** ``not_met``, ``partially_completed``, ``planned``, ``not_done``, ``not_documented``, and
      ``done_not_recommended`` (a step advised against, done anyway).
    """
    if status not in STATUS_CATEGORIES:
        raise ValueError(f"Unknown workup status {status!r}.")
    category = STATUS_CATEGORIES[status]
    if category == "not_applicable" or applicability == "not_applicable":
        return "not_applicable", None
    if category in ("met", "not_met") and applicability != "applies":
        return "unknown", "applicability_uncertain"
    if recommends_against and status == "not_met":
        return "met", "not_recommended_not_done"
    return category, None


def catalog_paths(catalogs):
    """Catalog folders: ``catalogs`` is a catalog root (the latest complete, audited catalog of each disease,
    ``latest_guideline_catalogs``), one catalog folder or ``paradigms.jsonl``, or a list of them."""
    from matchminer_ai.trials.guideline_catalog import latest_guideline_catalogs

    if isinstance(catalogs, (str, Path)):
        root = Path(catalogs)
        if root.suffix == ".jsonl" or (root / "paradigms.jsonl").is_file():
            return [str(root)]
        rows = latest_guideline_catalogs(root)
        if not rows:
            raise ValueError("No complete, audited guideline catalogs under the catalog root.")
        return [row["path"] for row in rows]
    paths = [str(Path(c)) for c in catalogs]
    if not paths:
        raise ValueError("Supply at least one guideline catalog.")
    return paths


def _records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))


def _model_identity(config):
    """Path-free names of TrialSpace and the decision model (a local folder by its name)."""
    from matchminer_ai.site import load_site_config

    embedding = config.embedding
    model = embedding.get("model_path") or load_site_config().get("paths.trialspace_model") \
        or embedding.get("default_model")
    revision = embedding.get("revision") if embedding.get("model_path") else (
        None if load_site_config().get("paths.trialspace_model") else embedding.get("default_revision"))
    decision = config.raw.get("decision_model") or {}

    def name(value):  # a Hub id as it is, a local folder by its name
        value = str(value)
        return Path(value).name if value.startswith(("/", ".", "~")) else value

    return {"trialspace": [name(model), revision],
            "decision_model": [name(decision.get("model_name")), decision.get("revision")]}


def match_guideline_spaces(summaries, catalogs, *, config, candidate_k=20, min_reasonableness=0.5,
                           embedding_cache_dir=None, progress_callback=None):
    """Each patient's top guideline space: ``(matches, spaces, metadata)``.

    ``matches`` has ``patient_id``, ``space_trial_id``, ``space_name``, ``disease``,
    ``guideline``, ``catalog_version``, ``retrieval_rank``, ``similarity_score``,
    ``match_quality_score``, ``match_quality_reasonableness`` (P(reasonable)), the other
    match questions' P(yes), ``excluded`` and ``exclusion_reason`` (``NO_MATCH`` below
    ``min_reasonableness``). ``spaces`` holds the matched spaces' catalog records by ID.
    """
    from matchminer_ai.matching import retrieve_guideline_considerations

    result, metadata = retrieve_guideline_considerations(
        summaries[["patient_id", "cancer_history_summary"]], catalog_paths(catalogs), top_n=1,
        candidate_k=candidate_k, config=config, embedding_cache_dir=embedding_cache_dir, return_metadata=True,
        progress_callback=progress_callback)
    rows, spaces = [], {}
    heads = [c for c in result.columns if c.startswith("match_quality_") and c not in
             ("match_quality_score", "match_quality_pass")]
    for record in _records(result):
        source = record.get("source") or {}
        reasonable = float(record["match_quality_reasonableness"])
        excluded = reasonable < min_reasonableness
        rows.append({
            "patient_id": str(record["patient_id"]), "space_trial_id": record["space_trial_id"],
            "space_name": record.get("name"), "disease": source.get("disease"), "guideline": source.get("title"),
            "catalog_version": source.get("version"), "retrieval_rank": record.get("retrieval_rank"),
            "similarity_score": record.get("similarity_score"),
            "match_quality_score": record.get("match_quality_score"),
            **{head: record.get(head) for head in heads},
            "excluded": excluded, "exclusion_reason": NO_MATCH if excluded else None,
        })
        spaces.setdefault(record["space_trial_id"], {
            k: record.get(k) for k in ("space_trial_id", "trial_id", "name", "source", "clinical_space_summary",
                                       "cohort_key_criteria", "diagnostic_workup")})
    return pd.DataFrame(rows), spaces, metadata


def _population_context(space):
    return space.get("cohort_key_criteria") or space.get("clinical_space_summary") or ""


def _writer_cache(folder, config, section):
    """Compiled queries apart for each writing model, in the dashboard's layout (``writer-<model hash>``)."""
    import hashlib

    from matchminer_ai._storage import QueryCache

    model = str((getattr(config, section).get("remote") or {}).get("model_name") or "")
    return QueryCache(Path(folder) / f"writer-{hashlib.sha256(model.encode()).hexdigest()[:16]}")


def _error_review(items, exc):
    """One error row per item for a review that failed (only the exception type: messages can echo notes)."""
    return {"assessments": [
        dict(recommendation_index=number, recommendation=copy.deepcopy(item), name=item["name"], status="error",
             applicability="uncertain", evidence=[], bottom_line="No validated documentation assessment is "
             "available.", limitations=[f"The review failed ({type(exc).__name__})."], metadata={})
        for number, item in enumerate(items)], "metadata": {"error": type(exc).__name__}}


def _note_search_reviews(cohort, items, config, max_concurrent_patients):
    from matchminer_ai.patients import review_patient_workup_with_note_search

    grouped = dict(tuple(cohort.notes.groupby("patient_id", sort=False)))
    summaries = cohort.summaries.set_index("patient_id").cancer_history_summary.to_dict()

    def review(pid):
        try:
            return review_patient_workup_with_note_search(
                grouped[pid].drop(columns="patient_id"), items, config=config, patient_summary=summaries[pid],
                population_context=_population_context(cohort.space))
        except Exception as exc:  # cancellation is a BaseException and propagates
            return _error_review(items, exc)

    ids = cohort.summaries.patient_id.tolist()
    with ThreadPoolExecutor(max_workers=max_concurrent_patients) as pool:
        return [{"patient_id": pid, "review": result} for pid, result in zip(ids, pool.map(review, ids))]


def _evidence(entries, limit=16):
    out = []
    for entry in (entries or [])[:limit]:
        text = entry.get("quote") or entry.get("text") or ""
        out.append({"note_id": entry.get("note_id"), "note_date": entry.get("note_date"),
                    "quote": text[:1000], "start": entry.get("start"), "end": entry.get("end")})
    return out


def assess_space(space, patient_ids, prepared, *, method, config, compile_config, query_cache, sql_database=None,
                 sql_cache=None, extensions=None, max_concurrent_patients=4):
    """One space's workup results for its matched patients: ``(patients, queries)``, where ``patients`` is the
    population report's ``patients`` list (one review per patient, one assessment per item) and ``queries`` the
    compile summary (None for the note-reading methods)."""
    from matchminer_ai.patients.workup import without_history_and_physical
    from matchminer_ai.population import (
        PatientCohort, check_population_workup_with_extractions, check_population_workup_with_sql,
        review_population_workup,
    )

    space = without_history_and_physical(space)
    items = space["diagnostic_workup"]
    ids = set(patient_ids)
    summaries = prepared.summaries.loc[prepared.summaries.patient_id.isin(ids)]
    summaries = summaries.set_index("patient_id").loc[list(patient_ids)].reset_index()[
        ["patient_id", "cancer_history_summary", "general_exclusion_criteria_evidence"]]
    notes = prepared.notes.loc[prepared.notes.patient_id.isin(ids),
                               ["patient_id", "note_text", "note_date", "note_type", "note_id"]]
    cohort = PatientCohort(summaries.reset_index(drop=True), notes.reset_index(drop=True), space=space)
    anchors = prepared.anchors.set_index("patient_id").anchor_date.to_dict()
    references = {pid: anchors[pid] for pid in patient_ids}
    context = _population_context(space)

    def records(frame):
        return None if frame is None else frame.loc[frame.patient_id.astype(str).isin(ids)]

    if method == "extraction_queries":
        from matchminer_ai.patients import compile_workup_queries

        compiled, metadata = compile_workup_queries(items, config=compile_config, population_context=context,
                                                    cache=query_cache, extensions=extensions, return_metadata=True)
        result = check_population_workup_with_extractions(
            cohort, compiled, document_records=records(prepared.documents),
            patient_records=records(prepared.patient_records), reference_dates=references)
        return result.report["patients"], {**{k: metadata.get(k) for k in (
            "distinct", "cached", "compiled", "first_attempt_valid", "requests")},
            "failed": sum(e["status"] != "compiled" for e in compiled)}
    if method == "extraction_sql":
        from matchminer_ai.patients import compile_workup_sql

        compiled, metadata = compile_workup_sql(items, database=sql_database, config=compile_config,
                                                population_context=context, cache=sql_cache, return_metadata=True)
        result = check_population_workup_with_sql(
            cohort, compiled, document_records=records(prepared.documents),
            patient_records=records(prepared.patient_records), reference_dates=references, config=compile_config)
        return result.report["patients"], {k: metadata.get(k) for k in (
            "distinct", "cached", "compiled", "failed", "requests", "prompt_tokens", "completion_tokens")}
    if method == "full_notes":
        result = review_population_workup(cohort, config=config, max_concurrent_patients=max_concurrent_patients)
        return result.report["patients"], None
    return _note_search_reviews(cohort, items, config, max_concurrent_patients), None


def _item_rows(patients, space, method, fields):
    rows = []
    for patient in patients:
        pid = patient["patient_id"]
        for item in patient["review"]["assessments"]:
            requirement = item.get("recommendation") or {}
            against = bool(item.get("recommends_against"))
            category, note = concordance_category(item["status"], item["applicability"],
                                                  recommends_against=against)
            rows.append({
                "patient_id": pid, **fields[pid],
                "space_trial_id": space["space_trial_id"], "space_name": space.get("name"),
                "disease": (space.get("source") or {}).get("disease"),
                "item_index": int(item["recommendation_index"]), "item_name": requirement.get("name"),
                "item_conditions": requirement.get("conditions"), "item_category": requirement.get("category"),
                "method": method, "raw_status": item["status"], "reason": item.get("reason"),
                "applicability": item["applicability"], "recommends_against": against,
                "category": category, "category_note": note, "bottom_line": item.get("bottom_line"),
                "query_key": item.get("query_key"),
                "evidence_json": json.dumps(_evidence(item.get("evidence")), ensure_ascii=False),
            })
    return rows


def assess_concordance(prepared, *, catalogs, output_dir, method="extraction_queries", min_reasonableness=0.5,
                       candidate_k=20, config=None, compile_config=None, query_cache=None,
                       embedding_cache_dir=None, synthetic=False, max_concurrent_spaces=4,
                       max_concurrent_patients=4, check=False, progress_callback=None):
    """Match each prepared patient to a guideline space, check its workup items and write the concordance report.

    ``prepared``: part 1's folder or :class:`~.prepare.PreparedPopulation` (with
    ``prepared.provenance`` from :func:`~.prepare.load_prepared_population`).
    ``catalogs``: see :func:`catalog_paths`. ``config`` runs TrialSpace and the decision
    model (``embedding``, ``decision_model``) and, for ``full_notes`` and ``note_search``,
    the ``patient`` endpoint. ``compile_config`` writes the queries
    (``workup_query_compile``, or ``workup_sql_agent`` for ``extraction_sql``).
    ``query_cache``: a folder of compiled queries shared across runs (default: the
    output's ``query_cache/``). ``embedding_cache_dir``: guideline vectors (default: the
    output's ``guideline_embeddings/``; guideline text only). Spaces are assessed
    ``max_concurrent_spaces`` at a time. Writes the outputs of
    :func:`~matchminer_ai.population.report.write_concordance_report` into
    ``output_dir`` and returns them (:func:`~matchminer_ai.population.report.read_concordance_report`);
    with ``check``, only each stage's ``(code, reason)``. For the extraction methods, a
    patient without any usable extraction record is excluded (``no_extraction_records``).
    """
    from matchminer_ai import load_default_preset
    from matchminer_ai._storage import QueryCache

    from . import report

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    output = check_output_location(output_dir, synthetic=synthetic)
    if method == "extraction_sql" and query_cache is not None:
        # Accepted SQL entries keep aggregate counts of the records they ran on.
        check_output_location(query_cache, synthetic=synthetic)
    progress = progress_callback or (lambda _: None)
    if not isinstance(prepared, (str, Path)) and not getattr(prepared, "provenance", None):
        raise ValueError("Pass part 1's folder, or a PreparedPopulation from load_prepared_population.")
    if isinstance(prepared, (str, Path)):
        prepared = load_prepared_population(prepared)
    config = config or load_default_preset()
    compile_config = compile_config or config
    paths = catalog_paths(catalogs)
    from matchminer_ai.trials import load_guideline_catalog

    _, catalog_metadata = load_guideline_catalog(paths, return_metadata=True)
    receipts = prepared.provenance["receipts"]
    match_identity = {
        "version": VERSION, "summaries": receipts["summaries"],
        "catalogs": {"sha256": catalog_metadata["catalog_sha256"],
                     "sources": [[s.get("disease"), s.get("version")] for s in catalog_metadata["sources"]]},
        "candidate_k": candidate_k, "min_reasonableness": min_reasonableness, "models": _model_identity(config),
    }
    store = {**(prepared.provenance.get("extraction_store") or {}),
             "model": prepared.provenance.get("extraction_model")}
    extensions = prepared.provenance.get("extensions")
    section = "workup_sql_agent" if method == "extraction_sql" else "workup_query_compile"
    writer = getattr(compile_config, section).get("remote") or {}
    assess_identity = {
        "version": VERSION, "method": method, "extractions": receipts["extractions"],
        "writer": [writer.get("model_name"), compile_config.remote.get("provider")]
        if method.startswith("extraction") else None,
        "reviewer": [config.patient["remote"].get("model_name"), config.remote.get("provider")]
        if not method.startswith("extraction") else None,
    }
    matches_dir = output / MATCHES
    if check:
        statuses = {MATCHES: stage_status(matches_dir, "population-matches", match_identity)}
        code = statuses[MATCHES][0]
        statuses["assessment"] = stage_status(output, "population-concordance", assess_identity) if code == 0 \
            else (code, "the matches stage is not complete" if code == 10 else "the matches stage differs")
        return statuses
    with private_files():
        def match():
            progress(f"Matching {len(prepared.summaries)} patients to guideline spaces")
            matches, spaces, metadata = match_guideline_spaces(
                prepared.summaries, paths, config=config, candidate_k=candidate_k,
                min_reasonableness=min_reasonableness,
                embedding_cache_dir=embedding_cache_dir or output / "guideline_embeddings",
                progress_callback=progress)
            matches.to_parquet(matches_dir / "guideline_matches.parquet", index=False)
            (matches_dir / "spaces.json").write_text(json.dumps(spaces, ensure_ascii=False))
            return ["guideline_matches.parquet", "spaces.json"], {
                "matched": int((~matches.excluded).sum()), "excluded": int(matches.excluded.sum()),
                "model_metadata": metadata.get("model_metadata"), "retrieval": metadata.get("retrieval")}

        match_receipt = run_stage(matches_dir, "population-matches", match_identity, match,
                                  upstream={"summaries": receipts["summaries"]})
        matches = pd.read_parquet(matches_dir / "guideline_matches.parquet")
        spaces = json.loads((matches_dir / "spaces.json").read_text())

        def assess():
            from matchminer_ai.patients.workup import without_history_and_physical

            fields = prepared.anchors.set_index("patient_id")[list(GROUP_FIELDS)].to_dict("index")
            missing = report.patients_without_records(prepared) if method.startswith("extraction") else set()
            included = matches.loc[~matches.excluded & ~matches.patient_id.isin(missing)]
            groups = {sid: group.patient_id.tolist() for sid, group in included.groupby("space_trial_id",
                                                                                          sort=True)}
            no_workup = [sid for sid in groups if not without_history_and_physical(spaces[sid])["diagnostic_workup"]]
            cache = _writer_cache(query_cache or output / "query_cache", compile_config, section)
            sql_database = sql_cache = None
            if method == "extraction_sql":
                from matchminer_ai.patients import build_extraction_database, write_extraction_database

                progress("Building the SQL agent's exploration database from the run's records")
                frames = [f for f in (prepared.documents, prepared.patient_records) if f is not None]
                frames = [f.drop(columns=[c for c in ("note_ids",) if c in f]) for f in frames]
                sql_database = write_extraction_database(output / "sql" / "exploration.duckdb",
                                                         build_extraction_database(frames), private=True)
                sql_cache = QueryCache(Path(query_cache or output / "query_cache") / "workup_sql")
            todo = [sid for sid in groups if sid not in no_workup]
            done = {"spaces": 0}

            def one(sid):
                patients, queries = assess_space(
                    spaces[sid], groups[sid], prepared, method=method, config=config,
                    compile_config=compile_config, query_cache=cache, sql_database=sql_database,
                    sql_cache=sql_cache, extensions=extensions, max_concurrent_patients=max_concurrent_patients)
                done["spaces"] += 1
                progress(f"Assessed {done['spaces']}/{len(todo)} guideline spaces")
                return sid, patients, queries

            log = output / "gateway_requests.jsonl"
            rows, compile_summary = [], {}
            with gateway_log(log), ThreadPoolExecutor(max_workers=max(1, max_concurrent_spaces)) as pool:
                for sid, patients, queries in pool.map(one, todo):
                    rows.extend(_item_rows(patients, without_history_and_physical(spaces[sid]), method, fields))
                    if queries:
                        compile_summary[sid] = queries
            items = pd.DataFrame(rows)
            exclusions = report.all_exclusions(prepared, matches, missing, {
                pid for sid in no_workup for pid in groups[sid]})
            tables = report.write_concordance_report(
                output, items=items, matches=matches, prepared=prepared, exclusions=exclusions,
                design={"method": method, "min_reasonableness": min_reasonableness, "candidate_k": candidate_k,
                        "catalogs": [[s.get("disease"), s.get("title"), s.get("version")]
                                     for s in catalog_metadata["sources"]],
                        "models": _model_identity(config), "writer": assess_identity["writer"],
                        "reviewer": assess_identity["reviewer"], "extraction_store": store,
                        "compile": _compile_totals(compile_summary), "gateway_usage": gateway_usage(log),
                        "prepared": prepared.provenance.get("identity", {}).get("anchors"),
                        "prepared_receipts": receipts, "match_receipt": match_receipt["receipt_id"],
                        "matching": match_receipt.get("content", {}).get("retrieval")})
            return tables["outputs"], {"flow": tables["flow"], "compile": _compile_totals(compile_summary),
                                       "gateway_usage": gateway_usage(log), "shared_query_cache": bool(query_cache)}

        run_stage(output, "population-concordance", assess_identity, assess,
                  upstream={MATCHES: match_receipt["receipt_id"]})
    completed_receipt(output, "population-concordance")
    return report.read_concordance_report(output)


def _compile_totals(summary):
    totals: dict = {}
    for queries in summary.values():
        for key, value in queries.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return {**totals, "spaces": len(summary)}


__all__ = ["CATEGORIES", "CAVEAT", "METHODS", "STATUS_CATEGORIES", "assess_concordance", "assess_space",
           "catalog_paths", "concordance_category", "match_guideline_spaces"]
