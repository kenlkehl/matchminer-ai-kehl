"""The Population tab: a browser session's patient population, its guideline-space cohort and the concordance
review (``guidelines.population``). Real mode adds the population cache and DuckDB sources."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from ..core.errors import BusyError, InputError
from ..guidelines.state import GuidelineState
from ..llm_decisions import search_connection
from .common import owner as session_owner
from ..guidelines.population import (
    PopulationInference,
    adapt_tables,
    browse_databases,
    inspect_database,
    load_population_sample,
    parse_table,
    records,
    resolve_database,
)


class PopulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1, max_length=100)


class CacheRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    path: str | None = Field(default=None, max_length=4096)


class SampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = Field(default=1000, ge=1, le=100_000, strict=True)
    seed: int = Field(default=23, ge=0, le=2**32 - 1, strict=True)


class SyntheticSampleRequest(SampleRequest):
    """Synthetic mode: ``include_training`` samples every split, not only the configured (test) split."""

    include_training: bool = Field(default=False, strict=True)


class DatabaseSampleRequest(SampleRequest):
    """PHI mode: sample from an operator-selected DuckDB database."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    database: str | None = Field(default=None, max_length=4096)


class PopulationSearch(PopulationRequest):
    catalog_id: str = Field(min_length=1, max_length=100)
    space_trial_id: str = Field(min_length=1, max_length=1000)
    candidate_k: int = Field(default=50, ge=1, le=100_000, strict=True)
    min_reasonableness: float | None = Field(default=0.75, ge=0, le=1)
    # Who selects and ranks TrialSpace's candidates: the decision model (min_reasonableness), or the LLM decision
    # checks' connection (its reasonableness answer; min_reasonableness does not apply).
    scorer: Literal["decision_model", "llm"] = "decision_model"
    confirm: bool = False  # the user confirmed a large LLM run's estimate


class PopulationReview(PopulationRequest):
    max_concurrent_patients: int = Field(default=4, ge=1, le=32, strict=True)
    # full_notes: the LLM reads every patient's notes. extraction_queries: compiled queries
    # run in code on precomputed extraction records (no patient text reaches an LLM).
    # extraction_sql: the SQL agent explores the store's extracted values (they reach the
    # concordance connection), then each accepted query runs once over the cohort in code.
    method: Literal["full_notes", "extraction_queries", "extraction_sql"] = "full_notes"
    extraction_records: str | None = Field(default=None, max_length=4096)


def router(g: GuidelineState) -> APIRouter:
    api = APIRouter()
    settings, catalogs, jobs, sessions = g.settings, g.catalogs, g.jobs, g.population_sessions
    current_settings = g.current_settings

    def submit(owner, kind, run):
        job = jobs.submit(kind, kind.replace("-", " "), run, owner=owner, device=settings.device)
        return {"job": job.describe(with_result=False)}

    def idle(owner):
        if owner in jobs.active_owners():
            raise BusyError("Wait for the current job to finish or stop it first.")

    def entry_for(owner, version=None, active=False):
        entry = sessions.get(owner)
        if not entry or (version is not None and entry["version"] != version):
            raise InputError(
                "Load the population again; this cohort is unavailable or has changed."
            )
        if active and entry["active"] is None:
            raise InputError("Fetch patients for a guideline space first.")
        return entry

    @api.get("/api/population")
    def state(request: Request):
        latest = jobs.latest(session_owner(request), prefix="population-")
        return {
            "population": sessions.view(session_owner(request)),
            "job": latest.describe() if latest else None,
            "defaults": {
                "size": settings.population_sample_size,
                "seed": settings.population_seed,
                "extraction_records": str(settings.extraction_records),
                **({} if settings.phi else {"split": settings.population_split}),
                **({"database": str(settings.population_duckdb),
                    "cache": str(settings.population_cache)} if settings.phi else {}),
            },
        }

    if settings.phi:
        @api.get("/api/population/cache")
        def cache(path: str | None = None):
            from ..guidelines.population_cache import inspect_cache

            return inspect_cache(settings, path)

        @api.post("/api/population/cache", status_code=202)
        def load_cache(body: CacheRequest, request: Request):
            from ..guidelines.population_cache import cached_population, open_cache, resolve_cache

            owner = session_owner(request)
            resolve_cache(settings, body.path)

            def run(progress):
                progress("Loading preprocessed summaries and TrialSpace embeddings")
                return sessions.save(owner, cached_population(open_cache(settings, body.path)))

            return submit(owner, "population-cache", run)

        @api.get("/api/population/databases")
        def databases(path: str | None = None):
            return browse_databases(settings, path)

        @api.get("/api/population/database")
        def database(path: str | None = None):
            return inspect_database(settings, path)

    @api.get("/api/population/extractions")
    def extractions(path: str | None = None):
        """Manifest facts of an extraction record store; no patient rows are read."""
        from ..guidelines.extraction_workup import inspect_store

        return inspect_store(settings, path)

    @api.get("/api/population/spaces")
    def spaces(catalog_id: str):
        from matchminer_ai.trials import load_guideline_catalog

        try:
            frame = load_guideline_catalog(catalogs.resolve([catalog_id]))
            columns = [
                c
                for c in (
                    "space_trial_id", "name", "cohort_key_criteria", "clinical_space_summary",
                    "disease", "title",
                )
                if c in frame
            ]
            return {"spaces": records(frame[columns])}
        except InputError:
            raise
        except Exception:
            raise InputError(
                "Could not validate this completed catalog. Refresh the library."
            ) from None

    # PHI mode samples a DuckDB database; the non-PHI app, its synthetic Parquets.
    sample_request = DatabaseSampleRequest if settings.phi else SyntheticSampleRequest

    @api.post("/api/population/sample", status_code=202)
    def sample(body: sample_request, request: Request):
        owner = session_owner(request)
        database = getattr(body, "database", None)
        if settings.phi:
            resolve_database(settings, database)

        worker = PopulationInference(current_settings(owner))

        def run(progress):
            cohort = load_population_sample(
                settings, body.size, body.seed, progress, database=database,
                include_training=getattr(body, "include_training", False),
            )
            return sessions.save(owner, cohort, embeddings=worker.embed(cohort, progress))

        return submit(owner, "population-sample", run)

    @api.post("/api/population/tables", status_code=202)
    async def tables(
        request: Request, summaries: UploadFile = File(...), notes: UploadFile = File(...)
    ):
        owner = session_owner(request)
        idle(owner)
        try:
            a = await summaries.read(settings.population_upload_limit + 1)
            b = await notes.read(settings.population_upload_limit + 1)
        finally:
            await summaries.close()
            await notes.close()
        if len(a) + len(b) > settings.population_upload_limit:
            raise InputError("Combined cohort tables exceed the 512 MB upload limit.")

        worker = PopulationInference(current_settings(owner))

        def run(progress):
            from matchminer_ai.population import create_patient_cohort

            progress("Validating patient summaries and source notes")
            try:
                s, n = adapt_tables(
                    parse_table(summaries.filename, a), parse_table(notes.filename, b)
                )
                cohort = create_patient_cohort(s, n, provenance={"source": "uploaded_tables"})
            except InputError:
                raise
            except Exception:
                raise InputError(
                    "Invalid cohort tables. Check unique patient IDs, matching notes, text and dates."
                ) from None
            return sessions.save(owner, cohort, embeddings=worker.embed(cohort, progress))

        return submit(owner, "population-tables", run)

    @api.post("/api/population/import", status_code=202)
    async def restore(request: Request, file: UploadFile = File(...)):
        """Restore an exported cohort archive, then embed it (a job; the upload is streamed to a private temporary
        folder first, which the job removes)."""
        from matchminer_ai.population import load_patient_cohort

        owner = session_owner(request)
        idle(owner)
        folder = Path(tempfile.mkdtemp(prefix="guideline-cohort-import-"))
        path = folder / "cohort.zip"
        try:
            size = 0
            with path.open("wb") as output:
                while block := await file.read(1024 * 1024):
                    size += len(block)
                    if size > settings.population_archive_limit:
                        raise InputError("Cohort archive exceeds the configured archive upload limit.")
                    output.write(block)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            await file.close()
        worker = PopulationInference(current_settings(owner))

        def run(progress):
            try:
                progress("Reading the cohort archive")
                try:
                    cohort = load_patient_cohort(path)
                except Exception:
                    raise InputError("Invalid or incompatible cohort archive. Export a fresh cohort file.") from None
            finally:
                shutil.rmtree(folder, ignore_errors=True)
            embeddings = worker.embed(cohort, progress)
            return sessions.save(owner, cohort, cohort if cohort.space else None, embeddings=embeddings)

        try:
            return submit(owner, "population-import", run)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    @api.post("/api/population/retrieve", status_code=202)
    def retrieve(body: PopulationSearch, request: Request):
        owner = session_owner(request)
        entry = entry_for(owner, body.version)
        paths = catalogs.resolve([body.catalog_id])
        worker = PopulationInference(current_settings(owner))
        llm = None
        if body.scorer == "llm":
            candidates = min(body.candidate_k, len(entry["source"].summaries))
            llm = search_connection(g, owner, candidates, body.confirm)

        def select(selected, metadata, progress):
            return g.llm_decisions.select_guideline_patients(owner, selected, metadata, llm, progress)

        def run(progress):
            active = worker.retrieve(
                entry["source"],
                paths,
                body.space_trial_id,
                body.candidate_k,
                None if llm else body.min_reasonableness,
                progress,
                embeddings=entry.get("embeddings"),
                select=select if llm else None,
            )
            return sessions.save(owner, entry["source"], active, expected=entry)

        return submit(owner, "population-retrieve", run)

    def stage(owner, body, kind, action):
        entry = entry_for(owner, body.version, active=True)
        if entry["active"].summaries.empty:
            raise InputError(
                "No patients met this selection. Adjust the candidate count or threshold."
            )
        worker = PopulationInference(current_settings(owner))

        def run(progress):
            cohort = action(worker, entry["active"], progress)
            return sessions.save(owner, entry["source"], cohort, expected=entry)

        return submit(owner, kind, run)

    @api.post("/api/population/review", status_code=202)
    def review(body: PopulationReview, request: Request):
        entry_for(session_owner(request), body.version, active=True)  # a changed cohort is reported first
        if body.method in ("extraction_queries", "extraction_sql"):
            from ..guidelines.extraction_workup import resolve_store

            path = resolve_store(settings, body.extraction_records)
            return stage(
                session_owner(request), body, "population-review",
                lambda w, c, p: w.review_extractions(c, path, p, method=body.method),
            )
        return stage(
            session_owner(request),
            body,
            "population-review",
            lambda w, c, p: w.review(c, p, max_concurrent_patients=body.max_concurrent_patients),
        )

    @api.get("/api/population/items/{number}")
    def patients_for_item(number: int, request: Request, version: str):
        cohort = entry_for(session_owner(request), version, active=True)["active"]
        if not cohort.report or not 0 <= number < len(cohort.report["aggregate"]):
            raise HTTPException(404, "This workup review is unavailable.")
        # Evidence is fetched for an individual patient, avoiding an enormous dashboard payload.
        return {
            "patients": [
                dict(
                    patient_id=p["patient_id"],
                    **{
                        k: v
                        for k, v in p["review"]["assessments"][number].items()
                        if k not in {"evidence", "metadata", "recommendation"}
                    },
                )
                for p in cohort.report["patients"]
            ]
        }

    @api.get("/api/population/patient")
    def patient(request: Request, version: str, patient_id: str):
        cohort = entry_for(session_owner(request), version, active=True)["active"]
        frame = cohort.summaries.loc[cohort.summaries.patient_id == patient_id]
        if frame.empty:
            raise HTTPException(404, "Patient is not in this active cohort.")
        review = next(
            (
                p["review"]
                for p in (cohort.report or {}).get("patients", [])
                if p["patient_id"] == patient_id
            ),
            None,
        )
        from matchminer_ai.patients.extraction.notes import note_ids

        mine = cohort.notes.loc[cohort.notes.patient_id == patient_id]
        # Extraction evidence names notes by the extraction stages' note ids.
        mine = mine.assign(note_id=note_ids(mine))
        return {
            "patient": records(frame)[0],
            "notes": records(mine),
            "review": review,
            "space": cohort.space,
        }

    @api.get("/api/population/results")
    def results(request: Request, version: str, format: Literal["json", "csv"] = "json"):
        from matchminer_ai.population import population_results_table

        cohort = entry_for(session_owner(request), version, active=True)["active"]
        if cohort.report is None:
            raise InputError("Run the population review first.")
        if format == "json":
            data = json.dumps(cohort.report, ensure_ascii=False)
            media = "application/json"
        else:
            frame = population_results_table(cohort)

            # Neutralize formula prefixes in all user/model-controlled spreadsheet cells.
            def safe_cell(value):
                return (
                    "'" + value
                    if isinstance(value, str)
                    and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
                    else value
                )

            data = frame.map(safe_cell).to_csv(index=False)
            media = "text/csv; charset=utf-8"
        return Response(
            data,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="population-results.{format}"'},
        )

    @api.get("/api/population/export")
    def export(request: Request, version: str):
        from matchminer_ai.population import save_patient_cohort

        owner = session_owner(request)
        idle(owner)
        cohort = entry_for(owner, version, active=True)["active"]
        if cohort.summaries.empty:
            raise InputError("There are no active patients to export.")
        tmp = tempfile.mkdtemp(prefix="guideline-cohort-export-")
        try:
            path = save_patient_cohort(cohort, Path(tmp) / "cohort.zip")
            return FileResponse(
                path,
                media_type="application/zip",
                filename="guideline-cohort.zip",
                background=BackgroundTask(shutil.rmtree, tmp),
            )
        except Exception:
            shutil.rmtree(tmp)
            raise

    @api.delete("/api/population", status_code=204)
    def clear(request: Request):
        owner = session_owner(request)
        with jobs.lock:
            idle(owner)
            sessions.clear(owner)
            jobs.clear(owner, prefix="population-")
        return Response(status_code=204)

    return api
