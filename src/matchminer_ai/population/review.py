"""Anchored guideline-concordance review from the command line (``python -m matchminer_ai.population.review``).

- ``prepare``: anchors, then summaries and extraction records through each anchor (part 1;
  ``matchminer_ai.population.prepare``).
- ``assess``: each patient's guideline space and its workup items, then the concordance report (part 2;
  ``matchminer_ai.population.concordance``), on a ``prepare`` folder.
- ``run``: both, into ``<output>/prepared`` and ``<output>/assessment``.

LLMs default to Claude through the site config's Databricks gateway: Haiku 5.5 for summaries, extraction and note
reviews, Sonnet 5.5 for compiling the workup queries. ``--summary-model``, ``--extraction-model``,
``--compile-model`` and ``--review-model`` name another model of the site config's ``databricks.endpoints``
(haiku, sonnet, opus) or a full endpoint name; with ``--<task>-endpoint URL`` the task uses that OpenAI-compatible
server instead and the model option names its served model. Every note through an anchor reaches the summary and
extraction endpoints: use only endpoints authorized for the notes. Gateway token counts (never content) go to each
stage's ``gateway_requests.jsonl``; see ``python -m matchminer_ai.llm.gateway_usage``.

The outputs hold patient data: the process umask is 077, and a real run writes only under the data root's ``phi/``
or the site config's ``paths.phi_roots``. ``--synthetic`` (synthetic notes only) allows other folders, never a
``phi`` one. Logs show counts only. ``--check`` reports whether each stage is complete for these settings (exit 0
complete, 10 not yet, 20 settings or outputs differ) without running anything.

Example (synthetic notes)::

  python -m matchminer_ai.population.review run --synthetic --patients patients.csv --notes notes.parquet \\
      --start-date 2023-01-01 --end-date 2024-12-31 --output /path/to/no_phi/scratch/run1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def log(message):
    from matchminer_ai.evaluation.retro_fidelity.common import mask

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {mask(message)}", flush=True)


def connection(model, endpoint, default, thinking=None):
    """A gateway Claude model by site-config name (or endpoint name), or an OpenAI-compatible server's model."""
    from matchminer_ai.llm import connections

    if endpoint:
        if not model:
            raise ValueError("With an endpoint, name its served model too.")
        return connections.Connection(connections.LOCAL_BACKEND, model, connections.normalize_endpoint(endpoint),
                                      model, thinking)
    return connections.claude_connection(model or default)


def _base(timeout, concurrency, conn, cache):
    from matchminer_ai import load_default_preset
    from matchminer_ai.llm import connections

    config = load_default_preset()
    config.debug_mode = False
    config.model_metadata_cache_dir = str(cache)
    config.remote.update(max_concurrent_requests=concurrency, request_timeout=timeout)
    # The gateway's adaptive limit starts at 32 in flight and grows while requests succeed.
    initial = min(32, concurrency) if conn.backend == connections.CLAUDE_BACKEND else concurrency
    config.remote["adaptive_concurrency"] = {**(config.remote.get("adaptive_concurrency") or {}), "initial": initial}
    return config


def summary_config(conn, *, timeout, concurrency, cache):
    """The ``patient`` section (summaries, and the note-reading reviews) on ``conn``."""
    from matchminer_ai.llm import connections

    config = _base(timeout, concurrency, conn, cache)
    connections.connect_patient_section(config, conn)
    config.patient["prompt_build_workers"] = max(1, min(32, concurrency))
    return config


def extraction_config(conn, *, timeout, concurrency, cache):
    """The document and patient extraction sections on ``conn``."""
    from matchminer_ai.llm import connections

    config = _base(timeout, concurrency, conn, cache)
    connections.connect_remote(config, conn)
    for name in ("document_extraction", "patient_extraction"):
        connections.connect_section(getattr(config, name), name, conn)
    return config


def compile_config(conn, *, timeout, cache):
    """The query writers (compiled workup queries and the SQL agent) on ``conn``."""
    from matchminer_ai.llm import connections

    config = _base(timeout, 64, conn, cache)
    connections.connect_remote(config, conn)
    for name in ("workup_query_compile", "workup_sql_agent"):
        connections.connect_section(getattr(config, name), name, conn, own_connection=True)
    return config


def models_config(review, *, device, timeout, concurrency, cache):
    """TrialSpace and the decision model on ``device``, and the note-reading reviews on ``review``."""
    config = summary_config(review, timeout=timeout, concurrency=concurrency, cache=cache)
    config.embedding["device"] = device
    config.raw["decision_model"]["device"] = device
    return config


def _columns(values):
    mapping = {}
    for value in values or []:
        key, sep, column = value.partition("=")
        if not sep or not key.strip() or not column.strip():
            raise ValueError(f"--columns takes field=column, not {value!r}.")
        mapping[key.strip()] = column.strip()
    return mapping


def _thinking(value):
    return None if value is None else value == "on"


def prepare(args, output):
    from matchminer_ai.population.prepare import prepare_anchored_population

    cache = Path(output) / "model_metadata"
    summary = connection(args.summary_model, args.summary_endpoint, "haiku", _thinking(args.thinking))
    extraction = connection(args.extraction_model, args.extraction_endpoint, "haiku", _thinking(args.thinking))
    result = prepare_anchored_population(
        args.patients, args.notes, output_dir=output, start_date=args.start_date, end_date=args.end_date,
        seed=args.seed, columns=_columns(args.columns),
        summary_config=summary_config(summary, timeout=args.request_timeout, concurrency=args.concurrency,
                                      cache=cache),
        extraction_config=extraction_config(extraction, timeout=args.request_timeout,
                                            concurrency=args.concurrency, cache=cache),
        levels=args.levels, reuse_documents=args.reuse_documents, synthetic=args.synthetic, check=args.check,
        progress_callback=log)
    if args.check:
        return result
    flow = result.provenance["content"]
    log(f"Prepared: {len(result.anchors)} anchored, {len(result.summaries)} summarized; "
        f"extraction {flow['extractions'].get('statuses')}")
    return None


def assess(args, prepared, output):
    from matchminer_ai.population.concordance import assess_concordance
    from matchminer_ai.trials.guideline_catalog import default_guideline_catalog_root

    cache = Path(output) / "model_metadata"
    review = connection(args.review_model, args.review_endpoint, "haiku", _thinking(args.thinking))
    writer = connection(args.compile_model, args.compile_endpoint, "sonnet", _thinking(args.thinking))
    result = assess_concordance(
        prepared, catalogs=args.catalogs or default_guideline_catalog_root(), output_dir=output, method=args.method,
        min_reasonableness=args.min_reasonableness, candidate_k=args.candidate_k,
        config=models_config(review, device=args.device, timeout=args.request_timeout,
                             concurrency=args.concurrency, cache=cache),
        compile_config=compile_config(writer, timeout=args.request_timeout, cache=cache),
        query_cache=args.query_cache, embedding_cache_dir=args.embedding_cache, synthetic=args.synthetic,
        max_concurrent_spaces=args.max_concurrent_spaces, max_concurrent_patients=args.max_concurrent_patients,
        check=args.check, progress_callback=log)
    if args.check:
        return result
    flow = result["flow"]
    log("Flow: " + ", ".join(f"{step} {flow[step]}" for step in
                             ("given", "anchored", "summarized", "extracted", "matched", "assessed")))
    return None


def _report_checks(statuses):
    code = 0
    for name, (status, reason) in statuses.items():
        log(f"[check] {name}: {reason}")
        code = max(code, status)
    return code


def parser():
    top = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = top.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--output", required=True, help="the folder to write (created; private)")
        p.add_argument("--synthetic", action="store_true",
                       help="synthetic notes: the output may be outside the PHI folders (never in a phi folder)")
        p.add_argument("--thinking", choices=["on", "off"],
                       help="an OpenAI-compatible model's thinking switch (default: none sent)")
        p.add_argument("--request-timeout", type=int, default=14400, help="seconds per request (default %(default)s)")
        p.add_argument("--concurrency", type=int, default=32, help="most requests in flight (default %(default)s)")
        p.add_argument("--check", action="store_true", help="only report whether each stage is complete")

    def part1(p):
        p.add_argument("--patients", required=True, help="one row per patient (CSV or Parquet)")
        p.add_argument("--notes", required=True, help="the patients' notes (CSV or Parquet)")
        p.add_argument("--start-date", help="the ascertainment window's first day (inclusive)")
        p.add_argument("--end-date", help="the window's last day (inclusive)")
        p.add_argument("--seed", type=int, default=23, help="anchor choice seed (default %(default)s)")
        p.add_argument("--columns", nargs="+", action="extend", metavar="FIELD=COLUMN",
                       help="rename input columns (patients.FIELD= or notes.FIELD= for one table)")
        p.add_argument("--levels", choices=["document_and_patient", "document"], default="document_and_patient")
        p.add_argument("--reuse-documents", metavar="CHECKPOINT",
                       help="an extraction checkpoint folder whose matching document records are reused")
        p.add_argument("--summary-model", help="default: haiku")
        p.add_argument("--summary-endpoint", help="an OpenAI-compatible /v1 URL instead of the gateway")
        p.add_argument("--extraction-model", help="default: haiku")
        p.add_argument("--extraction-endpoint", help="an OpenAI-compatible /v1 URL instead of the gateway")

    def part2(p):
        p.add_argument("--method", default="extraction_queries",
                       choices=["extraction_queries", "extraction_sql", "full_notes", "note_search"])
        p.add_argument("--catalogs", help="a guideline catalog root, or one catalog folder (default: the site's "
                       "NCCN catalogs)")
        p.add_argument("--min-reasonableness", type=float, default=0.5,
                       help="exclude a patient whose top space's P(reasonable) is below this (default %(default)s)")
        p.add_argument("--candidate-k", type=int, default=20, help="TrialSpace candidates per patient")
        p.add_argument("--compile-model", help="default: sonnet")
        p.add_argument("--compile-endpoint", help="an OpenAI-compatible /v1 URL instead of the gateway")
        p.add_argument("--review-model", help="full_notes and note_search (default: haiku)")
        p.add_argument("--review-endpoint", help="an OpenAI-compatible /v1 URL instead of the gateway")
        p.add_argument("--device", default="cpu", help="TrialSpace and the decision model (default %(default)s)")
        p.add_argument("--query-cache", help="compiled queries shared across runs (default: the output's)")
        p.add_argument("--embedding-cache", help="guideline vectors shared across runs (default: the output's)")
        p.add_argument("--max-concurrent-spaces", type=int, default=4)
        p.add_argument("--max-concurrent-patients", type=int, default=4)

    p = commands.add_parser("prepare", help="part 1")
    common(p)
    part1(p)
    p = commands.add_parser("assess", help="part 2 on a prepare folder")
    common(p)
    p.add_argument("--prepared", required=True, help="a prepare folder")
    part2(p)
    p = commands.add_parser("run", help="both parts")
    common(p)
    part1(p)
    part2(p)
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    os.umask(0o077)
    try:
        output = Path(args.output).expanduser().resolve()
        if args.command == "prepare":
            statuses = prepare(args, output)
        elif args.command == "assess":
            statuses = assess(args, Path(args.prepared).expanduser().resolve(), output)
        else:
            statuses = prepare(args, output / "prepared")
            if args.check:
                statuses = {**statuses, **(assess(args, output / "prepared", output / "assessment")
                                           if all(code == 0 for code, _ in statuses.values()) else {})}
            else:
                statuses = assess(args, output / "prepared", output / "assessment")
    except ValueError as exc:
        from matchminer_ai.evaluation.retro_fidelity.common import mask

        sys.exit(f"error: {mask(exc)}")
    if args.check:
        sys.exit(_report_checks(statuses))


if __name__ == "__main__":
    main()
