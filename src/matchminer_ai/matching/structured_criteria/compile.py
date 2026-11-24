"""Compile a trial cohort's criteria lines into criterion queries, once per distinct line (never per patient).

Each checkable line (typed inclusion or exclusion, as in ``cohort_criteria_items``) is one request with
``config.criteria_query_compile``. A key line sees the cohort's key criteria as context; a boilerplate line compiles
without context, so its query is shared by every trial with the same line. A cached compilation is revalidated and
reused without a request, as for workup queries (:func:`~matchminer_ai.patients.workup_queries.compile_workup_queries`).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping, Sequence

from matchminer_ai._storage import digest
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.patients.extraction.common import fill
from matchminer_ai.patients.extraction.schema import SCHEMA_VERSION
from matchminer_ai.patients.workup_queries.compile import compile_queries, default_extensions
from matchminer_ai.patients.workup_queries.language import (
    MAX_CRITERIA,
    MAX_GROUPS,
    MAX_LITERALS,
    QUERY_LANGUAGE_VERSION,
    render_operator_guide,
    render_query_field_guide,
)

from .language import (
    CRITERION_QUERY_VERSION,
    LINE_TYPES,
    criterion_query_response_schema,
    render_criterion_examples,
    validate_criterion_query,
)

SECTION = "criteria_query_compile"
PROMPT_FILE = "patient.criteria_query.compile.txt"


def line_text(line: dict) -> dict:
    """The fields a line's query is compiled from: ``text``, ``criterion_type`` and ``boilerplate``."""
    if not isinstance(line, dict):
        raise ValueError("Each criteria line must be a dict.")
    text = line.get("text")
    kind = line.get("criterion_type")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each criteria line needs nonempty text.")
    if kind not in LINE_TYPES:
        raise ValueError("Each compiled line must be typed inclusion or exclusion.")
    return {"text": text.strip(), "criterion_type": kind, "boilerplate": bool(line.get("boilerplate"))}


def checkable(line: dict) -> bool:
    """Whether a line can be compiled: typed inclusion or exclusion, with text (as the decision model checks)."""
    return (isinstance(line, dict) and line.get("criterion_type") in LINE_TYPES
            and isinstance(line.get("text"), str) and bool(line["text"].strip()))


def criteria_compile_contract(extensions: Iterable[str] = ()) -> dict:
    """What a compiled criterion query depends on besides its line and context."""
    extensions = list(extensions)
    return {
        "query_version": CRITERION_QUERY_VERSION,
        "language_version": QUERY_LANGUAGE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "extensions": extensions,
        "prompt_sha256": digest({
            "template": load_prompt_text(PROMPT_FILE),
            "limits": [MAX_CRITERIA, MAX_GROUPS, MAX_LITERALS],
            "field_guide": render_query_field_guide(extensions),
            "operators": render_operator_guide(),
            "examples": render_criterion_examples(),
        }),
        "response_schema_sha256": digest(criterion_query_response_schema(extensions)),
    }


def line_context(line: dict, cohort_context: str | None) -> str:
    """The context a line compiles with: the cohort's key criteria for a key line, none for a boilerplate line."""
    return "" if line_text(line)["boilerplate"] else (cohort_context or "").strip()


def criteria_query_cache_key(line: dict, cohort_context: str | None = None, *, extensions: Iterable[str] = (),
                             contract: dict | None = None) -> str:
    """The cache key of one line's compiled query."""
    contract = contract or criteria_compile_contract(extensions)
    values = line_text(line)
    return digest({"line": values["text"], "criterion_type": values["criterion_type"],
                   "context": line_context(line, cohort_context), **contract})


def render_criteria_compile_prompt(line: dict, cohort_context: str | None = None, *,
                                   extensions: Iterable[str] = ()) -> str:
    """The compile request for one line."""
    extensions = list(extensions)
    values = line_text(line)
    return fill(load_prompt_text(PROMPT_FILE), {
        "field_guide": render_query_field_guide(extensions),
        "operators": render_operator_guide(),
        "examples": render_criterion_examples(),
        "context": line_context(line, cohort_context) or "(not given: the line is checked on its own)",
        "line_type": values["criterion_type"],
        "line": values["text"],
        "max_criteria": str(MAX_CRITERIA),
        "max_groups": str(MAX_GROUPS),
        "max_literals": str(MAX_LITERALS),
    })


def compile_criteria_queries(
    lines: Sequence[dict],
    *,
    cohort_context: str | None = None,
    config: MMAIConfig | None = None,
    cache: MutableMapping[str, dict] | None = None,
    extensions: Iterable[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    return_metadata: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Compile each checkable criteria line into a validated criterion query.

    ``lines`` are ``cohort_criteria_items`` rows (``number``, ``text``, ``criterion_type``, ``boilerplate``);
    ``cohort_context`` is the cohort's key criteria, shown to the model for key lines. Returns one entry per line,
    in order: the ``line`` (with its ``number``), ``status`` (``compiled``, ``failed``, or ``unchecked`` for a line
    without a type), ``query``, ``cache_key``, ``error``, ``attempts``, ``cached``, ``model`` and the ``contract``.
    No patient data is involved.
    """
    resolved = config or load_default_preset()
    if not isinstance(resolved, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    if cohort_context is not None and not isinstance(cohort_context, str):
        raise TypeError("cohort_context must be text or None.")
    extensions = default_extensions(resolved) if extensions is None else list(extensions)
    contract = criteria_compile_contract(extensions)
    numbered = [(n, line) for n, line in enumerate(lines) if checkable(line)]
    subjects = [{**line_text(line), "number": line.get("number")} for _, line in numbered]
    keys = [criteria_query_cache_key(line, cohort_context, contract=contract) for _, line in numbered]
    compiled = compile_queries(
        subjects, keys, config=resolved, section=SECTION, schema_name="criterion_query",
        schema=criterion_query_response_schema(extensions), contract=contract,
        render=lambda subject: render_criteria_compile_prompt(subject, cohort_context, extensions=extensions),
        validate=lambda query: validate_criterion_query(query, extensions), subject="line",
        context_sha256=lambda subject: digest(line_context(subject, cohort_context)),
        cache=cache, progress_callback=progress_callback, return_metadata=return_metadata,
        label="criteria",
    )
    results, metadata = compiled if return_metadata else (compiled, None)
    by_position = {n: entry for (n, _), entry in zip(numbered, results, strict=True)}
    entries = []
    for n, line in enumerate(lines):
        if n in by_position:
            # Each line keeps its own number (a shared compilation may carry another line's).
            entry = by_position[n]
            entries.append({**entry, "line": {**entry["line"], "number": line.get("number")}})
        else:
            entries.append({"line": {"text": (line or {}).get("text"), "number": (line or {}).get("number"),
                                     "criterion_type": (line or {}).get("criterion_type"),
                                     "boilerplate": bool((line or {}).get("boilerplate"))},
                            "status": "unchecked", "query": None, "cache_key": None,
                            "error": "the criterion has no inclusion or exclusion type", "attempts": [],
                            "cached": False, "model": None, "contract": contract})
    if not return_metadata:
        return entries
    metadata["unchecked_lines"] = len(lines) - len(numbered)
    return entries, metadata


__all__ = [
    "PROMPT_FILE",
    "SECTION",
    "checkable",
    "compile_criteria_queries",
    "criteria_compile_contract",
    "criteria_query_cache_key",
    "line_context",
    "line_text",
    "render_criteria_compile_prompt",
]
