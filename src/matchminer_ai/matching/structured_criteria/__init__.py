"""Criteria-level trial screening from structured extraction records (the analogue of workup concordance checks).

Each criteria line of a cohort is compiled once, by an LLM, into a query over extracted clinical items (compiled
queries, :mod:`.compile`, or the SQL agent, :mod:`.sql`); code then runs it on each patient's extraction records with
no further model call. Results use the decision model's polarity: a line is ``clearly_fails`` only when documented
facts contradict eligibility, never because something is not documented.
"""
from .compile import (
    checkable,
    compile_criteria_queries,
    criteria_compile_contract,
    criteria_query_cache_key,
    line_text,
)
from .evaluate import (
    CHECKED,
    NOTICE,
    STATUSES,
    aggregate_structured_criteria,
    check_criteria_with_extractions,
    evaluate_criterion_query,
    line_counts,
)
from .language import CRITERION_EXAMPLES, CRITERION_QUERY_VERSION, check_criterion_query, validate_criterion_query
from .sql import CRITERIA_SQL_TASK, check_criteria_with_sql, compile_criteria_sql, criteria_sql_cache_key

__all__ = [
    "CHECKED",
    "CRITERIA_SQL_TASK",
    "CRITERION_EXAMPLES",
    "CRITERION_QUERY_VERSION",
    "NOTICE",
    "STATUSES",
    "aggregate_structured_criteria",
    "check_criteria_with_extractions",
    "check_criteria_with_sql",
    "check_criterion_query",
    "checkable",
    "compile_criteria_queries",
    "compile_criteria_sql",
    "criteria_compile_contract",
    "criteria_query_cache_key",
    "criteria_sql_cache_key",
    "evaluate_criterion_query",
    "line_counts",
    "line_text",
    "validate_criterion_query",
]
