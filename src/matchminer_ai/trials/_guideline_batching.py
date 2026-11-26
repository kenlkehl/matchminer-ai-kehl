"""Split exhausted consolidation batches without changing clinical content."""

import re

from matchminer_ai._storage import atomic_json, digest, read_json

VERSION = "adaptive-canonical-batches-v1"
MAX_RETRY_BATCH = 12


# Failures a smaller batch cannot answer: transport and server errors, credentials, cancellation, and degenerate
# repetition (an abort, not a validation finding).
NOT_VALIDATION = re.compile(
    r"Endpoint request failed|\bHTTP \d{3}\b|streaming error event|Expected one streamed completion choice"
    r"|Inference stopped|credentials|unavailable|timed out|client_aborted_repetition",
    re.I,
)


def exhausted_validation(error):
    """A call that used up its attempts on the pipeline's own validation findings (each fed back to the next attempt)."""
    return "exhausted" in error and NOT_VALIDATION.search(error) is None


def can_split(error, size):
    """Split a multi-population batch whose attempts were exhausted by validation findings, down to single
    populations; never for transport errors. Smaller batches give each retry fewer states to correct."""
    return size > 1 and exhausted_validation(error)


def validate_ledger(ledger, guideline, candidates):
    if (
        ledger.get("version") != VERSION
        or ledger.get("source_fingerprint") != guideline.fingerprint
        or ledger.get("candidates_sha256") != digest(candidates)
    ):
        raise ValueError("Canonical split plan differs from source or candidates")
    known = {c["candidate_id"]: c for c in candidates}
    for record in ledger["splits"].values():
        ids = record["input_candidate_ids"]
        if len(set(ids)) != len(ids) or set(ids) - known.keys():
            raise ValueError("Canonical split receipt contains invalid candidate membership")
        if (
            record["input_sha256"] != digest([known[i] for i in ids])
            or type(record["child_size"]) is not int
            or not 1 <= record["child_size"] < len(ids)
            or not can_split(record["reason"], len(ids))
        ):
            raise ValueError("Canonical split receipt differs from its input batch")


def run_batches(planned, guideline, candidates, output, workers, run_jobs, local, build, log,
                *, coverage_repairs=()):
    """Retain successes; bound each failed group's retry to smaller source tasks."""
    path = output / "canonical_batch_splits.json"
    identity = {
        "version": VERSION,
        "source_fingerprint": guideline.fingerprint,
        "candidates_sha256": digest(candidates),
    }
    ledger = read_json(path) if path.exists() else {**identity, "splits": {}}
    validate_ledger(ledger, guideline, candidates)

    # Older versions saved terminal failures without a split plan. Use them only
    # when both the source and exact extracted candidates still match this run.
    previous = {}
    missing = {
        candidate["candidate_id"]
        for report in coverage_repairs
        for candidate, row in zip(candidates, report["reviews"])
        if row["status"] != "represented"
    }
    saved = output / "canonical_batches.json"
    config_path, candidates_path = output / "run_config.json", output / "candidates.json"
    if saved.exists() and config_path.exists() and candidates_path.exists():
        if (
            read_json(config_path).get("source_fingerprint") == guideline.fingerprint
            and digest(read_json(candidates_path)) == digest(candidates)
        ):
            previous = read_json(saved).get("failures", {})

    def mark(key, members, error):
        ledger["splits"][key] = {
            "input_candidate_ids": [c["candidate_id"] for c in members],
            "input_sha256": digest(members),
            "reason": error,
            "child_size": min(MAX_RETRY_BATCH, max(1, len(members) // 2)),
        }
        atomic_json(path, ledger)
        log(f"{guideline.disease}: splitting exhausted {key} ({len(members)} populations) into smaller source-backed batches")

    def expand(key, data):
        members, packed = data
        record = ledger["splits"].get(key)
        error = previous.get(key, "")
        if record is None and len(members) > 1 and missing.intersection(
            c["candidate_id"] for c in members
        ):
            # A global review can expose a miss that the local judge overlooked.
            # Regenerating the same large batch can erase earlier repairs. Narrow
            # only affected leaves, retaining all pages and every candidate.
            error = "exhausted final coverage: Catalog omitted or broadened an input population"
        if record is None and can_split(error, len(members)):
            mark(key, members, error)
            record = ledger["splits"][key]
        if record is None:
            return [(key, data)]
        if (
            record["input_sha256"] != digest(members)
            or record["input_candidate_ids"] != [c["candidate_id"] for c in members]
            or type(record["child_size"]) is not int
            or not 1 <= record["child_size"] < len(members)
        ):
            raise ValueError("Canonical split receipt differs from its input batch")
        children = []
        size = record["child_size"]
        for number, start in enumerate(range(0, len(members), size), 1):
            subset = members[start:start + size]
            children.extend(expand(f"{key}-part{number:04d}", (subset, build(subset))))
        return children

    jobs = [
        child for number, data in enumerate(planned, 1)
        for child in expand(f"catalog-content-{number:04d}", data)
    ]
    values, failures = {}, {}
    while jobs:
        results, errors = run_jobs(jobs, workers, local)
        values.update(results)
        failures.update(errors)
        atomic_json(saved, {"batches": values, "failures": failures})
        next_jobs = []
        for key, data in jobs:
            error = errors.get(key, "")
            if can_split(error, len(data[0])):
                mark(key, data[0], error)
                failures.pop(key)
                next_jobs.extend(expand(key, data))
        jobs = next_jobs
    return values, failures
