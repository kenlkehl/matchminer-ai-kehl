"""Outputs of the anchored concordance review: per patient and item, per patient, by provider and department.

Every file is private (``evaluation.demographics.write_private``: files 0600 in folders 0700):

- ``item_results.parquet``: one row per patient and workup item (``ITEM_COLUMNS``), with the raw status, its
  reason, the category and the evidence quotes;
- ``patient_results.parquet`` and ``.csv``: items, not applicable, applicable (met + not met), met, not met,
  unknown and concordance (met / applicable; empty when nothing applies);
- ``provider_summary.csv`` and ``department_summary.csv``: patients, inclusions and exclusions by reason, mean
  per-patient concordance with a 95% bootstrap interval over patients, the median, and pooled item concordance
  with an interval from resampling patients (``evaluation.retro_fidelity.metrics``); groups under 11 included
  patients are flagged (``small_group``), not suppressed;
- ``provider_items.csv`` and ``item_summary.csv``: rates per provider, space and item, and per space and item;
- ``exclusions.parquet``, ``flow.json`` (given, anchored, summarized, extracted, matched, assessed, with exclusion
  reasons), ``design_notes.json`` and ``report.md`` (counts and metrics only).
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .anchors import GROUP_FIELDS

SMALL_GROUP = 11
STAGES = ("anchor", "summary", "extraction", "match", "assessment")
FLOW_STEPS = ("given", "anchored", "summarized", "extracted", "matched", "assessed")
ITEM_COLUMNS = ("patient_id", *GROUP_FIELDS, "anchor_date", "space_trial_id", "space_name", "disease",
                "item_index", "item_name", "item_conditions", "item_category", "method", "raw_status", "reason",
                "applicability", "recommends_against", "category", "category_note", "bottom_line", "query_key",
                "evidence_json")
COUNTS = ("items", "not_applicable", "applicable", "met", "not_met", "unknown")
PATIENT_COLUMNS = ("patient_id", *GROUP_FIELDS, "anchor_date", "space_trial_id", "space_name", "disease",
                   "match_quality_reasonableness", *COUNTS, "concordance")
PROVIDER, DEPARTMENT = ("provider_id", "provider_name"), ("department_id", "department_name")
OUTPUTS = ("item_results.parquet", "patient_results.parquet", "patient_results.csv", "provider_summary.csv",
           "department_summary.csv", "provider_items.csv", "item_summary.csv", "exclusions.parquet", "flow.json",
           "design_notes.json", "report.md")


def _write(path, data):
    from matchminer_ai.evaluation.demographics import write_private

    write_private(Path(path), data)


def _parquet(frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def patients_without_records(prepared):
    """Prepared patients with no complete or partial document or patient record."""
    usable = set()
    for frame in (prepared.documents, prepared.patient_records):
        if frame is not None and len(frame):
            usable |= set(frame.loc[frame.extraction_status.isin(["complete", "partial"]),
                                    "patient_id"].astype(str))
    return set(prepared.summaries.patient_id) - usable


def all_exclusions(prepared, matches, no_records, no_workup):
    """Every excluded patient once, at the first stage that excluded it: ``patient_id``, the provider and
    department fields, ``stage`` (``STAGES``) and ``reason``."""
    from .concordance import NO_MATCH, NO_RECORDS, NO_SUMMARY, NO_WORKUP

    fields = prepared.anchors.set_index("patient_id")[list(GROUP_FIELDS)].to_dict("index")
    rows = prepared.exclusions.to_dict("records")
    seen = {r["patient_id"] for r in rows}

    def add(ids, stage, reason):
        for pid in sorted(set(ids) - seen):
            rows.append({"patient_id": pid, **fields[pid], "stage": stage, "reason": reason})
            seen.add(pid)

    add(set(prepared.anchors.patient_id) - set(prepared.summaries.patient_id), "summary", NO_SUMMARY)
    add(no_records, "extraction", NO_RECORDS)
    add(matches.loc[matches.excluded, "patient_id"], "match", NO_MATCH)
    add(no_workup, "assessment", NO_WORKUP)
    return pd.DataFrame(rows, columns=["patient_id", *GROUP_FIELDS, "stage", "reason"])


def flow_counts(people, exclusions):
    """Patients at each step and the exclusions by stage and reason."""
    counts, remaining = {"given": len(people)}, len(people)
    excluded = {}
    for step, stage in zip(FLOW_STEPS[1:], STAGES, strict=True):
        reasons = exclusions.loc[exclusions.stage == stage, "reason"].value_counts().sort_index()
        excluded[stage] = {str(k): int(v) for k, v in reasons.items()}
        remaining -= int(reasons.sum())
        counts[step] = remaining
    return {**counts, "excluded": excluded}


def patient_results(items, matches, anchors):
    """One row per assessed patient (``PATIENT_COLUMNS``)."""
    if items.empty:
        return pd.DataFrame(columns=list(PATIENT_COLUMNS))
    scores = matches.set_index("patient_id").match_quality_reasonableness.to_dict()
    dates = anchors.set_index("patient_id").anchor_date.to_dict()
    rows = []
    for pid, group in items.groupby("patient_id", sort=True):
        first = group.iloc[0]
        counts = group.category.value_counts()
        met, not_met = int(counts.get("met", 0)), int(counts.get("not_met", 0))
        rows.append({
            "patient_id": pid, **{k: first[k] for k in GROUP_FIELDS}, "anchor_date": dates.get(pid),
            "space_trial_id": first["space_trial_id"], "space_name": first["space_name"],
            "disease": first["disease"], "match_quality_reasonableness": scores.get(pid),
            "items": len(group), "not_applicable": int(counts.get("not_applicable", 0)),
            "applicable": met + not_met, "met": met, "not_met": not_met, "unknown": int(counts.get("unknown", 0)),
            "concordance": met / (met + not_met) if met + not_met else None,
        })
    return pd.DataFrame(rows, columns=list(PATIENT_COLUMNS))


def concordance_metrics(patients, items):
    """Mean per-patient concordance (95% bootstrap CI over patients), the median, and pooled item concordance (met
    / applicable over every applicable item, 95% CI resampling patients)."""
    from matchminer_ai.evaluation.retro_fidelity.metrics import bootstrap_mean, cluster_bootstrap_mean

    values = pd.to_numeric(patients.concordance, errors="coerce").dropna()
    applicable = items.loc[items.category.isin(["met", "not_met"])] if len(items) else items
    mean = bootstrap_mean(values.to_numpy()) if len(values) else {"mean": None, "low": None, "high": None}
    pooled = cluster_bootstrap_mean((applicable.category == "met").astype(float).to_numpy(),
                                    applicable.patient_id.astype(str).to_numpy()) if len(applicable) else \
        {"mean": None, "low": None, "high": None}
    counts = items.category.value_counts() if len(items) else pd.Series(dtype=int)
    return {
        "included": len(patients), "with_applicable": int((patients.applicable > 0).sum()) if len(patients) else 0,
        "mean_concordance": mean["mean"], "mean_ci_low": mean["low"], "mean_ci_high": mean["high"],
        "median_concordance": float(values.median()) if len(values) else None,
        "applicable_items": len(applicable), "met_items": int(counts.get("met", 0)),
        "not_met_items": int(counts.get("not_met", 0)), "unknown_items": int(counts.get("unknown", 0)),
        "not_applicable_items": int(counts.get("not_applicable", 0)),
        "pooled_concordance": pooled["mean"], "pooled_ci_low": pooled["low"], "pooled_ci_high": pooled["high"],
    }


def group_key(row, keys):
    """A provider's or department's identity, as anchors match it: its ID when the patient row gives one, else its
    name without case or repeated spaces (None when neither is given)."""
    from .anchors import _name

    key_id, name = row.get(keys[0]), row.get(keys[1])
    if key_id not in (None, "") and not (isinstance(key_id, float) and pd.isna(key_id)):
        return ("id", str(key_id))
    name = _name(name) if isinstance(name, str) else None
    return ("name", name) if name else None


def _with_groups(frame, keys, column):
    frame = frame.copy()
    frame[column] = [group_key(row, keys) for row in frame[list(keys)].to_dict("records")] if len(frame) else []
    return frame


def group_summary(people, patients, items, exclusions, keys):
    """One row per provider or department (``keys``: its ID and name fields; see :func:`group_key`), shown with its
    ID and the first name given for it."""
    reasons = sorted(exclusions.reason.unique())
    members: dict = {}
    labels: dict = {}
    for row in people.to_dict("records"):
        key = group_key(row, keys)
        members.setdefault(key, []).append(row["patient_id"])
        label = labels.setdefault(key, {k: None for k in keys})
        for field in keys:
            if label[field] in (None, "") and row.get(field) not in (None, ""):
                label[field] = row[field]
    rows = []
    for key in sorted(members, key=lambda k: ("", "") if k is None else k):
        ids = set(members[key])
        excluded = exclusions.loc[exclusions.patient_id.isin(ids)]
        metrics = concordance_metrics(patients.loc[patients.patient_id.isin(ids)],
                                      items.loc[items.patient_id.isin(ids)] if len(items) else items)
        rows.append({**labels[key], "patients": len(ids), **metrics, "excluded": len(excluded),
                     **{f"excluded_{r}": int((excluded.reason == r).sum()) for r in reasons},
                     "small_group": metrics["included"] < SMALL_GROUP})
    return pd.DataFrame(rows)


def item_rates(items, keys, by=None):
    """Counts and met / applicable per ``keys`` (such as space and item), and per provider when ``by`` names the
    provider fields (grouped by :func:`group_key`)."""
    columns = [*(by or ()), *keys, "patients", "applicable", "met", "not_met", "unknown", "not_applicable",
               "concordance"]
    if items.empty:
        return pd.DataFrame(columns=columns)
    frame = _with_groups(items, by, "_group") if by else items.assign(_group=None)
    frame["_group"] = frame["_group"].map(lambda k: "" if k is None else "\x1f".join(k))
    rows = []
    for key, group in frame.groupby(["_group", *keys], dropna=False, sort=True):
        counts = group.category.value_counts()
        met, not_met = int(counts.get("met", 0)), int(counts.get("not_met", 0))
        shown = {f: next((v for v in group[f] if v not in (None, "")), None) for f in (by or ())}
        rows.append({**shown, **dict(zip(keys, key[1:], strict=True)), "patients": group.patient_id.nunique(),
                     "applicable": met + not_met, "met": met, "not_met": not_met,
                     "unknown": int(counts.get("unknown", 0)), "not_applicable": int(counts.get("not_applicable", 0)),
                     "concordance": met / (met + not_met) if met + not_met else None})
    return pd.DataFrame(rows, columns=columns)


def _number(value, digits=2):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _interval(row, prefix):
    point = row.get(f"{prefix}_concordance")
    if point is None or pd.isna(point):
        return ""
    low, high = row.get(f"{prefix}_ci_low"), row.get(f"{prefix}_ci_high")
    return f"{point:.2f} ({low:.2f}-{high:.2f})"


def _group_table(frame, keys, label):
    lines = [f"| {label} | Patients | Included | With applicable items | Mean (95% CI) | Median | "
             "Pooled items (95% CI) | Applicable items | Unknown items | Excluded |",
             "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |"]
    for row in frame.to_dict("records"):
        name = " / ".join(str(row[k]) for k in keys if isinstance(row[k], str) and row[k]) or "(none given)"
        flag = " †" if row["small_group"] else ""
        lines.append(f"| {name}{flag} | {row['patients']} | {row['included']} | {row['with_applicable']} | "
                     f"{_interval(row, 'mean')} | {_number(row['median_concordance'])} | "
                     f"{_interval(row, 'pooled')} | {row['applicable_items']} | {row['unknown_items']} | "
                     f"{row['excluded']} |")
    return lines


def render_report(flow, overall, providers, departments, items_by_space, design, *, synthetic):
    """``report.md``: counts and metrics only, with the PHI banner and the concordance caveat."""
    from .concordance import CAVEAT

    banner = ("**Synthetic patients.** Counts and metrics only." if synthetic else
              "**Contains aggregates of real patients' records (PHI).** Internal use only; keep this report with the "
              "run's PHI. Counts and metrics only.")
    window = design.get("prepared") or {}
    lines = [
        "# Anchored guideline concordance review", "", f"> {banner}", ">", f"> {CAVEAT}", "",
        "## Run", "",
        f"- Method: `{design['method']}`; minimum P(reasonable) {design['min_reasonableness']}; TrialSpace "
        f"candidates {design['candidate_k']}; guideline catalogs: {len(design['catalogs'])}.",
        f"- Ascertainment window: {window.get('start_date') or 'open'} to {window.get('end_date') or 'open'}; "
        f"anchor seed {window.get('seed')}.",
        "", "## Flow", "", "| Step | Patients | Excluded at this step |", "| --- | ---: | --- |",
    ]
    for step, stage in zip(FLOW_STEPS, (None, *STAGES), strict=True):
        reasons = flow["excluded"].get(stage, {}) if stage else {}
        text = ", ".join(f"{reason.replace('_', ' ')}: {count}" for reason, count in reasons.items())
        lines.append(f"| {step} | {flow[step]} | {text} |")
    lines += ["", "## All patients", "",
              "| Included | With applicable items | Mean per-patient concordance (95% CI) | Median | "
              "Pooled item concordance (95% CI) | Applicable items | Met | Not met | Unknown | Not applicable |",
              "| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
              f"| {overall['included']} | {overall['with_applicable']} | {_interval(overall, 'mean')} | "
              f"{_number(overall['median_concordance'])} | {_interval(overall, 'pooled')} | "
              f"{overall['applicable_items']} | {overall['met_items']} | {overall['not_met_items']} | "
              f"{overall['unknown_items']} | {overall['not_applicable_items']} |", ""]
    note = f"† fewer than {SMALL_GROUP} included patients: flagged, not suppressed (this report is internal)."
    lines += ["## By provider", "", *_group_table(providers, PROVIDER, "Provider"), "", note, ""]
    lines += ["## By department", "", *_group_table(departments, DEPARTMENT, "Department"), "", note, ""]
    lines += ["## Workup items (most applicable first, up to 30)", "",
              "| Guideline space | Item | Patients | Applicable | Met | Not met | Unknown | Not applicable | "
              "Concordance |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    shown = items_by_space.sort_values(["applicable", "patients"], ascending=False).head(30) \
        if len(items_by_space) else items_by_space
    for row in shown.to_dict("records"):
        lines.append(f"| {row['space_name']} | {row['item_name']} | {row['patients']} | {row['applicable']} | "
                     f"{row['met']} | {row['not_met']} | {row['unknown']} | {row['not_applicable']} | "
                     f"{_number(row['concordance'])} |")
    return "\n".join(lines) + "\n"


def write_concordance_report(output, *, items, matches, prepared, exclusions, design):
    """Write every output (``OUTPUTS``) into ``output``; returns ``{"outputs", "flow"}``."""
    from .concordance import CATEGORIES, STATUS_CATEGORIES

    output = Path(output)
    items = items.reindex(columns=list(ITEM_COLUMNS)) if len(items) else pd.DataFrame(columns=list(ITEM_COLUMNS))
    if len(items):
        items["anchor_date"] = items.patient_id.map(prepared.anchors.set_index("patient_id").anchor_date)
    people = pd.concat([prepared.anchors[["patient_id", *GROUP_FIELDS]],
                        prepared.exclusions[["patient_id", *GROUP_FIELDS]]], ignore_index=True)
    patients = patient_results(items, matches, prepared.anchors)
    providers = group_summary(people, patients, items, exclusions, PROVIDER)
    departments = group_summary(people, patients, items, exclusions, DEPARTMENT)
    space_items = ("space_trial_id", "space_name", "item_index", "item_name")
    provider_items = item_rates(items, space_items, by=PROVIDER)
    item_summary = item_rates(items, space_items)
    flow = flow_counts(people, exclusions)
    overall = concordance_metrics(patients, items)
    synthetic = (prepared.provenance.get("extraction_store") or {}).get("mode") == "synthetic"
    notes = {**design, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "synthetic": synthetic, "categories": list(CATEGORIES), "status_categories": STATUS_CATEGORIES,
             "concordance": "met / (met + not met) per patient; unknown and not applicable stay out",
             "small_group": SMALL_GROUP, "overall": overall}
    _write(output / "item_results.parquet", _parquet(items))
    _write(output / "patient_results.parquet", _parquet(patients))
    _write(output / "patient_results.csv", patients.to_csv(index=False))
    _write(output / "provider_summary.csv", providers.to_csv(index=False))
    _write(output / "department_summary.csv", departments.to_csv(index=False))
    _write(output / "provider_items.csv", provider_items.to_csv(index=False))
    _write(output / "item_summary.csv", item_summary.to_csv(index=False))
    _write(output / "exclusions.parquet", _parquet(exclusions))
    _write(output / "flow.json", json.dumps(flow, indent=1))
    _write(output / "design_notes.json", json.dumps(notes, indent=1, default=str))
    _write(output / "report.md", render_report(flow, overall, providers, departments, item_summary, design,
                                               synthetic=synthetic))
    return {"outputs": list(OUTPUTS), "flow": flow}


def read_concordance_report(folder):
    """The written tables (DataFrames by name) and ``flow``."""
    folder = Path(folder)
    tables: dict = {name.split(".")[0]: pd.read_parquet(folder / name) for name in
                    ("item_results.parquet", "patient_results.parquet", "exclusions.parquet")}
    for name in ("provider_summary", "department_summary", "provider_items", "item_summary"):
        tables[name] = pd.read_csv(folder / f"{name}.csv", dtype={k: str for k in (*PROVIDER, *DEPARTMENT)})
    tables["flow"] = json.loads((folder / "flow.json").read_text())
    return tables


__all__ = ["ITEM_COLUMNS", "OUTPUTS", "PATIENT_COLUMNS", "all_exclusions", "concordance_metrics", "flow_counts",
           "group_key", "group_summary", "item_rates", "patient_results", "read_concordance_report", "render_report",
           "write_concordance_report"]
