"""Anchor notes of an anchored population review, and each patient's notes through the anchor.

Each patient row names a provider (by ID and/or name) and optionally a department. The patient's
anchor note is a seeded random choice among the notes that provider wrote (in that department),
within an optional ascertainment window. Everything the review reads about the patient is then
taken from the notes dated on or before the anchor (:func:`notes_through_anchor`), by any author.

No model calls; the tables are validated and nothing is written.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

import pandas as pd

PATIENT_FIELDS = ("patient_id", "provider_id", "provider_name", "department_id", "department_name")
NOTE_FIELDS = ("patient_id", "note_id", "note_date", "note_text", "note_type",
               "provider_id", "provider_name", "department_id", "department_name")
GROUP_FIELDS = ("provider_id", "provider_name", "department_id", "department_name")
#: Why a patient has no anchor note, in the order the rules apply.
ANCHOR_REASONS = {
    "no_notes": "no notes",
    "no_provider_note": "no note by the provider",
    "no_department_note": "no note by the provider in the department",
    "no_dated_note": "no dated note by the provider",
    "no_note_in_window": "no note by the provider in the ascertainment window",
}
ANCHOR_COLUMNS = ("patient_id", *GROUP_FIELDS, "anchor_note_id", "anchor_date", "anchor_time",
                  "candidate_notes", "notes_through_anchor", "undated_notes")
EXCLUSION_COLUMNS = ("patient_id", *GROUP_FIELDS, "stage", "reason")


def _text(value):
    """A stripped string, or None for a missing or blank value."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


def _name(value):
    """A name compared without case or repeated spaces."""
    value = _text(value)
    return " ".join(value.split()).casefold() if value else None


def _rename(frame, columns, table, fields):
    """``frame`` with the caller's column names mapped to ``fields``.

    ``columns`` maps a field to the caller's column; a key ``patients.<field>`` or
    ``notes.<field>`` applies to that table only.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError(f"Supply the {table} as a DataFrame with unique columns.")
    renames = {}
    for key, source in (columns or {}).items():
        scope, _, field = str(key).rpartition(".")
        if scope not in ("", table):
            if scope not in ("patients", "notes"):
                raise ValueError(f"Unknown column mapping {key!r}.")
            continue
        if field not in (*PATIENT_FIELDS, *NOTE_FIELDS):
            raise ValueError(f"Unknown column mapping {key!r}.")
        if field in fields and source in frame and source != field:
            renames[source] = field
    out = frame.rename(columns=renames)
    if not out.columns.is_unique:
        raise ValueError(f"The {table}' column mapping makes two columns with one name.")
    return out


def _timestamps(values, what):
    """Wall-clock timestamps (NaT when missing); an unreadable value is an error."""
    supplied = values.notna() & values.astype(str).str.strip().ne("")
    try:
        parsed = pd.to_datetime(values.where(supplied), errors="coerce", format="mixed")
    except ValueError:
        raise ValueError(f"{what} mix time zones; give local times.") from None
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    if (parsed.isna() & supplied).any():
        raise ValueError(f"Invalid {what}; leave unavailable dates empty.")
    return parsed


def _bound(value, what):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {what}.") from None
    if pd.isna(stamp):
        raise ValueError(f"Invalid {what}.")
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def _iso(stamp):
    return stamp.isoformat() if stamp != stamp.normalize() else stamp.strftime("%Y-%m-%d")


def _has_time(stamps):
    return stamps.notna() & stamps.ne(stamps.dt.normalize())


def _not_after(stamps, bound):
    """``stamps`` on or before ``bound``: by timestamp when both carry a time, else by day."""
    days = stamps.dt.normalize() <= bound.normalize()
    if bound == bound.normalize():
        return days
    return days.where(~_has_time(stamps), stamps <= bound)


def _not_before(stamps, bound):
    days = stamps.dt.normalize() >= bound.normalize()
    if bound == bound.normalize():
        return days
    return days.where(~_has_time(stamps), stamps >= bound)


def patient_table(patients, columns=None):
    """The validated patients table: ``patient_id`` and the provider and department fields (None when absent)."""
    frame = _rename(patients, columns, "patients", PATIENT_FIELDS)
    if "patient_id" not in frame:
        raise ValueError("The patients table needs a patient_id column.")
    if not {"provider_id", "provider_name"} & set(frame):
        raise ValueError("The patients table needs a provider_id or provider_name column.")
    out = pd.DataFrame({field: [_text(v) for v in frame[field]] if field in frame else [None] * len(frame)
                        for field in PATIENT_FIELDS}, dtype=object)
    if out.patient_id.isna().any() or out.patient_id.duplicated().any():
        raise ValueError("Patient IDs must be present and unique.")
    if (out.provider_id.isna() & out.provider_name.isna()).any():
        raise ValueError("Every patient needs a provider ID or name.")
    return out


def note_table(notes, columns=None):
    """The validated notes table with ``note_id`` (supplied, else ``extraction.notes.note_ids``) and parsed
    ``_timestamp``; rows without text are dropped and exact duplicate notes kept once."""
    from matchminer_ai.patients.extraction.notes import note_ids

    frame = _rename(notes, columns, "notes", NOTE_FIELDS)
    missing = [c for c in ("patient_id", "note_date", "note_text") if c not in frame]
    if missing:
        raise ValueError(f"The notes table needs columns {missing}.")
    frame = frame[[c for c in NOTE_FIELDS if c in frame]].copy()
    if frame.patient_id.isna().any():
        raise ValueError("Every note needs a patient_id.")
    frame["patient_id"] = [str(p).strip() for p in frame.patient_id]
    frame = frame.loc[[isinstance(t, str) and bool(t.strip()) for t in frame.note_text]].copy()
    if "note_type" not in frame:
        frame["note_type"] = None
    frame["_timestamp"] = _timestamps(frame.note_date, "note dates")
    if "note_id" in frame:
        if frame.note_id.map(_text).isna().any():
            raise ValueError("note_id must be present for every note when supplied.")
        frame["note_id"] = frame.note_id.map(_text)
        if frame.note_id.duplicated().any():
            raise ValueError("note_id must be unique.")
    else:
        frame["note_id"] = note_ids(frame)
        frame = frame.drop_duplicates("note_id")
    for field in GROUP_FIELDS:
        if field in frame:
            frame[field] = frame[field].map(_name if field.endswith("_name") else _text)
    return frame.reset_index(drop=True)


def _rows_matching(group, patient, kind):
    """Which of a patient's notes match the patient row's ``kind`` (provider or department), or None when the row
    names none: by ID when it gives one, else by name."""
    key, name = patient[f"{kind}_id"], patient[f"{kind}_name"]
    if key is not None:
        if f"{kind}_id" not in group:
            raise ValueError(f"A patient names a {kind} by ID, but the notes have no {kind}_id column.")
        return group[f"{kind}_id"].eq(key)
    if name is not None:
        if f"{kind}_name" not in group:
            raise ValueError(f"A patient names a {kind} by name, but the notes have no {kind}_name column.")
        return group[f"{kind}_name"].eq(_name(name))
    return None


def through_mask(stamps, anchor_stamp):
    """Notes dated on or before the anchor: by timestamp when both carry a time, else by day; undated notes are
    left out."""
    return stamps.notna() & _not_after(stamps, anchor_stamp)


def choose_anchor_notes(patients, notes, *, start_date=None, end_date=None, seed=23, columns=None):
    """Each patient's anchor note: ``(anchors, exclusions)``.

    ``patients`` has one row per patient: ``patient_id``, ``provider_id`` and/or
    ``provider_name`` (at least one per row), and optionally ``department_id`` and/or
    ``department_name``. ``notes`` has ``patient_id``, ``note_date``, ``note_text``,
    optionally ``note_type`` and ``note_id`` (default: the extraction stages' code-derived
    id, so evidence links back), and the note's provider and department fields.
    ``columns`` maps these names to the caller's (``patients.<field>`` or
    ``notes.<field>`` for one table only).

    A note is a candidate when it is the patient's, written by the patient's provider
    (matched on ID when the patient row gives one, otherwise on name, ignoring case and
    spacing), in the patient's department when the row gives one, dated, and within
    ``start_date``-``end_date`` (each optional and inclusive; a bound without a time
    covers its whole day). The anchor is ``random.Random(f"{seed}:{patient_id}")``'s
    choice among the candidates in date and note-ID order, so it does not depend on row
    order.

    ``anchors``: ``patient_id``, the patient row's provider and department fields,
    ``anchor_note_id``, ``anchor_date`` (YYYY-MM-DD), ``anchor_time`` (ISO timestamp,
    None for a date without a time), ``candidate_notes``, ``notes_through_anchor`` and
    ``undated_notes`` (left out). ``exclusions``: ``patient_id``, the same fields,
    ``stage`` (``anchor``) and ``reason`` (a key of ``ANCHOR_REASONS``).
    """
    people = patient_table(patients, columns)
    frame = note_table(notes, columns)
    start = _bound(start_date, "start date")
    end = _bound(end_date, "end date")
    if start is not None and end is not None and start > end:
        raise ValueError("The start date is after the end date.")
    groups = {pid: group for pid, group in frame.groupby("patient_id", sort=False)}
    anchors, exclusions = [], []
    for patient in people.to_dict("records"):
        pid = patient["patient_id"]
        fields = {k: patient[k] for k in GROUP_FIELDS}

        def exclude(reason, pid=pid, fields=fields):
            exclusions.append({"patient_id": pid, **fields, "stage": "anchor", "reason": reason})

        group = groups.get(pid)
        if group is None or group.empty:
            exclude("no_notes")
            continue
        mask = _rows_matching(group, patient, "provider")
        if not mask.any():
            exclude("no_provider_note")
            continue
        department = _rows_matching(group, patient, "department")
        if department is not None:
            mask &= department
            if not mask.any():
                exclude("no_department_note")
                continue
        stamps = group["_timestamp"]
        mask &= stamps.notna()
        if not mask.any():
            exclude("no_dated_note")
            continue
        if start is not None:
            mask &= _not_before(stamps, start)
        if end is not None:
            mask &= _not_after(stamps, end)
        if not mask.any():
            exclude("no_note_in_window")
            continue
        candidates = group.loc[mask].sort_values(["_timestamp", "note_id"], kind="stable")
        chosen = candidates.iloc[random.Random(f"{seed}:{pid}").randrange(len(candidates))]
        stamp = chosen["_timestamp"]
        timed = stamp != stamp.normalize()
        anchors.append({
            "patient_id": pid, **fields,
            "anchor_note_id": chosen["note_id"],
            "anchor_date": stamp.strftime("%Y-%m-%d"),
            "anchor_time": _iso(stamp) if timed else None,
            "candidate_notes": int(len(candidates)),
            "notes_through_anchor": int(through_mask(stamps, stamp).sum()),
            "undated_notes": int(stamps.isna().sum()),
        })
    return (pd.DataFrame(anchors, columns=list(ANCHOR_COLUMNS), dtype=object),
            pd.DataFrame(exclusions, columns=list(EXCLUSION_COLUMNS), dtype=object))


def anchor_timestamps(anchors):
    """Each anchor's timestamp by patient: its time when it has one, else its day."""
    return {row["patient_id"]: pd.Timestamp(row["anchor_time"] or row["anchor_date"])
            for row in anchors.to_dict("records")}


def notes_through_anchor(notes, anchors, *, columns=None):
    """Every note of each anchored patient, by any author, dated on or before the anchor (inclusive).

    When both the note and the anchor carry a time, timestamps are compared (a note at the
    anchor's timestamp is included); otherwise days are. Undated notes are left out.
    Returns ``patient_id``, ``note_id``, ``note_date`` (YYYY-MM-DD, or an ISO timestamp when the note has a
    time), ``note_type`` and
    ``note_text`` in patient and date order; ``notes`` may already be a :func:`note_table`.
    """
    frame = notes if "_timestamp" in notes else note_table(notes, columns)
    stamps = anchor_timestamps(anchors) if not isinstance(anchors, Mapping) else dict(anchors)
    frame = frame.loc[frame.patient_id.isin(set(stamps))]
    keep = pd.Series(False, index=frame.index)
    for pid, group in frame.groupby("patient_id", sort=False):
        keep.loc[group.index] = through_mask(group["_timestamp"], stamps[pid]).to_numpy()
    out = frame.loc[keep].sort_values(["patient_id", "_timestamp", "note_id"], kind="stable").copy()
    out["note_date"] = [_iso(s) for s in out["_timestamp"]]
    out["note_type"] = [_text(t) for t in out["note_type"]]
    return out[["patient_id", "note_id", "note_date", "note_type", "note_text"]].reset_index(drop=True)


__all__ = ["ANCHOR_REASONS", "NOTE_FIELDS", "PATIENT_FIELDS", "choose_anchor_notes", "note_table",
           "notes_through_anchor", "patient_table"]
