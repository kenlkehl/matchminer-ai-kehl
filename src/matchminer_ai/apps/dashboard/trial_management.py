"""Trial management: enrollment slots, edited cohort criteria, user-defined subcohorts and group pins, kept as an
overlay on the AI-extracted cohorts (Clinical trial management's **Trial management** sidebar).

**Units.** Slots and criteria belong to a unit:

- a *merged* cohort: a dashboard cohort, which may stand for several extracted cohorts with the same criteria;
- an *original* cohort: one of those extracted cohorts (``original_cohorts``), offered when a merged cohort stands for
  two or more (``<NCT>-original-<n>``, ``<NCT>-ctgov-original-<n>``: the extraction's cohort index);
- a *subcohort* (``<NCT>-sub-<hex>``): defined by a user under a merged or an original cohort, for a sponsor's
  requirements beyond the protocol's. It starts with its parent's current criteria, which are then edited.

Each unit's slots (available, consented, enrolled) are its own: a parent's and its children's are never added up.

**Overlay.** The AI-extracted criteria are never changed. An edit of a merged or original cohort is stored with the
AI version it was based on (its name and lines; ``fingerprint``), and a subcohort with its parent's criteria. When the
trial data are re-extracted and that base changes, the edit stays in effect and is flagged (``stale``) until the user
keeps it or reverts. Reverting removes the edit, so the current AI version applies. ``effective`` gives a unit as the
store's cohorts look, with its current criteria: every matching path reads cohorts through it (``DashState.cohort``).

**Lines.** Criteria lines are kept as the cohort pipeline splits criteria (``cohort_boilerplate.split_criteria``: one
nonblank line each, under the last "Inclusion criteria:" / "Exclusion criteria:" heading above it), each with its
labels (``criterion_type``; ``boilerplate``, shown as common) and who gave them (``labeled_by``: ``ai``,
``classifier``, ``manual``; none for an unclassified line). An unchanged line keeps its labels; a changed or new line
is classified by ``criteria_labels``; an unclassified line counts as key criteria and is skipped by per-criterion
checks, as in the package.

**File.** One versioned JSON file per mode (``dashboard.<mode>.trial_management``), shared by everyone using that
dashboard, written atomically and owner-only after every change under a lock (a thread lock and a lock file), and
re-read when another process changed it. Its log records what changed and when, never criteria text. Export
downloads it; Import replaces it after validation (a file from the other mode is refused) and keeps the replaced file
beside it.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading

from .core.cache import file_stamp, text_key
from .core.errors import ConflictError, InputError

FORMAT = "mmai-dash-trial-management-v1"
KINDS = ("merged", "original", "subcohort")
HEADINGS = ("Inclusion criteria:", "Exclusion criteria:")
CRITERION_TYPES = ("inclusion", "exclusion")
LABELED_BY = ("ai", "classifier", "manual")
SLOT_FIELDS = ("available", "consented", "enrolled")
MAX_SLOTS = 1_000_000
MAX_LINES = 400
MAX_LINE = 4000
MAX_NAME = 200
MAX_SUBCOHORTS = 50
MAX_GROUPS = 100
LOG_LIMIT = 2000
_LABEL = re.compile(r"\s*(inclusion|exclusion)\s+criteria\s*:?\s*", re.IGNORECASE)
_ORIGINAL = re.compile(r"^(?P<trial>.+?)-(?:(?P<ctgov>ctgov)-)?original-(?P<index>\d+)$")
_SUBCOHORT = re.compile(r"^(?P<trial>.+?)-sub-[0-9a-f]{8}$")
_GROUP = re.compile(r"^user-[0-9a-f]{8}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty(mode: str) -> dict:
    return {"format": FORMAT, "mode": mode, "revision": 0, "updated_at": None, "units": {}, "centers": {}, "log": []}


def original_id(trial_id: str, version: str, index: int) -> str:
    return f"{trial_id}-original-{index}" if version == "protocol" else f"{trial_id}-ctgov-original-{index}"


def parse_original(unit_id: str) -> tuple[str, str, int] | None:
    match = _ORIGINAL.match(unit_id)
    if not match:
        return None
    return match.group("trial"), "ctgov" if match.group("ctgov") else "protocol", int(match.group("index"))


# ---------------------------------------------------------------------------------------------- criteria lines


def normalize_items(raw, *, manual_ok: bool = True) -> list[dict]:
    """Lines as the package splits criteria: a line's newlines make more lines, blank lines go, a typed heading
    starts its section, and a line without a heading takes the last one above it. Labels are kept only when complete
    (``labeled_by`` names who gave them); otherwise the line is unclassified."""
    if not isinstance(raw, list):
        raise InputError("criteria lines must be a list")
    out, heading = [], ""
    for entry in raw:
        if not isinstance(entry, dict):
            raise InputError("each criteria line must be an object")
        given = str(entry.get("heading") or "")
        if given and given not in HEADINGS:
            raise InputError(f"a line's heading must be one of {', '.join(HEADINGS)}")
        if given:
            heading = given
        kind, boilerplate = entry.get("criterion_type"), entry.get("boilerplate")
        by = entry.get("labeled_by")
        labeled = kind in CRITERION_TYPES and isinstance(boilerplate, bool) and by in LABELED_BY
        if by == "manual" and not manual_ok:
            labeled = False
        for line in str(entry.get("text") or "").replace("\r", "").split("\n"):
            if not line.strip():
                continue
            found = _LABEL.fullmatch(line)
            if found:
                heading = f"{found.group(1).capitalize()} criteria:"
                continue
            if len(line) > MAX_LINE:
                raise InputError(f"a criteria line is longer than {MAX_LINE} characters")
            out.append({"heading": heading, "text": line.rstrip(),
                        "criterion_type": kind if labeled else None, "boilerplate": bool(boilerplate) if labeled else False,
                        "labeled_by": by if labeled else None})
    if len(out) > MAX_LINES:
        raise InputError(f"a cohort may have at most {MAX_LINES} criteria lines")
    return out


def ai_items(items: list[dict], labeled_by: str = "ai") -> list[dict]:
    """A store cohort's tagged lines as management lines."""
    return [{"heading": i.get("heading") or "", "text": i["text"],
             "criterion_type": i.get("criterion_type") if i.get("criterion_type") in CRITERION_TYPES else None,
             "boilerplate": bool(i.get("boilerplate")) if i.get("criterion_type") in CRITERION_TYPES else False,
             "labeled_by": (i.get("labeled_by") or labeled_by) if i.get("criterion_type") in CRITERION_TYPES else None}
            for i in items]


def line_key(item: dict) -> tuple[str, str]:
    return item.get("heading") or "", item["text"].strip()


# A list marker at a line's start ("1.", "(a)", "ii)", "-"): extracted cohorts of one trial often number the same
# lines differently.
_MARKER = re.compile(r"^\s*(?:[-*\u2022\u00b7]|\(?(?:\d{1,3}|[a-zA-Z]|[ivxIVX]{1,5})[.)])\s+")


def label_key(item: dict) -> tuple[str, str]:
    """The same line for reusing labels: its heading and text, without a list marker and with spaces collapsed."""
    return item.get("heading") or "", " ".join(_MARKER.sub("", item["text"]).split())


def resolve_labels(items: list[dict], references: list[list[dict]]) -> list[dict]:
    """Each line's labels: a manual label as given; else those of the same line (heading and text, ``label_key``)
    in the first reference that has it labeled (the unit's current lines, then its AI version or its parent's);
    else none."""
    known: dict[tuple[str, str], dict] = {}
    for reference in references:
        for line in reference:
            if line.get("criterion_type") in CRITERION_TYPES:
                known.setdefault(label_key(line), line)
    out = []
    for item in items:
        if item.get("labeled_by") == "manual" and item.get("criterion_type") in CRITERION_TYPES:
            out.append(dict(item))
            continue
        found = known.get(label_key(item))
        out.append({**item, "criterion_type": found["criterion_type"] if found else None,
                    "boilerplate": bool(found["boilerplate"]) if found else False,
                    "labeled_by": (found.get("labeled_by") or "ai") if found else None})
    return out


def fingerprint(name: str | None, items: list[dict]) -> str:
    """A cohort version's identity: its name and its lines under their headings (labels left out, so classifying a
    line does not change it)."""
    return text_key("mmai-dash-cohort-version-v1", name or "", [list(line_key(i)) for i in items])[:20]


def public_items(items: list[dict]) -> list[dict]:
    """Lines as the matching paths read them (``cohort_criteria_items``)."""
    return [{"number": n, "heading": i["heading"], "text": i["text"], "criterion_type": i["criterion_type"],
             "boilerplate": bool(i["boilerplate"])} for n, i in enumerate(items, 1)]


def criteria_texts(items: list[dict]) -> tuple[str, str]:
    """(key criteria, common criteria) as the package gathers them; a cohort whose lines are all common keeps all
    its criteria as key criteria."""
    from matchminer_ai.trials.cohort_run import criteria_from_items

    lines = [{"heading": i["heading"], "text": i["text"], "boilerplate": bool(i["boilerplate"])} for i in items]
    key = criteria_from_items(lines, boilerplate=False)
    if not key:
        return criteria_from_items(lines), ""
    return key, criteria_from_items(lines, boilerplate=True)


def same_lines(a: list[dict], b: list[dict]) -> bool:
    """The same lines with the same labels."""
    def view(items):
        return [(*line_key(i), i.get("criterion_type"), bool(i.get("boilerplate"))) for i in items]

    return view(a) == view(b)


def changed_lines(before: list[dict], after: list[dict]) -> str:
    """A short, text-free summary of an edit for the log."""
    old, new = {line_key(i) for i in before}, {line_key(i) for i in after}
    added, removed = len(new - old), len(old - new)
    return f"{added} lines added or changed, {removed} removed ({len(after)} lines)"


def check_slots(values: dict) -> dict:
    out = {}
    for name in SLOT_FIELDS:
        value = values.get(name)
        if value is None or value == "":
            out[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise InputError(f"{name} must be a whole number")
        if not 0 <= int(value) <= MAX_SLOTS:
            raise InputError(f"{name} must be between 0 and {MAX_SLOTS:,}")
        out[name] = int(value)
    return out


def check_name(name, what: str = "name") -> str:
    name = " ".join(str(name or "").split())
    if not name:
        raise InputError(f"give a {what}")
    if len(name) > MAX_NAME:
        raise InputError(f"a {what} may be at most {MAX_NAME} characters")
    return name


# ---------------------------------------------------------------------------------------------- the file


def validate(doc, mode: str) -> dict:
    """The document, checked: its format and mode, and every unit, group and pin."""
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise InputError(f"not a trial management file ({FORMAT})")
    if doc.get("mode") != mode:
        raise InputError(f"this file belongs to the {doc.get('mode')} dashboard, not the {mode} one")
    units, centers, log = doc.get("units"), doc.get("centers"), doc.get("log", [])
    if not isinstance(units, dict) or not isinstance(centers, dict) or not isinstance(log, list):
        raise InputError("the file's units, centers or log are malformed")
    for unit_id, unit in units.items():
        if not isinstance(unit, dict) or unit.get("id") != unit_id or unit.get("kind") not in KINDS:
            raise InputError(f"unit {unit_id!r} is malformed")
        if not unit.get("trial_id") or unit.get("version") not in ("protocol", "ctgov"):
            raise InputError(f"unit {unit_id!r} has no trial or version")
        if unit["kind"] == "subcohort" and (not unit.get("parent") or not _SUBCOHORT.match(unit_id)
                                             or unit.get("items") is None):
            raise InputError(f"subcohort {unit_id!r} needs a parent and criteria")
        if unit["kind"] == "original" and not parse_original(unit_id):
            raise InputError(f"original cohort {unit_id!r} has a malformed ID")
        for field in ("items", "previous_items"):
            if unit.get(field) is not None:
                unit[field] = normalize_items(unit[field])
        if unit.get("slots") is not None:
            slots = unit["slots"]
            if not isinstance(slots, dict):
                raise InputError(f"unit {unit_id!r} has malformed slots")
            unit["slots"] = {**check_slots(slots), "updated_at": slots.get("updated_at")}
        if unit.get("base") is not None and not isinstance(unit["base"], dict):
            raise InputError(f"unit {unit_id!r} has a malformed base")
        unit.setdefault("revision", 0)
    for center_id, center in centers.items():
        groups, pins = center.get("groups", []), center.get("pins", {})
        if not isinstance(groups, list) or not isinstance(pins, dict):
            raise InputError(f"center {center_id!r} is malformed")
        for group in groups:
            if not isinstance(group, dict) or not _GROUP.match(str(group.get("id"))):
                raise InputError(f"center {center_id!r} has a malformed group")
            group["name"] = check_name(group.get("name"), "group name")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in pins.items()):
            raise InputError(f"center {center_id!r} has malformed pins")
        center["groups"], center["pins"] = groups, pins
    doc["log"] = [e for e in log if isinstance(e, dict)][-LOG_LIMIT:]
    doc["revision"] = int(doc.get("revision") or 0)
    return doc


class ManagementFile:
    """The mode's file: read at start, re-read when another process wrote it, written after every change."""

    def __init__(self, path: Path, mode: str):
        self.path, self.mode = Path(path), mode
        self.lock = threading.RLock()
        self.doc = empty(mode)
        self.error: str | None = None
        self._stamp: str | None = None
        self._read()

    def _read(self) -> None:
        stamp = file_stamp(self.path)
        if stamp is None:
            self.doc, self._stamp, self.error = empty(self.mode), None, None
            return
        try:
            self.doc = validate(json.loads(self.path.read_text(encoding="utf-8")), self.mode)
            self._stamp, self.error = stamp, None
        except (OSError, ValueError) as exc:  # kept as it is (never overwritten) until fixed or replaced by Import
            self._stamp, self.error = stamp, f"The trial management file could not be read: {exc}"

    def refresh(self) -> None:
        with self.lock:
            if file_stamp(self.path) != self._stamp:
                self._read()

    @contextmanager
    def _file_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with open(self.path.with_name(f".{self.path.name}.lock"), "a") as handle:
            os.fchmod(handle.fileno(), 0o600)
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _write(self, doc: dict) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(doc, stream, ensure_ascii=False, indent=1)
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        self._stamp = file_stamp(self.path)

    def snapshot(self) -> dict:
        self.refresh()
        return self.doc

    def change(self, function, entries: list[dict]):
        """Apply ``function(doc)`` to a copy of the current document and write it, with log ``entries``; returns
        the function's value."""
        with self.lock, self._file_lock():
            self._read() if file_stamp(self.path) != self._stamp else None
            if self.error:
                raise InputError(self.error + " Fix the file, or import one, before making changes.")
            doc = copy.deepcopy(self.doc)
            value = function(doc)
            at = now()
            doc["revision"] = int(doc.get("revision") or 0) + 1
            doc["updated_at"] = at
            doc["log"] = (doc.get("log") or []) + [{"at": at, **e} for e in entries]
            doc["log"] = doc["log"][-LOG_LIMIT:]
            self._write(doc)
            self.doc = doc
            return value

    def replace(self, doc: dict) -> None:
        """Import: a validated document replaces the file; the replaced one is kept beside it."""
        doc = validate(copy.deepcopy(doc), self.mode)
        with self.lock, self._file_lock():
            if self.path.exists():
                previous = self.path.with_name(f"{self.path.stem}.before-import{self.path.suffix}")
                data = self.path.read_bytes()
                fd = os.open(previous, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
            at = now()
            doc["revision"] = max(int(doc.get("revision") or 0), int(self.doc.get("revision") or 0)) + 1
            doc["updated_at"] = at
            doc["log"] = (doc.get("log") or [])[-(LOG_LIMIT - 1):] + [{"at": at, "action": "imported",
                                                                        "detail": f"{len(doc['units'])} units"}]
            self._write(doc)
            self.doc, self.error = doc, None


# ---------------------------------------------------------------------------------------------- the manager


class TrialManager:
    """The overlay over the app's trial store: effective cohorts, the views of the Trial management sidebar and the
    center groups, and every change (the routes in ``routes/management.py``)."""

    def __init__(self, state, path: Path):
        from .original_cohorts import OriginalCohorts

        self.state = state
        self.file = ManagementFile(path, state.settings.mode)
        self.originals = OriginalCohorts(state.settings)
        self.classifying: dict[str, object] = {}  # unit -> the job classifying its lines
        self.errors: dict[str, str] = {}  # unit -> why its last classification failed
        # (original cohort, its lines' fingerprint) classified automatically once (a trial opened on the pane);
        # lines still unlabeled after that are retried with Classify, not again on every open.
        self.auto_tried: set[tuple[str, str]] = set()
        self.lock = threading.Lock()

    @property
    def store(self) -> dict:
        return self.state.store

    @property
    def revision(self) -> int:
        return int(self.file.snapshot().get("revision") or 0)

    def units(self) -> dict:
        return self.file.snapshot()["units"]

    # ------------------------------------------------------------------ the AI versions

    def original_groups(self, trial_id: str, version: str):
        return self.originals.groups(trial_id, version)

    def original(self, trial_id: str, version: str, index: int) -> dict | None:
        """An extracted cohort as a store cohort (its AI version): lines labeled like its merged cohort's same lines,
        else by an earlier classification, else unclassified."""
        from matchminer_ai.trials.cohort_boilerplate import split_criteria

        from . import grouping
        from .pipeline import cohort_id

        trial = self.store["trials"].get(trial_id)
        found = self.original_groups(trial_id, version)
        if trial is None or isinstance(found, str):
            return None
        groups, members = found
        number = next((n for n, g in groups.items() if index in g), None)
        merged = self.store["cohorts"].get(cohort_id(trial_id, version, number)) if number is not None else None
        if merged is None or index not in members:
            return None
        member = members[index]
        lines = [{"heading": h, "text": t} for h, t in split_criteria(member["criteria"])]
        labels = self.cached_labels(fingerprint(member["name"], lines))
        items = resolve_labels(lines, [ai_items(merged["items"]), labels])
        cohort = {"id": original_id(trial_id, version, index), "trial_id": trial_id, "version": version,
                  "primary": False, "number": index, "name": member["name"], "merged_from": 1,
                  "merged_into": merged["id"], "lines": items}
        cohort["cancer_types"] = grouping.cohort_cancer_types({"name": member["name"], "items": public_items(items)},
                                                              trial)
        return cohort

    def cached_labels(self, key: str) -> list[dict]:
        value = self.state.cache.get(f"original_labels_{key}")
        return [i for i in value if isinstance(i, dict)] if isinstance(value, list) else []

    def ai_version(self, unit_id: str) -> dict | None:
        """The AI version of a merged or original cohort: ``{"cohort", "lines", "fingerprint"}``."""
        cohort = self.store["cohorts"].get(unit_id)
        if cohort is not None:
            lines = ai_items(cohort["items"])
        else:
            parsed = parse_original(unit_id)
            cohort = self.original(*parsed) if parsed else None
            if cohort is None:
                return None
            lines = cohort["lines"]
        return {"cohort": cohort, "lines": lines, "fingerprint": fingerprint(cohort["name"], lines)}

    # ------------------------------------------------------------------ effective cohorts

    def lines(self, unit_id: str, units: dict | None = None) -> tuple[list[dict], dict] | None:
        """A unit's current lines and its base cohort (the AI version, or a subcohort's parent's base), or None."""
        units = self.units() if units is None else units
        unit = units.get(unit_id)
        if unit is not None and unit["kind"] == "subcohort":
            parent = self.lines(unit["parent"], units)
            return unit["items"], (parent[1] if parent else None)
        ai = self.ai_version(unit_id)
        if ai is None:
            return None
        return (unit["items"] if unit and unit.get("items") is not None else ai["lines"]), ai["cohort"]

    def effective(self, unit_id: str) -> dict | None:
        """The unit with its current criteria, shaped as the store's cohorts (``key``, ``boilerplate``, ``items``),
        with its management state (``management``); None for an unknown ID."""
        units = self.units()
        unit = units.get(unit_id)
        stored = self.store["cohorts"].get(unit_id)
        if stored is not None and unit is None:
            return stored  # unedited and without slots: the store's own
        found = self.lines(unit_id, units)
        if found is None:
            return None
        lines, base = found
        if stored is not None and (unit.get("items") is None):
            cohort = dict(stored)
        else:
            from . import grouping

            key, common = criteria_texts(lines)
            source = stored or base or {}
            trial = self.store["trials"].get((unit or source).get("trial_id"))
            cohort = {**{k: v for k, v in source.items() if k not in ("lines", "merged_into")},
                      "id": unit_id, "key": key, "boilerplate": common, "items": public_items(lines)}
            if unit is not None and unit["kind"] == "subcohort":
                cohort.update(name=unit["name"], primary=False, merged_from=1, number=source.get("number", 0),
                              trial_id=unit["trial_id"], version=unit["version"], group=None, group_reason=None)
            if not stored:
                cohort.update(group=None, group_reason=None)
            cohort["cancer_types"] = grouping.cohort_cancer_types(cohort, trial) if trial is not None else []
        cohort["management"] = self.unit_state(unit_id, unit, lines, units)
        if cohort["management"]["kind"] == "original":
            cohort["management"]["parent"] = (base or {}).get("merged_into")  # the merged cohort it is part of
        return cohort

    def unit_state(self, unit_id: str, unit: dict | None, lines: list[dict], units: dict) -> dict:
        kind = unit["kind"] if unit else ("merged" if unit_id in self.store["cohorts"] else "original")
        return {"kind": kind, "parent": unit.get("parent") if unit else None,
                "edited": bool(unit and unit["kind"] != "subcohort" and unit.get("items") is not None),
                "stale": self.stale(unit_id, unit, units) is not None,
                "slots": (unit or {}).get("slots"), "unclassified": sum(1 for i in lines if not i["criterion_type"])}

    def parent_lines(self, unit: dict, units: dict) -> list[dict] | None:
        found = self.lines(unit["parent"], units)
        return found[0] if found else None

    def stale(self, unit_id: str, unit: dict | None, units: dict) -> dict | None:
        """Why the unit's edit is flagged: its AI version changed since the edit (``ai_changed``), a subcohort's
        parent's criteria changed (``parent_changed``) or its parent is gone (``parent_missing``)."""
        if not unit:
            return None
        if unit["kind"] == "subcohort":
            parent = self.parent_lines(unit, units)
            if parent is None:
                return {"reason": "parent_missing"}
            if (unit.get("base") or {}).get("fingerprint") != fingerprint(None, parent):
                return {"reason": "parent_changed", "base_lines": (unit.get("base") or {}).get("lines") or [],
                        "current_lines": parent}
            return None
        if unit.get("items") is None:
            return None
        ai = self.ai_version(unit_id)
        if ai is None:
            return {"reason": "cohort_missing"}
        if (unit.get("base") or {}).get("fingerprint") != ai["fingerprint"]:
            return {"reason": "ai_changed", "base_lines": (unit.get("base") or {}).get("lines") or [],
                    "current_lines": ai["lines"]}
        return None

    def managed_texts(self) -> list[str]:
        """The key criteria of every edited cohort and subcohort (embedded with TrialSpace)."""
        out = []
        for unit_id, unit in self.units().items():
            if unit.get("items") is not None:
                cohort = self.effective(unit_id)
                if cohort is not None:
                    out.append(cohort["key"])
        return out

    # ------------------------------------------------------------------ views

    def trial_view(self, trial_id: str) -> dict:
        """A trial opened in the Trial management sidebar: its merged cohorts, their originals and subcohorts, with slots, criteria,
        flags and groups; entries whose cohort is gone; the center's groups; the trial's recent changes."""
        from .routes.common import trial_summary

        trial = self.store["trials"].get(trial_id)
        if trial is None:
            from .state import NotFound

            raise NotFound(trial_id)
        doc = self.file.snapshot()
        units = doc["units"]
        version = trial.get("primary_version")
        groups_found = self.original_groups(trial_id, version) if version else "The trial has no extracted cohorts."
        originals_reason = groups_found if isinstance(groups_found, str) else None
        groups, _ = groups_found if not isinstance(groups_found, str) else ({}, {})
        center = self.center_groups(trial["center_id"])
        placed = {u: g for g in center for u in g["unit_ids"]}
        shown, out = set(), []

        pins = (doc["centers"].get(trial["center_id"]) or {}).get("pins", {})

        def add(unit_id: str):
            view = self.unit_view(unit_id, units, placed, pins)
            if view is not None:
                out.append(view)
                shown.add(unit_id)
            return view

        children = {}
        for unit in units.values():
            if unit["kind"] == "subcohort" and unit["trial_id"] == trial_id:
                children.setdefault(unit["parent"], []).append(unit["id"])
        for cid in trial["cohort_ids"]:
            merged = add(cid)
            if merged is None:
                continue
            number = self.store["cohorts"][cid]["number"]
            if not isinstance(merged["merged_from"], int):  # synthetic mode's store: from the merge groups
                merged["merged_from"] = max(1, len(groups.get(number, [])))
            for index in groups.get(number, []):
                originals = add(original_id(trial_id, version, index))
                if originals is not None:
                    merged["originals"].append(originals["id"])
            for parent in [cid, *merged["originals"]]:
                for sub in sorted(children.get(parent, []), key=lambda s: units[s].get("created_at") or ""):
                    if add(sub) is not None:
                        next(v for v in out if v["id"] == parent)["children"].append(sub)
        orphans = [{"id": u["id"], "kind": u["kind"], "name": u.get("name"), "slots": u.get("slots"),
                    "edited": u.get("items") is not None, "parent": u.get("parent"), "version": u["version"]}
                   for u in units.values() if u["trial_id"] == trial_id and u["id"] not in shown]
        log = [e for e in doc.get("log") or [] if e.get("trial_id") == trial_id][-30:][::-1]
        return {"trial": trial_summary(trial), "version": version, "center_id": trial["center_id"],
                "units": out, "orphans": orphans, "originals_reason": originals_reason,
                "groups": [{"id": g["id"], "name": g["name"], "custom": g["custom"]} for g in center],
                "log": log, "revision": doc["revision"]}

    def unit_view(self, unit_id: str, units: dict, placed: dict, pins: dict | None = None) -> dict | None:
        cohort = self.effective(unit_id)
        if cohort is None:
            return None
        unit = units.get(unit_id)
        found = self.lines(unit_id, units)
        lines = found[0]
        state = cohort.get("management") or self.unit_state(unit_id, unit, lines, units)
        ai = self.ai_version(unit_id) if state["kind"] != "subcohort" else None
        group = placed.get(unit_id)
        job = self.classifying.get(unit_id)
        if job is not None and not job.active:
            self.classifying.pop(unit_id, None)
            job = None
        pinned_to = (pins or {}).get(unit_id)
        return {"id": unit_id, "kind": state["kind"], "name": cohort["name"], "number": cohort.get("number"),
                "version": cohort["version"], "parent": state["parent"], "merged_from": cohort.get("merged_from", 1),
                "merged_into": (ai["cohort"].get("merged_into") if ai else None), "originals": [], "children": [],
                "lines": lines, "ai_lines": ai["lines"] if ai else None, "edited": state["edited"],
                "stale": self.stale(unit_id, unit, units), "previous_lines": (unit or {}).get("previous_items"),
                "slots": state["slots"], "unclassified": state["unclassified"], "classifying": job is not None,
                "classification_error": self.errors.get(unit_id),
                "group": {"id": group["id"], "name": group["name"], "pinned": unit_id in group["pinned"]} if group else None,
                "pin_missing": bool(pinned_to and not (group and group["id"] == pinned_to)),
                "revision": (unit or {}).get("revision", 0), "created_at": (unit or {}).get("created_at"),
                "key_count": sum(1 for i in lines if not i["boilerplate"]), "line_count": len(lines)}

    def center_groups(self, center_id: str) -> list[dict]:
        """The center's groups in order: the users' own, then the rule groups (``grouping``), with pins applied: a
        pinned merged cohort leaves its rule group, and pinned originals and subcohorts join their group. A pin to a
        group or a cohort that no longer exists is left out (the pane lists it)."""
        center = next((c for c in self.store["centers"] if c["id"] == center_id), None)
        if center is None:
            return []
        doc = self.file.snapshot()
        mine = doc["centers"].get(center_id) or {"groups": [], "pins": {}}
        groups = [{"id": g["id"], "name": g["name"], "custom": True, "unit_ids": [], "pinned": set()}
                  for g in mine["groups"]]
        groups += [{"id": g["id"], "name": g["name"], "custom": False, "unit_ids": list(g["cohort_ids"]),
                    "pinned": set()} for g in center["groups"]]
        by_id = {g["id"]: g for g in groups}
        trials = set(center["trial_ids"])
        for unit_id, group_id in mine["pins"].items():
            target = by_id.get(group_id)
            if target is None:
                continue
            cohort = self.store["cohorts"].get(unit_id)
            if cohort is None:
                cohort = self.effective(unit_id)
                if cohort is None or cohort["trial_id"] not in trials:
                    continue
            elif cohort["trial_id"] not in trials or not cohort.get("primary"):
                continue
            for group in groups:
                if unit_id in group["unit_ids"]:
                    group["unit_ids"].remove(unit_id)
            target["unit_ids"].append(unit_id)
            target["pinned"].add(unit_id)
        for group in groups:
            group["unit_ids"].sort(key=self.sort_key)
        return groups

    def sort_key(self, unit_id: str):
        """Trial order (protocol number), then each merged cohort, its originals and its subcohorts."""
        units = self.units()
        unit = units.get(unit_id)
        if unit and unit["kind"] == "subcohort":
            return (*self.sort_key(unit["parent"])[:3], 2, unit.get("created_at") or "")
        cohort = self.store["cohorts"].get(unit_id)
        if cohort is None:
            parsed = parse_original(unit_id)
            ai = self.original(*parsed) if parsed else None
            merged = self.store["cohorts"].get(ai["merged_into"]) if ai else None
            if merged is None:
                return ("", unit_id, 0, 3, "")
            trial = self.store["trials"][merged["trial_id"]]
            return (trial["protocol_no"], merged["trial_id"], merged["number"], 1, f"{ai['number']:06d}")
        trial = self.store["trials"][cohort["trial_id"]]
        return (trial["protocol_no"], cohort["trial_id"], cohort["number"], 0, "")

    def group_of_rule(self, unit_id: str) -> tuple[str, str] | None:
        cohort = self.store["cohorts"].get(unit_id)
        return (cohort.get("group"), cohort.get("group_reason")) if cohort else None

    # ------------------------------------------------------------------ changes

    def _unit(self, doc: dict, unit_id: str, revision: int | None) -> dict:
        """The unit to change in ``doc`` (made for a merged or original cohort's first change), checking that it is
        the version the user saw."""
        unit = doc["units"].get(unit_id)
        if revision is not None and (unit or {}).get("revision", 0) != revision:
            raise ConflictError("This cohort was changed by someone else since you opened it. Reload to see the "
                                "changes, then try again.")
        if unit is not None:
            return unit
        ai = self.ai_version(unit_id)
        if ai is None:
            raise InputError("This cohort is not among the trial's extracted cohorts.")
        cohort = ai["cohort"]
        unit = {"id": unit_id, "kind": "merged" if unit_id in self.store["cohorts"] else "original",
                "trial_id": cohort["trial_id"], "version": cohort["version"], "parent": None, "name": None,
                "items": None, "base": None, "previous_items": None, "slots": None, "revision": 0, "created_at": now()}
        doc["units"][unit_id] = unit
        return unit

    @staticmethod
    def _touch(unit: dict) -> None:
        unit["revision"] = int(unit.get("revision") or 0) + 1
        unit["updated_at"] = now()

    @staticmethod
    def _prune(doc: dict, unit_id: str) -> None:
        """Drop an AI cohort's entry when it holds nothing (no edit, no slots, no pin)."""
        unit = doc["units"].get(unit_id)
        pinned = any(unit_id in c.get("pins", {}) for c in doc["centers"].values())
        if unit and unit["kind"] != "subcohort" and unit.get("items") is None and unit.get("slots") is None \
                and not pinned and not any(u.get("parent") == unit_id for u in doc["units"].values()):
            del doc["units"][unit_id]

    def set_slots(self, unit_id: str, values: dict, revision: int | None = None) -> dict:
        slots = check_slots(values)

        def change(doc):
            unit = self._unit(doc, unit_id, revision)
            unit["slots"] = None if all(v is None for v in slots.values()) else {**slots, "updated_at": now()}
            self._touch(unit)
            self._prune(doc, unit_id)

        self._require_cohort(unit_id)
        detail = ", ".join(f"{k} {'-' if v is None else v}" for k, v in slots.items())
        self.file.change(change, [self._entry("slots", unit_id, detail)])
        return self.effective(unit_id)

    def _require_cohort(self, unit_id: str) -> dict:
        cohort = self.effective(unit_id)
        if cohort is None:
            from .state import NotFound

            raise NotFound(unit_id)
        return cohort

    def _entry(self, action: str, unit_id: str | None, detail: str = "", trial_id: str | None = None) -> dict:
        if trial_id is None and unit_id:
            unit = self.units().get(unit_id)
            cohort = self.store["cohorts"].get(unit_id)
            parsed = parse_original(unit_id)
            trial_id = (unit or {}).get("trial_id") or (cohort or {}).get("trial_id") or (parsed[0] if parsed else None)
        return {"action": action, "unit": unit_id, "trial_id": trial_id, "detail": detail}

    def _children(self, doc: dict, unit_id: str) -> list[dict]:
        return [u for u in doc["units"].values() if u["kind"] == "subcohort" and u.get("parent") == unit_id]

    def _propagate(self, doc: dict, unit_id: str, new_lines: list[dict], propagate: bool | None) -> int:
        """After a parent's criteria change: reset its subcohorts to them (keeping each one's previous criteria for
        reference), or let them keep theirs; either way they are based on the new criteria."""
        children = self._children(doc, unit_id)
        if children and propagate is None:
            raise InputError("This cohort has subcohorts: choose whether its new criteria replace theirs.")
        for child in children:
            if propagate:
                child["previous_items"] = child["items"]
                child["items"] = [dict(i) for i in new_lines]
            child["base"] = {"fingerprint": fingerprint(None, new_lines), "lines": new_lines}
            self._touch(child)
        return len(children)

    def set_criteria(self, unit_id: str, raw_items, *, revision: int | None = None,
                     propagate: bool | None = None) -> list[str]:
        """Save a unit's criteria (labels resolved; changed and new lines unclassified until classified). Returns
        the units whose lines changed (the unit, and its subcohorts when the change was applied to them)."""
        items = normalize_items(raw_items)
        if not items:
            raise InputError("A cohort needs at least one criteria line.")
        self._require_cohort(unit_id)
        found = self.lines(unit_id)
        current, _ = found
        units = self.units()
        unit = units.get(unit_id)
        kind = unit["kind"] if unit else None
        references = [current]
        if kind == "subcohort":
            parent = self.parent_lines(unit, units)
            references.append(parent or [])
            ai = None
        else:
            ai = self.ai_version(unit_id)
            references.append(ai["lines"])
        items = resolve_labels(items, references)
        unchanged = same_lines(items, current)
        changed: list[str] = [unit_id]

        def change(doc):
            target = self._unit(doc, unit_id, revision)
            if target["kind"] == "subcohort":
                target["items"] = items
                target["previous_items"] = None
            else:
                overlay = None if same_lines(items, ai["lines"]) else items
                target["items"] = overlay
                target["base"] = ({"fingerprint": ai["fingerprint"], "name": ai["cohort"]["name"], "lines": ai["lines"]}
                                  if overlay is not None else None)
                if not unchanged and self._propagate(doc, unit_id, items, propagate) and propagate:
                    changed.extend(c["id"] for c in self._children(doc, unit_id))
            self._touch(target)
            self._prune(doc, unit_id)

        self.file.change(change, [self._entry("criteria", unit_id, changed_lines(current, items))])
        return changed

    def revert(self, unit_id: str, *, revision: int | None = None, propagate: bool | None = None) -> list[str]:
        """A merged or original cohort back to its current AI-extracted criteria."""
        ai = self.ai_version(unit_id)
        if ai is None:
            raise InputError("Only an extracted cohort can be reverted to its AI-extracted criteria.")
        if (self.units().get(unit_id) or {}).get("items") is None:
            return []
        changed: list[str] = [unit_id]

        def change(doc):
            target = self._unit(doc, unit_id, revision)
            target["items"], target["base"] = None, None
            n = self._propagate(doc, unit_id, ai["lines"], propagate)
            if propagate and n:
                changed.extend(c["id"] for c in self._children(doc, unit_id))
            self._touch(target)
            self._prune(doc, unit_id)

        self.file.change(change, [self._entry("reverted", unit_id, "AI-extracted criteria restored")])
        return changed

    def keep(self, unit_id: str, *, revision: int | None = None) -> None:
        """Resolve a flag by keeping the unit's criteria: based from now on the current AI version (or parent)."""
        units = self.units()
        unit = units.get(unit_id)
        if unit is None:
            raise InputError("This cohort has no edits to keep.")

        def change(doc):
            target = self._unit(doc, unit_id, revision)
            if target["kind"] == "subcohort":
                parent = self.parent_lines(target, doc["units"])
                if parent is None:
                    raise InputError("This subcohort's parent cohort is no longer extracted.")
                target["base"] = {"fingerprint": fingerprint(None, parent), "lines": parent}
            else:
                ai = self.ai_version(unit_id)
                if ai is None:
                    raise InputError("This cohort is no longer extracted.")
                target["base"] = {"fingerprint": ai["fingerprint"], "name": ai["cohort"]["name"], "lines": ai["lines"]}
            self._touch(target)

        self.file.change(change, [self._entry("kept", unit_id, "edits kept on the new base")])

    def reset_to_parent(self, unit_id: str, *, revision: int | None = None) -> None:
        """A subcohort's criteria back to its parent's current ones (its own kept for reference)."""
        def change(doc):
            target = doc["units"].get(unit_id)
            if target is None or target["kind"] != "subcohort":
                raise InputError("Only a subcohort can be reset to its parent's criteria.")
            self._unit(doc, unit_id, revision)
            parent = self.parent_lines(target, doc["units"])
            if parent is None:
                raise InputError("This subcohort's parent cohort is no longer extracted.")
            target["previous_items"], target["items"] = target["items"], [dict(i) for i in parent]
            target["base"] = {"fingerprint": fingerprint(None, parent), "lines": parent}
            self._touch(target)

        self.file.change(change, [self._entry("reset", unit_id, "criteria reset to the parent's")])

    def dismiss_previous(self, unit_id: str) -> None:
        def change(doc):
            target = doc["units"].get(unit_id)
            if target is not None:
                target["previous_items"] = None
                self._touch(target)

        self.file.change(change, [])

    def create_subcohort(self, parent_id: str, name: str) -> str:
        name = check_name(name, "subcohort name")
        units = self.units()
        parent = units.get(parent_id)
        if parent is not None and parent["kind"] == "subcohort":
            raise InputError("A subcohort is defined under a merged or an original cohort.")
        found = self.lines(parent_id)
        cohort = self.effective(parent_id)
        if found is None or cohort is None:
            raise InputError("The parent is not among the trial's extracted cohorts.")
        lines = found[0]
        if sum(1 for u in units.values() if u.get("parent") == parent_id) >= MAX_SUBCOHORTS:
            raise InputError(f"A cohort may have at most {MAX_SUBCOHORTS} subcohorts.")
        unit_id = f"{cohort['trial_id']}-sub-{secrets.token_hex(4)}"

        def change(doc):
            at = now()
            doc["units"][unit_id] = {
                "id": unit_id, "kind": "subcohort", "trial_id": cohort["trial_id"], "version": cohort["version"],
                "parent": parent_id, "name": name, "items": [dict(i) for i in lines],
                "base": {"fingerprint": fingerprint(None, lines), "lines": lines}, "previous_items": None,
                "slots": None, "revision": 1, "created_at": at, "updated_at": at}

        self.file.change(change, [self._entry("subcohort created", unit_id, f"under {parent_id}", cohort["trial_id"])])
        return unit_id

    def rename_subcohort(self, unit_id: str, name: str, revision: int | None = None) -> None:
        name = check_name(name, "subcohort name")

        def change(doc):
            target = doc["units"].get(unit_id)
            if target is None or target["kind"] != "subcohort":
                raise InputError("Only a subcohort can be renamed.")
            self._unit(doc, unit_id, revision)
            target["name"] = name
            self._touch(target)

        self.file.change(change, [self._entry("subcohort renamed", unit_id)])

    def delete_unit(self, unit_id: str) -> None:
        """A subcohort, or everything kept for an AI cohort (its edit, slots and pin); a deleted cohort's subcohorts
        are deleted with it."""
        units = self.units()
        if unit_id not in units:
            raise InputError("Nothing is kept for this cohort.")
        trial_id = units[unit_id]["trial_id"]

        def change(doc):
            gone = {unit_id} | {u["id"] for u in self._children(doc, unit_id)}
            for unit in gone:
                doc["units"].pop(unit, None)
                for center in doc["centers"].values():
                    center.get("pins", {}).pop(unit, None)

        self.file.change(change, [self._entry("deleted", unit_id, "", trial_id)])

    def create_group(self, center_id: str, name: str) -> str:
        name = check_name(name, "group name")
        self._require_center(center_id)
        group_id = f"user-{secrets.token_hex(4)}"

        def change(doc):
            center = doc["centers"].setdefault(center_id, {"groups": [], "pins": {}})
            if len(center["groups"]) >= MAX_GROUPS:
                raise InputError(f"A center may have at most {MAX_GROUPS} groups of its own.")
            center["groups"].append({"id": group_id, "name": name})

        self.file.change(change, [{"action": "group created", "unit": None, "trial_id": None, "center_id": center_id,
                                   "detail": group_id}])
        return group_id

    def rename_group(self, center_id: str, group_id: str, name: str) -> None:
        name = check_name(name, "group name")

        def change(doc):
            group = next((g for g in (doc["centers"].get(center_id) or {}).get("groups", []) if g["id"] == group_id),
                         None)
            if group is None:
                raise InputError("Only a group made here can be renamed.")
            group["name"] = name

        self.file.change(change, [{"action": "group renamed", "unit": None, "trial_id": None, "center_id": center_id,
                                   "detail": group_id}])

    def delete_group(self, center_id: str, group_id: str) -> None:
        """A group made here; its cohorts go back to their rule groups."""
        def change(doc):
            center = doc["centers"].get(center_id) or {"groups": [], "pins": {}}
            if not any(g["id"] == group_id for g in center["groups"]):
                raise InputError("Only a group made here can be deleted.")
            center["groups"] = [g for g in center["groups"] if g["id"] != group_id]
            for unit_id in [u for u, g in center["pins"].items() if g == group_id]:
                del center["pins"][unit_id]
                self._prune(doc, unit_id)

        self.file.change(change, [{"action": "group deleted", "unit": None, "trial_id": None, "center_id": center_id,
                                   "detail": group_id}])

    def _require_center(self, center_id: str) -> dict:
        center = next((c for c in self.store["centers"] if c["id"] == center_id), None)
        if center is None:
            from .state import NotFound

            raise NotFound(center_id)
        return center

    def pin(self, unit_id: str, group_id: str | None) -> None:
        """Put a cohort into one of its center's groups (overriding the rules), or back to its rule group (None);
        an original or a subcohort without a pin is in no group."""
        cohort = self._require_cohort(unit_id)
        trial = self.store["trials"][cohort["trial_id"]]
        center_id = trial["center_id"]
        if group_id is not None and group_id not in {g["id"] for g in self.center_groups(center_id)}:
            raise InputError("That group is not one of this center's groups.")
        if unit_id in self.store["cohorts"] and not self.store["cohorts"][unit_id].get("primary"):
            raise InputError("Only the cohorts the dashboard shows can be put into its groups.")

        def change(doc):
            center = doc["centers"].setdefault(center_id, {"groups": [], "pins": {}})
            if group_id is None:
                center["pins"].pop(unit_id, None)
                self._prune(doc, unit_id)
            else:
                center["pins"][unit_id] = group_id

        self.file.change(change, [self._entry("pinned" if group_id else "unpinned", unit_id, group_id or "")])

    # ------------------------------------------------------------------ classification

    def unlabeled(self, unit_ids: list[str]) -> list[str]:
        out, units = [], self.units()
        for unit_id in unit_ids:
            if unit_id in self.store["cohorts"] and (units.get(unit_id) or {}).get("items") is None:
                continue  # an unedited merged cohort: the AI version as extracted
            found = self.lines(unit_id, units)
            if found and any(not i["criterion_type"] for i in found[0]):
                out.append(unit_id)
        return out

    def apply_labels(self, cohorts: list[dict], results: list[list[dict | None]]) -> int:
        """Write classified labels into the units still holding those lines unclassified (and into the trials' other
        units holding the same line, such as subcohorts that copied it); an unedited original's go to the label
        cache. Returns how many lines got labels."""
        found: dict[str, dict[tuple[str, str], dict]] = {}
        for cohort, labels in zip(cohorts, results):
            for item, label in zip(cohort["items"], labels):
                if label is not None and not item.get("criterion_type"):
                    found.setdefault(cohort["trial_id"], {})[line_key(item)] = label
        if not found:
            return 0
        counted = {"n": 0}
        for cohort in cohorts:  # unedited originals: their AI version's labels are cached
            if cohort.get("cache_key"):
                labels = found.get(cohort["trial_id"], {})
                cached = self.cached_labels(cohort["cache_key"])
                merged = {line_key(i): i for i in cached}
                for item in cohort["items"]:
                    if line_key(item) in labels:
                        merged[line_key(item)] = {"heading": item["heading"], "text": item["text"],
                                                  "labeled_by": "classifier", **labels[line_key(item)]}
                        counted["n"] += 1
                self.state.cache.put(f"original_labels_{cohort['cache_key']}", list(merged.values()))

        def change(doc):
            for unit in doc["units"].values():
                labels = found.get(unit["trial_id"])
                if not labels:
                    continue
                touched = False
                for field in ("items",):
                    for item in unit.get(field) or []:
                        label = labels.get(line_key(item))
                        if label is not None and not item.get("criterion_type"):
                            item.update(criterion_type=label["criterion_type"], boilerplate=label["boilerplate"],
                                        labeled_by="classifier")
                            counted["n"] += 1
                            touched = True
                if touched:
                    self._touch(unit)

        self.file.change(change, [])
        return counted["n"]

    def originals_to_classify(self, trial_id: str) -> list[str]:
        """The opened trial's unedited original cohorts with lines no label could be reused for (neither the merged
        cohort's nor a cached classification), not being classified and not tried automatically before."""
        trial = self.store["trials"].get(trial_id)
        version = (trial or {}).get("primary_version")
        found = self.original_groups(trial_id, version) if version else None
        if not found or isinstance(found, str):
            return []
        units = self.units()
        out = []
        for index in sorted({i for group in found[0].values() for i in group}):
            unit_id = original_id(trial_id, version, index)
            if (units.get(unit_id) or {}).get("items") is not None:
                continue  # edited: classified when saved, retried with Classify
            ai = self.ai_version(unit_id)
            job = self.classifying.get(unit_id)
            if ai is None or (job is not None and job.active) or (unit_id, ai["fingerprint"]) in self.auto_tried:
                continue
            if any(not line["criterion_type"] for line in ai["lines"]):
                out.append(unit_id)
        return out

    def mark_auto_tried(self, unit_ids: list[str]) -> None:
        for unit_id in unit_ids:
            ai = self.ai_version(unit_id)
            if ai is not None:
                self.auto_tried.add((unit_id, ai["fingerprint"]))

    def center_summary(self, center_id: str) -> dict:
        """Per trial of the center with anything kept (the Trial management list; reads the file only, never asks a
        model): its cohorts with slots and their totals, its subcohorts, edits and flags. Totals add up only cohorts of
        different families (a merged cohort, its originals and their subcohorts are one family); a trial with slots at
        two levels of one family gets none (``overlap``)."""
        center = self._require_center(center_id)
        trials = set(center["trial_ids"])
        units = self.units()
        out: dict[str, dict] = {}

        def family(unit_id: str) -> str:
            unit = units.get(unit_id)
            if unit and unit["kind"] == "subcohort":
                return family(unit["parent"])
            if parse_original(unit_id):
                ai = self.ai_version(unit_id)
                return (ai["cohort"].get("merged_into") if ai else None) or unit_id
            return unit_id

        for unit in units.values():
            if unit["trial_id"] not in trials:
                continue
            entry = out.setdefault(unit["trial_id"], {"with_slots": 0, "subcohorts": 0, "edited": 0, "stale": 0,
                                                      "families": {}})
            entry["subcohorts"] += unit["kind"] == "subcohort"
            entry["edited"] += unit["kind"] != "subcohort" and unit.get("items") is not None
            entry["stale"] += self.stale(unit["id"], unit, units) is not None
            if unit.get("slots"):
                entry["with_slots"] += 1
                entry["families"].setdefault(family(unit["id"]), []).append(unit["slots"])
        for entry in out.values():
            families = entry.pop("families")
            entry["overlap"] = any(len(v) > 1 for v in families.values())
            slots = [s for v in families.values() for s in v]
            entry["slots"] = None if entry["overlap"] or not slots else {
                k: (sum(s[k] for s in slots if s.get(k) is not None) if any(s.get(k) is not None for s in slots)
                    else None) for k in SLOT_FIELDS}
        return {"trials": out, "revision": self.revision}

    def classification_input(self, unit_ids: list[str]) -> list[dict]:
        """The cohorts to classify; a unit with the same lines as one before it (a subcohort just reset to its
        parent's) is left out, since the labels reach every unit of the trial holding the line."""
        units = self.units()
        out, seen = [], set()
        for unit_id in unit_ids:
            cohort = self.effective(unit_id)
            found = self.lines(unit_id, units)
            if cohort is None or found is None:
                continue
            same = (cohort["trial_id"], tuple(line_key(i) for i in found[0]))
            if same in seen:
                continue
            seen.add(same)
            entry = {"unit": unit_id, "trial_id": cohort["trial_id"], "name": cohort["name"],
                     "items": [dict(i) for i in found[0]]}
            unit = units.get(unit_id)
            if (unit is None or unit.get("items") is None) and parse_original(unit_id):
                ai = self.ai_version(unit_id)
                entry["cache_key"] = ai["fingerprint"]
            out.append(entry)
        return out

    # ------------------------------------------------------------------ export, import, status

    def export(self) -> bytes:
        return json.dumps(self.file.snapshot(), ensure_ascii=False, indent=1).encode("utf-8")

    def import_file(self, data: bytes) -> dict:
        try:
            doc = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise InputError("The file is not JSON.") from None
        self.file.replace(doc)
        return self.describe()

    def describe(self) -> dict:
        doc = self.file.snapshot()
        units = doc["units"].values()
        return {"path": str(self.file.path), "error": self.file.error, "revision": doc["revision"],
                "updated_at": doc.get("updated_at"), "format": FORMAT,
                "counts": {"subcohorts": sum(1 for u in units if u["kind"] == "subcohort"),
                           "edited": sum(1 for u in units if u["kind"] != "subcohort" and u.get("items") is not None),
                           "with_slots": sum(1 for u in units if u.get("slots")),
                           "groups": sum(len(c.get("groups", [])) for c in doc["centers"].values()),
                           "pins": sum(len(c.get("pins", {})) for c in doc["centers"].values())}}
