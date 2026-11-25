"""The assistant's tools: read-only searches over the running dashboard's own data, in its mode.

Each tool calls the app's own services, so it finds what the tabs find:

- ``search_trials``: the store's dashboard cohorts (the version each trial shows), matched with the patient and cohort
  lists' regular-expression search (``core.textsearch``), and the Clinical trial management filters (cancer type, center,
  phase, open status, GoodOption drugs and classes);
- ``match_trials``: TrialSpace retrieval from a patient description, embedded as the Patient tab embeds an entered
  summary and searched as Trials for a patient searches (one cohort per trial), optionally scored by the decision
  model;
- ``trial_details``: a trial's or cohort's store entry (key and boilerplate lines);
- ``search_patients``: the patient list's search (``/api/patients``);
- ``match_patients``: Find patients (a cohort's TrialSpace vector), or an ad-hoc criteria text embedded with the
  cohort prompt, optionally scored by the decision model;
- ``patient_details``: the patient's summary sections and, when the mode's extraction store holds the patient, its
  records' disease course (the Structured data view's);
- ``screen_records``: Screen with structured data (``matching.structured_criteria``, compiled queries written by the
  session's query-writing connection, run in code over the store's records), for a cohort's lines or ad-hoc lines;
- ``score_pairs``: the decision model's match and boilerplate questions;
- ``guideline_spaces``: the Patient tab's guideline-space retrieval.

Results are compact (IDs, short fields, totals; at most ``Limits.max_rows`` rows) and every ID they name is kept for
the page's links. Nothing is written to disk except what the tabs already write: decision-model scores (keyed by
text digests) and compiled queries of a cohort's own lines; an ad-hoc line's compiled query is kept in the
conversation's memory only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import threading
import time

import numpy as np

from ..core import cancer_types
from ..core.decisions import PairText
from ..core.encoder import EncoderUnavailable
from ..core.errors import InputError
from ..core.textsearch import TextIndex, parse_terms
from ..guidelines import records as rec
from ..guidelines.settings import SQL_TASK
from ..state import NotFound
from .actions import Limits

STATUSES = ("meets", "clearly_fails", "not_documented", "uncertain", "conflicting", "not_applicable",
            "not_checkable", "no_records", "query_unavailable")
ROMAN = {"IV": "4", "III": "3", "II": "2", "I": "1"}


class ToolError(Exception):
    """A tool could not run as asked; the message (the app's own text) goes back to the model."""


@dataclass
class ToolResult:
    tool: str
    ok: bool
    #: One line for the page: what was done and what came back.
    summary: str
    #: What the model reads next.
    result: dict
    #: IDs the result names: {id: {"kind": "trial" | "cohort" | "patient", ...}}.
    links: dict = field(default_factory=dict)
    seconds: float = 0.0
    #: The LLM answers a scoring step got (the page opens each one; the model does not see the keys).
    llm_answers: list = field(default_factory=list)


def short(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def phases_of(label: str) -> set[str]:
    """{"1", "2"} from "Phase I/II", "Phase 1/2" or "Early Phase 1"."""
    found = set(re.findall(r"\d", label or ""))
    for roman in re.findall(r"\b(I{1,3}V?|IV)\b", (label or "").upper()):
        found.add(ROMAN.get(roman, ""))
    return found - {""}


_INDEXES: dict[tuple, tuple[list[str], TextIndex]] = {}
_INDEX_LOCK = threading.Lock()


def trial_index(store: dict, common: bool) -> tuple[list[str], TextIndex]:
    """The dashboard cohorts' search texts (built once per store and scope): trial ID, protocol number, titles,
    cohort name, key criteria (and boilerplate), drug names and classes."""
    key = (store["built_at"], len(store["cohorts"]), common)
    with _INDEX_LOCK:
        if key in _INDEXES:
            return _INDEXES[key]
    ids, texts = [], []
    for cohort in store["cohorts"].values():
        if not cohort.get("primary"):
            continue
        trial = store["trials"][cohort["trial_id"]]
        drugs = [part for d in trial.get("drugs") or [] for part in (d.get("name"), d.get("listed_name"),
                                                                      *(d.get("all_classes") or []))]
        parts = [trial["id"], trial.get("nct_id"), trial.get("protocol_no"), trial.get("short_title"),
                 trial.get("long_title"), trial.get("official_title"), cohort.get("name"), cohort.get("key"),
                 *(p for p in drugs if p)]
        if common:
            parts.append(cohort.get("boilerplate"))
        ids.append(cohort["id"])
        texts.append("\n".join(str(p) for p in parts if p))
    value = (ids, TextIndex(texts))
    with _INDEX_LOCK:
        while len(_INDEXES) >= 4:
            _INDEXES.pop(next(iter(_INDEXES)))
        _INDEXES[key] = value
    return value


def status_note(state) -> str:
    """What a trial's status means in this mode (the assistant must not imply more)."""
    if state.settings.real:
        built = time.strftime("%Y-%m-%d", time.localtime(state.store.get("built_at") or time.time()))
        return ("Each trial's status is the institution's own accrual status (OnCore) from the OA trial-table "
                f"snapshot the store was built from (store built {built}); 'Open to accrual' (open: true) means open "
                "at this institution. Recruitment at other sites is not in the data. " + SLOTS_NOTE)
    return ("Each trial's status is its ClinicalTrials.gov overall status from the synthetic run's export (trials first "
            "posted 2020 onward); open: true means Recruiting or Not yet recruiting. It is not site-specific and may be "
            "out of date. " + SLOTS_NOTE)


SLOTS_NOTE = ("Cohort-level slots are only those staff entered on Clinical trial management: trial_details shows them "
              "as a cohort's slots (available, consented, enrolled, and when they were updated); a cohort without "
              "them has none entered, which says nothing about its availability.")


class Toolbox:
    """The tools of one session's conversation, over the running app's state."""

    def __init__(self, state, owner: str, limits: Limits, memory_cache: dict, *, scorer: str = "decision_model",
                 llm_connection=None):
        self.state, self.g, self.owner, self.limits = state, state.guidelines, owner, limits
        #: Ad-hoc criteria lines' compiled queries (this conversation only; never on disk).
        self.memory_cache = memory_cache
        #: Where scores come from: the session's "Scores from" choice when the message was sent, and for the LLM
        #: the session's LLM decision checks connection then.
        self.scorer, self.llm_connection = scorer, llm_connection
        self.answers: list = []

    # ------------------------------------------------------------------ what is available

    def unavailable(self) -> dict[str, str]:
        """Tools that cannot run in this app now, and why (fixed texts: the app's own errors can name paths)."""
        s, out = self.state, {}
        if s.vectors is None:
            for name in ("match_trials", "match_patients"):
                out[name] = "no valid TrialSpace round"
        elif not s.encoder.available:
            out["match_trials"] = "the TrialSpace query encoder is unavailable"
        if s.patient_vectors is None and "match_patients" not in out:
            out["match_patients"] = "the patients' TrialSpace vectors are unavailable (see Data and models)"
        if not s.corpus.patients:
            for name in ("search_patients", "patient_details", "match_patients"):
                out[name] = "no patients are loaded (see Data and models)"
        if self.g.records.get() is None:
            out["screen_records"] = "no structured record store in this mode"
        try:
            if not self.g.catalogs.list():
                out["guideline_spaces"] = "no completed guideline catalogs"
        except Exception:
            out["guideline_spaces"] = "the guideline catalogs could not be read"
        return out

    def available(self) -> list[str]:
        from .actions import TOOLS

        missing = self.unavailable()
        return [name for name in TOOLS if name not in missing]

    def overview(self) -> dict:
        """The data overview the model is given with every request."""
        s, store = self.state, self.state.store
        shown = [t for t in store["trials"].values() if t["cohort_ids"]]
        records = self.g.records.describe()
        with_drugs = sum(1 for t in shown if t.get("drugs"))
        try:
            catalogs = [f"{c['title']} ({c['spaces']} spaces)" for c in self.g.catalogs.list()]
        except Exception:
            catalogs = []
        return {
            "mode": s.settings.mode,
            "patients_are": "real patients (a population sample; PHI)" if s.settings.real else "synthetic patients",
            "trials": {"with_dashboard_cohorts": len(shown), "open": sum(1 for t in shown if t.get("is_open")),
                       "dashboard_cohorts": store["counts"]["dashboard_cohorts"],
                       "version_note": "Each trial shows its protocol cohorts when extracted, else its "
                                       "ClinicalTrials.gov listing's cohorts.",
                       "status_note": status_note(s)},
            "centers": [{"id": c["id"], "name": c["short_name"], "trials": len(c["trial_ids"])}
                        for c in store["centers"]],
            "cancer_type_filters": cancer_types.TYPE_LABELS,
            "cancer_type_note": "Cancer-type filters are keyword screens of cohorts' key inclusion lines and of the "
                                "summaries' cancer-type header, for browsing; they can miss or over-include.",
            "patients": {"count": len(s.corpus.patients), "loaded": s.patients_error is None,
                         "summaries": "Each patient has an LLM-written cancer-history summary (header lines Age, Sex, "
                                      "Cancer type, Histology, then extent, biomarkers and treatment history) and "
                                      "other relevant history (common-exclusion evidence)."},
            "structured_records": ({"patients": records.get("patients"), "schema": records.get("schema_version"),
                                    "extracted_by": records.get("model")} if records.get("available")
                                   else {"available": False}),
            "goodoption_drugs": (f"{with_drugs} trials have catalog drug lists (names, roles, NCIt classes)"
                                 if with_drugs else "no GoodOption drug catalog in this mode: trials have no drug "
                                                    "lists"),
            "guideline_catalogs": catalogs[:30],
            "unavailable_tools": self.unavailable(),
        }

    # ------------------------------------------------------------------ running a call

    def run(self, call: dict, progress) -> ToolResult:
        name = call["tool"]
        started = time.monotonic()
        try:
            self.answers = []
            summary, result, links = getattr(self, name)(call, progress)
            out = ToolResult(name, True, summary, result, links, llm_answers=self.answers)
        except (ToolError, InputError) as exc:
            out = ToolResult(name, False, f"{LABELS[name]}: {exc}", {"error": str(exc)})
        except EncoderUnavailable:
            out = ToolResult(name, False, f"{LABELS[name]}: the TrialSpace query encoder is unavailable",
                             {"error": "the TrialSpace query encoder is unavailable in this app"})
        except NotFound as exc:
            out = ToolResult(name, False, f"{LABELS[name]}: unknown ID {exc.args[0]!r}",
                             {"error": f"unknown ID {exc.args[0]!r}"})
        except ValueError:  # its type only: a library's message may quote data
            out = ToolResult(name, False, f"{LABELS[name]}: the tool failed (ValueError)",
                             {"error": "the tool failed (ValueError)"})
        out.seconds = round(time.monotonic() - started, 1)
        return out

    # ------------------------------------------------------------------ helpers

    def _trial(self, trial: dict) -> dict:
        out = {"trial_id": trial["id"], "title": short(trial.get("short_title") or trial.get("long_title"), 160),
               "phase": trial.get("phase") or "", "status": trial.get("status") or "",
               "open": bool(trial.get("is_open")), "center": trial.get("center_short_name") or trial.get("center")}
        if trial.get("protocol_no"):
            out["protocol_no"] = trial["protocol_no"]
        return out

    def _trial_link(self, trial: dict) -> dict:
        return {"kind": "trial", "center_id": trial.get("center_id"), "open": bool(trial.get("is_open"))}

    def _cohort_link(self, cohort: dict) -> dict:
        trial = self.state.store["trials"][cohort["trial_id"]]
        return {"kind": "cohort", "trial_id": trial["id"], "center_id": trial.get("center_id"),
                "open": bool(trial.get("is_open"))}

    def _patient(self, pid: str, held=None) -> dict:
        s = self.state.patient_summary(pid)
        out = {"patient_id": pid, **{k: s.get(k) for k in ("age", "sex", "cancer_type", "histology") if s.get(k)}}
        if held is not None:
            out["has_records"] = pid in held
        return out

    def _cohort_mask(self, cohorts: list[dict], cancer_type: str, open_only: bool) -> np.ndarray:
        if cancer_type and cancer_type not in cancer_types.TYPE_LABELS:
            raise ToolError(f"unknown cancer type {cancer_type!r}")
        trials = self.state.store["trials"]
        return np.array([(not cancer_type or cancer_type in c["cancer_types"])
                         and (not open_only or bool(trials[c["trial_id"]].get("is_open"))) for c in cohorts], dtype=bool)

    def _patient_mask(self, cancer_type: str, sex: str, with_records: bool):
        mask = self.state.patient_mask(cancer_type, sex, "")
        if with_records:
            held = self.state.records_mask()
            mask = held if mask is None else mask & held
        return mask

    def _k(self, value: int) -> int:
        return max(1, min(int(value or 10), self.limits.max_rows))

    @property
    def llm(self) -> bool:
        return self.scorer == "llm"

    def scorer_label(self) -> str:
        return f"the LLM ({self.llm_connection.label})" if self.llm else "the decision model"

    def score_note(self) -> str:
        """How a result's scores read, for the model."""
        if self.llm:
            return (f"match: {self.scorer_label()}'s 0-6 match score from five yes/no questions (llm_score, out of 6) "
                    f"and its reasonableness answer; the user chose LLM scores. Only the first {self.limits.max_llm_pairs} "
                    "candidates of a call are scored; the others are ordered by TrialSpace similarity.")
        return "match: the decision model's match score (0-1) and P(reasonable consideration)."

    def _decision_progress(self, progress, what: str):
        name = "LLM" if self.llm else "Decision model"

        def report(done, total, message=""):
            progress(f"{name}: {what} {done}/{total}" if total else f"{name}: {message}")
        return report

    def _scores(self, pairs: list[PairText], families, progress, what: str) -> list[dict]:
        """The pairs' scores from the session's scorer: per pair {"match": ..., "common_criteria": ...}, each
        compact for the model (the first max_score_pairs pairs, or max_llm_pairs for the LLM)."""
        if not self.llm:
            pairs = pairs[: self.limits.max_score_pairs]
            scored = self.state.decisions.score(pairs, families, self._decision_progress(progress, what))
            return [{"match": self._match(v), "common_criteria": self._common(v)} for v in scored]
        pairs = pairs[: self.limits.max_llm_pairs]
        decisions = self.state.llm_decisions
        scored = decisions.score(self.owner, pairs, families, self.llm_connection,
                                 self._decision_progress(progress, what))
        rows = []
        for pair, value in zip(pairs, scored):
            for family in families:
                answer = value.get(family)
                if isinstance(answer, dict) and answer.get("key"):
                    self.answers.append({"patient_id": pair.patient_id, "cohort_id": pair.cohort_id,
                                         "family": family, "key": answer["key"]})
            match = value.get("match") or {}
            bp = value.get("boilerplate")
            rows.append({
                "match": ({"llm_score": match["score"], "of": match.get("max", 6), "reasonable": match["reasonable"]}
                          if match.get("score") is not None else
                          {"llm_score": None, "note": "the LLM's answer could not be read"} if match else None),
                "common_criteria": (None if bp is None else "the cohort has no common criteria" if "skipped" in bp
                                    else {"clearly_excluded": bp.get("excluded")})})
        return rows

    @staticmethod
    def _match(scored: dict | None) -> dict | None:
        match = (scored or {}).get("match")
        if not match:
            return None
        return {"score": round(match["score"], 3), "reasonable": bool(match["pass"]),
                "p_reasonable": round(match["reasonableness"], 3)}

    @staticmethod
    def _common(scored: dict | None):
        if not scored or "boilerplate" not in scored:
            return None
        bp = scored.get("boilerplate") or {}
        return ({"p_pass": round(bp["p"], 3), "pass": bool(bp["pass"])} if "p" in bp
                else "the cohort has no common criteria")

    @staticmethod
    def _rank(row: dict) -> tuple:
        """Scored first, by score (the LLM's 0-6 or the decision model's 0-1), then TrialSpace similarity."""
        match = row.get("match") or {}
        score = match.get("llm_score", match.get("score"))
        return (score is None, -(score if score is not None else 0), -row.get("cosine", 0.0))

    @staticmethod
    def _match_text(index: TextIndex, text: str) -> np.ndarray:
        try:
            return index.match(text)
        except ValueError as exc:  # an invalid pattern: the message names the term (the model's own text)
            raise ToolError(f"{exc}; put a term with spaces in double quotes") from None

    def _encode(self, text: str, prompt: str) -> np.ndarray:
        if not self.state.encoder.available:
            raise ToolError("the TrialSpace query encoder is unavailable in this app")
        return np.asarray(self.state.encoder.encode([text], prompt)[0], dtype=np.float32)

    # ------------------------------------------------------------------ trials

    def search_trials(self, a: dict, progress):
        store = self.state.store
        ids, index = trial_index(store, a["include_common_criteria"])
        cohorts = [store["cohorts"][cid] for cid in ids]
        keep = self._cohort_mask(cohorts, a["cancer_type"], a["open_only"])
        if parse_terms(a["text"]):
            keep &= self._match_text(index, a["text"])
        trials = store["trials"]
        if a["drug"].strip():
            if not any(t.get("drugs") for t in trials.values()):
                raise ToolError("there is no GoodOption drug catalog in this mode (trials have no drug lists): "
                                "search the text instead")
            try:
                pattern = re.compile(a["drug"], re.IGNORECASE)
            except re.error:
                raise ToolError(f"invalid drug pattern {a['drug']!r}") from None
            hit = {tid for tid, t in trials.items() if any(
                pattern.search(str(part)) for d in t.get("drugs") or []
                for part in (d.get("name"), d.get("listed_name"), *(d.get("all_classes") or [])) if part)}
            keep &= np.array([c["trial_id"] in hit for c in cohorts], dtype=bool)
        center = a["center"].strip().lower()
        if center:
            matched = {c["id"] for c in store["centers"] if center in (c["id"], c["name"].lower(),
                                                                       c["short_name"].lower())}
            matched = matched or {c["id"] for c in store["centers"] if center in c["name"].lower()}
            if not matched:
                raise ToolError(f"no center matches {a['center']!r} (see the data overview's centers)")
            keep &= np.array([trials[c["trial_id"]].get("center_id") in matched for c in cohorts], dtype=bool)
        if a["phases"]:
            wanted = set(a["phases"])
            keep &= np.array([bool(phases_of(trials[c["trial_id"]].get("phase") or "") & wanted) for c in cohorts],
                             dtype=bool)
        by_trial: dict[str, list[dict]] = {}
        for cohort, kept in zip(cohorts, keep):
            if kept:
                by_trial.setdefault(cohort["trial_id"], []).append(cohort)
        order = sorted(by_trial, key=lambda t: (not trials[t].get("is_open"), trials[t].get("center") or "", t))
        rows, links = [], {}
        for tid in order[: self.limits.max_rows]:
            trial = trials[tid]
            found = by_trial[tid]
            rows.append({**self._trial(trial), "matching_cohorts": [{"cohort_id": c["id"], "name": short(c["name"], 120)}
                                                                    for c in found[:6]],
                         "matching_cohort_count": len(found), "dashboard_cohort_count": len(trial["cohort_ids"]),
                         **({"drugs": [d["name"] for d in trial.get("drugs") or []][:6]} if trial.get("drugs") else {})})
            links[tid] = self._trial_link(trial)
            for c in found[:6]:
                links[c["id"]] = self._cohort_link(c)
        total_cohorts = int(keep.sum())
        # Every matching trial once, by the data's own phase label (the rows below may be cut).
        by_phase: dict[str, int] = {}
        for tid in order:
            label = (trials[tid].get("phase") or "").strip() or "not given"
            by_phase[label] = by_phase.get(label, 0) + 1
        result = {"trials_total": len(order), "cohorts_total": total_cohorts, "shown": len(rows),
                  "truncated": len(order) > len(rows), "searched_cohorts": len(cohorts),
                  "trials_by_phase": dict(sorted(by_phase.items())), "trials": rows}
        what = " · ".join(x for x in (a["text"].strip(), a["drug"].strip() and f"drug {a['drug'].strip()}",
                                      a["cancer_type"] and cancer_types.TYPE_LABELS.get(a["cancer_type"]),
                                      a["center"].strip() and f"center {a['center'].strip()}",
                                      a["phases"] and "phase " + "/".join(a["phases"]),
                                      a["open_only"] and "open only") if x) or "all"
        return (f"Searched trials: {what} → {total_cohorts} cohorts in {len(order)} trials", result, links)

    def match_trials(self, a: dict, progress):
        from matchminer_ai.embedding.trialspace_contract import PATIENT_PROMPT
        from matchminer_ai.matching.retrieval import stable_top_k

        s = self.state
        if s.vectors is None:
            raise ToolError("no valid TrialSpace round")
        description = a["description"].strip()
        if not description:
            raise ToolError("give a patient description")
        progress("Embedding the description with TrialSpace")
        vector = self._encode(description, PATIENT_PROMPT)
        with s.round_lock:
            store = s.store
            scope = [c["id"] for c in store["cohorts"].values() if c["primary"]]
            ids, matrix = s.cohort_matrix(scope)
            if not ids:
                raise ToolError("the trial cohorts' TrialSpace vectors are still being made; try again shortly")
            scores = (matrix @ vector).astype(np.float64)
        cohorts = [s.cohort(cid) for cid in ids]
        allowed = self._cohort_mask(cohorts, a["cancer_type"], a["open_only"])
        scores = np.where(allowed, scores, -np.inf)
        k = self._k(a["k"])
        order, seen = [], set()
        for i in stable_top_k(scores, int(np.isfinite(scores).sum())):
            if cohorts[i]["trial_id"] in seen:
                continue
            seen.add(cohorts[i]["trial_id"])
            order.append(i)
            if len(order) == k:
                break
        rows, links = [], {}
        for rank, i in enumerate(order, 1):
            cohort, trial = cohorts[i], store["trials"][cohorts[i]["trial_id"]]
            rows.append({"rank": rank, "cohort_id": cohort["id"], "cohort_name": short(cohort["name"], 120),
                         **self._trial(trial), "cosine": round(float(scores[i]), 4)})
            links[cohort["id"]] = self._cohort_link(cohort)
            links[trial["id"]] = self._trial_link(trial)
        if a["rerank"] and rows:
            current = {r["cohort_id"]: s.cohort(r["cohort_id"]) for r in rows[: self.limits.max_score_pairs]}
            pairs = [PairText("assistant-query", cid, description, "", c["key"], c["boilerplate"], c["items"])
                     for cid, c in current.items()]
            for row, scored in zip(rows, self._scores(pairs, ("match",), progress, "match questions")):
                row["match"] = scored["match"]
            rows.sort(key=self._rank)
        result = {"searched_cohorts": int(allowed.sum()), "round": s.round.number, "shown": len(rows),
                  "results": rows, "note": "cosine: TrialSpace similarity (a ranking signal, not eligibility); "
                                           + self.score_note()}
        scorer = f", {'LLM' if self.llm else 'decision model'}" if a["rerank"] else ""
        return f"Matched trials (TrialSpace{scorer}) → {len(rows)} trials", result, links

    def trial_details(self, a: dict, progress):
        store, limits = self.state.store, self.limits
        wanted = [str(i).strip() for i in a["ids"] if str(i).strip()]
        if not wanted:
            raise ToolError("give trial or cohort IDs")
        out, missing, links = [], [], {}
        for ident in list(dict.fromkeys(wanted))[: limits.max_detail_ids]:
            if ident in store["trials"]:
                trial = store["trials"][ident]
                cohort_ids = trial["cohort_ids"][: limits.max_detail_cohorts]
            elif self.state.management.effective(ident) is not None:  # a cohort, an original or a subcohort
                cohort = self.state.cohort(ident)
                trial = store["trials"][cohort["trial_id"]]
                cohort_ids = [ident]
            else:
                missing.append(ident)
                continue
            other = sum(len(v["cohort_ids"]) for name, v in trial["versions"].items()
                        if name != trial.get("primary_version"))
            entry = {**self._trial(trial), "shown_version": trial.get("primary_version"),
                     "summary": short(trial.get("brief_summary"), 600),
                     "dashboard_cohort_count": len(trial["cohort_ids"]), "other_version_cohort_count": other}
            for key in ("site_pi", "overall_pi"):
                if trial.get(key):
                    entry[key] = trial[key]
            if trial.get("drugs"):
                entry["drugs"] = [{"name": d["name"], "role": d.get("role"),
                                   "classes": (d.get("mechanism_types") or [])[:3]} for d in trial["drugs"][:8]]
            cohorts = []
            for cid in cohort_ids:
                cohort = self.state.cohort(cid)  # its current criteria
                items = cohort.get("items") or []

                def lines(kind, boilerplate, items=items):
                    return [short(i["text"], 300) for i in items if i.get("criterion_type") == kind
                            and bool(i.get("boilerplate")) == boilerplate][:15]

                view = {"cohort_id": cid, "name": short(cohort["name"], 160), "version": cohort["version"],
                        "key_inclusion": lines("inclusion", False), "key_exclusion": lines("exclusion", False),
                        "common_line_count": sum(1 for i in items if i.get("boilerplate"))}
                if not items:
                    view["key_criteria"] = short(cohort.get("key"), 2000)
                if a["show_common_criteria"]:
                    view["common_inclusion"] = lines("inclusion", True)
                    view["common_exclusion"] = lines("exclusion", True)
                managed = cohort.get("management") or {}
                if managed.get("slots"):  # entered by staff on Clinical trial management
                    view["slots"] = managed["slots"]
                if managed.get("kind") == "subcohort":
                    view["subcohort_of"] = managed.get("parent")
                cohorts.append(view)
                links[cid] = self._cohort_link(cohort)
            entry["cohorts"] = cohorts
            out.append(entry)
            links[trial["id"]] = self._trial_link(trial)
        result = {"trials": out, **({"unknown_ids": missing} if missing else {})}
        return (f"Read trial details: {', '.join(e['trial_id'] for e in out) or 'none'}"
                + (f" ({len(missing)} unknown)" if missing else ""), result, links)

    # ------------------------------------------------------------------ patients

    def search_patients(self, a: dict, progress):
        s = self.state
        mask = self._patient_mask(a["cancer_type"], "", a["with_records"])
        try:
            found = s.corpus.search_patients(a["text"], "", a["sex"], 0, self.limits.max_rows, mask, "text")
        except ValueError as exc:  # an invalid pattern: the message names the term (the model's own text)
            raise ToolError(f"{exc}; put a term with spaces in double quotes") from None
        held = self.g.records.patient_ids()
        rows = [self._patient(r["id"], held) for r in found["rows"]]
        what = " · ".join(x for x in (a["text"].strip(), a["cancer_type"] and cancer_types.TYPE_LABELS.get(
            a["cancer_type"]), a["sex"], a["with_records"] and "with records") if x) or "all"
        result = {"total": found["total"], "of": found["of"], "shown": len(rows),
                  "truncated": found["total"] > len(rows), "patients": rows}
        return (f"Searched patients: {what} → {found['total']} patients", result,
                {r["patient_id"]: {"kind": "patient"} for r in rows})

    def match_patients(self, a: dict, progress):
        from matchminer_ai.embedding.trialspace_contract import COHORT_PROMPT

        s = self.state
        if s.vectors is None or s.patient_vectors is None:
            raise ToolError("the patients' TrialSpace vectors are unavailable in this app")
        cohort, criteria = None, a["criteria"].strip()
        if a["cohort_id"].strip():
            cohort = s.cohort(a["cohort_id"].strip())
            with s.round_lock:
                vector = s.cohort_vectors.vector(cohort["key"])
            if vector is None:
                s.embed_cohorts([cohort["key"]])
                raise ToolError("this cohort's TrialSpace vector is still being made; try again shortly")
        elif criteria:
            progress("Embedding the criteria with TrialSpace")
            vector = self._encode(criteria, COHORT_PROMPT)
        else:
            raise ToolError("give a cohort_id or criteria")
        mask = self._patient_mask(a["cancer_type"], a["sex"], a["with_records"])
        k = self._k(a["k"])
        with s.round_lock:
            hits = s.patient_vectors.search(np.asarray(vector, dtype=np.float32), "patients", k, mask)
        held = self.g.records.patient_ids()
        rows = [{"rank": rank, **self._patient(s.corpus.patients[i]["id"], held), "cosine": round(c, 4)}
                for rank, (i, c) in enumerate(hits, 1)]
        if a["rerank"] and rows:
            if cohort is not None:
                pairs = [s.pair(r["patient_id"], cohort["id"]) for r in rows]
            else:
                pairs = []
                for r in rows:
                    entity = s.patient_entity(r["patient_id"])
                    pairs.append(PairText(r["patient_id"], "assistant-criteria", entity["key"], entity["boilerplate"],
                                          criteria, "", None))
            for row, scored in zip(rows, self._scores(pairs, ("match",), progress, "match questions")):
                row["match"] = scored["match"]
            rows.sort(key=self._rank)
        links = {r["patient_id"]: {"kind": "patient"} for r in rows}
        if cohort is not None:
            links[cohort["id"]] = self._cohort_link(cohort)
        searched = len(s.corpus.patients) if mask is None else int(mask.sum())
        result = {"query": cohort["id"] if cohort else "ad-hoc criteria", "searched_patients": searched,
                  "shown": len(rows), "results": rows,
                  "note": "cosine: TrialSpace similarity (a ranking signal, not eligibility); " + self.score_note()}
        scorer = f", {'LLM' if self.llm else 'decision model'}" if a["rerank"] else ""
        return (f"Matched patients to {cohort['id'] if cohort else 'the criteria'} (TrialSpace{scorer}) → "
                f"{len(rows)} of {searched} patients", result, links)

    def patient_details(self, a: dict, progress):
        s, limits = self.state, self.limits
        pid = a["patient_id"].strip()
        if pid.startswith("entered-"):
            raise ToolError("only the mode's corpus patients can be read")
        entity = s.patient_entity(pid)
        result = {**self._patient(pid), "cancer_history_summary": entity["key"][: limits.max_text_chars],
                  "other_relevant_history": (entity["boilerplate"] or "")[: limits.max_text_chars // 2]}
        if a["include_records"]:
            stored = rec.store_records(self.g.records, pid)
            if stored is None:
                result["structured_records"] = {"available": False,
                                                "reason": "the mode's extraction store does not hold this patient"}
            else:
                documents, patients, _ = stored
                view = rec.records_view(documents, patients, {})
                course = view["disease_course"]["rows"]
                result["structured_records"] = {
                    "available": True, "counts": view["counts"],
                    "disease_course": [{k: v for k, v in (("date", r.get("date") or r.get("date_text")),
                                                          ("type", r["type"]), ("item", short(r["label"], 160)),
                                                          ("tag", r.get("tag"))) if v}
                                       for r in course[: 2 * limits.max_rows]],
                    "disease_course_rows": len(course)}
        return (f"Read patient {pid}" + (" with structured records" if a["include_records"] else ""), result,
                {pid: {"kind": "patient"}})

    # ------------------------------------------------------------------ structured records

    def screen_records(self, a: dict, progress):
        from matchminer_ai.matching.structured_criteria import compile_criteria_queries

        g, limits = self.g, self.limits
        store = g.records.get()
        if store is None:
            raise ToolError("there is no structured record store in this mode")
        writer = g.task_settings(self.owner, SQL_TASK).extraction_config()
        links = {}
        if a["cohort_id"].strip():
            cohort = self.state.cohort(a["cohort_id"].strip())
            compiled, _ = rec.compile_cohort_criteria(cohort, "queries", settings=g.settings, config=writer,
                                                      store=store, progress=progress)
            subject = cohort["id"]
            links[cohort["id"]] = self._cohort_link(cohort)
        else:
            lines = [{"number": n, "text": line["text"].strip(), "criterion_type": line["type"], "boilerplate": False}
                     for n, line in enumerate(a["lines"], 1) if line["text"].strip()]
            if not lines:
                raise ToolError("give criteria or a cohort_id")
            if len(lines) > limits.max_screen_lines:
                raise ToolError(f"screen at most {limits.max_screen_lines} criteria at a time")
            context = "\n".join(f"{'Inclusion' if line['criterion_type'] == 'inclusion' else 'Exclusion'}: {line['text']}"
                                for line in lines)
            compiled, _ = compile_criteria_queries(
                lines, cohort_context=context, config=writer, cache=self.memory_cache, return_metadata=True,
                progress_callback=lambda done, total: progress(f"Compiling criteria queries · {done}/{total} ready"))
            subject = f"{len(lines)} {'criterion' if len(lines) == 1 else 'criteria'}"
        ids = list(dict.fromkeys(str(p).strip() for p in a["patient_ids"] if str(p).strip()))[:1000] \
            or sorted(g.records.patient_ids())
        progress(f"Reading stored records of {len(ids)} patients")
        documents, patients = store.records(ids)
        held = sorted(set(documents.patient_id if documents is not None else [])
                      | set(patients.patient_id if patients is not None else []))
        frames = [f for f in (documents, patients) if f is not None and len(f)]
        report = (rec.run_criteria(frames, compiled, "queries", held, writer, progress) if held
                  else {"patients": [], "aggregate": []})
        lines_out = []
        for row, entry in zip(report["aggregate"], compiled):
            counts = {k: v for k, v in row["statuses"].items() if v}
            query = entry.get("query") or {}
            checkable = entry.get("status") == "compiled" and query.get("expressible", True) and \
                row["statuses"].get("not_checkable", 0) < max(1, row["patient_count"])
            lines_out.append({"number": row["number"], "text": short(row["text"], 300), "type": row["criterion_type"],
                              "checkable": bool(checkable),
                              "query": short(query.get("summary") or query.get("not_expressible_reason")
                                             or entry.get("error"), 300),
                              "counts": counts})
        inclusion = [i for i, line in enumerate(lines_out) if line["type"] == "inclusion" and line["checkable"]]
        meeting, not_failing = [], 0
        for p in report["patients"]:
            statuses = [line["status"] for line in p["lines"]]
            fails = any(line["clearly_fails"] for line in p["lines"])
            not_failing += not fails
            if inclusion and not fails and all(statuses[i] == "meets" for i in inclusion):
                meeting.append(p["patient_id"])
        rows = [self._patient(pid) for pid in meeting[: limits.max_rows]]
        links.update({r["patient_id"]: {"kind": "patient"} for r in rows})
        result = {
            "screened": subject, "patients_asked": len(ids), "patients_with_records": len(held),
            "lines": lines_out,
            "documented_to_meet_every_checkable_inclusion_line": {"total": len(meeting), "patients": rows,
                                                                 "truncated": len(meeting) > len(rows)},
            "no_line_clearly_failed": not_failing,
            "polarity": "meets: documented facts satisfy the line (for an exclusion line: the condition documented "
                        "absent); clearly_fails: documented facts contradict eligibility; not_documented: no extracted "
                        "item shows either, which is not failing (absence never fails a line); not_checkable: no "
                        "extracted item type can check the line. Extracted by an LLM: review the records.",
        }
        unchecked = [line["number"] for line in lines_out if not line["checkable"]]
        return (f"Screened structured records: {subject} on {len(held)} patients → {len(meeting)} documented to meet"
                + (f"; criteria not checkable: {', '.join(map(str, unchecked))}" if unchecked else ""), result, links)

    # ------------------------------------------------------------------ the decision model and guidelines

    def score_pairs(self, a: dict, progress):
        s = self.state
        wanted = [(p["patient_id"].strip(), p["cohort_id"].strip()) for p in a["pairs"]]
        if not wanted:
            raise ToolError("give patient-cohort pairs")
        most = self.limits.max_llm_pairs if self.llm else self.limits.max_score_pairs
        if len(wanted) > most:
            raise ToolError(f"score at most {most} pairs at a time")
        pairs = [s.pair(pid, cid) for pid, cid in wanted]
        scored = self._scores(pairs, ("match", "boilerplate"), progress, "pairs")
        rows, links = [], {}
        for (pid, cid), value in zip(wanted, scored):
            rows.append({"patient_id": pid, "cohort_id": cid, **value})
            links[pid] = {"kind": "patient"}
            links[cid] = self._cohort_link(s.cohort(cid))
        common = ("common_criteria: whether the LLM finds a common exclusion clearly applies (clearly_excluded)."
                  if self.llm else "common_criteria: P(no common exclusion clearly applies).")
        result = {"pairs": rows, "note": self.score_note() + " " + common}
        return f"Scored {len(rows)} pairs with {self.scorer_label()}", result, links

    def guideline_spaces(self, a: dict, progress):
        description = a["description"].strip()
        if not description:
            raise ToolError("give a patient description")
        top_n = max(1, min(int(a["top_n"] or 3), 5))
        paths = self.g.catalogs.resolve([])
        progress("Finding guideline spaces")
        found = self.g.inference.retrieve(description, paths, top_n, max(10, 3 * top_n),
                                          lambda *args: progress("Finding guideline spaces"))
        spaces = []
        for space in found.get("spaces") or []:
            entry = {k: short(space.get(k), 300) for k in ("catalog_title", "disease", "space", "space_name",
                                                           "clinical_space_summary") if space.get(k)}
            source = space.get("source")
            if isinstance(source, dict) and source.get("label"):  # e.g. "Derived from PDQ®: ..."
                entry["source"] = short(source["label"], 300)
            for key in ("match_quality_score", "similarity"):
                if isinstance(space.get(key), (int, float)):
                    entry[key] = round(float(space[key]), 3)
            for menu in ("treatment_options", "diagnostic_workup"):
                entry[menu] = [short(o.get("name"), 160) + (f" ({o['category']})" if o.get("category") else "")
                               for o in (space.get(menu) or [])[:10]]
            spaces.append(entry)
        return (f"Found guideline spaces → {len(spaces)}",
                {"spaces": spaces, "note": "Guideline spaces from the finalized catalogs (LLM-assisted from NCCN "
                                           "and PDQ sources); verify against the source guideline."}, {})


LABELS = {
    "search_trials": "Searched trials", "match_trials": "Matched trials", "trial_details": "Read trial details",
    "search_patients": "Searched patients", "match_patients": "Matched patients",
    "patient_details": "Read patient", "screen_records": "Screened structured records",
    "score_pairs": "Scored pairs", "guideline_spaces": "Found guideline spaces",
}
