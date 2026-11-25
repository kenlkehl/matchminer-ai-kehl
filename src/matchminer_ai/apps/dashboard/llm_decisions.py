"""LLM decision checks: the LLM alternative to the decision model wherever the dashboard scores a pair.

Each of the decision model's three question families has an LLM function in the inference package, which asks the
training teacher's prompt (no dashboard prompt):

- **match** (``score_match_quality_with_llm``, ``match_quality.labels.system.txt``): the five match labels of the
  patient's key summary and the cohort's key criteria in one request, each yes or no with an evidence status and a
  reason, combined into the package's 0-6 match score (0 unless reasonable; then 1, plus 1 each for cancer type,
  disease burden and treatment history and 2 for biomarker);
- **boilerplate** (``exclusion_criteria_check_with_llm``, ``llm_exclusion_criteria.user.txt``): is the patient
  clearly excluded by one of the cohort's boilerplate criteria? The answer's last word is Yes! or No!;
- **criteria** (``check_eligibility_criteria_with_llm``): one request per tagged criteria line; yes means the patient
  does not clearly fail the line.

None of these is a probability, and the dashboard never mixes them with the decision model's: a view shows one
scorer's answers at a time, on its own scale, and ranks by them (lists: the 0-6 score, then TrialSpace similarity).

Every answer keeps the model's raw output, the parsed answer and the reasoning the backend returned: vLLM's
``reasoning_content``, Claude's thinking summaries (asked for here; the gateway omits them otherwise), nothing from
hosted Gemini (thought summaries are not requested). ``reasoning_note`` says which.

**Data handling.** Answers and reasoning stay in this process's memory, per browser session (``SessionResults``),
keyed by the pair's texts, the question family, the connection and the prompt contract, and expire with the
session. Runs are background jobs of the session (``core.jobs``: only it sees them, and errors are reduced to their
type), with at most ``IN_FLIGHT`` requests in flight per question family. Nothing patient-derived is logged. The patient summary
and the cohort's criteria go to the session's connection for the ``llm_decisions`` task (Data and Models).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .core.cache import text_key
from .core.decisions import PairText
from .core.errors import InputError
from .core.jobs import JobManager
from .llm import NOTICE, Connection

CACHE_VERSION = "mmai-dash-llm-decisions-v1"
FAMILIES = ("match", "boilerplate", "criteria")
SECTIONS = ("llm_match_quality", "llm_exclusion_criteria", "llm_criteria_check")
#: Requests a run keeps in flight per question family (a shared endpoint; the gateway sheds load above a few
#: hundred): a page's first ten pairs are one round.
IN_FLIGHT = 10
#: A run of more requests than this is shown as an estimate and needs the user's confirmation first.
LARGE_RUN = 25
#: Pairs one list run may score; candidates a guideline-space or population search may rank with the LLM.
MAX_PAIRS = 50
MAX_CANDIDATES = 200
#: Answers a session keeps (the oldest are dropped first).
MAX_SESSION_ANSWERS = 5000
SKIPPED = {"skipped": "the cohort has no boilerplate criteria"}
#: The answer scales, as the views label them.
SCALES = {
    "match": {"name": "LLM match score", "range": [0, 6],
              "meaning": "The package's 0-6 match score from the LLM's five yes/no answers: 0 unless the patient is a "
                         "reasonable candidate; then 1, plus 1 each for cancer type, disease extent and treatment "
                         "history and 2 for biomarker (requirements the cohort specifically has and the patient "
                         "documentedly meets).",
              "ranking": "Ranked by the LLM match score (0-6), then TrialSpace similarity; pairs the LLM has not "
                         "scored follow by similarity."},
    "boilerplate": {"name": "LLM common-exclusion check", "range": ["No", "Yes"],
                    "meaning": "The LLM's answer to \"Is this patient clearly excluded from this cohort?\" from the "
                               "cohort's common (boilerplate) criteria and the patient's other relevant history."},
    "criteria": {"name": "LLM per-criterion check", "range": ["yes", "no"],
                 "meaning": "Per line: yes, the patient does not clearly fail it; no, the patient clearly fails it "
                            "(an inclusion line clearly not met, or an exclusion line clearly met)."},
}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _bool(value):
    """True, False or None (pandas' NA and NaN become None)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return bool(value)
    except (TypeError, ValueError):
        return None


def reasoning_note(connection: Connection) -> str:
    """What reasoning text this connection's answers carry."""
    if connection.claude:
        return ("Claude's thinking summaries (requested through the gateway; Claude does not return its raw "
                "thinking). An answer for which Claude did not think has none.")
    if connection.gemini:
        return "Hosted Gemini returns no reasoning text through this connection (thought summaries are not requested)."
    if connection.vertex:
        return "The model's reasoning, when its thinking switch is on (Data and Models) and it returns any."
    return ("The server's reasoning text (vLLM's reasoning_content, or the reasoning the package's parser splits off) "
            "when the model thinks; otherwise none.")


@dataclass
class _Entry:
    updated: float
    values: dict


class SessionResults:
    """Answers per browser session, in memory only; a session's answers expire ``ttl`` seconds after its last use."""

    def __init__(self, ttl: float, limit: int = MAX_SESSION_ANSWERS):
        self.ttl, self.limit = ttl, limit
        self.sessions: dict[str, _Entry] = {}
        self.lock = threading.Lock()

    def _clean(self) -> None:
        now = time.monotonic()
        for owner in [o for o, e in self.sessions.items() if now - e.updated > self.ttl]:
            del self.sessions[owner]

    def get_many(self, owner: str, keys) -> dict:
        with self.lock:
            self._clean()
            entry = self.sessions.get(owner)
            if entry is None:
                return {}
            entry.updated = time.monotonic()
            return {k: entry.values[k] for k in keys if k in entry.values}

    def put_many(self, owner: str, values: dict) -> None:
        with self.lock:
            self._clean()
            entry = self.sessions.setdefault(owner, _Entry(time.monotonic(), {}))
            entry.updated = time.monotonic()
            for key, value in values.items():
                entry.values.pop(key, None)
                entry.values[key] = value
            while len(entry.values) > self.limit:
                entry.values.pop(next(iter(entry.values)))

    def count(self, owner: str) -> int:
        with self.lock:
            entry = self.sessions.get(owner)
            return len(entry.values) if entry else 0

    def clear(self, owner: str) -> None:
        with self.lock:
            self.sessions.pop(owner, None)


class LLMDecisions:
    def __init__(self, ttl: float = 3600, *, jobs: JobManager | None = None, metadata_dir: Path | None = None):
        self.results = SessionResults(ttl)
        self.jobs = jobs or JobManager(workers=4, ttl=ttl, name="mmai-llm-decisions")
        self.metadata_dir = metadata_dir  # the package's model metadata (server listings), never patient text
        self._base = None
        self._contracts: dict[str, str] = {}

    # ------------------------------------------------------------------ configuration

    def base_config(self):
        from matchminer_ai import load_default_preset

        if self._base is None:
            self._base = load_default_preset()
            if self.metadata_dir is not None:
                self._base.model_metadata_cache_dir = str(self.metadata_dir)
        return self._base

    def config(self, connection: Connection):
        return connection.configure(self.base_config(), SECTIONS, thinking_summaries=True)

    @staticmethod
    def package_version() -> str | None:
        import matchminer_ai

        return getattr(matchminer_ai, "__version__", None)

    def contract(self, family: str) -> str:
        """The prompt (and its answer contract) of a family, with the package version: a new prompt is a new answer."""
        if family not in self._contracts:
            from matchminer_ai.llm.prompts import load_prompt_text

            if family == "match":
                from matchminer_ai.matching.match_labels import match_labels_contract

                prompt = {"labels": match_labels_contract(), "file": self.base_config().llm_match_quality.get("prompt_file")}
            elif family == "boilerplate":
                name = self.base_config().llm_exclusion_criteria["prompt_file"]
                prompt = {"file": name, "sha256": hashlib.sha256(load_prompt_text(name).encode()).hexdigest()}
            else:
                from matchminer_ai.labeling.tasks import contract_hash

                prompt = {"criteria": contract_hash("criteria")}
            self._contracts[family] = text_key(prompt, self.package_version())
        return self._contracts[family]

    def unavailable(self, connection: Connection) -> str | None:
        if not connection.configured:
            return ("No LLM connection is ready for the LLM decision checks: start the local model server or choose a "
                    "connection on Data and models.")
        if connection.vertex:
            from matchminer_ai.llm.connections import vertex_credentials_error

            return vertex_credentials_error()
        return None

    def describe(self, connection: Connection, owner: str | None = None) -> dict:
        reason = self.unavailable(connection)
        return {"available": reason is None, "reason": reason, "connection": connection.describe(), "notice": NOTICE,
                "reasoning": reasoning_note(connection), "scales": SCALES, "large_run": LARGE_RUN,
                "max_pairs": MAX_PAIRS, "max_candidates": MAX_CANDIDATES, "in_flight": IN_FLIGHT,
                "cached": self.results.count(owner) if owner else 0}

    def require(self, connection: Connection) -> None:
        reason = self.unavailable(connection)
        if reason:
            raise InputError(reason)

    # ------------------------------------------------------------------ keys

    def key(self, family: str, pair: PairText, connection: Connection) -> str:
        if family == "match":
            texts = (pair.patient_key.strip(), pair.cohort_key.strip())
        elif family == "boilerplate":
            texts = (pair.patient_boilerplate.strip(), pair.cohort_boilerplate.strip())
        else:
            texts = (pair.patient_key.strip(), pair.patient_boilerplate.strip(), pair.items)
        return text_key(CACHE_VERSION, family, connection.identity(), self.contract(family), *texts)

    @staticmethod
    def asks(family: str, pair: PairText) -> bool:
        """Whether the family has a question for the pair (a cohort without boilerplate criteria has none)."""
        return family != "boilerplate" or bool(pair.cohort_boilerplate.strip())

    def answer(self, owner: str, key: str) -> dict | None:
        """One cached answer of the session (its full entry: parsed answer, raw output, reasoning)."""
        return self.results.get_many(owner, [key]).get(key)

    # ------------------------------------------------------------------ lookups and estimates

    def lookup(self, owner: str, pairs: list[PairText], families, connection: Connection) -> list[dict]:
        """The session's cached answers for these pairs (no model calls)."""
        keys = [{f: self.key(f, p, connection) for f in families if f != "criteria"} for p in pairs]
        found = self.results.get_many(owner, [k for row in keys for k in row.values()])
        rows = []
        for pair, row in zip(pairs, keys):
            out = {"patient_id": pair.patient_id, "cohort_id": pair.cohort_id}
            for family, key in row.items():
                out[family] = found.get(key) if self.asks(family, pair) else SKIPPED
            rows.append(out)
        return rows

    def estimate(self, owner: str, pairs: list[PairText], families, connection: Connection, *,
                 fresh: bool = False) -> dict:
        """What scoring the pairs would send: requests per family (cached answers, unless ``fresh``, and skipped
        questions are not asked again), the model and approximate input tokens (characters / 3)."""
        from matchminer_ai.llm.prompts import load_prompt_text

        base = self.base_config()
        prompts = {"match": len(load_prompt_text(base.llm_match_quality["prompt_file"])),
                   "boilerplate": len(load_prompt_text(base.llm_exclusion_criteria["prompt_file"])),
                   "criteria": 2000}
        requests, characters, cached = {}, 0, 0
        for family in families:
            todo = [p for p in pairs if self.asks(family, p)]
            keys = [self.key(family, p, connection) for p in todo]
            found = {} if fresh else self.results.get_many(owner, keys)
            todo = [p for p, k in zip(todo, keys) if k not in found]
            cached += len(keys) - len(todo)
            if family == "criteria":
                lines = [sum(1 for i in p.items or [] if i.get("criterion_type") in ("inclusion", "exclusion"))
                         for p in todo]
                requests[family] = sum(lines)
                characters += sum(n * (prompts[family] + len(p.patient_key) + len(p.patient_boilerplate)
                                       + len(json.dumps(p.items or [])))
                                  for n, p in zip(lines, todo))
            else:
                requests[family] = len(todo)
                characters += sum(prompts[family] + (len(p.patient_key) + len(p.cohort_key) if family == "match"
                                                     else len(p.patient_boilerplate) + len(p.cohort_boilerplate))
                                  for p in todo)
        total = sum(requests.values())
        return {"pairs": len(pairs), "requests": total, "by_family": requests, "cached": cached,
                "approx_input_tokens": characters // 3, "model": connection.label, "backend": connection.backend,
                "large": total > LARGE_RUN, "large_run": LARGE_RUN}

    def check_size(self, estimate: dict, confirm: bool) -> None:
        if estimate["large"] and not confirm:
            raise InputError(f"This run sends {estimate['requests']} requests to {estimate['model']} (more than "
                             f"{LARGE_RUN}); confirm it first.")

    # ------------------------------------------------------------------ runs

    def start(self, owner: str, label: str, run) -> dict:
        """A background job of the session; ``run(progress)`` returns its result."""
        job = self.jobs.submit("llm_decisions", label, run, owner=owner, device="endpoint")
        return {"job": job.describe(with_result=False)}

    def score(self, owner: str, pairs: list[PairText], families, connection: Connection, progress=None,
              rerun: bool = False) -> list[dict]:
        """Ask the LLM what the session has no answer to (everything, with ``rerun``); the pairs' answers."""
        progress = progress or (lambda *args: None)
        config = self.config(connection)
        todo: dict[str, list[tuple[str, PairText]]] = {}
        for family in families:
            if family == "criteria":
                raise ValueError("per-criterion checks run one pair at a time (criteria)")
            wanted = {self.key(family, p, connection): p for p in pairs if self.asks(family, p)}
            found = {} if rerun else self.results.get_many(owner, list(wanted))
            todo[family] = [(k, p) for k, p in wanted.items() if k not in found]
        total, done, lock = sum(len(v) for v in todo.values()), [0], threading.Lock()
        progress(0, total, f"asking {connection.label}" if total else "cached")

        def ask_family(family, items):
            ask = self._ask_match if family == "match" else self._ask_boilerplate
            for start in range(0, len(items), IN_FLIGHT):
                chunk = items[start:start + IN_FLIGHT]
                self.results.put_many(owner, ask(chunk, connection, config))  # kept as each chunk answers
                with lock:
                    done[0] += len(chunk)
                    now = done[0]
                progress(now, total, f"{now} of {total} {'request' if total == 1 else 'requests'} answered")

        # The families run side by side (each in the job's context, so a stop reaches the package's requests).
        streams = [(family, items) for family, items in todo.items() if items]
        if len(streams) == 1:
            ask_family(*streams[0])
        elif streams:
            with ThreadPoolExecutor(len(streams), thread_name_prefix="mmai-llm-decisions") as pool:
                for future in [pool.submit(contextvars.copy_context().run, ask_family, *stream) for stream in streams]:
                    future.result()
        return self.lookup(owner, pairs, families, connection)

    def _common(self, key: str, family: str, connection: Connection, seconds: float) -> dict:
        return {"scorer": "llm", "family": family, "key": key, "scale": SCALES[family]["name"],
                "connection": {k: connection.describe()[k] for k in ("backend", "model", "label")},
                "reasoning_note": reasoning_note(connection), "contract": self.contract(family)[:12],
                "seconds": round(seconds, 1), "checked_at": time.time()}

    def _ask_match(self, chunk, connection, config) -> dict:
        import pandas as pd
        from matchminer_ai.matching import score_match_quality_with_llm
        from matchminer_ai.matching.match_labels import LABELS, MAX_SCORE, parse_match_labels

        started = time.monotonic()
        frame = pd.DataFrame([{"patient_id": p.patient_id, "cohort_trial_id": p.cohort_id,
                               "cancer_history_summary": p.patient_key, "cohort_key_criteria": p.cohort_key}
                              for _, p in chunk])
        out = score_match_quality_with_llm(frame, config=config)
        seconds = (time.monotonic() - started) / max(1, len(chunk))
        values = {}
        for (key, _), row in zip(chunk, out.to_dict("records")):
            output = str(row.get("llm_match_quality_answer_text") or "")
            try:
                answers = parse_match_labels(output)
            except ValueError:
                answers = None
            score = _number(row.get("llm_match_quality_score"))
            values[key] = {**self._common(key, "match", connection, seconds),
                           "score": int(score) if answers and score is not None and score >= 0 else None,
                           "max": MAX_SCORE,
                           "reasonable": (answers["reasonableness"]["answer"] == "yes") if answers else None,
                           "answers": answers, "labels": list(LABELS),
                           "parse_status": str(row.get("llm_match_quality_parse_status") or ""),
                           "output": output, "reasoning": str(row.get("llm_match_quality_reasoning_text") or "")}
        return values

    def _ask_boilerplate(self, chunk, connection, config) -> dict:
        import pandas as pd
        from matchminer_ai.matching import exclusion_criteria_check_with_llm

        started = time.monotonic()
        frame = pd.DataFrame([{"patient_id": p.patient_id, "cohort_trial_id": p.cohort_id,
                               "general_exclusion_criteria_evidence": p.patient_boilerplate,
                               "cohort_boilerplate_criteria": p.cohort_boilerplate} for _, p in chunk])
        out = exclusion_criteria_check_with_llm(frame, config=config)
        seconds = (time.monotonic() - started) / max(1, len(chunk))
        values = {}
        for (key, _), row in zip(chunk, out.to_dict("records")):
            passed = _bool(row.get("llm_exclusion_criteria_pass"))
            values[key] = {**self._common(key, "boilerplate", connection, seconds), "pass": passed,
                           "excluded": None if passed is None else not passed,
                           "answer": None if passed is None else ("No" if passed else "Yes"),
                           "parse_status": str(row.get("llm_exclusion_criteria_parse_status") or ""),
                           "output": str(row.get("llm_exclusion_criteria_answer_text") or ""),
                           "reasoning": str(row.get("llm_exclusion_criteria_reasoning_text") or "")}
        return values

    # ------------------------------------------------------------------ per-criterion checks

    def criteria_cached(self, owner: str, pair: PairText, connection: Connection) -> dict | None:
        found = self.answer(owner, self.key("criteria", pair, connection))
        return {**found, "cached": True} if found else None

    def criteria(self, owner: str, pair: PairText, connection: Connection, progress=None) -> dict:
        """Every tagged line of the cohort, asked about the patient's complete summary, one request per line."""
        import pandas as pd
        from matchminer_ai.matching import check_eligibility_criteria_with_llm

        if not pair.items:
            raise InputError("this cohort has no tagged criteria (per-criterion checks need them)")
        progress = progress or (lambda *args: None)
        key = self.key("criteria", pair, connection)
        config = self.config(connection)
        progress(0, len(pair.items), f"asking {connection.label}")
        started = time.monotonic()
        frame = pd.DataFrame([{"patient_id": pair.patient_id, "cohort_trial_id": pair.cohort_id,
                               "cancer_history_summary": pair.patient_key,
                               "general_exclusion_criteria_evidence": pair.patient_boilerplate,
                               "cohort_criteria_items": json.dumps(pair.items, ensure_ascii=False)}])
        out, metadata = check_eligibility_criteria_with_llm(
            frame, config=config, return_metadata=True,
            progress=lambda done, total: progress(
                done, total, f"{done} of {total} {'criterion' if total == 1 else 'criteria'} answered"))
        details = metadata["model_metadata"]["check_eligibility_criteria_with_llm"]
        lines = [{"number": int(r["criterion_number"]), "type": r["criterion_type"], "boilerplate": bool(r["boilerplate"]),
                  "text": r["criterion_text"], "answer": r["llm_criterion_answer"],
                  "fails": _bool(r["llm_criterion_clearly_fails"]), "status": r["llm_criterion_status"],
                  "reason": r["llm_criterion_reason"], "finish_reason": r["llm_criterion_finish_reason"],
                  "attempts": int(r["llm_criterion_attempts"]), "parse_status": r["llm_criterion_parse_status"],
                  "output": str(r.get("llm_criterion_answer_text") or ""),
                  "reasoning": str(r.get("llm_criterion_reasoning_text") or "")} for r in out.to_dict("records")]
        counts = {"lines": len(lines), "clearly_fail": sum(1 for line in lines if line["fails"] is True),
                  "unanswered": sum(1 for line in lines if line["answer"] is None),
                  **{k: details.get(k) for k in ("requests", "retries", "parse_failures", "unchecked_lines")}}
        value = {**self._common(key, "criteria", connection, time.monotonic() - started), "kind": "criteria_llm",
                 "patient_id": pair.patient_id, "cohort_id": pair.cohort_id, "lines": lines, "counts": counts,
                 "structured_outputs": details.get("structured_outputs"),
                 "criteria_contract": metadata.get("criteria_contract")}
        self.results.put_many(owner, {key: value})
        return value

    # ------------------------------------------------------------------ guideline spaces and populations

    def rank_guideline_spaces(self, owner: str, result: dict, top_n: int, connection: Connection,
                              progress=None) -> dict:
        """A guideline search's candidate spaces (every TrialSpace candidate, ``result["spaces"]``) ranked by the
        LLM's match score instead of the decision model's: the top ``top_n``, each with its ``llm_match`` answer.
        The decision model's scores are removed from the spaces (the package report keeps its own table)."""
        spaces = list(result["spaces"])
        summary = result["summary"]
        pairs = [PairText("workspace-patient", str(s["space_trial_id"]), summary, "",
                          str(s.get("cohort_key_criteria") or s.get("clinical_space_summary") or ""), "")
                 for s in spaces]
        answers = self.score(owner, pairs, ("match",), connection, _messages(progress))
        for space, answer in zip(spaces, answers):
            space["llm_match"] = answer["match"]
            space["decision_model_rank"] = space.get("rank")
            for name in ("match_quality_score", "match_quality_pass"):
                space.pop(name, None)
        ranked = sorted(spaces, key=lambda s: rank_key((s["llm_match"] or {}).get("score"), s.get("similarity_score"),
                                                      s["space_trial_id"]))[:top_n]
        for rank, space in enumerate(ranked, 1):
            space["rank"] = rank
        ranking = {"scorer": "llm", "model": connection.label, "candidates": len(spaces), "top_n": top_n,
                   "scale": SCALES["match"]["name"], "ranking": SCALES["match"]["ranking"]}
        note = (f"## Ranking in the dashboard: {SCALES['match']['name']}\n\n"
                f"The dashboard ranked the {len(spaces)} TrialSpace candidates by {connection.label}'s 0-6 match "
                "score, then TrialSpace similarity, and shows the top "
                f"{len(ranked)}:\n\n| Rank | Space | LLM match score | Reasonable |\n| --- | --- | --- | --- |\n"
                + "".join(f"| {s['rank']} | {_cell(s.get('name'))} | {_score(s['llm_match'])} | "
                          f"{_yes((s['llm_match'] or {}).get('reasonable'))} |\n" for s in ranked)
                + "\nThe report below is the inference package's: its match-quality columns are the decision "
                  "model's, which this ranking does not use.\n\n---\n\n")
        return {**result, "spaces": ranked, "ranking": ranking, "report": note + str(result.get("report") or "")}

    def select_guideline_patients(self, owner: str, selected, metadata: dict, connection: Connection,
                                  progress=None):
        """A population search's candidates (every scored candidate, the package's ``selected`` with no threshold)
        selected and ranked by the LLM instead: kept when its reasonableness answer is yes, ranked by its 0-6 match
        score, then TrialSpace similarity. Returns (selected, metadata) without the decision model's scores."""
        import pandas as pd

        space = metadata["space"]
        key = str(space.get("cohort_key_criteria") or space.get("clinical_space_summary") or "")
        frame = selected.reset_index(drop=True)
        pairs = [PairText(str(pid), str(space["space_trial_id"]), str(summary), "", key, "")
                 for pid, summary in zip(frame["patient_id"], frame["cancer_history_summary"])]
        answers = self.score(owner, pairs, ("match",), connection, _messages(progress))
        frame = frame.drop(columns=[c for c in ("match_quality_score", "match_quality_pass",
                                                "match_quality_reasonableness", "rank") if c in frame])
        scores = [(a["match"] or {}).get("score") for a in answers]
        reasonable = [(a["match"] or {}).get("reasonable") for a in answers]
        frame["llm_match_score"] = pd.Series(scores, dtype=object)
        frame["llm_reasonable"] = pd.Series(reasonable, dtype=object)
        frame["llm_answer_key"] = [(a["match"] or {}).get("key") for a in answers]
        order = sorted(range(len(frame)), key=lambda i: rank_key(scores[i], frame["similarity_score"][i],
                                                                frame["patient_id"][i]))
        ranked = frame.iloc[order].reset_index(drop=True)
        ranked["rank"] = range(1, len(ranked) + 1)
        chosen = ranked.loc[[v is True for v in ranked["llm_reasonable"]]].reset_index(drop=True)
        candidates = [{k: v for k, v in row.items() if k != "cancer_history_summary"}
                      for row in ranked.to_dict("records")]
        metadata = {**metadata, "candidates": candidates, "selected_count": len(chosen), "min_reasonableness": None,
                    "scorer": {"kind": "llm", "model": connection.label, "selection": "the LLM's reasonableness answer "
                               "is yes", "ranking": SCALES["match"]["ranking"]}}
        return chosen, metadata


def search_connection(guidelines, owner: str, candidates: int, confirm: bool) -> Connection:
    """The session's connection for ranking a guideline search's candidates with the LLM (``guidelines``: the
    app's ``GuidelineState``), after the size rules: at most ``MAX_CANDIDATES``, and more than ``LARGE_RUN`` only
    when the user confirmed the estimate."""
    from .guidelines.settings import LLM_DECISIONS_TASK

    decisions = guidelines.llm_decisions
    if decisions is None:
        raise InputError("The LLM decision checks are unavailable in this app.")
    connection = Connection.for_task(guidelines.task_settings(owner, LLM_DECISIONS_TASK))
    decisions.require(connection)
    if candidates > MAX_CANDIDATES:
        raise InputError(f"The LLM ranks at most {MAX_CANDIDATES} candidates; lower the candidate count.")
    if candidates > LARGE_RUN and not confirm:
        raise InputError(f"Ranking {candidates} candidates sends up to {candidates} requests to {connection.label} "
                         f"(more than {LARGE_RUN}); confirm it first.")
    return connection


def rank_key(score, similarity, ident) -> tuple:
    """LLM ranking: scored pairs first by the 0-6 score, then TrialSpace similarity, then the id; unscored pairs
    (no valid answer) after them by similarity."""
    similarity = _number(similarity) or 0.0
    return (0, -score, -similarity, str(ident)) if score is not None else (1, 0, -similarity, str(ident))


def _messages(progress):
    """A guideline job's progress takes messages only."""
    if progress is None:
        return None
    return lambda done=None, total=None, message=None: progress(message or f"{done} of {total}")


def _cell(value) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _score(answer) -> str:
    score = (answer or {}).get("score")
    return "no valid answer" if score is None else f"{score} / {(answer or {}).get('max', 6)}"


def _yes(value) -> str:
    return "?" if value is None else ("yes" if value else "no")
