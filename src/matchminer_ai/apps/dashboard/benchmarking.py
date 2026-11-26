"""The Benchmarking tab: finished and running evaluation runs, read from disk, and adjudication of judge runs.

- **Distillation fidelity** (``matchminer_ai.evaluation.retro_fidelity``): a run directory with ``run_status.json``.
  The tab shows its stage states, the stage receipts' counts and ``metrics/metrics.json`` (retrieval P@20 and mean
  teacher score with intervals; the decision model's discrimination and calibration per question family, and by
  ranker denominator when the metrics have ``decision.by_ranker``). **Export
  results** (``export``, a background job) draws the page's results as a PDF (``fidelity_report``) into the run's
  ``reports/`` (owner-only); in real mode it adds Table 1 and the metrics by demographic group
  (``matchminer_ai.evaluation.demographics``, from the EHR extract at the site config's ``paths.retro_source_root``,
  aggregates with small cells suppressed). Synthetic mode never reads the extract.
- **LLM as judge** (``matchminer_ai.evaluation.dual_judge``): a run directory with ``build.json`` and ``items/``.
  The tab shows what was built, each judge's answers and cost, ``compare/summary.json`` and ``metrics/metrics.json``,
  reruns compare and metrics (standard-library steps, in a subprocess that keeps files owner-only), and adjudicates
  the queued units (``dual_judge.adjudication.Store``).

Runs are found under the mode's roots (site config ``dashboard.<mode>.benchmark_roots``), up to ``DEPTH`` folders
down. Real mode's runs hold PHI (patients, notes, judge answers quoting them); synthetic mode never reads a ``phi``
folder. Everything returned here is counts, ids and metrics, except an adjudication item, which shows the item the
judges read. Fidelity and judge runs themselves are started from the command line.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from matchminer_ai.site import load_site_config

from .core.cache import text_key
from .core.errors import InputError
from .core.jobs import JobManager
from .settings import mode_side_error

DEPTH = 3
DISCOVERY_SECONDS = 60
# retro_fidelity's stages (run.py), in pipeline order; each writes stage_receipt.json last.
FIDELITY_STAGES = ("extraction", "patients", "cohort_table", "corpus", "retrieval", "lists_ab", "lists_c",
                   "decide/match", "decide/boilerplate", "decide/criteria", "labels/match_ab", "labels/match_c",
                   "labels/boilerplate_ab", "labels/boilerplate_c", "labels/criteria_ab", "labels/criteria_c",
                   "metrics")
RECOMPUTE_STEPS = ("compare", "metrics")
REPORT = "fidelity_report.pdf"  # in the run's reports/, beside evaluation.demographics' cache
SYNTHETIC_NOTE = ("Synthetic patients carry no age, race, ethnicity or sex, so this report has no Table 1 and no "
                  "metrics by demographic group.")


def benchmark_roots(mode: str) -> list[Path]:
    values = load_site_config().get(f"dashboard.{mode}.benchmark_roots") or []
    return [Path(v).expanduser() for v in (values if isinstance(values, list) else [values])]


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def kind_of(directory: Path) -> str | None:
    if (directory / "run_status.json").is_file():
        return "fidelity"
    if (directory / "build.json").is_file() and (directory / "items").is_dir():
        return "judge"
    return None


def updated_at(directory: Path, names: tuple[str, ...]) -> float | None:
    stamps = [p.stat().st_mtime for p in (directory / n for n in names) if p.exists()]
    return max(stamps) if stamps else None


class Benchmarks:
    def __init__(self, mode: str, roots: list[Path] | None = None, user: str | None = None,
                 extract_root: Path | None = None):
        self.mode = mode
        self.extract_root = extract_root  # real mode's EHR extract; default: the site config's paths.retro_source_root
        self.roots = [Path(r) for r in (roots if roots is not None else benchmark_roots(mode))]
        for root in self.roots:
            error = mode_side_error(mode, root)
            if error:
                raise ValueError(f"{error} (benchmark root {root})")
        self.user = user if user is not None else os.environ.get("USER", "")
        self.jobs = JobManager(name="mmai-benchmarks")
        self.lock = threading.Lock()
        self._found: tuple[float, dict[str, Path]] | None = None
        self._stores: dict[str, tuple[float | None, object]] = {}

    # ------------------------------------------------------------------ finding runs

    def _discover(self) -> dict[str, Path]:
        found: dict[str, Path] = {}

        def visit(directory: Path, depth: int):
            if kind_of(directory):
                found[text_key("benchmark-run", str(directory.resolve()))[:16]] = directory
                return
            if depth >= DEPTH:
                return
            try:
                children = sorted(e.path for e in os.scandir(directory) if e.is_dir(follow_symlinks=False)
                                  and not e.name.startswith("."))
            except OSError:
                return
            for child in children:
                visit(Path(child), depth + 1)

        for root in self.roots:
            if root.is_dir():
                visit(root, 0)
        return found

    def found(self) -> dict[str, Path]:
        with self.lock:
            if self._found is None or time.monotonic() - self._found[0] > DISCOVERY_SECONDS:
                self._found = (time.monotonic(), self._discover())
            return self._found[1]

    def available(self) -> bool:
        return bool(self.roots) and bool(self.found())

    def run(self, run_id: str, kind: str | None = None) -> Path:
        directory = self.found().get(run_id)
        if directory is None or (kind and kind_of(directory) != kind):
            raise InputError("That benchmark run is not available.")
        return directory

    def runs(self) -> list[dict]:
        rows = []
        for run_id, directory in self.found().items():
            kind = kind_of(directory)
            row = {"id": run_id, "kind": kind, "name": directory.name, "folder": directory.parent.name}
            if kind == "fidelity":
                status = read_json(directory / "run_status.json") or {}
                states: dict[str, int] = {}
                for job in (status.get("jobs") or {}).values():
                    states[job.get("state", "unknown")] = states.get(job.get("state", "unknown"), 0) + 1
                row.update(jobs=states, has_metrics=(directory / "metrics" / "metrics.json").is_file(),
                           updated=updated_at(directory, ("run_status.json", "metrics/metrics.json")))
            else:
                build = read_json(directory / "build.json") or {}
                answered = (sorted(p.name for p in (directory / "responses").iterdir() if p.is_dir())
                            if (directory / "responses").is_dir() else [])
                # The two judges compared (the run's judges); others' earlier answers are kept but not compared.
                compared = list((read_json(directory / "compare" / "summary.json") or {}).get("judges") or [])
                row.update(components={c: (v or {}).get("items") for c, v in (build.get("components") or {}).items()},
                           judges=compared or answered, other_judges=[j for j in answered if compared and j not in compared],
                           compared=(directory / "compare" / "summary.json").is_file(),
                           has_metrics=(directory / "metrics" / "metrics.json").is_file(),
                           adjudications=self._decision_count(directory),
                           updated=updated_at(directory, ("build.json", "compare/summary.json", "metrics/metrics.json",
                                                          "adjudications.jsonl")))
            rows.append(row)
        return sorted(rows, key=lambda r: -(r.get("updated") or 0))

    @staticmethod
    def _decision_count(directory: Path) -> int:
        from matchminer_ai.evaluation.dual_judge.judging import latest_decisions, read_jsonl

        return len(latest_decisions(read_jsonl(directory / "adjudications.jsonl")))

    # ------------------------------------------------------------------ fidelity runs

    def fidelity(self, run_id: str) -> dict:
        directory = self.run(run_id, "fidelity")
        status = read_json(directory / "run_status.json") or {}
        stages = {}
        embeddings = sorted(str(p.relative_to(directory)) for p in directory.glob("embeddings/*/*") if p.is_dir())
        for name in (*FIDELITY_STAGES[:4], *embeddings, *FIDELITY_STAGES[4:]):  # embedding comes after the corpus
            receipt = read_json(directory / name / "stage_receipt.json")
            if receipt is not None:
                stages[name] = {"content": receipt.get("content"),
                                "finished_at": (receipt.get("provenance") or {}).get("finished_at"),
                                "commit": ((receipt.get("provenance") or {}).get("repository") or {}).get("commit")}
        lists = read_json(directory / "lists_c" / "lists.json") or {}
        report = directory / "reports" / REPORT
        return {"id": run_id, "name": directory.name, "jobs": status.get("jobs") or {},
                "updated": status.get("updated_at"), "released": status.get("released"),
                "design_notes": read_json(directory / "design_notes.json") or [], "stages": stages,
                "k_binding": lists.get("k_binding"), "metrics": read_json(directory / "metrics" / "metrics.json"),
                "report": {"built": report.stat().st_mtime, "bytes": report.stat().st_size} if report.is_file() else None,
                "exports": [j.describe(with_result=True) for j in self.jobs.jobs.values()
                            if j.kind == "fidelity-report" and (j.restore or {}).get("run_id") == run_id][-3:]}

    def export(self, run_id: str):
        """Draw a fidelity run's results as a PDF in ``<run>/reports/`` (a background job; one at a time per run)."""
        directory = self.run(run_id, "fidelity")
        if not (directory / "metrics" / "metrics.json").is_file():
            raise InputError("This run has no results yet: export them when the metrics stage has run.")
        with self.jobs.lock:
            for job in self.jobs.jobs.values():
                if job.kind == "fidelity-report" and job.active and (job.restore or {}).get("run_id") == run_id:
                    return job

        def work(progress):
            from .fidelity_report import build_report

            progress(0, 3, "reading the results")
            run = self.fidelity(run_id)
            demographics, note = None, None
            if self.mode == "real":
                progress(1, 3, "linking demographics")
                demographics, note = self._demographics(directory, lambda message: progress(1, 3, message))
            else:  # synthetic mode never reads the EHR extract
                note = SYNTHETIC_NOTE
            progress(2, 3, "drawing the PDF")
            pdf = build_report(run, directory, mode=self.mode, demographics=demographics, note=note)
            from matchminer_ai.evaluation.demographics import write_private

            write_private(directory / "reports" / REPORT, pdf)
            return {"bytes": len(pdf), "demographics": demographics is not None, "note": note}

        return self.jobs.submit("fidelity-report", f"PDF of {directory.name}", work, restore={"run_id": run_id})

    def _demographics(self, directory: Path, progress):
        """Real mode: the run's Table 1 and metrics by group (cached in its reports/), or why there are none."""
        from matchminer_ai.evaluation.demographics import DemographicsError, NoDemographics, run_demographics

        root = self.extract_root or load_site_config().get("paths.retro_source_root")
        if not root:
            return None, "The site config names no EHR extract (paths.retro_source_root), so this report has no Table 1."
        error = mode_side_error(self.mode, root)
        if error:
            return None, error
        try:
            return run_demographics(directory, Path(root).expanduser(), progress), None
        except NoDemographics:
            return None, SYNTHETIC_NOTE
        except DemographicsError as exc:  # our own messages name files, never their contents
            return None, f"Table 1 is unavailable: {exc}"

    def table1(self, run_id: str) -> dict:
        """A fidelity run's Table 1 for its page, from the cache Export results writes (never computed here: linking
        the run's patients takes minutes). ``status`` is ``ready`` (with the suppressed table), ``not_built`` or
        ``unavailable`` (with why)."""
        from matchminer_ai.evaluation.demographics import TABLE1_METHOD, cached_demographics, table1_intro

        from .fidelity_report import eval_of

        directory = self.run(run_id, "fidelity")
        info = eval_of(directory)
        if self.mode != "real" or info["kind"] == "synthetic":
            return {"status": "unavailable", "note": SYNTHETIC_NOTE}
        root = self.extract_root or load_site_config().get("paths.retro_source_root")
        if not root:
            return {"status": "unavailable",
                    "note": "The site config names no EHR extract (paths.retro_source_root), so there is no Table 1."}
        error = mode_side_error(self.mode, root)
        if error:
            return {"status": "unavailable", "note": error}
        try:
            found = cached_demographics(directory, Path(root).expanduser())
        except (OSError, ValueError):  # a moved denominator or extract: Export results says why
            found = None
        if found is None:
            return {"status": "not_built", "eval": info}
        return {"status": "ready", "eval": info, "built_at": found["built_at"], "table1": found["table1"],
                "linkage": found["linkage"], "rule": found["rule"], "intro": table1_intro(info["label"]),
                "method": TABLE1_METHOD}

    def report_file(self, run_id: str) -> tuple[Path, str]:
        directory = self.run(run_id, "fidelity")
        path = directory / "reports" / REPORT
        if not path.is_file():
            raise InputError("Export the results first: this run has no PDF yet.")
        return path, f"fidelity-{directory.name}.pdf"

    # ------------------------------------------------------------------ judge runs

    def judge(self, run_id: str) -> dict:
        directory = self.run(run_id, "judge")
        build = read_json(directory / "build.json") or {}
        components = {}
        for name, value in (build.get("components") or {}).items():
            value = value or {}
            components[name] = {k: value.get(k) for k in ("items", "requests", "request_chars", "max_request_chars",
                                                          "built_at", "seconds")}
        responses = {}
        root = directory / "responses"
        for judge in sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            responses[judge] = {f.stem: sum(1 for _ in f.open(encoding="utf-8")) for f in sorted((root / judge).glob("*.jsonl"))}
        from matchminer_ai.evaluation.dual_judge.judging import JUDGE_NAMES

        return {"id": run_id, "name": directory.name, "version": build.get("version"), "components": components,
                "judge_names": JUDGE_NAMES,
                "responses": responses, "compare": read_json(directory / "compare" / "summary.json"),
                "metrics": read_json(directory / "metrics" / "metrics.json"),
                "adjudications": self._decision_count(directory),
                "recompute": [j.describe(with_result=True) for j in self.jobs.jobs.values()
                              if (j.restore or {}).get("run_id") == run_id][-3:]}

    def recompute(self, run_id: str, steps: tuple[str, ...]):
        """Rerun compare and/or metrics in a subprocess (they set their own owner-only umask); a background job."""
        directory = self.run(run_id, "judge")
        if not steps or any(s not in RECOMPUTE_STEPS for s in steps):
            raise InputError(f"steps must be among {RECOMPUTE_STEPS}")

        # A rerun keeps the comparison's own settings (its audit sample, seed and judges).
        summary = read_json(directory / "compare" / "summary.json") or {}
        compare_args = []
        if summary.get("audit_fraction") is not None:
            compare_args += ["--audit-fraction", str(summary["audit_fraction"])]
        if summary.get("seed") is not None:
            compare_args += ["--seed", str(summary["seed"])]
        if summary.get("judges"):
            compare_args += ["--judges", ",".join(summary["judges"])]

        def run(progress):
            out = {}
            for n, step in enumerate(steps):
                progress(n, len(steps), f"running {step}")
                command = [sys.executable, "-m", f"matchminer_ai.evaluation.dual_judge.{step}", "--run-dir",
                           str(directory), *(compare_args if step == "compare" else [])]
                done = subprocess.run(command, capture_output=True, text=True, timeout=1800)
                # The steps log counts only; the last lines say what they did.
                out[step] = {"returncode": done.returncode, "log": done.stdout.strip().splitlines()[-6:]}
                if done.returncode != 0:
                    raise InputError(f"{step} failed (exit {done.returncode}); run it from the command line to see why.")
            with self.lock:
                self._stores.pop(run_id, None)
            return out

        return self.jobs.submit("recompute", f"{'+'.join(steps)} for {directory.name}", run,
                                restore={"run_id": run_id, "steps": list(steps)})

    # ------------------------------------------------------------------ adjudication

    def store(self, run_id: str):
        from matchminer_ai.evaluation.dual_judge.adjudication import Store

        directory = self.run(run_id, "judge")
        if not (directory / "compare" / "units.jsonl").is_file():
            raise InputError("Run compare first: this run has no units to adjudicate yet.")
        items_stamp = updated_at(directory, ("items",))
        with self.lock:
            cached = self._stores.get(run_id)
            if cached is None or cached[0] != items_stamp:  # rebuilt items: a new store
                cached = (items_stamp, Store(directory, user=self.user))
                self._stores[run_id] = cached
            return cached[1]

    def describe(self) -> dict:
        return {"roots": [str(r) for r in self.roots], "runs": len(self.found()) if self.roots else 0}
