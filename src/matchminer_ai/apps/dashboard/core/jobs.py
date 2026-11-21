"""Background jobs with polled progress, for the whole app.

A job runs ``function(progress)`` on its manager's thread pool. Model work runs one job at a time per manager (one
model in process); the app has a manager per kind of work (the decision model, background store work, LLM checks,
guideline inference), and ``JobRegistry`` finds a job in any of them.

- **Owner:** a job started from a browser belongs to its session (the session cookie), and only that session can
  see or stop it. System jobs (store rebuilds, model loading) have no owner and are visible to every session.
- **Busy:** a manager can allow one active job per session (``exclusive``), as guideline inference does.
- **Stopping:** ``progress`` checks the job's cancellation token, and the job runs inside
  ``matchminer_ai.cancellation.cancellation_scope``, so package code that checks for cancellation stops too.
- **Errors:** an ``InputError``'s message is shown as it is. Anything else is reduced to its type and a fixed
  message, because provider errors and exception text can quote patient text or file names; the server log gets
  only the type and the code locations.
- **Expiry:** finished jobs are dropped after ``ttl`` seconds, and at most ``keep`` jobs are kept.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import secrets
import sys
import threading
import time
import traceback

from matchminer_ai.cancellation import CancellationToken, InferenceCancelled, cancellation_scope

from .errors import BusyError, InputError

ACTIVE = frozenset({"queued", "running", "cancelling"})
FINISHED = frozenset({"done", "error", "cancelled"})
MAX_ACTIVE_SESSION_JOBS = 4
GENERIC_ERROR = "the job could not finish. Check the model connection and the server, then retry."


@dataclass
class Job:
    id: str
    kind: str
    label: str
    owner: str | None = None
    status: str = "queued"  # queued, running, cancelling, done, error, cancelled
    done: int = 0
    total: int = 0
    message: str = ""
    device: str | None = None  # where the model runs ("cpu", "cuda:3", "endpoint"), shown with the progress
    result: object = None
    error: str | None = None
    restore: object = None  # what a reloaded page needs to show the job's inputs again
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    token: CancellationToken = field(default_factory=CancellationToken, repr=False)
    future: Future | None = field(default=None, repr=False)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    def describe(self, with_result: bool = True) -> dict:
        value = {k: getattr(self, k) for k in ("id", "kind", "label", "status", "done", "total", "message", "error",
                                               "device", "created", "updated", "started", "finished")}
        value["seconds"] = round((self.finished or time.time()) - self.started, 1) if self.started else None
        value["restore"] = self.restore
        if with_result:
            value["result"] = self.result
        return value


def public_error(exc: BaseException) -> str:
    """What the page shows for a failed job."""
    if isinstance(exc, InputError):
        return str(exc)
    return f"{type(exc).__name__}: {GENERIC_ERROR}"


def log_failure(job: Job, exc: BaseException) -> None:
    """The exception's type and where it was raised (file, line, function), never its message."""
    frames = traceback.extract_tb(exc.__traceback__)[-4:]
    where = "; ".join(f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}" for f in frames)
    print(f"job {job.kind} failed: {type(exc).__name__} at {where}", file=sys.stderr, flush=True)


class JobManager:
    def __init__(self, workers: int = 1, keep: int = 200, ttl: float = 3600, exclusive: bool = False,
                 name: str = "mmai-job"):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.lock = threading.RLock()
        self.keep, self.ttl, self.exclusive = keep, ttl, exclusive

    # ------------------------------------------------------------------ bookkeeping

    def clean(self) -> None:
        now = time.time()
        with self.lock:
            for key, job in list(self.jobs.items()):
                if job.status in FINISHED and now - job.updated > self.ttl:
                    del self.jobs[key]
            while len(self.jobs) > self.keep:
                oldest = next((k for k, j in self.jobs.items() if not j.active), None)
                if oldest is None:
                    break
                del self.jobs[oldest]

    def active_owners(self) -> set[str]:
        with self.lock:
            return {j.owner for j in self.jobs.values() if j.active and j.owner}

    # ------------------------------------------------------------------ running

    def submit(self, kind: str, label: str, function, *, device: str | None = None, owner: str | None = None,
               restore=None) -> Job:
        """Run ``function(progress)``. ``progress(done, total, message)`` or ``progress(message)`` updates the job
        and raises ``InferenceCancelled`` once it is stopped."""
        with self.lock:
            self.clean()
            if self.exclusive and owner is not None:
                active = [j for j in self.jobs.values() if j.active and j.owner]
                if len(active) >= MAX_ACTIVE_SESSION_JOBS or any(j.owner == owner for j in active):
                    raise BusyError("A job is already running. Wait for it to finish before starting another.")
            job = Job(secrets.token_urlsafe(18), kind, label, owner=owner, device=device, restore=restore,
                      message="waiting for the worker")
            self.jobs[job.id] = job

        def progress(done=None, total=None, message=None):
            job.token.check()
            if isinstance(done, str):
                done, message = None, done
            with self.lock:
                if done is not None:
                    job.done = int(done)
                if total is not None:
                    job.total = int(total)
                if message is not None:
                    job.message = str(message)
                job.updated = time.time()

        def run():
            try:
                with cancellation_scope(job.token):
                    with self.lock:
                        job.token.check()
                        job.status, job.started, job.message = "running", time.time(), ""
                    result = function(progress)
                    job.token.check()
                with self.lock:
                    job.result, job.status = result, "done"
            except InferenceCancelled:
                with self.lock:
                    job.status, job.message = "cancelled", "Stopped. No further inference requests will be sent."
            except Exception as exc:
                log_failure(job, exc)
                with self.lock:
                    if job.token.event.is_set():
                        job.status, job.message = "cancelled", "Stopped."
                    else:
                        job.status, job.error = "error", public_error(exc)
            finally:
                with self.lock:
                    job.finished = job.updated = time.time()

        with self.lock:
            job.future = self.pool.submit(run)
        return job

    def get(self, job_id: str, owner: str | None = None) -> Job | None:
        """The job, if ``owner`` may see it (its own session's jobs and system jobs)."""
        with self.lock:
            self.clean()
            job = self.jobs.get(job_id)
            if job is None or (job.owner is not None and job.owner != owner):
                return None
            return job

    def cancel(self, job_id: str, owner: str | None) -> Job | None:
        """Stop a session's job: a queued job never starts; a running one stops at its next progress check."""
        with self.lock:
            job = self.get(job_id, owner)
            if job is None or job.owner is None:  # system jobs are not stopped from a page
                return None
            if job.active:
                job.token.cancel()
                queued = job.future is not None and job.future.cancel()
                job.status = "cancelled" if queued else "cancelling"
                job.message = "Stopped." if queued else "Stopping…"
                job.updated = time.time()
                if queued:
                    job.finished = job.updated
            return job

    def latest(self, owner: str, *, prefix: str | None = None, exclude_prefix: str | None = None) -> Job | None:
        """The session's most recently updated job (of kinds starting with ``prefix``, not ``exclude_prefix``)."""
        with self.lock:
            self.clean()
            mine = [j for j in self.jobs.values() if j.owner == owner
                    and (prefix is None or j.kind.startswith(prefix))
                    and not (exclude_prefix and j.kind.startswith(exclude_prefix))]
            return max(mine, key=lambda j: j.updated) if mine else None

    def clear(self, owner: str, *, prefix: str | None = None) -> None:
        """Forget a session's jobs (those whose kind starts with ``prefix``); refused while one is active."""
        with self.lock:
            mine = [k for k, j in self.jobs.items() if j.owner == owner and (prefix is None or j.kind.startswith(prefix))]
            if any(self.jobs[k].active for k in mine):
                raise BusyError("Wait for the current job to finish before clearing the workspace.")
            for key in mine:
                del self.jobs[key]

    def recent(self, n: int = 20, owner: str | None = None) -> list[dict]:
        with self.lock:
            visible = [j for j in self.jobs.values() if j.owner is None or j.owner == owner]
        return [job.describe(with_result=False) for job in visible[-n:]][::-1]

    def shutdown(self):
        with self.lock:
            for job in self.jobs.values():
                if job.active:
                    job.token.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)


class JobRegistry:
    """The app's job managers, so one route can find and stop any job."""

    def __init__(self, **managers: JobManager):
        self.managers = dict(managers)

    def add(self, name: str, manager: JobManager) -> JobManager:
        self.managers[name] = manager
        return manager

    def get(self, job_id: str, owner: str | None) -> Job | None:
        for manager in self.managers.values():
            job = manager.get(job_id, owner)
            if job is not None:
                return job
        return None

    def cancel(self, job_id: str, owner: str | None) -> Job | None:
        for manager in self.managers.values():
            job = manager.cancel(job_id, owner)
            if job is not None:
                return job
        return None

    def recent(self, n: int = 10, owner: str | None = None) -> list[dict]:
        rows = [row for manager in self.managers.values() for row in manager.recent(n, owner)]
        return sorted(rows, key=lambda r: r["created"], reverse=True)[:3 * n]

    def active_owners(self) -> set[str]:
        return set().union(*(m.active_owners() for m in self.managers.values())) if self.managers else set()

    def shutdown(self):
        for manager in self.managers.values():
            manager.shutdown()
