"""The Assistant tab: a chat agent that answers questions over the dashboard's own data, in its mode.

- **Driver** (``driver``): an in-app, provider-neutral loop. Each step is one request whose answer is one JSON action
  (``actions``): a read-only tool, a clarifying question (``ask_user``) or the answer. It runs on every backend the
  app offers, chosen per browser session as the ``assistant`` LLM task (Data and models; by default Claude Opus 5.5
  through the Databricks gateway).
- **Tools** (``tools``): the app's own services over its trials, patients, structured extraction records, guideline
  catalogs and GoodOption drugs; read-only, with no shell, file or network access.
- **Clarification:** the system prompt keeps a low threshold for asking what a clinical concept means (metastatic,
  first-line, EGFR mutant, open), one question at a time with options, and the answer states the interpretation used.
- **Data handling:** conversations live in memory per browser session (``conversation``), expire with the session
  and are never written to disk or logged. Patient summaries and records go to the selected model connection, so
  use only one authorized for the mode's data (in real mode, the Databricks gateway is approved for PHI).
- **Turns** run as session jobs (one at a time per session), polled with the conversation and stopped like any job.
"""
from __future__ import annotations

from typing import Callable

from ..core.errors import BusyError, InputError
from ..core.jobs import JobManager
from ..guidelines.settings import ASSISTANT_TASK, LLM_DECISIONS_TASK
from ..llm import Connection
from .actions import Limits
from .conversation import Conversation, Conversations
from .driver import JSONActionDriver
from .llm import connection_label, make_client
from .tools import Toolbox


class Assistant:
    """The app's assistant: each session's conversation, its turns (jobs) and the driver."""

    def __init__(self, state, *, limits: Limits | None = None, client_factory: Callable | None = None,
                 driver=None):
        self.state = state
        self.limits = limits or Limits()
        self.driver = driver or JSONActionDriver(self.limits)
        #: Makes the turn's model client from the session's ``assistant`` connection (tests pass a fake).
        self.client_factory = client_factory or make_client
        ttl = state.guidelines.settings.ttl
        self.conversations = Conversations(ttl)
        # Turns of different sessions run side by side; one at a time per session.
        self.jobs = JobManager(workers=4, ttl=ttl, exclusive=True, name="mmai-assistant")

    def connection(self, owner: str):
        return self.state.guidelines.task_settings(owner, ASSISTANT_TASK)

    def running(self, conversation: Conversation | None):
        if conversation is None or not conversation.job_id:
            return None
        job = self.jobs.get(conversation.job_id, conversation.owner)
        return job if job is not None and job.active else None

    def view(self, owner: str) -> dict:
        conversation = self.conversations.get(owner)
        selected = self.connection(owner)
        toolbox = Toolbox(self.state, owner, self.limits, {})
        running = self.running(conversation)
        return {"conversation": conversation.view() if conversation is not None else None,
                "running": running.describe(with_result=False) if running is not None else None,
                "model": {"task": ASSISTANT_TASK, "label": connection_label(selected),
                          "llm_backend": selected.llm_backend, "model": selected.model},
                "mode": self.state.settings.mode, "tools": toolbox.available(), "unavailable": toolbox.unavailable(),
                "limits": {"max_tool_calls": self.limits.max_tool_calls, "max_messages": self.limits.max_messages,
                           "max_message_chars": self.limits.max_message_chars}}

    def score_connection(self, owner: str, scorer: str):
        """The LLM connection the turn's scores come from (None for the decision model): the session's LLM decision
        checks connection, as every view that scores pairs uses when the session's "Scores from" is LLM."""
        if scorer != "llm":
            return None
        decisions = getattr(self.state, "llm_decisions", None)
        if decisions is None:
            raise InputError("The LLM decision checks are unavailable in this app; choose the decision model.")
        connection = Connection.for_task(self.state.guidelines.task_settings(owner, LLM_DECISIONS_TASK))
        decisions.require(connection)
        return connection

    def send(self, owner: str, text: str, scorer: str = "decision_model"):
        """Start a turn for the session's message (a job); the conversation's view shows its steps as they run.
        ``scorer``: where the turn's scores come from (the session's "Scores from" choice when it was sent)."""
        text = (text or "").strip()
        if not text:
            raise InputError("Write a message first.")
        if len(text) > self.limits.max_message_chars:
            raise InputError(f"Keep a message under {self.limits.max_message_chars:,} characters.")
        conversation = self.conversations.current(owner)
        if self.running(conversation) is not None:
            raise BusyError("The assistant is still answering. Wait, or stop it first.")
        if conversation.user_messages >= self.limits.max_messages:
            raise InputError("This conversation has reached its message limit. Start a new conversation.")
        # The connections are fixed when the turn is submitted, as for every job.
        selected = self.connection(owner)
        selected.require_connection()  # e.g. --local-server-only before the server is ready
        score_connection = self.score_connection(owner, scorer)
        model = connection_label(selected)

        def turn(progress):
            try:
                client = self.client_factory(selected)
            except Exception as exc:  # its type only: the text may name an endpoint's internals
                entry = conversation.begin_turn(text, model)
                entry.update(status="error", progress="", error=(
                    f"The Chat assistant connection could not be used ({type(exc).__name__}). Check it on Data "
                    "and models."))
                return {"conversation_id": conversation.id}
            toolbox = Toolbox(self.state, owner, self.limits, conversation.memory_cache, scorer=scorer,
                              llm_connection=score_connection)
            self.driver.run_turn(conversation, text, toolbox=toolbox, client=client, progress=progress, model=model)
            return {"conversation_id": conversation.id}

        try:
            job = self.jobs.submit("assistant", "assistant turn", turn, owner=owner, device="endpoint")
        except BusyError:
            raise BusyError("The assistant is still answering. Wait, or stop it first.") from None
        conversation.job_id = job.id
        return job

    def stop(self, owner: str) -> bool:
        conversation = self.conversations.get(owner)
        running = self.running(conversation)
        if running is None:
            return False
        self.jobs.cancel(running.id, owner)
        return True

    def reset(self, owner: str) -> None:
        """A new conversation: a running turn is stopped first."""
        self.stop(owner)
        self.conversations.clear(owner)

    def clean(self, active_owners: set[str] | None = None) -> None:
        self.jobs.clean()
        self.conversations.clean(active_owners if active_owners is not None else self.jobs.active_owners())

    def shutdown(self) -> None:
        self.jobs.shutdown()


__all__ = ["Assistant", "Limits"]
