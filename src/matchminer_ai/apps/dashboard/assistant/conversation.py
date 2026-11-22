"""Conversations, in memory per browser session: the page's transcript, the model's messages and the token use.

A conversation is never written to disk or logged. It expires ``ttl`` seconds after its last use (the app's watch
loop calls ``Conversations.clean``), or when the session starts a new one.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import secrets
import threading
import time

USAGE_KEYS = ("requests", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def zero_usage() -> dict:
    return dict.fromkeys(USAGE_KEYS, 0)


@dataclass
class Conversation:
    owner: str
    id: str = field(default_factory=lambda: secrets.token_urlsafe(12))
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    #: The page's transcript: user messages, and assistant turns with their steps and outcome.
    entries: list[dict] = field(default_factory=list)
    #: The model's messages after the system blocks: {"role": "user" | "assistant", "content": str}.
    history: list[dict] = field(default_factory=list)
    #: Every ID the tools returned: {id: {"kind": "trial" | "cohort" | "patient", ...}} (the page's links).
    links: dict[str, dict] = field(default_factory=dict)
    usage: dict = field(default_factory=zero_usage)
    #: Ad-hoc criteria lines' compiled queries, for this conversation only.
    memory_cache: dict = field(default_factory=dict)
    job_id: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def user_messages(self) -> int:
        return sum(1 for e in self.entries if e["role"] == "user")

    def pending_question(self) -> dict | None:
        """The clarifying question the last turn ended with, if the user has not answered it."""
        last = self.entries[-1] if self.entries else None
        return last if last and last["role"] == "assistant" and last.get("kind") == "question" else None

    def touch(self) -> None:
        self.updated = time.time()

    def begin_turn(self, text: str, model: str) -> dict:
        with self.lock:
            self.entries.append({"id": len(self.entries), "role": "user", "text": text, "at": time.time()})
            turn = {"id": len(self.entries), "role": "assistant", "status": "running", "model": model, "steps": [],
                    "progress": "Starting", "kind": None, "text": "", "options": [], "why": "",
                    "interpretation": "", "error": None, "at": time.time(), "seconds": None}
            self.entries.append(turn)
            self.touch()
            return turn

    def view(self) -> dict:
        with self.lock:
            return {"id": self.id, "created": self.created, "updated": self.updated,
                    "entries": copy.deepcopy(self.entries), "links": dict(self.links), "usage": dict(self.usage),
                    "job_id": self.job_id, "user_messages": self.user_messages}


class Conversations:
    """Each browser session's current conversation."""

    def __init__(self, ttl: float = 3600):
        self.ttl = ttl
        self.items: dict[str, Conversation] = {}
        self.lock = threading.Lock()

    def get(self, owner: str) -> Conversation | None:
        with self.lock:
            found = self.items.get(owner)
            if found is not None:
                found.touch()
            return found

    def current(self, owner: str) -> Conversation:
        with self.lock:
            found = self.items.get(owner)
            if found is None:
                found = self.items[owner] = Conversation(owner)
            found.touch()
            return found

    def clear(self, owner: str) -> None:
        with self.lock:
            self.items.pop(owner, None)

    def clean(self, active_owners: set[str] | None = None) -> None:
        now = time.time()
        with self.lock:
            for owner, conversation in list(self.items.items()):
                if now - conversation.updated > self.ttl and owner not in (active_owners or set()):
                    del self.items[owner]
