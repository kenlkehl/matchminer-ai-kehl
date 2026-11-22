"""The Assistant tab: the session's conversation, a message (a turn, run as a job), stopping and a new conversation.

- ``GET /api/assistant``: the conversation (transcript, steps so far, links, token use), the running turn, and the
  session's Chat assistant connection; the page polls it while a turn runs.
- ``POST /api/assistant/messages``: send a message (or the answer to a clarifying question); 409 while a turn runs.
- ``POST /api/assistant/stop``: stop the running turn.
- ``DELETE /api/assistant``: start a new conversation (a running turn is stopped).

The model connection is the session's ``assistant`` task (``/api/models``, Data and models).
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from ..state import DashState
from .common import owner


class MessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20_000)
    #: The session's "Scores from" choice: the turn's reranks and pair scores use it.
    scorer: Literal["decision_model", "llm"] = "decision_model"


def router(state: DashState) -> APIRouter:
    api = APIRouter()
    assistant = state.assistant

    @api.get("/api/assistant")
    def conversation(request: Request):
        return assistant.view(owner(request))

    @api.post("/api/assistant/messages", status_code=202)
    def message(body: MessageRequest, request: Request):
        job = assistant.send(owner(request), body.text, body.scorer)
        return {"job": job.describe(with_result=False), **assistant.view(owner(request))}

    @api.post("/api/assistant/stop")
    def stop(request: Request):
        return {"stopped": assistant.stop(owner(request))}

    @api.delete("/api/assistant", status_code=204)
    def new_conversation(request: Request):
        assistant.reset(owner(request))
        return Response(status_code=204)

    return api
