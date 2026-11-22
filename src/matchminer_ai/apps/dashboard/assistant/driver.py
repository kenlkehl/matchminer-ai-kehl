"""The assistant's agent loop, behind a small driver interface.

A driver runs one user message's turn: model steps and tool calls until the model answers or asks a clarifying
question, recording each step in the conversation as it happens (the page polls it). ``JSONActionDriver`` is the
provider-neutral loop the app uses: each step is one request whose answer is one JSON action (``actions``), validated
in code, the pattern of the SQL workup agent; it works on every backend the app offers. Another driver (such as the
Claude Agent SDK restricted to these tools) would implement the same ``run_turn``; the tools stay the app's own
read-only functions either way, never shell or file access.

The model's messages are the system prompt (cached by Claude), the data overview, then the conversation: user
messages, the model's actions and the tool results, each kept verbatim so that a later step's request starts with
the earlier one's (Claude reads it from the cache). A result names its step (numbered through the conversation, as
the page shows them, so that a chart can cite where its numbers came from) and says how many tool calls are left;
once none are, the response schema offers only ``ask_user`` and ``answer``. An answer with ``draw_charts`` is followed by
one more request for its charts (``actions.charts_schema``); the answer stands without them if that request fails.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Protocol

from matchminer_ai.cancellation import InferenceCancelled
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.llm.remote_auth import DATABRICKS_ANTHROPIC_PROVIDER
from matchminer_ai.llm.structured import CredentialsError, EndpointError

from .actions import (
    Limits, chart_view, charts_schema, compact, parse_call, response_schema, validate, validate_charts,
)
from .conversation import USAGE_KEYS, Conversation
from .tools import LABELS, ToolResult

PROMPT_FILE = "dashboard.assistant.system.txt"
OMITTED = "[This tool result was omitted to fit the model's context; run the tool again if it is needed.]"


class ContextFull(Exception):
    """The conversation does not fit the model's context even with every tool result omitted."""


class Driver(Protocol):
    name: str

    def _charts(self, conversation: Conversation, overview: dict, client) -> tuple[list[dict], str | None]:
        """The charts of the answer just given (its follow-up request), and why there are none if it failed."""
        limits = self.limits
        with conversation.lock:
            conversation.history.append({"role": "user", "content": charts_request(limits)})
        cited = conversation.result_steps()
        failed_checks = len(getattr(client, "validation_errors", None) or [])
        try:
            value = client.complete("assistant", self._messages(conversation, overview, client), charts_schema(),
                                    lambda v: validate_charts(v, limits, cited))
        except EndpointError:
            checks = len(getattr(client, "validation_errors", None) or []) > failed_checks
            with conversation.lock:
                conversation.history.pop()  # the request: the answer stands without charts
            return [], ("The charts did not pass the checks, so none are shown." if checks else
                        "The charts could not be drawn: the model connection failed.")
        with conversation.lock:
            conversation.history.append({"role": "assistant", "content": compact(value)})
        return [chart_view(c) for c in value["charts"]], None

    def run_turn(self, conversation: Conversation, text: str, *, toolbox, client, progress: Callable,
                 model: str) -> None:
        """Answer one user message, recording its steps and outcome in ``conversation``."""


def system_prompt(limits: Limits) -> str:
    return (load_prompt_text(PROMPT_FILE).replace("{max_tool_calls}", str(limits.max_tool_calls))
            .replace("{max_rows}", str(limits.max_rows)))


def render_result(result: ToolResult, calls_left: int, limits: Limits, step: int) -> str:
    body = json.dumps(result.result, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(body) > limits.max_result_chars:
        body = body[: limits.max_result_chars] + " … [cut at the result size limit]"
    state = "ok" if result.ok else "error"
    return (f"Result of step {step}, {result.tool} ({state}; {calls_left} tool call{'' if calls_left == 1 else 's'} "
            f"left for this message):\n{body}")


def charts_request(limits: Limits) -> str:
    """The message that asks for an answer's charts, with the checks they must pass (a failed check is retried
    without saying which one failed)."""
    return (f"Now the charts for your answer: one JSON object {{\"charts\": [...]}} with 1-{limits.max_charts} "
            "charts, following the chart rules. Copy each number exactly from the tool results above and list those "
            f"results' step numbers in steps. Checks: a pie or donut has 2-{limits.max_pie_slices} labels and one "
            "series, with values of 0 or more that add up to total; bars have "
            f"1-{limits.max_chart_categories} labels and 1-{limits.max_chart_series} series (each named when there "
            "are two or more; stacked values 0 or more; when total is above 0, no bar is above it); a line has "
            f"2-{limits.max_chart_categories} points; every series has one value per label; labels are distinct.")


def endpoint_message(client, exc: Exception) -> str:
    """What the page says about a failed model request: fixed texts and an HTTP status only."""
    if isinstance(exc, CredentialsError):
        return str(exc)
    statuses = [m.get("http_status") for m in getattr(client, "request_metrics", []) if m.get("http_status")]
    if statuses:
        return (f"The model connection failed (HTTP {statuses[-1]}). Check the Chat assistant connection on Data "
                "and models, then send the message again.")
    if getattr(client, "validation_errors", None):
        return "The model's answers did not fit the assistant's action format. Send the message again."
    return "The model connection failed. Check the Chat assistant connection on Data and models, then try again."


class JSONActionDriver:
    name = "json-actions"

    def __init__(self, limits: Limits | None = None):
        self.limits = limits or Limits()
        self.system = system_prompt(self.limits)

    def _messages(self, conversation: Conversation, overview: dict, client) -> list[dict]:
        context = "Data overview (this dashboard, now):\n" + json.dumps(overview, ensure_ascii=False, default=str)
        head = ([{"role": "system", "content": self.system}, {"role": "system", "content": context}]
                if getattr(client, "provider", "") == DATABRICKS_ANTHROPIC_PROVIDER
                else [{"role": "system", "content": self.system + "\n\n" + context}])
        while True:
            messages = head + conversation.history
            if client.fits(messages):
                return messages
            # The oldest tool result still shown makes way (the conversation's prefix changes only now).
            oldest = next((m for m in conversation.history if m["role"] == "user"
                           and m["content"].startswith("Result of ") and m["content"] != OMITTED), None)
            if oldest is None:
                raise ContextFull()
            oldest["content"] = OMITTED

    def _charts(self, conversation: Conversation, overview: dict, client) -> tuple[list[dict], str | None]:
        """The charts of the answer just given (its follow-up request), and why there are none if it failed."""
        limits = self.limits
        with conversation.lock:
            conversation.history.append({"role": "user", "content": charts_request(limits)})
        cited = conversation.result_steps()
        failed_checks = len(getattr(client, "validation_errors", None) or [])
        try:
            value = client.complete("assistant", self._messages(conversation, overview, client), charts_schema(),
                                    lambda v: validate_charts(v, limits, cited))
        except EndpointError:
            checks = len(getattr(client, "validation_errors", None) or []) > failed_checks
            with conversation.lock:
                conversation.history.pop()  # the request: the answer stands without charts
            return [], ("The charts did not pass the checks, so none are shown." if checks else
                        "The charts could not be drawn: the model connection failed.")
        with conversation.lock:
            conversation.history.append({"role": "assistant", "content": compact(value)})
        return [chart_view(c) for c in value["charts"]], None

    def run_turn(self, conversation: Conversation, text: str, *, toolbox, client, progress: Callable,
                 model: str) -> None:
        limits = self.limits
        reply = conversation.pending_question() is not None
        turn = conversation.begin_turn(text, model)
        started = time.monotonic()
        base = dict(conversation.usage)
        tools = toolbox.available()
        framing = "The user's answer to your question" if reply else "User message"
        with conversation.lock:
            conversation.history.append({"role": "user", "content": (
                f"{framing} ({limits.max_tool_calls} tool calls available for this message):\n{text}")})

        def update(**values):
            with conversation.lock:
                turn.update(values)
                usage = client.usage() if hasattr(client, "usage") else {}
                conversation.usage = {k: base.get(k, 0) + usage.get(k, 0) for k in USAGE_KEYS}
                conversation.touch()

        calls_left = limits.max_tool_calls
        made: dict[str, int] = {}  # each call's arguments this turn, and its step
        try:
            overview = toolbox.overview()
            while True:
                update(progress="Thinking")
                progress("Thinking")
                final_only = calls_left <= 0
                schema = response_schema(final_only, tools)
                messages = self._messages(conversation, overview, client)
                value = client.complete("assistant", messages, schema,
                                        lambda v: validate(v, limits, final_only=final_only, tools=tools))
                call = parse_call(value)
                with conversation.lock:
                    conversation.history.append({"role": "assistant", "content": compact(value)})
                if call["tool"] == "answer":
                    # The answer shows at once; its charts, when asked for, follow from one more request.
                    update(kind="answer", text=call["markdown"].strip(),
                           interpretation=call["interpretation"].strip(),
                           progress="Drawing charts" if call["draw_charts"] else "")
                    if call["draw_charts"]:
                        progress("Drawing charts")
                        charts, why = self._charts(conversation, overview, client)
                        update(charts=charts, chart_error=why)
                    update(status="done", progress="")
                    break
                if call["tool"] == "ask_user":
                    update(status="done", kind="question", text=call["question"].strip(),
                           options=[o.strip() for o in call["options"]], why=call["why"].strip(), progress="")
                    break
                calls_left -= 1
                with conversation.lock:
                    step = {"number": conversation.step_count() + 1, "tool": call["tool"],
                            "label": LABELS[call["tool"]], "purpose": value["purpose"].strip(), "summary": "",
                            "ok": None, "seconds": None}
                    turn["steps"].append(step)
                update(progress=f"{LABELS[call['tool']]}…")

                def step_progress(*args):
                    progress(*args)  # raises once the turn is stopped
                    message = next((a for a in args if isinstance(a, str)), None)
                    if message:
                        update(progress=message)

                same = made.get(compact(call))
                made.setdefault(compact(call), step["number"])
                try:
                    if same is not None:  # a repeated call: its result is already in the conversation
                        result = ToolResult(call["tool"], False, f"{LABELS[call['tool']]}: the same call as step "
                                            f"{same}", {"error": f"This is the same call as step {same}; use its "
                                                                 "result above, or change the arguments."})
                    else:
                        result = toolbox.run(call, step_progress)
                except Exception as exc:  # never the exception's text: it may quote data
                    result = ToolResult(call["tool"], False, f"{LABELS[call['tool']]}: the tool failed "
                                        f"({type(exc).__name__})", {"error": f"the tool failed ({type(exc).__name__})"})
                with conversation.lock:
                    step.update(summary=result.summary, ok=result.ok, seconds=result.seconds,
                                **({"llm_answers": result.llm_answers} if result.llm_answers else {}))
                    conversation.links.update(result.links)
                    conversation.history.append({"role": "user", "content": render_result(
                        result, calls_left, limits, step["number"])})
        except ContextFull:
            update(status="error", error="The conversation is too long for this model's context. Start a new "
                                         "conversation.", progress="")
        except EndpointError as exc:
            update(status="error", error=endpoint_message(client, exc), progress="")
        except InferenceCancelled:
            update(status="stopped", progress="")
            with conversation.lock:
                conversation.history.append({"role": "user", "content": "(The user stopped this turn.)"})
            raise
        except Exception as exc:  # a fault of the app: its type only (exception text may quote data)
            update(status="error", error=f"The assistant failed ({type(exc).__name__}).", progress="")
            raise
        finally:
            update(seconds=round(time.monotonic() - started, 1))

