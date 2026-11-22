"""The Assistant tab's chat agent: its action schema, its model default, and a scripted conversation through the
routes (fabricated trials and patients, a fake model client; no endpoint is called)."""
from __future__ import annotations

import pytest

from . import test_dash
from .conftest import wait_job

# The fabricated dashboard's fixtures (test_dash.py).
dash, dash_client, dash_settings, fake_ncit, trial_data = (test_dash.dash, test_dash.dash_client, test_dash.dash_settings,
                                                         test_dash.fake_ncit, test_dash.trial_data)


def search(text="", **overrides):
    return {"search_trials": {"text": text, "drug": "", "cancer_type": "", "center": "", "phases": [],
                              "open_only": False, "include_common_criteria": False, **overrides}}


def ask(question, options, why=""):
    return {"ask_user": {"question": question, "options": options, "why": why}}


def answer(markdown, interpretation=""):
    return {"answer": {"markdown": markdown, "interpretation": interpretation}}


def test_action_schema_and_validation():
    from matchminer_ai.apps.dashboard.assistant.actions import TOOLS, Limits, parse_call, response_schema, validate

    limits = Limits()
    names = [next(iter(v["properties"])) for v in response_schema()["properties"]["call"]["anyOf"]]
    assert names == [*TOOLS, "ask_user", "answer"]
    final = response_schema(final_only=True)["properties"]["call"]["anyOf"]
    assert [next(iter(v["properties"])) for v in final] == ["ask_user", "answer"]
    assert parse_call(validate({"purpose": "Search", "call": search("EGFR")}, limits))["text"] == "EGFR"
    question = ask("Which EGFR alterations?", ["Sensitizing (exon 19 deletion, L858R)", "Exon 20 insertion",
                                                "Any EGFR alteration"], "It changes the cohorts.")
    assert validate({"purpose": "Clarify", "call": question}, limits)
    bad = [
        ({"purpose": "x", "call": {"search_trials": {**search()["search_trials"], "extra": 1}}}, "exactly the fields"),
        ({"purpose": "x", "call": {**search(), **answer("Two calls")}}, "one key"),
        ({"purpose": "x", "call": search(open_only="yes")}, "true or false"),
        ({"purpose": "x", "call": search(cancer_type="not-a-type")}, "allowed values"),
        ({"purpose": "x", "call": ask("Which?", ["Only one"])}, "2-5"),
        ({"purpose": "x", "call": answer(" ")}, "Markdown"),
    ]
    for value, message in bad:
        with pytest.raises(ValueError, match=message):
            validate(value, limits)
    with pytest.raises(ValueError, match="ask_user or answer now"):  # the tool budget is spent
        validate({"purpose": "x", "call": search()}, limits, final_only=True)


def test_the_assistant_task_defaults_to_opus_through_the_gateway(tmp_path, monkeypatch):
    from matchminer_ai.apps.dashboard.assistant.llm import structured_config
    from matchminer_ai.apps.dashboard.settings import DashSettings, guideline_settings
    from matchminer_ai.llm import anthropic_messages

    site = tmp_path / "gateway.yaml"
    site.write_text("databricks:\n  host: https://workspace.example.cloud.databricks.com\n  profile: p\n"
                    "  endpoints:\n    haiku: catalog.schema.team-claude-haiku-5-5\n"
                    "    sonnet: catalog.schema.team-claude-sonnet-5-5\n    opus: catalog.schema.team-claude-opus-5-5\n")
    monkeypatch.setenv("MMAI_SITE_CONFIG", str(site))
    tasks = guideline_settings(DashSettings(mode="synthetic", store_dir=tmp_path / "s", device="cpu"))
    selected = tasks.for_task("assistant")
    assert (selected.llm_backend, selected.model) == ("claude_databricks", "catalog.schema.team-claude-opus-5-5")
    assert tasks.for_task("summarize").model.endswith("haiku-5-5")  # the app-wide default is unchanged
    config = structured_config(selected)
    assert config.provider == "databricks_anthropic" and config.model.endswith("opus-5-5")
    assert config.base_url == "https://workspace.example.cloud.databricks.com/ai-gateway/anthropic"
    assert config.request_params == {"reasoning_effort": "medium", "cache_control": {"type": "ephemeral"}}
    request = anthropic_messages.messages_request({
        "model": config.model, "max_tokens": 10, "cache_control": {"type": "ephemeral"},
        "messages": [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Hi."}]})
    assert request["cache_control"] == {"type": "ephemeral"}  # the conversation so far is cached


class FakeClient:
    """Answers each step from a script, through the real validator; records the messages it was sent."""

    provider = "openai"

    def __init__(self, script):
        self.script, self.sent = list(script), []

    def fits(self, messages):
        return True

    def complete(self, job, messages, schema, validator):
        self.sent.append([dict(m) for m in messages])
        return validator(self.script.pop(0))

    def usage(self):
        steps = len(self.sent)
        return {"requests": steps, "input_tokens": 100 * steps, "output_tokens": 10 * steps,
                "cache_read_tokens": 0, "cache_write_tokens": 0}


def test_a_scripted_conversation_asks_then_searches_and_answers(dash_client, dash):
    clients = [FakeClient([{"purpose": "Clarify EGFR", "call": ask(
                   "Which EGFR alterations do you mean?", ["Exon 19 deletion or L858R", "Any EGFR alteration"],
                   "Exon 19 deletion cohorts differ from exon 20 insertion cohorts.")}]),
               FakeClient([{"purpose": "Search open EGFR trials", "call": search("EGFR exon", open_only=True)},
                           {"purpose": "Answer", "call": answer(
                               "NCT90000001 has NCT90000001-cohort-0 for EGFR exon 19 deletions.",
                               "Interpreted EGFR mutant as exon 19 deletion or L858R.")}])]
    dash.assistant.client_factory = lambda selected: clients.pop(0)
    view = dash_client.get("/api/assistant").json()
    assert view["conversation"] is None and "search_trials" in view["tools"]
    assert "screen_records" in view["unavailable"]  # no extraction store in this fixture

    started = dash_client.post("/api/assistant/messages", json={"text": "Open trials for EGFR mutant NSCLC?"})
    assert started.status_code == 202
    wait_job(dash_client, started.json()["job"]["id"])
    entries = dash_client.get("/api/assistant").json()["conversation"]["entries"]
    assert [e["role"] for e in entries] == ["user", "assistant"]
    assert entries[1]["kind"] == "question" and entries[1]["options"][0] == "Exon 19 deletion or L858R"

    wait_job(dash_client, dash_client.post("/api/assistant/messages",
                                           json={"text": "Exon 19 deletion or L858R"}).json()["job"]["id"])
    view = dash_client.get("/api/assistant").json()["conversation"]
    turn = view["entries"][-1]
    assert turn["status"] == "done" and turn["kind"] == "answer" and turn["interpretation"].startswith("Interpreted")
    step = turn["steps"][0]
    assert step["tool"] == "search_trials" and step["ok"] and "→ 1 cohorts in 1 trials" in step["summary"]
    assert view["links"]["NCT90000001-cohort-0"]["kind"] == "cohort" and view["links"]["NCT90000001"]["kind"] == "trial"
    assert view["usage"]["requests"] == 3
    (history,) = [c.history for c in dash.assistant.conversations.items.values()]
    assert history[2]["content"].startswith("The user's answer to your question")  # carried forward as an answer
    assert "tool calls left" in history[4]["content"] and "NCT90000001-cohort-0" in history[4]["content"]

    assert dash_client.post("/api/assistant/messages", json={"text": ""}).status_code in (400, 422)
    assert dash_client.delete("/api/assistant").status_code == 204
    assert dash_client.get("/api/assistant").json()["conversation"] is None
