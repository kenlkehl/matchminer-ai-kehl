"""The assistant's action schema: one JSON object per step, ``{"purpose", "call"}``, validated in code.

``call`` has one key, the name of a tool (``TOOLS``: read-only searches over the dashboard's own data), of
``ask_user`` (a clarifying question with options; ends the turn) or of ``answer`` (the final Markdown and the
interpretation used; ends the turn), and its value holds the arguments: ``{"search_trials": {"text": "EGFR", ...}}``.
The name is the key, not a ``tool`` field beside the arguments, because constrained decoding then fixes the
arguments' schema by the name (with a ``tool`` field among ``anyOf`` variants, Claude's structured outputs returned
another tool's empty arguments). Every argument is required (strict JSON-schema outputs), with ``""``, ``[]``,
``false`` or ``0`` for an unused one. The endpoint's JSON-schema response format constrains the shape;
:func:`validate` checks it again with fixed messages (never input text), and each tool checks its arguments' meaning
when it runs. :func:`parse_call` gives the code's form, ``{"tool": name, **arguments}``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from ..core import cancer_types

FINAL = ("ask_user", "answer")
SEXES = ("", "male", "female")
PHASES = ("1", "2", "3", "4")
LINE_TYPES = ("inclusion", "exclusion")


@dataclass(frozen=True)
class Limits:
    """Budgets of one conversation."""

    #: Tool calls per user message; then the model must answer or ask.
    max_tool_calls: int = 8
    #: Rows a tool result lists (totals are always given).
    max_rows: int = 25
    #: Characters of one tool result in the conversation.
    max_result_chars: int = 12000
    #: Characters of a patient summary section a tool result shows.
    max_text_chars: int = 4000
    #: Trials or cohorts one trial_details call shows, and cohorts per trial.
    max_detail_ids: int = 5
    max_detail_cohorts: int = 8
    #: Criteria lines one screen shows per cohort, and ad-hoc lines per screen.
    max_screen_lines: int = 10
    #: Pairs the decision model scores per call (rerank or score_pairs).
    max_score_pairs: int = 20
    #: Pairs the LLM scores per call when the session's scores come from the LLM (the views' first batch).
    max_llm_pairs: int = 10
    #: Candidates a TrialSpace search ranks.
    max_k: int = 50
    #: User messages per conversation, and characters per message.
    max_messages: int = 40
    max_message_chars: int = 4000
    max_answer_chars: int = 16000
    max_question_chars: int = 600
    max_option_chars: int = 200
    max_purpose_chars: int = 300
    max_text_arg_chars: int = 4000

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"assistant limit {item.name} must be a positive integer")


TEXT = {"type": "string"}
BOOL = {"type": "boolean"}
INT = {"type": "integer"}


def _string(description: str, enum=None) -> dict:
    out = {"type": "string", "description": description}
    if enum is not None:
        out["enum"] = list(enum)
    return out


def _cancer_type() -> dict:
    return _string("A cancer-type filter key from the data overview, or '' for none.",
                   ["", *cancer_types.TYPE_LABELS])


#: Each tool's arguments (JSON schema properties) and what it is for, in the order the prompt lists them.
TOOLS: dict[str, dict] = {
    "search_trials": {
        "description": "Text search over the dashboard's trial cohorts (trial ID, title, cohort name, key criteria, "
                       "drug names), with filters.",
        "properties": {
            "text": _string("Search terms: case-insensitive regular expressions, all must match (AND); a term with "
                            "spaces in double quotes. '' for none."),
            "drug": _string("A regular expression over the trials' GoodOption drug names and drug classes; '' for "
                            "none."),
            "cancer_type": _cancer_type(),
            "center": _string("A center ID or part of its name from the data overview; '' for any."),
            "phases": {"type": "array", "items": _string("A phase number.", PHASES),
                       "description": "Keep trials with any of these phases; [] for any."},
            "open_only": {**BOOL, "description": "Only trials the data marks open (see the data overview)."},
            "include_common_criteria": {**BOOL, "description": "Also search the boilerplate (common) criteria."},
        },
    },
    "match_trials": {
        "description": "TrialSpace semantic retrieval: the trial cohorts nearest a patient description, one per "
                       "trial, optionally scored by the decision model.",
        "properties": {
            "description": _string("A patient cancer-history summary in the corpus style (age, sex, cancer type, "
                                   "histology, extent, biomarkers, treatment history)."),
            "cancer_type": _cancer_type(),
            "open_only": BOOL,
            "k": {**INT, "description": "How many trials to return (1-25)."},
            "rerank": {**BOOL, "description": "Score the returned cohorts with the decision model's match "
                                              "questions and order by that score."},
        },
    },
    "trial_details": {
        "description": "A trial's or cohort's details: status, phase, center, drugs and each cohort's key "
                       "inclusion and exclusion lines.",
        "properties": {
            "ids": {"type": "array", "items": TEXT, "description": "Trial IDs or cohort IDs (1-5)."},
            "show_common_criteria": {**BOOL, "description": "Also list the boilerplate (common) lines."},
        },
    },
    "search_patients": {
        "description": "Text search over the patients' cancer-history summaries and other relevant history.",
        "properties": {
            "text": _string("Search terms: case-insensitive regular expressions, all must match (AND); '' for "
                            "none."),
            "cancer_type": _cancer_type(),
            "sex": _string("'' for any.", SEXES),
            "with_records": {**BOOL, "description": "Only patients the structured record store holds."},
        },
    },
    "match_patients": {
        "description": "TrialSpace semantic retrieval: the patients nearest a cohort (cohort_id) or an ad-hoc cohort "
                       "written as eligibility criteria, optionally scored by the decision model.",
        "properties": {
            "cohort_id": _string("A cohort ID, or '' to use criteria."),
            "criteria": _string("With no cohort_id: the cohort's key eligibility criteria as protocol-style "
                                "inclusion and exclusion lines; else ''."),
            "cancer_type": _cancer_type(),
            "sex": _string("'' for any.", SEXES),
            "with_records": {**BOOL, "description": "Only patients the structured record store holds."},
            "k": {**INT, "description": "How many patients to return (1-25)."},
            "rerank": {**BOOL, "description": "Score them with the decision model's match questions."},
        },
    },
    "patient_details": {
        "description": "One patient's summary sections and, when the store holds the patient, the structured "
                       "records' disease course.",
        "properties": {
            "patient_id": TEXT,
            "include_records": {**BOOL, "description": "Include the structured records' disease course."},
        },
    },
    "screen_records": {
        "description": "Check criteria lines on the structured extraction records of the store's patients (or the "
                       "given ones): per-line counts and the patients documented to meet them.",
        "properties": {
            "cohort_id": _string("Screen this cohort's own criteria lines; '' to screen the lines below."),
            "lines": {"type": "array", "description": "With no cohort_id: 1-10 criteria lines; else [].",
                      "items": {"type": "object", "additionalProperties": False, "required": ["text", "type"],
                                "properties": {"text": TEXT, "type": _string("The line's type.", LINE_TYPES)}}},
            "patient_ids": {"type": "array", "items": TEXT,
                            "description": "Patients to screen (for example a search's results); [] for every "
                                           "patient the store holds."},
        },
    },
    "score_pairs": {
        "description": "Decision-model scores of patient-cohort pairs (match questions and the boilerplate "
                       "question).",
        "properties": {
            "pairs": {"type": "array", "description": "1-20 pairs.",
                      "items": {"type": "object", "additionalProperties": False,
                                "required": ["patient_id", "cohort_id"],
                                "properties": {"patient_id": TEXT, "cohort_id": TEXT}}},
        },
    },
    "guideline_spaces": {
        "description": "The guideline catalog's spaces (NCCN-derived treatment and workup menus) nearest a patient "
                       "description.",
        "properties": {
            "description": _string("A patient cancer-history summary in the corpus style."),
            "top_n": {**INT, "description": "Spaces to return (1-5)."},
        },
    },
}

CALLS: dict[str, dict] = {
    **TOOLS,
    "ask_user": {
        "description": "Ask the user one clarifying question; ends this turn.",
        "properties": {
            "question": _string("One question about one concept."),
            "options": {"type": "array", "items": TEXT,
                        "description": "2-5 concrete readings the user can pick (the user may also type one)."},
            "why": _string("One short sentence: what changes in the result depending on the answer."),
        },
    },
    "answer": {
        "description": "The final answer for this turn.",
        "properties": {
            "markdown": _string("The answer in Markdown, citing the tools' IDs exactly."),
            "interpretation": _string("One sentence stating how ambiguous concepts were read (\"Interpreted "
                                      "metastatic as ...\"), or '' when none needed reading."),
        },
    },
}


def _arguments(name: str) -> dict:
    properties = CALLS[name]["properties"]
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _variant(name: str) -> dict:
    return {"type": "object", "description": CALLS[name]["description"], "properties": {name: _arguments(name)},
            "required": [name], "additionalProperties": False}


def response_schema(final_only: bool = False, tools=None) -> dict:
    """The step's JSON schema: every call, or (the tool budget spent) only ``ask_user`` and ``answer``."""
    names = list(FINAL) if final_only else [*(tools or TOOLS), *FINAL]
    return {
        "type": "object",
        "properties": {
            "purpose": _string("What this step does, in one short sentence the user sees."),
            "call": {"anyOf": [_variant(name) for name in names]},
        },
        "required": ["purpose", "call"],
        "additionalProperties": False,
    }


def _check(value: dict, schema: dict, where: str) -> None:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict) or set(value) != set(schema["properties"]):
            raise ValueError(f"{where} must have exactly the fields of its schema.")
        for key, sub in schema["properties"].items():
            _check(value[key], sub, f"{where}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{where} must be a list.")
        for item in value:
            _check(item, schema["items"], where)
    elif kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"{where} must be a string.")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{where} must be one of the allowed values.")
    elif kind == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{where} must be true or false.")
    elif kind == "integer":
        if type(value) is not int:
            raise ValueError(f"{where} must be an integer.")


def validate(value, limits: Limits, *, final_only: bool = False, tools=None) -> dict:
    """The step's action, checked against the schema and the limits (fixed messages, never input text)."""
    if not isinstance(value, dict) or set(value) != {"purpose", "call"}:
        raise ValueError("Return exactly the fields purpose and call.")
    if not isinstance(value["purpose"], str) or len(value["purpose"]) > limits.max_purpose_chars:
        raise ValueError("purpose must be one short sentence.")
    allowed = list(FINAL) if final_only else [*(tools or TOOLS), *FINAL]
    if not isinstance(value["call"], dict) or len(value["call"]) != 1 or next(iter(value["call"])) not in allowed:
        raise ValueError("call must have one key, an allowed tool" + (" (ask_user or answer now)."
                                                                      if final_only else "."))
    name = next(iter(value["call"]))
    _check(value["call"][name], _arguments(name), f"call.{name}")
    call = parse_call(value)
    if name == "ask_user":
        if not call["question"].strip() or len(call["question"]) > limits.max_question_chars:
            raise ValueError("ask_user needs one short question.")
        options = [o.strip() for o in call["options"]]
        if not 2 <= len(options) <= 5 or any(not o or len(o) > limits.max_option_chars for o in options) \
                or len(set(options)) != len(options):
            raise ValueError("ask_user needs 2-5 distinct, short options.")
    elif name == "answer":
        if not call["markdown"].strip() or len(call["markdown"]) > limits.max_answer_chars:
            raise ValueError("answer needs Markdown text within the length limit.")
        if len(call["interpretation"]) > limits.max_question_chars:
            raise ValueError("interpretation must be one sentence.")
    else:
        for key, sub in CALLS[name]["properties"].items():
            if sub.get("type") == "string" and len(call[key]) > limits.max_text_arg_chars:
                raise ValueError(f"call.{key} is too long.")
    return value


def parse_call(value: dict) -> dict:
    """``{"tool": name, **arguments}`` from a step's ``call``."""
    ((name, arguments),) = value["call"].items()
    return {"tool": name, **arguments}


def compact(value: dict) -> str:
    """The action as the conversation keeps it (the assistant's turn)."""
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
