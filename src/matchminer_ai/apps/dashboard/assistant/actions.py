"""The assistant's action schema: one JSON object per step, ``{"purpose", "call"}``, validated in code.

``call`` has one key, the name of a tool (``TOOLS``: read-only searches over the dashboard's own data), of
``ask_user`` (a clarifying question with options; ends the turn) or of ``answer`` (the final Markdown, the
interpretation used and any charts; ends the turn), and its value holds the arguments:
``{"search_trials": {"text": "EGFR", ...}}``.
The name is the key, not a ``tool`` field beside the arguments, because constrained decoding then fixes the
arguments' schema by the name (with a ``tool`` field among ``anyOf`` variants, Claude's structured outputs returned
another tool's empty arguments). Every argument is required (strict JSON-schema outputs), with ``""``, ``[]``,
``false`` or ``0`` for an unused one. The endpoint's JSON-schema response format constrains the shape;
:func:`validate` checks it again with fixed messages (never input text), and each tool checks its arguments' meaning
when it runs. :func:`parse_call` gives the code's form, ``{"tool": name, **arguments}``.

**Charts** (``answer.charts``, :data:`CHART`): data, never diagram code. Each is a kind (pie, donut, bar or line), a
title and caption, its labels and one or more series of numbers, a unit, the denominator, the bar layout and the steps
its numbers came from; :func:`check_charts` checks the shapes, caps and signs the page draws from, and that the steps
are this conversation's successful tool results. The page draws them as SVG.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import re

from ..core import cancer_types

FINAL = ("ask_user", "answer")
SEXES = ("", "male", "female")
PHASES = ("1", "2", "3", "4")
LINE_TYPES = ("inclusion", "exclusion")
CHART_KINDS = ("pie", "donut", "bar", "line")
ORIENTATIONS = ("", "vertical", "horizontal")
#: A fenced block of diagram code in the Markdown (the page would show it as text): charts go in ``charts``.
DIAGRAM_FENCE = re.compile(r"^\s*(```|~~~)\s*(mermaid|vega(-lite)?|plantuml|graphviz|dot|chart|flowchart)\b",
                           re.IGNORECASE | re.MULTILINE)


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
    #: Charts per answer; categories (bars, line points) and series per chart; slices of a pie or donut.
    max_charts: int = 3
    max_chart_categories: int = 24
    max_chart_series: int = 4
    max_pie_slices: int = 6
    #: Characters of a chart's title, caption, and of a label, series name or unit.
    max_chart_title_chars: int = 120
    max_chart_caption_chars: int = 400
    max_chart_label_chars: int = 80

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"assistant limit {item.name} must be a positive integer")


TEXT = {"type": "string"}
BOOL = {"type": "boolean"}
INT = {"type": "integer"}
NUMBER = {"type": "number"}


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

#: One chart of an answer (every field required, as for the tools' arguments).
CHART: dict = {
    "type": "object", "additionalProperties": False,
    "required": ["kind", "title", "caption", "labels", "series", "unit", "total", "orientation", "stacked", "steps"],
    "properties": {
        "kind": _string("pie or donut: the parts of one whole (2-6 parts, one series, values summing to total); bar: "
                        "counts compared across categories; line: values along an ordered x axis (such as years).",
                        CHART_KINDS),
        "title": _string("A short title: what is counted, in which population."),
        "caption": _string("One or two short sentences: the denominator (n), where the numbers came from, and what "
                           "'not documented' means when it appears."),
        "labels": {"type": "array", "items": TEXT,
                   "description": "The categories (slices or bars), or the x-axis values in order (line)."},
        "series": {"type": "array", "description": "1-4 series (exactly one for pie and donut), each with one value "
                                                   "per label.",
                   "items": {"type": "object", "additionalProperties": False, "required": ["name", "values"],
                             "properties": {
                                 "name": _string("The series' name for the legend; '' for a chart's only series."),
                                 "values": {"type": "array", "items": NUMBER,
                                            "description": "One number per label, in the labels' order, each from a "
                                                           "tool result."}}}},
        "unit": _string("What the values count, such as 'patients', 'trials' or '%'; '' for none."),
        "total": {**NUMBER, "description": "The denominator n percentages are out of: a pie's or donut's sum of "
                                           "values; for bars, the population every bar is a part of, or 0 when the "
                                           "bars are not parts of one population (then no percentages); 0 for line."},
        "orientation": _string("Bar charts: 'vertical' (columns) or 'horizontal' (bars, for many or long labels); "
                               "'' for other kinds.", ORIENTATIONS),
        "stacked": {**BOOL, "description": "Bar charts with 2+ series: stack them in each bar (the parts of each "
                                           "bar's whole) instead of grouping them side by side; else false."},
        "steps": {"type": "array", "items": INT,
                  "description": "The step numbers of the tool results the numbers came from (each result names its "
                                 "step); [] only when every number is from the data overview."},
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
            "charts": {"type": "array", "items": CHART,
                       "description": "0-3 charts the page draws below the Markdown, from tool results' numbers "
                                      "only; [] for none."},
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
    elif kind == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{where} must be a finite number.")


def _text(value: str, limit: int, where: str, *, required: bool = True) -> None:
    if (required and not value.strip()) or len(value) > limit:
        raise ValueError(f"{where} needs {'a ' if required else ''}short text within its length limit.")


def check_charts(charts: list[dict], limits: Limits, steps=None) -> None:
    """An answer's charts (already of the schema's shape), checked against the caps and the kinds' rules: lengths,
    signs, a pie's sum, and the steps (``steps``: this conversation's successful tool results' step numbers; None
    skips that check). Fixed messages, never input text."""
    if len(charts) > limits.max_charts:
        raise ValueError(f"answer.charts holds at most {limits.max_charts} charts.")
    for n, chart in enumerate(charts):
        where, kind = f"answer.charts[{n}]", chart["kind"]
        _text(chart["title"], limits.max_chart_title_chars, f"{where}.title")
        _text(chart["caption"], limits.max_chart_caption_chars, f"{where}.caption")
        _text(chart["unit"], limits.max_chart_label_chars, f"{where}.unit", required=False)
        labels, series = chart["labels"], chart["series"]
        pie = kind in ("pie", "donut")
        low, high = ((2, limits.max_pie_slices) if pie else (2, limits.max_chart_categories) if kind == "line"
                     else (1, limits.max_chart_categories))
        if not low <= len(labels) <= high:
            raise ValueError(f"{where}: a {kind} chart takes {low}-{high} labels (use bars for more categories).")
        for label in labels:
            _text(label, limits.max_chart_label_chars, f"{where}.labels")
        if len({label.strip().casefold() for label in labels}) != len(labels):
            raise ValueError(f"{where}.labels must be distinct.")
        if not 1 <= len(series) <= (1 if pie else limits.max_chart_series):
            raise ValueError(f"{where}: a {kind} chart takes "
                             + ("exactly one series." if pie else f"1-{limits.max_chart_series} series."))
        names = [s["name"].strip() for s in series]
        for name in names:
            _text(name, limits.max_chart_label_chars, f"{where}.series.name", required=len(series) > 1)
        if len(set(n.casefold() for n in names)) != len(names):
            raise ValueError(f"{where}: series names must be distinct.")
        if any(len(s["values"]) != len(labels) for s in series):
            raise ValueError(f"{where}: every series needs one value per label.")
        values = [v for s in series for v in s["values"]]
        total = chart["total"]
        if total < 0:
            raise ValueError(f"{where}.total must be 0 or more.")
        stacked = kind == "bar" and chart["stacked"] and len(series) > 1
        if (pie or stacked or (kind == "bar" and total > 0)) and any(v < 0 for v in values):
            raise ValueError(f"{where}: a {'stacked ' if stacked else ''}{kind} chart's values must be 0 or more.")
        if pie:
            whole = sum(values)
            if whole <= 0:
                raise ValueError(f"{where}: a {kind} chart needs a positive sum.")
            if abs(whole - total) > 0.005 * whole:
                raise ValueError(f"{where}: a {kind} chart's total must be the sum of its values (add a part such "
                                 "as 'Not documented' for the rest of the population).")
        elif kind == "bar" and total > 0:
            heights = ([sum(col) for col in zip(*(s["values"] for s in series))] if stacked else values)
            if max(heights) > total * 1.005:
                raise ValueError(f"{where}: a bar is larger than total, the population the percentages are out "
                                 "of; give the right total, or 0 for no percentages.")
        cited = chart["steps"]
        if len(cited) > limits.max_tool_calls * 2 or len(set(cited)) != len(cited) or any(s < 1 for s in cited):
            raise ValueError(f"{where}.steps must be distinct step numbers.")
        if steps is not None and any(s not in steps for s in cited):
            raise ValueError(f"{where}.steps must name steps of this conversation whose tool results succeeded.")


def chart_view(chart: dict) -> dict:
    """A checked chart as the page draws it: trimmed text, and the fields its kind does not use set to their
    defaults (bars default to vertical; a line has no denominator)."""
    kind = chart["kind"]
    bar = kind == "bar"
    return {"kind": kind, "title": chart["title"].strip(), "caption": chart["caption"].strip(),
            "labels": [label.strip() for label in chart["labels"]],
            "series": [{"name": s["name"].strip(), "values": list(s["values"])} for s in chart["series"]],
            "unit": chart["unit"].strip(), "total": 0 if kind == "line" else chart["total"],
            "orientation": (chart["orientation"] or "vertical") if bar else "",
            "stacked": bool(bar and chart["stacked"] and len(chart["series"]) > 1), "steps": list(chart["steps"])}


def validate(value, limits: Limits, *, final_only: bool = False, tools=None, steps=None) -> dict:
    """The step's action, checked against the schema and the limits (fixed messages, never input text). ``steps``:
    the step numbers an answer's charts may cite (None skips that check)."""
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
        if DIAGRAM_FENCE.search(call["markdown"]):
            raise ValueError("answer.markdown must not hold diagram code (such as Mermaid); put a chart in "
                             "answer.charts.")
        check_charts(call["charts"], limits, steps)
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
