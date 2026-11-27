"""Resolve final citations from exact source excerpts instead of model-selected offsets."""

import copy
import functools
import html
import re

from ._guideline_schema import (
    STATE,
    STRING,
    arr,
    materialize_evidence,
    obj,
    validate_shape,
    validate_state,
)

VERSION = "source-excerpts-v3"
# The excerpt-matching rule of each recorded detail_citations version (resolve_excerpt): v1 matches whitespace only;
# v2 also matches an excerpt's visible words across the page's markup; v3 also equates the page's HTML character
# references with the characters they stand for. A catalog re-audits under the rule its run config records, so a
# catalog built under v1 or v2 keeps its exact audit.
EXCERPT_RULES = {"source-excerpts-v1": "exact", "source-excerpts-v2": "markup", VERSION: "entities"}
RULE = EXCERPT_RULES[VERSION]
QUOTED_EVIDENCE = arr(obj({"page_id": STRING, "source_text": STRING}))
QUOTED_DETAIL = copy.deepcopy(STATE)
QUOTED_DETAIL["properties"]["evidence"] = QUOTED_EVIDENCE
for _kind in ("diagnostic_workup", "treatment_options"):
    QUOTED_DETAIL["properties"][_kind]["items"]["properties"]["evidence"] = (
        QUOTED_EVIDENCE
    )


# An HTML tag in a source page (PDQ tables, links and inline markup are kept as HTML).
MARKUP = r"<[A-Za-z/!][^<>]*>"


# An HTML character reference in a source page: named (&gt;), decimal (&#8805;) or hexadecimal (&#x2265;).
ENTITY = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")


@functools.lru_cache(maxsize=256)
def _visible(text):
    """``text`` with its character references decoded, and the raw span each decoded character came from."""
    chars, spans, position = [], [], 0
    for reference in ENTITY.finditer(text):
        value = html.unescape(reference.group())
        if value == reference.group():
            continue  # not a reference html.unescape knows; kept as written
        chars.append(text[position:reference.start()])
        spans.extend((i, i + 1) for i in range(position, reference.start()))
        chars.append(value)
        spans.extend([(reference.start(), reference.end())] * len(value))
        position = reference.end()
    chars.append(text[position:])
    spans.extend((i, i + 1) for i in range(position, len(text)))
    return "".join(chars), tuple(spans)


def _words_pattern(words, separator):
    return re.compile(separator.join(re.escape(word) for word in words))


def resolve_excerpt(page, excerpt, *, rule="exact"):
    """Match whitespace only; never fuzzy-match words or guess an ambiguous location.

    ``rule`` widens what may lie between and within the excerpt's words, one level at a time, and only when the
    previous level finds nothing: ``"markup"`` (v2) matches an excerpt quoting a table row or linked text by its visible
    words across the page's HTML tags (tags in the excerpt are skipped, so their attribute quoting need not match);
    ``"entities"`` (v3) also equates the page's character references (``&gt;``, ``&#8805;``) with the characters the
    excerpt writes. The returned line addresses and quote always come from the page's own raw text.
    """
    if rule not in {"exact", "markup", "entities"}:
        raise ValueError(f"Unknown excerpt-matching rule: {rule!r}")
    if not isinstance(excerpt, str) or not excerpt.strip():
        raise ValueError("Citation source_text must be a nonempty verbatim excerpt")
    text = page.text
    matches = [m.span() for m in _words_pattern(excerpt.split(), r"\s+").finditer(text)]
    separator = rf"(?:\s|{MARKUP})+"
    words = re.sub(MARKUP, " ", excerpt).split()
    if not matches and rule != "exact" and words:
        matches = [m.span() for m in _words_pattern(words, separator).finditer(text)]
    if not matches and rule == "entities" and ENTITY.search(text + excerpt):
        visible, spans = _visible(text)
        decoded = re.sub(MARKUP, " ", html.unescape(excerpt)).split()
        if decoded:
            matches = [(spans[m.start()][0], spans[m.end() - 1][1])
                       for m in _words_pattern(decoded, separator).finditer(visible) if m.end() > m.start()]
    if not matches:
        raise ValueError(
            f"{page.id}: source_text does not occur verbatim on this supplied page: "
            f"{excerpt[:240]!r}. Copy exact source words; do not paraphrase, add ellipses, "
            "or join text across the other PDF column. Use separate excerpts for wrapped lines."
        )
    if len(matches) != 1:
        raise ValueError(
            f"{page.id}: source_text occurs {len(matches)} times; include more surrounding "
            "source words to identify the intended branch unambiguously."
        )
    start, end = matches[0]
    first = text.count("\n", 0, start) + 1
    last = text.count("\n", 0, end - 1) + 1
    if last - first >= 12:
        raise ValueError(f"{page.id}: split excerpts spanning more than 12 lines")
    return list(range(first, last + 1)), text[start:end]


def _owners(state):
    yield state
    for kind in ("diagnostic_workup", "treatment_options"):
        yield from state[kind]


def materialize_quoted_state(state, pages, *, rule=RULE):
    """Derive addresses and displayed quotations in code, preserving all clinical fields."""
    validate_shape(state, QUOTED_DETAIL)
    result = copy.deepcopy(state)
    quotes = []
    errors = []
    for owner in _owners(result):
        refs, owner_quotes = [], []
        for number, item in enumerate(owner["evidence"], 1):
            try:
                if item["page_id"] not in pages:
                    raise ValueError(f"Unknown or unsupplied evidence page: {item['page_id']}")
                ids, quote = resolve_excerpt(pages[item["page_id"]], item["source_text"], rule=rule)
            except ValueError as exc:
                errors.append(f"{owner['name']} excerpt {number}: {exc}")
                continue
            refs.append({"page_id": item["page_id"], "line_ids": ids})
            owner_quotes.append(quote)
        owner["evidence"] = refs
        quotes.append(owner_quotes)
    if errors:
        raise ValueError("Correct every invalid excerpt in the draft:\n" + "\n".join(errors))
    validate_state(result, pages)
    result = materialize_evidence(result, pages)
    for owner, owner_quotes in zip(_owners(result), quotes):
        for evidence, quote in zip(owner["evidence"], owner_quotes):
            # Full original lines remain in source_lines for provenance. Display
            # the verified excerpt so neighboring PDF columns are not interleaved.
            evidence["quote"] = quote
    return result
