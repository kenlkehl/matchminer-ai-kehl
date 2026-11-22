"""The How this works tab: a curated, read-only set of documentation pages.

The pages are listed in ``docs_pages.json`` (an id, a title, and a source under the repository's ``docs/``). From a
source checkout they are read from ``docs/`` itself, so the tab shows what the documentation site shows; an
installed package reads the copies the frontend build writes to ``static/docs/<id>.md``. The extraction data model
page has no source: it is generated from the schema declarations (``matchminer_ai.patients.extraction.schema`` and
its vocabularies, the declarations the extraction prompts, the JSON schemas constraining the model's answers, the
validator and the SQL agent's tables are all built from), so it cannot drift from the code. Only listed ids are
served, and no other file is read.

Before a page is served, what only the documentation site renders is removed (API reference blocks, ``::: module``
with their options, and Mermaid diagrams), links to other listed pages become ``page:<id>`` links, and links to
unlisted pages keep only their text.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import posixpath
import re

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "docs_pages.json"
STATIC_DOCS = HERE / "static" / "docs"
_LINK = re.compile(r"(!?)\[([^\]]+)\]\(([^)\s]+)(\s+\"[^\"]*\")?\)")


class UnknownPage(KeyError):
    pass


@lru_cache(maxsize=1)
def pages() -> tuple[dict, ...]:
    return tuple(json.loads(MANIFEST.read_text(encoding="utf-8"))["pages"])


def repository_docs() -> Path | None:
    """The source checkout's ``docs/`` (None in an installed package)."""
    root = HERE.parents[3]
    return root / "docs" if (root / "mkdocs.yml").is_file() and (root / "docs").is_dir() else None


def index() -> list[dict]:
    return [{"id": p["id"], "title": p["title"], "generated": p["source"] is None} for p in pages()]


def _source_text(page: dict) -> str | None:
    docs = repository_docs()
    candidates = ([docs / page["source"]] if docs is not None else []) + [STATIC_DOCS / f"{page['id']}.md"]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def prepare(text: str, source: str) -> str:
    """A documentation page as the tab shows it (see the module docstring)."""
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(":::"):  # an API reference block and its indented options
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith((" ", "\t"))):
                i += 1
            continue
        if line.strip().startswith("```mermaid"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
            continue
        out.append(line)
        i += 1
    by_source = {p["source"]: p["id"] for p in pages() if p["source"]}
    folder = posixpath.dirname(source)

    def link(match: re.Match) -> str:
        image, label, target = match.group(1), match.group(2), match.group(3)
        if image:
            return label
        if re.match(r"https?://", target):
            return match.group(0)
        path = target.split("#", 1)[0]
        if not path:
            return label
        page = by_source.get(posixpath.normpath(posixpath.join(folder, path)))
        return f"[{label}](page:{page})" if page else label

    return re.sub(r"\n{3,}", "\n\n", _LINK.sub(link, "\n".join(out))).strip() + "\n"


def page(page_id: str) -> dict:
    found = next((p for p in pages() if p["id"] == page_id), None)
    if found is None:
        raise UnknownPage(page_id)
    if found["source"] is None:
        markdown = extraction_data_model()
    else:
        text = _source_text(found)
        if text is None:
            raise UnknownPage(page_id)
        markdown = prepare(text, found["source"])
    return {"id": found["id"], "title": found["title"], "markdown": markdown}


# ---------------------------------------------------------------------------
# The extraction data model, from the schema declarations
# ---------------------------------------------------------------------------

KINDS = {"text": "text", "text_list": "list of text", "choice": "one code", "choice_list": "list of codes",
         "date": "date", "integer": "integer"}


def _cell(text) -> str:
    return " ".join(str(text or "").split()).replace("|", "\\|")


def extraction_data_model() -> str:
    from matchminer_ai.patients.extraction import vocabulary as v
    from matchminer_ai.patients.extraction.extensions import registered_extensions
    from matchminer_ai.patients.extraction.schema import (
        CORE_ITEM_TYPES, ITEM_TYPES_SINCE, LEVELS, MAX_EVIDENCE_ITEMS, MAX_QUOTE_CHARACTERS, SCHEMA_VERSION,
        SCHEMA_VERSIONS, _DATE_RECORD, item_types_for, record_json_schema,
    )

    vocabularies: dict[str, object] = {}

    def vocabulary_of(field) -> str:
        if field.vocabulary is None:
            return ""
        vocabularies.setdefault(field.vocabulary.name, field.vocabulary)
        return f"`{field.vocabulary.name}` (Vocabularies)"

    lines = [
        "# Structured extraction: data model", "",
        f"Schema `{SCHEMA_VERSION}`. This page is generated from the extraction schema's declarations "
        "(`matchminer_ai.patients.extraction.schema` and its vocabularies). The extraction prompts' field guide, the "
        "JSON schemas that constrain the model's answers, the validator of those answers and the SQL agent's tables "
        "(one per item type) are built from the same declarations. "
        "See [Structured extraction](page:structured-extraction) for how records are made.", "",
        "## Records", "",
        "| Level | One record per | Record fields | Item lists |", "| --- | --- | --- | --- |",
    ]
    per = {"document": "note", "patient": "patient"}
    for level in LEVELS:
        fields = ", ".join(f"`{k}`" for k in record_json_schema(level)["properties"])
        items = ", ".join(f"`{t.key}`" for t in item_types_for(level))
        lines.append(f"| {level} | {per[level]} | {fields} | {items} |")
    sources = _DATE_RECORD["properties"]["source"]["enum"]
    lines += [
        "", "## Every item", "",
        "- `item_type`; at the patient level also `item_id`, the type's prefix and a number (`D1`, `B2`, ...).",
        "- Dates are objects: `value` (`YYYY`, `YYYY-MM` or `YYYY-MM-DD`), `precision` ("
        + ", ".join(f"`{p}`" for p in v.DATE_PRECISIONS) + ") and `source` (" + ", ".join(f"`{s}`" for s in sources)
        + ").",
        f"- `evidence`: 1 to {MAX_EVIDENCE_ITEMS} exact quotes from the note (at most {MAX_QUOTE_CHARACTERS} "
        "characters each), each with the note's `note_id`, `note_date`, the `quote` and its `start` and `end` "
        "character offsets.",
        "- Occurrence items (studies, procedures, tests, counseling) say how they occurred: at the document level "
        "`relation` (" + ", ".join(f"`{c}`" for c in v.DOCUMENT_RELATIONS.codes) + ") and `source_of_document`; at "
        "both levels `status` (" + ", ".join(f"`{c}`" for c in v.OCCURRENCE_STATUSES.codes) + ").",
        "- Linked items carry `cancer_ids`: the patient-level diagnosis ids (`D1`, ...) they concern (`[]` when "
        "unclear, or when the document stage ran without patient records).",
        "- Text a note does not state is `\"\"`, a choice it does not settle is `not_stated` where the vocabulary "
        "has it, and an unstated list is `[]`.", "",
        "## Item types", "",
    ]
    for item_type in CORE_ITEM_TYPES:
        facts = [f"Levels: {', '.join(sorted(item_type.levels))}",
                 f"patient-level ids `{item_type.id_prefix}1`, `{item_type.id_prefix}2`, ..."
                 if "patient" in item_type.levels else "",
                 f"dated by `{item_type.primary_date}`" if item_type.primary_date else "dated by its note",
                 "an occurrence item (relation and status)" if item_type.occurrence else "",
                 f"linked to cancers (`cancer_ids`) at the {' and '.join(sorted(item_type.linkable_levels))} level"
                 if item_type.linkable_levels else "",
                 f"since `{ITEM_TYPES_SINCE[item_type.name]}`" if item_type.name in ITEM_TYPES_SINCE else ""]
        lines += [f"### `{item_type.key}` (`{item_type.name}`)", "", _cell(item_type.description), "",
                  "; ".join(f for f in facts if f) + ".", "",
                  "| Field | Kind | Meaning | Codes |", "| --- | --- | --- | --- |"]
        for field in item_type.fields:
            kind = KINDS[field.kind]
            if field.minimum is not None or field.maximum is not None:
                kind += f" ({field.minimum if field.minimum is not None else ''}-"
                kind += f"{field.maximum if field.maximum is not None else ''})"
            meaning = _cell(field.description)
            if field.other_text:
                meaning += f" With `other`, `{field.other_text}` names the value."
            lines.append(f"| `{field.name}` | {kind} | {meaning} | {vocabulary_of(field)} |")
        lines.append("")
    extensions = registered_extensions()
    if extensions:
        lines += ["## Extensions", "",
                  "Disease-specific item types a record includes when its extraction selected the extension "
                  "(the record's `extensions`).", ""]
        for name, extension in sorted(extensions.items()):
            for item_type in extension.item_types:
                lines += [f"### `{item_type.key}` (`{name}` extension)", "", _cell(item_type.description), "",
                          "| Field | Kind | Meaning | Codes |", "| --- | --- | --- | --- |"]
                for field in item_type.fields:
                    lines.append(f"| `{field.name}` | {KINDS[field.kind]} | {_cell(field.description)} | "
                                 f"{vocabulary_of(field)} |")
                lines.append("")
    vocabularies.setdefault(v.DOCUMENT_TYPES.name, v.DOCUMENT_TYPES)
    lines += ["## Vocabularies", "",
              "Closed code lists, rendered into the prompts and enforced by the JSON schemas. `other` needs its "
              "companion text field; `not_stated` is a document's silence.", ""]
    for name in sorted(vocabularies):
        vocabulary = vocabularies[name]
        lines += [f"### {name}", "", _cell(vocabulary.description), "", "| Code | Meaning |", "| --- | --- |"]
        lines += [f"| `{t.code}` | {_cell(t.description)}{f' (group `{t.group}`)' if t.group else ''} |"
                  for t in vocabulary.terms]
        lines.append("")
    lines += ["## Schema versions", "",
              "Versions in order: " + ", ".join(f"`{s}`" for s in SCHEMA_VERSIONS) + ". A record of an earlier "
              "version lacks the item types added later ("
              + ", ".join(f"`{k}` since `{s}`" for k, s in ITEM_TYPES_SINCE.items())
              + "); checks treat those types as not extracted for it, never as not done.", ""]
    return "\n".join(lines)
