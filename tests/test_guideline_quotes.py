import copy
from dataclasses import replace

import pytest

from matchminer_ai.trials._guideline_quotes import (
    EXCERPT_RULES, VERSION, materialize_quoted_state, resolve_excerpt,
)
from matchminer_ai.trials._guideline_sources import Page
from test_guideline_extraction import quoted_state


@pytest.fixture
def page():
    return Page("p0002", 2, "SYN-1", "synthetic.md", (
        "Fictional metastatic disease.\n\n"
        "Obtain synthetic blood test A.                 An unrelated column about imaging.\n"
        "Obtain synthetic chemistry panel B.            Another unrelated column.\n"
    ), "synthetic", ())


def test_quote_first_addresses_come_from_source_and_exclude_adjacent_columns(page):
    value = quoted_state()
    value["evidence"] = [{"page_id": page.id, "source_text": "Fictional metastatic disease."}]
    value["diagnostic_workup"][0]["evidence"] = [
        {"page_id": page.id, "source_text": "Obtain synthetic blood test A."}]
    before = copy.deepcopy(value)
    result = materialize_quoted_state(value, {page.id: page})
    evidence = result["diagnostic_workup"][0]["evidence"][0]
    assert evidence["line_ids"] == [3]
    assert evidence["quote"] == "Obtain synthetic blood test A."
    assert "unrelated column" in evidence["source_lines"][0]["text"]
    assert evidence["pdf_page"] == 2
    assert value == before


def test_wrong_numeric_reference_is_not_an_accepted_final_citation(page):
    value = quoted_state()
    value["evidence"] = [{"page_id": page.id, "line_ids": [4]}]
    with pytest.raises(ValueError, match="source_text"):
        materialize_quoted_state(value, {page.id: page})


@pytest.mark.parametrize("text", [
    "Obtain a different synthetic blood test.",
    "Obtain synthetic blood test A. Obtain synthetic chemistry panel B.",
    "Obtain synthetic ... A.",
])
def test_invented_paraphrased_or_cross_column_joined_quotes_fail(page, text):
    with pytest.raises(ValueError, match="does not occur verbatim"):
        resolve_excerpt(page, text)


def test_ambiguous_excerpt_requires_more_context(page):
    with pytest.raises(ValueError, match="occurs 2 times"):
        resolve_excerpt(page, "Obtain synthetic")


def test_only_whitespace_normalization_is_permitted_and_original_bytes_are_returned(page):
    altered = replace(page, text="Use synthetic\t\tpanel B.\nFor the stated population.")
    ids, quote = resolve_excerpt(altered, "Use synthetic panel B.\nFor the stated population.")
    assert ids == [1, 2]
    assert quote == altered.text
    with pytest.raises(ValueError, match="does not occur verbatim"):
        resolve_excerpt(altered, "Use Synthetic panel B.")


def test_markup_matching_is_gated_and_returns_the_source_markup(page):
    row = '<tr><td rowspan="3">I</td><td>T1, N0, M0</td><td>T1 = Synthetic tumor size.</td></tr>'
    table = replace(page, text=f"Table 1.\n{row}\n<tr><td>II</td><td>T2 = Synthetic tumor size.</td></tr>\n")
    for excerpt in ("I T1, N0, M0 T1 = Synthetic tumor size.",
                    "<tr><td rowspan='3'>I</td><td>T1, N0, M0</td><td>T1 = Synthetic tumor size.</td></tr>"):
        with pytest.raises(ValueError, match="does not occur verbatim"):
            resolve_excerpt(table, excerpt)  # v1 catalogs re-audit with whitespace-only matching
        ids, quote = resolve_excerpt(table, excerpt, rule="markup")
        assert ids == [2] and quote in row and quote.startswith("I</td>")
    with pytest.raises(ValueError, match="occurs 2 times"):
        resolve_excerpt(table, "Synthetic tumor size.", rule="markup")
    with pytest.raises(ValueError, match="does not occur verbatim"):
        resolve_excerpt(table, "I T1, N0, M0 T1 = Synthetic tumour size.", rule="markup")


def test_entities_rule_equates_character_references_and_keeps_raw_offsets(page):
    from matchminer_ai.trials._guideline_quote_repair import targets

    row = "<tr><td>T2 = Synthetic tumor &gt;7 cm and &#8804;10&nbsp;cm.</td></tr>"
    table = replace(page, text=f"Table 2.\nIntro line.\n{row}\n")
    ids, quote = resolve_excerpt(table, "T2 = Synthetic tumor >7 cm and \u226410 cm.", rule="entities")
    assert ids == [3] and quote == "T2 = Synthetic tumor &gt;7 cm and &#8804;10&nbsp;cm."
    with pytest.raises(ValueError, match="does not occur verbatim"):
        resolve_excerpt(table, "T2 = Synthetic tumor >7 cm", rule="markup")
    # A catalog replays each repair under the rule its receipt records, so a v2 repair still saw this as invalid.
    record = quoted_state()
    record["evidence"] = [{"page_id": table.id, "source_text": "Synthetic tumor >7 cm"}]
    assert (0, 0) in [t[:2] for t in targets(record, {table.id: table}, rule="markup")]
    assert (0, 0) not in [t[:2] for t in targets(record, {table.id: table}, rule="entities")]
    assert EXCERPT_RULES == {"source-excerpts-v1": "exact", "source-excerpts-v2": "markup", VERSION: "entities"}


def test_unsupplied_page_and_excessive_spans_are_rejected(page):
    with pytest.raises(ValueError, match="Unknown or unsupplied"):
        materialize_quoted_state(quoted_state(), {})
    long_page = replace(page, text="\n".join(f"Synthetic line {i}." for i in range(13)))
    with pytest.raises(ValueError, match="more than 12 lines"):
        resolve_excerpt(long_page, long_page.text)
