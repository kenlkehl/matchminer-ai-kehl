# Structured-Record Criteria Screening

!!! warning "Review aids, not eligibility determinations"
    Results say whether structured items extracted from the notes show a trial criteria line satisfied or clearly
    failed. An LLM wrote each line's query and an LLM extracted the items: check every cited quote against its note
    and the current protocol. *Not documented* means no extracted item showed either outcome, not that the patient
    is eligible.

Criteria-level screening from extraction records is the trial-side analogue of the
[extraction-based workup checks](workup-queries.md):

1. **Compile once per line.** Each criteria line of a cohort (a `cohort_criteria_items` row typed inclusion or
   exclusion) is compiled once, by an LLM, into a query over the item types of
   [Structured Clinical Extraction](extraction.md). Key lines see the cohort's key criteria as context; boilerplate
   lines compile without context, so one query serves every trial with the same line. Queries are cached.
2. **Run in code.** The query runs on each patient's extraction records with no model call: one patient, or every
   candidate of a cohort.

Two engines write the queries:

- **Compiled queries** (`compile_criteria_queries`, `check_criteria_with_extractions`): a declarative JSON query, the
  workup query language's matchers (item types, filters, date windows) with two expressions, `meets_when` and
  `fails_when`. Validated in code; the model never sees patient data.
- **The SQL agent** (`compile_criteria_sql`, `check_criteria_with_sql`): the workup SQL agent's loop, read-only
  sandbox, store profile and search, exploring an extraction store and writing one DuckDB query per line. It can also
  read lab values from result text, take the latest value (such as the most recent ECOG score) and use treatment end
  dates. Exploration results (extracted values, never patient ids or note text) reach the agent's endpoint.

```python
from matchminer_ai.matching.structured_criteria import (
    check_criteria_with_extractions,
    compile_criteria_queries,
)

compiled = compile_criteria_queries(cohort_criteria_items, cohort_context=cohort_key_criteria, config=config,
                                    cache={})
report = check_criteria_with_extractions([document_records, patient_records], compiled,
                                         patient_ids=["patient-1", "patient-2"])
report["patients"][0]["lines"][0]["status"]  # e.g. "clearly_fails"
report["aggregate"]                          # per line: statuses across patients
```

## Polarity and statuses

Results use the [decision model](criteria-check.md)'s polarity: yes means the patient does not clearly fail the line.
A line is `clearly_fails` only when documented facts contradict eligibility (an inclusion line contradicted, such as a
negative result for a required marker; or an exclusion line's condition documented), the facts that would satisfy
it do not also hold, and the line applies. A missing item never fails a line: the validator rejects a `fails_when`
or `meets_when` built only from negated (absent) criteria.

| Status | Meaning |
| --- | --- |
| `clearly_fails` | Documented facts fail the line |
| `meets` | Documented facts satisfy the line |
| `not_documented` | No extracted item shows either outcome (not failing) |
| `uncertain` | A deciding item's date or details are not certain, or a failing fact under a condition that may not apply |
| `conflicting` | Documented facts both satisfy and fail the line |
| `not_applicable` | Documented facts show the line's condition does not apply |
| `not_checkable` | No item type can check the line (age, lab thresholds in the compiled language, comorbidities) |
| `no_records` | The patient has no extraction records |
| `query_unavailable` | No valid query could be compiled, or the line has no inclusion or exclusion type |

Conditional lines ("for patients with prior surgery", "women of childbearing potential") encode their condition in
`applies_when`, or say why it cannot be documented (`condition_not_expressible_reason`); such a line is never reported
as clearly failed while its condition is uncertain.

## Limits

The extraction schema records cancers only (no comorbidities), anticancer treatments only, labs as a test with a free
text result, and performance status as stated numbers. Lines about age, organ-function thresholds, cardiac function,
pregnancy, consent, measurable disease or whether brain metastases are active are therefore usually not checkable by
compiled queries; the SQL agent may read some lab values from result text. In the compiled language a treatment is
dated by its start, so washouts are approximated (and say so).

## Dashboard

The dashboard uses both engines with the session's query-writing connection: the Patient-centric trial dialog and the
Trial management patient dialog (**Check with structured data**), and Trial management **Find patients** (**Screen with
structured data**, over every candidate the mode's extraction store holds). See [Dashboard](../apps/dashboard.md).
