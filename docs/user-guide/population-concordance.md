# Anchored guideline concordance

!!! warning "Concordance measured from documentation"
    Concordance here is measured from documentation in the notes through each patient's anchor note. A step that
    was done but not documented there counts as **not met**, and a step the records cannot settle is **unknown**
    and left out of the fraction. Guideline spaces are chosen by TrialSpace and the decision model, and workup
    results come from LLM summaries, extraction and compiled queries: review them against the source notes. The
    results are research aids, not a measure of the quality of care, and need clinical review.

`matchminer_ai.population` also runs a population-level guideline-concordance review without the dashboard. Each
patient is anchored at one note written by their provider. The review then reads only the notes up to that anchor:

1. **Anchor.** A seeded random note by the patient's provider (in their department, within an optional window).
2. **Prepare** (part 1). A summary and structured extraction records from the notes through the anchor.
3. **Assess** (part 2). The patient's most relevant guideline space (TrialSpace and the decision model). Then that
   space's workup items, checked by the compiled structured-extraction queries (by default) or another method.
4. **Report.** Concordance per patient and item, per patient, and by provider and department.

Part 2 runs on part 1's saved outputs, so one prepared population can be assessed again, for example with another
method or catalog. The [population workup review](population-workup.md) is the related cohort-level tool: it
checks one guideline space across a cohort.

## Inputs

**Patients**, one row per patient:

- `patient_id`;
- `provider_id` and/or `provider_name` (at least one per row);
- optionally `department_id` and/or `department_name`.

**Notes:**

- `patient_id`, `note_date`, `note_text` and optionally `note_type`;
- optionally `note_id`. Without it, each note gets the extraction stages' code-derived id (a hash of patient, date,
  type and text; `patients.extraction.notes.note_ids`), so evidence links back to the note;
- the note's `provider_id` and/or `provider_name`, and `department_id` and/or `department_name`, as needed to match
  the patients table.

Tables are DataFrames, or CSV (read as text) or Parquet files. `columns` (`--columns FIELD=COLUMN` on the command
line) renames the caller's columns; a key `patients.FIELD` or `notes.FIELD` applies to one table only, for example
`notes.provider_id=AUTHOR_ID`.

## Anchor notes

`choose_anchor_notes(patients, notes, start_date=None, end_date=None, seed=23)` returns `(anchors, exclusions)`.

**Which notes are candidates:**

- **Provider.** The note's provider must match the patient's: by ID when the patient row gives one, otherwise by
  name, ignoring case and repeated spaces.
- **Department.** When the patient row gives a department, the note must be in it, matched the same way.
- **Window.** The note must be dated, with `start_date <= note_date <= end_date`. Each bound is optional and
  inclusive, and a bound without a time covers its whole day.

**Choice.** The anchor is `random.Random(f"{seed}:{patient_id}")`'s choice among the candidates in date and note-id
order, so it depends on the seed and the patient, not on row order.

**Exclusions.** A patient without a candidate is excluded, with one reason:

| Reason | Meaning |
| --- | --- |
| `no_notes` | No notes |
| `no_provider_note` | No note by the provider |
| `no_department_note` | No note by the provider in the department |
| `no_dated_note` | No dated note by the provider |
| `no_note_in_window` | No note by the provider in the window |

**Notes through the anchor** (`notes_through_anchor`). Every note of the patient, by any author, dated on or before
the anchor date. When both the note and the anchor carry a time, timestamps are compared, and a note at the
anchor's timestamp is included. Undated notes are left out.

`anchors` has `patient_id`, the patient row's provider and department fields, `anchor_note_id`, `anchor_date`,
`anchor_time` (None when the anchor note has no time), `candidate_notes`, `notes_through_anchor` and
`undated_notes`.

## Part 1: prepare

`prepare_anchored_population(patients, notes, output_dir=..., start_date=..., end_date=..., seed=...,
summary_config=..., extraction_config=...)` runs three stages into one folder:

| Stage | What it does | Outputs |
| --- | --- | --- |
| `anchors/` | Anchors, exclusions and the notes through each anchor | `anchors.parquet`, `exclusions.parquet`, `notes.parquet` |
| `summaries/` | [`summarize_patients`](../api/patients.md) on those notes | `summaries.parquet` (`patient_id`, provider and department fields, `anchor_date`, `cancer_history_summary`, `general_exclusion_criteria_evidence`, `note_count`), `qc.parquet` |
| `extractions/` | [`extract_patient_records`, then `extract_document_records`](../api/extraction.md) linked to them | `store/`: an extraction record store (`document_records.parquet`, `patient_records.parquet`, `manifest.json`) |

**Stages and resuming:**

- Each stage writes a path-free `stage_receipt.json` last. Its identity holds the input file stamps (size and
  modification time, never hashes), the window, the seed, the models and the prompt contracts.
- A complete stage is skipped. An interrupted one resumes from its checkpoints: the summaries' checkpoint folder is
  keyed by the stage identity, which includes the anchors receipt, and the extractors check their own.
- A folder begun under other settings is refused; use a fresh one.
- `check=True` (`--check`) reports each stage without running anything: 0 complete, 10 not yet, 20 settings or
  inputs differ.

**Options:**

- `levels="document"` extracts document records only.
- `reuse_documents` names an extraction checkpoint folder. Its document records of the same notes are reused when
  their input hash and contract match, which is the extractor's own rule. Records linked to different patient
  records do not match.

**Which records the dashboard can open.** The store's format is the dashboard's
(`matchminer_ai.patients.extraction.store`). Its mode is `synthetic` with `synthetic=True` and `real` otherwise.

Every note through an anchor is sent to the summary and extraction endpoints, which must be authorized for the notes.

## Part 2: assess

`assess_concordance(prepared, catalogs=..., output_dir=..., method="extraction_queries")` reads part 1's folder.

**Match.** `retrieve_guideline_considerations(summaries, catalog, top_n=1, candidate_k=20)` runs over the catalogs:
TrialSpace's `candidate_k` nearest spaces get the decision model's match questions, and the space ranked first by
`match_quality_score` is the patient's.

- **Catalogs.** By default, the latest complete, audited catalog of each disease under the site's catalog root
  (`trials.latest_guideline_catalogs`, the dashboard's rule). The root is the site config's
  `dashboard.guideline_catalogs`, else `<data root>/no_phi/space_paradigms_nccn`.
- **No reasonable match.** A patient whose top space's P(reasonable) (`match_quality_reasonableness`) is below
  `min_reasonableness` (default 0.5) is excluded with the reason `no_reasonable_guideline_match`. The candidate and
  its scores are kept in `matches/guideline_matches.parquet`.
- **Reuse.** The match stage has its own receipt (`matches/`). Guideline vectors are cached in the output's
  `guideline_embeddings/` (guideline text only) unless `embedding_cache_dir` names a shared folder.

**Assess.** Patients are grouped by their space. The space's workup items are checked, without the items that ask
only for a history and physical (`patients.workup.is_history_and_physical`). The space's key criteria are the
population context, and each patient's evaluation date is the anchor date. Methods:

| Method | What it does |
| --- | --- |
| `extraction_queries` (default) | [`compile_workup_queries`](../api/workup-queries.md) once per item and population, then code runs the queries on the patient's extraction records. Queries are cached under `query_cache/writer-<model hash>/`, the dashboard's layout; `query_cache` names a shared folder |
| `extraction_sql` | The [SQL agent](../api/workup-sql.md) explores a database of the run's own records (`sql/exploration.duckdb`); its accepted queries then run once over each space's patients. Exploration results (extracted values, no ids or note text) reach the query-writing endpoint. Accepted queries keep aggregate counts, so a shared `query_cache` follows the outputs' PHI rule |
| `full_notes` | [`review_population_workup`](population-workup.md#full-note-review) on each patient's notes through the anchor |
| `note_search` | `review_patient_workup_with_note_search` on the same notes |

For the extraction methods, a patient without any complete or partial extraction record is excluded
(`no_extraction_records`). A patient matched to a space with no workup items left is excluded too
(`no_workup_items`).

## Status mapping

Each result keeps its raw status and reason, and maps to one category (`concordance_category`):

| Category | Compiled queries and SQL agent | Full-note and note-search reviews |
| --- | --- | --- |
| met | `met` | `completed` |
| not met | `not_met`, `done_not_recommended` | `partially_completed`, `planned`, `not_done`, `not_documented` |
| unknown | `unknown` | `unclear`, `error` |
| not applicable | `not_applicable` | applicability `not_applicable` |

Two rules refine the table:

- A met or not-met result whose applicability is uncertain is unknown (`category_note`
  `applicability_uncertain`). The documentation fractions leave such results out too.
- For a step the guideline advises against (SQL agent, `recommends_against`), a step done anyway
  (`done_not_recommended`) is not met. A step not documented as done (`not_met`) is met (`category_note`
  `not_recommended_not_done`).

**Concordance** is met / applicable, where applicable = met + not met. Unknown results are counted and shown, but
they are not in the fraction.

## Outputs and metrics

Every file is private (files 0600, folders 0700) and the process umask is 077 while a stage runs.

| File | Contents |
| --- | --- |
| `item_results.parquet` | One row per patient and item: provider and department, anchor date, space, item index, name, conditions and category, method, raw status, reason, applicability, category and note, bottom line, query key, and evidence (note id, date, quote, offsets) |
| `patient_results.parquet`, `.csv` | Per patient: items, not applicable, applicable, met, not met, unknown, and concordance (empty when nothing applies) |
| `provider_summary.csv`, `department_summary.csv` | Per group: patients, included, excluded in all and by reason, patients with an applicable item, mean per-patient concordance with a 95% bootstrap interval over patients, the median, and pooled item concordance (met / applicable over every applicable item) with a 95% interval from resampling patients. `small_group` flags groups under 11 included patients |
| `provider_items.csv`, `item_summary.csv` | Counts and concordance per provider, space and item, and per space and item |
| `exclusions.parquet` | Every excluded patient once, at the first stage that excluded it |
| `flow.json` | Patients given, anchored, summarized, extracted, matched and assessed, with exclusion reasons at each step |
| `report.md` | Counts and metrics only, with the PHI banner and the concordance caveat. Small groups are flagged, not suppressed, since the report is internal |
| `design_notes.json` | Method, thresholds, catalogs, models, window and seed, compile totals and gateway usage |

A provider or department is identified as the anchors match it: by ID when the patient row gives one, else by
normalized name. The bootstrap intervals use 2,000 draws with seed 42 (`evaluation.retro_fidelity.metrics`).

## Command line

```bash
export MMAI_SITE_CONFIG=/path/to/site.yaml
python -m matchminer_ai.population.review prepare --patients patients.csv --notes notes.parquet \
    --start-date 2025-01-01 --end-date 2026-06-30 \
    --columns notes.provider_id=AUTHOR_ID notes.department_name=DEPARTMENT \
    --output /data/phi/concordance/run1/prepared
python -m matchminer_ai.population.review assess --prepared /data/phi/concordance/run1/prepared \
    --output /data/phi/concordance/run1/assessment
```

`run` does both, into `<output>/prepared` and `<output>/assessment`.

**Models.** LLMs default to Claude through the site config's Databricks gateway:

- Haiku 5.5 for summaries, extraction and the note-reading methods;
- Sonnet 5.5 for compiling the workup queries (the SQL agent's default too).

`--summary-model`, `--extraction-model`, `--compile-model` and `--review-model` name another model of the site
config's `databricks.endpoints` (`haiku`, `sonnet`, `opus`) or a full endpoint name. With `--<task>-endpoint URL`,
the task uses that OpenAI-compatible server instead, and the model option names its served model. `--thinking on|off`
sets a local model's thinking switch.

**Other options:**

- `--device` runs TrialSpace and the decision model (default `cpu`). The decision model scores each patient's
  `candidate_k` spaces, which is slow on a CPU: for a large population use `--device cuda` with
  `CUDA_VISIBLE_DEVICES`.
- `--concurrency` (default 32) bounds requests in flight; the gateway's adaptive limit starts at 32.
- `--query-cache` and `--embedding-cache` share compiled queries and guideline vectors across runs.
- `--max-concurrent-spaces` (default 4) assesses several spaces at once.

**Logs and privacy:**

- **Token logs.** Gateway token counts (never content) go to each stage's `gateway_requests.jsonl`. Summarize them
  with `python -m matchminer_ai.llm.gateway_usage <log>`.
- **Where outputs go.** A real run writes only under `<data root>/phi` or the site config's `paths.phi_roots`, and
  never in a `no_phi` folder. `--synthetic` (synthetic notes only) allows other folders, but never a `phi` folder.
- **Logs** show counts only.

## Python

```python
from matchminer_ai.population import assess_concordance, prepare_anchored_population

prepared = prepare_anchored_population(
    patients, notes, output_dir="/data/phi/concordance/run1/prepared",
    start_date="2025-01-01", end_date="2026-06-30",
    summary_config=summary_config, extraction_config=extraction_config,
)
tables = assess_concordance(
    "/data/phi/concordance/run1/prepared", catalogs="/data/no_phi/space_paradigms_nccn",
    output_dir="/data/phi/concordance/run1/assessment",
    config=models_config, compile_config=compile_config,
)
print(tables["flow"], tables["provider_summary"])
```

`matchminer_ai.population.review` builds the four configurations from connections
(`summary_config`, `extraction_config`, `models_config`, `compile_config`). A connection is
`llm.connections.claude_connection("haiku")` or `llm.connections.Connection("local", model, endpoint)`.

## API reference

::: matchminer_ai.population.anchors
    options:
      members:
        - choose_anchor_notes
        - notes_through_anchor

::: matchminer_ai.population.prepare
    options:
      members:
        - prepare_anchored_population
        - load_prepared_population
        - PreparedPopulation
        - check_output_location

::: matchminer_ai.population.concordance
    options:
      members:
        - assess_concordance
        - match_guideline_spaces
        - concordance_category
        - catalog_paths

::: matchminer_ai.population.report
    options:
      members:
        - write_concordance_report
        - read_concordance_report
