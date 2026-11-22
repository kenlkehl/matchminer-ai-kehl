# Dashboard

`matchminer_ai.apps.dashboard` is the MatchMiner-AI web app: trials, patient populations, single patients and
benchmarks in one FastAPI server with a React frontend. The frontend's source is in `frontend/dashboard/`, and it is
built into the package's `static/` folder. It is for research: scores and guideline considerations are screening
signals for review, not eligibility determinations or treatment recommendations.

## Two modes

The mode is chosen only on the command line (`--mode`, required), never by the environment. It decides where the data
come from and how the server is reached.

| | `real` | `synthetic` |
| --- | --- | --- |
| Trials | A site's extracted trial cohorts (an extraction run and its trial table) | The synthetic training run's trials (ClinicalTrials.gov listings and synthetic protocols), grouped by cancer type |
| Patients | A preprocessed population cache of real patients | The synthetic run's patients |
| Populations | The population cache, a sample from a population DuckDB database, uploaded tables, cohort archives | A sample of the synthetic population, uploaded tables, cohort archives |
| Benchmarking | Evaluation runs under `dashboard.real.benchmark_roots` | Runs under `dashboard.synthetic.benchmark_roots` (the tab is hidden without any) |
| Serving | Loopback only (`127.0.0.1:8781`); a non-loopback `--host` is refused; the access token is always required | `0.0.0.0:8034` by default; any Host unless `--allowed-hosts` names them; a token only with `--require-token` |
| Data side | Writes never go to a `no_phi` folder | Never reads or writes a `phi` folder |

TrialSpace, its rounds and the decision model come from the synthetic run in both modes. Real mode compares the
population cache's vectors with cohort vectors made by the round's encoder, so it requires both to be the same
TrialSpace model and says so on Data and models if they differ.

## Tabs

- **Clinical trial management:** per disease center (real mode) or cancer type (synthetic mode), an accordion of trial groups, one row
  per cohort with its key criteria, the trial's drugs and mechanisms (from a GoodOption catalog), contacts and
  **Find patients**. Trial details show both versions' cohorts (protocol and ClinicalTrials.gov). Find patients ranks
  the mode's patients nearest a cohort's key criteria in TrialSpace, scored by the decision model or the LLM (see
  [Decision model or LLM](#decision-model-or-llm)); the per-criterion check and the GoodOption check run on demand.
  **Screen with structured data** checks the cohort's criteria lines on the structured records of every candidate the
  mode's extraction store holds (each line's query is compiled once; see
  [Structured-Record Criteria Screening](../api/structured-criteria.md)), and the search can be limited to patients
  with stored records. Trials not open to enrollment (OnCore status other than open to accrual; ClinicalTrials.gov
  status other than recruiting or not yet recruiting) are hidden until **Show closed trials** is checked, here and in
  the Patient-centric Trials view.
- **Guideline concordance reporting:** load a population, choose a guideline space, fetch its matching patients (TrialSpace and the
  decision model's reasonableness, 0.75 by default, or the LLM's reasonableness answer), and check guideline concordance across the
  cohort (full-note review, compiled extraction queries, or the SQL agent over extraction records). Drill into a
  workup item or a patient, and export the cohort and its results.
- **Patient-centric search:** one patient for three views. The patient is a corpus patient, an entered summary, or clinical
  notes (pasted, uploaded as PDF/CSV/TXT, or a synthetic example) that are summarized.
  - *Trials:* the ranked trial cohorts with the decision model's or the LLM's scores, and the GoodOption check. A
    trial's criteria check runs on the decision model or the LLM (the toggle), or on the patient's structured records
    (**Check with structured data**).
  - *Guidelines:* guideline spaces for the patient's summary (TrialSpace's candidates ranked by the decision model
    or the LLM), their workup and treatment menus with sources, a workup review, and a Markdown report.
  - *Structured data:* the clinical facts extracted from the notes, patient level or note by note, each with its
    quotes and its raw JSON (each item, each note's record, the patient record). Imaging, labs and counseling are
    extracted note by note; Patient level lists each study or test once, merged from the notes in code.
    **Disease course** (the first view, shown by default) puts every item type on one timeline in date order: each fact once (the patient-level
    record's, else merged from the notes) and each note's disease assessments and performance status. Each row has
    its type and a tag, the type's category: disease status, event type (such as progression or recurrence),
    treatment type, biomarker result, specimen result, modality, ECOG score. Type chips filter the rows (labs and
    counseling start hidden), and a patient with several cancers gets a cancer filter. A corpus patient the mode's
    extraction store holds shows its stored records; otherwise the notes are
    extracted on request with the guideline concordance connection. The Guidelines workup review (extraction
    queries or the SQL agent) and the criteria checks use the same records: the store's, else the session's
    extraction of the notes (one with patient level also serves a document-level review), and they extract only
    when neither exists. The store holds a sample of the patients (in synthetic mode, 1,000 of the corpus), so the
    patient search marks the patients it holds (**Structured records**) and, by default, lists only them.
- **Assistant:** a chat over the mode's own data (see [Assistant](#assistant) below).
- **Benchmarking:** distillation-fidelity runs (retrieval precision and teacher score per ranker; the decision
  model's discrimination and calibration per question family), LLM-as-judge runs (cost, agreement, results), and
  adjudication of the units the two judges disagree on. Compare and metrics can be rerun from the tab; the runs
  themselves start from the command line ([Evaluation](../evaluation/index.md)).
- **How this works:** how the models are trained and fit together at run time, and the structured extraction data
  model (see [How this works](#how-this-works) below).
- **Data and models** (the last tab): the data sources, the TrialSpace round, the decision model, background jobs and
  the **model connections**, one per LLM task for this browser session: patient summarization,
  guideline concordance checks, query writing (workup and criteria queries, the SQL agent), the two backup reviews, the
  GoodOption check, the LLM decision checks and the chat assistant. Three connectors: an OpenAI-compatible server
  (**Detect served model** reads its `/v1/models`), Google Vertex / Agent Platform (Gemini 4 Hydrogen, Gemini 3.8
  Flash or Gemma 4 26B, with the server's Application Default Credentials), and the Databricks AI Gateway (the site's
  Claude endpoints).

## Assistant

The Assistant tab answers questions about the mode's own data ("what trials are open for EGFR-mutant lung cancer?",
"find patients with metastatic colorectal cancer") in a chat. It is an agent loop inside the app
(`matchminer_ai.apps.dashboard.assistant`), not a separate agent runtime: each step is one request to the chosen
model whose answer is one JSON action, validated in code, so it works with every connector the app offers. The
action is a tool call, a clarifying question or the answer.

**Tools.** Read-only, over the running app's data, through the services the other tabs use; the model has no shell,
file or network access:

| Tool | What it does |
| --- | --- |
| `search_trials` | Regular-expression search over the dashboard cohorts (trial ID, titles, cohort names, key criteria, drug names), with the Clinical trial management filters: cancer type, center, phase, open status, GoodOption drugs and classes |
| `match_trials` | TrialSpace retrieval from a patient description (embedded as an entered summary is), one cohort per trial, optionally reranked by the decision model's match questions |
| `trial_details` | A trial's or cohort's status, phase, center, drugs and key (and, on request, boilerplate) criteria lines |
| `search_patients` | The patient list's search over the summaries |
| `match_patients` | Find patients: TrialSpace retrieval from a cohort, or from ad-hoc criteria text, optionally reranked |
| `patient_details` | A patient's summary sections and, when the extraction store holds the patient, the records' disease course |
| `screen_records` | Screen with structured data: criteria lines (a cohort's, or ad-hoc lines) compiled into queries by the query-writing connection and run in code over the store's records, with per-line counts and the patients documented to meet every checkable inclusion line. Absence never fails a line, and lines no item type can check are reported as not checkable |
| `score_pairs` | The match and boilerplate questions for given patient-cohort pairs, from the decision model or the LLM (the session's **Scores from**) |
| `guideline_spaces` | The guideline catalog's spaces nearest a patient description |

A user message gets at most 8 tool calls; then the model must answer or ask. Results list at most 25 rows with their
totals, and a repeated call is not run again. Each tool step shows on the page as it runs, as a line that opens to
the step's purpose.

**Trial status.** In real mode a trial's status is the institution's own accrual status (OnCore) from the OA trial
table the store was built from; in synthetic mode it is the ClinicalTrials.gov overall status of the run's export
(open: recruiting or not yet recruiting), not site-specific. Neither says which cohorts have open slots, and the
assistant says what the status means rather than implying more.

**Clarification.** The assistant asks before searching when a clinical concept has materially different readings:
metastatic (stage IV at diagnosis, recurrent metastatic disease, or either), first-line and line counting
(maintenance, adjuvant therapy followed by early recurrence), EGFR mutant (any alteration, sensitizing exon 19
deletion or L858R, exon 20 insertion, resistance mutations), lung cancer (NSCLC or SCLC), open, and time windows. It
asks one question at a time with 2-5 options, shown as buttons (an answer can also be typed), carries the answer
through the conversation, and states the interpretation it used under each answer. It does not ask when the reading
would not change the result.

**Answers** are Markdown. Every trial, cohort and patient ID a tool returned is a link: a trial opens its details
and a cohort its criteria in Clinical trial management, and a patient opens in Patient-centric search.

**Model.** The chat assistant is its own LLM task on Data and models (**Change model** on the tab edits the same
connection). Its default is Claude Opus 5.5 through the Databricks gateway (`databricks.endpoints.opus`), at medium
effort, with the conversation so far read from Claude's prompt cache on each step; without a gateway it uses the
app's default connection. The page shows the conversation's requests and tokens. **Stop** ends a running turn, and
**New conversation** starts over.

**Data handling.** A conversation lives in the server's memory for its browser session (it expires with the session)
and is never written to disk or logged; an ad-hoc criteria line's compiled query is kept with the conversation only.
Patient summaries and structured records go to the selected model connection, so use one authorized for the mode's
data.

## How this works

The How this works tab shows a curated set of the documentation's pages: how matching works at run time, training
(TrialSpace, the decision model, the teacher labels), patient summaries, cohort extraction, structured extraction and
criteria screening. The extraction data model page is generated from the schema's declarations (item types, fields,
vocabularies, versions), so it matches the code. From a source checkout the pages are read from `docs/`; the frontend
build copies them into the package (`static/docs/`) for an installed one. Only the pages listed in
`docs_pages.json` are served.

## Run

From a source checkout:

```bash
uv pip install -e ".[apps]"          # add databricks for Claude in the SQL agent
cd frontend/dashboard && pnpm install && pnpm build && cd -   # once, and after frontend changes
mmai-dashboard --mode synthetic --site-config site.yaml --device cuda:0
```

`python -m matchminer_ai.apps.dashboard` is the same command. `mmai-dashboard --mode ... build` builds the trial
store once and exits. `pnpm dev` serves the frontend on :5173 and proxies `/api` to a running server
(`MMAI_DASH_API_URL`, default `http://127.0.0.1:8781`).

**Access.** Real mode serves on loopback only, because it shows patient records and confidential protocol content;
forward the port with `ssh -L` to reach it from another machine. It accepts only loopback Host headers, refuses
cross-origin writes and requires the access token: `MMAI_DASH_ACCESS_TOKEN`, else a private
`~/.config/mmai-dashboard/access_token` created on first use. That keeps out other accounts on the same machine. On an
interactive terminal the server prints a one-click sign-in link; with redirected output it prints only where the token
is. Each browser gets a session: its jobs, model connections, patient workspace and population are its own and
expire after an hour unused.

## Inputs

Paths come from the command line, `MMAI_DASH_*` variables (for example `MMAI_DASH_STORE_DIR`), the site config
(`dashboard.<mode>.<key>` for a mode's own data, `dashboard.<key>` for shared values), or folders under the data root
(`--data-root`, `MMAI_DATA_ROOT` or `paths.data_root`):

| Setting | Mode | Default under the data root | Contents |
| --- | --- | --- | --- |
| `--trial-data` (`dashboard.real.trial_data`) | real | `phi/real_time/trial_data` | The trial extraction run (`inputs.csv`, `protocol/` and `ctgov/` with per-trial results); new results appear within a poll (`--rebuild-poll-seconds`, 30) |
| `--oa-db` (`dashboard.real.oa_db`) | real | `phi/real_time/oa_db_*.duckdb` | The DuckDB copy holding the trial table (the newest file of a glob) |
| `--store-dir` (`dashboard.<mode>.store_dir`) | both | `phi/real_time/dashboard`, `no_phi/dashboard_synthetic` | The trial store, cohort vectors and caches (owner only) |
| `--good-options-catalog` (`dashboard.<mode>.good_options_catalog`) | both | real: `phi/real_time/good_options_catalog/catalog`; synthetic: none | Trials' drugs and mechanism classes; missing: no drugs |
| `--grouping` (`dashboard.grouping`) | both | the bundled example rules | Trial-group rules per center |
| `--synthetic-run` (`dashboard.synthetic_run`) | both | `no_phi/synthetic_run` | TrialSpace rounds, the decision model, synthetic mode's trials and patients |
| `dashboard.real.population_cache` | real | beside the population database | The real patients (see below) |
| `dashboard.guideline_catalogs` | both | `no_phi/space_paradigms_nccn` | Finalized guideline catalogs |

The decision model is the synthetic run's trained model (`--decision-model` overrides it, with fallbacks while the
run is still training; Data and models shows which one loaded). It scores in every tab unless a view's toggle chooses
the LLM (Benchmarking always uses it) and runs on `--device`, by default `cuda:0` when CUDA is available, else the
CPU.

The default model connection (`--llm-backend`, `--llm-model`) is Claude Haiku 5.5 through a Databricks AI Gateway
when the site config names the gateway (`databricks.host`, `databricks.profile`) and a Haiku endpoint
(`databricks.endpoints.haiku`), else the Vertex profile Gemini 4 Hydrogen (in the site's `vertex.project`). Claude
serves every task: Haiku's default effort (medium) for all of them except the SQL agent (high). The other
endpoints the site names (`sonnet`, `opus`) can be chosen per task. An OpenAI-compatible endpoint is also offered
(`--llm-backend openai`, `--llm-endpoint`, default the site's `endpoints.teacher.url`). Each session can change the
connection per task on Data and models. Patient text and cohort criteria go to the selected endpoints, so use only
endpoints authorized for the data.

## Decision model or LLM

Wherever the dashboard scores a patient-cohort pair, a **Scores from: Decision model | LLM** toggle chooses who
answers. The choice belongs to the browser session: the same toggle appears in every such view, changing it in one
changes it in all, and it lasts until the browser tab closes. Benchmarking always uses the decision model (an
evaluation must). The Assistant follows it too: each message carries the choice, and its reranks and `score_pairs`
then use the LLM decision checks connection (at most 10 pairs per call, ranked by the LLM's 0-6 score), name the
source and scale in their results, and list each LLM answer under the step, opening the same answer dialog.

| View | Decision model | LLM |
| --- | --- | --- |
| Clinical trial management, Find patients: the candidates' scores, ranking and **Hide patients likely excluded** | match questions and the common-exclusion question | `score_match_quality_with_llm` and `exclusion_criteria_check_with_llm` |
| Its patient dialog (Criteria check tab) | the pair's answers and the per-criterion check | the same, from the LLM |
| Patient-centric search, Trials: the cohorts' scores, ranking and **Hide trials with likely common exclusions** | as Find patients | as Find patients |
| Its trial dialog | the pair's answers and the per-criterion check | the same, from the LLM |
| The per-criterion check (both dialogs) | `check_eligibility_criteria` | `check_eligibility_criteria_with_llm` (one request per line) |
| Patient-centric search, Guidelines: ranking TrialSpace's candidate spaces | the match score | the LLM's match score |
| Guideline concordance reporting: selecting and ranking a space's candidates | P(reasonable) at least the threshold | the LLM's reasonableness answer is yes |

**Questions and scales.** The LLM answers the prompts the training teacher labeled with (the package's LLM
functions; the dashboard writes no prompt of its own), on their own scales, never mixed with the decision model's
probabilities:

- *Match:* five yes/no answers (reasonable candidate; cancer type, disease extent, treatment history and biomarker,
  each "required by the cohort and documentedly met"), each with an evidence status and a reason, combined into the
  package's 0-6 match score: 0 unless reasonable, else 1 plus 1 each for cancer type, extent and treatment history and
  2 for biomarker. The badge reads *Possible match (LLM)* or *Unlikely match (LLM)* from the reasonableness answer,
  with the score (for example 3/6).
- *Common exclusions:* "Is this patient clearly excluded from this cohort?" from the cohort's common criteria and the
  patient's other relevant history: yes or no.
- *Per criterion:* per line, yes (does not clearly fail it) or no (clearly fails it), as the decision model's
  question 7.

Under the LLM, lists rank by the 0-6 score, then TrialSpace similarity; pairs the LLM has not scored follow by
similarity. A guideline search ranks every TrialSpace candidate the same way and keeps the top N; a population
search keeps the candidates the LLM finds reasonable, ranked the same way (the decision model's threshold does not
apply). Both searches still come from the package's retrieval, whose decision-model scores are removed from what the
view shows; the guideline report keeps the package's table, marked as the decision model's, below the LLM's ranking.
When both have answered a pair's criteria, the criteria list shows the other scorer's answers too, labeled, with the
lines they disagree on.

**Cost and latency.** LLM scoring is one request per pair and question (per line for the criteria check):

- Lists score only what the view shows: the first ten results as the page opens, then **Score the next 10 ... with
  the LLM**. A run asks only what the session has no answer to.
- Every run is a background job of the browser session (its progress shows beside the list), with at most ten
  requests in flight per question family.
- Before a run, the view shows its estimate: requests, the model, and approximate input tokens. A run of more than
  25 requests (a long criteria list, a guideline or population search over many candidates) waits for **confirm**;
  the server refuses it unconfirmed. A list run scores at most 50 pairs, and a search ranks at most 200 candidates.
- Answers are cached in the server's memory for the session, per pair texts, question, connection (backend, endpoint,
  model) and prompt version, and expire with the session.

**Reasoning.** Every LLM answer opens (click the badge, **reasoning**, or **raw output**) to its parsed answer, the
reasoning the backend returned, its raw output, and the whole record as JSON (Copy and Download). What reasoning
there is depends on the backend: an OpenAI-compatible server returns its reasoning text (vLLM's
`reasoning_content`, or what the package's reasoning parser splits off) when the model thinks; Claude through the
Databricks gateway returns summaries of its thinking, which the dashboard requests (the 5.5 models otherwise omit
them; the billing is the same); hosted Gemini on Vertex returns none through this connection. An answer without
reasoning says so.

**The model.** One LLM task on Data and models, **LLM decision checks**, serves the match, common-exclusion and
per-criterion questions. Its default is the app's default connection (Claude Haiku 5.5 through the gateway when the
site names it), and any connector can serve it: an OpenAI-compatible server (**Detect served model**), a Vertex
profile, or another of the gateway's Claude endpoints. It replaced the per-criterion task (`criteria_llm`); a
connection saved under the old name serves the new task.

**Data.** The patient's summary sections and the cohort's criteria go to the task's connection, so use an endpoint
authorized for the data, as for every LLM task. Answers and reasoning stay in the server's memory, visible only to the
session; nothing patient-derived is logged or written to disk.

## Real patients

Real mode's patients are a preprocessed population cache: summaries with precomputed TrialSpace vectors, and the
source notes, read on demand. It is built offline from a population DuckDB database (`dashboard.real.population_duckdb`)
with `patient_summary` (`mrn`, `patient_summary`, `patient_boilerplate`, `is_current`) and `patient_notes` (`id`,
`mrn`, `rpt_text`, `narrative_text`, `event_date`, `note_source`) tables:

- `python -m matchminer_ai.apps.dashboard.guidelines.preprocess_population` builds the cache (0700, with 0600 files)
  beside the database; `--limit N` builds it for a seeded sample of N patients.
- `python -m matchminer_ai.apps.dashboard.guidelines.preprocess_extractions` builds structured extraction records for
  the Structured data view, the extraction-based workup checks and criteria screening (`--phi` for a population
  cache; `--population-sample` for synthetic mode's default Population sample). Every note goes to the LLM endpoint
  given, so use one authorized for the notes:
  - an OpenAI-compatible server (`--endpoint`, `--server-registry`); the adaptive limit starts at `--concurrency`;
  - or Claude through the site's Databricks gateway (`--claude haiku`): the limit starts at 32 and grows to
    `--concurrency` while requests succeed, halving on rate limits or dropped connections. Each request's status and
    token counts (never content) go to `<output>.gateway_requests.jsonl`, totaled in the store's manifest;
    `python -m matchminer_ai.llm.gateway_usage <log> --rates <input>,<output> --long-rates <input>,<output>
    --discount <fraction>` reports tokens, cache reads, throughput, latency, the concurrency over time, when rate limits
    began, and the cost, each request priced in its own band (prompts over 100,000 tokens at the long rates).
  - Patient-level prompts through Claude stay under a budget (85,000 tokens for the notes and the record so far): each
    request takes only the notes that fit beside the record, so long histories need more, smaller requests rather than
    prompts in Haiku's long-prompt price band. Only a record too large to fit with any note goes over.

Both read `MMAI_SITE_CONFIG` and `MMAI_DATA_ROOT`. `examples/guidelines/` has workup spot checks and benchmarks on
synthetic notes.
