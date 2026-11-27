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
  **Screen with structured data** checks each of the cohort's criteria on the structured records of every candidate the
  mode's extraction store holds (each criterion's query is compiled once; see
  [Structured-Record Criteria Screening](../api/structured-criteria.md)), and the search can be limited to patients
  with stored records. Trials not open to enrollment (OnCore status other than open to accrual; ClinicalTrials.gov
  status other than recruiting or not yet recruiting) are hidden until **Show closed trials** is checked, here and in
  the Patient-centric Trials view. **Trial management** opens a sidebar beside the groups for the trials' enrollment
  slots, edited criteria, subcohorts and group placements (see [Trial management](#trial-management) below); each
  group row shows its slots.
- **Guideline concordance reporting:** load a population, choose a guideline space, fetch its matching patients (TrialSpace and the
  decision model's reasonableness, 0.75 by default, or the LLM's reasonableness answer), and check guideline concordance across the
  cohort (full-note review, compiled extraction queries, or the SQL agent over extraction records). Drill into a
  workup item or a patient, and export the cohort and its results. In synthetic mode the sample comes from
  `synthetic.population_split` (the run's test split, which the decision model never trained on); **Include training
  patients** samples every split, with the decision model's scores for training patients optimistic.
- **Patient-centric search:** one patient for three views. The patient is a corpus patient, an entered summary, or clinical
  notes (pasted, uploaded as PDF/CSV/TXT, or a synthetic example) that are summarized.
  - *Trials:* the ranked trial cohorts with the decision model's or the LLM's scores, and the GoodOption check. A
    trial's criteria check runs on the decision model or the LLM (the toggle), or on the patient's structured records
    (**Check with structured data**). **Only cohorts with open slots** (off by default) searches only the cohorts with
    slots available on Trial management, each matched and scored on its own criteria: a subcohort or an original
    cohort with slots is searched as itself, not as its merged cohort, so a subcohort limited to, say, KRAS G12C
    ranks for patients who fit it. A trial appears once per such cohort, and each result shows its own slots. With
    the option off, the results are the dashboards' merged cohorts, one per trial.
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
- **Benchmarking:** distillation-fidelity runs (retrieval precision and teacher score per ranker, (a) to (d) when the
  run has (d), the app's composite ordering, and the full pipeline's over every query in a full-denominator run; the
  decision model's discrimination and calibration per question family, pooled and per ranker), LLM-as-judge runs
  (cost, agreement, results), and adjudication of the units the two judges disagree on. Compare and metrics can be
  rerun from the tab; the runs themselves start from the command line ([Evaluation](../evaluation/index.md)).
  - A fidelity run whose metrics have it (metrics v4, `decision.by_ranker`) also shows **Decision model by ranker
    denominator**: per question, the decision model against the teacher on each ranker's own top-20 lists (A:
    EmbeddingGemma 2 top 20; B: TrialSpace top 20; C: TrialSpace top K reranked by the decision model; D: TrialSpace
    top K reranked by the composite match score, as in the app), with n,
    prevalence, AUROC, AUPRC (yes), Brier, ECE and calibration intercept and slope with their 95% intervals, pooled or
    per workflow, and each ranker's boilerplate coverage (combinations with a teacher label of all its combinations).
    Older and synthetic runs without it show no such section.
  - **Export results** on a fidelity run draws its results as a PDF and downloads it: a title page (the eval, the
    models, the denominators), the page's retrieval and decision-model tables and charts (with the by-ranker tables
    when the run has them; in real mode a row with fewer than 11 pairs or clusters shows n<11), a methods note, counts
    and design, and the stages. A background job builds it into the run's `reports/fidelity_report.pdf` (owner-only);
    `GET /api/benchmarks/fidelity/{run_id}/report.pdf` downloads the last one.
  - In real mode the PDF also has **Table 1** of the eval's patients and treatment starts (age at treatment start,
    race, ethnicity, sex, cancer type) and the **headline metrics by demographic group** (the full pipeline's P@20 and
    mean teacher score, the decision model's reasonableness AUROC). They come from `evaluation.demographics`, which
    links the run's patients to the EHR extract at the site config's `paths.retro_source_root` and caches the
    aggregates in the run's `reports/demographics.json`. Small cells are suppressed: counts of 1 to 10 show as <11,
    a larger count hidden alongside them as *, and a group with fewer than 11 patients shows n<11.
  - Synthetic patients have no demographics, so a synthetic run's PDF says so instead; synthetic mode never reads
    the extract.
- **How this works:** how the models are trained and fit together at run time, and the structured extraction data
  model (see [How this works](#how-this-works) below).
- **Data and models** (the last tab): the data sources, the TrialSpace round, the decision model, background jobs and
  the **model connections**, one per LLM task for this browser session: patient summarization,
  guideline concordance checks, query writing (workup and criteria queries, the SQL agent), the two backup reviews, the
  GoodOption check, the LLM decision checks, the chat assistant and criteria classification (edited cohort criteria).
  Three connectors: an OpenAI-compatible server
  (**Detect served model** reads its `/v1/models`), Google Vertex / Agent Platform (Gemini 4 Hydrogen, Gemini 3.8
  Flash or Gemma 4 26B, with the server's Application Default Credentials), and the Databricks AI Gateway (the site's
  Claude endpoints). The tab also has the [local model server](#local-model-server).

## Trial management

**Trial management** (Clinical trial management) tracks enrollment slots and the criteria the center actually enrolls
by. Its button opens a sidebar to the right of the center's groups (resizable; the browser remembers whether it is
open and how wide). The sidebar lists every trial the page's search and filters show, with a quick filter of its own;
each row shows the trial's protocol and NCT numbers, its title, and what is kept for it: slot totals (when they can be
added without counting a cohort twice; see Slots), subcohorts, edits and edits to review. The list reads only the
trial management file. A trial opens in place to its cohorts, each a card with its slots, criteria, group and actions,
under a diagram of where the cohorts come from (**Enlarge** shows it full size). The criteria editor and the
criteria comparisons open in dialogs. **Manage trial**, under a group row's **Trial details**, opens the sidebar on that
row's trial with its cohort (merged, original or subcohort) highlighted and in view; the page's address keeps both
(`manageTrial`, `manageCohort`), so a reload or a shared link opens the same place. Everything is saved as it is changed, to one file per mode shared by everyone
using that dashboard (`matchminer_ai.apps.dashboard.trial_management`).

**Units.** A trial's cohorts are shown at three levels, each with its own slots:

- *merged cohorts*: the dashboard cohorts, each standing for the extracted cohorts the merge step found to have
  the same criteria;
- *original cohorts*: those extracted cohorts, shown under a merged cohort that stands for two or more
  (`original_cohorts`: real mode reads the extraction's `extraction.json` or the listing run's `run_metadata.json` with
  `merge.json`; synthetic mode the run's unmerged and merged cohort tables). Their criteria take the merged cohort's
  labels where the criterion is the same (under the same heading, apart from its list number). When a trial is opened in
  the sidebar, its original cohorts' other criteria are classified automatically (the Criteria classification
  connection, below), once per version of their criteria, and the labels cached, so they are never asked again; criteria
  a failed run left unlabeled are retried with **Classify**. Without a working connection nothing is sent and the trial
  says so. Listing trials never sends a request;
- *subcohorts*: defined by a user under a merged or an original cohort, for a sponsor's requirements beyond the
  protocol's. A subcohort starts with its parent's current criteria, which are then edited.

A diagram shows where each comes from (original cohorts merged into a cohort, subcohorts defined from one).

**Slots.** Available, consented and enrolled counts per unit, entered by hand, with when they were last changed. A
unit's slots are its own: a parent's and its children's are never added up, and the trial says when a family (a
merged cohort, its original cohorts and their subcohorts) has slots at more than one level. The sidebar's list adds
up a trial's slots across its families only when no family has them at two levels, and otherwise says how many
cohorts have slots. The group rows show each cohort's slots.

**Criteria.** Any unit's criteria can be edited, added or removed. They keep the cohort pipeline's layout (one per line,
under "Inclusion criteria:" or "Exclusion criteria:"). An unchanged criterion keeps its labels; a changed or new
criterion is labeled inclusion or exclusion and key or common by the pipeline's own classification step
(`remove_boilerplate_criteria`, its prompt and one request per criterion with the whole cohort as context, the unchanged
criteria passed as known), as a job of the browser session on its **Criteria classification** connection (Data and
models; Claude Haiku 5.5 through the gateway at medium effort when the site names it, else the app's default). A
criterion's labels can also be set by hand. Without a working connection the criteria are saved with the new ones
unclassified: they count as key criteria and are skipped by per-criterion checks, as in the package. **Revert to the
AI-extracted criteria** restores a merged or original cohort's current AI version. Edits are an overlay; the extracted
criteria are never changed.

**Subcohorts follow their parent on request.** When a cohort with subcohorts is edited or reverted, the pane asks
whether the new criteria replace the subcohorts'. Yes: each subcohort's criteria are reset to them, and its previous
criteria are shown while its own requirements are re-entered. No: each keeps its own.

**What reads the current criteria.** Every cohort lookup goes through the overlay (`DashState.cohort`): Find patients
(TrialSpace retrieval from the cohort's current key criteria, embedded with the round's encoder when saved), the
decision model's match, common-exclusion and per-criterion questions, the LLM decision checks, structured-data screening
(its queries are cached by criterion, so an edited criterion is compiled anew), trial details, and the Assistant's tools
except `search_trials`, whose text search reads the AI-extracted criteria. Original cohorts and subcohorts can be
searched with Find patients; they are not added to the Patient-centric Trials results, whose index holds the dashboard
cohorts (with their edited criteria).

**Re-extraction.** Real mode re-reads its trial data as extraction runs finish, so a cohort's AI version can change
under an edit. Each edit is stored with the AI version it was based on (its name and criteria); when that changes, the
edit stays in effect and is flagged (**Base changed**, with a comparison) until the user keeps it or reverts. A
subcohort whose parent's criteria changed is flagged the same way (keep its criteria, or reset to the parent's). What
is kept for a cohort the trial no longer has is listed until deleted.

**Groups.** Dragging a cohort card (merged, original or subcohort) from the sidebar by its head, or a group row while
the sidebar is open, onto a group in the groups section puts it there instead of where the grouping rules
(`grouping.yaml`) place it; while a cohort is dragged every group is outlined and the one under it highlighted. The
card's **back to its rule group** (or **remove from the group**) undoes it. Original cohorts and subcohorts are in a
group only when put in one. **New group** above the groups adds a group of the center's own (listed first, and shown
while empty so cohorts can be dropped on it); its header renames or deletes it. The groups' **Slots** column (on by
default; **Columns** hides it) shows each cohort's slots available, consented and enrolled as entered in the sidebar.

**The file.** `dashboard.<mode>.trial_management` (`--trial-management`), by default
`dashboard_trial_management/trial_management.json` under the mode's folder of the data root (real: `phi/real_time/`;
synthetic: `no_phi/`), outside the store folder, which can be deleted to rebuild the store. It is versioned JSON
(`mmai-dash-trial-management-v1`), written atomically and owner-only (0600 in a 0700 folder) under a lock, re-read
when another process wrote it, with a log of what changed and when (never criteria text). A change based on an older
version of a unit than the file holds is refused, so one user's edit does not silently replace another's. **Export**
downloads the file; **Import** replaces it after confirming (a file from the other mode is refused, and the replaced
file is kept beside it as `trial_management.before-import.json`). Synthetic mode's file is never in a `phi` folder,
and real mode's never in a `no_phi` one. In synthetic mode, anyone who can reach the dashboard can change it.

**Clearing.** Two actions remove what users entered; each confirmation lists what goes, with counts, and offers
**Export first**. Before either writes, the file is kept beside itself, owner-only, as
`trial_management.before-clear-<UTC time>.json` or `trial_management.before-reset-<trial>-<UTC time>.json`, so
importing that copy undoes it; the newest 20 such copies are kept. Both are recorded in the log with their counts.

- **Clear all** (the sidebar's header) removes everything in the mode's file: slots, criteria edits, subcohorts,
  group placements and the centers' own groups, in every center. The button enables only once **clear** is typed.
  Cached classifier labels of extracted criteria are not user entries and stay.
- **Reset this trial** (an opened trial) removes that trial's slots, criteria edits (its cohorts go back to the AI
  extraction), subcohorts, entries kept for cohorts it no longer shows, and its cohorts' group placements. The
  centers' own groups stay, even when left empty. It is refused if anything kept for the trial changed since the
  trial was opened, so one user's reset does not remove another's newer change.

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
| `trial_details` | A trial's or cohort's status, phase, center, drugs and key (and, on request, boilerplate) criteria |
| `search_patients` | The patient list's search over the summaries |
| `match_patients` | Find patients: TrialSpace retrieval from a cohort, or from ad-hoc criteria text, optionally reranked |
| `patient_details` | A patient's summary sections and, when the extraction store holds the patient, the records' disease course |
| `screen_records` | Screen with structured data: criteria (a cohort's, or ad-hoc criteria) compiled into queries by the query-writing connection and run in code over the store's records, with per-criterion counts and the patients documented to meet every checkable inclusion criterion. Absence never fails a criterion, and criteria no item type can check are reported as not checkable |
| `score_pairs` | The match and boilerplate questions for given patient-cohort pairs, from the decision model or the LLM (the session's **Scores from**) |
| `guideline_spaces` | The guideline catalog's spaces nearest a patient description |

A user message gets at most 8 tool calls; then the model must answer or ask. Results list at most 25 rows with their
totals, and a repeated call is not run again. Each tool step shows on the page as it runs, as a line that opens to
the step's purpose; steps are numbered through the conversation.

**Charts.** When asked for a chart, or when a few counts compare naturally, the answer asks for charts, and one more
request returns up to 3 of them as data, never diagram code: a pie or donut (2-6 parts of one whole, adding up to its
denominator), bars (vertical or horizontal, one to four series, grouped or stacked) or a line (an ordered x axis), each
with the steps its numbers came from. (In every step's schema, charts would make Claude's structured-output grammar
too large.) Code checks them like the actions (lengths, caps, signs, a pie's sum, and that the cited steps' tool
results succeeded), and the page draws them in SVG with every value and share printed, a table view and a CSV
download. "Not documented" parts are grey: absence in the records is not a negative finding.

**Trial status.** In real mode a trial's status is the institution's own accrual status (OnCore) from the OA trial
table the store was built from; in synthetic mode it is the ClinicalTrials.gov overall status of the run's export
(open: recruiting or not yet recruiting), not site-specific. Neither says which cohorts have open slots, and the
assistant says what the status means rather than implying more. Slots entered on [Trial management](#trial-management)
appear in `trial_details` as a cohort's `slots`; a cohort without them has none entered.

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

**Data handling.** A conversation lives in the server's memory for its browser session (it expires with the session) and
is never written to disk or logged; an ad-hoc criterion's compiled query is kept with the conversation only. Patient
summaries and structured records go to the selected model connection, so use one authorized for the mode's data.

## How this works

The How this works tab shows a curated set of the documentation's pages. It opens on an
[overview of the app](dashboard-overview.md), then how matching works at run time, training (TrialSpace, the
decision model, the teacher labels), patient summaries, cohort extraction, structured extraction and criteria
screening. The extraction data model page is generated from the schema's declarations (item types, fields,
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
store once and exits; `mmai-dashboard --mode ... prepare` only builds the synthetic run from its published dataset
(below) and exits. `pnpm dev` serves the frontend on :5173 and proxies `/api` to a running server
(`MMAI_DASH_API_URL`, default `http://127.0.0.1:8781`).

**Framing.** No other site may show the app in a frame, except, in synthetic mode, the https origins of
`--frame-ancestors` (`MMAI_DASH_FRAME_ANCESTORS`, site config `dashboard.synthetic.frame_ancestors`), such as
`https://huggingface.co` for a Space page. The session cookie is then a partitioned third-party cookie, which browsers
keep in a frame.

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
| `--trial-management` (`dashboard.<mode>.trial_management`) | both | `phi/real_time/dashboard_trial_management/trial_management.json`, `no_phi/dashboard_trial_management/trial_management.json` | Slots, edited criteria, subcohorts and group placements ([Trial management](#trial-management)); outside the rebuildable store folder |
| `--synthetic-run` (`dashboard.synthetic_run`) | both | `no_phi/synthetic_run` | TrialSpace rounds, the decision model, synthetic mode's trials and patients; or a published run (below) |
| `dashboard.synthetic.trial_cohorts`, `dashboard.synthetic.trial_tables` | synthetic | the run's corpus cohorts and trial tables | Another trial set: a cohort table (or corpus JSON lines) and trial tables merged by `nct_id`; cohorts without the round's vectors are embedded at startup (`--embedding-device`) |
| `dashboard.real.population_cache` | real | beside the population database | The real patients (see below) |
| `dashboard.guideline_catalogs` | both | `no_phi/space_paradigms_nccn` | Finalized guideline catalogs; without any, the Guidelines sub-tab and Guideline concordance reporting are hidden |

**Published data.** A site config value may name a file or folder on the Hugging Face Hub at a pinned revision,
`hf://datasets/<org>/<name>@<revision>/<path>`; the app reads it from the Hugging Face cache, downloading it on first
use with `--allow-downloads` ([site configuration](../reference/site-config.md#hugging-face-hub-files)). Read-only
inputs only: the extraction store, the GoodOption catalog, the synthetic population and guideline catalogs.
`dashboard.synthetic_run` may name a whole published run, `hf://datasets/ksg-dfci/mmai-synthetic-1226@<revision>`:
before serving, the app builds the run folder it reads under the data root's `no_phi/synthetic_runs/` from the
dataset's corpus, trial tables, patient metadata and notes, and embeds the corpus with the published TrialSpace
(minutes on a GPU, once per data root and revision; `apps.dashboard.hub_run`). The decision model is then the
published MatchEval-1226. The [Hugging Face Space](hf-space.md) runs this way.

The decision model is the synthetic run's trained model (`--decision-model` overrides it, with fallbacks while the
run is still training; Data and models shows which one loaded). It scores in every tab unless a view's toggle chooses
the LLM (Benchmarking always uses it) and runs on `--device`, by default `cuda:0` when CUDA is available, else the
CPU. Its dtype (`--dtype`, `MMAI_DASH_DTYPE`) is `auto` by default: bfloat16, the checkpoint's own, from Ampere on and on
the CPU, and float16 on an older GPU such as a T4, which gave the same answers (see the
[lite Space](hf-space.md#the-lite-space)).

The default model connection (`--llm-backend`, `--llm-model`) is Claude Haiku 5.5 through a Databricks AI Gateway
when the site config names the gateway (`databricks.host`, `databricks.profile`) and a Haiku endpoint
(`databricks.endpoints.haiku`), else the Vertex profile Gemini 4 Hydrogen (in the site's `vertex.project`). Claude
serves every task: Haiku's default effort (medium) for all of them except the SQL agent (high). The other
endpoints the site names (`sonnet`, `opus`) can be chosen per task. An OpenAI-compatible endpoint is also offered
(`--llm-backend openai`, `--llm-endpoint`, default the site's `endpoints.teacher.url`), and in synthetic mode
[OpenRouter](#openrouter) with a user's own key. Each session can change the connection per task on Data and models.
Patient text and cohort criteria go to the selected endpoints, so use only endpoints authorized for the data.

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
| The per-criterion check (both dialogs) | `check_eligibility_criteria` | `check_eligibility_criteria_with_llm` (one request per criterion) |
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
- *Per criterion:* for each criterion, yes (does not clearly fail it) or no (clearly fails it), as the decision model's
  question 7.

Under the LLM, lists rank by the 0-6 score, then TrialSpace similarity; pairs the LLM has not scored follow by
similarity. A guideline search ranks every TrialSpace candidate the same way and keeps the top N; a population
search keeps the candidates the LLM finds reasonable, ranked the same way (the decision model's threshold does not
apply). Both searches still come from the package's retrieval, whose decision-model scores are removed from what the
view shows; the guideline report keeps the package's table, marked as the decision model's, below the LLM's ranking.
When both have answered a pair's criteria, the criteria list shows the other scorer's answers too, labeled, with the
criteria they disagree on.

**Cost and latency.** LLM scoring is one request per pair and question (per criterion for the criteria check):

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

## Local model server

Every dashboard, in both modes, can run its own model server: Data and models has a **Local model server** section
above Models. It starts one vLLM or llama.cpp (`llama-server`) server on the app's own machine. Both engines serve the
same OpenAI-compatible API on loopback (`127.0.0.1`, a free port), so every LLM task can use it. A machine that cannot
reach a hosted endpoint, such as a Hugging Face Space with a GPU, runs the app with `--local-server-only` (below).

- **Who:** anyone using the app may start, stop and configure the server, on any GPU it finds (user, 2026-11-24).
- **Nothing installed:** without a GPU the section says so (llama.cpp can still run a small model on the CPU); without
  vLLM or llama-server it says which is missing and why, and Start stays off.
- **GPUs:** the GPUs are found automatically (NVML, else `nvidia-smi`, else PyTorch) and refreshed every 5 seconds.
  Each shows its compute capability and a memory bar split into this dashboard, the local server and other
  processes. In a container the driver may list no processes, so all used memory shows as other.
- **Form:** the engine (one that is not installed is disabled, with the reason), the model (a Hugging Face
  repository; default Nemotron 3.5 Bolt), the GGUF file (llama.cpp), the GPUs (default: the one with the most free
  memory), the engine parameters with **Reset to recommended**, and under Advanced extra flags, environment variables
  and vLLM's speculative decoding config. Below the form are the command and the fit warnings: weights against free
  memory, vLLM's memory share against each GPU's free memory, the recipe's hardware limits and the cache's free disk.
- **Recipes:** a known model (`presets/local_servers.yaml`) brings its model card's settings for each engine, and the
  form proposes an engine for the chosen GPUs and says why. Other models get generic settings: vLLM's tensor
  parallelism is the number of GPUs and its memory share is the tightest GPU's (free - 1 GB) / total (vLLM counts the
  dashboard's own models against its share); llama.cpp puts every layer on the GPUs, with four slots sharing one
  unified KV cache.
- **Start:** the files the engine needs are downloaded into the Hugging Face cache first, after a free-space check;
  the status line shows "Downloading 12.3 of 22.0 GB", then the phase (loading weights, allocating the KV cache,
  capturing CUDA graphs) and "Ready" once `/v1/models` lists the served name, with the elapsed time. An engine that
  ends on its own shows "Exited" and its code. The log tail (the command line, then the engine's messages; request
  logging is off) is under **Log**. An app without `--allow-downloads` uses only what the cache already holds.
- **Use:** once ready, **Use for every task** points this browser session's tasks at the server, with its endpoint,
  model and tokenizer, and the **Reasoning** switch beside it. On a task's own form the OpenAI-compatible connector
  has **Use the local server**. With `--local-server-default`, the ready server also becomes the default connection
  of every task in sessions that chose none, until it stops.
- **Stop:** **Stop** ends the engine's whole process group (SIGTERM, then SIGKILL after 45 seconds), so no engine
  worker keeps a GPU. The app stops its server when it exits, on Ctrl-C or SIGTERM; on Linux the engine also gets
  SIGTERM if the app is killed or crashes.

**The default model, Nemotron 3.5 Bolt 74B-A9B (2-bit, about 22 GB).** Its settings are copied from NVIDIA's model
cards (verified 2026-11-24), text only:

- **vLLM** (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`): the card's recipe needs a vLLM nightly with the b12x
  kernels, which run on Blackwell (SM 12.0 and 12.1) only, so on other GPUs the form proposes llama.cpp. The card's
  flags are kept (fp8 KV cache, b12x and FlashInfer backends, the Mamba cache settings, the LiLiCorr DFlash drafter as
  `speculative_config`, `VLLM_USE_V2_MODEL_RUNNER=1`). The card registers its reasoning parser, `ultra_v3`, from a file
  in the model repository; the spec's `reasoning_parser_plugin` names it, and the download includes it. Left out:
  `--allowed-local-media-path /`, which would let a request read any file on the machine.
- **llama.cpp** (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`): the card's GGUF, every layer on the GPU, one
  unified KV cache and its DFlash drafter (`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF`). The card's
  two recipes are variants chosen by the chosen GPUs' free memory: the 32GB recipe (the full 262,144-token context)
  when 32 GB are free, else the 28GB recipe (65,536 tokens, a q8_0 KV cache). The vision projector (`mmproj`) is left
  out. The app downloads the files itself and starts `llama-server --model <path>`, so the engine never downloads.
- **Requests:** the app sends Bolt NVIDIA's sampling (temperature 1.0, top_p 0.95) in every task, in place of a task's
  own fixed sampling, as for Nemotron 3.5 Super; its thinking switch follows the **Reasoning** setting.

On an L40S (48 GB) the GGUF, the decision model (about 5 GB) and TrialSpace (about 1 GB) fit on one GPU with llama.cpp.
`Qwen/Qwen3-0.6B` is also known, as a small model for checking a setup.

**Only the local server.** With `--local-server-only` every LLM task uses the app's own local server, and the
server refuses every other model connection, wherever one is chosen or used:

- **Refused:** the Databricks gateway, Vertex, Azure and any OpenAI-compatible URL, including other loopback
  addresses. A session's or a task's connection, **Use for every task**, the connection check, the Assistant, the
  **Scores from: LLM** checks, criteria classification and trial management's labeling all accept only the ready
  local server (matched by endpoint and served name), and a refused one gets a 400 that says why. The package's
  connection builders apply the same rule (`llm.connections.set_connection_policy`), so no preset section the app
  builds can point anywhere else.
- **Defaults:** the ready server is every task's default (`--local-server-default` is implied). Before it is ready
  there is no default connection: LLM tasks fail with a message pointing to Data and models, and nothing falls back
  to a hosted endpoint the site config names. A connection saved for a server since stopped or restarted is dropped.
- **Site config:** no gateway, Vertex or Azure settings are needed (the `--llm-*` options and the site's hosted
  defaults are ignored).
- **Frontend:** Data and models' connection forms offer only the local server. Benchmarking reads stored answers
  and is unaffected.
- **OpenRouter:** with `--allow-openrouter` (synthetic mode), [OpenRouter](#openrouter) with a session's own key is
  accepted too, for example while the server loads or on a machine without a suitable GPU. The forms then offer the
  local server and OpenRouter; nothing else.
- **No local server:** `--no-local-server` (`MMAI_DASH_LOCAL_SERVER_ENABLED=false`, site config
  `local_server.enabled: false`) is for a machine that cannot host an LLM: Data and models says so instead of showing
  the server's controls, starting or using one is refused, and no autostart runs. With `--local-server-only` and
  `--allow-openrouter` too (the [lite Space](hf-space.md#the-lite-space)), OpenRouter is the only connection and every
  task's default (the presets' default model, reasoning on), served by each session's own key, or the
  [hosted key](#hosted-key)'s model, served by the owner's key until a session pastes its own.
- **Other network use is unchanged:** Hugging Face model downloads (with `--allow-downloads`) and ClinicalTrials.gov
  fetches.

**Flags** (also `MMAI_DASH_LOCAL_SERVER_ONLY`, `MMAI_DASH_LOCAL_SERVER_AUTOSTART`, `MMAI_DASH_LOCAL_SERVER_DEFAULT`,
`MMAI_DASH_LLAMA_SERVER`, `MMAI_DASH_ALLOW_OPENROUTER` and `MMAI_DASH_LOCAL_SERVER_ENABLED`, or the site config's
`dashboard.<mode>.local_server` with `only`, `autostart`, `default`, `llama_server`, `allow_openrouter` and
`enabled`):

| Flag | Effect |
| --- | --- |
| `--local-server-only` | Every LLM task uses the app's local model server; every other connection is refused (implies `--local-server-default`) |
| `--local-server-autostart JSON\|PATH` | Start a server spec (JSON, or a JSON or YAML file) once the decision model has loaded, so vLLM's memory profile sees it. The engine, model and GPUs are enough: the rest comes from the model's recipe, with the variant for the GPUs' free memory |
| `--local-server-default` | Once ready, the server is the default connection of every task in sessions that chose none |
| `--llama-server PATH` | The `llama-server` binary (default: `$MMAI_LLAMA_SERVER`, else `llama-server` on `PATH`) |
| `--allow-openrouter` | Synthetic mode, with `--local-server-only`: also accept OpenRouter with a session's own key |
| `--no-local-server` | Offer no local model server (a machine that cannot host an LLM); with the two above, OpenRouter only |

For example, a Space (the [Hugging Face Space](hf-space.md) sets the same in its site config): Bolt with llama.cpp
on the first GPU, the only model connection:

```bash
mmai-dashboard --mode synthetic --allow-downloads --local-server-only \
  --local-server-autostart '{"engine": "llama_cpp", "model": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "gpus": [0]}'
```

The package API is described in [Local model server](../api/local-server.md).

## OpenRouter

In synthetic mode Data and models also offers **OpenRouter (your own API key)** for any task or **Use for every
task**: an OpenRouter model ID and the user's own OpenRouter API key, at the user's own risk. It needs no site
config and no GPU, so it also serves when no local model server can run.

- **Models:** any OpenRouter model ID may be typed; the form suggests some (`presets/openrouter.yaml`, IDs from
  OpenRouter's model list): Nemotron 3.5 Super (the model the synthetic training data was generated and labeled
  with; the default) and its free tier, Claude Haiku, Sonnet and Opus 5.5, and open models with JSON-schema output
  (Qwen3.8 Flash, DeepSeek V4.1 Flash, GLM 5.3, Kimi K3).
- **The key:** pasted once per browser session and kept only in the app's memory for that session: never written
  to disk, logged, put in an error or an export, or sent back to the browser (the form shows only "key set" and its
  last four characters). It is sent only to `https://openrouter.ai/api/v1`, and forgotten when the session expires
  or the user removes it. The key and the synthetic data the tasks send go to OpenRouter under the user's own
  account, which is billed for them.
- **Requests** (`matchminer_ai.llm.openrouter`): every task answers in its JSON schema (`response_format`), with
  `provider.require_parameters` so that only providers honouring every parameter serve it. Reasoning goes through
  OpenRouter's `reasoning` parameter: Claude gets the same efforts as through the gateway (medium, high for the query
  writers) as `reasoning.effort` and `verbosity`, and no temperature, top_p or top_k; other models get their model
  profile's sampling (Nemotron 3.5 Super: NVIDIA's) and the **Reasoning** switch as `reasoning.enabled`. Outputs are
  capped at 65,536 tokens; prompt budgets count UTF-8 bytes.
- **Errors:** fixed texts for a refused key (401), too few credits (402), a refused model or input (403), no
  provider for the parameters (404) and rate limits (429); response bodies are not kept.
- **Connection check:** reads the key's own record and the public model list, and checks that the model offers
  JSON-schema output; it sends no prompt.
- **Real mode never offers OpenRouter.** Its routes, the connection checks and the package's connection builders
  (`llm.connections.set_connection_policy`, the mechanism `--local-server-only` uses) refuse it, and
  `allow_openrouter` is an error there. OpenRouter cannot be the app's default connection (`--llm-backend`), since
  the key is per session.

### Hosted key

A synthetic-mode app can also offer its owner's own OpenRouter key to the sessions that pasted none (the
[lite Space](hf-space.md#the-lite-space) does), for one model: **Hosted: Nemotron 3.5 Super (free, shared quota)**
on Data and models (`apps.dashboard.hosted_openrouter`, `llm.openrouter.HostedQuota`).

- **When:** the site config has `dashboard.synthetic.hosted_openrouter` (below) *and* the environment has the key in
  `MMAI_OPENROUTER_HOSTED_KEY` (a Space secret). Without the secret, nothing is offered, so one site config serves an
  app with the secret and one without. Real mode refuses the block and drops the secret.
- **The key:** read once at startup and removed from the app's environment before anything else starts, so no child
  process inherits it (the Space's `entrypoint.sh` runs its helpers without it too). It is kept in memory like a
  session's key, but nothing of it is shown, not even its last four characters: the page says "this Space's key".
- **One model:** a request through the hosted key for another model is refused in `llm.openrouter` before anything
  is sent; restrict the key on OpenRouter to the same model (a guardrail's model allowlist) as the second lock. A
  session without its own key that saves another model gets "This Space's key serves only <model>; paste your own
  key for other models".
- **Default:** with OpenRouter the only connection (no local server, `--local-server-only` and `--allow-openrouter`),
  the hosted model is every task's default, the chat assistant's included. Each session's requests count against
  its own share. A session that pastes its own key uses it instead, for any model, as before.
- **Limits** (OpenRouter's free models allow 20 requests a minute, and 50 a day per account, or 1,000 once it has
  bought $10 of credits; failed requests count):
  - at most `requests_per_minute` requests start in any minute, for the whole app; the others wait for a slot;
  - `session_daily_requests` per browser session and UTC day;
  - the account's free-model requests left today, from OpenRouter's `GET /api/v1/key` (`free_model_daily_requests`;
    read at most once a minute, uses no model request) minus what the app sent since (and in the counter's lag
    before): at `daily_reserve` left, requests stop with a message to paste one's own key. When OpenRouter reports
    no counter, the other limits still hold;
  - a 429 through the hosted key is never retried (a retry would spend the day's requests), and the key then pauses
    for a minute: requests are refused in the app, before anything is sent, so no retry of a task above the request
    (a summary's quality retries, say) reaches OpenRouter.
- **Messages:** every refusal is a fixed text that says what to do (wait, or paste one's own key); a patient summary
  that fails on one shows it instead of the generic failure.
- **Data and models** shows the option with its model locked, today's requests left for visitors and the session's
  share, and the notice: requests go to OpenRouter under the Space's account, free models' providers may log
  prompts, synthetic data only. The connection check reads the key's record through the same once-a-minute reading.

| `hosted_openrouter` key | Default | Meaning |
| --- | --- | --- |
| `model` | `nvidia/nemotron-3.5-super-vl:free` | The one model the hosted key serves |
| `label` | `Nemotron 3.5 Super` | Its name on Data and models |
| `requests_per_minute` | 16 | Requests a minute for the whole app (at most 20, OpenRouter's free-model limit) |
| `session_daily_requests` | 150 | Requests per browser session and UTC day |
| `daily_reserve` | 20 | Requests stop when this many of the account's free-model requests are left today |

The block may be `{}` for the defaults.

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
