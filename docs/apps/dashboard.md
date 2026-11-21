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
| Benchmarks | Evaluation runs under `dashboard.real.benchmark_roots` | Runs under `dashboard.synthetic.benchmark_roots` (the tab is hidden without any) |
| Serving | Loopback only (`127.0.0.1:8781`); a non-loopback `--host` is refused; the access token is always required | `0.0.0.0:8034` by default; any Host unless `--allowed-hosts` names them; a token only with `--require-token` |
| Data side | Writes never go to a `no_phi` folder | Never reads or writes a `phi` folder |

TrialSpace, its rounds and the decision model come from the synthetic run in both modes. Real mode compares the
population cache's vectors with cohort vectors made by the round's encoder, so it requires both to be the same
TrialSpace model and says so on Data and Models if they differ.

## Tabs

- **Trial-centric:** per disease center (real mode) or cancer type (synthetic mode), an accordion of trial groups, one row
  per cohort with its key criteria, the trial's drugs and mechanisms (from a GoodOption catalog), contacts and
  **Find patients**. Trial details show both versions' cohorts (protocol and ClinicalTrials.gov). Find patients ranks
  the mode's patients nearest a cohort's key criteria in TrialSpace, scored by the decision model; the per-criterion
  and GoodOption LLM checks run on demand. **Screen with structured data** checks the cohort's criteria lines on the
  structured records of every candidate the mode's extraction store holds (each line's query is compiled once; see
  [Structured-Record Criteria Screening](../api/structured-criteria.md)), and the search can be limited to patients
  with stored records. Trials not open to enrollment (OnCore status other than open to accrual; ClinicalTrials.gov
  status other than recruiting or not yet recruiting) are hidden until **Show closed trials** is checked, here and in
  the Patient-centric Trials view.
- **Population management:** load a population, choose a guideline space, fetch its matching patients (TrialSpace and the
  decision model's reasonableness, 0.75 by default), compress their notes, and check guideline concordance across the
  cohort (full-note review, compiled extraction queries, or the SQL agent over extraction records). Drill into a
  workup item or a patient, and export the cohort and its results.
- **Patient-centric:** one patient for three views. The patient is a corpus patient, an entered summary, or clinical
  notes (pasted, uploaded as PDF/CSV/TXT, or a synthetic example) that are compressed and summarized.
  - *Trials:* the ranked trial cohorts with decision-model scores and the GoodOption check. A trial's criteria check
    runs on the decision model, the LLM, or the patient's structured records (**Check with structured data**).
  - *Guidelines:* guideline spaces for the patient's summary, their workup and treatment menus with sources, a
    workup review, and a Markdown report.
  - *Structured data:* the clinical facts extracted from the notes, patient level or note by note, each with its
    quotes and its raw JSON (each item, each note's record, the patient record). Imaging, labs and counseling are
    extracted note by note; Patient level lists each study or test once, merged from the notes in code. **Disease course** lists each
    cancer's assessments in date order: status, sites with cancer and the note. A corpus patient the mode's extraction store holds shows its stored records; otherwise the notes are
    extracted on request with the guideline concordance connection, and the Guidelines extraction review reuses
    them.
- **Benchmarking:** distillation-fidelity runs (retrieval precision and teacher score per ranker; the decision
  model's discrimination and calibration per question family), LLM-as-judge runs (cost, agreement, results), and
  adjudication of the units the two judges disagree on. Compare and metrics can be rerun from the tab; the runs
  themselves start from the command line ([Evaluation](../evaluation/index.md)).
- **Data and Models** (the last tab): the data sources, the TrialSpace round, the decision model, background jobs and
  the **model connections**, one per LLM task for this browser session: patient summarization, note compression,
  guideline concordance checks, query writing (workup and criteria queries, the SQL agent), the two backup reviews, the
  GoodOption check and the per-criterion LLM check.

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
run is still training; Data and Models shows which one loaded). It scores in every tab and runs on `--device`, by
default `cuda:0` when CUDA is available, else the CPU.

The default model connection (`--llm-backend`, `--llm-model`) is Claude Haiku 5.5 through a Databricks AI Gateway
when the site config names the gateway (`databricks.host`, `databricks.profile`) and a Haiku endpoint
(`databricks.endpoints.haiku`), else the Vertex profile Gemini 4 Hydrogen (in the site's `vertex.project`). Claude
serves every task: Haiku's default effort (medium) for all of them except the SQL agent (high). The other
endpoints the site names (`sonnet`, `opus`) can be chosen per task. An OpenAI-compatible endpoint is also offered
(`--llm-backend openai`, `--llm-endpoint`, default the site's `endpoints.teacher.url`). Each session can change the
connection per task on Data and Models. Patient text and cohort criteria go to the selected endpoints, so use only
endpoints authorized for the data.

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
