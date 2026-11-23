# Evaluations

`matchminer_ai.evaluation` holds the evaluations of the pipeline on real patients. Their inputs and outputs are
clinical data (PHI): they read and write only under the data root's `phi/` tree (or a PHI directory the site config
names, `paths.phi_roots`), create files 600 and directories 700, and log counts only. Run them only where that data is
authorized, with LLM endpoints approved for it (the site config's `endpoints.*`).

Install with the evaluation extra: `uv pip install -e ".[training,evaluation]"`.

| Module | What it does |
| --- | --- |
| `evaluation.retro_cohorts` | Retrospective denominators from an institution's structured EHR extract (its location is the site config's `paths.retro_source_root`): trial-treatment starts and standard-of-care (SOC) treatment starts with their index dates (`pull_denominators`), the SOC trial sample (`sample_soc_trials`), note-level datasets cut before the index date with the event's own trial identifiers masked (`build_note_level_dataset`), and their serial summaries on the site's teacher server (`summarize_retro_cohorts`, `endpoints.teacher.url`). |
| `evaluation.retro_fidelity` | Distillation fidelity on those patients, with the teacher's labels as the reference: P@20 and the mean teacher score of three rankers' top-20 lists (base EmbeddingGemma 2; TrialSpace; TrialSpace reranked by the decision model), and the decision model's AUROC, AUPRC, Brier score and calibration for every question family. |
| `evaluation.dual_judge` | Two frontier LLM judges (Claude Opus through a Databricks AI Gateway and Gemini on Vertex AI; GPT on Azure before 2026-11-23) grade cohort extraction, patient summaries and the decision questions; the user adjudicates disagreements on the dashboard's Benchmarking tab. |
| `evaluation.cohort_lineitems` | One row per cohort for a table of ctgov-style listings, extracted with training's ctgov extractor (`training.trials.extract_ctgov_cohorts`), under the column names the evaluations read. |

## Retrospective fidelity

The stages (`cohorts`, `prepare`, `embed`, `retrieve`, `lists`, `label`, `decide`, `metrics`) are modules of
`evaluation.retro_fidelity`. Each writes a path-free `stage_receipt.json` last and answers `--check` (0 complete, 10
not yet, 20 begun under other settings); the orchestrator runs them in dependency order and resumes:

```bash
python -m matchminer_ai.evaluation.retro_fidelity.run --root <data root>/phi/<evaluation> \
    --trials <denominators>/soc_trial_sample.csv --summaries <denominators>/summaries/soc_validation/summaries \
    --denominator <denominators>/soc_validation/processed_soc_treatments.csv
```

Teacher servers come from the site config (`endpoints.teacher.url`, and optionally `endpoints.secondary_teacher.url`,
reached through `retro_fidelity.teacher_proxy`) or `--teacher-url` / `--secondary-teacher-url`. By default the trial
cohorts are extracted with the training run's frozen programs, so they match training's ctgov cohort tables; with
`--extraction-source SNAPSHOT` they are extracted by a code snapshot (`git archive` with a `commits.json`) instead.
Results go to `metrics.json` and `report.md`; progress to `run_status.json` and `logs/`.
`--extraction-workers N` passes `--workers N` to the extraction job (part of its identity with `--extraction-source`),
so an extraction made with other settings can be linked in.

**Full-denominator mode.** By default the queries are a seeded sample (`--patient-queries 500`, `--cohort-queries
200`). With `--all-queries --comparison-patient-queries N --comparison-cohort-queries M`, every patient and every
cohort is a query: TrialSpace's top K is reranked by the decision model (ranker (c), the full pipeline) and labeled by
the teacher for all of them, and rankers (a) and (b) run only for a seeded comparison subset of N patients and M
cohorts (`corpus/sample.json` `comparison_patient_queries` / `comparison_cohort_queries`, the same draw a default run
with those counts makes). `report.md` then has the ranker comparison ((a), (b), (c), paired differences) over the
comparison queries, a full-pipeline table of (c)'s P@20 and mean score over every query (95% CIs from bootstrapping
persons, or trials for the trial-centric workflow), and the decision-model metrics over every labeled pair, as before.
Its Design section says which queries each table covers. `--no-criteria` (or `--criteria-lines 0`) leaves out
per-criterion checking entirely: no line sample, no teacher labels or decision scores for it, and no per-criterion
rows in the metrics. Without these options a run's stages, identities and outputs are unchanged.

### On synthetic data

A synthetic run shows how the evaluation works without PHI, for the dashboard's synthetic mode. Its patients are one
generator split of the n=5000 run (`test`: the decision model never trained on them), and its cohorts are the run's
ctgov cohorts, both taken from its TrialSpace corpus instead of the extraction, patients and cohort table stages:

```bash
python -m matchminer_ai.evaluation.retro_fidelity.run --root <data root>/no_phi/benchmarks/<evaluation> \
    --synthetic-corpus <n=5000 run>/trialspace/trialspace_corpus --synthetic-split test \
    --patient-queries 100 --cohort-queries 50 --criteria-lines 1000 --embed-device cuda:0,cuda:1 --gpu 0 \
    --no-proxy --teacher-after <files that must exist first>
```

`--embed-device` encodes on GPUs instead of the CPU. `--teacher-after` holds the label jobs until other teacher work
is done (such as builds whose manifests it names). TrialSpace round 2 was fine-tuned on every synthetic patient, the
test split included, so its synthetic retrieval results are optimistic; the run's `design_notes.json` says so. The
real-patient evaluation is the meaningful one.

## Dual judge

```bash
python -m matchminer_ai.evaluation.dual_judge.build_items --run-dir RUN ...    # where the data is
python -m matchminer_ai.evaluation.dual_judge.judge_worker --run-dir RUN --judge databricks
python -m matchminer_ai.evaluation.dual_judge.judge_worker --run-dir RUN --judge gemini
python -m matchminer_ai.evaluation.dual_judge.compare --run-dir RUN          # [--judges databricks,gemini]
python -m matchminer_ai.evaluation.dual_judge.metrics --run-dir RUN
```

`compare` compares the two judges whose answers the run holds (while only one has answered, the other is the first
of Databricks and Gemini not taken, and its units wait); name them with `--judges` when a run has more. It queues
disputed and single-judge units for adjudication; `--audit-fraction` also queues a seeded sample of agreed units
(off by default). The judge,
comparison, adjudication and metrics modules need only the package and the standard library (plus `anthropic` and
`databricks-sdk` for the Databricks judge, and `google-auth` for Gemini), so they can run on another machine that can read and write the run
directory. Claude judges only through the Databricks gateway (Claude Opus 5.5): the site config's `databricks.host`
and `databricks.profile`, and `--model` or `databricks.endpoints.opus` for the endpoint name. Runs judged before
2026-11-21 through Vertex keep those answers (`responses/vertex`), which compare and metrics still read. The second
judge is Gemini 4 Neon (`--model`, default `google/gemini-4-neon`) on Vertex's OpenAI-compatible route in the site
config's `vertex.project`, with Application Default Credentials and the answer schema enforced. GPT-6.1 Sol on Azure
(`--judge azure`, the second judge until 2026-11-23) uses `--endpoint` and `--deployment` (or `AZURE_OPENAI_ENDPOINT`
and `AZURE_OPENAI_DEPLOYMENT`) with an Entra token from the Azure CLI (`--az`), refreshed every 30 minutes and on a
401 or 403; a run that holds its answers too compares the pair `--judges` names. `judge_worker --dry-run` prices the remaining
requests without sending any, and `--max-usd` caps a judge's spending. The judge prompts are `prompts/judge.*.txt`.

A synthetic judge run builds its items from a synthetic fidelity run: `build_items --synthetic-notes
<all_synthetic_notes.parquet> --synthetic-summaries <patient_summaries_with_spaces.parquet> --synthetic-prompts
<the prompts of the code snapshot that wrote the summaries> --components summary decision`. The extraction component
needs the extraction stage's per-trial checkpoints, which the n=5000 run did not keep.

Adjudication, browsing runs and rerunning compare and metrics are on the [dashboard](../apps/dashboard.md)'s
Benchmarking tab, which finds runs under the site config's `dashboard.<mode>.benchmark_roots`. Real mode serves this
PHI on loopback only, with the access token. Decisions are appended to the run's `adjudications.jsonl` (owner only)
under the server account's user name.

## Caveats

The teacher is the reference, not clinician review: these numbers measure agreement with the teacher on real
summaries, not clinical accuracy. MatchMiner-AI is a research prioritization tool; its scores are screening signals
for human review.
