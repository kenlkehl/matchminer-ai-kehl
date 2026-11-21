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
| `evaluation.dual_judge` | Two frontier LLM judges (Claude on Vertex AI or a Databricks AI Gateway, GPT on Azure) grade cohort extraction, patient summaries and the decision questions; the user adjudicates disagreements on the dashboard's Benchmarking tab. |
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
python -m matchminer_ai.evaluation.dual_judge.judge_worker --run-dir RUN --judge vertex       # or --judge databricks
python -m matchminer_ai.evaluation.dual_judge.judge_worker --run-dir RUN --judge azure
python -m matchminer_ai.evaluation.dual_judge.compare --run-dir RUN          # [--judges vertex,databricks]
python -m matchminer_ai.evaluation.dual_judge.metrics --run-dir RUN
```

`compare` compares the two judges whose answers the run holds (while only one has answered, the other is the first
of Vertex and Azure not taken, and its units wait); name them with `--judges` when a run has more. The judge,
comparison, adjudication and metrics modules need only the package and the standard library (plus
`anthropic[vertex]` for the Vertex judge), so they can run on another machine that can read and write the run
directory. The Vertex project is `--project`, `ANTHROPIC_VERTEX_PROJECT_ID` or the site config's `vertex.project`.
The `databricks` judge sends the same requests to Claude Opus through a Databricks AI Gateway: the site config's
`databricks.host` and `databricks.profile`, and `--model` or `databricks.endpoints.opus` for the endpoint name.
Azure uses `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT`. `judge_worker --dry-run` prices the remaining
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
