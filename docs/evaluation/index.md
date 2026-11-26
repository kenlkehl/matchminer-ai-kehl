# Evaluations

`matchminer_ai.evaluation` holds the evaluations of the pipeline on real patients. Their inputs and outputs are
clinical data (PHI): they read and write only under the data root's `phi/` tree (or a PHI directory the site config
names, `paths.phi_roots`), create files 600 and directories 700, and log counts only. Run them only where that data is
authorized, with LLM endpoints approved for it (the site config's `endpoints.*`).

Install with the evaluation extra: `uv pip install -e ".[training,evaluation]"`.

| Module | What it does |
| --- | --- |
| `evaluation.retro_cohorts` | Retrospective denominators from an institution's structured EHR extract (its location is the site config's `paths.retro_source_root`): trial-treatment starts and standard-of-care (SOC) treatment starts with their index dates (`pull_denominators`), the SOC trial sample (`sample_soc_trials`), note-level datasets cut before the index date with the event's own trial identifiers masked (`build_note_level_dataset`), and their serial summaries on the site's teacher server (`summarize_retro_cohorts`, `endpoints.teacher.url`). |
| `evaluation.retro_fidelity` | Distillation fidelity on those patients, with the teacher's labels as the reference: P@20 and the mean teacher score of four rankers' top-20 lists (base EmbeddingGemma 2; TrialSpace; TrialSpace reranked by the decision model's P(reasonable); TrialSpace reranked by the app's composite match score), and the decision model's AUROC, AUPRC, Brier score and calibration for every question family, pooled and per ranker. |
| `evaluation.demographics` | Table 1 and fidelity metrics by demographic group for a real fidelity run (the dashboard's PDF export): links the run's patients to the EHR extract's registration and treatment plan tables, and suppresses small cells. |
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

**Rankers.** Every run compares four rankers, each a top-20 list per query (`lists.py`; stages `lists_ab`,
`lists_c`, `lists_d`):

| | Ranker | List |
| --- | --- | --- |
| A | `embeddinggemma2` | base EmbeddingGemma 2, cosine top 20 |
| B | `trialspace_round2` | TrialSpace round 2, cosine top 20 |
| C | `trialspace_round2_decision` | TrialSpace's top K (`--k`, 100) reranked by the decision model's P(reasonable) |
| D | `trialspace_round2_composite` | the same top K reranked by the composite match score, as the app ranks |

D exists because C is not the ordering users see: `score_match_quality` and the dashboard rank pairs by
`match_quality_score` = P(reasonable) x (1 + P(cancer type) + P(burden) + P(treatment history) + 2 P(biomarker)) / 6,
the expected fraction of the teacher's 0-6 score (`matching.rerank`; `lists.py` calls the same formula). C and D
rerank the same candidates, ties by TrialSpace rank, and both read the decision model's match scores, so D needs no
new decision-model work. Each list stage adds the pairs the earlier ones did not have (`union_c_extra.jsonl`,
`union_d_extra.jsonl`), and only those get new teacher labels (`labels/match_d`, `labels/boilerplate_d`); the
boilerplate decisions cover the union of all four (`lists_d/union.jsonl`). D has no per-criterion strata. In
full-denominator mode C and D rank every query and A and B the comparison subset (all of them with `--compare-all`).
`--reuse-run` reuses an earlier run's decision scores and teacher labels for all four, so only new pairs are labeled.
The retrieval tables and paired differences (D-C, D-B, D-A beside B-A, C-B, C-A) include D; the full-pipeline table
over every query stays C's.

**By ranker denominator (metrics v4, 2026-11-26; D since v5).** Besides the merged union of the rankers' top-20 lists,
the metrics stage scores the decision model against the teacher on each ranker's own lists:
`decision.by_ranker.<ranker>` in `metrics.json` (A to D), with `pooled`,
`patient_centric` and `trial_centric` tables of the five match questions and boilerplate (no per-criterion rows), the
union tables' fields and clusters, and `counts`: list rows, unique pairs, pairs with match labels, and the boilerplate
combinations among the ranker's pairs, those scored and those labeled. `report.md` has a section per ranker. A run
with `--no-new-boilerplate` has boilerplate labels only for the combinations an earlier run labeled, so a ranker's
boilerplate rows can cover only part of its combinations; the counts give that coverage. To add D or these tables
to a finished run, run `lists.py d` (and, for D's boilerplate rows, `decide.py boilerplate` over `lists_d/union.jsonl`
with the run's own `--reuse`), label `union_d_extra.jsonl` like `labels/match_c`, move `metrics/` aside and rerun the
metrics stage with `--lists lists_d` and the new label directory beside its receipt's other arguments
(`provenance.argv`).

### Demographics

`evaluation.demographics.run_demographics(run_dir, extract_root)` gives a real fidelity run's Table 1 and its
headline metrics by demographic group, aggregates only, with small cells suppressed (the dashboard's Export results
draws them into the PDF):

- **Linkage.** The run's patients are its corpus's index events. The patients stage's receipt names the denominator
  CSV it read (checked against the receipt's size and mtime stamp), which gives each event's patient, index date (the
  treatment plan's start) and plan. From the extract's `structured_data/` it reads selected columns of `PT_INFO.csv`
  (birth date, sex), `PT_DEMOGRAPHICS.csv` (race, ethnicity) and `TREATMENT_PLAN.csv` (the index plan's ICD-10
  codes). The categories are v22's (`real_eval_code/eval_phi_*/demographics_utils.py`): age at treatment start in
  `<40, 40-49, 50-59, 60-69, 70-79, 80+, Unknown`; race from the derived race (several races: Multiple); ethnicity
  from the Hispanic indicator; sex from the gender, else the sex at birth. Cancer type groups the plan's ICD-10 codes
  by site, a primary cancer code first.
- **Table 1.** A patients column (race, ethnicity and sex; age and cancer type at each patient's first treatment start
  in the eval) and a treatment-starts column (every index event), with the median (IQR) age.
- **By group.** For age group, race, ethnicity and sex: the full pipeline's (ranker (c)) P@20 and mean teacher score
  over the group's patient-centric queries, and the decision model's reasonableness AUROC over the teacher-labeled
  pairs of the group's patients (both workflows pooled), with person-cluster bootstrap 95% intervals. Trial-centric
  queries are cohorts, so trial-centric retrieval is not split by group.
- **Small cells.** Counts of 1 to 10 show as <11 without a percentage; zero shows as 0; totals are exact. Within each
  section of a column, larger counts are also hidden (*), smallest first, until no hidden count can be worked out
  from the total: at least two are hidden, they add up to 11 or more, and their sum has more than one split. A
  group's metrics are hidden (n<11) when it has fewer than 11 patients.
- **Cache.** The unsuppressed aggregates are cached in the run's `reports/demographics.json` (file 600, directory 700,
  inside the run's own PHI directory) and reused while the run's metrics, its denominator file and the three extract
  tables are unchanged; suppression is applied each time the tables are read.

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
