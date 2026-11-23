# Site configuration

Values that belong to one installation (endpoints, cloud projects, local paths, institution-specific table maps and
names) live in a site config: a YAML file outside the repository, named by `MMAI_SITE_CONFIG` (`matchminer_ai.site`).
The two apps also take `--site-config`, which sets that variable. Nothing in the repository depends on a particular
site. Without a site config, site-only features are off and their values must be given explicitly (on the command line
or as environment variables).

Code reads the file with dotted keys:

```python
from matchminer_ai.site import load_site_config

site = load_site_config()                # empty when no site config is configured
site.get("endpoints.teacher.url")        # None when the site does not set it
site.require("endpoints.teacher.url")    # KeyError naming the key, the file and how to set it
site.section("dashboard.branding")       # a dict, empty when unset
```

## Data root

The data root holds `phi/` (real clinical data) and `no_phi/` (synthetic and public data). It comes from the apps'
`--data-root`, then `MMAI_DATA_ROOT`, then the site config's `paths.data_root`, and has no default
(`matchminer_ai.site.data_root()` raises without one). When the package runs from a source checkout, a data root or
models root inside the checkout is refused, so clinical data cannot land in the repository.

## Keys

All keys are optional. A step that needs an unset key fails with an error naming it, unless the value is given on
its command line.

### Paths, endpoints and cloud projects

| Key | Used by | Meaning |
| --- | --- | --- |
| `paths.data_root` | everything that reads data | The data root, when neither `--data-root` nor `MMAI_DATA_ROOT` is set |
| `paths.models_root` | `training.from_summaries` | Where trained models go (default `<data root>/../models`) |
| `paths.hf_cache` | `evaluation.retro_fidelity` | Hugging Face cache for the teacher tokenizer (`--cache-dir` overrides it) |
| `paths.trialspace_model` | `embed_for_matching` and everything that embeds with TrialSpace, including the dashboard's guideline features | A local TrialSpace checkpoint, used when the configuration's `embedding.model_path` is unset (otherwise the published `ksg-dfci/TrialSpace-1226`) |
| `paths.embeddinggemma2_cache` | `evaluation.retro_fidelity` | A cache holding the pinned EmbeddingGemma 2 snapshot (default: the Hugging Face cache) |
| `paths.retro_source_root` | `evaluation.retro_cohorts` | The institution's structured EHR extract (`structured_data/`, `derived_data/`) |
| `paths.phi_roots` | `evaluation.retro_cohorts.build_note_level_dataset` | Directories besides `<data root>/phi` where note text may be written |
| `endpoints.teacher.url` | `evaluation.retro_fidelity`, `evaluation.retro_cohorts.summarize_retro_cohorts`, the dashboard | The on-premises OpenAI-compatible teacher server (`/v1`), approved for PHI. The dashboard offers it as its OpenAI-compatible endpoint |
| `endpoints.secondary_teacher.url` | `evaluation.retro_fidelity` | An optional second teacher, reached through `teacher_proxy`; unset labels on the first teacher only |
| `vertex.project` | the apps' Vertex profiles, `evaluation.dual_judge.judge_worker` | Google Cloud project of the Vertex (Agent Platform) endpoints, and of the Claude judge on Vertex |
| `databricks.host` | the dashboard's model connections, `provider: databricks_anthropic`, `evaluation.dual_judge.judge_worker --judge databricks` | Workspace URL whose AI Gateway serves Claude |
| `databricks.profile` | the same | Databricks CLI login profile (default `unity-gateway`) |
| `databricks.endpoints.<name>` | the dashboard (`haiku`: its default model for every LLM task, except query writing, `sonnet`, and the chat assistant, `opus`; the others selectable), the Databricks judge (`opus`) | Full three-part gateway endpoint names of the Claude models (`haiku`, `sonnet`, `opus`) |

Without `vertex.project` the Vertex profiles have no project, and without `databricks.host` the Claude option has no
gateway.

### `dashboard`

Read by the [dashboard](../apps/dashboard.md). The command line and `MMAI_DASH_*` variables override the paths. A
mode's own data sit under `dashboard.real` and `dashboard.synthetic` (no shared fallback, so one mode never reads the
other's folders); shared values sit directly under `dashboard`.

| Key | Default | Meaning |
| --- | --- | --- |
| `real.trial_data` | `<data root>/phi/real_time/trial_data` | The trial extraction run real mode ingests |
| `real.oa_db` | `<data root>/phi/real_time/oa_db_*.duckdb` | The DuckDB copy holding the trial table (a glob; the newest file is used) |
| `real.store_dir`, `synthetic.store_dir` | `<data root>/phi/real_time/dashboard`, `<data root>/no_phi/dashboard_synthetic` | Each mode's store, vectors and caches |
| `real.good_options_catalog`, `synthetic.good_options_catalog` | real: `<data root>/phi/real_time/good_options_catalog/catalog`; synthetic: none | The GoodOption catalog of the trials' drugs |
| `real.population_duckdb` | `<data root>/phi/guideline_fetcher_population.duckdb` | Real mode's population database |
| `real.population_cache` | beside the population database | The preprocessed real population (the patients of real mode) |
| `real.duckdb_roots` | the data root | Folders real mode may browse for DuckDB files |
| `real.extraction_records`, `synthetic.extraction_records` | beside the population database; `<data root>/no_phi/guideline_fetcher_extractions/synthetic` | Extraction records: the Patient tab's Structured data, extraction-based workup checks, and criteria screening |
| `synthetic.population_summaries`, `synthetic.population_notes` | `<data root>/no_phi/patient_summaries.parquet`, `<data root>/no_phi/all_synthetic_notes.parquet` | Synthetic mode's population (Population management samples it) |
| `synthetic.population_split` | none (every patient) | The split the synthetic sample is drawn from, when the summaries have a `split` column; a sample with **Include training patients** draws from every split |
| `synthetic.population_key` | `pseudo_mrn` | The synthetic tables' patient key; `patient_id` when it is the patient's own ID (the patient corpus's), not a generation-prompt ID |
| `real.benchmark_roots`, `synthetic.benchmark_roots` | none | Folders searched for evaluation runs (Benchmarking tab); synthetic mode's never under `phi` |
| `synthetic_run` | `<data root>/no_phi/synthetic_run` | TrialSpace rounds, the decision model, synthetic mode's trials and patients |
| `guideline_catalogs` | `<data root>/no_phi/space_paradigms_nccn` | Finalized guideline catalogs |
| `grouping` | the bundled example rules | Trial-group rules per center |
| `real.trial_management`, `synthetic.trial_management` | `<data root>/phi/real_time/dashboard/trial_management.json`, `<data root>/no_phi/dashboard_synthetic/trial_management.json` | The file of each mode's slots, edited criteria, subcohorts and group placements (Clinical trial management's Manage a trial), shared by the mode's users; synthetic mode's never under `phi` |
| `embedding_python` | the server's own Python | Python that runs the query encoder |
| `embedding_caches` | the standard Hugging Face cache | Caches searched for the encoder's weights |
| `trial_table.name` | `trials` | The trial table in the DuckDB copy |
| `trial_table.columns` | each field's own name | Map from a dashboard field to the table's column. Fields: `nct_id`, `protocol_no`, `short_title`, `long_title`, `management_group`, `contact_name`, `contact_email`, `contact_role`, `site_pi`, `overall_pi`, `status`, `phase`, `brief_summary`, `site_list`, `ct_response` |
| `site_names` | none | Map from site names in the table's `site_list` to the short names shown; with a map, only mapped sites are shown |
| `center_prefix` | none | Regular expression removed from center names in the sidebar |
| `higher_threshold_groups` | none | Coordinating centers left out of the Patient tab's trials unless included |
| `branding.institution` | none | The site's name |
| `branding.confidentiality` | none | Confidentiality notice, shown in real mode only |
| `branding.pi_label` | `Site PI` | Label of the site PI |
| `branding.higher_threshold_label` | `Exclude higher-threshold programs` | Label of the `higher_threshold_groups` filter |
| `branding.institutions` | none | Sidebar groups of the disease centers: `label` and name `prefixes` |
| `branding.trial_links` | none | Links from a trial's protocol number: `label` and a `url` containing `{protocol_no}` |

## Example

Placeholder values only:

```yaml
paths:
  data_root: /srv/mmai/data            # holds phi/ and no_phi/
  models_root: /srv/mmai/models
  trialspace_model: /srv/mmai/models/trialspace   # when embedding.model_path is unset
  hf_cache: /srv/models/hf
  retro_source_root: /srv/ehr/extract_2026

endpoints:
  teacher:
    url: http://gpu-server.example:8000/v1
  secondary_teacher:
    url: http://gpu-server-2.example:8000/v1

vertex:
  project: my-gcp-project

databricks:
  host: https://workspace.example.cloud.databricks.com
  profile: unity-gateway
  endpoints:
    haiku: catalog.schema.example-claude-haiku-5-5
    sonnet: catalog.schema.example-claude-sonnet-5-5
    opus: catalog.schema.example-claude-opus-5-5

dashboard:
  real:
    trial_data: /srv/mmai/data/phi/real_time/trial_data
    oa_db: /srv/mmai/data/phi/real_time/trials_*.duckdb
    store_dir: /srv/mmai/data/phi/real_time/dashboard
    good_options_catalog: /srv/mmai/data/phi/real_time/good_options_catalog/catalog
    population_duckdb: /srv/mmai/data/phi/population.duckdb
    duckdb_roots: [/srv/mmai/data]
    benchmark_roots: [/srv/mmai/data/phi]
    trial_management: /srv/mmai/data/phi/real_time/dashboard/trial_management.json
  synthetic:
    store_dir: /srv/mmai/data/no_phi/dashboard_synthetic
    benchmark_roots: [/srv/mmai/data/no_phi/benchmarks]
  synthetic_run: /srv/mmai/data/no_phi/synthetic_run
  guideline_catalogs: /srv/mmai/data/no_phi/space_paradigms_nccn
  grouping: /srv/mmai/site/grouping.yaml
  trial_table:
    name: trials
    columns:
      management_group: managing_group
      site_pi: local_pi
  site_names:
    Example Cancer Center: ECC Main
  center_prefix: '^ECC/\s*'
  higher_threshold_groups: [ECC Phase I]
  branding:
    institution: Example Cancer Center
    confidentiality: The information shown is confidential.
    pi_label: ECC PI
    institutions:
      - {label: ECC, prefixes: [ECC/]}
    trial_links:
      - {label: Portal, url: 'https://portal.example.org/protocols/{protocol_no}'}
```
