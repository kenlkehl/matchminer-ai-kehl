# Hugging Face Space

[`ksg-dfci/matchminer-ai-1226`](https://huggingface.co/spaces/ksg-dfci/matchminer-ai-1226) runs the
[dashboard](dashboard.md) in synthetic mode as a public Docker Space, on one L40S (48 GB) that sleeps after an hour
without visitors. Everything it shows comes from the Hub: the trials, patients, corpus and guideline catalogs of
[mmai-synthetic-1226](https://huggingface.co/datasets/ksg-dfci/mmai-synthetic-1226), TrialSpace-1226 and
MatchEval-1226 at the preset's pinned revisions, and NVIDIA's Nemotron 3.5 Bolt GGUF for every LLM task, run by
llama.cpp on the Space's GPU. It reaches no other model service, needs no sign-in and has no Benchmarking tab.

## Files

The Space's files are `deploy/hf_space/space/` in this repository:

| File | Contents |
| --- | --- |
| `README.md` | The Space card, with its metadata: Docker SDK, port 7860, suggested hardware, startup timeout, models and dataset |
| `Dockerfile` | Four stages: the source at `MMAI_COMMIT` (from GitHub), its React frontend, `llama-server` built for compute capability 8.9 (a pinned llama.cpp commit), and the Python runtime (CUDA runtime image, Python 3.13, the package with `apps` and `sql`) |
| `constraints.txt` | The pinned versions of the heavy dependencies (torch, vLLM, transformers, sentence-transformers, huggingface_hub, ...), installed in their own layer before the package |
| `site.yaml` | The Space's site config: the published data as `hf://` references at `${MMAI_DATASET_REVISION:-main}`, the local model server, framing by the Space page |
| `entrypoint.sh` | `mmai-dashboard --mode synthetic --port 7860 --allow-downloads serve`, with the data under `/data` when a storage bucket is mounted there |

`MMAI_COMMIT`, the commit whose code the image installs, is not in the repository: `publish.py` writes it.
`deploy/hf_space/refresh.py` restarts the Space after a dataset update (below).

## Publishing

`deploy/hf_space/publish.py` uploads `space/` and `MMAI_COMMIT` to the Space repository in one commit, which
rebuilds the Space. Docker's layer cache keeps llama.cpp and the pinned dependencies, so a code update rebuilds only
the source, frontend and package layers. The commit must be on GitHub, since the image downloads its code there.

```bash
python deploy/hf_space/publish.py                    # this checkout's HEAD (pushed, no uncommitted changes)
python deploy/hf_space/publish.py --commit <sha>
python deploy/hf_space/publish.py --create           # once: create the public Space with its GPU and sleep time
python deploy/hf_space/publish.py --factory-rebuild  # also rebuild without the layer cache
python deploy/hf_space/publish.py --dry-run          # stage only
```

**Automatic rebuilds.** The workflow `.github/workflows/hf-space.yml` runs `publish.py --commit $GITHUB_SHA` on every
push to `main` that changes `src/`, `frontend/dashboard/`, `docs/` (the How this works tab serves its pages),
`pyproject.toml` or `deploy/hf_space/`, and on demand (Actions, **Run workflow**, optionally a factory rebuild). It
needs the repository secret `HF_TOKEN`: a Hugging Face token with write access to the Space.

## Data and models

`site.yaml` names every input (see [site configuration](../reference/site-config.md#hugging-face-hub-files)), all of
the dataset at one revision: `${MMAI_DATASET_REVISION:-main}`, the Space variable `MMAI_DATASET_REVISION` when it is
set, else the dataset's main branch. So a dataset update reaches the Space with a restart and no code change; to hold
the Space at a dataset commit, set the variable (`refresh.py --pin <commit>`; `--unpin` follows main again). A file or
folder the revision does not have reads as a missing path, so that feature is unavailable rather than an error.

- `dashboard.synthetic_run`: the published run, `hf://datasets/ksg-dfci/mmai-synthetic-1226@<commit>`. Before
  serving, the app builds its run folder from the dataset (`apps.dashboard.hub_run`): it decompresses the corpus,
  converts the two trial tables, links the patient metadata and notes, and adds round 3's TrialSpace vectors. Those
  are the dataset's own when it publishes them (`trialspace/trialspace_round3/mining/{patients,cohorts}.npy` and the
  mining `manifest.json`, checked against the corpus rows); otherwise the app embeds the corpus with the preset's
  pinned TrialSpace-1226 (the model that made them). The decision model is the preset's pinned MatchEval-1226.
- `dashboard.synthetic.trial_cohorts` and `trial_tables`: the trials the Space shows, the dataset's cohort table
  (`trials/retrieval_cohorts.parquet`) and trial tables, now the run's 5,000 trials. Another trial set in the dataset
  (a fresh ClinicalTrials.gov extraction in its own folder, in the same table form) is a change of these lines and a
  restart. The patients stay the run's.
- `dashboard.synthetic.population_summaries`, `population_notes`, `extraction_records`, `good_options_catalog`: files
  and folders of the dataset. A new extraction store in another folder is a change of that line.
- `dashboard.guideline_catalogs`: the dataset's `guideline_catalogs/pdq/`, the NCI PDQ catalogs, shown in the
  Patient-centric Guidelines sub-tab and Guideline concordance reporting. While the dataset has none, both are
  hidden. Catalogs derived from licensed guidelines (NCCN) must never be published there.
- `dashboard.synthetic.local_server`: the only model connection, started at boot: Bolt with llama.cpp on GPU 0, its
  recipe chosen by the GPU's free memory (`presets/local_servers.yaml`). Bolt's GGUF and its draft model download on
  first start.
- `dashboard.synthetic.frame_ancestors`: `https://huggingface.co`, so the Space page can show the app.

Clinical trial management's file (`no_phi/dashboard_trial_management/` under the data root) starts empty and is shared
by everyone using the Space.

## After a dataset update

A new dataset commit (new guideline catalogs, published vectors, a new extraction store) needs only a restart, no
rebuild:

```bash
python deploy/hf_space/refresh.py --dry-run   # the dataset's main, and the catalogs the Space would load
python deploy/hf_space/refresh.py --wait      # restart, and wait until it runs again
```

It warns when `MMAI_DATASET_REVISION` pins another commit. The token is `HF_TOKEN` (or the huggingface_hub login), as
for `publish.py`. A restart repeats the startup below.

## Startup

A Space has no persistent disk unless a storage bucket is mounted, so every start after sleep repeats the work:

1. Download the dataset's files (about 15 GB, mostly the notes) and the models into the Hugging Face cache.
2. Build the run folder and, unless the dataset publishes the vectors, embed about 88,000 patients and 15,000
   cohorts with TrialSpace on the GPU: by far the longest step (about 14 minutes on a shared H100, so expect several
   times that on an L40S). With published vectors the folder takes under a minute.
3. Load the decision model and build the trial store; then serve. The local model server downloads Bolt (about
   23 GB) and starts in the background; Data and models shows its progress.
4. Embed the trial cohorts that have no TrialSpace vector of the round, in the background. The run's own 5,000 trials
   take theirs from the round, so nothing is embedded now; another trial set is embedded at every start (its vectors
   live in the data root), by the query encoder on `--embedding-device` (`MMAI_DASH_EMBEDDING_DEVICE`; CPU by default,
   far too slow for tens of thousands of cohorts, while the GPU must leave Bolt's memory free).

The card's `startup_duration_timeout` allows for this. With a storage bucket mounted at `/data`
(`HfApi.set_space_volumes`), `entrypoint.sh` keeps the data root and the Hugging Face cache there, and a restart
skips steps 1 and 2.

## Testing without Docker

The image is built only on the Hub. The entrypoint's logic runs anywhere with the package installed: copy `site.yaml`,
point `paths.data_root` at a scratch folder, and run

```bash
HF_HUB_CACHE=<cache> MMAI_SITE_CONFIG=<site.yaml> MMAI_LLAMA_SERVER=<llama-server> \
  mmai-dashboard --mode synthetic --port <port> --allow-downloads --device cuda:0 serve
```

`mmai-dashboard --mode synthetic prepare` stops after the run folder is built. A smaller local model (for example
`Qwen/Qwen3-0.6B`) in `local_server.autostart` checks the server path on a smaller GPU.

## Licenses

The code, TrialSpace-1226 and the dataset are CC BY-NC-ND 4.0; MatchEval-1226 is under the LFM Open License v1.0;
Nemotron 3.5 Bolt and its draft model are under OpenMDW-1.1, which permits use and redistribution with the license
kept. The Space card lists them.
