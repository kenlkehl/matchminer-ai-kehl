# Hugging Face Space

[`ksg-dfci/matchminer-ai-1226`](https://huggingface.co/spaces/ksg-dfci/matchminer-ai-1226) runs the
[dashboard](dashboard.md) in synthetic mode as a public Docker Space, on one RTX PRO 6000 (96 GB, Blackwell) that
sleeps after five hours without visitors. Everything it shows comes from the Hub: the trials, patients, corpus and
guideline catalogs of [mmai-synthetic-1226](https://huggingface.co/datasets/ksg-dfci/mmai-synthetic-1226),
TrialSpace-1226 and MatchEval-1226 at the preset's pinned revisions, and NVIDIA's Nemotron 3.5 Super NVFP4 (the
training teacher) for every LLM task, served by vLLM on the Space's GPU. On a GPU under 90 GB (an L40S, say) the
same image serves Nemotron 3.5 Bolt's GGUF with llama.cpp instead. The only other model connection it accepts is
[OpenRouter](dashboard.md#openrouter) with a visitor's own API key (`allow_openrouter`), at the visitor's own risk:
while the local server loads, or on hardware without a suitable GPU. It needs no sign-in and has no Benchmarking tab.

## Files

The Space's files are `deploy/hf_space/space/` in this repository:

| File | Contents |
| --- | --- |
| `README.md` | The Space card, with its metadata: Docker SDK, port 7860, suggested hardware, startup timeout, models and dataset |
| `Dockerfile` | Four stages: the source at `MMAI_COMMIT` (from GitHub), its React frontend, `llama-server` built for compute capabilities 8.9 and 12.0 (a pinned llama.cpp commit), and the Python runtime (CUDA runtime image with a C/C++ compiler for vLLM's kernel builds, Python 3.13, vLLM 0.33 and the package with `apps` and `sql`) |
| `constraints.txt` | The pinned versions of the heavy dependencies (torch, vLLM, transformers, sentence-transformers, huggingface_hub, ...), installed in their own layer before the package |
| `site.yaml` | The Space's site config: the published data as `hf://` references at `${MMAI_DATASET_REVISION:-main}`, the local model server, framing by the Space page |
| `entrypoint.sh` | `mmai-dashboard --mode synthetic --port 7860 --allow-downloads serve`, with the data under `/data` when a storage bucket is mounted there; it logs the GPU and torch's view of it, falls back to Bolt on a GPU under 90 GB, and starts `prefetch.py` |
| `prefetch.py` | Downloads the local model server's weights while the app starts (the files the server would download itself) |

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
- `dashboard.synthetic.local_server`: the only model connection, started once the decision model has loaded: vLLM
  serving Nemotron 3.5 Super NVFP4 (below).
- `dashboard.synthetic.frame_ancestors`: `https://huggingface.co`, so the Space page can show the app.

Clinical trial management's file (`no_phi/dashboard_trial_management/` under the data root) starts empty and is shared
by everyone using the Space.

## The LLM server

Until the local server is ready (or on hardware without a GPU), Data and models offers OpenRouter: a visitor pastes
their own key, which the app keeps only in memory for that browser session and sends only to OpenRouter, and picks a
model (by default `nvidia/nemotron-3.5-super-vl`, the same Nemotron 3.5 Super). Nothing else is accepted.

`site.yaml`'s `local_server.autostart` is a full vLLM spec, so it needs no recipe in the package:
- **Model:** `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at a pinned revision, text only
  (`--language-model-only`), served under its own ID. So the package's Super profile applies: the `nemotron_v3`
  reasoning parser, the `qwen3_coder` tool parser and NVIDIA's sampling, as for the training teacher.
- **Flags:** those of NVIDIA's single-GPU recipes on the model card and of the production servers: remote code,
  prefix caching, the hybrid Mamba cache settings, MTP speculative decoding with 5 tokens (the card's RTX PRO 6000
  setting), 8,192 batched tokens, blocks of 64, asynchronous scheduling.
- **Memory (96 GB, about 95 GiB usable):**

  | | GiB |
  | --- | --- |
  | vLLM's share (`gpu_memory_utilization` 0.84) | ~80 |
  | — Super's weights (the language model) | ~70 |
  | — activations, CUDA graphs | ~3 |
  | — KV and Mamba caches | ~7 |
  | The app: decision model (~6), and TrialSpace when guideline or population retrieval runs (up to ~12 in all) | ~15 left |

  The KV cache is small for this hybrid model: 8 of its 88 layers are attention layers with 2 KV heads, about 9 KB
  per token (2.25 GiB for one 262,144-token context), and each sequence's Mamba state is about 80 MiB. So
  `max_model_len` 262,144 and `max_num_seqs` 8 fit, with the default (bf16) KV cache.
- **Engine:** vLLM 0.33.0, whose wheels carry SM 12.0 kernels (the card lists vLLM on one RTX PRO 6000), with torch
  2.14 (CUDA 13: host driver 580 or later; `entrypoint.sh` logs torch's view of the GPU). FlashInfer's precompiled
  kernels for SM 12.0 are installed (`flashinfer-cubin`, and `flashinfer-jit-cache` with its `sm120f` provider), so
  vLLM loads them instead of compiling them. The one FlashInfer kernel the jit-cache cannot hold, the Mamba state
  update with site.yaml's flashinfer Mamba backend (fp16 state cache, Philox-5 stochastic rounding, 6 tokens per
  step), is compiled during the image build (about 40 s on 8 cores) into FlashInfer's cache, with the settings vLLM's
  run-time build uses, so a cold start loads it. Anything still compiled at start uses the pip CUDA toolkit, which
  the local server finds by itself:
  - `constraints.txt` pins its compiler to torch's CUDA 13.0, since a newer nvcc than the 13.0 headers fails CCCL's
    check ("CUDA compiler and CUDA toolkit headers are incompatible");
  - the Dockerfile gives it the layout FlashInfer's and nvcc's links expect: `lib64` (a link to the wheels' `lib`),
    unversioned names such as `libcudart.so` beside the versioned libraries, and the CUDA toolkit's driver stub as
    `lib/stubs/libcuda.so` (only linked against; the host driver's `libcuda.so.1` is loaded at run time).

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

1. Download the dataset's files (about 15 GB, mostly the notes) and the models into the Hugging Face cache. Super's
   weights (75 GB) download at the same time (`prefetch.py`), over Xet's high-performance mode without its chunk
   cache.
2. Build the run folder and, unless the dataset publishes the vectors, embed about 88,000 patients and 15,000
   cohorts with TrialSpace on the GPU (about 14 minutes on a shared H100). With the published vectors the folder takes
   under a minute.
3. Load the decision model and build the trial store; then serve. The local model server then starts vLLM: loading
   about 70 GiB of weights, compiling kernels and capturing CUDA graphs (nothing is cached between starts); Data and
   models shows its progress, and the LLM tasks wait for it.
4. Embed the trial cohorts that have no TrialSpace vector of the round, in the background. The run's own 5,000 trials
   take theirs from the round, so nothing is embedded now; another trial set is embedded at every start (its vectors
   live in the data root), by the query encoder on `--embedding-device` (`MMAI_DASH_EMBEDDING_DEVICE`; CPU by default,
   far too slow for tens of thousands of cohorts, while the GPU must leave vLLM's memory free).

Expected cold start, not measured (camus has no Blackwell GPU): the app serves after about 5–10 minutes, mostly
downloads at the Hub's internal bandwidth, and the LLM is ready after about 15–30 minutes. That is why the Space
sleeps only after five hours without visitors. The card's `startup_duration_timeout` allows for this. With a storage bucket mounted at `/data`
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
`Qwen/Qwen3-0.6B` with a small `gpu_memory_utilization`) in `local_server.autostart` checks the vLLM path on another
GPU; Super itself needs about 80 GiB on a Blackwell GPU.

## Licenses

The code, TrialSpace-1226 and the dataset are CC BY-NC-ND 4.0; MatchEval-1226 is under the LFM Open License v1.0;
Nemotron 3.5 Super and Bolt (with Bolt's draft model) are under OpenMDW-1.1, which permits use and redistribution with
the license kept. The Space card lists them.
