# Hugging Face Space

Two Spaces run the [dashboard](dashboard.md) in synthetic mode from the same files and commit. The public one,
[`ksg-dfci/matchminer-ai-1226`](https://huggingface.co/spaces/ksg-dfci/matchminer-ai-1226), is
[the lite Space](#the-lite-space): a T4 with no LLM of its own. `ksg-dfci/matchminer-ai-1226-rtxpro` (named
`matchminer-ai-1226` until 2026-11-27, when the lite Space took the name to save cost) is private and paused; it
runs as a Docker Space on one RTX PRO 6000 (96 GB, Blackwell) that
sleeps after five hours without visitors. Everything it shows comes from the Hub: the trials, patients, corpus and
guideline catalogs of [mmai-synthetic-1226](https://huggingface.co/datasets/ksg-dfci/mmai-synthetic-1226),
TrialSpace-1226 and MatchEval-1226 at the preset's pinned revisions, and NVIDIA's Nemotron 3.5 Super NVFP4 (the
training teacher) for every LLM task, served by vLLM on the Space's GPU. On a GPU under 90 GB (an L40S, say) the
same image serves Nemotron 3.5 Bolt's GGUF with llama.cpp instead. The only other model connection it accepts is
[OpenRouter](dashboard.md#openrouter) with a visitor's own API key (`allow_openrouter`), at the visitor's own risk:
while the local server loads, or on hardware without a suitable GPU. It needs no sign-in and has no Benchmarking tab.
The lite Space runs the same image on a T4, with OpenRouter as its only LLM connection: the
Space's own key for one free model (shared quota), or a visitor's own key.

## Files

The Space's files are `deploy/hf_space/space/` in this repository:

| File | Contents |
| --- | --- |
| `README.md` | The Space card, with its metadata: Docker SDK, port 7860, suggested hardware, startup timeout, models and dataset |
| `Dockerfile` | The source at `MMAI_COMMIT` (from GitHub), its React frontend, `llama-server` built for compute capabilities 8.9 and 12.0 (a pinned llama.cpp commit), and the Python runtime (Python 3.13 with a C/C++ compiler and the package with `apps` and `sql`), in two variants (`MMAI_VARIANT`): `full` adds vLLM 0.33, FlashInfer's kernels and `llama-server` on the CUDA runtime image; `lite` (the [lite Space](#the-lite-space)) is the app alone on plain Ubuntu |
| `constraints.txt` | The pinned versions of the heavy dependencies both variants share (torch, transformers, sentence-transformers, huggingface_hub, ...), installed in their own layer before the package |
| `constraints-engines.txt` | The full variant's: vLLM, flashinfer-python and the CUDA 13.0 compiler pins |
| `no-engines.txt` | The lite variant's uv override that leaves the package's vLLM dependency out |
| `site.yaml` | The Space's site config: the published data as `hf://` references at `${MMAI_DATASET_REVISION:-main}`, the local model server, the hosted OpenRouter key's limits, framing by the Space page |
| `entrypoint.sh` | `mmai-dashboard --mode synthetic --port 7860 --allow-downloads serve`, with the data under `/data` when a storage bucket is mounted there; it logs the GPU, sets the tier (`tier.py`) and, with a local server, starts `prefetch.py` (both without the hosted OpenRouter key in their environment) |
| `tier.py` | The tier by the GPU's memory: Super with vLLM from 90 GB, Bolt with llama.cpp from 40 GB, else no local model server ([the lite Space](#the-lite-space)) |
| `prefetch.py` | Downloads the local model server's weights while the app starts (the files the server would download itself) |

`MMAI_COMMIT`, the commit whose code the image installs, is not in the repository: `publish.py` writes it. A Space
may have its own card in `deploy/hf_space/cards/`, which `publish.py` uploads as its `README.md`.
`deploy/hf_space/refresh.py` restarts the Space after a dataset update (below).

## Publishing

`deploy/hf_space/publish.py` uploads `space/` and `MMAI_COMMIT` to a Space repository in one commit, which
rebuilds the Space. Its `SPACES` lists the Spaces with their hardware: `ksg-dfci/matchminer-ai-1226` (the default,
the lite Space on a T4, with its own card and image variant) and `ksg-dfci/matchminer-ai-1226-rtxpro` (an RTX PRO
6000); `--space` picks one. Docker's layer cache keeps llama.cpp and the pinned dependencies, so a code update rebuilds only
the source, frontend and package layers. The commit must be on GitHub, since the image downloads its code there.

```bash
python deploy/hf_space/publish.py                    # this checkout's HEAD (pushed, no uncommitted changes)
python deploy/hf_space/publish.py --commit <sha>
python deploy/hf_space/publish.py --create           # once: create the public Space with its GPU and sleep time
python deploy/hf_space/publish.py --factory-rebuild  # also rebuild without the layer cache
python deploy/hf_space/publish.py --dry-run          # stage only
python deploy/hf_space/publish.py --space ksg-dfci/matchminer-ai-1226-rtxpro   # the RTX PRO Space (likewise --create)
```

**Automatic rebuilds.** The workflow `.github/workflows/hf-space.yml` runs `publish.py --commit $GITHUB_SHA` on every
push to `main` that changes `src/`, `frontend/dashboard/`, `docs/` (the How this works tab serves its pages),
`pyproject.toml` or `deploy/hf_space/`, and on demand (Actions, **Run workflow**, optionally a factory rebuild), once for
each Space in its matrix, as separate jobs: one Space's failure does not stop another's. The matrix lists only the
public Space; the private, paused RTX PRO Space is published by hand before it is restarted. It needs the repository secret `HF_TOKEN`:
a Hugging Face token with write access to every Space (a fine-grained token must list each). A token without access
fails that Space's job with a message saying so; a Space not created yet is skipped with a warning.

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
- `dashboard.synthetic.hosted_openrouter`: the [hosted OpenRouter key](dashboard.md#hosted-key)'s model and limits,
  used only where the Space has the secret `MMAI_OPENROUTER_HOSTED_KEY` ([the lite Space](#the-lite-space)).

Clinical trial management's file (`no_phi/dashboard_trial_management/` under the data root) starts empty and is shared
by everyone using the Space.

## The LLM server

Until the local server is ready (or on hardware without a GPU), Data and models offers OpenRouter: a visitor pastes
their own key, which the app keeps only in memory for that browser session and sends only to OpenRouter, and picks a
model (by default `nvidia/nemotron-3.5-super-vl`, the same Nemotron 3.5 Super). Nothing else is accepted. The main
Space has no hosted OpenRouter key (that is [the lite Space](#the-lite-space)'s).

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

## The lite Space

[`ksg-dfci/matchminer-ai-1226`](https://huggingface.co/spaces/ksg-dfci/matchminer-ai-1226) (named
`matchminer-ai-1226-lite` until 2026-11-27; the old name redirects) runs the same image and commit on one T4 (16 GB, `t4-small`), which sleeps after five hours too. It hosts no LLM:

- **Tier:** `tier.py` finds under 40 GB of GPU memory (or none) and sets `MMAI_DASH_LOCAL_SERVER_ENABLED=false`; no
  weights are prefetched and no server starts. With site.yaml's `only` and `allow_openrouter`, OpenRouter with each
  visitor's own key is then the only LLM connection the app accepts, and every task's default (Nemotron 3.5 Super,
  `nvidia/nemotron-3.5-super-vl`, with reasoning off). Data and models says why there is no local server and shows the
  key field; until a visitor pastes a key, the LLM tasks say to, and everything else works.
- **Hosted key:** with the Space secret `MMAI_OPENROUTER_HOSTED_KEY` (this Space only; the RTX PRO Space has none, so
  the same `site.yaml` offers it nothing), the owner's OpenRouter key is every task's default instead, for one model:
  **Hosted: Nemotron 3.5 Super (free, shared quota)**, `nvidia/nemotron-3.5-super-vl:free`, with site.yaml's
  `hosted_openrouter` limits (16 requests a minute for the whole Space, 150 per browser session a day, a stop at 20
  of the account's free-model requests left today; a 429 is not retried). The key is restricted on OpenRouter (a
  guardrail's model allowlist) to that model, and the app refuses any other model through it before sending. The app
  reads the secret once and removes it from its environment; nothing of it is ever shown. Data and models shows the
  model locked, today's requests left and the session's share, and says that requests go to OpenRouter under the
  Space's account, that free models' providers may log prompts, and to enter synthetic data only. A visitor may still
  paste their own key, which then serves instead, for any model.
- **Decision model:** the T4 (Turing, compute capability 7.5) has no native bfloat16, so `dtype: auto` loads
  MatchEval-1226 in float16 there (`matching.decision_model.PRE_AMPERE_DTYPE`; `MMAI_DASH_DTYPE` or `--dtype` choose
  another). Its own attention code takes the masked-SDPA path below compute capability 8.0 (its flash path needs
  Ampere). Checked on 390 synthetic requests of the published corpus (1,798 answers: 300 match pairs of five questions,
  60 boilerplate, 30 per-criterion sets), with the pre-Ampere path forced on an H100 (November 2026):

  | Against bfloat16 with the flash kernels (production) | max abs. difference in P(yes) | answers on the other side of 0.5 |
  | --- | --- | --- |
  | float16, masked SDPA | 0.0054 | none |
  | float32, masked SDPA | 0.0054 | none |
  | bfloat16, masked SDPA | 0.0069 | 1 (0.496 vs 0.500) |

  float16 is the closest to float32 (max 0.0008), with no non-finite values (the largest layer output was 48), the
  same memory as bfloat16 (4.8 GiB at the app's 16,384-token budget) and bfloat16's speed; float32 took 7.8 GiB and
  four times as long. TrialSpace keeps its published settings (float32 weights, bfloat16 autocast, which torch
  emulates on a T4).
- **Image:** the Dockerfile's `lite` variant, which `publish.py` sets for this Space (`SPACE_VARIANTS`, rewriting
  `ARG MMAI_VARIANT=full`): the shared base and package layers on plain Ubuntu, with no llama.cpp stage, vLLM,
  FlashInfer, CUDA compiler or Mamba prebuild (BuildKit builds only the stages a variant uses). The package is
  installed with `no-engines.txt`, so its vLLM dependency is left out (the app needs vLLM only to serve or parse a
  local model). About 7 GB installed against about 16 GB for the full image (estimates from installed package
  sizes: 6.5 GB of Python packages, mostly torch and its CUDA libraries, against about 10 GB, plus the full image's
  CUDA runtime base, FlashInfer's 1.7 GB of cubins and llama-server). torch 2.14's CUDA 13 wheels carry SM 7.5
  kernels. On any GPU the lite image picks the no-server tier, since it has no engine to start.

## After a dataset update

A new dataset commit (new guideline catalogs, published vectors, a new extraction store) needs only a restart, no
rebuild:

```bash
python deploy/hf_space/refresh.py --dry-run   # the dataset's main, and the catalogs the Space would load
python deploy/hf_space/refresh.py --wait      # restart, and wait until it runs again
```

It warns when `MMAI_DATASET_REVISION` pins another commit. The token is `HF_TOKEN` (or the huggingface_hub login), as
for `publish.py`. A restart repeats the startup below. Add `--space ksg-dfci/matchminer-ai-1226-rtxpro` for the RTX PRO
Space (a restart also unpauses it).

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
