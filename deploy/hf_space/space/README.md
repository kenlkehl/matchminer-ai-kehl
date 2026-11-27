---
title: MatchMiner-AI 1226 RTX PRO
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
startup_duration_timeout: 2h
pinned: false
license: cc-by-nc-nd-4.0
short_description: Cancer trial matching on synthetic patients (research use)
models:
  - ksg-dfci/TrialSpace-1226
  - ksg-dfci/MatchEval-1226
  - nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
  - nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
  - nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF
datasets:
  - ksg-dfci/mmai-synthetic-1226
tags:
  - clinical-trials
  - oncology
  - synthetic
  - matchminer-ai
---

# MatchMiner-AI 1226 RTX PRO

The [MatchMiner-AI](https://github.com/kenlkehl/matchminer-ai-kehl) dashboard on the synthetic data of the
[MatchMiner-AI 1226 release](https://huggingface.co/collections/ksg-dfci/matchminer-ai-1226-6b02079780cbeb2146ee081c):
the 5,000 sampled ClinicalTrials.gov cancer trials and their cohorts, and 87,890 synthetic patients with their notes
and summaries. MatchMiner-AI ranks patient-cohort pairs for human review: it is a research tool, not an eligibility
engine, and nothing it shows is medical advice.

**All patients here are synthetic.** Do not enter real patient information.

## What runs

- **Retrieval:** [TrialSpace-1226](https://huggingface.co/ksg-dfci/TrialSpace-1226) embeds the patients and cohorts.
- **Scoring:** [MatchEval-1226](https://huggingface.co/ksg-dfci/MatchEval-1226), the decision model, answers the match,
  boilerplate and per-criterion questions.
- **LLM tasks** (summaries, checks, the chat assistant): NVIDIA
  [Nemotron 3.5 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) (NVFP4, text only),
  the model that labeled the training data, in vLLM on this Space's GPU (on a smaller GPU,
  [Nemotron 3.5 Bolt](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) in llama.cpp). The
  only other model service it uses is [OpenRouter](https://openrouter.ai), if you choose it on Data and models with
  your own API key, at your own risk: the key and the synthetic data the tasks send then go to OpenRouter under your
  account. The Space keeps the key only in memory for your browser session.
- **Data:** [mmai-synthetic-1226](https://huggingface.co/datasets/ksg-dfci/mmai-synthetic-1226), downloaded at
  startup, with its guideline catalogs (NCI PDQ) when it has them.

The Space sleeps after five hours without visitors. After it wakes, it needs a while before it serves (it downloads
the data and models and builds its search index), and longer before the LLM is ready. Clinical trial management edits are shared by everyone using the Space and do not survive a restart.

## Code

The Space installs the commit of the repository named in `MMAI_COMMIT`; its own files are the repository's
`deploy/hf_space/space/` (see `docs/apps/hf-space.md` there). A push to the repository's main branch that changes the
app republishes them, which rebuilds the Space.

## Licenses

- The app's code, TrialSpace-1226 and mmai-synthetic-1226: CC BY-NC-ND 4.0.
- MatchEval-1226: LFM Open License v1.0 (fine-tuned from LiquidAI's d1-2B).
- Nemotron 3.5 Super and Bolt (NVIDIA): the OpenMDW-1.1 license.
- Trial listings: ClinicalTrials.gov (U.S. National Library of Medicine). Drug classes: NCI Thesaurus (CC BY 4.0).
- Guideline catalogs: machine extractions from NCI's PDQ® cancer treatment summaries (Health Professional Versions),
  cited to them. They are not PDQ summaries and are not endorsed by NCI. PDQ® is a registered trademark of the U.S.
  Department of Health and Human Services.
