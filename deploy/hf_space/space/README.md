---
title: MatchMiner-AI 1226
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
suggested_hardware: l40sx1
startup_duration_timeout: 2h
pinned: false
license: cc-by-nc-nd-4.0
short_description: Cancer trial matching on synthetic patients (research use)
models:
  - ksg-dfci/TrialSpace-1226
  - ksg-dfci/MatchEval-1226
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

# MatchMiner-AI 1226

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
  [Nemotron 3.5 Bolt](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) in llama.cpp on this
  Space's GPU. The Space reaches no other model service.
- **Data:** [mmai-synthetic-1226](https://huggingface.co/datasets/ksg-dfci/mmai-synthetic-1226), downloaded at
  startup at a pinned revision.

After the Space wakes, it needs some minutes before it serves: it downloads the data and models and builds its
search index. Clinical trial management edits are shared by everyone using the Space and do not survive a restart.

## Code

The Space installs the commit of the repository named in `MMAI_COMMIT`; its own files are the repository's
`deploy/hf_space/space/` (see `docs/apps/hf-space.md` there). A push to the repository's main branch that changes the
app republishes them, which rebuilds the Space.

## Licenses

- The app's code, TrialSpace-1226 and mmai-synthetic-1226: CC BY-NC-ND 4.0.
- MatchEval-1226: LFM Open License v1.0 (fine-tuned from LiquidAI's d1-2B).
- Nemotron 3.5 Bolt (NVIDIA): the OpenMDW-1.1 license.
- Trial listings: ClinicalTrials.gov (U.S. National Library of Medicine). Drug classes: NCI Thesaurus (CC BY 4.0).
