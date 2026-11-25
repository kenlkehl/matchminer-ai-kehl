---
title: MatchMiner-AI 1226 Lite
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
startup_duration_timeout: 2h
pinned: false
license: cc-by-nc-nd-4.0
short_description: Synthetic-patient trial matching; LLMs via OpenRouter key
models:
  - ksg-dfci/TrialSpace-1226
  - ksg-dfci/MatchEval-1226
datasets:
  - ksg-dfci/mmai-synthetic-1226
tags:
  - clinical-trials
  - oncology
  - synthetic
  - matchminer-ai
---

# MatchMiner-AI 1226 Lite

The [MatchMiner-AI](https://github.com/kenlkehl/matchminer-ai-kehl) dashboard on the synthetic data of the
[MatchMiner-AI 1226 release](https://huggingface.co/collections/ksg-dfci/matchminer-ai-1226-6b02079780cbeb2146ee081c),
on a small GPU. It is [MatchMiner-AI 1226](https://huggingface.co/spaces/ksg-dfci/matchminer-ai-1226) without a
language model of its own: retrieval and scoring run here, and the LLM tasks use [OpenRouter](https://openrouter.ai)
with your own API key. MatchMiner-AI ranks patient-cohort pairs for human review: it is a research tool, not an
eligibility engine, and nothing it shows is medical advice.

**All patients here are synthetic.** Do not enter real patient information.

## What runs

- **Retrieval:** [TrialSpace-1226](https://huggingface.co/ksg-dfci/TrialSpace-1226) embeds the patients and cohorts.
- **Scoring:** [MatchEval-1226](https://huggingface.co/ksg-dfci/MatchEval-1226), the decision model, answers the match,
  boilerplate and per-criterion questions (in float16 on this Space's T4).
- **LLM tasks** (summaries, checks, the chat assistant): **OpenRouter with your own API key, at your own risk.** Paste
  the key on Data and models; the key and the synthetic data the tasks send go to OpenRouter under your account, which
  is billed for them. The Space keeps the key only in memory for your browser session and never shows it again. The
  default model is NVIDIA's Nemotron 3.5 Super (`nvidia/nemotron-3.5-super-vl`), the model that labeled the training
  data; Claude and open models are offered too, and any OpenRouter model ID may be typed. Without a key, everything
  but the LLM tasks works.
- **Data:** [mmai-synthetic-1226](https://huggingface.co/datasets/ksg-dfci/mmai-synthetic-1226), downloaded at
  startup, with its guideline catalogs (NCI PDQ) when it has them.

The Space sleeps after five hours without visitors. After it wakes, it needs a while before it serves (it downloads
the data and models and builds its search index). Clinical trial management edits are shared by everyone using the
Space and do not survive a restart.

## Code

The Space installs the commit of the repository named in `MMAI_COMMIT`; its own files are the repository's
`deploy/hf_space/space/`, with this card (see `docs/apps/hf-space.md` there). A push to the repository's main branch
that changes the app republishes them, which rebuilds the Space.

## Licenses

- The app's code, TrialSpace-1226 and mmai-synthetic-1226: CC BY-NC-ND 4.0.
- MatchEval-1226: LFM Open License v1.0 (fine-tuned from LiquidAI's d1-2B).
- Models used through OpenRouter: their own licenses and OpenRouter's and the providers' terms.
- Trial listings: ClinicalTrials.gov (U.S. National Library of Medicine). Drug classes: NCI Thesaurus (CC BY 4.0).
- Guideline catalogs: machine extractions from NCI's PDQ® cancer treatment summaries (Health Professional Versions),
  cited to them. They are not PDQ summaries and are not endorsed by NCI. PDQ® is a registered trademark of the U.S.
  Department of Health and Human Services.
