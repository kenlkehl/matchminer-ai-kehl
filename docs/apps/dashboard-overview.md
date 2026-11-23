# Overview

MatchMiner-AI matches cancer patients with clinical trials and compares patients' care with treatment guidelines. It
reads patient summaries, clinical notes and the clinical facts extracted from them, and ranks or checks them so a
clinician or trial team knows where to look first.

## Two modes

- **Real mode:** your institution's trials and patients, on the local machine only, after signing in.
- **Synthetic mode:** the synthetic training run's trials and patients, so everything here can be tried without
  patient data.

Both use the same models: TrialSpace, which finds the patients and trials nearest each other, and the decision model,
which answers yes/no questions about a patient and a trial cohort.

## The tabs

### Clinical trial management

Browse the trials by disease center (real mode) or cancer type (synthetic mode). Each trial cohort shows its key
eligibility criteria, the trial's drugs and how they work, and contacts. Trial details show the cohorts from both the
protocol and the ClinicalTrials.gov listing. Trials not open to enrollment are hidden until **Show closed trials** is
checked.

**Find patients** ranks the patients nearest a cohort's key criteria and scores each candidate. For a candidate you
can also check each criteria line, and run the GoodOption check on the trial's drugs. **Screen with structured data**
checks the cohort's criteria lines against the clinical facts extracted from every candidate's notes. A line fails
only when the records contradict it; a fact that is simply not documented never fails a line.

### Patient-centric search

Choose one patient: a patient from the list, a summary you enter, or clinical notes (pasted, or uploaded as PDF, CSV
or text) that the app summarizes. Then:

- **Trials:** the trial cohorts that fit the patient, ranked and scored, with filters by phase and disease center.
  Each trial's criteria can be checked line by line, and the GoodOption check weighs the trial's drugs for this
  patient using the published evidence gathered for each drug and drug class.
- **Guidelines:** the guideline populations (NCCN and NCI PDQ) that fit the patient, with their recommended workup
  and treatment options and the sources they come from. A workup review checks which recommended steps the notes
  document, and the results can be saved as a report.
- **Structured data:** the clinical facts extracted from the notes, such as diagnoses, biomarkers, treatments,
  procedures, imaging, labs, performance status and disease assessments, each with the quotes it rests on. **Disease
  course** puts them on one timeline. A patient without stored records can be extracted on request.

### Guideline concordance reporting

Load a population of patients (a sample, tables you upload, or a saved cohort), choose a guideline population, and
find the patients who fit it. Then check across the whole cohort whether each patient's documented workup matches
what the guideline recommends: by reading the full notes, or by queries over the extracted clinical facts. Drill into
a workup step or a patient, and export the cohort and its results.

### Assistant

Ask questions in plain language, such as "what trials are open for EGFR-mutant lung cancer?" or "find patients with
metastatic colorectal cancer". The assistant answers from the app's own data with the same search, matching, scoring
and screening the other tabs use, asks when a question is unclear, shows each step it takes, and can draw charts of
its results.

### Benchmarking

The evaluation runs: how closely the models reproduce the large language model (LLM) they were trained from, how
LLM judges rate the pipeline's outputs, and review of the cases two judges disagree on.

### How this works

These pages: how matching works, how the models are trained, what each model answers, and the data model of the
extracted clinical facts. Start with [How matching works](../user-guide/patient-centric-matching.md).

### Data and models

The data sources and models in use, background jobs, and the model connection for each LLM task in your browser
session (summarizing notes, guideline checks, writing queries, the GoodOption check, LLM scoring and the assistant).

## Decision model or LLM

Wherever the app scores a patient and a trial cohort, **Scores from** switches between the decision model, a small
model that answers in a fraction of a second, and an LLM, which shows its reasoning. Both answer the same questions,
such as whether the pair is a reasonable consideration and whether the patient clearly fails a criteria line.
