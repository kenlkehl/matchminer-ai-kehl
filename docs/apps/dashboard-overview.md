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

**Trial management** opens a sidebar beside the groups. It lists the trials the page shows (narrow them with the
search and filters, or the sidebar's own filter), with each trial's slots and what has been changed. Open a trial to
work on its cohorts:

- **Slots:** enter the slots available and the numbers consented and enrolled for each cohort. Slots can be kept for
  a cohort, for each of the original cohorts it combines (cohorts with the same criteria are shown as one), or for a
  subcohort. Each keeps its own numbers; they are never added up.
- **Subcohorts:** when a sponsor takes only patients with requirements beyond the protocol's, define a subcohort. It
  starts with its cohort's criteria; add the sponsor's requirements to them. A diagram shows where each cohort and
  subcohort comes from.
- **Criteria:** edit, add or remove any cohort's criteria lines. New and changed lines are labeled inclusion or
  exclusion and key or common automatically, or by you. When you open a trial, the lines of its original cohorts
  are labeled automatically too. **Revert to the AI-extracted criteria** undoes the edits. When a cohort with
  subcohorts changes, you choose whether the subcohorts take the new criteria. If the trial is extracted again and
  its criteria change, edited cohorts are flagged for review.
- **Groups:** drag a cohort from the sidebar onto a group on the left to show it there. **New group** adds a group
  of your own to drop cohorts onto. The groups' **Slots** column shows each cohort's slots available, consented and
  enrolled.

Find patients, scoring, the criteria checks and screening all use a cohort's current criteria, and subcohorts can be
searched like any cohort. Changes are saved as you make them and shared with everyone using the dashboard; **Export**
and **Import** save and load them as a file. **Reset this trial** removes one trial's slots, edits, subcohorts and
group placements, and **Clear all** removes everything entered on Trial management. Both ask first, offer an export,
and keep a copy of the file as it was, which you can import to undo them.

### Patient-centric search

Choose one patient: a patient from the list, a summary you enter, or clinical notes (pasted, or uploaded as PDF, CSV
or text) that the app summarizes. Then:

- **Trials:** the trial cohorts that fit the patient, ranked and scored, with filters by phase and disease center.
  **Only cohorts with open slots** searches only the cohorts and subcohorts with slots available on Trial
  management, each on its own criteria.
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

On a distillation fidelity run, **Export results** downloads the results as a PDF to share. For real patients it also
has a Table 1 describing the patients (age, race, ethnicity, sex and cancer type) and the main results for each of
these groups. Small numbers are hidden there to protect patient privacy.

### How this works

These pages: how matching works, how the models are trained, what each model answers, and the data model of the
extracted clinical facts. Start with [How matching works](../user-guide/patient-centric-matching.md).

### Data and models

The data sources and models in use, background jobs, and the model connection for each LLM task in your browser
session (summarizing notes, guideline checks, writing queries, the GoodOption check, LLM scoring, the assistant and
labeling edited criteria).

## Decision model or LLM

Wherever the app scores a patient and a trial cohort, **Scores from** switches between the decision model, a small
model that answers in a fraction of a second, and an LLM, which shows its reasoning. Both answer the same questions,
such as whether the pair is a reasonable consideration and whether the patient clearly fails a criteria line.
