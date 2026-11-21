/**
 * What the Patient tab does with its shared state: set the patient (a corpus patient, an entered
 * summary, notes from an upload or a synthetic example), run the session's guideline jobs and follow
 * them, and restore the session's latest job after a reload (the guideline fetcher's app.js flow).
 */
import { api, ApiError, type Job, type JobStart, type PatientDetail, stopJob } from "@/lib/api";

import { corpusNotes, savedCompressionMatches } from "./notes";
import {
  CANCER_SUMMARY_SCAFFOLD,
  getPatient,
  hasSummary,
  initialPatientState,
  isBusy,
  type ActiveJob,
  type NotesOrigin,
  type PatientState,
  setPatient,
  type StatusMessage,
  usesExtraction,
} from "./store";
import { resetTrials } from "./trials/store";
import type {
  CompressionResult,
  GuidelineJobKind,
  ImportedNotes,
  NoteSource,
  NotesInput,
  RetrieveResult,
  SavedCompression,
  SessionView,
  SummaryResult,
  WorkupMethod,
  WorkupResult,
} from "./types";

const POLL_MS = 2000;
const KINDS: GuidelineJobKind[] = ["summarize", "compress", "retrieve", "workup"];
const METHODS: WorkupMethod[] = ["full_notes", "agentic", "extractions", "extraction_sql"];

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const areaOf = (kind: GuidelineJobKind): StatusMessage["where"] =>
  kind === "retrieve" || kind === "workup" ? "guidelines" : "input";

export function say(where: StatusMessage["where"], kind: StatusMessage["kind"], text: string) {
  setPatient({ status: { where, kind, text } });
}

export function dismissStatus() {
  setPatient({ status: null });
}

const noGuidelines = (): Partial<PatientState> => ({
  retrieval: null,
  retrievalJobId: null,
  reviews: {},
  openSpaces: [],
});

// ---------------------------------------------------------------- the saved note compression

/** A pending DELETE of the server's saved compression; jobs wait for it. */
let compressionReset: Promise<unknown> = Promise.resolve();

function discardSavedCompression(clearServer = true) {
  const had = Boolean(getPatient().noteCompression);
  setPatient({ noteCompression: null, noteSource: "original" });
  if (clearServer && had) {
    compressionReset = compressionReset
      .catch(() => undefined)
      .then(() => api.delete("/api/patient/note-compression"));
    compressionReset.catch((e: unknown) => {
      say("input", "error", errorText(e));
    });
  }
}

/** Keep the server's saved compression when it was made from the current notes. */
function restoreSavedCompression(saved: SavedCompression | null, restoreNotes = false) {
  const s = getPatient();
  let { notesText, notes } = s;
  const patch: Partial<PatientState> = {};
  if (saved && restoreNotes && !notesText) {
    ({ text: notesText, notes } = saved.notes_input);
    Object.assign(patch, {
      notesText,
      notes,
      notesOrigin: { kind: "restored" },
      notesVersion: s.notesVersion + 1,
    } satisfies Partial<PatientState>);
  }
  const match = savedCompressionMatches(saved, notesText, notes) ? saved : null;
  setPatient({ ...patch, noteCompression: match, ...(match ? {} : { noteSource: "original" }) });
}

async function refreshSavedCompression() {
  try {
    const saved = await api.get<{ note_compression: SavedCompression | null }>(
      "/api/patient/note-compression",
    );
    restoreSavedCompression(saved.note_compression);
  } catch {
    // The finished job's results stay available if this refresh fails.
  }
}

// ---------------------------------------------------------------- the patient

/** Replace the notes: their workup reviews go, and a compression made from other notes. */
function setNotes(input: NotesInput, origin: NotesOrigin | null, notice: string | null) {
  if (!savedCompressionMatches(getPatient().noteCompression, input.text, input.notes)) {
    discardSavedCompression();
  }
  setPatient((s) => ({
    notesText: input.text,
    notes: input.notes,
    notesOrigin: origin,
    notesNotice: notice,
    notesVersion: s.notesVersion + 1,
    reviews: {},
  }));
}

/** A corpus patient: its summary sections fill the summary, and its notes load for the workup. */
export function pickCorpusPatient(p: PatientDetail) {
  setPatient({
    patientId: p.id,
    fromCorpus: true,
    summary: p.key,
    otherHistory: p.boilerplate,
    summaryNote: null,
    summaryCompression: null,
    corpusNotesPending: p.id,
    status: null,
    ...noGuidelines(),
  });
  setNotes({ text: "", notes: null }, { kind: "corpus", patientId: p.id }, null);
}

/** The corpus patient's notes have loaded (ignored if another patient was chosen meanwhile). */
export function fillCorpusNotes(patientId: string, rows: Parameters<typeof corpusNotes>[0]) {
  const s = getPatient();
  if (s.corpusNotesPending !== patientId) return;
  const input = corpusNotes(rows);
  setPatient({ corpusNotesPending: null });
  setNotes(
    input,
    { kind: "corpus", patientId },
    input.notes
      ? `${String(input.notes.length)} ${input.notes.length === 1 ? "note" : "notes"} of the chosen patient, with their dates.`
      : "The chosen patient has no notes.",
  );
}

export function corpusNotesFailed(patientId: string, message: string) {
  if (getPatient().corpusNotesPending !== patientId) return;
  setPatient({ corpusNotesPending: null, notesNotice: message });
}

export function editSummary(patch: { summary?: string; otherHistory?: string }) {
  setPatient({ ...patch, fromCorpus: false, summaryNote: null, summaryCompression: null });
}

export function clearSummary() {
  editSummary({ summary: CANCER_SUMMARY_SCAFFOLD, otherHistory: "" });
}

export function editNotes(text: string) {
  setNotes(
    { text, notes: null },
    { kind: "pasted" },
    "Edited free text will be summarized. Dates are preserved only when stated in the text.",
  );
}

/** Load a synthetic example's notes (synthetic mode only): a new patient, so the summary starts over. */
export async function loadSyntheticNotes(source: string) {
  if (isBusy(getPatient()) || !source) return;
  setPatient({ loading: true, status: null });
  try {
    const data = await api.get<ImportedNotes>(
      `/api/patient/synthetic-notes?source=${encodeURIComponent(source)}`,
    );
    setNotes(data, { kind: "synthetic", label: data.label ?? source }, data.notice);
    setPatient({
      patientId: null,
      fromCorpus: false,
      summary: CANCER_SUMMARY_SCAFFOLD,
      otherHistory: "",
      summaryNote: "Create a summary from the synthetic notes",
      summaryCompression: null,
      corpusNotesPending: null,
      ...noGuidelines(),
    });
    say("input", "info", "Synthetic notes loaded. Create a patient summary, then search.");
  } catch (e) {
    say("input", "error", errorText(e));
  } finally {
    setPatient({ loading: false });
  }
}

/** Read an uploaded record (PDF, CSV, TXT or Markdown) into the notes. */
export async function uploadRecord(file: File, maxMb: number) {
  if (isBusy(getPatient())) return;
  if (file.size > maxMb * 1024 * 1024) {
    say("input", "error", `The file exceeds the ${String(maxMb)} MB upload limit.`);
    return;
  }
  setPatient({ loading: true, status: null, notesNotice: "Reading the uploaded record…" });
  try {
    const data = await api.upload<ImportedNotes>("/api/patient/upload", { file });
    discardSavedCompression();
    setNotes(data, { kind: "upload", name: file.name }, data.notice);
    setPatient({ corpusNotesPending: null });
    say("input", "info", "Record imported. Review the text, then create a summary.");
  } catch (e) {
    setPatient({ notesNotice: "No record was imported. Choose another file or paste notes." });
    say("input", "error", errorText(e));
  } finally {
    setPatient({ loading: false });
  }
}

/** Forget the patient, the guideline results and the trial matches (refused while a job runs). */
export async function clearWorkspace() {
  try {
    await api.delete("/api/patient/session");
  } catch (e) {
    say("input", "error", errorText(e));
    return;
  }
  const keep = getPatient();
  setPatient({
    ...initialPatientState(),
    restored: true,
    syntheticCase: keep.syntheticCase,
    catalog: keep.catalog,
    topN: keep.topN,
    candidateK: keep.candidateK,
    input: keep.input,
    status: { where: "input", kind: "info", text: "Patient workspace cleared." },
  });
  resetTrials();
}

// ---------------------------------------------------------------- restoring results

function restoreNoteInput(result: { notes_input?: NotesInput; note_source?: NoteSource }) {
  const input = result.notes_input;
  if (input) {
    const s = getPatient();
    const same =
      s.notesText === input.text && JSON.stringify(s.notes) === JSON.stringify(input.notes);
    if (!same) {
      setPatient({
        notesText: input.text,
        notes: input.notes,
        notesOrigin: { kind: "restored" },
        notesVersion: s.notesVersion + 1,
      });
    }
  }
  setPatient({ noteSource: result.note_source ?? "original" });
}

function restoreSummary(result: SummaryResult, how: "generated" | "restored") {
  restoreNoteInput(result);
  const compressed = Boolean(result.compression);
  setPatient({
    summary: result.summary,
    otherHistory: result.exclusion_evidence,
    fromCorpus: false,
    summaryCompression: result.compression,
    summaryNote:
      how === "generated"
        ? compressed
          ? "Generated from compressed notes · review before searching"
          : "Generated summary · review before searching"
        : compressed
          ? "Restored summary from compressed notes"
          : "Restored from this browser session",
  });
}

function scopeOf(result: RetrieveResult): Partial<PatientState> {
  const search = result.search;
  if (!search) return {};
  return {
    catalog: search.catalogs.length === 1 ? (search.catalogs[0] ?? "") : "",
    topN: search.top_n,
    candidateK: search.candidate_k,
  };
}

function restoreRetrieval(result: RetrieveResult, jobId: string) {
  setPatient((s) => ({
    retrieval: result,
    retrievalJobId: jobId,
    reviews: {},
    openSpaces: result.spaces.length ? [0] : [],
    ...scopeOf(result),
    ...(hasSummary(s.summary) ? {} : { summary: result.summary }),
  }));
}

function methodOf(result: WorkupResult): WorkupMethod {
  const reported = result.review?.metadata.method;
  const method =
    result.method ?? (reported === "extraction_queries" ? "extractions" : reported) ?? "full_notes";
  return METHODS.find((m) => m === method) ?? "full_notes";
}

function restoreWorkup(result: WorkupResult) {
  setPatient((s) => {
    const same = s.retrievalJobId === result.retrieval_job_id;
    return {
      retrieval: result.retrieval,
      retrievalJobId: result.retrieval_job_id,
      reviews: {
        ...(same ? s.reviews : {}),
        ...(result.review ? { [result.space_index]: result.review } : {}),
      },
      workupMethod: methodOf(result),
      extractionLevels: result.extraction_levels ?? s.extractionLevels,
      openSpaces: [...new Set([...(same ? s.openSpaces : []), result.space_index])],
      ...scopeOf(result.retrieval),
      ...(hasSummary(s.summary) ? {} : { summary: result.retrieval.summary }),
    };
  });
  restoreNoteInput(result);
}

function restoreCompression(result: CompressionResult) {
  if (result.summary_result) restoreSummary(result.summary_result, "restored");
  if (result.workup_result) restoreWorkup(result.workup_result);
  else if (result.retrieval && result.retrieval_job_id) {
    restoreRetrieval(result.retrieval, result.retrieval_job_id);
  }
  restoreNoteInput(result);
  if (result.summary && !hasSummary(getPatient().summary)) setPatient({ summary: result.summary });
}

// ---------------------------------------------------------------- jobs

function jobTitle(kind: GuidelineJobKind, s: PatientState): string {
  const compressed = s.noteSource === "compressed";
  switch (kind) {
    case "retrieve":
      return "Finding relevant guideline spaces";
    case "compress":
      return "Compressing clinical notes";
    case "workup":
      return s.workupMethod === "extraction_sql"
        ? "Checking workup with the SQL agent"
        : s.workupMethod === "extractions"
          ? "Checking workup with extraction queries"
          : compressed
            ? "Reviewing workup in saved compressed notes"
            : "Reviewing workup in original notes";
    case "summarize":
      return compressed
        ? "Creating the summary from saved compressed notes"
        : "Creating the patient summary from original notes";
  }
}

function patchJob(patch: Partial<ActiveJob>) {
  setPatient((s) => (s.job ? { job: { ...s.job, ...patch } } : {}));
}

const progressText = (message: string) =>
  message.replace(/^Embedding 1 patient summaries\b/, "Embedding 1 patient summary");

function finish(kind: GuidelineJobKind, job: Job) {
  switch (kind) {
    case "summarize":
      restoreSummary(job.result as SummaryResult, "generated");
      setPatient({ input: "summary", inputOpen: true });
      say(
        "input",
        "info",
        "Summary ready. Review or edit it, then find trials or guideline spaces.",
      );
      break;
    case "compress":
      say(
        "input",
        "info",
        "Notes compressed and saved. Choose original or compressed notes for the next summary or workup check.",
      );
      break;
    case "workup":
      restoreWorkup(job.result as WorkupResult);
      say(
        "guidelines",
        "info",
        "Workup documentation review ready. Inspect the findings and source excerpts.",
      );
      break;
    case "retrieve":
      restoreRetrieval(job.result as RetrieveResult, job.id);
      say("guidelines", "info", "Guideline considerations ready for review.");
      break;
  }
}

/** Start a job (or follow one already running) until it ends, keeping the store up to date. */
async function followJob(
  kind: GuidelineJobKind,
  start: (() => Promise<JobStart>) | Job,
  space?: number,
) {
  const s = getPatient();
  const running = typeof start === "function" ? null : start;
  const restoredSpace = (running?.restore as Partial<WorkupResult> | null | undefined)?.space_index;
  setPatient({
    job: {
      id: running?.id ?? null,
      kind,
      title: jobTitle(kind, s),
      status: running?.status ?? "queued",
      message: running?.message ? progressText(running.message) : "Preparing your request",
      done: running?.done ?? 0,
      total: running?.total ?? 0,
      started: (running?.started ?? running?.created ?? Date.now() / 1000) * 1000,
      stopping: running?.status === "cancelling",
      space: space ?? restoredSpace,
    },
    status: null,
  });
  try {
    await compressionReset.catch(() => undefined);
    const id = running?.id ?? (await (start as () => Promise<JobStart>)()).job.id;
    patchJob({ id });
    let failures = 0;
    for (;;) {
      let job: Job;
      try {
        job = await api.get<Job>(`/api/jobs/${encodeURIComponent(id)}`);
        failures = 0;
      } catch (e) {
        // An expired job is gone; a dropped connection may come back.
        if ((e instanceof ApiError && e.status < 500) || ++failures >= 5) throw e;
        patchJob({ message: "Connection interrupted. Reconnecting to the job…" });
        await sleep(3000);
        continue;
      }
      patchJob({
        status: job.status,
        message: progressText(job.message),
        done: job.done,
        total: job.total,
        ...(job.status === "cancelling" ? { stopping: true } : {}),
      });
      if (job.status === "error") throw new ApiError(500, job.error ?? "The job could not finish.");
      if (job.status === "cancelled") {
        say(
          areaOf(kind),
          "info",
          kind === "workup"
            ? "Workup review stopped. Your notes and guideline results are available to use again."
            : "Job stopped. You can start another job.",
        );
        break;
      }
      if (job.status === "done") {
        finish(kind, job);
        break;
      }
      await sleep(POLL_MS);
    }
  } catch (e) {
    say(areaOf(kind), "error", errorText(e));
  } finally {
    setPatient({ job: null });
    await refreshSavedCompression();
  }
}

function startJob(kind: GuidelineJobKind, path: string, body: unknown, space?: number) {
  if (isBusy(getPatient())) return;
  void followJob(kind, () => api.post<JobStart>(path, body), space);
}

export async function stopActiveJob() {
  const job = getPatient().job;
  if (!job?.id || job.stopping) return;
  patchJob({ stopping: true });
  try {
    await stopJob(job.id);
  } catch (e) {
    if (getPatient().job?.id === job.id) {
      patchJob({ stopping: false });
      say(areaOf(job.kind), "error", errorText(e));
    }
  }
}

const notesBody = (s: PatientState) => ({ text: s.notesText, notes: s.notes });

export function summarize() {
  const s = getPatient();
  startJob("summarize", "/api/patient/summarize", { ...notesBody(s), note_source: s.noteSource });
}

export function compressNotes() {
  const s = getPatient();
  startJob("compress", "/api/patient/compress", {
    ...notesBody(s),
    summary: hasSummary(s.summary) ? s.summary : "",
    retrieval_job_id: s.retrievalJobId,
    note_source: s.noteSource,
  });
}

/** Find the guideline spaces for the summary; false (with a message) when the scope is invalid. */
export function findGuidelineSpaces(): boolean {
  const s = getPatient();
  if (!Number.isInteger(s.candidateK) || s.candidateK < s.topN || s.candidateK > 100) {
    say(
      "guidelines",
      "error",
      "Candidate count must be an integer between the requested result count and 100.",
    );
    return false;
  }
  setPatient({ ...noGuidelines(), inputOpen: false });
  startJob("retrieve", "/api/guidelines/retrieve", {
    summary: s.summary,
    catalogs: s.catalog ? [s.catalog] : [],
    top_n: s.topN,
    candidate_k: s.candidateK,
  });
  return true;
}

export function reviewWorkup(spaceIndex: number) {
  const s = getPatient();
  if (!s.retrievalJobId || isBusy(s)) return;
  const extraction = usesExtraction(s.workupMethod);
  setPatient((p) => {
    const reviews = { ...p.reviews };
    delete reviews[spaceIndex];
    return { reviews, openSpaces: [...new Set([...p.openSpaces, spaceIndex])] };
  });
  startJob(
    "workup",
    "/api/guidelines/workup",
    {
      ...notesBody(s),
      retrieval_job_id: s.retrievalJobId,
      space_index: spaceIndex,
      method: s.workupMethod,
      note_source: extraction ? "original" : s.noteSource,
      ...(extraction ? { extraction_levels: s.extractionLevels } : {}),
    },
    spaceIndex,
  );
}

// ---------------------------------------------------------------- the saved workspace

/** After a page load: show the session's latest guideline job (its inputs, its result, or follow it). */
export async function restoreSession() {
  if (getPatient().restored) return;
  setPatient({ restored: true });
  let saved: SessionView;
  try {
    saved = await api.get<SessionView>("/api/patient/session");
  } catch (e) {
    say("input", "error", `Could not load the saved workspace. ${errorText(e)}`);
    return;
  }
  const s = getPatient();
  const job = saved.job;
  const kind = KINDS.find((k) => k === job?.kind);
  if (job && kind && !hasSummary(s.summary) && !s.notesText && !s.retrieval && !s.job) {
    if (job.restore) {
      if (kind === "workup") restoreWorkup(job.restore as WorkupResult);
      else if (kind === "compress") restoreCompression(job.restore as CompressionResult);
      else if (kind === "retrieve") {
        // A server that saves a search's inputs: its summary and scope.
        const restore = job.restore as { summary?: string; search?: RetrieveResult["search"] };
        if (restore.summary) setPatient({ summary: restore.summary });
        if (restore.search) setPatient(scopeOf({ search: restore.search } as RetrieveResult));
      } else {
        restoreNoteInput(job.restore as { notes_input?: NotesInput; note_source?: NoteSource });
      }
    }
    if (job.status === "error") {
      say(areaOf(kind), "error", job.error ?? "The previous job could not finish.");
    } else if (job.status === "cancelled") {
      say(areaOf(kind), "info", "The previous job was stopped. You can start another job.");
    } else if (job.status === "done") {
      if (kind === "retrieve") restoreRetrieval(job.result as RetrieveResult, job.id);
      else if (kind === "workup") restoreWorkup(job.result as WorkupResult);
      else if (kind === "compress") restoreCompression(job.result as CompressionResult);
      else restoreSummary(job.result as SummaryResult, "restored");
      if (kind !== "compress" && kind !== "summarize") {
        setPatient({ summaryNote: "Restored from this browser session" });
      }
    } else {
      void followJob(kind, job);
    }
  }
  restoreSavedCompression(saved.note_compression, true);
  const after = getPatient();
  if (after.input === "corpus" && !after.patientId) {
    if (hasSummary(after.summary)) setPatient({ input: "summary" });
    else if (after.notesText) setPatient({ input: "notes" });
  }
}
