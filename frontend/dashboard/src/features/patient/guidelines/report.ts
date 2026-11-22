/**
 * Labels for workup reviews and the Markdown report export (the guideline fetcher's report.mjs and
 * extraction_view.mjs): they format completed reviews without changing their findings.
 */
import type {
  BackupMetadata,
  ExtractionCost,
  RetrieveResult,
  WorkupAssessment,
  WorkupReview,
} from "../types";

export const statusLabels: Record<string, string> = {
  completed: "Completed",
  partially_completed: "Partially completed",
  planned: "Planned / pending",
  not_done: "Explicitly not done",
  not_documented: "Not documented",
  unclear: "Unclear / conflicting",
  error: "Review failed",
};

export const applicabilityLabels: Record<string, string> = {
  applies: "Applies",
  not_applicable: "Not applicable",
  uncertain: "Uncertain",
};

export const extractionStatusLabels: Record<string, string> = {
  met: "Met · documented in extraction",
  not_met: "Not documented in extraction",
  unknown: "Unknown",
  not_applicable: "Not applicable",
  done_not_recommended: "Done though not recommended",
};

export const extractionReasonLabels: Record<string, string> = {
  documented: "Documented by extracted items",
  not_documented: "No extracted item matched",
  planned_only: "Only planned or ordered",
  uncertain_dates: "Dates too imprecise to tell",
  fields_not_stated: "Needed details not stated",
  anchor_unavailable: "Reference event undated or missing",
  incomplete_extraction: "Some notes lack extraction records",
  no_extraction: "No extraction records",
  not_expressible: "Not checkable from extracted items",
  query_unavailable: "No valid query compiled",
  documented_not_applicable: "Documented facts make it not apply",
  documented_not_recommended: "Documented although the guideline advises against it",
};

export const stateLabels: Record<string, string> = {
  yes: "yes",
  unknown: "unknown",
  no: "no",
  planned: "planned",
};

const REVIEW_AID =
  "Review aid, not a determination: an LLM compiled each workup item into a query and extracted the " +
  "items from the notes; code ran the query. Check each cited quote against its note. Not documented " +
  "does not mean not done.";
const SQL_REVIEW_AID =
  "Review aid, not a determination: an LLM agent wrote each workup item's SQL query after exploring " +
  "extracted items, an LLM extracted the items from the notes, and code ran each query once over the " +
  "cohort. Check each cited quote against its note. Not documented does not mean not done.";

export const reviewAid = (method: string | undefined) =>
  method === "extraction_sql" ? SQL_REVIEW_AID : REVIEW_AID;

const label = (labels: Record<string, string>, key: string) => labels[key] ?? key;

/** The review methods that check extraction records (statuses met / not_met / unknown / not_applicable). */
export function isExtractionReview(review: WorkupReview | undefined): boolean {
  const method = review?.metadata.method;
  return method === "extraction_queries" || method === "extraction_sql";
}

export function isAgentic(review: WorkupReview | undefined): boolean {
  return review?.metadata.method === "agentic" || review?.metadata.scope === "searched_excerpts";
}

export function reviewMethodLabel(review: WorkupReview): string {
  if (isExtractionReview(review)) {
    return review.metadata.method === "extraction_sql"
      ? "SQL agent over extraction records"
      : "Structured extraction queries";
  }
  return !isAgentic(review)
    ? "Full-note review"
    : (review.metadata.followup_review?.max_review_passes ?? 0) > 0
      ? "Agentic note search · coverage review"
      : "Agentic note search · selected excerpts";
}

export function workupStatusLabel(item: WorkupAssessment, review: WorkupReview): string {
  if (isExtractionReview(review)) return label(extractionStatusLabels, item.status);
  const complete = item.metadata?.followup_review?.full_record?.complete;
  if (isAgentic(review) && item.status === "not_documented") {
    return complete ? "Not documented in supplied notes" : "Not located in searched excerpts";
  }
  return label(statusLabels, item.status);
}

export function workupReviewDetail(item: WorkupAssessment): string {
  const audit = item.metadata?.followup_review;
  if (!audit) return "";
  const parts: string[] = [];
  if (audit.full_record?.complete) {
    parts.push("Full-record fallback reviewed all supplied notes.");
  } else if (audit.full_record) {
    parts.push("Full-record fallback stopped before all supplied notes were reviewed.");
  }
  parts.push(
    audit.status === "incomplete"
      ? "Follow-up review incomplete; see limitations."
      : "Search coverage review finished.",
  );
  return parts.join(" ");
}

export function workupBackupDetail(metadata: BackupMetadata | null | undefined): string {
  if (metadata?.backup_resolution_failed) {
    return "The configured backup endpoint could not be prepared; see limitations.";
  }
  if (!metadata?.backup_used) return "";
  const events = metadata.backup_events ?? [];
  if (!events.length) {
    const count = metadata.backup_items ?? 0;
    return `Backup model used for ${String(count)} workup ${count === 1 ? "item" : "items"}; see each item's details.`;
  }
  return events
    .map((event) => {
      const reason =
        event.reason === "consecutive_python_errors"
          ? "consecutive Python errors"
          : "failed request or validation attempts";
      return `Backup model ${event.backup_model} used after ${String(event.failure_threshold)} ${reason} with ${event.primary_model}.`;
    })
    .join(" ");
}

export function automaticExcerpts(item: WorkupAssessment, review: WorkupReview): boolean {
  return (
    (item.metadata?.evidence_selection ?? review.metadata.evidence_selection) ===
    "automatic_reviewed_excerpts"
  );
}

export function patientEvidenceLabel(item: WorkupAssessment, review: WorkupReview): string {
  if (isExtractionReview(review)) return "Extracted-item evidence";
  return automaticExcerpts(item, review)
    ? "Reviewed patient-note excerpts"
    : "Patient-note evidence";
}

export function patientEvidenceCountLabel(item: WorkupAssessment, review: WorkupReview): string {
  const count = item.evidence?.length ?? 0;
  const noun = automaticExcerpts(item, review) ? "fragment" : "quote";
  return `${String(count)} ${noun}${count === 1 ? "" : "s"}`;
}

export function missingEvidenceLabel(item: WorkupAssessment, review: WorkupReview): string {
  if (isExtractionReview(review)) return "No extracted item quotes for this result.";
  if (item.status === "error") return "No documentation assessment available.";
  return automaticExcerpts(item, review)
    ? "No patient-note excerpts retained for review."
    : "No supporting evidence located in the supplied notes.";
}

/** What an extraction review would send, in a sentence or two. */
export function costLine(cost: ExtractionCost): string {
  const plural = (n: number, noun: string) => `${String(n)} ${noun}${n === 1 ? "" : "s"}`;
  const extraction = cost.reused_extraction
    ? cost.records_source === "store"
      ? "The extraction store's records of this patient are used; nothing is extracted."
      : "This session already extracted these notes (in Structured data or an earlier review); they will be reused."
    : `On-demand extraction: ${plural(cost.notes, "note")} (${cost.characters.toLocaleString()} characters) → ` +
      `${plural(cost.document_requests, "LLM request")}` +
      (cost.patient_requests
        ? ` + ${plural(cost.patient_requests, "patient-level chunk request")}`
        : "") +
      ` to ${cost.model}, each reasoning ${cost.typical_reasoning_tokens} with up to ` +
      `${cost.output_reserve_per_request.toLocaleString()} output tokens.`;
  const agent = cost.sql_agent;
  const queries = agent
    ? agent.items_to_explore
      ? ` SQL agent: ${String(agent.items_to_explore)} of ${String(cost.workup_items)} workup items to explore ` +
        `(${String(cost.cached_queries)} cached), up to ${String(agent.max_actions_per_item)} requests each, ` +
        `over the extraction records of ${agent.store} (${agent.patients.toLocaleString()} patients` +
        `${agent.database_built ? "" : "; its database is built first"}). The agent's query results ` +
        "(extracted values, without patient ids, note text or quotes) go to the SQL agent's connection" +
        `${agent.model ? ` (${agent.model})` : ""}.`
      : ` All ${String(cost.workup_items)} workup SQL queries are cached for ${agent.store}.`
    : cost.compile_requests
      ? ` Query compilation: ${plural(cost.compile_requests, "request")} (${String(cost.cached_queries)} of ` +
        `${String(cost.workup_items)} workup items already cached).`
      : ` All ${String(cost.workup_items)} workup queries are cached.`;
  return `${extraction}${queries} ${agent ? "At most" : "About"} ${String(cost.requests)} requests before retries.`;
}

// ---------------------------------------------------------------- the Markdown export

/** Text safe inside Markdown: HTML and Markdown characters escaped, on one line. */
function markdownText(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/([\\`*_{}[\]|#+.!-])/g, "\\$1")
    .replace(/[\r\n]+/g, " ");
}

/**
 * The server's Markdown report of a guideline search, followed by the completed workup reviews shown
 * for its spaces (keyed by space index).
 */
export function buildExportReport(
  result: RetrieveResult,
  reviews: Partial<Record<number, WorkupReview>> = {},
): string {
  const sections: string[] = [];
  result.spaces.forEach((space, index) => {
    const review = reviews[index];
    if (!review?.assessments.length) return;
    const metadata = review.metadata;
    const lines = [
      `### ${markdownText(space.rank)}. ${markdownText(space.name)}`,
      "",
      `**Guideline:** ${markdownText(space.source.title)} · ${markdownText(space.source.version)}`,
      "",
      `**Space ID:** ${markdownText(space.space_trial_id)}`,
      "",
    ];
    if (review.notice) lines.push(markdownText(review.notice), "");
    lines.push(`**Review method:** ${markdownText(reviewMethodLabel(review))}`, "");
    if (metadata.note_count != null) {
      const counted = isExtractionReview(review)
        ? "Notes extracted"
        : isAgentic(review)
          ? "Raw notes available to search"
          : "Raw notes reviewed";
      lines.push(`**${counted}:** ${markdownText(metadata.note_count)}`, "");
    }
    if (metadata.model) lines.push(`**Review model:** ${markdownText(metadata.model)}`, "");
    const backup = workupBackupDetail(metadata);
    if (backup) lines.push(markdownText(backup), "");
    review.assessments.forEach((item, itemIndex) => {
      lines.push(
        `#### ${String(itemIndex + 1)}. ${markdownText(item.name ?? item.recommendation?.name)}`,
        "",
        `**Status:** ${markdownText(workupStatusLabel(item, review))}`,
        "",
        `**Applicability:** ${markdownText(label(applicabilityLabels, item.applicability))}`,
        "",
        markdownText(item.bottom_line),
        "",
      );
      const detail = workupReviewDetail(item);
      const itemBackup = workupBackupDetail(item.metadata);
      if (itemBackup) lines.push(markdownText(itemBackup), "");
      if (detail) lines.push(`**Review coverage:** ${markdownText(detail)}`, "");
      if (item.limitations?.length) {
        lines.push("**Limitations:** " + item.limitations.map(markdownText).join("; "), "");
      }
      if (item.evidence?.length) {
        lines.push(`**${patientEvidenceLabel(item, review)}**`, "");
        if (automaticExcerpts(item, review)) {
          lines.push(
            "Automatically retained review context; each excerpt is not necessarily support for the assessment.",
            "",
          );
        }
        for (const evidence of item.evidence) {
          lines.push(
            `Note ${markdownText(evidence.note_number)} · ${markdownText(evidence.note_date || "Date unavailable")}` +
              (evidence.note_type ? " · " + markdownText(evidence.note_type) : ""),
            "",
            ...String(evidence.quote)
              .split(/\r?\n/)
              .map((line) => `> ${markdownText(line)}  `),
            "",
          );
        }
      } else {
        lines.push(missingEvidenceLabel(item, review), "");
      }
    });
    sections.push(lines.join("\n"));
  });
  if (!sections.length) return result.report;
  return (
    result.report.trimEnd() +
    "\n\n---\n\n## Guideline workup concordance review\n\n" +
    "Completed documentation reviews available for the current patient and guideline spaces at export time.\n\n" +
    sections.join("\n")
  );
}
