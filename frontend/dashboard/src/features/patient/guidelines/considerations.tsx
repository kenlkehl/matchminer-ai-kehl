import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";

import type { CitationReview, Consideration, GuidelineSpace, SourceExcerpt } from "../types";

const UNSPECIFIED = new Set([
  "",
  "na",
  "n/a",
  "not specified",
  "not specified in supplied evidence",
  "not specified in the supplied evidence",
]);

function Conditions({ value }: { value: string | null | undefined }) {
  const text = String(value ?? "").trim();
  const normalized = text.toLowerCase().replace(/\s+/g, " ").replace(/\.+$/, "");
  if (UNSPECIFIED.has(normalized)) return null;
  return <p className="text-sm whitespace-pre-wrap text-gray-700">{text}</p>;
}

/** A warning box: the catalog's caveats, or source support that needs review. */
export function Caution({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-950">
      <strong className="mb-1 block">{title}</strong>
      {children}
    </div>
  );
}

export function CitationIssues({ owner }: { owner: { citation_review?: CitationReview | null } }) {
  const issues = owner.citation_review?.issues ?? [];
  if (!issues.length) return null;
  return (
    <Caution title="Source support needs review">
      <ul className="list-disc space-y-1 pl-5">
        {issues.map((issue, i) => (
          <li key={i}>{issue}</li>
        ))}
      </ul>
      <p className="mt-1">Any excerpts below support only part of this catalog item.</p>
    </Caution>
  );
}

/** The guideline passages that support a catalog item, folded. */
export function SourceEvidence({ items }: { items: SourceExcerpt[] | null | undefined }) {
  if (!items?.length) return null;
  return (
    <details className="group text-sm">
      <summary className="cursor-pointer text-brand-blue hover:underline">
        View source evidence · {items.length} {items.length === 1 ? "excerpt" : "excerpts"}
      </summary>
      <div className="mt-2 flex flex-col gap-2">
        {items.map((item, i) => (
          <blockquote
            key={i}
            className="border-l-2 border-gray-300 pl-3 whitespace-pre-wrap text-gray-800"
          >
            {item.quote}
            <cite className="mt-1 block text-xs text-gray-500 not-italic">
              {item.printed_label || item.page_id}
              {item.pdf_page ? ` · PDF page ${String(item.pdf_page)}` : ""}
              {item.line_ids?.length ? ` · Lines ${item.line_ids.join(", ")}` : ""}
            </cite>
          </blockquote>
        ))}
      </div>
    </details>
  );
}

/** A space's workup items or treatment options, each with its category, conditions and sources. */
export function Considerations({ items }: { items: Consideration[] }) {
  if (!items.length) {
    return (
      <p className="text-sm text-gray-500 italic">
        No considerations in this category were recorded for this catalog space.
      </p>
    );
  }
  return (
    <ol className="flex flex-col gap-2">
      {items.map((item, i) => (
        <li key={i} className="flex gap-3 rounded-md border border-gray-200 bg-white p-3">
          <span className="font-mono text-xs text-gray-400">{String(i + 1).padStart(2, "0")}</span>
          <div className="flex min-w-0 flex-1 flex-col items-start gap-1.5">
            <h5 className="font-medium text-gray-900">{item.name}</h5>
            <Badge variant="outline" className="h-auto whitespace-normal">
              {item.category}
            </Badge>
            <Conditions value={item.conditions} />
            <CitationIssues owner={item} />
            <SourceEvidence items={item.evidence} />
          </div>
        </li>
      ))}
    </ol>
  );
}

export function PopulationTab({ space }: { space: GuidelineSpace }) {
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm whitespace-pre-wrap text-gray-800">
        {space.cohort_key_criteria || space.clinical_space_summary}
      </p>
      <p className="text-xs text-gray-500">
        The population's key criteria, as TrialSpace and the decision model read them, written like
        a trial cohort's key criteria. A match does not establish that a patient meets the full
        guideline population.
      </p>
      <CitationIssues owner={space} />
      <SourceEvidence items={space.evidence} />
    </div>
  );
}

export function SourceTab({ space }: { space: GuidelineSpace }) {
  const facts: [string, string][] = [
    ["Guideline", space.source.title],
    ["Edition", space.source.version],
    ["Space ID", space.space_trial_id],
    space.llm_match !== undefined
      ? [
          "LLM match score (0–6)",
          space.llm_match?.score == null ? "no valid answer" : String(space.llm_match.score),
        ]
      : ["Match quality", Number(space.match_quality_score).toFixed(4)],
    ["TrialSpace cosine", Number(space.similarity_score).toFixed(4)],
    ["Retrieval rank", String(space.retrieval_rank)],
  ];
  return (
    <div className="flex flex-col gap-3">
      <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2 lg:grid-cols-3">
        {facts.map(([label, value]) => (
          <div key={label}>
            <dt className="text-xs text-gray-500">{label}</dt>
            <dd className="break-words text-gray-900">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="text-xs text-gray-500">
        {space.llm_match !== undefined
          ? "The LLM's match score (its five yes/no answers combined, 0–6) is a ranking signal, not a probability. Spaces it scored low are kept; the decision model's scores are not used for this ranking."
          : "Match-quality scores from the decision model are ranking signals, not probabilities. The cutoff flag is kept; low-scoring results are not silently removed."}
      </p>
      {space.uncertainties.length ? (
        <Caution title="Catalog uncertainties">
          <ul className="list-disc space-y-1 pl-5">
            {space.uncertainties.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
        </Caution>
      ) : (
        <p className="text-xs text-gray-500">
          No explicit uncertainties recorded. This does not establish clinical correctness.
        </p>
      )}
    </div>
  );
}
