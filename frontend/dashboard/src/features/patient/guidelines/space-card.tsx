import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { useEffect, useState } from "react";

import { AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";

import { type LLMAnswer, LLMAnswerDialog, LLMMatchBadge } from "../../matching/llm-score-view";
import { reviewWorkup } from "../actions";
import { EndpointNotice } from "../fields";
import { JobProgress } from "../job-progress";
import { MAX_NOTE_CHARACTERS } from "../notes";
import { getPatient, isBusy, recordsBody, setPatient, usePatient, usesExtraction } from "../store";
import type { ExtractionCost, GuidelineSpace, WorkupMethod } from "../types";
import { Considerations, PopulationTab, SourceTab } from "./considerations";
import { Findings } from "./findings";
import { costLine } from "./report";

const METHODS: { id: WorkupMethod; label: string; help: string }[] = [
  {
    id: "full_notes",
    label: "Full-note question answering",
    help: "Reviews every supplied note chunk for each item.",
  },
  {
    id: "agentic",
    label: "Agentic note search",
    help: "Searches each item in parallel, checks coverage, and reviews more notes when needed. May miss evidence.",
  },
  {
    id: "extractions",
    label: "Structured extraction queries",
    help: "Extracts each original note into structured items (one LLM request per note), compiles each workup item once into a cached query, and checks the queries in code. Each result cites its notes.",
  },
  {
    id: "extraction_sql",
    label: "SQL agent over extractions",
    help: "Extracts each original note into structured items, then an LLM agent explores the extraction record store with read-only SQL and writes one cached query per workup item; code runs it over this patient's items. Each result cites its notes. The agent's query results (extracted values, without patient ids or note text) go to the SQL agent's connection.",
  },
];

/** One ranked guideline space: its rank and scores, and its workup, treatment, population and source. */
export function SpaceCard({ space, index }: { space: GuidelineSpace; index: number }) {
  const job = usePatient().job;
  const reviewing = job?.kind === "workup" && job.space === index ? job : null;
  const caveats = space.uncertainties.length;
  const [answer, setAnswer] = useState<LLMAnswer | null>(null);
  return (
    <AccordionItem
      value={index}
      className="rounded-lg border border-gray-200 bg-white not-last:border-b"
    >
      <AccordionTrigger className="px-4 py-3">
        <span className="flex items-start gap-3">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-brand-blue text-sm font-semibold text-white">
            {space.rank}
          </span>
          <span className="flex min-w-0 flex-col gap-1">
            <span className="text-base leading-snug font-semibold text-gray-900">{space.name}</span>
            <span className="text-xs font-normal text-gray-600">
              {space.source.title} · {space.source.version}
            </span>
            <span className="flex flex-wrap items-center gap-2 text-xs font-normal">
              {space.llm_match !== undefined ? (
                // In the accordion's trigger (a button): a plain badge; its answers open from the content.
                <LLMMatchBadge answer={space.llm_match} scoring={false} />
              ) : (
                <span
                  className={cn(
                    "rounded-md px-2 py-0.5 font-semibold",
                    space.match_quality_pass
                      ? "bg-brand-blue/10 text-brand-midnight"
                      : "bg-amber-100 text-amber-900",
                  )}
                >
                  Match quality {Number(space.match_quality_score).toFixed(3)}
                  {space.match_quality_pass ? "" : " · Below cutoff"}
                </span>
              )}
              <span className="text-gray-600">
                {space.diagnostic_workup.length} workup · {space.treatment_options.length} treatment
                {caveats ? ` · ${String(caveats)} ${caveats === 1 ? "caveat" : "caveats"}` : ""}
              </span>
            </span>
          </span>
        </span>
      </AccordionTrigger>
      <AccordionContent className="flex flex-col gap-3 px-4">
        {reviewing ? <JobProgress job={reviewing} /> : null}
        {space.llm_match ? (
          <p className="flex flex-wrap items-center gap-2 text-xs text-gray-600">
            Ranked by the LLM's match answers ({space.llm_match.connection.label}).
            <Button
              size="xs"
              variant="outline"
              onClick={() => {
                setAnswer(space.llm_match ?? null);
              }}
            >
              LLM answers and reasoning
            </Button>
          </p>
        ) : null}
        <Tabs defaultValue="workup">
          <TabsList variant="line" className="w-full justify-start border-b">
            <TabsTrigger value="workup" className="flex-none">
              Workup
            </TabsTrigger>
            <TabsTrigger value="treatment" className="flex-none">
              Treatment
            </TabsTrigger>
            <TabsTrigger value="population" className="flex-none">
              Population
            </TabsTrigger>
            <TabsTrigger value="source" className="flex-none">
              Source &amp; caveats
            </TabsTrigger>
          </TabsList>
          <TabsContent value="workup" className="flex flex-col gap-3 pt-2">
            {space.diagnostic_workup.length ? (
              <WorkupReviewPanel index={index} space={space} />
            ) : null}
            <Considerations items={space.diagnostic_workup} />
          </TabsContent>
          <TabsContent value="treatment" className="pt-2">
            <Considerations items={space.treatment_options} />
          </TabsContent>
          <TabsContent value="population" className="pt-2">
            <PopulationTab space={space} />
          </TabsContent>
          <TabsContent value="source" className="pt-2">
            <SourceTab space={space} />
          </TabsContent>
        </Tabs>
        {answer ? (
          <LLMAnswerDialog
            answer={answer}
            onClose={() => {
              setAnswer(null);
            }}
          />
        ) : null}
      </AccordionContent>
    </AccordionItem>
  );
}

/** "Was this workup done?": the review method, the review's cost, and its findings. */
function WorkupReviewPanel({ index, space }: { index: number; space: GuidelineSpace }) {
  const s = usePatient();
  const busy = isBusy(s);
  const extraction = usesExtraction(s.workupMethod);
  const hasNotes = !!s.notesText.trim();
  const tooLong = s.notesText.length > MAX_NOTE_CHARACTERS;
  const review = s.reviews[index];
  const action =
    s.workupMethod === "extraction_sql"
      ? "Check workup with the SQL agent"
      : s.workupMethod === "extractions"
        ? "Check workup with extraction queries"
        : "Review workup in the notes";
  return (
    <section className="flex flex-col gap-3 rounded-md border border-gray-200 bg-slate-50 p-3">
      <div>
        <h4 className="text-sm font-semibold">Was this workup done?</h4>
        <p className="text-xs text-gray-600">
          Choose how to review the listed items in the clinical notes. Each method uses the
          guideline concordance connection and its enabled backup. The patient summary is not used
          as evidence.
        </p>
      </div>
      <fieldset className="flex flex-col gap-2" disabled={busy}>
        <legend className="mb-1 text-xs font-medium text-gray-600">Review method</legend>
        <RadioGroup
          value={s.workupMethod}
          disabled={busy}
          onValueChange={(value: WorkupMethod) => {
            setPatient({ workupMethod: value });
          }}
        >
          {METHODS.map((m) => (
            <Label key={m.id} className="items-start gap-2 font-normal">
              <RadioGroupItem value={m.id} className="mt-0.5" />
              <span className="flex flex-col gap-0.5 leading-snug">
                <strong className="text-sm font-medium">{m.label}</strong>
                <small className="text-xs text-gray-600">{m.help}</small>
              </span>
            </Label>
          ))}
        </RadioGroup>
      </fieldset>
      {extraction ? (
        <div className="flex flex-col gap-1.5">
          <Label className="gap-2 font-normal">
            <Checkbox
              checked={s.extractionLevels === "document_and_patient"}
              disabled={busy}
              onCheckedChange={(checked) => {
                setPatient({ extractionLevels: checked ? "document_and_patient" : "document" });
              }}
            />
            Also build a patient-level record from all notes (more requests)
          </Label>
          <CostEstimate index={index} />
        </div>
      ) : null}
      <div className="flex flex-wrap items-end gap-2">
        <Button
          disabled={busy || !hasNotes || tooLong || !s.retrievalJobId}
          onClick={() => {
            reviewWorkup(index);
          }}
        >
          {action}
        </Button>
      </div>
      <p className="text-xs text-gray-600">
        {hasNotes
          ? "Conditional items may not apply. Missing documentation does not mean “not done.”"
          : "Add notes under Patient → Clinical notes, or choose a patient with notes, to review the workup."}
        {extraction ? " Structured extraction quotes the original notes." : ""}
      </p>
      <EndpointNotice
        tasks={s.workupMethod === "extraction_sql" ? ["workup", "workup_sql"] : ["workup"]}
      />
      {review ? <Findings review={review} space={space} /> : null}
    </section>
  );
}

/** What an extraction review of this space would send (the server estimates it; no model calls). */
function CostEstimate({ index }: { index: number }) {
  const s = usePatient();
  const method = s.workupMethod === "extraction_sql" ? "extraction_sql" : "extractions";
  const ready = !!s.retrievalJobId && !!s.notesText.trim();
  // Estimated once typing in the notes pauses, not at each keystroke.
  const [version, setVersion] = useState(s.notesVersion);
  useEffect(() => {
    const timer = setTimeout(() => {
      setVersion(s.notesVersion);
    }, 600);
    return () => {
      clearTimeout(timer);
    };
  }, [s.notesVersion]);
  const cost = useQuery({
    queryKey: [
      "extraction-cost",
      s.retrievalJobId,
      index,
      s.extractionLevels,
      method,
      version,
      recordsBody(s).patient_id,
    ],
    queryFn: () => {
      const now = getPatient();
      return api.post<ExtractionCost>("/api/guidelines/workup/extraction-cost", {
        text: now.notesText,
        notes: now.notes,
        retrieval_job_id: now.retrievalJobId,
        space_index: index,
        extraction_levels: now.extractionLevels,
        method,
        patient_id: recordsBody(now).patient_id,
      });
    },
    enabled: ready && version === s.notesVersion,
    staleTime: 60_000,
    retry: false,
  });
  const text = !ready
    ? "Add clinical notes to estimate what extraction sends."
    : cost.data
      ? `${costLine(cost.data)}${
          cost.data.reused_extraction
            ? ""
            : " Notes go to the guideline concordance connection; use only an endpoint authorized for them."
        }`
      : cost.isError
        ? cost.error.message
        : "Estimating what this sends…";
  return (
    <p role="status" className={cn("text-xs", cost.isError ? "text-red-700" : "text-gray-700")}>
      {text}
    </p>
  );
}
