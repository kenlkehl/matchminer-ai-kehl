import { useMutation, useQuery } from "@tanstack/react-query";
import { getRouteApi } from "@tanstack/react-router";
import { cn } from "cn";
import { ChevronUpIcon, PillIcon, SearchIcon } from "lucide-react";
import {
  type PointerEvent,
  type ReactNode,
  type RefObject,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { SidebarSlot } from "@/components/shell/sidebar-slot";
import { SummarySplit } from "@/components/summary-split";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { api, type CohortHit, type PairScores, type TrialsForPatientResult } from "@/lib/api";
import { type LLMPairResult, type Scorer, useScorer } from "@/lib/llm-decisions";
import { metaOptions } from "@/lib/queries";
import { useSite } from "@/lib/site";

import {
  byLLMScore,
  type LLMAnswer,
  LLMAnswerDialog,
  LLMExclusionLine,
  llmLikelyExcluded,
  LLMMatchBadge,
  LLMRankingNote,
  LLMRequirements,
  LLMRunStatus,
} from "../../matching/llm-score-view";
import { ScoringStatus } from "../../matching/patients-for-cohort-page";
import { ScorerToggle } from "../../matching/scorer-toggle";
import {
  byScore,
  ExclusionLine,
  likelyExcluded,
  MatchBadge,
  Requirements,
} from "../../matching/score-view";
import { useLLMScoring } from "../../matching/use-llm-scoring";
import { useScoring } from "../../matching/use-scoring";
import { CatalogEvidenceDialog } from "../../overview/catalog-evidence-dialog";
import { hasSummary, type PatientSearch, setPatient, usePatientValue } from "../store";
import { GoodOptionButton, GoodOptionDialog } from "./good-option";
import { type AiFilters, setAiState, useAiState } from "./store";
import { TrialDialog } from "./trial-dialog";
import { drugsUnavailable, TrialDrugsDialog } from "./trial-drugs-dialog";

const route = getRouteApi("/_app/patient/");

const AI_DISCLAIMER =
  "Trial matches are based on probabilistic AI and can be incorrect. Trials displayed may be " +
  "biomarker-selected even if the patient is not known to have the biomarker, since biomarker " +
  "testing is often part of trial screening.";

type Row = CohortHit & { id: string };

/** "synthetic patient" in synthetic mode, "patient" in real mode. */
function usePatientNoun(): string {
  return useQuery(metaOptions()).data?.mode === "real" ? "patient" : "synthetic patient";
}

/**
 * The Patient tab's Trials view (mm-dash's AI search): the open trials' cohorts that match the shared
 * patient's summary, with the session scorer's answers (the decision model, or the LLM: the toggle) and the
 * common-exclusion check.
 */
export function TrialsView({ active }: { active: boolean }) {
  const ai = useAiState();
  return (
    <div className="flex flex-col gap-4">
      <SearchBar />
      {ai.result ? <Results result={ai.result} active={active} key={ai.searches} /> : null}
    </div>
  );
}

/** Search with the shared patient: a corpus patient by its ID, else the summary and other history. */
function SearchBar() {
  const patientId = usePatientValue((s) => s.patientId);
  const fromCorpus = usePatientValue((s) => s.fromCorpus);
  const summary = usePatientValue((s) => s.summary);
  const otherHistory = usePatientValue((s) => s.otherHistory);
  const { allCohorts, openSlotsOnly, result } = useAiState();
  const noun = usePatientNoun();
  const byPatient = !!patientId && fromCorpus;
  const ready = byPatient || hasSummary(summary);
  const search = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      api.post<TrialsForPatientResult>("/api/match/trials-for-patient", body),
    onSuccess: (result) => {
      setAiState((s) => ({
        result,
        searches: s.searches + 1,
        filters: { ...s.filters, excludeHigherThreshold: true },
      }));
      setPatient({ inputOpen: false });
    },
  });
  const submit = (slotsOnly = openSlotsOnly) => {
    const common = {
      k: 50,
      per_trial: !allCohorts,
      scope: "dashboard",
      open_slots_only: slotsOnly,
    };
    search.mutate(
      byPatient
        ? { ...common, patient_id: patientId }
        : { ...common, summary, boilerplate: otherHistory },
    );
  };
  return (
    <section
      aria-label="Find trials"
      className="flex flex-wrap items-center gap-3 rounded-lg border border-gray-200 bg-white p-4"
    >
      <div className="flex min-w-64 flex-1 flex-col gap-1 text-sm text-gray-700">
        <p>
          {byPatient ? (
            <>
              Matches {noun} <strong>{patientId}</strong>'s cancer summary against the open trials'
              cohorts on the disease-center dashboards and flags likely common exclusions.
            </>
          ) : ready ? (
            "Matches the patient summary above against the open trials' cohorts on the disease-center dashboards and flags likely common exclusions."
          ) : (
            "Choose a patient, enter a summary, or create one from clinical notes above to find trials."
          )}
        </p>
        {search.error ? <p className="text-red-700">{search.error.message}</p> : null}
      </div>
      <Label
        className="gap-1.5 text-sm font-normal text-gray-700"
        title="Search only the cohorts with slots available on Clinical trial management's Trial management, each on its own criteria: a subcohort or original cohort with slots is matched as itself"
      >
        <Checkbox
          checked={openSlotsOnly}
          onCheckedChange={(checked) => {
            setAiState({ openSlotsOnly: checked });
            if (result && ready && !search.isPending) submit(checked);
          }}
        />
        Only cohorts with open slots
      </Label>
      <Button
        disabled={!ready || search.isPending}
        onClick={() => {
          submit();
        }}
      >
        {search.isPending ? <Spinner /> : <SearchIcon />}
        Find trials
      </Button>
    </section>
  );
}

/** Whether a row is likely excluded by common criteria, by the session's scorer. */
type Excluded = (id: string) => boolean;

function filterRows(rows: Row[], f: AiFilters, higher: string[], excluded: Excluded) {
  const any = (chosen: string[], value: string) => chosen.length === 0 || chosen.includes(value);
  return rows.filter(
    (r) =>
      (f.showClosed || r.cohort.trial.is_open) &&
      any(f.phases, r.cohort.trial.phase) &&
      any(f.centers, r.cohort.trial.center) &&
      (f.locations.length === 0 || r.cohort.trial.sites.some((s) => f.locations.includes(s))) &&
      !(f.excludeHigherThreshold && higher.includes(r.cohort.trial.center)) &&
      !(f.hideLikelyExcluded && excluded(r.id)),
  );
}

/** The results; kept mounted while the Guidelines view shows (its filters show only when active). */
function Results({ result, active }: { result: TrialsForPatientResult; active: boolean }) {
  const ai = useAiState();
  const noun = usePatientNoun();
  const stale = usePatientValue((s) =>
    s.patientId && s.fromCorpus
      ? result.patient.custom || result.patient.id !== s.patientId
      : !result.patient.custom ||
        result.patient.key.trim() !== s.summary.trim() ||
        result.patient.boilerplate.trim() !== s.otherHistory.trim(),
  );
  const meta = useQuery(metaOptions()).data;
  const higher = meta?.higher_threshold_groups ?? [];
  const rows = useMemo(
    () => result.results.map((r) => ({ ...r, id: r.cohort.cohort_id })),
    [result],
  );
  // Scored in the order the filters show them, so the trials on screen are scored first.
  const visible = useMemo(
    () => filterRows(rows, ai.filters, higher, () => false),
    // The first scoring pass follows the default filters; later passes follow the current ones.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rows],
  );
  const scorer = useScorer();
  const usingLLM = scorer === "llm";
  const autoN = meta?.auto_score_n ?? 10;
  const scoring = useScoring(
    "trials_for_patient",
    result.patient.id,
    visible,
    usingLLM ? 0 : autoN,
  );
  const llm = useLLMScoring("trials_for_patient", result.patient.id, visible, autoN, usingLLM);
  const excluded: Excluded = (id) =>
    usingLLM ? llmLikelyExcluded(llm.results[id]) : likelyExcluded(scoring.scores[id]);
  const filtered = filterRows(rows, ai.filters, higher, excluded);
  const shown = usingLLM ? byLLMScore(filtered, llm.results) : byScore(filtered, scoring.scores);
  const unscored = shown
    .filter((r) => !(usingLLM ? llm.results[r.id]?.match : scoring.scores[r.id]?.match))
    .map((r) => r.id);
  const busyScoring = usingLLM
    ? llm.running.length > 0 || !!llm.pending
    : scoring.running.length > 0;
  const [details, setDetails] = useState<Row | null>(null);
  const [answer, setAnswer] = useState<{ answer: LLMAnswer; id: string } | null>(null);
  // The Drugs / mechanism dialog and the catalog evidence over it are in the URL, so Back closes them.
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const go = (patch: PatientSearch) => {
    void navigate({ search: (prev) => ({ ...prev, ...patch }), resetScroll: false });
  };
  const evidence = search.evidence ? (
    <CatalogEvidenceDialog
      target={search.evidence}
      onClose={() => {
        go({ evidence: undefined });
      }}
      onOpen={(ref) => {
        go({ evidence: ref });
      }}
      onTrial={(trialId) => {
        go({ drugs: trialId, good: undefined, evidence: undefined });
      }}
    />
  ) : null;
  const goodRow = search.good ? rows.find((r) => r.cohort.trial.id === search.good) : undefined;

  return (
    <>
      {active ? <AiFilterPanel rows={rows} excluded={excluded} higher={higher} /> : null}
      <ResultsLayout result={result}>
        <section className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-xl font-bold">Matched Clinical Trials</h2>
            <Badge variant="secondary">
              {shown.length} {shown.length === 1 ? "trial" : "trials"}
            </Badge>
            {usingLLM ? (
              <LLMRunStatus scoring={llm} noun="trials" />
            ) : (
              <ScoringStatus scoring={scoring} />
            )}
          </div>
          <ScorerToggle />
          {usingLLM ? <LLMRankingNote /> : null}
          <p className="text-sm">
            {result.patient.custom ? (
              "Displaying trial suggestions for the entered summary."
            ) : (
              <>
                Displaying trial suggestions for {noun} <strong>{result.patient.id}</strong>.
              </>
            )}
          </p>
          {stale ? (
            <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-950">
              The patient above has changed since this search. Find trials again to search with it.
            </p>
          ) : null}
          {result.unembedded ? (
            <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-950">
              {result.unembedded} {result.unembedded === 1 ? "cohort" : "cohorts"} with open slots
              {result.unembedded === 1 ? " is" : " are"} still being prepared for search; find
              trials again shortly to include {result.unembedded === 1 ? "it" : "them"}.
            </p>
          ) : null}
          <p className="text-sm text-gray-600 italic">{AI_DISCLAIMER}</p>
          {shown.length === 0 ? (
            <p className="py-6 text-center text-sm text-gray-500">
              {result.open_slots_only && result.results.length === 0
                ? "No cohorts have slots available on Trial management. Turn off Only cohorts with open slots to search them all."
                : "No trials match. Try changing filters to display more trials."}
            </p>
          ) : (
            <ol className="flex flex-col gap-3">
              {shown.map((row) => (
                <TrialCard
                  key={row.id}
                  row={row}
                  scorer={scorer}
                  scores={scoring.scores[row.id]}
                  llm={llm.results[row.id]}
                  scoring={(usingLLM ? llm.running : scoring.running).includes(row.id)}
                  onAnswer={(a) => {
                    setAnswer({ answer: a, id: row.id });
                  }}
                  onDetails={() => {
                    setDetails(row);
                  }}
                  drugsReason={drugsUnavailable(row.cohort.trial, meta)}
                  onDrugs={() => {
                    go({ drugs: row.cohort.trial.id, good: undefined, evidence: undefined });
                  }}
                  goodOption={
                    <GoodOptionButton
                      patientId={result.patient.id}
                      trial={row.cohort.trial}
                      onOpen={() => {
                        go({ good: row.cohort.trial.id, drugs: undefined, evidence: undefined });
                      }}
                    />
                  }
                />
              ))}
            </ol>
          )}
          {unscored.length ? (
            <Button
              variant="outline"
              className="self-start"
              disabled={busyScoring}
              onClick={() => {
                (usingLLM ? llm : scoring).score(unscored.slice(0, 10));
              }}
            >
              Score the next {Math.min(10, unscored.length)} trials
              {usingLLM ? " with the LLM" : ""}
            </Button>
          ) : null}
        </section>
      </ResultsLayout>
      {details ? (
        <TrialDialog
          row={details}
          patientId={result.patient.id}
          scorer={scorer}
          scores={scoring.scores[details.id]}
          llm={llm.results[details.id]}
          scoring={(usingLLM ? llm.running : scoring.running).includes(details.id)}
          onScoreLLM={() => {
            llm.score([details.id]);
          }}
          onAnswer={(a) => {
            setAnswer({ answer: a, id: details.id });
          }}
          onClose={() => {
            setDetails(null);
          }}
        />
      ) : null}
      {answer ? (
        <LLMAnswerDialog
          answer={answer.answer}
          onClose={() => {
            setAnswer(null);
          }}
          onRerun={() => {
            llm.score([answer.id], { rerun: true });
          }}
        />
      ) : null}
      {search.good ? (
        <GoodOptionDialog
          patientId={result.patient.id}
          trialId={search.good}
          trial={goodRow?.cohort.trial}
          onClose={() => {
            go({ good: undefined, evidence: undefined });
          }}
          onEvidence={(ref) => {
            go({ evidence: ref });
          }}
        >
          {evidence}
        </GoodOptionDialog>
      ) : search.drugs ? (
        <TrialDrugsDialog
          trialId={search.drugs}
          onClose={() => {
            go({ drugs: undefined, evidence: undefined });
          }}
          onEvidence={(ref) => {
            go({ evidence: ref });
          }}
        >
          {evidence}
        </TrialDrugsDialog>
      ) : (
        evidence
      )}
    </>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-1.5">
      <dt className="text-gray-500">{label}</dt>
      <dd className="text-gray-800">{children}</dd>
    </div>
  );
}

const SHOWN_SITES = 2;

function Locations({ sites }: { sites: string[] }) {
  const rest = sites.length - SHOWN_SITES;
  if (rest <= 0) return <>{sites.join(", ")}</>;
  return (
    <>
      {sites.slice(0, SHOWN_SITES).join(", ")}{" "}
      <Popover>
        <PopoverTrigger openOnHover delay={150} className="link cursor-pointer">
          +{rest} more
        </PopoverTrigger>
        <PopoverContent align="start" className="w-auto gap-1 p-3">
          <p className="text-xs font-medium text-gray-500">Locations</p>
          <ul className="text-sm">
            {sites.map((site) => (
              <li key={site}>{site}</li>
            ))}
          </ul>
        </PopoverContent>
      </Popover>
    </>
  );
}

/** mm-dash's trial card: title, what it is and where, the cohort it matched, the checks, and its details. */
function TrialCard({
  row,
  scorer,
  scores,
  llm,
  scoring,
  onAnswer,
  onDetails,
  drugsReason,
  onDrugs,
  goodOption,
}: {
  row: Row;
  scorer: Scorer;
  scores: PairScores | undefined;
  llm: LLMPairResult | undefined;
  scoring: boolean;
  onAnswer: (answer: LLMAnswer) => void;
  onDetails: () => void;
  /** Why the trial's drugs can't be shown (the button is disabled), or null. */
  drugsReason: string | null;
  onDrugs: () => void;
  /** The Good option check's button and state. */
  goodOption: ReactNode;
}) {
  const site = useSite();
  const t = row.cohort.trial;
  return (
    <li className="flex gap-4 rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <div className="flex flex-col gap-1">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <h3 className="text-base leading-snug font-semibold text-gray-900">{t.short_title}</h3>
            {scorer === "llm" ? (
              <LLMMatchBadge answer={llm?.match} scoring={scoring} onOpen={onAnswer} />
            ) : (
              <MatchBadge scores={scores} scoring={scoring} />
            )}
          </div>
          <dl className="flex flex-col gap-0.5 text-sm">
            <div className="flex flex-wrap gap-x-5 gap-y-0.5">
              {t.phase ? <Fact label="Phase">{t.phase}</Fact> : null}
              <Fact label="Protocol">{t.protocol_no || t.nct_id}</Fact>
              <Fact label="Managed by">{t.center}</Fact>
              {row.open_slots?.length ? (
                <Fact label="Open slots">
                  <span className="font-semibold text-emerald-800">
                    {row.open_slots
                      .map((s) =>
                        s.id === row.cohort.cohort_id
                          ? String(s.available)
                          : `${String(s.available)} (${s.kind === "subcohort" ? "subcohort" : "original cohort"} ${s.name ?? s.id})`,
                      )
                      .join(" · ")}
                  </span>
                </Fact>
              ) : null}
            </div>
            <div className="flex flex-wrap gap-x-5 gap-y-0.5">
              {t.sites.length ? (
                <Fact label="Locations">
                  <Locations sites={t.sites} />
                </Fact>
              ) : null}
              {t.site_pi ? (
                <Fact label={site.pi_label}>{t.site_pi}</Fact>
              ) : t.overall_pi ? (
                <Fact label="Overall PI">{t.overall_pi}</Fact>
              ) : null}
            </div>
          </dl>
        </div>
        <div className="border-l-2 border-gray-200 pl-3 text-sm">
          <p className="text-xs font-medium text-gray-500">
            {row.cohort.unit?.kind === "subcohort"
              ? `Subcohort${row.cohort.unit.parent_name ? ` of ${row.cohort.unit.parent_name}` : ""}`
              : row.cohort.unit?.kind === "original"
                ? `Original cohort${row.cohort.unit.parent_name ? ` of ${row.cohort.unit.parent_name}` : ""}`
                : "Most relevant trial cohort"}
          </p>
          <p className="font-medium text-gray-900">{row.cohort.name}</p>
          <ul className="mt-0.5 flex flex-col gap-0.5 text-gray-700">
            {row.cohort.key_inclusion.slice(0, 3).map((line, i) => (
              <li key={i} className="line-clamp-2">
                {line}
              </li>
            ))}
          </ul>
        </div>
        {scorer === "llm" ? (
          <>
            <LLMRequirements answer={llm?.match} />
            <LLMExclusionLine answer={llm?.boilerplate} scoring={scoring} onOpen={onAnswer} />
          </>
        ) : (
          <>
            <Requirements match={scores?.match} />
            <ExclusionLine boilerplate={scores?.boilerplate} scoring={scoring} />
          </>
        )}
      </div>
      <div className="flex shrink-0 flex-col items-stretch gap-2">
        <Button size="sm" onClick={onDetails}>
          Trial Details
        </Button>
        <DrugsButton reason={drugsReason} onClick={onDrugs} />
        {goodOption}
      </div>
    </li>
  );
}

/** Opens the trial's drugs and mechanisms (GoodOption catalog); disabled, with a tooltip saying why,
 * when there are none to show. */
function DrugsButton({ reason, onClick }: { reason: string | null; onClick: () => void }) {
  const button = (
    <Button
      size="sm"
      variant="outline"
      disabled={reason !== null}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
    >
      <PillIcon />
      Drugs / mechanism
    </Button>
  );
  if (reason === null) return button;
  return (
    <Tooltip>
      {/* A disabled button gets no pointer or focus events, so the tooltip hangs on a wrapper. */}
      <TooltipTrigger render={<span className="inline-flex flex-col" tabIndex={0} />}>
        {button}
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{reason}</TooltipContent>
    </Tooltip>
  );
}

function ResultsLayout({
  result,
  children,
}: {
  result: TrialsForPatientResult;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-4">
      <PatientPanel patient={result.patient} />
      {children}
    </div>
  );
}

/** The patient panel's default and largest heights, in percent of the window height. */
const PANEL_DEFAULT = 25;
const PANEL_MAX = 75;
/** An arrow key moves the splitter by this much of the window height, as the other splitters do. */
const PANEL_STEP = 5;

/**
 * The patient's summary and common exclusions, stuck to the top of the window while the trials scroll
 * beneath it: a quarter of the window high until the splitter on its bottom edge makes it taller or
 * shorter, each text scrolling in its own box, and collapsible to one line.
 */
function PatientPanel({ patient }: { patient: TrialsForPatientResult["patient"] }) {
  const [collapsed, setCollapsed] = useState(false);
  const id = useId();
  const panel = useRef<HTMLElement>(null);
  const header = useRef<HTMLDivElement>(null);
  const body = useRef<HTMLDivElement>(null);
  useScrollPaddingFor(panel);
  const [limits, measure] = usePanelLimits(panel, header, body, collapsed);
  // Kept with the Trials view's other state, so later searches open at the chosen height until a reload.
  const chosen = useAiState().summaryHeight;
  const [dragged, setDragged] = useState<number | null>(null);
  const height = dragged ?? chosen ?? PANEL_DEFAULT;
  return (
    <aside
      ref={panel}
      id={`${id}-panel`}
      aria-label="Patient"
      className="sticky top-[var(--app-tabs-h,3rem)] z-20 flex flex-col rounded-lg border border-gray-200 bg-slate-50 shadow-sm"
      style={{
        maxHeight: `min(${String(PANEL_MAX)}svh, max(${String(limits?.min ?? 0)}px, ${String(height)}svh))`,
      }}
    >
      <div
        ref={header}
        className={cn(
          "flex shrink-0 items-center gap-2 px-4 py-2",
          !collapsed && "border-b border-gray-200",
        )}
      >
        <h2 className="shrink-0 text-xs font-semibold tracking-wider whitespace-nowrap text-gray-500 uppercase">
          Searched for
        </h2>
        <span className="truncate text-xs whitespace-nowrap text-gray-600">
          {patient.custom ? "Entered summary" : patient.id}
        </span>
        {collapsed ? (
          <span className="min-w-0 flex-1 truncate text-xs text-gray-500">
            {patient.key.replace(/\s+/g, " ")}
          </span>
        ) : null}
        <Button
          variant="outline"
          size="sm"
          className="ml-auto shrink-0"
          aria-expanded={!collapsed}
          aria-controls={`${id}-body`}
          onClick={() => {
            setCollapsed((c) => !c);
          }}
        >
          {collapsed ? "Expand" : "Collapse"}
          <ChevronUpIcon className={cn("transition-transform", collapsed && "rotate-180")} />
        </Button>
      </div>
      <div
        ref={body}
        id={`${id}-body`}
        hidden={collapsed}
        className="grid min-h-0 flex-1 grid-rows-[minmax(0,1fr)] px-4 pt-2 pb-3"
      >
        <SummarySplit
          className="min-h-0"
          history={<PatientText id={`${id}-key`} label="Cancer summary" text={patient.key} />}
          other={
            <PatientText
              id={`${id}-boilerplate`}
              label="Other relevant history"
              text={patient.boilerplate}
            />
          }
        />
      </div>
      <PanelSplitter
        controls={`${id}-panel`}
        hidden={collapsed}
        limits={limits}
        measure={measure}
        onDrag={setDragged}
        onChoose={(share) => {
          setAiState({ summaryHeight: share });
          setDragged(null);
        }}
      />
    </aside>
  );
}

/** The window height and the patient panel's current, smallest and largest heights, in pixels. */
interface PanelLimits {
  vh: number;
  now: number;
  min: number;
  max: number;
}

const percentOf = (px: number, l: PanelLimits) => (100 * px) / l.vh;
const clampTo = (px: number, l: PanelLimits) => Math.min(l.max, Math.max(l.min, px));

/**
 * The panel's heights now. The smallest shows the header and about two lines of the summary (the
 * body's padding, a label and two lines side by side; two labels and a line each when stacked). The
 * largest is three quarters of the window, so the trials stay in view, or the panel's whole content:
 * its height now plus what the texts' boxes scroll out of view.
 */
function measurePanel(el: HTMLElement, header: HTMLElement, body: HTMLElement): PanelLimits {
  const vh = window.innerHeight;
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
  const [first, second] = [...body.children] as (HTMLElement | undefined)[];
  const stacked = !!first && !!second && second.offsetTop > first.offsetTop;
  const min =
    header.offsetHeight + (el.offsetHeight - el.clientHeight) + (stacked ? 6.75 : 5) * rem;
  const unseen = [...body.querySelectorAll<HTMLElement>("[role=region]")].map(
    (box) => box.scrollHeight - box.clientHeight,
  );
  const now = el.offsetHeight;
  const whole = now + (stacked ? unseen.reduce((a, b) => a + b, 0) : Math.max(0, ...unseen));
  const max = Math.min((PANEL_MAX / 100) * vh, whole);
  return { vh, now, min: Math.min(min, max), max };
}

/** The panel's heights, measured while it is open as it, its content or the window change size. */
function usePanelLimits(
  panel: RefObject<HTMLElement | null>,
  header: RefObject<HTMLElement | null>,
  body: RefObject<HTMLElement | null>,
  collapsed: boolean,
) {
  const [limits, setLimits] = useState<PanelLimits | null>(null);
  const measure = useCallback(() => {
    if (!panel.current || !header.current || !body.current) return null;
    const next = measurePanel(panel.current, header.current, body.current);
    setLimits((prev) =>
      prev && (["vh", "now", "min", "max"] as const).every((k) => Math.abs(prev[k] - next[k]) < 0.5)
        ? prev
        : next,
    );
    return next;
  }, [panel, header, body]);
  useLayoutEffect(() => {
    const el = panel.current;
    if (collapsed || !el) return;
    measure();
    const observer = new ResizeObserver(() => {
      measure();
    });
    observer.observe(el);
    window.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [panel, collapsed, measure]);
  return [limits, measure] as const;
}

/**
 * While the panel is resized, the page's scroll anchoring is off: otherwise the browser may scroll the
 * page to hold the trials still (it does in the narrow layouts), which moves a panel not yet stuck to
 * the top away from the pointer. It comes back after the frame that lays out the new height.
 */
let anchoringHolds = 0;
function holdScrollAnchoring() {
  const hold = ++anchoringHolds;
  document.documentElement.style.overflowAnchor = "none";
  return () => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (hold === anchoringHolds)
          document.documentElement.style.removeProperty("overflow-anchor");
      });
    });
  };
}

/** Sets the panel's height while the page holds its scroll position. */
function resizeHolding(resize: () => void) {
  const release = holdScrollAnchoring();
  resize();
  release();
}

/**
 * The splitter on the panel's bottom edge, a window splitter like the other panes'
 * (react-resizable-panels): drag it or use the arrow keys, Home and End for the smallest and largest;
 * double-click or Enter restores the default height.
 */
function PanelSplitter({
  controls,
  hidden,
  limits,
  measure,
  onDrag,
  onChoose,
}: {
  controls: string;
  hidden: boolean;
  limits: PanelLimits | null;
  measure: () => PanelLimits | null;
  onDrag: (share: number) => void;
  onChoose: (share: number | null) => void;
}) {
  const drag = useRef<{
    y: number;
    from: number;
    limits: PanelLimits;
    release: () => void;
    share?: number;
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const stop = () => {
    const d = drag.current;
    drag.current = null;
    document.body.style.removeProperty("cursor");
    document.body.style.removeProperty("user-select");
    d?.release();
    return d;
  };
  const end = (e: PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    if (e.currentTarget.hasPointerCapture(e.pointerId))
      e.currentTarget.releasePointerCapture(e.pointerId);
    const d = stop();
    setDragging(false);
    // A click without a move keeps the default height a default.
    if (d?.share !== undefined) onChoose(d.share);
  };
  useEffect(
    () => () => {
      stop();
    },
    [],
  );
  const percent = (px: number) => (limits ? Math.round(percentOf(px, limits)) : undefined);
  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      aria-label="Resize patient summary"
      aria-controls={controls}
      aria-valuenow={limits ? percent(limits.now) : undefined}
      aria-valuemin={limits ? percent(limits.min) : undefined}
      aria-valuemax={limits ? percent(limits.max) : undefined}
      aria-valuetext={limits ? `${String(percent(limits.now))}% of the window height` : undefined}
      tabIndex={0}
      hidden={hidden}
      title="Drag or use the arrow keys to resize · double-click or Enter to reset"
      data-dragging={dragging || undefined}
      className="group absolute inset-x-0 -bottom-1 flex h-4 cursor-row-resize touch-none items-center justify-center rounded-b-lg outline-none select-none focus-visible:ring-3 focus-visible:ring-ring/50"
      onPointerDown={(e) => {
        if (e.button !== 0) return;
        const l = measure();
        if (!l) return;
        e.preventDefault();
        e.currentTarget.focus({ preventScroll: true });
        e.currentTarget.setPointerCapture(e.pointerId);
        drag.current = { y: e.clientY, from: l.now, limits: l, release: holdScrollAnchoring() };
        setDragging(true);
        document.body.style.cursor = "row-resize";
        document.body.style.userSelect = "none";
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d) return;
        d.share = percentOf(clampTo(d.from + e.clientY - d.y, d.limits), d.limits);
        onDrag(d.share);
      }}
      onPointerUp={end}
      onPointerCancel={end}
      onLostPointerCapture={end}
      onDoubleClick={() => {
        resizeHolding(() => {
          onChoose(null);
        });
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          resizeHolding(() => {
            onChoose(null);
          });
          return;
        }
        const l = measure();
        if (!l) return;
        const step = (PANEL_STEP / 100) * l.vh;
        const to: Partial<Record<string, number>> = {
          ArrowDown: l.now + step,
          ArrowUp: l.now - step,
          Home: l.min,
          End: l.max,
        };
        const px = to[e.key];
        if (px === undefined) return;
        e.preventDefault();
        resizeHolding(() => {
          onChoose(percentOf(clampTo(px, l), l));
        });
      }}
    >
      <div className="h-1 w-8 rounded-lg bg-gray-300 transition-colors group-hover:bg-gray-400 group-focus-visible:bg-gray-500 group-data-dragging:bg-gray-500" />
    </div>
  );
}

function PatientText({ id, label, text }: { id: string; label: string; text: string }) {
  return (
    <section className="flex min-h-0 flex-col gap-1">
      <h3 id={id} className="shrink-0 text-xs font-medium text-gray-500">
        {label}
      </h3>
      {text.trim() ? (
        // Focusable so the keyboard can scroll a long text.
        <div
          role="region"
          aria-labelledby={id}
          tabIndex={0}
          className="scrollbar-visible min-h-0 flex-1 overflow-y-auto rounded-sm font-mono text-sm break-words whitespace-pre-wrap outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          {text}
        </div>
      ) : (
        <p className="text-sm text-gray-500 italic">None entered</p>
      )}
    </section>
  );
}

/** While the panel shows, the window scrolls a focused control into view below it, not under it. */
function useScrollPaddingFor(panel: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const el = panel.current;
    if (!el) return;
    const root = document.documentElement;
    const observer = new ResizeObserver(() => {
      root.style.scrollPaddingTop = `${String(el.offsetHeight + 16)}px`;
    });
    observer.observe(el);
    return () => {
      observer.disconnect();
      root.style.removeProperty("scroll-padding-top");
    };
  }, [panel]);
}

/** mm-dash's AI filters (phase, managing group, location, the higher-threshold groups, likely excluded). */
function AiFilterPanel({
  rows,
  excluded,
  higher,
}: {
  rows: Row[];
  excluded: Excluded;
  higher: string[];
}) {
  const site = useSite();
  const { filters } = useAiState();
  const set = (patch: Partial<AiFilters>) => {
    setAiState((s) => ({ filters: { ...s.filters, ...patch } }));
  };
  const closed = new Set(rows.filter((r) => !r.cohort.trial.is_open).map((r) => r.cohort.trial.id))
    .size;
  const options = (facet: "phases" | "centers" | "locations", values: (r: Row) => string[]) => {
    const others = filterRows(rows, { ...filters, [facet]: [] }, higher, excluded);
    return [...new Set([...others.flatMap(values), ...filters[facet]])].sort();
  };
  const facets: {
    facet: "phases" | "centers" | "locations";
    label: string;
    values: (r: Row) => string[];
  }[] = [
    { facet: "phases", label: "Phase", values: (r) => [r.cohort.trial.phase] },
    { facet: "centers", label: "Managed by", values: (r) => [r.cohort.trial.center] },
    { facet: "locations", label: "Locations", values: (r) => r.cohort.trial.sites },
  ];
  return (
    <SidebarSlot>
      <div className="mt-2 flex flex-col gap-4 border-t border-white/20 pt-4 pb-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Filters</h2>
          <button
            type="button"
            className="text-xs text-gray-400 underline hover:text-white"
            onClick={() => {
              set({
                phases: [],
                centers: [],
                locations: [],
                excludeHigherThreshold: true,
                hideLikelyExcluded: false,
                showClosed: false,
              });
            }}
          >
            Clear All
          </button>
        </div>
        <Label className="items-start gap-1.5 text-xs font-normal text-gray-200">
          <Checkbox
            className="mt-px size-3.5"
            checked={filters.excludeHigherThreshold}
            onCheckedChange={(v) => {
              set({ excludeHigherThreshold: v });
            }}
          />
          {site.higher_threshold_label}
        </Label>
        <Label className="items-start gap-1.5 text-xs font-normal text-gray-200">
          <Checkbox
            className="mt-px size-3.5"
            checked={filters.hideLikelyExcluded}
            onCheckedChange={(v) => {
              set({ hideLikelyExcluded: v });
            }}
          />
          Hide trials with likely common exclusions
        </Label>
        {closed ? (
          <Label className="items-start gap-1.5 text-xs font-normal text-gray-200">
            <Checkbox
              className="mt-px size-3.5"
              checked={filters.showClosed}
              onCheckedChange={(v) => {
                set({ showClosed: v });
              }}
            />
            <span className="flex-1">Show closed trials</span>
            <span className="text-white/40 tabular-nums">{closed}</span>
          </Label>
        ) : null}
        {facets.map(({ facet, label, values }) => {
          const opts = options(facet, values);
          if (!opts.length) return null;
          return (
            <fieldset key={facet} className="flex flex-col gap-1.5">
              <legend className="mb-1.5 text-xs font-semibold tracking-wider text-sidebar-foreground/60 uppercase">
                {label}
              </legend>
              {opts.map((value) => (
                <Label
                  key={value}
                  className="items-start gap-1.5 text-xs leading-snug font-normal text-gray-200"
                >
                  <Checkbox
                    className="mt-px size-3.5 shrink-0"
                    checked={filters[facet].includes(value)}
                    onCheckedChange={(checked) => {
                      set({
                        [facet]: checked
                          ? [...filters[facet], value]
                          : filters[facet].filter((v) => v !== value),
                      });
                    }}
                  />
                  {value || "Not given"}
                </Label>
              ))}
            </fieldset>
          );
        })}
      </div>
    </SidebarSlot>
  );
}
