import { useQuery } from "@tanstack/react-query";
import { TrashIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { PageHeader } from "@/components/page-header";
import { Alert, AlertAction, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { isActive, type Job } from "@/lib/api";
import { metaOptions } from "@/lib/queries";

import { GuidelinePanel } from "./guideline-panel";
import { ItemDialog } from "./item-dialog";
import { PatientDialog } from "./patient-dialog";
import { ActiveCohort, Distribution, Metrics } from "./results";
import { ReviewPanel } from "./review-panel";
import { SourcePanel } from "./source-panel";
import { type PopulationController, usePopulation } from "./use-population";

const JOB_TITLES: Record<string, string> = {
  "population-sample": "Loading the population sample",
  "population-cache": "Loading the cached population",
  "population-tables": "Loading the cohort tables",
  "population-retrieve": "Fetching matching patients",
  "population-compress": "Compressing cohort notes",
  "population-review": "Checking guideline concordance",
};

/**
 * The Population tab: load a patient population (a synthetic sample; in real mode a population cache or
 * a DuckDB sample; or a cohort of one's own), fetch the patients matching a guideline space, then check
 * their workup documentation against the guideline, with drilldowns by item and by patient.
 */
const GAP = 16; // px kept between the workflow pane and the window's edges

/**
 * On wide screens, keep the workflow pane's bottom inside the window wherever the page is scrolled: its height is
 * the room from its top (or the sticky offset) to the window's bottom, so scrolling the pane alone always reaches
 * its last control.
 */
function useFitToWindow() {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const wide = window.matchMedia("(min-width: 64rem)");
    const fit = () => {
      if (!wide.matches) {
        element.style.maxHeight = "";
        return;
      }
      const top = Math.max(element.getBoundingClientRect().top, GAP);
      element.style.maxHeight = `${String(Math.max(200, window.innerHeight - top - GAP))}px`;
    };
    fit();
    window.addEventListener("resize", fit);
    document.addEventListener("scroll", fit, { passive: true, capture: true });
    wide.addEventListener("change", fit);
    return () => {
      window.removeEventListener("resize", fit);
      document.removeEventListener("scroll", fit, { capture: true });
      wide.removeEventListener("change", fit);
    };
  }, []);
  return ref;
}

export function PopulationPage() {
  const workflow = useFitToWindow();
  const mode = useQuery(metaOptions()).data?.mode;
  const pop = usePopulation();
  const version = pop.population?.version;
  const report = pop.population?.active?.report ?? null;
  const [item, setItem] = useState<{ version: string; number: number } | null>(null);
  const [patient, setPatient] = useState<{
    version: string;
    id: string;
    item: number | null;
  } | null>(null);
  const reload = () => {
    void pop.reload();
  };

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Population guideline concordance">
        <p>Identify a patient population and evaluate documented care for guideline concordance.</p>
      </PageHeader>
      {pop.error ? (
        <Alert variant="destructive">
          <AlertDescription className="text-destructive">{pop.error}</AlertDescription>
          {!pop.ready ? (
            <AlertAction>
              <Button
                size="sm"
                variant="outline"
                disabled={pop.initializing}
                onClick={() => void pop.init()}
              >
                Retry population connection
              </Button>
            </AlertAction>
          ) : null}
        </Alert>
      ) : null}
      <div className="grid items-start gap-4 lg:grid-cols-[22rem_minmax(0,1fr)]">
        {/* The workflow steps are their own pane: on wide screens it stays in view and scrolls by itself, so the
            results beside it stay visible. */}
        <aside
          ref={workflow}
          aria-label="Population workflow"
          className="scrollbar-visible rounded-xl bg-card shadow-xs ring-1 ring-foreground/10 lg:sticky lg:top-4 lg:overflow-y-auto"
        >
          {mode ? (
            <fieldset disabled={!pop.ready || pop.busy} className="min-w-0">
              <SourcePanel pop={pop} mode={mode} />
              <GuidelinePanel pop={pop} />
              <ReviewPanel pop={pop} />
            </fieldset>
          ) : (
            <div className="p-4">
              <Spinner />
            </div>
          )}
        </aside>
        <div className="flex min-w-0 flex-col gap-4">
          <Progress pop={pop} />
          {pop.notice && !pop.busy ? (
            <p role="status" className="rounded-md bg-gray-100 px-3 py-2 text-sm text-gray-700">
              {pop.notice}
            </p>
          ) : null}
          <div className="flex flex-wrap items-start gap-3">
            <Metrics population={pop.population} />
            <Button
              variant="outline"
              size="sm"
              disabled={pop.busy || !pop.population}
              onClick={() => void pop.clear()}
            >
              <TrashIcon />
              Clear population
            </Button>
          </div>
          <ActiveCohort
            pop={pop}
            onPatient={(id) => {
              if (version) setPatient({ version, id, item: null });
            }}
          />
          <Distribution
            pop={pop}
            onItem={(number) => {
              if (version) setItem({ version, number });
            }}
          />
        </div>
      </div>
      {/* A new cohort or run (a new version) closes the drilldowns opened on the old one. */}
      {item && item.version === version && report ? (
        <ItemDialog
          version={item.version}
          number={item.number}
          report={report}
          onClose={() => {
            setItem(null);
          }}
          onPatient={(id) => {
            setPatient({ version: item.version, id, item: item.number });
          }}
          onStale={reload}
        />
      ) : null}
      {patient && patient.version === version ? (
        <PatientDialog
          version={patient.version}
          patientId={patient.id}
          item={patient.item}
          report={report}
          onClose={() => {
            setPatient(null);
          }}
          onStale={reload}
        />
      ) : null}
    </div>
  );
}

/** The running job: what it does, its progress and Stop. */
function Progress({ pop }: { pop: PopulationController }) {
  const job: Job | null = pop.job;
  if (!job || !isActive(job)) return null;
  const canStop = !!job.id && isActive(job) && job.status !== "cancelling";
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-3 rounded-xl bg-brand-blue/5 px-4 py-3 ring-1 ring-brand-blue/20"
    >
      <Spinner className="size-5 text-brand-blue" />
      <div className="min-w-0 flex-1 text-sm">
        <p className="font-medium">
          {job.status === "cancelling" ? "Stopping…" : (JOB_TITLES[job.kind] ?? "Working…")}
        </p>
        <p className="break-words text-gray-700">
          {job.message}
          {job.total ? ` · ${String(job.done)}/${String(job.total)}` : ""}
          {job.seconds ? ` · ${String(Math.round(job.seconds))} s` : ""}
        </p>
      </div>
      <Button variant="outline" size="sm" disabled={!canStop} onClick={() => void pop.stop()}>
        Stop
      </Button>
    </div>
  );
}
