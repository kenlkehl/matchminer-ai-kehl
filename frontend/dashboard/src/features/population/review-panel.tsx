import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { ArrowUpRightIcon } from "lucide-react";
import { useEffect, useRef } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { connectionLabel, modelsOptions } from "@/lib/models";

import { CheckStatus, Helper, LabeledField, SELECT, Step, usePathCheck } from "./controls";
import { isExtractionMethod, METHOD_OPTIONS, storeLine } from "./labels";
import { useRemembered } from "./remember";
import type { ExtractionStoreInfo, ReviewMethod } from "./types";
import type { PopulationController } from "./use-population";

/** The cohort version whose report last set the review method. */
let syncedVersion: string | null = null;

/** 03: check the active cohort's guideline concordance. */
export function ReviewPanel({ pop }: { pop: PopulationController }) {
  const population = pop.population;
  const active = population?.active ?? null;
  const patients = active?.patient_count ?? 0;
  const workup = active?.space?.diagnostic_workup?.length ?? 0;
  const models = useQuery(modelsOptions()).data;

  const [method, setMethod] = useRemembered<ReviewMethod>("population.method", "full_notes");
  const [concurrency, setConcurrency] = useRemembered("population.concurrency", "4");
  const [records, setRecords] = useRemembered<string | null>("population.extractions", null);
  const recordsValue = records ?? pop.defaults?.extraction_records ?? "";
  const store = usePathCheck<ExtractionStoreInfo>("extractions", "/api/population/extractions");
  const extraction = isExtractionMethod(method);

  // A new cohort or run shows the method its report used (once per version, so a choice made since
  // survives leaving the tab).
  const report = active?.report;
  useEffect(() => {
    if (!population || syncedVersion === population.version) return;
    syncedVersion = population.version;
    if (report?.method) setMethod(report.method);
  }, [population, report, setMethod]);

  // Check the store when an extraction method is chosen (and on the path's changes).
  const storeChecked = useRef(false);
  useEffect(() => {
    if (!extraction || !pop.defaults || storeChecked.current) return;
    storeChecked.current = true;
    store.check(recordsValue);
  }, [extraction, pop.defaults, recordsValue, store]);

  const task = (name: "workup" | "workup_sql") =>
    models ? connectionLabel(models, models.tasks[name]) : "…";

  return (
    <Step number="03" eyebrow="Review" title="Active cohort notes">
      {models?.notice ? (
        <p role="note" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-900">
          {models.notice}
        </p>
      ) : null}
      <form
        className="flex flex-col gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (!population) return;
          void pop.start(
            "/api/population/review",
            extraction
              ? {
                  version: population.version,
                  method,
                  extraction_records: recordsValue.trim() || null,
                }
              : {
                  version: population.version,
                  method,
                  max_concurrent_patients: Number(concurrency),
                },
          );
        }}
      >
        <LabeledField label="Review method" htmlFor="pop-method">
          <select
            id="pop-method"
            className={SELECT}
            value={method}
            onChange={(e) => {
              setMethod(e.target.value as ReviewMethod);
            }}
          >
            {METHOD_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </LabeledField>
        {extraction ? (
          <>
            <LabeledField label="Extraction records" htmlFor="pop-extractions">
              <Input
                id="pop-extractions"
                type="text"
                spellCheck={false}
                autoComplete="off"
                aria-describedby="pop-extractions-status"
                className="font-mono text-xs md:text-xs"
                value={recordsValue}
                onChange={(e) => {
                  setRecords(e.target.value);
                }}
                onBlur={(e) => {
                  if (store.checkedPath !== e.target.value.trim()) store.check(e.target.value);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    store.check(e.currentTarget.value);
                  }
                }}
              />
            </LabeledField>
            <CheckStatus
              id="pop-extractions-status"
              query={store.query}
              checking="Checking extraction records…"
              describe={storeLine}
            />
            <Helper>
              Built offline with{" "}
              <code className="text-[0.7rem] break-all">
                python -m matchminer_ai.apps.dashboard.guidelines.preprocess_extractions
              </code>{" "}
              from this population's original notes.
            </Helper>
            {method === "extraction_sql" ? (
              <Helper>
                For each workup item, an LLM agent explores these records with read-only SQL (no
                files, network or other tools) and accepts one query, cached for this store; code
                then runs it once over the active cohort's extracted items. The agent's query
                results (extracted values from the whole store, without patient ids, note text or
                evidence quotes) go to the SQL agent's connection ({task("workup_sql")}): use only
                an endpoint authorized for these records. Results link each item to its quote in the
                source note; notes without records make a result unknown.
              </Helper>
            ) : (
              <Helper>
                Each workup item is compiled once into a query (cached; only guideline text goes to
                the concordance connection), then the query runs in code on every patient's
                extracted items. No patient text reaches an LLM in this step. Results link each item
                to its quote in the source note; notes without records make a result unknown.
              </Helper>
            )}
          </>
        ) : (
          <>
            <LabeledField label="Patients reviewed at once" htmlFor="pop-concurrency">
              <Input
                id="pop-concurrency"
                type="number"
                required
                min={1}
                max={32}
                step={1}
                value={concurrency}
                onChange={(e) => {
                  setConcurrency(e.target.value);
                }}
              />
            </LabeledField>
            <Helper>
              Each patient's notes are reviewed in full, chunk by chunk, for every workup item. Each
              request contains only one patient's notes.
            </Helper>
          </>
        )}
        <Button type="submit" className="w-full" disabled={!patients || !workup}>
          Check guideline concordance
        </Button>
      </form>
      <Helper>
        Guideline concordance: {task("workup")}
        {models?.tasks.workup_full_backup.backup_enabled && !extraction
          ? ", with its full-note backup"
          : ""}
        . Set the model connections in{" "}
        <Link to="/status" className="inline-flex items-center text-brand-blue hover:underline">
          Data and models
          <ArrowUpRightIcon className="size-3" />
        </Link>
        .
      </Helper>
    </Step>
  );
}
