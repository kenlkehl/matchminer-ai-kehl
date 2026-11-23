import { type ReactNode, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { Mode } from "@/lib/api";

import { BrowseDialog, type BrowseKind } from "./browse-dialog";
import { CheckStatus, Fold, Helper, LabeledField, Step, usePathCheck } from "./controls";
import { cacheLine, count, databaseLine } from "./labels";
import { useRemembered } from "./remember";
import type { CacheInfo, DatabaseInfo, PopulationView } from "./types";
import type { PopulationController } from "./use-population";

/** 01: load the session's source population (synthetic sample, or real cache / DuckDB), or a cohort of one's own. */
export function SourcePanel({ pop, mode }: { pop: PopulationController; mode: Mode }) {
  const defaults = pop.defaults;
  const [size, setSize] = useRemembered<string | null>("population.size", null);
  const [seed, setSeed] = useRemembered<string | null>("population.seed", null);
  const [includeTraining, setIncludeTraining] = useRemembered("population.includeTraining", false);
  const sizeValue = size ?? (defaults ? String(defaults.size) : "");
  const seedValue = seed ?? (defaults ? String(defaults.seed) : "");
  const real = mode === "real";

  const sampleFields = (
    <div className="grid grid-cols-2 gap-2">
      <LabeledField label="Sample size" htmlFor="pop-size">
        <Input
          id="pop-size"
          type="number"
          required
          min={1}
          max={100000}
          step={1}
          value={sizeValue}
          onChange={(e) => {
            setSize(e.target.value);
          }}
        />
      </LabeledField>
      <LabeledField label="Random seed" htmlFor="pop-seed">
        <Input
          id="pop-seed"
          type="number"
          required
          min={0}
          max={4294967295}
          step={1}
          value={seedValue}
          onChange={(e) => {
            setSeed(e.target.value);
          }}
        />
      </LabeledField>
    </div>
  );
  const sampleBody = () => ({ size: Number(sizeValue), seed: Number(seedValue) });

  return (
    <Step number="01" eyebrow="Source population" title="Load patients">
      {real ? (
        <RealSources pop={pop} sampleFields={sampleFields} sampleBody={sampleBody} />
      ) : (
        <>
          <Helper>Random sample of local synthetic summaries and their original notes.</Helper>
          <form
            className="flex flex-col gap-3"
            onSubmit={(e) => {
              e.preventDefault();
              void pop.start("/api/population/sample", {
                ...sampleBody(),
                include_training: includeTraining,
              });
            }}
          >
            {sampleFields}
            {defaults?.split ? (
              <Label className="items-start gap-1.5 text-xs font-normal text-gray-700">
                <Checkbox
                  className="mt-px size-3.5"
                  checked={includeTraining}
                  onCheckedChange={(v) => {
                    setIncludeTraining(v);
                  }}
                />
                <span>
                  Include training patients. Otherwise patients come from the run&apos;s{" "}
                  {defaults.split} split, which the decision model never trained on; its scores for
                  training patients are optimistic.
                </span>
              </Label>
            ) : null}
            <Button type="submit" variant="outline" className="w-full">
              Load synthetic population
            </Button>
          </form>
          <SourceStatus population={pop.population} />
        </>
      )}
      <OwnCohort pop={pop} />
    </Step>
  );
}

function SourceStatus({ population: p }: { population: PopulationView | null }) {
  const origin = p?.provenance.cache?.name ?? p?.provenance.database?.name;
  return (
    <Helper role="status" className="font-medium text-gray-800">
      {p
        ? `${count(p.source_count)} patients · ${count(p.source_note_count)} ${p.notes_on_demand ? "cached source notes, read for matched patients" : "source notes loaded"}${origin ? ` from ${origin}` : ""}`
        : "No population loaded."}
    </Helper>
  );
}

/** Real mode: a preprocessed population cache, or a sample from a DuckDB database. */
function RealSources({
  pop,
  sampleFields,
  sampleBody,
}: {
  pop: PopulationController;
  sampleFields: ReactNode;
  sampleBody: () => { size: number; seed: number };
}) {
  const defaults = pop.defaults;
  const [cache, setCache] = useRemembered<string | null>("population.cache", null);
  const [database, setDatabase] = useRemembered<string | null>("population.database", null);
  const cacheValue = cache ?? defaults?.cache ?? "";
  const databaseValue = database ?? defaults?.database ?? "";
  const cacheCheck = usePathCheck<CacheInfo>("cache", "/api/population/cache");
  const databaseCheck = usePathCheck<DatabaseInfo>("database", "/api/population/database");
  const [browsing, setBrowsing] = useState<{ kind: BrowseKind; start: string } | null>(null);

  // Check the configured (or remembered) paths once the defaults arrive.
  const checkedOnce = useRef(false);
  useEffect(() => {
    if (!defaults || checkedOnce.current) return;
    checkedOnce.current = true;
    cacheCheck.check(cacheValue);
    databaseCheck.check(databaseValue);
  }, [defaults, cacheValue, databaseValue, cacheCheck, databaseCheck]);

  const changed = (
    check: { check: (p: string) => void; checkedPath: string | null },
    v: string,
  ) => {
    if (check.checkedPath !== v.trim()) check.check(v);
  };

  return (
    <>
      <Helper>
        Load every real patient from a preprocessed population cache: current summaries with
        precomputed TrialSpace embeddings. Notes are read for matched patients. Build or refresh the
        cache with{" "}
        <code className="text-[0.7rem] break-all">
          python -m matchminer_ai.apps.dashboard.guidelines.preprocess_population
        </code>
        .
      </Helper>
      <form
        className="flex flex-col gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void pop.start("/api/population/cache", { path: cacheValue.trim() || null });
        }}
      >
        <LabeledField label="Preprocessed population cache" htmlFor="pop-cache">
          <PathInput
            id="pop-cache"
            value={cacheValue}
            onChange={setCache}
            onCommit={(v) => {
              changed(cacheCheck, v);
            }}
            onBrowse={() => {
              // Start beside the current cache rather than inside it.
              setBrowsing({ kind: "cache", start: cacheValue.trim().replace(/\/[^/]*\/?$/, "") });
            }}
          />
        </LabeledField>
        <CheckStatus
          id="pop-cache-status"
          query={cacheCheck.query}
          checking="Checking cache manifest…"
          describe={cacheLine}
        />
        <Button type="submit" variant="outline" className="w-full">
          Load cached population
        </Button>
      </form>
      <SourceStatus population={pop.population} />
      <Fold summary="Sample from a DuckDB database">
        <Helper>
          Random sample of patients with a current summary, with all of their source notes. The
          database is opened read-only, and summaries are embedded at each search.
        </Helper>
        <form
          className="flex flex-col gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void pop.start("/api/population/sample", {
              ...sampleBody(),
              database: databaseValue.trim() || null,
            });
          }}
        >
          <LabeledField label="DuckDB database" htmlFor="pop-database">
            <PathInput
              id="pop-database"
              value={databaseValue}
              onChange={setDatabase}
              onCommit={(v) => {
                changed(databaseCheck, v);
              }}
              onBrowse={() => {
                setBrowsing({ kind: "database", start: databaseValue.trim() });
              }}
            />
          </LabeledField>
          <CheckStatus
            id="pop-database-status"
            query={databaseCheck.query}
            checking="Checking database tables…"
            describe={databaseLine}
          />
          {sampleFields}
          <Button type="submit" variant="outline" className="w-full">
            Load sample from database
          </Button>
        </form>
      </Fold>
      {browsing ? (
        <BrowseDialog
          kind={browsing.kind}
          start={browsing.start}
          onClose={() => {
            setBrowsing(null);
          }}
          onPick={(path) => {
            setBrowsing(null);
            if (browsing.kind === "cache") {
              setCache(path);
              cacheCheck.check(path);
            } else {
              setDatabase(path);
              databaseCheck.check(path);
            }
          }}
        />
      ) : null}
    </>
  );
}

/** A server path with a Browse… button; it is checked when the entry changes or on Enter. */
function PathInput({
  id,
  value,
  onChange,
  onCommit,
  onBrowse,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
  onCommit: (value: string) => void;
  onBrowse: () => void;
}) {
  return (
    <div className="flex gap-1.5">
      <Input
        id={id}
        type="text"
        spellCheck={false}
        autoComplete="off"
        aria-describedby={`${id}-status`}
        className="font-mono text-xs md:text-xs"
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
        }}
        onBlur={(e) => {
          onCommit(e.target.value);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            onCommit(e.currentTarget.value);
          }
        }}
      />
      <Button type="button" variant="outline" size="sm" onClick={onBrowse}>
        Browse…
      </Button>
    </div>
  );
}

/** Both modes: a cohort's own summary and note tables, or an exported cohort archive. */
function OwnCohort({ pop }: { pop: PopulationController }) {
  const summaries = useRef<HTMLInputElement>(null);
  const notes = useRef<HTMLInputElement>(null);
  const archive = useRef<HTMLInputElement>(null);
  return (
    <Fold summary="Use your own cohort">
      <Helper>
        CSV or Parquet. Summaries: patient_id, cancer_history_summary. Notes: patient_id, note_text,
        optional note_date and note_type. All patients need notes. Combined limit: 512 MB.
      </Helper>
      <form
        className="flex flex-col gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          const s = summaries.current?.files?.[0];
          const n = notes.current?.files?.[0];
          if (!s || !n) {
            pop.setError("Choose both a summary table and a notes table.");
            return;
          }
          void pop.start("/api/population/tables", null, { summaries: s, notes: n });
        }}
      >
        <LabeledField label="Summary table" htmlFor="pop-summary-file">
          <Input
            ref={summaries}
            id="pop-summary-file"
            type="file"
            accept=".csv,.parquet"
            required
            className="text-xs md:text-xs"
          />
        </LabeledField>
        <LabeledField label="Notes table" htmlFor="pop-notes-file">
          <Input
            ref={notes}
            id="pop-notes-file"
            type="file"
            accept=".csv,.parquet"
            required
            className="text-xs md:text-xs"
          />
        </LabeledField>
        <Button type="submit" variant="outline" className="w-full">
          Load cohort tables
        </Button>
      </form>
      <form
        className="flex flex-col gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          const file = archive.current?.files?.[0];
          if (!file) {
            pop.setError("Choose an exported cohort ZIP.");
            return;
          }
          // A job: the archive is read, then its patients are embedded for the searches.
          void pop.start("/api/population/import", undefined, { file });
        }}
      >
        <LabeledField label="Restore an exported cohort" htmlFor="pop-archive">
          <Input
            ref={archive}
            id="pop-archive"
            type="file"
            accept=".zip"
            required
            className="text-xs md:text-xs"
          />
        </LabeledField>
        <Button type="submit" variant="outline" className="w-full">
          Import cohort archive
        </Button>
      </form>
    </Fold>
  );
}
