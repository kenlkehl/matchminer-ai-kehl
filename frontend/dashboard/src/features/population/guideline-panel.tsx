import { useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCwIcon } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { api } from "@/lib/api";

import { Helper, LabeledField, SELECT, Step } from "./controls";
import { useRemembered } from "./remember";
import type { CatalogInfo, GuidelineSpace } from "./types";
import { errorMessage, type PopulationController } from "./use-population";

type Catalogs = { catalogs: CatalogInfo[]; spaces: number };

const CATALOGS_KEY = ["population", "catalogs"] as const;

const spaceText = (s: GuidelineSpace) => s.cohort_key_criteria || s.clinical_space_summary || "";

/** 02: choose a finalized guideline space and fetch the loaded population's matching patients. */
export function GuidelinePanel({ pop }: { pop: PopulationController }) {
  const client = useQueryClient();
  const catalogs = useQuery({
    queryKey: CATALOGS_KEY,
    queryFn: () => api.get<Catalogs>("/api/guidelines/catalogs?refresh=false"),
    staleTime: 60_000,
  });
  const [catalogId, setCatalogId] = useRemembered("population.catalog", "");
  const [filter, setFilter] = useRemembered("population.filter", "");
  const [spaceId, setSpaceId] = useRemembered("population.space", "");
  const [candidates, setCandidates] = useRemembered("population.candidates", "50");
  const [threshold, setThreshold] = useRemembered("population.threshold", "0.75");
  const [refreshing, setRefreshing] = useState(false);

  const listed = catalogs.data?.catalogs ?? [];
  const catalog = listed.some((c) => c.id === catalogId) ? catalogId : "";
  const spaces = useQuery({
    queryKey: ["population", "spaces", catalog],
    queryFn: () =>
      api.get<{ spaces: GuidelineSpace[] }>(
        `/api/population/spaces?catalog_id=${encodeURIComponent(catalog)}`,
      ),
    enabled: !!catalog,
    staleTime: 5 * 60_000,
  });
  const visible = useMemo(() => {
    const query = filter.toLocaleLowerCase();
    return (spaces.data?.spaces ?? []).filter((s) =>
      `${s.name ?? ""} ${s.space_trial_id} ${spaceText(s)}`.toLocaleLowerCase().includes(query),
    );
  }, [spaces.data, filter]);
  // A space the filter hides is not selected (as in the fetcher); clearing the filter brings it back.
  const space = visible.find((s) => s.space_trial_id === spaceId) ?? null;

  const refresh = async () => {
    setRefreshing(true);
    try {
      const fresh = await api.get<Catalogs>("/api/guidelines/catalogs?refresh=true");
      client.setQueryData(CATALOGS_KEY, fresh);
      await client.invalidateQueries({ queryKey: ["population", "spaces"] });
    } catch (err) {
      pop.setError(errorMessage(err));
    } finally {
      setRefreshing(false);
    }
  };

  const ready = !!pop.population && !!space;
  return (
    <Step number="02" eyebrow="Guideline population" title="Find matching patients">
      <form
        className="flex flex-col gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (!pop.population || !space) return;
          void pop.start("/api/population/retrieve", {
            version: pop.population.version,
            catalog_id: catalog,
            space_trial_id: space.space_trial_id,
            candidate_k: Number(candidates),
            min_reasonableness: Number(threshold),
          });
        }}
      >
        <LabeledField label="Finalized guideline catalog" htmlFor="pop-catalog">
          <div className="flex gap-1.5">
            <select
              id="pop-catalog"
              className={SELECT}
              value={catalog}
              onChange={(e) => {
                setCatalogId(e.target.value);
                setSpaceId("");
              }}
            >
              <option value="">
                {catalogs.isPending ? "Loading catalogs…" : "Choose a catalog"}
              </option>
              {listed.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.title} · {c.version} · {c.spaces} spaces
                </option>
              ))}
            </select>
            <Button
              type="button"
              variant="outline"
              size="icon-sm"
              title="Refresh catalogs"
              aria-label="Refresh catalogs"
              disabled={refreshing}
              onClick={() => void refresh()}
            >
              {refreshing ? <Spinner /> : <RefreshCwIcon />}
            </Button>
          </div>
        </LabeledField>
        {catalogs.isError ? (
          <Helper className="text-red-700">{errorMessage(catalogs.error)}</Helper>
        ) : null}
        <LabeledField label="Filter trial spaces" htmlFor="pop-filter">
          <Input
            id="pop-filter"
            type="search"
            placeholder="Cancer, stage, biomarker…"
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
            }}
          />
        </LabeledField>
        <LabeledField label="Guideline trial space" htmlFor="pop-space">
          <select
            id="pop-space"
            className={SELECT}
            value={space?.space_trial_id ?? ""}
            onChange={(e) => {
              setSpaceId(e.target.value);
            }}
          >
            <option value="">
              {!catalog
                ? "Choose a catalog first"
                : spaces.isPending
                  ? "Loading spaces…"
                  : `${String(visible.length)} spaces · choose one`}
            </option>
            {visible.map((s) => (
              <option key={s.space_trial_id} value={s.space_trial_id}>
                {s.name || (s.clinical_space_summary ?? s.space_trial_id).slice(0, 180)}
              </option>
            ))}
          </select>
        </LabeledField>
        {spaces.isError ? (
          <Helper className="text-red-700">{errorMessage(spaces.error)}</Helper>
        ) : null}
        <div className="flex flex-col gap-1">
          <p className="text-xs font-medium text-gray-700">Selected population key criteria</p>
          <p className="max-h-40 overflow-y-auto rounded-md border border-gray-200 bg-gray-50 p-2 text-xs break-words whitespace-pre-wrap text-gray-800">
            {space ? spaceText(space) || "No key criteria recorded." : "No space selected."}
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <LabeledField label="Candidates to score" htmlFor="pop-candidates">
            <Input
              id="pop-candidates"
              type="number"
              required
              min={1}
              max={100000}
              step={1}
              value={candidates}
              onChange={(e) => {
                setCandidates(e.target.value);
              }}
            />
          </LabeledField>
          <LabeledField label="Minimum reasonableness" htmlFor="pop-threshold">
            <Input
              id="pop-threshold"
              type="number"
              required
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(e) => {
                setThreshold(e.target.value);
              }}
            />
          </LabeledField>
        </div>
        <Helper>
          TrialSpace searches the full loaded population. The decision model's match questions score
          up to this many candidates. A patient is kept when the decision model's reasonableness
          answer (its probability that the patient is potentially eligible) is at least this value;
          0 keeps all candidates. Kept patients are ranked by match quality.
        </Helper>
        <Button type="submit" className="w-full" disabled={!ready}>
          Fetch matching patients
        </Button>
      </form>
    </Step>
  );
}
