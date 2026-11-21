import { queryOptions } from "@tanstack/react-query";

import { api } from "@/lib/api";

import type { AdjudicationItem, AdjudicationState, FidelityRun, JudgeRun, RunList } from "./types";

const enc = encodeURIComponent;

/** The runs under the mode's benchmark roots (the server rescans them every 60 s). */
export const runsOptions = () =>
  queryOptions({
    queryKey: ["benchmarks"],
    queryFn: () => api.get<RunList>("/api/benchmarks"),
    refetchInterval: 30_000,
  });

/** A fidelity run; polled while any of its stages is still to run. */
export const fidelityOptions = (runId: string) =>
  queryOptions({
    queryKey: ["benchmarks", "fidelity", runId],
    queryFn: () => api.get<FidelityRun>(`/api/benchmarks/fidelity/${enc(runId)}`),
    refetchInterval: (query) => {
      const jobs = Object.values(query.state.data?.jobs ?? {});
      return jobs.some((j) => j.state === "running" || j.state === "pending") ? 15_000 : false;
    },
  });

export const judgeOptions = (runId: string) =>
  queryOptions({
    queryKey: ["benchmarks", "judge", runId],
    queryFn: () => api.get<JudgeRun>(`/api/benchmarks/judge/${enc(runId)}`),
    refetchInterval: 60_000,
  });

/** Every item of a judge run with its queued and decided counts (a 400 until compare has run). */
export const adjudicationStateOptions = (runId: string, enabled = true) =>
  queryOptions({
    queryKey: ["benchmarks", "judge", runId, "adjudication"],
    queryFn: () => api.get<AdjudicationState>(`/api/benchmarks/judge/${enc(runId)}/adjudication`),
    enabled,
    retry: false,
  });

export const adjudicationItemOptions = (
  runId: string,
  component: string,
  itemId: string,
  reference: boolean,
) =>
  queryOptions({
    queryKey: ["benchmarks", "judge", runId, "adjudication", "item", component, itemId, reference],
    queryFn: () =>
      api.get<AdjudicationItem>(
        `/api/benchmarks/judge/${enc(runId)}/adjudication/item?component=${enc(component)}` +
          `&item_id=${enc(itemId)}&reference=${String(reference)}`,
      ),
    placeholderData: (previous) => previous,
    retry: false,
  });
