import { useSyncExternalStore } from "react";

import type { TrialsForPatientResult } from "@/lib/api";

/** The Trials view's filters, as mm-dash's (MAIA's), narrowing the retrieved trials on the client. */
export interface AiFilters {
  phases: string[];
  centers: string[];
  locations: string[];
  /** Leave out the groups that need a higher match threshold (the site config's higher_threshold_groups). */
  excludeHigherThreshold: boolean;
  /** Leave out the trials whose common-exclusion check found the patient likely excluded. */
  hideLikelyExcluded: boolean;
  /** Show trials that are not open to enrollment (hidden by default). */
  showClosed: boolean;
}

export const DEFAULT_FILTERS: AiFilters = {
  phases: [],
  centers: [],
  locations: [],
  excludeHigherThreshold: true,
  hideLikelyExcluded: false,
  showClosed: false,
};

/**
 * What the Trials view holds while other views and tabs show: the last search's results and the
 * filters (the patient searched for is the Patient tab's shared patient). In memory only.
 */
export interface TrialsState {
  result: TrialsForPatientResult | null;
  allCohorts: boolean;
  /** Search only cohorts with slots available on Trial management (off by default). */
  openSlotsOnly: boolean;
  filters: AiFilters;
  searches: number;
  /**
   * The results' patient panel height chosen with its splitter, in percent of the window height;
   * null is the default.
   */
  summaryHeight: number | null;
}

const initial: TrialsState = {
  result: null,
  allCohorts: false,
  openSlotsOnly: false,
  filters: DEFAULT_FILTERS,
  searches: 0,
  summaryHeight: null,
};

let state: TrialsState = initial;
const listeners = new Set<() => void>();

export function setAiState(
  patch: Partial<TrialsState> | ((s: TrialsState) => Partial<TrialsState>),
) {
  state = { ...state, ...(typeof patch === "function" ? patch(state) : patch) };
  for (const listener of listeners) listener();
}

/** Forget the results (Clear workspace), keeping the panel height chosen. */
export function resetTrials() {
  setAiState((s) => ({ ...initial, summaryHeight: s.summaryHeight, searches: s.searches + 1 }));
}

export function useAiState(): TrialsState {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => state,
  );
}
