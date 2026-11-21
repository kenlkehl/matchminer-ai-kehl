import { createFileRoute } from "@tanstack/react-router";

import { AdjudicationPage } from "@/features/benchmarking/adjudication-page";

export interface AdjudicationSearch {
  component?: string;
  item?: string;
}

export const Route = createFileRoute("/_app/benchmarking/judge/$runId/adjudicate")({
  validateSearch: (search: Record<string, unknown>): AdjudicationSearch => ({
    component:
      typeof search.component === "string" && search.component ? search.component : undefined,
    item: typeof search.item === "string" && search.item ? search.item : undefined,
  }),
  component: AdjudicationPage,
});
