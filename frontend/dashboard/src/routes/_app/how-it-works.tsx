import { createFileRoute } from "@tanstack/react-router";

import { DocsPage } from "@/features/docs/docs-page";

export const Route = createFileRoute("/_app/how-it-works")({
  validateSearch: (search: Record<string, unknown>): { page?: string } => ({
    page: typeof search.page === "string" && search.page ? search.page : undefined,
  }),
  component: DocsPage,
});
