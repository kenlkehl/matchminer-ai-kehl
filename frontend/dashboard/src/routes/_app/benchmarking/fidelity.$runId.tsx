import { createFileRoute } from "@tanstack/react-router";

import { FidelityRunPage } from "@/features/benchmarking/fidelity-run-page";

export const Route = createFileRoute("/_app/benchmarking/fidelity/$runId")({
  component: FidelityRunPage,
});
