import { createFileRoute } from "@tanstack/react-router";

import { JudgeRunPage } from "@/features/benchmarking/judge-run-page";

export const Route = createFileRoute("/_app/benchmarking/judge/$runId/")({
  component: JudgeRunPage,
});
