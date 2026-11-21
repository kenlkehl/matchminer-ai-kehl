import { createFileRoute } from "@tanstack/react-router";

import { BenchmarksPage } from "@/features/benchmarking/benchmarks-page";

export const Route = createFileRoute("/_app/benchmarking/")({
  component: BenchmarksPage,
});
