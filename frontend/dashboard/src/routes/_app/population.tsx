import { createFileRoute } from "@tanstack/react-router";

import { PopulationPage } from "@/features/population/population-page";

export const Route = createFileRoute("/_app/population")({
  component: PopulationPage,
});
