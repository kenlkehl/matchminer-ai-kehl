import { createFileRoute } from "@tanstack/react-router";

import { PatientPage } from "@/features/patient/patient-page";
import { validatePatientSearch } from "@/features/patient/store";

export const Route = createFileRoute("/_app/patient/")({
  validateSearch: validatePatientSearch,
  component: PatientPage,
});
