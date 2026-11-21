import { createFileRoute } from "@tanstack/react-router";

import { PatientsForCohortPage } from "@/features/matching/patients-for-cohort-page";

interface PatientsSearch {
  cancer_type?: string;
  sex?: string;
  k?: number;
  /** Only patients with stored structured records. */
  records?: boolean;
}

export const Route = createFileRoute("/_app/cohorts/$cohortId/patients")({
  validateSearch: (search: Record<string, unknown>): PatientsSearch => ({
    cancer_type:
      typeof search.cancer_type === "string" && search.cancer_type ? search.cancer_type : undefined,
    sex: typeof search.sex === "string" && search.sex ? search.sex : undefined,
    k: typeof search.k === "number" ? search.k : undefined,
    records: search.records === true || search.records === "true" ? true : undefined,
  }),
  component: PatientsForCohortPage,
});
