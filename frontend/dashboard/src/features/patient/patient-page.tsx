import { getRouteApi } from "@tanstack/react-router";
import { useEffect } from "react";

import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { restoreSession } from "./actions";
import { GuidelinesView } from "./guidelines/guidelines-view";
import { PatientInput } from "./patient-input";
import { type PatientView, usePatientValue } from "./store";
import { TrialsView } from "./trials/trials-view";

const route = getRouteApi("/_app/patient/");

/**
 * The Patient tab: one patient (chosen from the corpus, entered as a summary, or made from clinical
 * notes) on top, and two views that use it: Trials (matching trial cohorts) and Guidelines (guideline
 * spaces and the documented workup). The sub-tab is the `view` search param.
 */
export function PatientPage() {
  const view = route.useSearch({ select: (s) => s.view ?? "trials" });
  const navigate = route.useNavigate();
  const guidelineJob = usePatientValue(
    (s) => s.job?.kind === "retrieve" || s.job?.kind === "workup",
  );
  const spaces = usePatientValue((s) => s.retrieval?.spaces.length ?? null);
  useEffect(() => {
    void restoreSession();
  }, []);
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Patient">
        <p>
          Choose a patient, enter a summary, or add clinical notes. Trials matches the patient's
          cancer summary against the open trials' cohorts; Guidelines finds the guideline
          populations that fit the patient and checks the documented workup in the notes.
        </p>
      </PageHeader>
      <PatientInput />
      <Tabs
        value={view}
        onValueChange={(value: PatientView) => {
          void navigate({
            search: (prev) => ({
              ...prev,
              view: value === "guidelines" ? "guidelines" : undefined,
              drugs: undefined,
              good: undefined,
              evidence: undefined,
            }),
          });
        }}
      >
        <TabsList variant="line" className="w-full justify-start border-b">
          <TabsTrigger value="trials" className="flex-none px-3">
            Trials
          </TabsTrigger>
          <TabsTrigger value="guidelines" className="flex-none px-3">
            Guidelines
            {guidelineJob ? (
              <Spinner className="size-3.5" />
            ) : spaces !== null ? (
              <Badge variant="secondary">{spaces}</Badge>
            ) : null}
          </TabsTrigger>
        </TabsList>
        {/* Kept mounted so its scores and dialogs survive a look at Guidelines. */}
        <TabsContent value="trials" keepMounted className="pt-2">
          <TrialsView active={view === "trials"} />
        </TabsContent>
        <TabsContent value="guidelines" className="pt-2">
          <GuidelinesView />
        </TabsContent>
      </Tabs>
    </div>
  );
}
