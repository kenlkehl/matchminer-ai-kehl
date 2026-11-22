import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { type CohortHit, type PairScores, VERSION_LABELS } from "@/lib/api";
import type { LLMPairResult, Scorer } from "@/lib/llm-decisions";
import { cohortOptions } from "@/lib/queries";

import { CriteriaCheck } from "../../matching/criteria-check";
import type { LLMAnswer } from "../../matching/llm-score-view";
import { PairScoresView } from "../../matching/patients-for-cohort-page";
import { getPatient, recordsBody } from "../store";

/** A matched trial's cohort: its facts, the session scorer's answers and the per-criterion check. */
export function TrialDialog({
  row,
  patientId,
  scorer,
  scores,
  llm,
  scoring,
  onScoreLLM,
  onAnswer,
  onClose,
}: {
  row: CohortHit;
  patientId: string;
  scorer: Scorer;
  scores: PairScores | undefined;
  llm: LLMPairResult | undefined;
  scoring: boolean;
  onScoreLLM: () => void;
  onAnswer: (answer: LLMAnswer) => void;
  onClose: () => void;
}) {
  const cohort = useQuery(cohortOptions(row.cohort.cohort_id));
  const t = row.cohort.trial;
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90svh] sm:max-w-3xl" bodyClassName="gap-4">
        <DialogHeader>
          <DialogTitle>
            {t.protocol_no || t.nct_id}: {t.short_title}
          </DialogTitle>
          <DialogDescription className="text-gray-600">
            {row.cohort.name} · {t.phase} · {t.center} · {VERSION_LABELS[row.cohort.version]}{" "}
            criteria
          </DialogDescription>
        </DialogHeader>
        <PairScoresView
          scorer={scorer}
          scores={scores}
          llm={llm}
          scoring={scoring}
          onScoreLLM={onScoreLLM}
          onAnswer={onAnswer}
        />
        {cohort.data ? (
          <CriteriaCheck
            patientId={patientId}
            cohortId={row.cohort.cohort_id}
            items={cohort.data.items}
            records={() => ({ ...recordsBody(getPatient()), levels: "document_and_patient" })}
          />
        ) : (
          <Spinner className="size-5" />
        )}
        <DialogFooter className="pt-1">
          <Button
            variant="outline"
            render={<Link to="/" search={{ center: t.center_id, details: t.id }} />}
          >
            Open on the dashboard
          </Button>
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
