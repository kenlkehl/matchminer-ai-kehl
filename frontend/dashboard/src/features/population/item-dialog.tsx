import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api } from "@/lib/api";

import { LabeledField, SELECT } from "./controls";
import {
  APPLICABILITY_LABELS,
  EXTRACTION_REASON_LABELS,
  isExtractionMethod,
  methodLabels,
} from "./labels";
import type { ItemPatient, PopulationReport } from "./types";
import { errorMessage } from "./use-population";

/** One workup item's patients, filtered by status; a patient opens their drilldown. */
export function ItemDialog({
  version,
  number,
  report,
  onPatient,
  onClose,
  onStale,
}: {
  version: string;
  number: number;
  report: PopulationReport;
  onPatient: (patientId: string) => void;
  onClose: () => void;
  onStale: () => void;
}) {
  const patients = useQuery({
    queryKey: ["population", "item", version, number],
    queryFn: async () => {
      try {
        return await api.get<{ patients: ItemPatient[] }>(
          `/api/population/items/${String(number)}?version=${encodeURIComponent(version)}`,
        );
      } catch (err) {
        onStale();
        throw err;
      }
    },
    retry: false,
    staleTime: Infinity,
  });
  const [filter, setFilter] = useState("");
  const names = methodLabels(report.method);
  const recommendation = report.aggregate[number]?.recommendation;
  const extraction = isExtractionMethod(report.method);
  const visible = (patients.data?.patients ?? []).filter((p) => !filter || p.status === filter);
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="h-[85svh] sm:max-w-5xl" bodyClassName="gap-4 auto-rows-min">
        <DialogHeader>
          <DialogTitle>{recommendation?.name ?? "Workup patients"}</DialogTitle>
          {recommendation?.conditions ? (
            <DialogDescription className="whitespace-pre-wrap">
              {recommendation.conditions}
            </DialogDescription>
          ) : null}
        </DialogHeader>
        <LabeledField
          label="Filter documentation status"
          htmlFor="pop-status-filter"
          className="max-w-sm"
        >
          <select
            id="pop-status-filter"
            className={SELECT}
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
            }}
          >
            <option value="">All patients</option>
            {Object.entries(names).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </LabeledField>
        {patients.isPending ? (
          <Spinner />
        ) : patients.isError ? (
          <p className="text-sm text-red-700">{errorMessage(patients.error)}</p>
        ) : (
          <Table variant="list">
            <TableCaption className="mt-2 text-left">{visible.length} patients shown</TableCaption>
            <TableHeader>
              <TableRow>
                <TableHead>Patient</TableHead>
                <TableHead>Documentation</TableHead>
                <TableHead>Applicability</TableHead>
                <TableHead>Finding</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {visible.map((p) => (
                <TableRow key={p.patient_id}>
                  <TableCell className="align-top">
                    <button
                      type="button"
                      className="text-left font-medium text-brand-blue hover:underline"
                      onClick={() => {
                        onPatient(p.patient_id);
                      }}
                    >
                      {p.patient_id}
                    </button>
                  </TableCell>
                  <TableCell className="align-top">
                    {names[p.status] ?? p.status}
                    {extraction && p.reason ? (
                      <span className="block text-xs text-gray-500">
                        {EXTRACTION_REASON_LABELS[p.reason] ?? p.reason}
                      </span>
                    ) : null}
                  </TableCell>
                  <TableCell className="align-top">
                    {APPLICABILITY_LABELS[p.applicability]}
                  </TableCell>
                  <TableCell className="align-top break-words whitespace-pre-wrap">
                    {p.bottom_line}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </DialogContent>
    </Dialog>
  );
}
