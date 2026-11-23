/**
 * Clinical trial management's Manage a trial pane (the server's trial_management.py and routes/management.py):
 * slots, edited criteria, subcohorts and group pins, kept in one file shared by everyone using this dashboard.
 */
import { queryOptions, useMutation, useQueryClient } from "@tanstack/react-query";

import { api, type Job, type Slots, type TrialSummary, type UnitKind, type Version } from "./api";

export type LabeledBy = "ai" | "classifier" | "manual" | null;

export const HEADINGS = ["Inclusion criteria:", "Exclusion criteria:"] as const;

/** A criteria line as the pipeline splits criteria, with its labels and who gave them (none: unclassified). */
export interface ManagedLine {
  heading: string;
  text: string;
  criterion_type: "inclusion" | "exclusion" | null;
  boilerplate: boolean;
  labeled_by: LabeledBy;
}

export interface StaleBase {
  reason: "ai_changed" | "parent_changed" | "parent_missing" | "cohort_missing";
  base_lines?: ManagedLine[];
  current_lines?: ManagedLine[];
}

export interface ManagedUnit {
  id: string;
  kind: UnitKind;
  name: string;
  number: number | null;
  version: Version;
  /** A subcohort's parent, an original cohort's merged cohort. */
  parent: string | null;
  merged_from: number;
  merged_into: string | null;
  /** A merged cohort's original cohorts (when it stands for two or more). */
  originals: string[];
  /** The subcohorts defined under this unit. */
  children: string[];
  lines: ManagedLine[];
  /** The current AI-extracted lines (merged and original cohorts). */
  ai_lines: ManagedLine[] | null;
  edited: boolean;
  stale: StaleBase | null;
  /** A subcohort reset to its parent's criteria: its criteria before, for reference. */
  previous_lines: ManagedLine[] | null;
  slots: Slots | null;
  unclassified: number;
  classifying: boolean;
  classification_error: string | null;
  group: { id: string; name: string; pinned: boolean } | null;
  /** Pinned to a group that no longer exists. */
  pin_missing: boolean;
  revision: number;
  created_at: string | null;
  key_count: number;
  line_count: number;
}

export interface Orphan {
  id: string;
  kind: UnitKind;
  name: string | null;
  slots: Slots | null;
  edited: boolean;
  parent: string | null;
  version: Version;
}

export interface LogEntry {
  at: string;
  action: string;
  unit: string | null;
  trial_id: string | null;
  detail: string;
}

export interface TrialManagementView {
  trial: TrialSummary;
  version: Version | null;
  center_id: string;
  units: ManagedUnit[];
  orphans: Orphan[];
  /** Why the trial's original cohorts are not shown, when they are not. */
  originals_reason: string | null;
  groups: { id: string; name: string; custom: boolean }[];
  log: LogEntry[];
  revision: number;
}

export interface ManagementStatus {
  path: string;
  error: string | null;
  revision: number;
  updated_at: string | null;
  counts: { subcohorts: number; edited: number; with_slots: number; groups: number; pins: number };
  classifier: {
    connection: { label: string; configured: boolean };
    unavailable: string | null;
  };
}

/** A criteria change: the units whose lines changed, and their classification (a job, or why it was skipped). */
export interface ChangeResult {
  changed?: string[];
  classifying: boolean;
  skipped: string | null;
  job?: Job;
}

export const managementStatusOptions = () =>
  queryOptions({
    queryKey: ["management", "status"],
    queryFn: () => api.get<ManagementStatus>("/api/management"),
    staleTime: 15_000,
  });

export const trialManagementOptions = (trialId: string) =>
  queryOptions({
    queryKey: ["management", "trial", trialId],
    queryFn: () =>
      api.get<TrialManagementView>(`/api/management/trials/${encodeURIComponent(trialId)}`),
    // Classification runs in the background: poll while a unit's lines are being classified.
    refetchInterval: (query) =>
      query.state.data?.units.some((u) => u.classifying) ? 2_000 : 30_000,
  });

export const KIND_LABELS: Record<UnitKind, string> = {
  merged: "Cohort",
  original: "Original cohort",
  subcohort: "Subcohort",
};

/** "Merged cohort" for a cohort standing for several extracted cohorts. */
export function kindLabel(unit: { kind: UnitKind; merged_from?: number }): string {
  return unit.kind === "merged" && (unit.merged_from ?? 1) > 1
    ? "Merged cohort"
    : KIND_LABELS[unit.kind];
}

const enc = encodeURIComponent;

/** A change, then every view that shows cohorts, slots or groups is refetched. */
export function useManagementChange<A, R = unknown>(run: (args: A) => Promise<R>) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: run,
    onSettled: async () => {
      await Promise.all(
        [
          "management",
          "center",
          "cohort",
          "trial",
          "patients-for-cohort",
          "llm-criteria",
          "meta",
        ].map((key) => client.invalidateQueries({ queryKey: [key] })),
      );
    },
  });
}

export const management = {
  slots: (unitId: string, slots: Omit<Slots, "updated_at">, revision: number) =>
    api.put<{ ok: boolean }>(`/api/management/units/${enc(unitId)}/slots`, { ...slots, revision }),
  criteria: (unitId: string, items: ManagedLine[], revision: number, propagate: boolean | null) =>
    api.put<ChangeResult>(`/api/management/units/${enc(unitId)}/criteria`, {
      items,
      revision,
      propagate,
    }),
  revert: (unitId: string, revision: number, propagate: boolean | null) =>
    api.post<ChangeResult>(`/api/management/units/${enc(unitId)}/revert`, { revision, propagate }),
  keep: (unitId: string, revision: number) =>
    api.post(`/api/management/units/${enc(unitId)}/keep`, { revision }),
  resetToParent: (unitId: string, revision: number) =>
    api.post<ChangeResult>(`/api/management/units/${enc(unitId)}/reset-to-parent`, { revision }),
  dismissPrevious: (unitId: string) =>
    api.post(`/api/management/units/${enc(unitId)}/dismiss-previous`, {}),
  classify: (unitId: string) =>
    api.post<ChangeResult>(`/api/management/units/${enc(unitId)}/classify`, {}),
  remove: (unitId: string) => api.delete(`/api/management/units/${enc(unitId)}`),
  createSubcohort: (parentId: string, name: string) =>
    api.post<ChangeResult & { id: string }>("/api/management/subcohorts", {
      parent_id: parentId,
      name,
    }),
  renameSubcohort: (unitId: string, name: string, revision: number) =>
    api.patch(`/api/management/subcohorts/${enc(unitId)}`, { name, revision }),
  createGroup: (centerId: string, name: string) =>
    api.post<{ id: string }>(`/api/management/centers/${enc(centerId)}/groups`, { name }),
  renameGroup: (centerId: string, groupId: string, name: string) =>
    api.patch(`/api/management/centers/${enc(centerId)}/groups/${enc(groupId)}`, { name }),
  deleteGroup: (centerId: string, groupId: string) =>
    api.delete(`/api/management/centers/${enc(centerId)}/groups/${enc(groupId)}`),
  pin: (unitId: string, groupId: string | null) =>
    api.put(`/api/management/pins/${enc(unitId)}`, { group_id: groupId }),
  importFile: (file: File) =>
    api.upload<ManagementStatus>("/api/management/import", { file, confirm: "replace" }),
};

/** The drag-and-drop payload of a cohort (Manage a trial's cards and the group rows). */
export const UNIT_MIME = "application/x-mmai-cohort";

export function slotText(slots: Slots | null | undefined): string {
  if (!slots) return "";
  const part = (value: number | null, label: string) =>
    value === null ? null : `${String(value)} ${label}`;
  return [
    part(slots.available, "available"),
    part(slots.consented, "consented"),
    part(slots.enrolled, "enrolled"),
  ]
    .filter(Boolean)
    .join(" · ");
}
