import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { cn } from "cn";
import {
  ChevronDownIcon,
  ChevronRightIcon,
  DownloadIcon,
  GripVerticalIcon,
  Maximize2Icon,
  PanelRightCloseIcon,
  PencilIcon,
  PlusIcon,
  Trash2Icon,
  UploadIcon,
  UsersIcon,
} from "lucide-react";
import { type ReactNode, useEffect, useRef, useState } from "react";

import { CriteriaList } from "@/components/criteria-list";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { download, type TrialSummary, VERSION_LABELS } from "@/lib/api";
import { isoWhen } from "@/lib/format";
import {
  type ChangeResult,
  type ClearResult,
  kindLabel,
  management,
  type ManagedLine,
  type ManagedUnit,
  managementStatusOptions,
  managementSummaryOptions,
  removedText,
  slotText,
  type TrialBadges,
  trialManagementOptions,
  type TrialManagementView,
  useManagementChange,
} from "@/lib/management";

import { CriteriaEditor } from "./criteria-editor";
import { dragUnit } from "./drag";
import { LineageDiagram } from "./lineage-diagram";

/** Lines as the criteria list shows them. */
function asItems(lines: ManagedLine[]) {
  return lines.map((l, i) => ({
    number: i + 1,
    heading: l.heading,
    text: l.text,
    criterion_type: l.criterion_type,
    boilerplate: l.boilerplate,
  }));
}

/**
 * Trial management, the sidebar beside the center's groups: every trial the page's search and filters show (with
 * a quick filter of its own), each opening in place to its cohorts (merged cohorts, the original cohorts they stand
 * for, and subcohorts) with their slots, criteria and groups. A cohort card dragged onto a group on the left is shown
 * in that group. Every change is saved at once to the dashboard's trial management file, shared by everyone using
 * it; Export and Import save and load it.
 */
export function TrialManagementSidebar({
  centerId,
  trials,
  managementRevision,
  trialId,
  cohortId,
  onTrial,
  onClose,
}: {
  centerId: string;
  /** The trials the page's search and filters show. */
  trials: TrialSummary[];
  managementRevision: number | undefined;
  trialId: string | undefined;
  /** The cohort to show in the open trial (the groups' Manage trial). */
  cohortId: string | undefined;
  onTrial: (trialId: string | undefined) => void;
  onClose: () => void;
}) {
  const [text, setText] = useState("");
  const summary = useQuery(managementSummaryOptions(centerId, managementRevision)).data;
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = trials.filter((t) => {
    if (!words.length) return true;
    const hay = [t.protocol_no, t.nct_id, t.short_title, t.site_pi].join(" ").toLowerCase();
    return words.every((w) => hay.includes(w));
  });
  // The open trial stays listed even when the filters would hide it.
  const open =
    trialId && !shown.some((t) => t.id === trialId) ? trials.find((t) => t.id === trialId) : null;
  return (
    <aside
      aria-label="Trial management"
      data-sidebar-scroll
      className="sticky top-[var(--app-tabs-h,3rem)] flex max-h-[calc(100svh_-_var(--app-tabs-h,3rem))] flex-col overflow-y-auto print:hidden"
    >
      <div
        data-sidebar-header
        className="sticky top-0 z-10 flex flex-col gap-2 border-b border-gray-200 bg-slate-50 px-3 pt-3 pb-2"
      >
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-bold text-gray-900">Trial management</h2>
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Close Trial management"
            className="ml-auto text-gray-500"
            onClick={onClose}
          >
            <PanelRightCloseIcon />
          </Button>
        </div>
        <FileActions />
        <p className="text-xs text-gray-600">
          Open a trial to keep its cohorts' slots, criteria and subcohorts. Drag a cohort onto a
          group on the left to show it there. Changes are saved as you make them and shared with
          everyone using this dashboard.
        </p>
        <Input
          aria-label="Filter the trials"
          placeholder="Filter by protocol, NCT number, title or PI…"
          className="h-8 bg-white"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
          }}
        />
        <p className="text-xs text-gray-500">
          {shown.length} of {trials.length} trials shown by the page's search and filters
        </p>
      </div>
      <ul className="flex flex-col">
        {[...(open ? [open] : []), ...shown].map((t) => (
          <TrialRow
            key={t.id}
            trial={t}
            badges={summary?.trials[t.id]}
            open={t.id === trialId}
            cohortId={t.id === trialId ? cohortId : undefined}
            onToggle={() => {
              onTrial(t.id === trialId ? undefined : t.id);
            }}
          />
        ))}
        {!shown.length && !open ? (
          <li className="px-3 py-4 text-sm text-gray-500">No trials match.</li>
        ) : null}
      </ul>
    </aside>
  );
}

function TrialRow({
  trial,
  badges,
  open,
  cohortId,
  onToggle,
}: {
  trial: TrialSummary;
  badges: TrialBadges | undefined;
  open: boolean;
  cohortId: string | undefined;
  onToggle: () => void;
}) {
  const row = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (open) scrollInSidebar(row.current, "nearest");
  }, [open]);
  const Chevron = open ? ChevronDownIcon : ChevronRightIcon;
  return (
    <li
      ref={row}
      className={cn(
        "border-b border-gray-200",
        open
          ? "bg-white"
          : "[contain-intrinsic-size:auto_4rem] [content-visibility:auto] hover:bg-white/70",
      )}
    >
      <button
        type="button"
        aria-expanded={open}
        className="flex w-full items-start gap-2 px-3 py-2 text-left"
        onClick={onToggle}
      >
        <Chevron className="mt-0.5 size-4 shrink-0 text-gray-400" />
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="text-sm font-semibold text-gray-900">
            {trial.protocol_no || trial.nct_id}
            {trial.protocol_no ? (
              <span className="font-normal text-gray-500"> · {trial.nct_id}</span>
            ) : null}
          </span>
          <span className="line-clamp-2 text-xs text-gray-600" title={trial.short_title}>
            {trial.short_title}
          </span>
          {badges ? <TrialBadgeLine badges={badges} /> : null}
        </span>
      </button>
      {open ? (
        <div className="px-3 pb-3">
          <TrialManager trialId={trial.id} cohortId={cohortId} />
        </div>
      ) : null}
    </li>
  );
}

/** A trial's slot totals (when no cohort family has slots at two levels) and what is kept for it. */
function TrialBadgeLine({ badges }: { badges: TrialBadges }) {
  return (
    <span className="flex flex-wrap gap-1 pt-0.5">
      {badges.slots ? (
        <Badge tone="green">Slots: {slotText({ ...badges.slots, updated_at: null })}</Badge>
      ) : badges.with_slots ? (
        <Badge tone="green">
          Slots in {badges.with_slots} cohorts{badges.overlap ? " (several levels, not added)" : ""}
        </Badge>
      ) : null}
      {badges.subcohorts ? (
        <Badge tone="teal">
          {badges.subcohorts} {badges.subcohorts === 1 ? "subcohort" : "subcohorts"}
        </Badge>
      ) : null}
      {badges.edited ? <Badge tone="blue">{badges.edited} edited</Badge> : null}
      {badges.stale ? <Badge tone="amber">{badges.stale} to review</Badge> : null}
    </span>
  );
}

/** Export (download) and Import (upload, after confirming) of the trial management file. */
function FileActions() {
  const status = useQuery(managementStatusOptions()).data;
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [clearing, setClearing] = useState(false);
  const replace = useManagementChange((chosen: File) => management.importFile(chosen));
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        size="xs"
        variant="outline"
        onClick={() => {
          void download("/api/management/export", "trial_management.json").catch((e: unknown) => {
            setMessage(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
          });
        }}
      >
        <DownloadIcon />
        Export
      </Button>
      <Button
        size="xs"
        variant="outline"
        onClick={() => {
          input.current?.click();
        }}
      >
        <UploadIcon />
        Import
      </Button>
      <input
        ref={input}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={(e) => {
          setFile(e.target.files?.[0] ?? null);
          e.target.value = "";
        }}
      />
      {status ? (
        <span className={cn("text-xs", status.error ? "text-red-700" : "text-gray-500")}>
          {status.error
            ? status.error
            : status.updated_at
              ? `Last change ${isoWhen(status.updated_at)}`
              : "No changes yet"}
        </span>
      ) : null}
      <Button
        size="xs"
        variant="destructive"
        className="ml-auto"
        onClick={() => {
          setClearing(true);
        }}
      >
        <Trash2Icon />
        Clear all
      </Button>
      {message ? <span className="w-full text-xs text-gray-700">{message}</span> : null}
      {clearing && status ? (
        <ClearAllDialog
          status={status}
          onClose={() => {
            setClearing(false);
          }}
          onCleared={(result) => {
            setClearing(false);
            setMessage(
              `Cleared ${removedText(result.removed)}. The file as it was is kept on the server as ${result.backup}; import it to undo.`,
            );
          }}
        />
      ) : null}
      {file ? (
        <Dialog
          open
          onOpenChange={(open) => {
            if (!open) setFile(null);
          }}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Replace the trial management file?</DialogTitle>
              <DialogDescription>
                Every slot count, edited criteria, subcohort and group of this dashboard is replaced
                by those in {file.name}, for everyone using it. The current file is kept on the
                server beside the new one.
              </DialogDescription>
            </DialogHeader>
            {replace.isError ? (
              <p className="text-sm text-red-700">Not imported: {replace.error.message}</p>
            ) : null}
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => {
                  setFile(null);
                }}
              >
                Cancel
              </Button>
              <Button
                disabled={replace.isPending}
                onClick={() => {
                  replace.mutate(file, {
                    onSuccess: () => {
                      setMessage(`Imported ${file.name}.`);
                      setFile(null);
                    },
                  });
                }}
              >
                {replace.isPending ? "Importing…" : "Replace"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      ) : null}
    </div>
  );
}

/** What labeling an opened trial's original cohorts did: running, nothing to do, or why it could not. */
function useOpenTrial(trialId: string) {
  const client = useQueryClient();
  const [notice, setNotice] = useState<{ tone: "info" | "warn"; text: string } | null>(null);
  const asked = useRef<string | null>(null);
  useEffect(() => {
    if (asked.current === trialId) return;
    asked.current = trialId;
    void management
      .openTrial(trialId)
      .then((result) => {
        if (!result.units.length) return;
        const n = result.units.length;
        const which = `${String(n)} original ${n === 1 ? "cohort's" : "cohorts'"}`;
        setNotice(
          result.classifying
            ? {
                tone: "info",
                text: `Labeling the criteria of ${which} that its cohort's labels do not cover…`,
              }
            : {
                tone: "warn",
                text: `The criteria of ${which} that its cohort's labels do not cover could not be labeled: ${
                  result.skipped ?? "no working connection"
                } Choose a connection for Criteria classification on Data and models, then use Classify.`,
              },
        );
        void client.invalidateQueries({ queryKey: ["management", "trial", trialId] });
      })
      .catch((e: unknown) => {
        setNotice({
          tone: "warn",
          text: `The original cohorts' criteria could not be labeled: ${e instanceof Error ? e.message : String(e)}`,
        });
      });
  }, [trialId, client]);
  return notice;
}

function TrialManager({ trialId, cohortId }: { trialId: string; cohortId: string | undefined }) {
  const query = useQuery(trialManagementOptions(trialId));
  const opened = useOpenTrial(trialId);
  const [editing, setEditing] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(cohortId ?? null);
  // The groups' Manage trial names a cohort: it is highlighted and scrolled to once its card is shown.
  const loaded = Boolean(query.data);
  const shownAt = useRef<string | null>(null);
  useEffect(() => {
    if (!cohortId || !loaded || shownAt.current === cohortId) return;
    shownAt.current = cohortId;
    setSelected(cohortId);
    requestAnimationFrame(() => {
      scrollInSidebar(document.getElementById(`unit-${cohortId}`), "center");
    });
  }, [cohortId, loaded]);
  const [notice, setNotice] = useState("");
  const [enlarged, setEnlarged] = useState(false);
  const [resetting, setResetting] = useState(false);
  const data = query.data;
  if (!data) {
    return query.isError ? (
      <p className="text-sm text-red-700">Could not load the trial: {query.error.message}</p>
    ) : (
      <div className="flex justify-center py-4">
        <Spinner className="size-5" />
      </div>
    );
  }
  const units = new Map(data.units.map((u) => [u.id, u]));
  const editUnit = editing ? units.get(editing) : undefined;
  const lineage = data.units.some((u) => u.kind !== "merged");
  const classifying = data.units.some((u) => u.classifying);
  const onSaved = (result: ChangeResult) => {
    setNotice(
      result.skipped
        ? `Saved. New and changed criteria are unclassified until classified: ${result.skipped}`
        : result.classifying
          ? "Saved. Classifying the new and changed criteria…"
          : "Saved.",
    );
  };
  const select = (id: string) => {
    setSelected(id);
    setEnlarged(false);
    scrollInSidebar(document.getElementById(`unit-${id}`), "nearest");
  };
  const t = data.trial;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <p className="text-xs text-gray-500">
          {[t.phase, t.status, data.version ? `${VERSION_LABELS[data.version]} cohorts` : null]
            .filter(Boolean)
            .join(" · ")}
        </p>
        {data.reset.entries || data.reset.pins ? (
          <Button
            size="xs"
            variant="outline"
            className="ml-auto"
            onClick={() => {
              setResetting(true);
            }}
          >
            Reset this trial
          </Button>
        ) : null}
      </div>
      {resetting ? (
        <ResetTrialDialog
          data={data}
          onClose={() => {
            setResetting(false);
          }}
          onReset={(result) => {
            setResetting(false);
            setNotice(
              `Reset: removed ${removedText(result.removed)}. The file as it was is kept on the server as ${result.backup}; import it to undo.`,
            );
          }}
        />
      ) : null}
      {data.originals_reason && data.units.some((u) => u.merged_from > 1) ? (
        <p className="text-xs text-gray-500">Original cohorts: {data.originals_reason}</p>
      ) : null}
      {opened && (opened.tone === "warn" || classifying) ? (
        <p
          role="status"
          className={cn(
            "flex items-center gap-1.5 rounded px-2 py-1.5 text-xs",
            opened.tone === "warn" ? "bg-amber-50 text-amber-900" : "bg-blue-50 text-blue-900",
          )}
        >
          {opened.tone === "info" ? <Spinner className="size-3" /> : null}
          {opened.text}
        </p>
      ) : null}
      <DoubleCounting data={data} />
      {notice ? (
        <p className="rounded bg-blue-50 px-2 py-1.5 text-xs text-blue-900" role="status">
          {notice}
        </p>
      ) : null}
      {lineage ? (
        <div className="flex flex-col gap-1">
          <LineageDiagram units={data.units} selected={selected} onSelect={select} />
          <Button
            size="xs"
            variant="ghost"
            className="self-end text-gray-600"
            onClick={() => {
              setEnlarged(true);
            }}
          >
            <Maximize2Icon />
            Enlarge the diagram
          </Button>
        </div>
      ) : null}
      <ul className="flex flex-col gap-2">
        {data.units.map((unit) => (
          <UnitCard
            key={unit.id}
            unit={unit}
            parent={unit.parent ? units.get(unit.parent) : undefined}
            selected={selected === unit.id}
            onEdit={() => {
              setEditing(unit.id);
            }}
            onSaved={onSaved}
            onCreated={(id) => {
              setEditing(id);
            }}
          />
        ))}
      </ul>
      <Orphans data={data} />
      <ChangeLog data={data} />
      {editUnit ? (
        <CriteriaEditor
          key={`${editUnit.id}-${String(editUnit.revision)}`}
          unit={editUnit}
          onClose={() => {
            setEditing(null);
          }}
          onSaved={onSaved}
        />
      ) : null}
      {enlarged ? (
        <Dialog
          open
          onOpenChange={(o) => {
            if (!o) setEnlarged(false);
          }}
        >
          <DialogContent className="sm:max-w-5xl">
            <DialogHeader>
              <DialogTitle>{t.protocol_no || t.nct_id}: where its cohorts come from</DialogTitle>
            </DialogHeader>
            <LineageDiagram units={data.units} selected={selected} onSelect={select} />
          </DialogContent>
        </Dialog>
      ) : null}
    </div>
  );
}

/** Export the file before a clear or reset (the dialogs offer it). */
function ExportButton() {
  const [error, setError] = useState("");
  return (
    <span className="flex items-center gap-2">
      <Button
        variant="outline"
        onClick={() => {
          setError("");
          void download("/api/management/export", "trial_management.json").catch((e: unknown) => {
            setError(e instanceof Error ? e.message : String(e));
          });
        }}
      >
        <DownloadIcon />
        Export first
      </Button>
      {error ? <span className="text-xs text-red-700">Export failed: {error}</span> : null}
    </span>
  );
}

/** Clear all: what goes, with counts; Export first; the destructive button waits for the word "clear". */
function ClearAllDialog({
  status,
  onClose,
  onCleared,
}: {
  status: {
    counts: {
      subcohorts: number;
      edited: number;
      with_slots: number;
      groups: number;
      pins: number;
    };
  };
  onClose: () => void;
  onCleared: (result: ClearResult) => void;
}) {
  const [word, setWord] = useState("");
  const clear = useManagementChange(() => management.clearAll(word));
  const c = status.counts;
  const items: [number, string, string][] = [
    [c.with_slots, "cohort with slots", "cohorts with slots (available, consented, enrolled)"],
    [c.edited, "cohort with edited criteria", "cohorts with edited criteria"],
    [c.subcohorts, "subcohort", "subcohorts, with their slots and criteria"],
    [c.pins, "placement of a cohort in a group", "placements of cohorts in groups"],
    [c.groups, "group a center made", "groups the centers made"],
  ];
  return (
    <Dialog
      open
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Clear all trial management?</DialogTitle>
          <DialogDescription>
            This removes everything entered on Trial management in this dashboard, in every center
            and for everyone using it:
          </DialogDescription>
        </DialogHeader>
        <ul className="list-disc pl-5 text-sm text-gray-800">
          {items.map(([n, one, many]) => (
            <li key={many} className={n ? "" : "text-gray-400"}>
              {n} {n === 1 ? one : many}
            </li>
          ))}
        </ul>
        <p className="text-sm text-gray-600">
          Edited cohorts go back to their AI-extracted criteria. Labels the classifier gave to
          extracted criteria are kept. The file as it is now is kept on the server beside the new
          one, and you can export it first; importing it undoes the clear.
        </p>
        <ExportButton />
        <label className="flex flex-col gap-1 text-sm">
          <span>
            Type <strong>clear</strong> to confirm
          </span>
          <Input
            aria-label="Type clear to confirm"
            value={word}
            autoFocus
            onChange={(e) => {
              setWord(e.target.value);
            }}
          />
        </label>
        {clear.isError ? <p className="text-sm text-red-700">{clear.error.message}</p> : null}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            disabled={word.trim().toLowerCase() !== "clear" || clear.isPending}
            onClick={() => {
              clear.mutate(undefined, { onSuccess: onCleared });
            }}
          >
            {clear.isPending ? "Clearing…" : "Clear all"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** Reset this trial: what goes, with counts; Export first. Refused if the trial changed since it was opened. */
function ResetTrialDialog({
  data,
  onClose,
  onReset,
}: {
  data: TrialManagementView;
  onClose: () => void;
  onReset: (result: ClearResult) => void;
}) {
  const reset = useManagementChange(() =>
    management.resetTrial(data.trial.id, data.trial_revision),
  );
  const r = data.reset;
  const items: [number, string, string][] = [
    [r.slots, "cohort's slots", "cohorts' slots (available, consented, enrolled)"],
    [r.edited, "cohort's edited criteria", "cohorts' edited criteria"],
    [r.subcohorts, "subcohort", "subcohorts, with their slots and criteria"],
    [r.pins, "placement of a cohort in a group", "placements of its cohorts in groups"],
  ];
  const orphans = data.orphans.length;
  return (
    <Dialog
      open
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Reset {data.trial.protocol_no || data.trial.nct_id}?</DialogTitle>
          <DialogDescription>
            This removes what is kept for this trial, for everyone using the dashboard:
          </DialogDescription>
        </DialogHeader>
        <ul className="list-disc pl-5 text-sm text-gray-800">
          {items.map(([n, one, many]) => (
            <li key={many} className={n ? "" : "text-gray-400"}>
              {n} {n === 1 ? one : many}
            </li>
          ))}
          {orphans ? (
            <li>
              {orphans} {orphans === 1 ? "entry" : "entries"} kept for cohorts the trial no longer
              shows
            </li>
          ) : null}
        </ul>
        <p className="text-sm text-gray-600">
          Its cohorts go back to their AI-extracted criteria and their rule groups. The groups the
          center made stay, even if they are left empty. The file as it is now is kept on the server
          beside the new one, and you can export it first; importing it undoes the reset.
        </p>
        <ExportButton />
        {reset.isError ? <p className="text-sm text-red-700">{reset.error.message}</p> : null}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            disabled={reset.isPending}
            onClick={() => {
              reset.mutate(undefined, { onSuccess: onReset });
            }}
          >
            {reset.isPending ? "Resetting…" : "Reset this trial"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** Slots entered at more than one level of one merged cohort's family: shown, never added up. */
function DoubleCounting({ data }: { data: TrialManagementView }) {
  const byId = new Map(data.units.map((u) => [u.id, u]));
  const families = data.units
    .filter((u) => u.kind === "merged")
    .map((m) => {
      const members = [
        m,
        ...m.originals.map((id) => byId.get(id)),
        ...[m.id, ...m.originals]
          .flatMap((id) => byId.get(id)?.children ?? [])
          .map((id) => byId.get(id)),
      ].filter((u) => u !== undefined);
      return { m, counted: members.filter((u) => u.slots) };
    })
    .filter((f) => f.counted.length > 1);
  if (!families.length) return null;
  return (
    <div className="rounded bg-amber-50 px-2 py-1.5 text-xs text-amber-900 ring-1 ring-amber-200">
      {families.map(({ m, counted }) => (
        <p key={m.id}>
          Slots are entered at more than one level of one cohort's family (
          {counted.map((u) => `${u.name}, ${kindLabel(u).toLowerCase()}`).join("; ")}). Each
          cohort's numbers are its own; they are not added up.
        </p>
      ))}
    </div>
  );
}

function Badge({
  tone,
  children,
}: {
  tone: "blue" | "gray" | "teal" | "amber" | "green";
  children: ReactNode;
}) {
  const tones = {
    blue: "bg-blue-50 text-blue-800 ring-blue-200",
    gray: "bg-gray-100 text-gray-700 ring-gray-200",
    teal: "bg-teal-50 text-teal-800 ring-teal-200",
    amber: "bg-amber-50 text-amber-900 ring-amber-200",
    green: "bg-emerald-50 text-emerald-900 ring-emerald-200",
  };
  return (
    <span className={cn("rounded px-1.5 py-0.5 text-[11px] font-medium ring-1", tones[tone])}>
      {children}
    </span>
  );
}

const KIND_TONE = { merged: "blue", original: "gray", subcohort: "teal" } as const;

/** Two versions of a cohort's criteria side by side (a dialog: the sidebar is too narrow for both). */
function CompareDialog({
  title,
  left,
  right,
  onClose,
}: {
  title: string;
  left: { label: string; lines: ManagedLine[] };
  right: { label: string; lines: ManagedLine[] };
  onClose: () => void;
}) {
  return (
    <Dialog
      open
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <DialogContent className="max-h-[92svh] sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 md:grid-cols-2">
          {[left, right].map((side) => (
            <div key={side.label}>
              <p className="mb-1 text-xs font-semibold text-gray-700">{side.label}</p>
              <CriteriaList items={asItems(side.lines)} />
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function UnitCard({
  unit,
  parent,
  selected,
  onEdit,
  onSaved,
  onCreated,
}: {
  unit: ManagedUnit;
  parent: ManagedUnit | undefined;
  selected: boolean;
  onEdit: () => void;
  onSaved: (result: ChangeResult) => void;
  onCreated: (unitId: string) => void;
}) {
  const [asking, setAsking] = useState<"revert" | null>(null);
  const [naming, setNaming] = useState<"new" | "rename" | null>(null);
  const [name, setName] = useState("");
  const [showLines, setShowLines] = useState(false);
  const [comparing, setComparing] = useState<"stale" | "ai" | null>(null);
  const card = useRef<HTMLLIElement>(null);
  const revert = useManagementChange((propagate: boolean | null) =>
    management.revert(unit.id, unit.revision, propagate),
  );
  const keep = useManagementChange(() => management.keep(unit.id, unit.revision));
  const reset = useManagementChange(() => management.resetToParent(unit.id, unit.revision));
  const dismiss = useManagementChange(() => management.dismissPrevious(unit.id));
  const classify = useManagementChange(() => management.classify(unit.id));
  const remove = useManagementChange(() => management.remove(unit.id));
  const unpin = useManagementChange(() => management.pin(unit.id, null));
  const create = useManagementChange((n: string) => management.createSubcohort(unit.id, n));
  const rename = useManagementChange((n: string) =>
    management.renameSubcohort(unit.id, n, unit.revision),
  );
  const error = [revert, keep, reset, classify, remove, unpin, create, rename].find(
    (m) => m.isError,
  )?.error;
  // Originals and the merged cohort's subcohorts one step in; an original's subcohorts two.
  const depth =
    unit.kind === "merged" ? 0 : unit.kind === "subcohort" && parent?.kind === "original" ? 2 : 1;
  const doRevert = (propagate: boolean | null) => {
    revert.mutate(propagate, {
      onSuccess: (r) => {
        onSaved(r);
        setAsking(null);
      },
    });
  };
  return (
    <li
      ref={card}
      id={`unit-${unit.id}`}
      className={cn(
        "flex flex-col gap-1.5 rounded-md border bg-white p-2 text-sm",
        unit.kind === "merged"
          ? "border-blue-200"
          : unit.kind === "original"
            ? "border-gray-200"
            : "border-teal-200",
        depth === 1 ? "ml-3" : depth === 2 ? "ml-6" : "",
        selected && "ring-2 ring-brand-blue",
      )}
    >
      {/* The card is dragged by its head, so its criteria and inputs stay selectable. */}
      <div
        draggable
        title="Drag onto a group on the left to show it there"
        className="-m-1 flex cursor-grab flex-col gap-1 rounded p-1 hover:bg-gray-50 active:cursor-grabbing"
        onDragStart={(e) => {
          dragUnit(e, unit.id);
          if (card.current) e.dataTransfer.setDragImage(card.current, 12, 12);
        }}
      >
        <div className="flex flex-wrap items-center gap-1">
          <GripVerticalIcon className="size-3.5 text-gray-400" aria-hidden />
          <Badge tone={KIND_TONE[unit.kind]}>{kindLabel(unit)}</Badge>
          {unit.edited ? <Badge tone="blue">Edited</Badge> : null}
          {unit.stale ? <Badge tone="amber">Base changed</Badge> : null}
          {unit.pin_missing ? <Badge tone="amber">Its group is gone</Badge> : null}
        </div>
        <p className="leading-snug font-semibold text-gray-900">{unit.name}</p>
      </div>
      {unit.kind === "merged" && unit.merged_from > 1 ? (
        <p className="text-xs text-gray-500">Combines {unit.merged_from} extracted cohorts</p>
      ) : unit.kind !== "merged" && parent ? (
        <p className="line-clamp-2 text-xs text-gray-500" title={parent.name}>
          {unit.kind === "original" ? "Part of" : "Under"} {parent.name}
        </p>
      ) : null}
      <p className="flex flex-wrap items-center gap-x-2 text-xs text-gray-600">
        {unit.group ? (
          <>
            <span>
              Group: <strong>{unit.group.name}</strong>
              {unit.group.pinned ? " (placed here)" : ""}
            </span>
            {unit.group.pinned ? (
              <button
                type="button"
                className="link"
                disabled={unpin.isPending}
                onClick={() => {
                  unpin.mutate(undefined);
                }}
              >
                {unit.kind === "merged" ? "back to its rule group" : "remove from the group"}
              </button>
            ) : null}
          </>
        ) : (
          "In no group: drag it onto one"
        )}
      </p>
      <SlotEditor unit={unit} />
      {unit.stale ? (
        <div className="flex flex-col gap-1 rounded bg-amber-50 px-2 py-1.5 text-xs text-amber-900 ring-1 ring-amber-200">
          <p>
            {unit.stale.reason === "ai_changed"
              ? "The AI extraction of this cohort changed after its criteria were edited. The edited criteria are still in use."
              : unit.stale.reason === "parent_changed"
                ? "Its parent's criteria changed after this subcohort was defined. Its own criteria are still in use."
                : unit.stale.reason === "parent_missing"
                  ? "Its parent cohort is no longer among the trial's extracted cohorts. Its own criteria are still in use."
                  : "This cohort is no longer among the trial's extracted cohorts."}
          </p>
          <div className="flex flex-wrap gap-1">
            {unit.stale.base_lines && unit.stale.current_lines ? (
              <Button
                size="xs"
                variant="outline"
                onClick={() => {
                  setComparing("stale");
                }}
              >
                Compare
              </Button>
            ) : null}
            {unit.stale.reason === "ai_changed" || unit.stale.reason === "parent_changed" ? (
              <Button
                size="xs"
                variant="outline"
                disabled={keep.isPending}
                onClick={() => {
                  keep.mutate(undefined);
                }}
              >
                Keep {unit.kind === "subcohort" ? "its" : "the edited"} criteria
              </Button>
            ) : null}
            {unit.stale.reason === "parent_changed" ? (
              <Button
                size="xs"
                variant="outline"
                disabled={reset.isPending}
                onClick={() => {
                  reset.mutate(undefined, { onSuccess: onSaved });
                }}
              >
                Reset to the parent's
              </Button>
            ) : null}
          </div>
        </div>
      ) : null}
      {unit.previous_lines ? (
        <div className="flex flex-wrap items-center gap-1 rounded bg-amber-50 px-2 py-1.5 text-xs text-amber-900 ring-1 ring-amber-200">
          Its criteria were reset to its parent's new ones: re-enter its own requirements.
          <Button size="xs" variant="outline" onClick={onEdit}>
            Edit criteria
          </Button>
          <Button
            size="xs"
            variant="ghost"
            onClick={() => {
              dismiss.mutate(undefined);
            }}
          >
            Dismiss
          </Button>
        </div>
      ) : null}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
        <button
          type="button"
          className="link"
          onClick={() => {
            setShowLines((v) => !v);
          }}
        >
          {showLines ? "Hide" : "Show"} its {unit.line_count}{" "}
          {unit.line_count === 1 ? "criterion" : "criteria"} ({unit.key_count} key)
        </button>
        {unit.edited && unit.ai_lines ? (
          <button
            type="button"
            className="link"
            onClick={() => {
              setComparing("ai");
            }}
          >
            compare with the AI-extracted
          </button>
        ) : null}
        {unit.classifying ? (
          <span className="flex items-center gap-1 text-gray-600">
            <Spinner className="size-3" /> classifying criteria…
          </span>
        ) : unit.unclassified ? (
          <span className="flex items-center gap-1 text-amber-800">
            {unit.unclassified} unclassified
            {unit.kind !== "merged" || unit.edited ? (
              <Button
                size="xs"
                variant="outline"
                disabled={classify.isPending}
                onClick={() => {
                  classify.mutate(undefined, { onSuccess: onSaved });
                }}
              >
                Classify
              </Button>
            ) : null}
          </span>
        ) : null}
      </div>
      {unit.classification_error && !unit.classifying && unit.unclassified ? (
        <p className="text-xs text-red-700">{unit.classification_error}</p>
      ) : null}
      {showLines ? <CriteriaList items={asItems(unit.lines)} /> : null}
      <div className="flex flex-wrap gap-1">
        <Button size="xs" variant="outline" onClick={onEdit}>
          <PencilIcon />
          Edit criteria
        </Button>
        {unit.kind !== "subcohort" && unit.edited ? (
          <Button
            size="xs"
            variant="outline"
            disabled={revert.isPending}
            onClick={() => {
              if (unit.children.length) setAsking("revert");
              else doRevert(null);
            }}
          >
            Revert to AI criteria
          </Button>
        ) : null}
        {unit.kind !== "subcohort" ? (
          <Button
            size="xs"
            variant="outline"
            onClick={() => {
              setNaming("new");
              setName("");
            }}
          >
            <PlusIcon />
            Subcohort
          </Button>
        ) : (
          <Button
            size="xs"
            variant="outline"
            onClick={() => {
              setNaming("rename");
              setName(unit.name);
            }}
          >
            Rename
          </Button>
        )}
        <Button
          size="xs"
          variant="outline"
          render={<Link to="/cohorts/$cohortId/patients" params={{ cohortId: unit.id }} />}
        >
          <UsersIcon />
          Find patients
        </Button>
        {unit.kind === "subcohort" ? (
          <Button
            size="xs"
            variant="ghost"
            aria-label={`Delete the subcohort ${unit.name}`}
            disabled={remove.isPending}
            onClick={() => {
              if (window.confirm(`Delete the subcohort ${unit.name}, with its slots and criteria?`))
                remove.mutate(undefined);
            }}
          >
            <Trash2Icon />
          </Button>
        ) : null}
      </div>
      {asking === "revert" ? (
        <div className="flex flex-col gap-1.5 rounded bg-blue-50 p-2 text-xs ring-1 ring-blue-200">
          <p>
            Apply the AI-extracted criteria to its {unit.children.length}{" "}
            {unit.children.length === 1 ? "subcohort" : "subcohorts"} too? Yes: their criteria are
            replaced and you re-enter their own requirements. No: they keep their own.
          </p>
          <div className="flex flex-wrap gap-1">
            <Button
              size="xs"
              onClick={() => {
                doRevert(true);
              }}
            >
              Yes, replace theirs
            </Button>
            <Button
              size="xs"
              variant="outline"
              onClick={() => {
                doRevert(false);
              }}
            >
              No, keep theirs
            </Button>
            <Button
              size="xs"
              variant="ghost"
              onClick={() => {
                setAsking(null);
              }}
            >
              Cancel
            </Button>
          </div>
        </div>
      ) : null}
      {naming ? (
        <form
          className="flex flex-col gap-1 rounded bg-teal-50 p-2 text-xs ring-1 ring-teal-200"
          onSubmit={(e) => {
            e.preventDefault();
            if (naming === "new") {
              create.mutate(name, {
                onSuccess: (r) => {
                  setNaming(null);
                  onCreated(r.id);
                },
              });
            } else {
              rename.mutate(name, {
                onSuccess: () => {
                  setNaming(null);
                },
              });
            }
          }}
        >
          <label className="text-teal-900" htmlFor={`name-${unit.id}`}>
            {naming === "new"
              ? "Subcohort name (it starts with this cohort's criteria, then you add its own):"
              : "Subcohort name:"}
          </label>
          <div className="flex gap-1">
            <Input
              id={`name-${unit.id}`}
              className="h-7 bg-white"
              value={name}
              autoFocus
              onChange={(e) => {
                setName(e.target.value);
              }}
            />
            <Button
              size="xs"
              type="submit"
              disabled={!name.trim() || create.isPending || rename.isPending}
            >
              {naming === "new" ? "Create" : "Rename"}
            </Button>
            <Button
              size="xs"
              variant="ghost"
              type="button"
              onClick={() => {
                setNaming(null);
              }}
            >
              Cancel
            </Button>
          </div>
        </form>
      ) : null}
      {error ? <p className="text-xs text-red-700">{error.message}</p> : null}
      {comparing === "stale" && unit.stale?.base_lines && unit.stale.current_lines ? (
        <CompareDialog
          title={`${unit.name}: what changed under the edit`}
          left={{
            label:
              unit.kind === "subcohort"
                ? "The parent's criteria it was defined from"
                : "The AI version the edits were based on",
            lines: unit.stale.base_lines,
          }}
          right={{
            label: unit.kind === "subcohort" ? "The parent's criteria now" : "The AI version now",
            lines: unit.stale.current_lines,
          }}
          onClose={() => {
            setComparing(null);
          }}
        />
      ) : null}
      {comparing === "ai" && unit.ai_lines ? (
        <CompareDialog
          title={`${unit.name}: edited and AI-extracted criteria`}
          left={{ label: "In use (edited)", lines: unit.lines }}
          right={{ label: "AI-extracted", lines: unit.ai_lines }}
          onClose={() => {
            setComparing(null);
          }}
        />
      ) : null}
    </li>
  );
}

function SlotEditor({ unit }: { unit: ManagedUnit }) {
  const initial = {
    available: unit.slots?.available ?? null,
    consented: unit.slots?.consented ?? null,
    enrolled: unit.slots?.enrolled ?? null,
  };
  const [values, setValues] = useState(initial);
  const [seen, setSeen] = useState(unit.revision);
  if (seen !== unit.revision) {
    setSeen(unit.revision);
    setValues(initial);
  }
  const save = useManagementChange(() => management.slots(unit.id, values, unit.revision));
  const dirty = (Object.keys(values) as (keyof typeof values)[]).some(
    (k) => values[k] !== initial[k],
  );
  const field = (key: keyof typeof values, label: string) => (
    <label className="flex flex-col gap-0.5 text-[11px] text-gray-600">
      {label}
      <Input
        type="number"
        min={0}
        step={1}
        inputMode="numeric"
        className="h-7 w-16 bg-white px-1.5"
        value={values[key] ?? ""}
        onChange={(e) => {
          const raw = e.target.value;
          setValues((v) => ({
            ...v,
            [key]: raw === "" ? null : Math.max(0, Math.floor(Number(raw))),
          }));
        }}
      />
    </label>
  );
  return (
    <form
      className="flex flex-wrap items-end gap-2 rounded bg-gray-50 px-2 py-1.5"
      onSubmit={(e) => {
        e.preventDefault();
        save.mutate(undefined);
      }}
    >
      {field("available", "Available")}
      {field("consented", "Consented")}
      {field("enrolled", "Enrolled")}
      <div className="flex flex-col gap-0.5">
        <span className="text-[11px] text-gray-400">
          {unit.slots?.updated_at ? isoWhen(unit.slots.updated_at) : "no slots yet"}
        </span>
        <Button size="xs" type="submit" disabled={!dirty || save.isPending}>
          {save.isPending ? "Saving…" : "Save slots"}
        </Button>
      </div>
      {save.isError ? <span className="text-xs text-red-700">{save.error.message}</span> : null}
    </form>
  );
}

function Orphans({ data }: { data: TrialManagementView }) {
  const remove = useManagementChange((id: string) => management.remove(id));
  if (!data.orphans.length) return null;
  return (
    <div className="flex flex-col gap-1 rounded bg-amber-50 px-2 py-1.5 text-xs text-amber-900 ring-1 ring-amber-200">
      <p className="font-semibold">Kept for cohorts this trial no longer shows</p>
      <p>
        The trial was extracted again (or its protocol's cohorts replaced its ClinicalTrials.gov
        ones). Their entries are kept until you delete them.
      </p>
      <ul className="flex flex-col gap-1">
        {data.orphans.map((o) => (
          <li key={o.id} className="flex items-center gap-1">
            <span className="flex-1">
              {o.name ?? o.id} ({kindLabel(o).toLowerCase()}
              {o.version !== data.version ? `, ${VERSION_LABELS[o.version]} cohorts` : ""})
              {o.slots ? ` · slots: ${slotText(o.slots)}` : ""}
              {o.edited ? " · criteria edited" : ""}
            </span>
            <Button
              size="icon-xs"
              variant="ghost"
              aria-label="Delete what is kept for this cohort"
              onClick={() => {
                if (window.confirm("Delete what is kept for this cohort?")) remove.mutate(o.id);
              }}
            >
              <Trash2Icon />
            </Button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ChangeLog({ data }: { data: TrialManagementView }) {
  if (!data.log.length) return null;
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-gray-600">
        Recent changes to this trial ({data.log.length})
      </summary>
      <ul className="mt-1 flex flex-col gap-0.5 text-gray-600">
        {data.log.map((e, i) => (
          <li key={i}>
            {isoWhen(e.at)} · {e.action}
            {e.unit ? ` · ${e.unit}` : ""}
            {e.detail ? ` · ${e.detail}` : ""}
          </li>
        ))}
      </ul>
    </details>
  );
}

/**
 * Scroll the sidebar (only) so ``el`` is in view below its sticky header: unlike scrollIntoView, the page itself
 * stays put.
 */
function scrollInSidebar(el: HTMLElement | null | undefined, block: "center" | "nearest") {
  const box = el?.closest<HTMLElement>("[data-sidebar-scroll]");
  if (!el || !box) return;
  const header = box.querySelector<HTMLElement>("[data-sidebar-header]")?.offsetHeight ?? 0;
  const outer = box.getBoundingClientRect();
  const inner = el.getBoundingClientRect();
  const offset = inner.top - outer.top + box.scrollTop;
  let top: number;
  if (block === "center") top = offset - header - (box.clientHeight - header - inner.height) / 2;
  else if (inner.top < outer.top + header) top = offset - header - 8;
  else if (inner.bottom > outer.bottom) top = offset - box.clientHeight + inner.height + 8;
  else return;
  box.scrollTo({ top: Math.max(0, top) });
}
