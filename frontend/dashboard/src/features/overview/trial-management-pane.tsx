import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { cn } from "cn";
import {
  DownloadIcon,
  GripVerticalIcon,
  PencilIcon,
  PlusIcon,
  Trash2Icon,
  UploadIcon,
  UsersIcon,
  XIcon,
} from "lucide-react";
import { type ReactNode, useMemo, useRef, useState } from "react";

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
import { type CenterView, download, type TrialSummary, VERSION_LABELS } from "@/lib/api";
import { isoWhen } from "@/lib/format";
import {
  type ChangeResult,
  kindLabel,
  management,
  type ManagedLine,
  type ManagedUnit,
  managementStatusOptions,
  slotText,
  trialManagementOptions,
  type TrialManagementView,
  useManagementChange,
} from "@/lib/management";

import { CriteriaEditor } from "./criteria-editor";
import { dragUnit, useDropTarget } from "./drag";
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
 * Manage a trial: choose one of the center's trials, then its cohorts (merged cohorts, the original cohorts they
 * stand for, and subcohorts) with their slots, criteria and groups. Every change is saved at once to the
 * dashboard's trial management file, shared by everyone using this dashboard; Export and Import save and load it.
 */
export function TrialManagementPane({
  view,
  trialId,
  onTrial,
  onClose,
}: {
  view: CenterView;
  trialId: string | undefined;
  onTrial: (trialId: string | undefined) => void;
  onClose: () => void;
}) {
  const trials = useMemo(() => {
    const seen = new Map<string, TrialSummary>();
    for (const g of view.groups) for (const r of g.rows) seen.set(r.trial.id, r.trial);
    return [...seen.values()].sort((a, b) =>
      (a.protocol_no || a.nct_id).localeCompare(b.protocol_no || b.nct_id),
    );
  }, [view]);
  return (
    <section
      aria-label="Manage a trial"
      className="flex flex-col gap-3 rounded-md border border-brand-blue/40 bg-white p-4 print:hidden"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-gray-900">Manage a trial</h2>
          <p className="text-sm text-gray-600">
            Enrollment slots, criteria and subcohorts of one trial's cohorts, and the groups they
            are shown in. Changes are saved as you make them and shared with everyone using this
            dashboard.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <FileActions />
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Close Manage a trial"
            onClick={onClose}
          >
            <XIcon />
          </Button>
        </div>
      </div>
      <TrialPicker trials={trials} trialId={trialId} onTrial={onTrial} />
      {trialId ? (
        <TrialManager trialId={trialId} />
      ) : (
        <p className="text-sm text-gray-500">
          Choose a trial by its protocol number, NCT number or title.
        </p>
      )}
    </section>
  );
}

function TrialPicker({
  trials,
  trialId,
  onTrial,
}: {
  trials: TrialSummary[];
  trialId: string | undefined;
  onTrial: (trialId: string | undefined) => void;
}) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const matches = trials
    .filter((t) => {
      const hay = [t.protocol_no, t.nct_id, t.short_title, t.site_pi].join(" ").toLowerCase();
      return words.every((w) => hay.includes(w));
    })
    .slice(0, 12);
  const chosen = trials.find((t) => t.id === trialId);
  return (
    <div className="relative flex flex-col gap-1">
      <Input
        aria-label="Find a trial to manage"
        placeholder={
          chosen
            ? `${chosen.protocol_no || chosen.nct_id}: ${chosen.short_title}`
            : "Protocol number, NCT number or title…"
        }
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setOpen(true);
        }}
        onFocus={() => {
          setOpen(true);
        }}
        onBlur={() => {
          setTimeout(() => {
            setOpen(false);
          }, 150);
        }}
      />
      {open && matches.length ? (
        <ul className="absolute top-10 z-20 max-h-80 w-full overflow-y-auto rounded-md border border-gray-200 bg-white shadow-lg">
          {matches.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                className="flex w-full flex-col items-start px-3 py-1.5 text-left text-sm hover:bg-brand-blue/5"
                onMouseDown={(e) => {
                  e.preventDefault();
                  onTrial(t.id);
                  setText("");
                  setOpen(false);
                }}
              >
                <span className="font-semibold">
                  {t.protocol_no || t.nct_id}
                  {t.protocol_no ? (
                    <span className="font-normal text-gray-500"> · {t.nct_id}</span>
                  ) : null}
                </span>
                <span className="text-xs text-gray-600">{t.short_title}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/** Export (download) and Import (upload, after confirming) of the trial management file. */
function FileActions() {
  const status = useQuery(managementStatusOptions()).data;
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const replace = useManagementChange((chosen: File) => management.importFile(chosen));
  return (
    <div className="flex flex-col items-end gap-1">
      <div className="flex items-center gap-2">
        <Button
          size="sm"
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
          size="sm"
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
      </div>
      {status ? (
        <span className="text-xs text-gray-500">
          {status.error
            ? status.error
            : status.updated_at
              ? `Last change ${isoWhen(status.updated_at)}`
              : "No changes yet"}
        </span>
      ) : null}
      {message ? <span className="text-xs text-gray-700">{message}</span> : null}
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

function TrialManager({ trialId }: { trialId: string }) {
  const query = useQuery(trialManagementOptions(trialId));
  const [editing, setEditing] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const data = query.data;
  if (!data) {
    return query.isError ? (
      <p className="text-sm text-red-700">Could not load the trial: {query.error.message}</p>
    ) : (
      <div className="flex justify-center py-6">
        <Spinner className="size-5" />
      </div>
    );
  }
  const units = new Map(data.units.map((u) => [u.id, u]));
  const editUnit = editing ? units.get(editing) : undefined;
  const lineage = data.units.some((u) => u.kind !== "merged");
  const onSaved = (result: ChangeResult) => {
    setNotice(
      result.skipped
        ? `Saved. New and changed lines are unclassified until classified: ${result.skipped}`
        : result.classifying
          ? "Saved. Classifying the new and changed lines…"
          : "Saved.",
    );
  };
  const t = data.trial;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
        <span className="font-bold text-gray-900">{t.protocol_no || t.nct_id}</span>
        <span className="text-gray-700">{t.short_title}</span>
        <span className="text-xs text-gray-500">
          {[
            t.phase,
            t.nct_id,
            t.status,
            data.version ? `${VERSION_LABELS[data.version]} cohorts` : null,
          ]
            .filter(Boolean)
            .join(" · ")}
        </span>
      </div>
      {data.originals_reason && data.units.some((u) => u.merged_from > 1) ? (
        <p className="text-xs text-gray-500">Original cohorts: {data.originals_reason}</p>
      ) : null}
      <DoubleCounting data={data} />
      {notice ? (
        <p className="rounded bg-blue-50 px-3 py-1.5 text-sm text-blue-900" role="status">
          {notice}
        </p>
      ) : null}
      <div className="grid gap-4 lg:grid-cols-[1fr_17rem]">
        <div className="flex min-w-0 flex-col gap-3">
          {lineage ? (
            <LineageDiagram
              units={data.units}
              selected={selected}
              onSelect={(id) => {
                setSelected(id);
                document.getElementById(`unit-${id}`)?.scrollIntoView({ block: "nearest" });
              }}
            />
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
        </div>
        <GroupsPanel data={data} />
      </div>
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
    </div>
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
    <div className="rounded bg-amber-50 px-3 py-2 text-sm text-amber-900 ring-1 ring-amber-200">
      {families.map(({ m, counted }) => (
        <p key={m.id}>
          Slots are entered at more than one level of {m.name}:{" "}
          {counted.map((u) => `${u.name} (${kindLabel(u).toLowerCase()})`).join(", ")}. Each
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
  tone: "blue" | "gray" | "teal" | "amber" | "red";
  children: ReactNode;
}) {
  const tones = {
    blue: "bg-blue-50 text-blue-800 ring-blue-200",
    gray: "bg-gray-100 text-gray-700 ring-gray-200",
    teal: "bg-teal-50 text-teal-800 ring-teal-200",
    amber: "bg-amber-50 text-amber-900 ring-amber-200",
    red: "bg-red-50 text-red-800 ring-red-200",
  };
  return (
    <span className={cn("rounded px-1.5 py-0.5 text-[11px] font-medium ring-1", tones[tone])}>
      {children}
    </span>
  );
}

const KIND_TONE = { merged: "blue", original: "gray", subcohort: "teal" } as const;

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
  const [comparing, setComparing] = useState(false);
  const revert = useManagementChange((propagate: boolean | null) =>
    management.revert(unit.id, unit.revision, propagate),
  );
  const keep = useManagementChange(() => management.keep(unit.id, unit.revision));
  const reset = useManagementChange(() => management.resetToParent(unit.id, unit.revision));
  const dismiss = useManagementChange(() => management.dismissPrevious(unit.id));
  const classify = useManagementChange(() => management.classify(unit.id));
  const remove = useManagementChange(() => management.remove(unit.id));
  const create = useManagementChange((n: string) => management.createSubcohort(unit.id, n));
  const rename = useManagementChange((n: string) =>
    management.renameSubcohort(unit.id, n, unit.revision),
  );
  const error = [revert, keep, reset, classify, remove, create, rename].find(
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
      id={`unit-${unit.id}`}
      draggable
      onDragStart={(e) => {
        dragUnit(e, unit.id);
      }}
      className={cn(
        "flex flex-col gap-2 rounded-md border bg-white p-3",
        unit.kind === "merged"
          ? "border-blue-200"
          : unit.kind === "original"
            ? "border-gray-200"
            : "border-teal-200",
        depth === 1 ? "ml-6" : depth === 2 ? "ml-12" : "",
        selected && "ring-2 ring-brand-blue",
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <GripVerticalIcon className="size-4 cursor-grab text-gray-300" aria-hidden />
        <Badge tone={KIND_TONE[unit.kind]}>{kindLabel(unit)}</Badge>
        <span className="font-semibold text-gray-900">{unit.name}</span>
        {unit.kind === "merged" && unit.merged_from > 1 ? (
          <span className="text-xs text-gray-500">
            combines {unit.merged_from} extracted cohorts
          </span>
        ) : null}
        {unit.kind !== "merged" && parent ? (
          <span className="text-xs text-gray-500">
            {unit.kind === "original" ? "part of" : "under"} {parent.name}
          </span>
        ) : null}
        {unit.edited ? <Badge tone="blue">Criteria edited</Badge> : null}
        {unit.stale ? <Badge tone="amber">Base changed</Badge> : null}
        {unit.pin_missing ? <Badge tone="amber">Its group no longer exists</Badge> : null}
        <span className="ml-auto text-xs text-gray-500">
          {unit.group ? (
            <>
              Group: <strong>{unit.group.name}</strong>
              {unit.group.pinned ? " (placed by hand)" : ""}
            </>
          ) : (
            "In no group: drag it onto one"
          )}
        </span>
      </div>
      <SlotEditor unit={unit} />
      {unit.stale ? (
        <div className="flex flex-col gap-1.5 rounded bg-amber-50 px-3 py-2 text-sm text-amber-900 ring-1 ring-amber-200">
          <p>
            {unit.stale.reason === "ai_changed"
              ? "The AI extraction of this cohort changed after its criteria were edited. The edited criteria are still in use."
              : unit.stale.reason === "parent_changed"
                ? "Its parent's criteria changed after this subcohort was defined. Its own criteria are still in use."
                : unit.stale.reason === "parent_missing"
                  ? "Its parent cohort is no longer among the trial's extracted cohorts. Its own criteria are still in use."
                  : "This cohort is no longer among the trial's extracted cohorts."}
          </p>
          <div className="flex flex-wrap gap-2">
            {unit.stale.base_lines ? (
              <Button
                size="xs"
                variant="outline"
                onClick={() => {
                  setComparing((v) => !v);
                }}
              >
                {comparing ? "Hide the comparison" : "Compare"}
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
                Reset to the parent's criteria
              </Button>
            ) : null}
          </div>
          {comparing && unit.stale.base_lines && unit.stale.current_lines ? (
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <p className="mb-1 text-xs font-semibold">
                  {unit.kind === "subcohort"
                    ? "The parent's criteria it was defined from"
                    : "The AI version the edits were based on"}
                </p>
                <CriteriaList items={asItems(unit.stale.base_lines)} />
              </div>
              <div>
                <p className="mb-1 text-xs font-semibold">
                  {unit.kind === "subcohort" ? "The parent's criteria now" : "The AI version now"}
                </p>
                <CriteriaList items={asItems(unit.stale.current_lines)} />
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
      {unit.previous_lines ? (
        <div className="flex flex-wrap items-center gap-2 rounded bg-amber-50 px-3 py-2 text-sm text-amber-900 ring-1 ring-amber-200">
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
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <button
          type="button"
          className="link text-xs"
          onClick={() => {
            setShowLines((v) => !v);
          }}
        >
          {showLines ? "Hide" : "Show"} its {unit.line_count} criteria lines ({unit.key_count} key)
        </button>
        {unit.classifying ? (
          <span className="flex items-center gap-1 text-xs text-gray-600">
            <Spinner className="size-3" /> classifying lines…
          </span>
        ) : unit.unclassified ? (
          <span className="flex items-center gap-1 text-xs text-amber-800">
            {unit.unclassified} unclassified {unit.unclassified === 1 ? "line" : "lines"}
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
        {unit.classification_error && !unit.classifying ? (
          <span className="text-xs text-red-700">{unit.classification_error}</span>
        ) : null}
      </div>
      {showLines ? (
        unit.edited && unit.ai_lines ? (
          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <p className="mb-1 text-xs font-semibold text-gray-700">In use (edited)</p>
              <CriteriaList items={asItems(unit.lines)} />
            </div>
            <div>
              <p className="mb-1 text-xs font-semibold text-gray-700">AI-extracted</p>
              <CriteriaList items={asItems(unit.ai_lines)} />
            </div>
          </div>
        ) : (
          <CriteriaList items={asItems(unit.lines)} />
        )
      ) : null}
      <div className="flex flex-wrap gap-2">
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
            Revert to the AI-extracted criteria
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
            New subcohort
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
            disabled={remove.isPending}
            onClick={() => {
              if (window.confirm(`Delete the subcohort ${unit.name}, with its slots and criteria?`))
                remove.mutate(undefined);
            }}
          >
            <Trash2Icon />
            Delete
          </Button>
        ) : null}
      </div>
      {asking === "revert" ? (
        <div className="flex flex-col gap-2 rounded bg-blue-50 p-2 text-sm ring-1 ring-blue-200">
          <p>
            Apply the AI-extracted criteria to its {unit.children.length}{" "}
            {unit.children.length === 1 ? "subcohort" : "subcohorts"} too? Yes: their criteria are
            replaced and you re-enter their own requirements. No: they keep their own.
          </p>
          <div className="flex gap-2">
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
          className="flex flex-wrap items-center gap-2 rounded bg-teal-50 p-2 text-sm ring-1 ring-teal-200"
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
          <label className="text-xs text-teal-900" htmlFor={`name-${unit.id}`}>
            {naming === "new"
              ? "Subcohort name (it starts with this cohort's criteria, then you add its own):"
              : "Subcohort name:"}
          </label>
          <Input
            id={`name-${unit.id}`}
            className="h-7 w-72"
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
        </form>
      ) : null}
      {error ? <p className="text-xs text-red-700">{error.message}</p> : null}
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
    <label className="flex items-center gap-1 text-xs text-gray-600">
      {label}
      <Input
        type="number"
        min={0}
        step={1}
        inputMode="numeric"
        className="h-7 w-20"
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
      className="flex flex-wrap items-center gap-3"
      onSubmit={(e) => {
        e.preventDefault();
        save.mutate(undefined);
      }}
    >
      <span className="text-xs font-semibold text-gray-700">Slots</span>
      {field("available", "Available")}
      {field("consented", "Consented")}
      {field("enrolled", "Enrolled")}
      <Button size="xs" type="submit" disabled={!dirty || save.isPending}>
        {save.isPending ? "Saving…" : "Save slots"}
      </Button>
      <span className="text-xs text-gray-400">
        {unit.slots?.updated_at ? `updated ${isoWhen(unit.slots.updated_at)}` : "none entered"}
      </span>
      {save.isError ? <span className="text-xs text-red-700">{save.error.message}</span> : null}
    </form>
  );
}

/** The center's groups as drop targets: a cohort dropped on one is shown in it. */
function GroupsPanel({ data }: { data: TrialManagementView }) {
  const [name, setName] = useState("");
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);
  const pin = useManagementChange(
    ({ unitId, groupId }: { unitId: string; groupId: string | null }) =>
      management.pin(unitId, groupId),
  );
  const create = useManagementChange((n: string) => management.createGroup(data.center_id, n));
  const rename = useManagementChange(({ id, n }: { id: string; n: string }) =>
    management.renameGroup(data.center_id, id, n),
  );
  const remove = useManagementChange((id: string) => management.deleteGroup(data.center_id, id));
  const error = [pin, create, rename, remove].find((m) => m.isError)?.error;
  return (
    <aside className="flex flex-col gap-2 self-start rounded-md bg-gray-50 p-3 lg:sticky lg:top-[calc(var(--app-tabs-h,48px)+0.5rem)]">
      <h3 className="text-sm font-bold text-gray-900">Groups</h3>
      <p className="text-xs text-gray-600">
        Drag a cohort onto a group to show it there; it stays there instead of where the group rules
        put it. Original cohorts and subcohorts are in a group only when placed in one.
      </p>
      <ul className="flex flex-col gap-1">
        {data.groups.map((g) => (
          <GroupTarget
            key={g.id}
            name={g.name}
            custom={g.custom}
            units={data.units.filter((u) => u.group?.id === g.id)}
            onDrop={(unitId) => {
              pin.mutate({ unitId, groupId: g.id });
            }}
            onRename={() => {
              setRenaming({ id: g.id, name: g.name });
            }}
            onDelete={() => {
              if (
                window.confirm(`Delete the group ${g.name}? Its cohorts go back to their groups.`)
              )
                remove.mutate(g.id);
            }}
          />
        ))}
        <GroupTarget
          name="Back to the rules' group"
          custom={false}
          units={[]}
          muted
          onDrop={(unitId) => {
            pin.mutate({ unitId, groupId: null });
          }}
        />
      </ul>
      {renaming ? (
        <form
          className="flex gap-1"
          onSubmit={(e) => {
            e.preventDefault();
            rename.mutate(
              { id: renaming.id, n: renaming.name },
              {
                onSuccess: () => {
                  setRenaming(null);
                },
              },
            );
          }}
        >
          <Input
            className="h-7"
            aria-label="Group name"
            value={renaming.name}
            autoFocus
            onChange={(e) => {
              setRenaming({ ...renaming, name: e.target.value });
            }}
          />
          <Button size="xs" type="submit" disabled={!renaming.name.trim()}>
            Rename
          </Button>
        </form>
      ) : null}
      <form
        className="flex gap-1"
        onSubmit={(e) => {
          e.preventDefault();
          create.mutate(name, {
            onSuccess: () => {
              setName("");
            },
          });
        }}
      >
        <Input
          className="h-7"
          aria-label="New group name"
          placeholder="New group"
          value={name}
          onChange={(e) => {
            setName(e.target.value);
          }}
        />
        <Button size="xs" type="submit" disabled={!name.trim() || create.isPending}>
          <PlusIcon />
          Add
        </Button>
      </form>
      {error ? <p className="text-xs text-red-700">{error.message}</p> : null}
    </aside>
  );
}

function GroupTarget({
  name,
  custom,
  units,
  muted,
  onDrop,
  onRename,
  onDelete,
}: {
  name: string;
  custom: boolean;
  units: ManagedUnit[];
  muted?: boolean;
  onDrop: (unitId: string) => void;
  onRename?: () => void;
  onDelete?: () => void;
}) {
  const target = useDropTarget(onDrop);
  return (
    <li
      {...target.props}
      className={cn(
        "flex flex-col gap-0.5 rounded border border-dashed px-2 py-1.5 text-sm",
        target.over ? "border-brand-blue bg-brand-blue/10" : "border-gray-300 bg-white",
        muted && "text-gray-500",
      )}
    >
      <div className="flex items-center gap-1">
        <span className={cn("flex-1", custom && "font-medium")}>{name}</span>
        {custom ? (
          <>
            <Button size="icon-xs" variant="ghost" aria-label={`Rename ${name}`} onClick={onRename}>
              <PencilIcon />
            </Button>
            <Button size="icon-xs" variant="ghost" aria-label={`Delete ${name}`} onClick={onDelete}>
              <Trash2Icon />
            </Button>
          </>
        ) : null}
      </div>
      {units.length ? (
        <ul className="flex flex-col gap-0.5">
          {units.map((u) => (
            <li key={u.id} className="truncate text-xs text-gray-600" title={u.name}>
              {u.name}
              {u.slots ? <span className="text-emerald-700"> · {slotText(u.slots)}</span> : null}
            </li>
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function Orphans({ data }: { data: TrialManagementView }) {
  const remove = useManagementChange((id: string) => management.remove(id));
  if (!data.orphans.length) return null;
  return (
    <div className="flex flex-col gap-1 rounded bg-amber-50 px-3 py-2 text-sm text-amber-900 ring-1 ring-amber-200">
      <p className="font-semibold">Kept for cohorts this trial no longer has</p>
      <p className="text-xs">
        The trial was extracted again (or its protocol's cohorts replaced its ClinicalTrials.gov
        ones), and these cohorts are not among the cohorts shown now. Their entries are kept until
        you delete them.
      </p>
      <ul className="flex flex-col gap-1">
        {data.orphans.map((o) => (
          <li key={o.id} className="flex items-center gap-2">
            <span>
              {o.name ?? o.id} ({kindLabel(o).toLowerCase()}
              {o.version !== data.version ? `, ${VERSION_LABELS[o.version]} cohorts` : ""})
              {o.slots ? ` · slots: ${slotText(o.slots)}` : ""}
              {o.edited ? " · criteria edited" : ""}
            </span>
            <Button
              size="xs"
              variant="ghost"
              onClick={() => {
                if (window.confirm("Delete what is kept for this cohort?")) remove.mutate(o.id);
              }}
            >
              <Trash2Icon />
              Delete
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
    <details className="text-sm">
      <summary className="cursor-pointer text-gray-600">
        Recent changes to this trial ({data.log.length})
      </summary>
      <ul className="mt-1 flex flex-col gap-0.5 text-xs text-gray-600">
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
