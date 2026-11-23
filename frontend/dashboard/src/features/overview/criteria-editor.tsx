import { cn } from "cn";
import { PlusIcon, Trash2Icon } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import {
  type ChangeResult,
  HEADINGS,
  kindLabel,
  management,
  type ManagedLine,
  type ManagedUnit,
  useManagementChange,
} from "@/lib/management";

interface Draft extends ManagedLine {
  key: number;
}

let nextKey = 1;

function drafts(lines: ManagedLine[]): Draft[] {
  return lines.map((line) => ({ ...line, key: nextKey++ }));
}

function same(a: ManagedLine[], b: ManagedLine[]) {
  const view = (lines: ManagedLine[]) =>
    lines
      .filter((l) => l.text.trim())
      .map((l) => [l.heading, l.text.trim(), l.criterion_type, l.boilerplate].join("\u0000"));
  return JSON.stringify(view(a)) === JSON.stringify(view(b));
}

const LABELED_BY: Record<string, string> = {
  ai: "AI",
  classifier: "classified",
  manual: "set by hand",
};

/**
 * Edit a unit's criteria: change, add or remove lines and set a line's labels by hand. Saving keeps unchanged
 * lines' labels; changed and new lines are classified (inclusion or exclusion, key or common) by the session's
 * Criteria classification connection, and stay unclassified without one. A cohort with subcohorts asks whether its
 * new criteria replace theirs.
 */
export function CriteriaEditor({
  unit,
  onClose,
  onSaved,
}: {
  unit: ManagedUnit;
  onClose: () => void;
  onSaved: (result: ChangeResult) => void;
}) {
  const [lines, setLines] = useState<Draft[]>(() => drafts(unit.lines));
  const [asking, setAsking] = useState(false);
  const headed = lines.some((l) => l.heading) || unit.lines.length === 0;
  const dirty = !same(lines, unit.lines);
  const save = useManagementChange((propagate: boolean | null) =>
    management.criteria(
      unit.id,
      lines.filter((l) => l.text.trim()).map(({ key: _key, ...line }) => line),
      unit.revision,
      propagate,
    ),
  );
  const submit = (propagate: boolean | null) => {
    save.mutate(propagate, {
      onSuccess: (result) => {
        onSaved(result);
        onClose();
      },
    });
  };
  const update = (key: number, patch: Partial<ManagedLine>) => {
    setLines((all) => all.map((l) => (l.key === key ? { ...l, ...patch } : l)));
  };
  const add = (heading: string) => {
    setLines((all) => {
      const line: Draft = {
        key: nextKey++,
        heading,
        text: "",
        criterion_type: null,
        boilerplate: false,
        labeled_by: null,
      };
      const last = all.map((l) => l.heading).lastIndexOf(heading);
      return last < 0 ? [...all, line] : [...all.slice(0, last + 1), line, ...all.slice(last + 1)];
    });
  };
  const insertAfter = (key: number) => {
    setLines((all) => {
      const i = all.findIndex((l) => l.key === key);
      const line: Draft = {
        key: nextKey++,
        heading: all[i]?.heading ?? "",
        text: "",
        criterion_type: null,
        boilerplate: false,
        labeled_by: null,
      };
      return [...all.slice(0, i + 1), line, ...all.slice(i + 1)];
    });
  };
  const pending = lines.filter((l) => l.text.trim() && !l.criterion_type).length;

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[92svh] sm:max-w-5xl" bodyClassName="gap-4">
        <DialogHeader>
          <DialogTitle>
            {kindLabel(unit)} criteria: {unit.name}
          </DialogTitle>
          <DialogDescription className="text-gray-600">
            Edit, add or remove lines. Unchanged lines keep their labels; when you save, changed and
            new lines are labeled inclusion or exclusion and key or common by the Criteria
            classification model (Data and models), the cohort pipeline's own step. You can also set
            a line's labels yourself. Find patients, scoring, criteria checks and structured-data
            screening use these criteria once saved.
          </DialogDescription>
        </DialogHeader>
        <div className={cn("grid gap-4", unit.previous_lines && "lg:grid-cols-[2fr_1fr]")}>
          <ol className="flex flex-col gap-1">
            {lines.map((line, i) => (
              <li key={line.key} className="flex flex-col gap-1">
                {line.heading && line.heading !== lines[i - 1]?.heading ? (
                  <h4 className="mt-2 text-sm font-bold">{line.heading.replace(":", "")}</h4>
                ) : null}
                <LineEditor
                  line={line}
                  onChange={(patch) => {
                    update(line.key, patch);
                  }}
                  onRemove={() => {
                    setLines((all) => all.filter((l) => l.key !== line.key));
                  }}
                  onInsert={() => {
                    insertAfter(line.key);
                  }}
                />
              </li>
            ))}
            <li className="flex flex-wrap gap-2 pt-2">
              {headed ? (
                HEADINGS.map((heading) => (
                  <Button
                    key={heading}
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      add(heading);
                    }}
                  >
                    <PlusIcon />
                    Add {heading.startsWith("Incl") ? "inclusion" : "exclusion"} line
                  </Button>
                ))
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    add("");
                  }}
                >
                  <PlusIcon />
                  Add line
                </Button>
              )}
            </li>
          </ol>
          {unit.previous_lines ? (
            <aside className="flex flex-col gap-2 rounded-md bg-amber-50 p-3 text-sm ring-1 ring-amber-200">
              <p className="font-semibold text-amber-900">
                This subcohort's criteria before its parent's changed
              </p>
              <p className="text-xs text-amber-900">
                Its criteria were reset to the parent's new ones. Re-enter its own requirements;
                these are for reference.
              </p>
              <ul className="flex flex-col gap-1">
                {unit.previous_lines.map((l, i) => (
                  <li key={i} className="whitespace-pre-wrap text-gray-800">
                    {l.text}
                  </li>
                ))}
              </ul>
            </aside>
          ) : null}
        </div>
        {asking ? (
          <div className="flex flex-col gap-2 rounded-md bg-blue-50 p-3 text-sm ring-1 ring-blue-200">
            <p className="font-semibold">
              This cohort has {unit.children.length}{" "}
              {unit.children.length === 1 ? "subcohort" : "subcohorts"}. Apply the new criteria to{" "}
              {unit.children.length === 1 ? "it" : "them"}?
            </p>
            <p className="text-gray-700">
              Yes: each subcohort's criteria are replaced by these, and you re-enter its own
              requirements (its previous criteria are shown while you do). No: each keeps its own
              criteria.
            </p>
            <div className="flex gap-2">
              <Button
                size="sm"
                disabled={save.isPending}
                onClick={() => {
                  submit(true);
                }}
              >
                Yes, replace theirs
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={save.isPending}
                onClick={() => {
                  submit(false);
                }}
              >
                No, keep theirs
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setAsking(false);
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        ) : null}
        {save.isError ? (
          <p className="text-sm text-red-700">Not saved: {save.error.message}</p>
        ) : null}
        <DialogFooter className="items-center">
          <span className="mr-auto text-xs text-gray-500">
            {pending
              ? `${String(pending)} ${pending === 1 ? "line" : "lines"} to classify when saved`
              : ""}
          </span>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            disabled={!dirty || save.isPending || !lines.some((l) => l.text.trim())}
            onClick={() => {
              if (unit.children.length && unit.kind !== "subcohort") setAsking(true);
              else submit(null);
            }}
          >
            {save.isPending ? "Saving…" : "Save criteria"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function LineEditor({
  line,
  onChange,
  onRemove,
  onInsert,
}: {
  line: Draft;
  onChange: (patch: Partial<ManagedLine>) => void;
  onRemove: () => void;
  onInsert: () => void;
}) {
  const label = (patch: Partial<ManagedLine>) => {
    onChange({
      criterion_type:
        line.criterion_type ?? (line.heading.startsWith("Excl") ? "exclusion" : "inclusion"),
      ...patch,
      labeled_by: "manual",
    });
  };
  return (
    <div className="flex items-start gap-2">
      <Textarea
        aria-label="Criteria line"
        value={line.text}
        rows={1}
        className={cn("min-h-0 flex-1 py-1", line.boilerplate && "text-gray-500")}
        onChange={(e) => {
          // A changed line is classified again when saved (unless its labels are then set by hand).
          onChange({
            text: e.target.value,
            criterion_type: null,
            boilerplate: false,
            labeled_by: null,
          });
        }}
      />
      <div className="flex shrink-0 flex-col items-end gap-0.5">
        <div className="flex gap-1">
          <Toggle
            options={[
              ["inclusion", "Incl."],
              ["exclusion", "Excl."],
            ]}
            value={line.criterion_type}
            onChange={(value) => {
              label({ criterion_type: value as ManagedLine["criterion_type"] });
            }}
          />
          <Toggle
            options={[
              ["key", "Key"],
              ["common", "Common"],
            ]}
            value={line.criterion_type ? (line.boilerplate ? "common" : "key") : null}
            onChange={(value) => {
              label({ boilerplate: value === "common" });
            }}
          />
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label="Insert a line below"
            onClick={onInsert}
          >
            <PlusIcon />
          </Button>
          <Button size="icon-xs" variant="ghost" aria-label="Remove the line" onClick={onRemove}>
            <Trash2Icon />
          </Button>
        </div>
        <span className="text-[11px] text-gray-400">
          {line.criterion_type
            ? (LABELED_BY[line.labeled_by ?? "ai"] ?? "")
            : line.text.trim()
              ? "unclassified: labeled when saved"
              : ""}
        </span>
      </div>
    </div>
  );
}

function Toggle({
  options,
  value,
  onChange,
}: {
  options: [string, string][];
  value: string | null;
  onChange: (value: string) => void;
}) {
  return (
    <div className="inline-flex overflow-hidden rounded border border-gray-300 text-[11px]">
      {options.map(([key, text]) => (
        <button
          key={key}
          type="button"
          className={cn(
            "px-1.5 py-0.5",
            value === key ? "bg-brand-blue text-white" : "bg-white text-gray-600 hover:bg-gray-100",
          )}
          aria-pressed={value === key}
          onClick={() => {
            onChange(key);
          }}
        >
          {text}
        </button>
      ))}
    </div>
  );
}
