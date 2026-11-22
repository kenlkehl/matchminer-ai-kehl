import { cn } from "cn";
import { useState } from "react";

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

import { noteDate, noteRows } from "./notes";
import { usePatient } from "./store";

/** The patient's notes as a table: each note's date and type, and its text. */
export function NotesDialog({ onClose }: { onClose: () => void }) {
  const s = usePatient();
  const rows = noteRows(s.notesText, s.notes);
  const [expanded, setExpanded] = useState<Set<number>>(() => new Set());
  const all = rows.length > 0 && expanded.size === rows.length;
  const toggle = (i: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90svh] sm:max-w-6xl" bodyClassName="gap-3 auto-rows-min">
        <DialogHeader>
          <DialogTitle>Clinical notes</DialogTitle>
          <DialogDescription className="text-gray-600">
            {rows.length} {rows.length === 1 ? "note" : "notes"} · original text and clinical dates
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-wrap items-center gap-2">
          <p className="flex-1 text-xs text-gray-600">
            Each note scrolls in its own box; Show whole note expands it.
          </p>
          <Button
            size="sm"
            variant="outline"
            disabled={!rows.length}
            onClick={() => {
              setExpanded(all ? new Set() : new Set(rows.map((_, i) => i)));
            }}
          >
            {all ? "Collapse notes" : "Show whole notes"}
          </Button>
        </div>
        <div className="overflow-x-auto rounded-md border border-gray-200">
          <table className="w-full border-collapse text-sm">
            <thead className="bg-gray-50 text-left text-xs text-gray-600">
              <tr>
                <th scope="col" className="w-40 px-3 py-2 font-medium">
                  Date / note
                </th>
                <th scope="col" className="px-3 py-2 font-medium">
                  Note
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {rows.map((row, i) => {
                const open = expanded.has(i);
                const box = cn(
                  "scrollbar-visible overflow-y-auto rounded-sm font-mono text-xs break-words whitespace-pre-wrap text-gray-800 outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
                  !open && "max-h-48",
                );
                return (
                  <tr key={i} className="align-top">
                    <th scope="row" className="px-3 py-2 text-left font-normal">
                      <div className="flex flex-col items-start gap-1">
                        <strong className="text-sm">Note {i + 1}</strong>
                        <span className="text-xs text-gray-600">{noteDate(row.note_date)}</span>
                        {row.note_type ? (
                          <span className="text-xs text-gray-500">{row.note_type}</span>
                        ) : null}
                        <Button
                          size="xs"
                          variant="outline"
                          aria-expanded={open}
                          onClick={() => {
                            toggle(i);
                          }}
                        >
                          {open ? "Collapse note" : "Show whole note"}
                        </Button>
                      </div>
                    </th>
                    <td className="px-3 py-2">
                      <div tabIndex={0} aria-label={`Note ${String(i + 1)}`} className={box}>
                        {row.note_text}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {!rows.length ? (
            <p className="p-4 text-sm text-gray-500 italic">No notes have been added.</p>
          ) : null}
        </div>
        <DialogFooter className="pt-1">
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
