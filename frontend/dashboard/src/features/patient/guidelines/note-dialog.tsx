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

import { splitSpan } from "../notes";

/** A note to show with one quoted span marked: the span's character offsets in the note's text. */
export interface NoteView {
  title: string;
  text: string;
  start: number | null;
  end: number | null;
  /** Why the span is not marked (e.g. a quote not found verbatim). */
  unmarked?: string;
}

/** One note's whole text, its quoted span marked and scrolled into view. */
export function NoteDialog({ note, onClose }: { note: NoteView; onClose: () => void }) {
  const parts = splitSpan(note.text, note.start, note.end);
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[85svh] sm:max-w-4xl" bodyClassName="gap-3 auto-rows-min">
        <DialogHeader>
          <DialogTitle>{note.title}</DialogTitle>
          <DialogDescription className="text-gray-600">
            {parts
              ? "The quoted span is marked in the note text. Findings are review aids; read the whole note."
              : (note.unmarked ?? "Findings are review aids; read the whole note.")}
          </DialogDescription>
        </DialogHeader>
        <div className="rounded-md border border-gray-200 bg-gray-50 p-3 font-mono text-xs break-words whitespace-pre-wrap text-gray-800">
          {parts ? (
            <>
              {parts[0]}
              <mark
                ref={(el) => {
                  el?.scrollIntoView({ block: "center" });
                }}
                className="rounded-sm bg-amber-200 px-0.5 text-gray-900"
              >
                {parts[1]}
              </mark>
              {parts[2]}
            </>
          ) : (
            note.text
          )}
        </div>
        <DialogFooter className="pt-1">
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
