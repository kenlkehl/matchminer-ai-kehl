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
import { saveBlob } from "@/lib/api";

/** A value to show as JSON: its title, a file name for download, the value and, optionally, what it is. */
export interface JsonView {
  title: string;
  name: string;
  value: unknown;
  description?: string;
}

/** Raw JSON (an extracted item or record, an LLM answer) in a dialog, with Copy and Download. */
export function JsonDialog({ view, onClose }: { view: JsonView; onClose: () => void }) {
  const text = JSON.stringify(view.value, null, 2);
  const [copied, setCopied] = useState(false);
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[85svh] sm:max-w-3xl" bodyClassName="gap-3 auto-rows-min">
        <DialogHeader>
          <DialogTitle>{view.title}</DialogTitle>
          <DialogDescription className="text-gray-600">
            {view.description ?? "As extracted: evidence offsets index the note's text."}
          </DialogDescription>
        </DialogHeader>
        <pre className="max-h-[60svh] overflow-auto rounded-md border border-gray-200 bg-gray-50 p-3 font-mono text-xs whitespace-pre-wrap text-gray-800">
          {text}
        </pre>
        <DialogFooter className="pt-1">
          <Button
            variant="outline"
            onClick={() => {
              void navigator.clipboard.writeText(text).then(() => {
                setCopied(true);
              });
            }}
          >
            {copied ? "Copied" : "Copy"}
          </Button>
          <Button
            variant="outline"
            onClick={() => {
              saveBlob(new Blob([text], { type: "application/json" }), view.name);
            }}
          >
            Download
          </Button>
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** A small "JSON" link button. */
export function JsonLink({ onClick, label = "JSON" }: { onClick: () => void; label?: string }) {
  return (
    <button type="button" className="shrink-0 link text-xs" onClick={onClick}>
      {label}
    </button>
  );
}
