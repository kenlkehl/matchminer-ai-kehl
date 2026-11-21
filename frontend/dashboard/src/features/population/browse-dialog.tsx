import { FolderIcon, FolderUpIcon } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

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

import { fileDate, fileSize } from "./labels";
import type { FolderListing } from "./types";
import { errorMessage } from "./use-population";

export type BrowseKind = "database" | "cache";

const COPY: Record<BrowseKind, [string, string]> = {
  database: [
    "Choose a DuckDB database",
    "Folders and .duckdb files on this server, within the configured database folders. Choosing a file checks its tables without reading patient rows.",
  ],
  cache: [
    "Choose a population cache",
    "Folders and preprocessed population caches on this server, within the configured database folders. Choosing a cache reads only its manifest.",
  ],
};

const LINK = "inline-flex items-center gap-1.5 text-left text-brand-blue hover:underline";

/** Real mode: pick a DuckDB file or a population cache from the server's database folders. */
export function BrowseDialog({
  kind,
  start,
  onPick,
  onClose,
}: {
  kind: BrowseKind;
  /** The folder to open first (a typed path outside the database folders opens the default one). */
  start: string;
  onPick: (path: string) => void;
  onClose: () => void;
}) {
  const [listing, setListing] = useState<FolderListing | null>(null);
  const [problem, setProblem] = useState("");
  const [loading, setLoading] = useState(false);
  const sequence = useRef(0);

  const browse = useCallback(async (path: string, fallback: boolean) => {
    const mine = ++sequence.current;
    setLoading(true);
    setProblem("");
    const list = (p: string) =>
      api.get<FolderListing>(`/api/population/databases?path=${encodeURIComponent(p)}`);
    try {
      let found: FolderListing;
      try {
        found = await list(path);
      } catch (err) {
        if (!fallback || !path) throw err;
        found = await list("");
      }
      if (mine === sequence.current) setListing(found);
    } catch (err) {
      if (mine === sequence.current) setProblem(errorMessage(err));
    } finally {
      if (mine === sequence.current) setLoading(false);
    }
  }, []);

  // The dialog opens on `start` (it is remounted for each opening).
  const [first] = useState(start);
  useEffect(() => {
    void browse(first, true);
  }, [browse, first]);

  const [title, help] = COPY[kind];
  const base = listing?.path.replace(/\/$/, "") ?? "";
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="h-[80svh] sm:max-w-3xl" bodyClassName="gap-3 auto-rows-min">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{help}</DialogDescription>
        </DialogHeader>
        <p className="flex items-center gap-2 font-mono text-xs break-all text-gray-700">
          {loading ? <Spinner className="size-3" /> : null}
          {listing?.path ?? ""}
        </p>
        {problem ? <p className="text-sm text-red-700">{problem}</p> : null}
        {listing ? (
          <Table variant="list">
            <TableCaption>
              {kind === "cache"
                ? `${String(listing.caches.length)} population caches`
                : `${String(listing.files.length)} DuckDB files`}{" "}
              · {listing.directories.length} folders
              {listing.truncated ? " · listing truncated" : ""}
            </TableCaption>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Modified</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {listing.parent ? (
                <TableRow>
                  <TableCell>
                    <button
                      type="button"
                      className={LINK}
                      onClick={() => void browse(listing.parent ?? "", false)}
                    >
                      <FolderUpIcon className="size-4" />
                      Parent folder
                    </button>
                  </TableCell>
                  <TableCell />
                  <TableCell />
                </TableRow>
              ) : null}
              {kind === "cache"
                ? listing.caches.map((c) => (
                    <TableRow key={c.path}>
                      <TableCell>
                        <button type="button" className={LINK} onClick={() => onPick(c.path)}>
                          {c.name}
                        </button>
                      </TableCell>
                      <TableCell>Population cache</TableCell>
                      <TableCell />
                    </TableRow>
                  ))
                : listing.files.map((f) => (
                    <TableRow key={f.path}>
                      <TableCell>
                        <button type="button" className={LINK} onClick={() => onPick(f.path)}>
                          {f.name}
                        </button>
                      </TableCell>
                      <TableCell className="tabular-nums">{fileSize(f.size)}</TableCell>
                      <TableCell>{fileDate(f.modified)}</TableCell>
                    </TableRow>
                  ))}
              {listing.directories.map((name) => (
                <TableRow key={name}>
                  <TableCell>
                    <button
                      type="button"
                      className={LINK}
                      onClick={() => void browse(`${base}/${name}`, false)}
                    >
                      <FolderIcon className="size-4" />
                      {name}/
                    </button>
                  </TableCell>
                  <TableCell>Folder</TableCell>
                  <TableCell />
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : null}
      </DialogContent>
    </Dialog>
  );
}
