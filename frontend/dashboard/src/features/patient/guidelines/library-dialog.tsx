import { queryOptions, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RotateCwIcon } from "lucide-react";
import { useState } from "react";

import { SearchInput } from "@/components/search-input";
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
import { api } from "@/lib/api";

import type { CatalogList } from "../types";

/** The completed guideline catalogs (each with a passing stored audit). */
export const catalogsOptions = () =>
  queryOptions({
    queryKey: ["guideline-catalogs"],
    queryFn: () => api.get<CatalogList>("/api/guidelines/catalogs"),
    staleTime: 60_000,
  });

/** The guideline library: the completed catalogs, filterable, each selectable for the next search. */
export function LibraryDialog({
  disabled,
  onSelect,
  onClose,
}: {
  disabled: boolean;
  onSelect: (catalogId: string) => void;
  onClose: () => void;
}) {
  const client = useQueryClient();
  const library = useQuery(catalogsOptions());
  const [q, setQ] = useState("");
  const refresh = useMutation({
    mutationFn: () => api.get<CatalogList>("/api/guidelines/catalogs?refresh=true"),
    onSuccess: (data) => {
      client.setQueryData(catalogsOptions().queryKey, data);
    },
  });
  const query = q.trim().toLowerCase();
  const rows = (library.data?.catalogs ?? []).filter((r) =>
    `${r.title} ${r.disease}`.toLowerCase().includes(query),
  );
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[85svh] sm:max-w-2xl" bodyClassName="gap-3 auto-rows-min">
        <DialogHeader>
          <DialogTitle>Guideline library</DialogTitle>
          <DialogDescription className="text-gray-600">
            Only completed catalogs with a passing stored audit appear here. Refresh as more
            guidelines finish processing.
          </DialogDescription>
        </DialogHeader>
        <div className="flex gap-2">
          <div className="flex-1">
            <SearchInput value={q} placeholder="Find a guideline…" onChange={setQ} />
          </div>
          <Button
            variant="outline"
            disabled={refresh.isPending}
            onClick={() => {
              refresh.mutate();
            }}
          >
            {refresh.isPending ? <Spinner /> : <RotateCwIcon />}
            Refresh
          </Button>
        </div>
        {refresh.isError ? <p className="text-sm text-red-700">{refresh.error.message}</p> : null}
        {library.data ? (
          <p className="text-xs text-gray-600">
            {library.data.catalogs.length} completed guidelines ·{" "}
            {library.data.spaces.toLocaleString()} catalog spaces
          </p>
        ) : null}
        {library.data ? (
          rows.length ? (
            <ul className="divide-y divide-gray-100 rounded-md border border-gray-200">
              {rows.map((r) => (
                <li key={r.id} className="flex items-center gap-3 px-3 py-2">
                  <div className="min-w-0 flex-1">
                    <h3 className="text-sm font-medium text-gray-900">{r.title}</h3>
                    <p className="text-xs text-gray-600">
                      {r.version} · {r.spaces} spaces · Stored audit passed
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={disabled}
                    onClick={() => {
                      onSelect(r.id);
                    }}
                  >
                    Select for search →
                  </Button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-500 italic">
              No completed guidelines match this search.
            </p>
          )
        ) : library.isError ? (
          <p className="text-sm text-red-700">{library.error.message}</p>
        ) : (
          <div className="flex justify-center py-6">
            <Spinner />
          </div>
        )}
        <DialogFooter className="pt-1">
          <DialogClose render={<Button variant="outline">Close</Button>} />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
