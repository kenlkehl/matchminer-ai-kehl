import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { type ComponentProps, type ReactNode, useState } from "react";

import { Spinner } from "@/components/ui/spinner";
import { api } from "@/lib/api";

import { errorMessage } from "./use-population";

/** The native select of the tab's forms (long lists of spaces read better there than in a popup). */
export const SELECT =
  "h-8 w-full min-w-0 rounded-md border border-input bg-background px-2 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50";

/** A numbered workflow step of the controls column. */
export function Step({
  number,
  eyebrow,
  title,
  children,
}: {
  number: string;
  eyebrow: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3 border-b border-gray-200 p-4 last:border-b-0">
      <div>
        <p className="text-[0.7rem] font-semibold tracking-wider text-brand-blue uppercase">
          {number} / {eyebrow}
        </p>
        <h2 className="text-base font-semibold">{title}</h2>
      </div>
      {children}
    </section>
  );
}

export function Helper({ className, ...props }: ComponentProps<"p">) {
  return <p className={cn("text-xs text-gray-600", className)} {...props} />;
}

/** A labelled control: the label above, any help below. */
export function LabeledField({
  label,
  htmlFor,
  children,
  className,
}: {
  label: string;
  htmlFor: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1", className)}>
      <label htmlFor={htmlFor} className="text-xs font-medium text-gray-700">
        {label}
      </label>
      {children}
    </div>
  );
}

/** A folded group of controls (the fetcher's "advanced" sections). */
export function Fold({
  summary,
  children,
  defaultOpen,
}: {
  summary: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details
      className="group rounded-md border border-gray-200 bg-gray-50/60 open:bg-white"
      open={defaultOpen}
    >
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-gray-700 select-none">
        {summary}
      </summary>
      <div className="flex flex-col gap-3 border-t border-gray-200 p-3">{children}</div>
    </details>
  );
}

/**
 * A server path checked when it is entered (on change, Enter or a Browse pick): the answer shown is
 * always the current path's, as the fetcher's sequence counters ensured.
 */
export function usePathCheck<T>(kind: string, url: string) {
  const [checked, setChecked] = useState<{ path: string; n: number } | null>(null);
  const query = useQuery({
    queryKey: ["population", "check", kind, checked?.path, checked?.n],
    queryFn: () => api.get<T>(`${url}?path=${encodeURIComponent(checked?.path ?? "")}`),
    enabled: checked !== null,
    retry: false,
    staleTime: Infinity,
    gcTime: 60_000,
  });
  return {
    check: (path: string) => {
      setChecked((current) => ({ path: path.trim(), n: (current?.n ?? 0) + 1 }));
    },
    checkedPath: checked?.path ?? null,
    query,
  };
}

/** A path check's line: checking, the server's facts, or why the path cannot be used. */
export function CheckStatus<T>({
  id,
  query,
  checking,
  describe,
}: {
  id: string;
  query: { isFetching: boolean; data: T | undefined; error: Error | null };
  checking: string;
  describe: (value: T) => string;
}) {
  return (
    <p id={id} role="status" className="text-xs break-words text-gray-600">
      {query.isFetching ? (
        <span className="inline-flex items-center gap-1">
          <Spinner className="size-3" />
          {checking}
        </span>
      ) : query.error ? (
        <span className="text-red-700">{errorMessage(query.error)}</span>
      ) : query.data !== undefined ? (
        describe(query.data)
      ) : null}
    </p>
  );
}
