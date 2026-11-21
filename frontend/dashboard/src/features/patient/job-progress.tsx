import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";

import { stopActiveJob } from "./actions";
import type { ActiveJob } from "./store";

function elapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  return `${String(Math.floor(seconds / 60))}:${String(seconds % 60).padStart(2, "0")}`;
}

/** The running guideline job: what it does, how long it has run, and Stop. */
export function JobProgress({ job }: { job: ActiveJob }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => {
      setNow(Date.now());
    }, 1000);
    return () => {
      clearInterval(timer);
    };
  }, []);
  const share = job.total > 0 ? Math.min(1, job.done / job.total) : null;
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex flex-col gap-2 rounded-lg border border-brand-blue/30 bg-brand-blue/5 p-4"
    >
      <div className="flex items-center gap-2">
        <Spinner className="text-brand-blue" />
        <strong className="text-sm">{job.title}</strong>
        <span className="text-sm text-gray-600 tabular-nums">{elapsed(now - job.started)}</span>
        <Button
          size="sm"
          variant="outline"
          className="ml-auto"
          disabled={!job.id || job.stopping}
          onClick={() => {
            void stopActiveJob();
          }}
        >
          {job.stopping ? "Stopping…" : "Stop"}
        </Button>
      </div>
      <p className="text-sm text-gray-700">{job.message}</p>
      {share !== null ? (
        <div className="h-1.5 overflow-hidden rounded-full bg-gray-200">
          <div
            className="h-full bg-brand-blue transition-all"
            style={{ width: `${String(share * 100)}%` }}
          />
        </div>
      ) : null}
      <p className="text-xs text-gray-500">
        {job.kind === "retrieve"
          ? "Stop ends the search at its next step. Reloading the page keeps the job running."
          : "Stop ends local work and prevents further requests; a request the LLM server already accepted may still finish there. Reloading the page keeps the job running."}
      </p>
    </div>
  );
}
