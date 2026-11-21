import { Link } from "@tanstack/react-router";
import { ArrowLeftIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";

import { LegendItem, STATE_COLORS, ThinSegmentedBar } from "./charts";

const STATE_ORDER = ["done", "running", "blocked", "failed", "pending"];
const GREY = "#d1d5db";

/** A titled section of a run page, with an optional action beside the title. */
export function Section({
  title,
  description,
  action,
  children,
}: {
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        {action}
      </div>
      {description ? <p className="max-w-4xl text-sm text-gray-600">{description}</p> : null}
      {children}
    </section>
  );
}

export function BackToRuns() {
  return (
    <Button
      variant="outline"
      size="sm"
      className="self-start"
      nativeButton={false}
      render={<Link to="/benchmarking" />}
    >
      <ArrowLeftIcon />
      Benchmarking
    </Button>
  );
}

/** Job states in the bar's order, then any others. */
function orderedStates(counts: Record<string, number>): string[] {
  return [
    ...STATE_ORDER.filter((s) => counts[s]),
    ...Object.keys(counts).filter((s) => !STATE_ORDER.includes(s) && counts[s]),
  ];
}

/** A thin bar of job counts by state with its legend (status colours, each named). */
export function StateBar({ counts }: { counts: Record<string, number> }) {
  const states = orderedStates(counts);
  return (
    <div className="flex flex-col gap-1.5">
      <ThinSegmentedBar
        label="Stages by state"
        parts={states.map((s) => ({
          key: s,
          label: s,
          value: counts[s] ?? 0,
          color: STATE_COLORS[s] ?? GREY,
        }))}
      />
      <div className="flex flex-wrap gap-x-3 gap-y-1">
        {states.map((s) => (
          <LegendItem key={s} color={STATE_COLORS[s] ?? GREY}>
            {counts[s]} {s}
          </LegendItem>
        ))}
      </div>
    </div>
  );
}
