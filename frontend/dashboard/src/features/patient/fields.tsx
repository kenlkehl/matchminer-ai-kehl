import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { cn } from "cn";
import { XIcon } from "lucide-react";
import type { ComponentProps, ReactNode } from "react";

import { connectionLabel, modelsOptions, type Task } from "@/lib/models";

import { dismissStatus } from "./actions";
import { type StatusMessage, usePatientValue } from "./store";

/** A native select, styled as the app's other filters. */
export function NativeSelect({ className, ...props }: ComponentProps<"select">) {
  return (
    <select
      className={cn(
        "h-9 rounded-md border border-input bg-background px-2 text-sm disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      {...props}
    />
  );
}

/** A labelled control, its label above it and an optional helper line below. */
export function Field({
  label,
  htmlFor,
  help,
  className,
  children,
}: {
  label: ReactNode;
  htmlFor?: string;
  help?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={cn("flex flex-col gap-1", className)}>
      <label htmlFor={htmlFor} className="text-sm font-medium">
        {label}
      </label>
      {children}
      {help ? <p className="text-xs text-gray-500">{help}</p> : null}
    </div>
  );
}

/** The latest message for one area of the page (an error, or what just finished). */
export function StatusLine({ where }: { where: StatusMessage["where"] }) {
  const status = usePatientValue((s) => s.status);
  if (!status || status.where !== where) return null;
  return (
    <div
      role={status.kind === "error" ? "alert" : "status"}
      className={cn(
        "flex items-start gap-2 rounded-md border px-3 py-2 text-sm",
        status.kind === "error"
          ? "border-red-200 bg-red-50 text-red-800"
          : "border-emerald-200 bg-emerald-50 text-emerald-900",
      )}
    >
      <p className="flex-1">{status.text}</p>
      <button
        type="button"
        aria-label="Dismiss"
        className="shrink-0 text-current opacity-60 hover:opacity-100"
        onClick={dismissStatus}
      >
        <XIcon className="size-4" />
      </button>
    </div>
  );
}

/**
 * Where the notes go: the server's notice on patient text and LLM endpoints, the tasks' connections,
 * and a link to change them on the Data and Models page.
 */
export function EndpointNotice({ tasks }: { tasks: Task[] }) {
  const config = useQuery(modelsOptions()).data;
  if (!config) return null;
  return (
    <p className="text-xs text-gray-600">
      {config.notice}{" "}
      {tasks.map((task, i) => {
        const connection = config.tasks[task];
        return (
          <span key={task}>
            {i ? " · " : ""}
            {connection.label}:{" "}
            <span className="text-gray-800">{connectionLabel(config, connection)}</span>
          </span>
        );
      })}
      {" · "}
      <Link to="/status" className="link">
        Model connections
      </Link>
    </p>
  );
}
