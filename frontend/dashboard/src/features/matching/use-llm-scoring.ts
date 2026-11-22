import { useCallback, useEffect, useRef, useState } from "react";

import {
  type Direction,
  type LLMEstimate,
  type LLMPairResult,
  lookupPairs,
  scorePairs,
} from "@/lib/llm-decisions";

/** At most this many pairs per lookup (the server's MAX_PAIRS). */
const LOOKUP = 50;

export interface LLMScoring {
  /** The session's LLM answers, by result id. */
  results: Record<string, LLMPairResult>;
  /** The ids being scored now, and the job's progress. */
  running: string[];
  progress: { done: number; total: number; message: string } | null;
  /** The running (or last) run's estimate. */
  estimate: LLMEstimate | null;
  /** A run waiting for the user's confirmation (more than the server's LARGE_RUN requests). */
  pending: { ids: string[]; estimate: LLMEstimate } | null;
  error: string | null;
  score: (ids: string[], options?: { rerun?: boolean }) => void;
  confirm: () => void;
  dismiss: () => void;
}

interface Batch {
  ids: string[];
  rerun: boolean;
  confirmed: boolean;
}

/**
 * The LLM decision checks of one search's results (when `enabled`): the session's answers the server already has,
 * then the first `autoN` results without an LLM match answer, scored in a session job as the view opens; more on
 * request. Every run is estimated first (requests and model); one of more than the server's LARGE_RUN requests
 * waits for the user's confirmation. Runs go one after another.
 */
export function useLLMScoring(
  direction: Direction,
  queryId: string | null,
  rows: { id: string }[] | null,
  autoN: number,
  enabled: boolean,
): LLMScoring {
  const [results, setResults] = useState<Record<string, LLMPairResult>>({});
  const [running, setRunning] = useState<string[]>([]);
  const [progress, setProgress] = useState<LLMScoring["progress"]>(null);
  const [estimate, setEstimate] = useState<LLMEstimate | null>(null);
  const [pending, setPending] = useState<LLMScoring["pending"]>(null);
  const [error, setError] = useState<string | null>(null);
  const queue = useRef<Batch[]>([]);
  const busy = useRef(false);
  const generation = useRef(0);

  const merge = useCallback(
    (found: LLMPairResult[]) => {
      setResults((current) => {
        const out = { ...current };
        for (const r of found) {
          const id = direction === "patients_for_cohort" ? r.patient_id : r.cohort_id;
          out[id] = {
            ...out[id],
            ...Object.fromEntries(Object.entries(r).filter(([, v]) => v)),
          } as LLMPairResult;
        }
        return out;
      });
    },
    [direction],
  );

  const pump = useCallback(() => {
    if (busy.current || !queryId) return;
    const next = queue.current.shift();
    if (!next) return;
    busy.current = true;
    const mine = generation.current;
    const body = { direction, query_id: queryId, target_ids: next.ids, rerun: next.rerun };
    const run = async () => {
      const looked = await lookupPairs(body);
      if (mine !== generation.current) return;
      merge(looked.results);
      const e = looked.estimate;
      if (!next.rerun && !e.requests) return;
      if (e.large && !next.confirmed) {
        setPending({ ids: next.ids, estimate: e });
        return;
      }
      setEstimate(e);
      setRunning(next.ids);
      setProgress({ done: 0, total: e.requests, message: "queued" });
      const found = await scorePairs({ ...body, confirm: next.confirmed }, (j) => {
        if (mine === generation.current)
          setProgress({ done: j.done, total: j.total, message: j.message });
      });
      if (mine === generation.current) merge(found);
    };
    run()
      .catch((e: unknown) => {
        if (mine === generation.current) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        busy.current = false;
        if (mine === generation.current) {
          setRunning([]);
          setProgress(null);
        }
        pump();
      });
  }, [direction, queryId, merge]);

  // A new search (or the LLM switched on): the session's answers, then the first results lacking one.
  useEffect(() => {
    generation.current += 1;
    const mine = generation.current;
    queue.current = [];
    busy.current = false;
    setRunning([]);
    setProgress(null);
    setPending(null);
    setError(null);
    setResults({});
    if (!enabled || !rows || !queryId) return;
    const ids = rows.map((r) => r.id);
    const lookups = [];
    for (let i = 0; i < ids.length; i += LOOKUP)
      lookups.push(
        lookupPairs({ direction, query_id: queryId, target_ids: ids.slice(i, i + LOOKUP) }),
      );
    void Promise.all(lookups)
      .then((found) => {
        if (mine !== generation.current) return;
        const all = found.flatMap((f) => f.results);
        merge(all);
        const answered = new Set(
          all
            .filter((r) => r.match)
            .map((r) => (direction === "patients_for_cohort" ? r.patient_id : r.cohort_id)),
        );
        const todo = ids.slice(0, autoN).filter((id) => !answered.has(id));
        if (todo.length) {
          queue.current.push({ ids: todo, rerun: false, confirmed: false });
          pump();
        }
      })
      .catch((e: unknown) => {
        if (mine === generation.current) setError(e instanceof Error ? e.message : String(e));
      });
  }, [rows, queryId, direction, autoN, enabled, merge, pump]);

  const score = useCallback(
    (ids: string[], options?: { rerun?: boolean }) => {
      const queued = new Set([...running, ...queue.current.flatMap((b) => b.ids)]);
      const todo = options?.rerun ? ids : ids.filter((id) => !queued.has(id));
      if (!todo.length) return;
      setPending(null);
      setError(null);
      queue.current.push({ ids: todo, rerun: Boolean(options?.rerun), confirmed: false });
      pump();
    },
    [running, pump],
  );

  const confirm = useCallback(() => {
    if (!pending) return;
    setPending(null);
    queue.current.unshift({ ids: pending.ids, rerun: false, confirmed: true });
    pump();
  }, [pending, pump]);

  const dismiss = useCallback(() => {
    setPending(null);
  }, []);

  return { results, running, progress, estimate, pending, error, score, confirm, dismiss };
}
