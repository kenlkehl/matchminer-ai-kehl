import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError, isActive, type Job, type JobStart, stopJob } from "@/lib/api";

import type { PopulationDefaults, PopulationState, PopulationView } from "./types";

const POLL_MS = 1000;

export const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : "The request failed. Check the inputs, then retry.";

/** The view's job as it starts, before the server's first answer. */
const preparing = (): Job<PopulationView | null> => ({
  id: "",
  kind: "population",
  label: "population",
  status: "queued",
  done: 0,
  total: 0,
  message: "Preparing population workflow…",
  error: null,
  device: null,
  created: Date.now() / 1000,
  updated: Date.now() / 1000,
  started: null,
  finished: null,
  seconds: null,
});

export interface PopulationController {
  /** The first `GET /api/population` answered. */
  ready: boolean;
  initializing: boolean;
  population: PopulationView | null;
  defaults: PopulationDefaults | null;
  /** The job this view follows (or the last one it followed). */
  job: Job<PopulationView | null> | null;
  /** A job runs or a request is in flight: every action waits. */
  busy: boolean;
  error: string;
  notice: string;
  setError: (message: string) => void;
  setNotice: (message: string) => void;
  init: () => Promise<void>;
  /** Start a population job (a JSON body, or a multipart upload) and follow it. */
  start: (path: string, body: unknown, upload?: Record<string, Blob>) => Promise<void>;
  /** Run a request that answers at once (an archive import, a download), keeping the view busy. */
  request: <T>(run: () => Promise<T>) => Promise<T | undefined>;
  apply: (population: PopulationView | null) => void;
  /** Reload the session's population (a drilldown found the view stale). */
  reload: () => Promise<void>;
  stop: () => Promise<void>;
  clear: () => Promise<void>;
}

/**
 * The browser session's population and its one population job: restored from `GET /api/population`
 * (resuming a running job), started from the panels, polled until it ends. Every mutation sends the
 * view's `version`; after a failure the view is reloaded, so a stale version is replaced before a retry.
 */
export function usePopulation(): PopulationController {
  const [ready, setReady] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [population, setPopulation] = useState<PopulationView | null>(null);
  const [defaults, setDefaults] = useState<PopulationDefaults | null>(null);
  const [job, setJob] = useState<Job<PopulationView | null> | null>(null);
  const [followId, setFollowId] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const follow = useCallback((started: Job<PopulationView | null>) => {
    setJob(started);
    setFollowId(started.id);
  }, []);

  /** Reload the session's population (after a failure, a stale version or a stopped job). */
  const resync = useCallback(async () => {
    try {
      const saved = await api.get<PopulationState>("/api/population");
      if (!alive.current) return;
      setPopulation((current) =>
        current?.version === saved.population?.version ? current : saved.population,
      );
      if (saved.job && isActive(saved.job)) follow(saved.job);
    } catch {
      // The failure that led here is already shown.
    }
  }, [follow]);

  // Follow one job at a time; a newer job (or leaving the page) ends the older poll.
  useEffect(() => {
    if (!followId) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const current = await api.get<Job<PopulationView | null>>(
          `/api/jobs/${encodeURIComponent(followId)}`,
        );
        if (!live) return;
        setJob(current);
        if (current.status === "done") {
          setFollowId(null);
          if (current.result !== undefined) setPopulation(current.result);
        } else if (current.status === "error") {
          setFollowId(null);
          setError(current.error ?? "The population job failed.");
          void resync();
        } else if (current.status === "cancelled") {
          setFollowId(null);
          setNotice("Stopped. The last completed cohort stage is preserved.");
          void resync();
        } else {
          timer = setTimeout(() => void tick(), POLL_MS);
        }
      } catch (err) {
        if (!live) return;
        setFollowId(null);
        setJob(null);
        setError(errorMessage(err));
      }
    };
    void tick();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [followId, resync]);

  const init = useCallback(async () => {
    setInitializing(true);
    setError("");
    try {
      const saved = await api.get<PopulationState>("/api/population");
      if (!alive.current) return;
      setDefaults(saved.defaults);
      setPopulation(saved.population);
      setReady(true);
      if (saved.job && isActive(saved.job)) follow(saved.job);
      else if (saved.job?.status === "error") setError(saved.job.error ?? "The last job failed.");
    } catch (err) {
      if (!alive.current) return;
      setError(
        err instanceof ApiError && err.status === 404
          ? "Population guideline concordance is unavailable on this server. Restart the dashboard to load the updated backend, then retry."
          : `Could not connect to population management. ${errorMessage(err)}`,
      );
    } finally {
      if (alive.current) setInitializing(false);
    }
  }, [follow]);

  useEffect(() => {
    void init();
  }, [init]);

  const start = useCallback(
    async (path: string, body: unknown, upload?: Record<string, Blob>) => {
      setError("");
      setNotice("");
      setPending(true);
      setJob(preparing());
      try {
        const started = upload
          ? await api.upload<JobStart>(path, upload)
          : await api.post<JobStart>(path, body);
        if (alive.current) follow(started.job as Job<PopulationView | null>);
      } catch (err) {
        if (!alive.current) return;
        setJob(null);
        setError(errorMessage(err));
        void resync();
      } finally {
        if (alive.current) setPending(false);
      }
    },
    [follow, resync],
  );

  const request = useCallback(
    async <T>(run: () => Promise<T>): Promise<T | undefined> => {
      setError("");
      setPending(true);
      try {
        return await run();
      } catch (err) {
        if (alive.current) {
          setError(errorMessage(err));
          void resync();
        }
        return undefined;
      } finally {
        if (alive.current) setPending(false);
      }
    },
    [resync],
  );

  const stop = useCallback(async () => {
    if (!followId) return;
    try {
      const stopping = await stopJob(followId);
      if (alive.current) {
        setJob((current) =>
          current?.id === stopping.id
            ? { ...current, status: stopping.status, message: stopping.message }
            : current,
        );
      }
    } catch (err) {
      if (alive.current) setError(errorMessage(err));
    }
  }, [followId]);

  const clear = useCallback(async () => {
    const done = await request(async () => {
      await api.delete("/api/population");
      return true;
    });
    if (done && alive.current) {
      setPopulation(null);
      setJob(null);
      setNotice("");
    }
  }, [request]);

  return {
    ready,
    initializing,
    population,
    defaults,
    job,
    busy: pending || followId !== null,
    error,
    notice,
    setError,
    setNotice,
    init,
    start,
    request,
    apply: setPopulation,
    reload: resync,
    stop,
    clear,
  };
}
