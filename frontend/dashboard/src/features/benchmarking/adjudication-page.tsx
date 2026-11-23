import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getRouteApi, Link } from "@tanstack/react-router";
import { ArrowLeftIcon, ChevronLeftIcon, ChevronRightIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";

import { ItemHeader, UnitCard } from "./adjudication-review";
import { type Highlight, scrollWithin, SourcePane } from "./adjudication-source";
import { componentLabel } from "./format";
import { adjudicationItemOptions, adjudicationStateOptions, judgeOptions } from "./queries";
import type { AdjudicationItem, AdjudicationItemRow, Choice, DecideBody, UnitView } from "./types";

const route = getRouteApi("/_app/benchmarking/judge/$runId/adjudicate");

const COMPONENTS = ["extraction", "summary", "decision"];

function hasOpen(row: AdjudicationItemRow): boolean {
  return row.decided < row.queued;
}

/**
 * Adjudication of a judge run's queued units: the item's source on the left, its units on the right, judges
 * blinded as Judge 1 and Judge 2 until a decision. Keys: j/k move, 1/2 take a judge's answer, a-h pick a choice,
 * n note, s skip, [ ] change item.
 */
export function AdjudicationPage() {
  const { runId } = route.useParams();
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const client = useQueryClient();
  const run = useQuery(judgeOptions(runId)).data;
  const state = useQuery(adjudicationStateOptions(runId));
  const [showAll, setShowAll] = useState(false);
  const [showRef, setShowRef] = useState(false);

  const components = COMPONENTS.filter((c) => state.data?.components[c]?.items.length);
  const component =
    search.component && components.includes(search.component)
      ? search.component
      : (components.find((c) => state.data?.components[c]?.items.some(hasOpen)) ?? components[0]);
  const items = (component && state.data?.components[component]?.items) || [];
  const itemId =
    search.item && items.some((i) => i.item_id === search.item)
      ? search.item
      : (items.find(hasOpen) ?? items.find((i) => i.queued > 0) ?? items[0])?.item_id;
  // The items the list and the arrows go through: those with queued units (all of them with "Show agreed
  // units"), and the item open now.
  const listed = items.filter((i) => showAll || i.queued > 0 || i.item_id === itemId);
  const at = listed.findIndex((i) => i.item_id === itemId);

  const itemQuery = useQuery({
    ...adjudicationItemOptions(runId, component ?? "", itemId ?? "", showRef),
    enabled: !!component && !!itemId,
  });
  // A previous item's data (placeholder) while the next loads is shown dimmed, never as this item.
  const item = itemQuery.data?.item_id === itemId ? itemQuery.data : undefined;

  const open = useCallback(
    (nextComponent: string, nextItem: string | undefined) => {
      void navigate({ search: { component: nextComponent, item: nextItem }, replace: true });
    },
    [navigate],
  );

  // Pin the item shown in the address, so deciding its last open unit does not move on to another.
  useEffect(() => {
    if (component && itemId && (search.component !== component || search.item !== itemId)) {
      open(component, itemId);
    }
  }, [component, itemId, search.component, search.item, open]);

  // One item back or on in the list, decided or not.
  const step = useCallback(
    (direction: number) => {
      const row = listed[at + direction];
      if (component && at >= 0 && row) open(component, row.item_id);
    },
    [component, listed, at, open],
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="outline"
          size="sm"
          nativeButton={false}
          render={<Link to="/benchmarking/judge/$runId" params={{ runId }} />}
        >
          <ArrowLeftIcon />
          {run?.name ?? "Run"}
        </Button>
        <h1 className="text-xl font-bold">Adjudication</h1>
        <p className="text-sm text-gray-600">
          Judges are shown as Judge 1 and Judge 2 and named once you decide.
        </p>
      </div>
      {state.isPending ? (
        <Spinner className="size-6" />
      ) : state.isError ? (
        <p className="text-sm text-red-700">{state.error.message}</p>
      ) : !component ? (
        <p className="text-sm text-gray-600">This run has no items to adjudicate.</p>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-y bg-gray-50 px-3 py-2">
            <Tabs
              value={component}
              onValueChange={(value) => {
                open(String(value), undefined);
              }}
            >
              <TabsList className="bg-gray-200/70">
                {components.map((c) => {
                  const s = state.data.components[c];
                  return (
                    <TabsTrigger key={c} value={c} className="px-3">
                      {componentLabel(c)}{" "}
                      <span className="text-xs text-gray-500 tabular-nums">
                        {s?.decided ?? 0}/{s?.queued ?? 0}
                      </span>
                    </TabsTrigger>
                  );
                })}
              </TabsList>
            </Tabs>
            <div className="flex items-center gap-1">
              <Button
                variant="outline"
                size="icon-sm"
                title="Previous item ([)"
                aria-label="Previous item"
                disabled={at <= 0}
                onClick={() => {
                  step(-1);
                }}
              >
                <ChevronLeftIcon />
              </Button>
              <select
                aria-label="Item"
                className="h-8 max-w-[28rem] rounded-md border border-input bg-white px-2 text-sm"
                value={itemId ?? ""}
                onChange={(e) => {
                  open(component, e.target.value);
                }}
              >
                {listed.map((i) => (
                  <option key={i.item_id} value={i.item_id}>
                    {i.title}
                    {i.stratum ? ` [${i.stratum}]` : ""} {i.decided}/{i.queued}
                  </option>
                ))}
              </select>
              <Button
                variant="outline"
                size="icon-sm"
                title="Next item (])"
                aria-label="Next item"
                disabled={at < 0 || at >= listed.length - 1}
                onClick={() => {
                  step(1);
                }}
              >
                <ChevronRightIcon />
              </Button>
            </div>
            <Label className="gap-1.5 text-sm font-normal">
              <Checkbox checked={showAll} onCheckedChange={setShowAll} />
              Show agreed units
            </Label>
            {component === "decision" ? (
              <Label
                className="gap-1.5 text-sm font-normal"
                title="Show the teacher's and the decision model's answers before you decide (unblinds you)"
              >
                <Checkbox checked={showRef} onCheckedChange={setShowRef} />
                Show teacher and model before deciding
              </Label>
            ) : null}
            <span className="text-xs text-gray-500">
              Keys: j/k move · 1/2 take judge 1/2 · a, b, c… pick a choice · n note · s skip · [ ]
              item · Esc leave a field
            </span>
          </div>
          {itemQuery.isError ? (
            <p className="text-sm text-red-700">{itemQuery.error.message}</p>
          ) : (
            <Workspace
              key={`${component}:${itemId ?? ""}`}
              runId={runId}
              component={component}
              item={item}
              loading={!item}
              placeholder={itemQuery.data}
              showAll={showAll}
              showRef={showRef}
              onStep={step}
              onDecided={(next) => {
                client.setQueryData(
                  adjudicationItemOptions(runId, component, next.item_id, showRef).queryKey,
                  next,
                );
                void client.invalidateQueries({
                  queryKey: adjudicationStateOptions(runId).queryKey,
                  exact: true,
                });
                void client.invalidateQueries({
                  queryKey: adjudicationItemOptions(runId, component, next.item_id, !showRef)
                    .queryKey,
                  exact: true,
                });
                void client.invalidateQueries({
                  queryKey: judgeOptions(runId).queryKey,
                  exact: true,
                });
              }}
            />
          )}
        </>
      )}
    </div>
  );
}

function Workspace({
  runId,
  component,
  item,
  loading,
  placeholder,
  showAll,
  showRef,
  onStep,
  onDecided,
}: {
  runId: string;
  component: string;
  item: AdjudicationItem | undefined;
  loading: boolean;
  placeholder: AdjudicationItem | undefined;
  showAll: boolean;
  showRef: boolean;
  onStep: (direction: number) => void;
  onDecided: (item: AdjudicationItem) => void;
}) {
  const shown = item ?? placeholder;
  const [highlight, setHighlight] = useState<Highlight | null>(null);
  const [notes, setNotes] = useState<Record<string, string>>({});
  const cards = useRef(new Map<string, HTMLDivElement>());
  const review = useRef<HTMLDivElement>(null);
  const noteInputs = useRef(new Map<string, HTMLInputElement>());

  const [focus, setFocusState] = useState(0);
  // Scroll to the focused card only once the user moves (not over the item's header on opening it).
  const moved = useRef(false);
  const setFocus = useCallback((value: number) => {
    moved.current = true;
    setFocusState(value);
  }, []);
  const visible = useMemo(
    () => (item?.units ?? []).filter((u) => showAll || u.queued || u.decision),
    [item, showAll],
  );

  // Opening an item: focus its first undecided unit.
  const started = useRef(false);
  useEffect(() => {
    if (!item || started.current) return;
    started.current = true;
    const first = visible.findIndex((u) => !u.decision);
    if (first > 0) setFocus(first);
  }, [item, visible, setFocus]);

  const decide = useMutation({
    mutationFn: (body: DecideBody) =>
      api.post<AdjudicationItem>(
        `/api/benchmarks/judge/${encodeURIComponent(runId)}/adjudication/decide`,
        body,
      ),
    onSuccess: (next, body) => {
      onDecided(next);
      // Move on to the next undecided unit after this one.
      const units = next.units.filter((u) => showAll || u.queued || u.decision);
      const at = units.findIndex((u) => u.unit_id === body.unit_id);
      const after = units.findIndex((u, k) => k > at && !u.decision);
      if (after >= 0) setFocus(after);
    },
  });

  const send = useCallback(
    (u: UnitView, value: string | null, choice: Choice) => {
      if (decide.isPending) return;
      decide.mutate({
        unit_id: u.unit_id,
        choice,
        value,
        note: notes[u.unit_id] ?? u.decision?.note ?? "",
        show_reference: showRef,
      });
    },
    [decide, notes, showRef],
  );

  // Keep the focused card in view.
  const focusedId = visible[Math.min(focus, visible.length - 1)]?.unit_id;
  useEffect(() => {
    if (focusedId && moved.current) {
      const el = cards.current.get(focusedId);
      if (el && review.current) scrollWithin(review.current, el, "nearest");
    }
  }, [focusedId]);

  // The keyboard: the latest values through a ref, one listener.
  const latest = useRef({ visible, focus, send, onStep });
  latest.current = { visible, focus, send, onStep };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        if (e.key === "Escape") target.blur();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const { visible: units, focus: at, send: decideUnit, onStep: stepItem } = latest.current;
      const u = units[Math.min(at, units.length - 1)];
      if (e.key === "j") setFocus(Math.min(at + 1, units.length - 1));
      else if (e.key === "k") setFocus(Math.max(at - 1, 0));
      else if (e.key === "]") stepItem(1);
      else if (e.key === "[") stepItem(-1);
      else if (!u) return;
      else if (e.key === "1" || e.key === "2") {
        const j = u.judges[Number(e.key) - 1];
        if (j && j.compared !== null)
          decideUnit(u, j.compared, e.key === "1" ? "judge1" : "judge2");
      } else if (e.key === "s") decideUnit(u, null, "skip");
      else if (e.key === "n") {
        e.preventDefault();
        noteInputs.current.get(u.unit_id)?.focus();
      } else if (/^[a-h]$/.test(e.key)) {
        const c = u.choices[e.key.charCodeAt(0) - 97];
        if (c) decideUnit(u, c, "other");
      } else return;
      e.preventDefault();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <ResizablePanelGroup
      orientation="horizontal"
      className="rounded-lg border"
      style={{ height: "calc(100svh - 13rem)", minHeight: "28rem" }}
    >
      <ResizablePanel id="source" defaultSize="50%" minSize="25%">
        {shown ? (
          <div className={loading ? "h-full opacity-60" : "h-full"}>
            <SourcePane source={shown.view.source} highlight={item ? highlight : null} />
          </div>
        ) : null}
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel id="review" defaultSize="50%" minSize="30%">
        <div ref={review} className="scrollbar-visible h-full overflow-auto bg-gray-50 p-3">
          {!item ? (
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Spinner /> Loading the item…
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              <ItemHeader item={item} />
              {decide.isError ? (
                <p className="text-sm text-red-700">{decide.error.message}</p>
              ) : null}
              {visible.length ? (
                visible.map((u, i) => (
                  <UnitCard
                    key={u.unit_id}
                    unit={u}
                    focused={u.unit_id === focusedId}
                    busy={decide.isPending}
                    note={notes[u.unit_id] ?? u.decision?.note ?? ""}
                    onNote={(value) => {
                      setNotes((n) => ({ ...n, [u.unit_id]: value }));
                    }}
                    noteRef={(el) => {
                      if (el) noteInputs.current.set(u.unit_id, el);
                      else noteInputs.current.delete(u.unit_id);
                    }}
                    cardRef={(el) => {
                      if (el) cards.current.set(u.unit_id, el);
                      else cards.current.delete(u.unit_id);
                    }}
                    onFocus={() => {
                      setFocus(i);
                    }}
                    onDecide={(value, choice) => {
                      setFocus(i);
                      send(u, value, choice);
                    }}
                    onQuote={(h) => {
                      setHighlight({ ...h, nonce: Date.now() });
                    }}
                  />
                ))
              ) : (
                <p className="rounded-lg border bg-white p-3 text-sm text-gray-600">
                  Nothing is queued in this item (turn on "Show agreed units" to review them
                  anyway).
                </p>
              )}
              {component === "decision" ? null : (
                <p className="text-xs text-gray-500">
                  Click a quoted passage to find it in the source.
                </p>
              )}
            </div>
          )}
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
