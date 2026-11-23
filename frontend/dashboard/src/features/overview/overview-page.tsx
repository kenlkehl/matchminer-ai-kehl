import { useQuery } from "@tanstack/react-query";
import { getRouteApi, Link } from "@tanstack/react-router";
import { cn } from "cn";
import {
  ChevronsDownUpIcon,
  ChevronsUpDownIcon,
  Columns3Icon,
  PanelRightOpenIcon,
  PlusIcon,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useDefaultLayout } from "react-resizable-panels";

import { PageHeader } from "@/components/page-header";
import { SearchInput } from "@/components/search-input";
import { Accordion } from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Field, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { Spinner } from "@/components/ui/spinner";
import type { CenterInfo, CenterView, CohortRow, TrialSummary } from "@/lib/api";
import { when } from "@/lib/format";
import { management, useManagementChange } from "@/lib/management";
import { centerOptions, metaOptions } from "@/lib/queries";

import { CatalogEvidenceDialog } from "./catalog-evidence-dialog";
import { CohortDialog } from "./cohort-dialog";
import { useUnitDragging } from "./drag";
import { closedTrials, FilterPanel } from "./filter-panel";
import {
  COLUMNS,
  filtersOf,
  hasFilters,
  keeps,
  type OverviewSearch,
  rowText,
  terms,
  visibleColumns,
} from "./search";
import { PendingSection, TrialGroupSection } from "./trial-group-section";
import { TrialDetailsDialog } from "./trial-details-dialog";
import { TrialManagementSidebar } from "./trial-management-sidebar";

const route = getRouteApi("/_app/");

export function OverviewPage() {
  const search = route.useSearch();
  const meta = useQuery(metaOptions());
  if (!meta.data) {
    return meta.isError ? (
      <p className="text-sm text-red-700">Could not load the dashboard: {meta.error.message}</p>
    ) : (
      <Loading />
    );
  }
  const center = meta.data.centers.find((c) => c.id === search.center);
  return center ? (
    <CenterOverview
      center={center}
      builtAt={meta.data.built_at}
      managementRevision={meta.data.management_revision}
    />
  ) : (
    <Landing
      centers={meta.data.centers}
      builtAt={meta.data.built_at}
      trials={meta.data.counts.trials}
    />
  );
}

function Loading() {
  return (
    <div className="flex justify-center py-16">
      <Spinner className="size-6" />
    </div>
  );
}

/** No center chosen: every disease center's dashboard, largest first. */
function Landing({
  centers,
  builtAt,
  trials,
}: {
  centers: CenterInfo[];
  builtAt: number;
  trials: number;
}) {
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Trials Overview">
        <p>
          {trials} trials across {centers.length} disease centers, from their protocols (or their
          ClinicalTrials.gov listings) as extracted so far; updated {when(builtAt)}. Choose a
          disease center to see its trials and cohorts.
        </p>
      </PageHeader>
      <ul className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-3">
        {centers.map((c) => (
          <li key={c.id}>
            <Link
              to="/"
              search={{ center: c.id }}
              className="flex h-full flex-col gap-1 rounded-lg border border-gray-200 bg-white p-4 hover:border-brand-blue hover:bg-brand-blue/5"
            >
              <span className="font-semibold text-gray-900">{c.short_name}</span>
              <span className="text-xs text-gray-500">{c.name}</span>
              <span className="mt-auto pt-2 text-sm text-gray-700">
                {c.trials} trials · {c.cohorts} cohorts
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

export type Overview = ReturnType<typeof useOverview>;

function useOverview(view: CenterView | undefined) {
  const search = route.useSearch();
  const navigate = route.useNavigate();
  const filters = filtersOf(search);
  const [clears, setClears] = useState(0);
  const go = (update: (prev: OverviewSearch) => OverviewSearch) => {
    void navigate({ search: update, resetScroll: false });
  };
  const rows = useMemo(() => (view ? view.groups.flatMap((g) => g.rows) : []), [view]);
  const texts = useMemo(() => new Map(rows.map((r) => [r.cohort_id, rowText(r)])), [rows]);
  const words = terms(filters.q);
  const shown = (row: CohortRow, except?: keyof typeof filters) =>
    keeps(row, filters, words, texts.get(row.cohort_id) ?? "", except);
  return {
    search,
    filters,
    rows,
    shown,
    clears,
    columns: visibleColumns(search),
    setFilters: (patch: Partial<OverviewSearch>) => {
      go((prev) => ({ ...prev, ...patch }));
    },
    clearFilters: () => {
      setClears((n) => n + 1);
      go((prev) => ({ center: prev.center, cols: prev.cols, manageTrial: prev.manageTrial }));
    },
    setColumn: (key: keyof ReturnType<typeof visibleColumns>, visible: boolean) => {
      go((prev) => ({ ...prev, cols: { ...prev.cols, [key]: visible } }));
    },
    open: (patch: Pick<OverviewSearch, "details" | "cohort">) => {
      go((prev) => ({
        ...prev,
        details: undefined,
        cohort: undefined,
        evidence: undefined,
        ...patch,
      }));
    },
    /** A drug's or class's catalog evidence (`drug:<id>`, `class:<id>`), over trial details if open. */
    showEvidence: (evidence: string | undefined) => {
      go((prev) => ({ ...prev, evidence }));
    },
    /** The trial opened in the Trial management sidebar (none: the list only). */
    manageTrial: (manageTrial: string | undefined) => {
      go((prev) => ({ ...prev, manageTrial }));
    },
  };
}

function CenterOverview({
  center,
  builtAt,
  managementRevision,
}: {
  center: CenterInfo;
  builtAt: number;
  managementRevision: number | undefined;
}) {
  const query = useQuery(centerOptions(center.id, builtAt, managementRevision));
  const view = query.data?.id === center.id ? query.data : undefined;
  const overview = useOverview(view);
  const [openGroups, setOpenGroups] = useState<string[] | null>(null);
  const { filters, search } = overview;
  const filtering = hasFilters(filters);

  // A group the center made stays in view while empty (a place to drop cohorts), unless filtering.
  const groups = (view?.groups ?? [])
    .map((g) => ({ ...g, rows: g.rows.filter((r) => overview.shown(r)) }))
    .filter((g) => g.rows.length > 0 || (g.custom && !filtering));
  const pin = useManagementChange(({ unitId, groupId }: { unitId: string; groupId: string }) =>
    management.pin(unitId, groupId),
  );
  const renameGroup = useManagementChange(({ id, name }: { id: string; name: string }) =>
    management.renameGroup(center.id, id, name),
  );
  const deleteGroup = useManagementChange((id: string) => management.deleteGroup(center.id, id));
  const [sidebar, setSidebar] = useStoredFlag(SIDEBAR_KEY);
  // The sidebar is open while the user keeps it open, or a link names a trial to manage.
  const managing = sidebar || Boolean(search.manageTrial);
  const layout = useDefaultLayout({
    id: "mmai-dash-trial-management",
    panelIds: managing ? ["groups", "trial-management"] : ["groups"],
    storage: localStorage,
  });
  const dragging = useUnitDragging();
  // The sidebar lists the trials the page's search and filters show.
  const listed = useMemo(() => {
    const seen = new Map<string, TrialSummary>();
    for (const g of groups) for (const r of g.rows) seen.set(r.trial.id, r.trial);
    return [...seen.values()].sort((a, b) =>
      (a.protocol_no || a.nct_id).localeCompare(b.protocol_no || b.nct_id),
    );
  }, [groups]);
  const closedHidden = closedTrials(overview.rows);
  const pending = filtering ? [] : (view?.pending ?? []);
  // While filtering, the groups with matches open; otherwise as the user left them (all closed at first).
  const open = openGroups ?? (filtering ? groups.map((g) => g.id) : []);
  const allIds = [...groups.map((g) => g.id), ...(pending.length ? ["pending"] : [])];
  const trialCount = new Set(groups.flatMap((g) => g.rows.map((r) => r.trial.id))).size;
  // The catalog evidence dialog: nested in trial details when those are open, so it stacks over them.
  const evidence = search.evidence ? (
    <CatalogEvidenceDialog
      target={search.evidence}
      onClose={() => {
        overview.showEvidence(undefined);
      }}
      onOpen={overview.showEvidence}
      onTrial={(details) => {
        overview.open({ details });
      }}
    />
  ) : null;
  const cohortCount = groups.reduce((n, g) => n + g.rows.length, 0);

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title={`${center.short_name} Trials Overview`}>
        <p>
          {center.name}: {center.trials} trials, with each cohort's eligibility criteria as
          extracted from its protocol (or, before its protocol is processed, its ClinicalTrials.gov
          listing). Updated {when(builtAt)}; new extractions appear here as they finish.
        </p>
      </PageHeader>
      {view ? <FilterPanel overview={overview} view={view} /> : null}
      <div className="flex items-end gap-2 print:hidden">
        <Field className="flex-1">
          <FieldLabel htmlFor="overview-search" className="text-sm font-normal text-gray-600">
            Search by Protocol, PI, Title, Cohort, Criteria, Drug
          </FieldLabel>
          <SearchInput
            key={`${center.id}-${String(overview.clears)}`}
            id="overview-search"
            placeholder="Type to search..."
            value={filters.q}
            onChange={(q) => {
              setOpenGroups(null);
              overview.setFilters({ q: q || undefined });
            }}
          />
        </Field>
        <Button
          variant="outline"
          className="font-normal"
          onClick={() => {
            setOpenGroups(open.length < allIds.length ? allIds : []);
          }}
        >
          {open.length < allIds.length ? <ChevronsUpDownIcon /> : <ChevronsDownUpIcon />}
          {open.length < allIds.length ? "Expand all" : "Collapse all"}
        </Button>
        <Button
          variant={managing ? "secondary" : "outline"}
          className="font-normal"
          aria-pressed={managing}
          onClick={() => {
            setSidebar(!managing);
            if (managing) overview.manageTrial(undefined);
          }}
        >
          <PanelRightOpenIcon />
          Trial management
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="outline" className="font-normal" />}>
            <Columns3Icon />
            Columns
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuGroup>
              <DropdownMenuLabel>Show / hide columns</DropdownMenuLabel>
              {COLUMNS.map((column) => (
                <DropdownMenuCheckboxItem
                  key={column.key}
                  checked={overview.columns[column.key]}
                  onCheckedChange={(checked) => {
                    overview.setColumn(column.key, checked);
                  }}
                >
                  {column.label}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      {!view ? (
        query.isError ? (
          <p className="text-sm text-red-700">
            Could not load {center.short_name}: {query.error.message}
          </p>
        ) : (
          <Loading />
        )
      ) : (
        <ResizablePanelGroup
          orientation="horizontal"
          id="mmai-dash-trial-management"
          defaultLayout={layout.defaultLayout}
          onLayoutChanged={layout.onLayoutChanged}
          className="-mr-6"
          style={{ width: "calc(100% + 1.5rem)", height: "auto", overflow: "visible" }}
        >
          <ResizablePanel id="groups" minSize="35%" style={{ overflow: "visible" }}>
            <div className={cn("flex flex-col gap-3", managing ? "pr-3" : "pr-6")}>
              {pin.isError || renameGroup.isError || deleteGroup.isError ? (
                <p className="text-sm text-red-700">
                  {(pin.error ?? renameGroup.error ?? deleteGroup.error)?.message}
                </p>
              ) : null}
              {managing ? <NewGroup centerId={center.id} /> : null}
              <Accordion
                multiple
                value={open}
                onValueChange={(value) => {
                  setOpenGroups(value as string[]);
                }}
                className="overflow-hidden rounded-md border border-gray-300"
              >
                {groups.map((group) => (
                  <TrialGroupSection
                    key={group.id}
                    group={group}
                    columns={overview.columns}
                    labels={view.cancer_type_labels}
                    onDetails={(details) => {
                      overview.open({ details });
                    }}
                    onCriteria={(cohort) => {
                      overview.open({ cohort });
                    }}
                    onEvidence={overview.showEvidence}
                    dragRows={managing}
                    dragging={dragging}
                    onDropUnit={(unitId) => {
                      pin.mutate({ unitId, groupId: group.id });
                    }}
                    onRename={(name) => {
                      renameGroup.mutate({ id: group.id, name });
                    }}
                    onDelete={() => {
                      if (
                        window.confirm(
                          `Delete the group ${group.name}? Its cohorts go back to their groups.`,
                        )
                      )
                        deleteGroup.mutate(group.id);
                    }}
                  />
                ))}
                {pending.length ? (
                  <PendingSection
                    trials={pending}
                    onDetails={(details) => {
                      overview.open({ details });
                    }}
                  />
                ) : null}
              </Accordion>
              <p className="text-sm text-gray-500">
                {cohortCount > 0
                  ? `Showing ${String(trialCount)} trials, ${String(cohortCount)} cohorts across ${String(groups.length)} groups`
                  : "No trials found."}
                {pending.length
                  ? ` · ${String(pending.length)} trials without extracted cohorts`
                  : ""}
                {!filters.closed && closedHidden
                  ? ` · ${String(closedHidden)} closed ${closedHidden === 1 ? "trial" : "trials"} hidden`
                  : ""}
              </p>
            </div>
          </ResizablePanel>
          {managing ? (
            <>
              <ResizableHandle withHandle className="hover:bg-gray-400" />
              <ResizablePanel
                id="trial-management"
                defaultSize="34%"
                minSize="22%"
                maxSize="60%"
                className="bg-slate-50"
                style={{ overflow: "visible" }}
              >
                <TrialManagementSidebar
                  centerId={center.id}
                  trials={listed}
                  managementRevision={managementRevision}
                  trialId={search.manageTrial}
                  onTrial={overview.manageTrial}
                  onClose={() => {
                    setSidebar(false);
                    overview.manageTrial(undefined);
                  }}
                />
              </ResizablePanel>
            </>
          ) : null}
        </ResizablePanelGroup>
      )}
      {search.details ? (
        <TrialDetailsDialog
          trialId={search.details}
          onClose={() => {
            overview.open({});
          }}
          onCriteria={(cohort) => {
            overview.open({ cohort });
          }}
          onEvidence={overview.showEvidence}
        >
          {evidence}
        </TrialDetailsDialog>
      ) : (
        evidence
      )}
      {search.cohort ? (
        <CohortDialog
          cohortId={search.cohort}
          onClose={() => {
            overview.open({});
          }}
          onDetails={(details) => {
            overview.open({ details });
          }}
        />
      ) : null}
    </div>
  );
}

const SIDEBAR_KEY = "mmai-dash-trial-management-open";

/** A flag kept in the browser between visits (whether the Trial management sidebar is open). */
function useStoredFlag(key: string): [boolean, (value: boolean) => void] {
  const [value, setValue] = useState(() => localStorage.getItem(key) === "1");
  return [
    value,
    (next: boolean) => {
      localStorage.setItem(key, next ? "1" : "0");
      setValue(next);
    },
  ];
}

/** A group of the center's own, made here so cohorts can be dropped onto it. */
function NewGroup({ centerId }: { centerId: string }) {
  const [name, setName] = useState<string | null>(null);
  const create = useManagementChange((n: string) => management.createGroup(centerId, n));
  if (name === null) {
    return (
      <Button
        size="sm"
        variant="outline"
        className="self-start font-normal"
        onClick={() => {
          setName("");
        }}
      >
        <PlusIcon />
        New group
      </Button>
    );
  }
  return (
    <form
      className="flex flex-wrap items-center gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        create.mutate(name, {
          onSuccess: () => {
            setName(null);
          },
        });
      }}
    >
      <Input
        aria-label="New group name"
        placeholder="Name of the new group"
        className="h-8 w-72"
        value={name}
        autoFocus
        onChange={(e) => {
          setName(e.target.value);
        }}
      />
      <Button size="sm" type="submit" disabled={!name.trim() || create.isPending}>
        Add the group
      </Button>
      <Button
        size="sm"
        variant="ghost"
        type="button"
        onClick={() => {
          setName(null);
        }}
      >
        Cancel
      </Button>
      <span className="text-xs text-gray-500">
        It is listed first, and cohorts dragged onto it are shown in it.
      </span>
      {create.isError ? <span className="text-xs text-red-700">{create.error.message}</span> : null}
    </form>
  );
}
