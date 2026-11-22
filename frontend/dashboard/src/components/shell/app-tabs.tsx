import { useQuery } from "@tanstack/react-query";
import { Link, useRouterState } from "@tanstack/react-router";
import { useEffect, useRef } from "react";

import type { AppTabName } from "@/lib/api";
import { metaOptions } from "@/lib/queries";

export type AppTab = AppTabName;

const PREFIXES: [string, AppTab][] = [
  ["/population", "population"],
  ["/patient", "patient"],
  ["/assistant", "assistant"],
  ["/benchmarking", "benchmarking"],
  ["/how-it-works", "docs"],
  ["/status", "status"],
];

/** Which tab the current page belongs to (Trials: the overview and patients for a cohort). */
export function useAppTab(): AppTab {
  return useRouterState({
    select: (s) =>
      PREFIXES.find(([prefix]) => s.location.pathname.startsWith(prefix))?.[1] ?? "trials",
  });
}

export type TabPath =
  "/" | "/population" | "/patient" | "/assistant" | "/benchmarking" | "/how-it-works" | "/status";

const TABS: {
  tab: AppTab;
  to: TabPath;
  label: string;
}[] = [
  { tab: "trials", to: "/", label: "Clinical trial management" },
  { tab: "patient", to: "/patient", label: "Patient-centric search" },
  { tab: "population", to: "/population", label: "Guideline concordance reporting" },
  { tab: "assistant", to: "/assistant", label: "Assistant" },
  { tab: "benchmarking", to: "/benchmarking", label: "Benchmarking" },
  { tab: "docs", to: "/how-it-works", label: "How this works" },
  { tab: "status", to: "/status", label: "Data and models" },
];

/**
 * The tab bar's height in px, as it is now (labels wrap on narrow windows). The bar publishes it as the CSS variable
 * --app-tabs-h, which the panes that stay in view sit below.
 */
export function appTabsHeight(): number {
  return (
    parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--app-tabs-h")) || 48
  );
}

/** The tabs above every page: those the server's mode shows. They stay in view while the page scrolls. */
export function AppTabs() {
  const current = useAppTab();
  const bar = useRef<HTMLElement>(null);
  useEffect(() => {
    const element = bar.current;
    if (!element) return;
    const publish = () => {
      document.documentElement.style.setProperty(
        "--app-tabs-h",
        `${String(element.offsetHeight)}px`,
      );
    };
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, []);
  // The server's tabs for its mode; Data and models is always last.
  const shown = [
    ...(useQuery(metaOptions()).data?.tabs ?? ["trials", "population", "patient"]),
    "status",
  ];
  return (
    <nav
      ref={bar}
      aria-label="Applications"
      className="sticky top-0 z-40 -mx-6 -mt-6 mb-6 flex border-b border-gray-200 bg-gray-100 px-6 pt-2 print:static print:hidden"
    >
      {TABS.filter(({ tab }) => shown.includes(tab)).map(({ tab, to, label }) => (
        <Link
          key={tab}
          to={to}
          aria-current={tab === current ? "page" : undefined}
          className="-mb-px flex items-center gap-1.5 rounded-t-md border border-transparent px-4 py-2 text-sm font-semibold text-gray-500 hover:text-gray-900 aria-[current=page]:border-t-2 aria-[current=page]:border-gray-200 aria-[current=page]:border-t-brand-blue aria-[current=page]:border-b-white aria-[current=page]:bg-white aria-[current=page]:text-gray-900"
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
