import { cn } from "cn";
import type { ReactNode } from "react";
import type { Layout } from "react-resizable-panels";

import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";

/** The split the user last dragged to, shared by every summary display for the page's lifetime. */
let remembered: Layout | undefined;

/**
 * A patient summary's two sections side by side, the cancer history and the other relevant history, with a
 * splitter between them so either can be given more room. Each section fills its pane's height (a pane of fixed
 * height scrolls its section's text inside it).
 */
export function SummarySplit({
  history,
  other,
  className,
}: {
  history: ReactNode;
  other: ReactNode;
  className?: string;
}) {
  return (
    <ResizablePanelGroup
      orientation="horizontal"
      // Auto height: the group stretches to a fixed-height parent (a percentage height did not resolve there and
      // took the texts' full height), and is as tall as its content elsewhere.
      className={cn("h-auto", className)}
      defaultLayout={remembered}
      onLayoutChanged={(layout) => {
        remembered = layout;
      }}
    >
      <ResizablePanel
        id="history"
        defaultSize="75%"
        minSize="20%"
        className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)] pr-3"
      >
        {history}
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel
        id="other"
        defaultSize="25%"
        minSize="10%"
        className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)] pl-3"
      >
        {other}
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
