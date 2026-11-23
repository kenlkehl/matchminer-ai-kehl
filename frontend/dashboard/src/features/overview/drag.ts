import { type DragEvent, useEffect, useState } from "react";

import { UNIT_MIME } from "@/lib/management";

/** A cohort being dragged onto a group (the sidebar's cohort cards and the group rows). */
export function dragUnit(event: DragEvent, unitId: string) {
  event.dataTransfer.setData(UNIT_MIME, unitId);
  event.dataTransfer.setData("text/plain", unitId);
  event.dataTransfer.effectAllowed = "move";
}

/** Drop handlers for a group: highlighted while a cohort is dragged over it. */
export function useDropTarget(onDrop: ((unitId: string) => void) | undefined) {
  const [over, setOver] = useState(false);
  const accepts = (event: DragEvent) => !!onDrop && event.dataTransfer.types.includes(UNIT_MIME);
  return {
    over,
    props: {
      onDragOver: (event: DragEvent) => {
        if (!accepts(event)) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        setOver(true);
      },
      onDragLeave: () => {
        setOver(false);
      },
      onDrop: (event: DragEvent) => {
        setOver(false);
        if (!accepts(event)) return;
        event.preventDefault();
        const unitId = event.dataTransfer.getData(UNIT_MIME);
        if (unitId) onDrop?.(unitId);
      },
    },
  };
}

/** Whether a cohort is being dragged anywhere on the page (the groups then show where it can be dropped). */
export function useUnitDragging() {
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    const start = (event: globalThis.DragEvent) => {
      if (event.dataTransfer?.types.includes(UNIT_MIME)) setDragging(true);
    };
    const end = () => {
      setDragging(false);
    };
    document.addEventListener("dragstart", start);
    document.addEventListener("dragend", end);
    document.addEventListener("drop", end);
    return () => {
      document.removeEventListener("dragstart", start);
      document.removeEventListener("dragend", end);
      document.removeEventListener("drop", end);
    };
  }, []);
  return dragging;
}
