import { type Dispatch, type SetStateAction, useEffect, useState } from "react";

/** The Population tab's form inputs while other tabs show (in memory only; a reload starts over). */
const memory = new Map<string, unknown>();

/** `useState` whose value outlives the page: leaving the tab and coming back keeps what was entered. */
export function useRemembered<T>(key: string, initial: T): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => (memory.has(key) ? (memory.get(key) as T) : initial));
  useEffect(() => {
    memory.set(key, value);
  }, [key, value]);
  return [value, setValue];
}
