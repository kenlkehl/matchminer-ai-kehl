import { createFileRoute, redirect } from "@tanstack/react-router";

/** The former Trials-for-a-patient address: the Patient tab. */
export const Route = createFileRoute("/_app/ai")({
  beforeLoad: () => {
    throw redirect({ to: "/patient" });
  },
});
