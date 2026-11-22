import { copyFileSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";

import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

// The build goes into the Python package (src/matchminer_ai/apps/dashboard/static), which FastAPI serves at "/".
// `pnpm dev` serves the frontend on :5173 and proxies /api to a running `mmai-dashboard`.
const API = process.env.MMAI_DASH_API_URL ?? "http://127.0.0.1:8781";
const OUT = path.resolve(import.meta.dirname, "../../src/matchminer_ai/apps/dashboard/static");

/** Copies the How this works tab's pages (docs_pages.json) into the build, so an installed package serves them. */
function docsPages(): Plugin {
  const root = path.resolve(import.meta.dirname, "../..");
  return {
    name: "mmai-docs-pages",
    apply: "build",
    closeBundle() {
      const manifest = path.join(root, "src/matchminer_ai/apps/dashboard/docs_pages.json");
      const { pages } = JSON.parse(readFileSync(manifest, "utf8")) as {
        pages: { id: string; source: string | null }[];
      };
      mkdirSync(path.join(OUT, "docs"), { recursive: true });
      for (const page of pages) {
        if (page.source) {
          copyFileSync(
            path.join(root, "docs", page.source),
            path.join(OUT, "docs", `${page.id}.md`),
          );
        }
      }
    },
  };
}

export default defineConfig({
  base: "/",
  plugins: [
    tanstackRouter({ target: "react", autoCodeSplitting: true }),
    react(),
    tailwindcss(),
    docsPages(),
  ],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  build: { outDir: OUT, emptyOutDir: true },
  server: { host: "127.0.0.1", proxy: { "/api": { target: API } } },
});
