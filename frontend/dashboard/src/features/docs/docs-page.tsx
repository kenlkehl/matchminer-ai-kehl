import { useQuery } from "@tanstack/react-query";
import { getRouteApi, useNavigate } from "@tanstack/react-router";

import { Markdown } from "@/components/markdown";
import { PageHeader } from "@/components/page-header";
import { Spinner } from "@/components/ui/spinner";
import { api } from "@/lib/api";

interface DocIndex {
  pages: { id: string; title: string; generated: boolean }[];
}

interface DocPage {
  id: string;
  title: string;
  markdown: string;
}

const route = getRouteApi("/_app/how-it-works");

/**
 * How this works: how the models are trained and fit together, and the structured extraction data
 * model. The server serves a curated set of the documentation's pages (and generates the data model
 * from the schema); links between them stay in the tab.
 */
export function DocsPage() {
  const search = route.useSearch();
  const navigate = useNavigate();
  const index = useQuery({
    queryKey: ["docs"],
    queryFn: () => api.get<DocIndex>("/api/docs"),
    staleTime: Infinity,
  });
  const pages = index.data?.pages ?? [];
  const current = search.page ?? pages[0]?.id;
  const page = useQuery({
    queryKey: ["docs", current],
    queryFn: () => api.get<DocPage>(`/api/docs/${encodeURIComponent(current ?? "")}`),
    enabled: !!current,
    staleTime: Infinity,
  });
  const open = (id: string) => {
    void navigate({ to: "/how-it-works", search: { page: id } });
    window.scrollTo({ top: 0 });
  };
  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="How this works">
        <p>
          How the models are trained and how they work together when you search, and the data model
          of the structured records extracted from notes.
        </p>
      </PageHeader>
      <div className="flex flex-col gap-6 md:flex-row">
        <nav
          aria-label="Pages"
          className="shrink-0 md:sticky md:top-16 md:max-h-[calc(100svh-5rem)] md:w-64 md:self-start md:overflow-y-auto"
        >
          <ul className="flex flex-col gap-1 text-sm">
            {pages.map((p) => (
              <li key={p.id}>
                <button
                  type="button"
                  aria-current={p.id === current ? "page" : undefined}
                  className="w-full rounded-md px-2 py-1 text-left text-gray-700 hover:bg-gray-100 aria-[current=page]:bg-brand-light-blue/40 aria-[current=page]:font-semibold aria-[current=page]:text-gray-950"
                  onClick={() => {
                    open(p.id);
                  }}
                >
                  {p.title}
                </button>
              </li>
            ))}
          </ul>
        </nav>
        <article className="max-w-4xl min-w-0 flex-1">
          {index.isError ? <p className="text-sm text-red-700">{index.error.message}</p> : null}
          {page.isError ? <p className="text-sm text-red-700">{page.error.message}</p> : null}
          {page.data ? (
            <Markdown
              text={page.data.markdown}
              onLink={(href) =>
                href.startsWith("page:")
                  ? () => {
                      open(href.slice(5));
                    }
                  : null
              }
            />
          ) : page.isPending && current ? (
            <Spinner />
          ) : null}
        </article>
      </div>
    </div>
  );
}
