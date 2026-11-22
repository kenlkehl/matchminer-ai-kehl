import { Fragment, type ReactNode } from "react";

/**
 * A small, safe Markdown renderer (React elements only, never HTML): headings, paragraphs, nested lists,
 * tables, fenced code, quotes, rules, MkDocs admonitions, and inline code, bold, italics and links.
 * Links go out only to http(s) addresses (a new tab); other targets are offered to `onLink` (the
 * documentation's own pages), else shown as text.
 */
export interface MarkdownProps {
  text: string;
  /** Renders a run of plain text, e.g. turning known IDs into links (default: the text). */
  renderText?: (text: string, key: string) => ReactNode;
  /** A relative link's action (e.g. open another documentation page), or null to show it as text. */
  onLink?: (href: string) => (() => void) | null;
  className?: string;
}

type Block =
  | { kind: "heading"; level: number; text: string }
  | { kind: "paragraph"; text: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "code"; text: string }
  | { kind: "table"; header: string[]; rows: string[][] }
  | { kind: "quote"; text: string }
  | { kind: "admonition"; type: string; title: string; body: string }
  | { kind: "rule" };

const LIST = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const indentOf = (line: string) => line.length - line.trimStart().length;

function cells(line: string): string[] {
  const inner = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  return inner.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
}

function dedent(lines: string[]): string {
  const widths = lines.filter((l) => l.trim()).map(indentOf);
  const cut = widths.length ? Math.min(...widths) : 0;
  return lines.map((l) => l.slice(Math.min(cut, indentOf(l)))).join("\n");
}

export function parseBlocks(text: string): Block[] {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i] ?? "";
    const trimmed = line.trim();
    if (!trimmed) {
      i += 1;
      continue;
    }
    if (trimmed.startsWith("```")) {
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !(lines[i] ?? "").trim().startsWith("```"))
        body.push(lines[i++] ?? "");
      i += 1;
      blocks.push({ kind: "code", text: dedent(body) });
      continue;
    }
    const heading = /^(#{1,6})\s+(.*?)\s*#*$/.exec(trimmed);
    if (heading) {
      blocks.push({ kind: "heading", level: heading[1]?.length ?? 1, text: heading[2] ?? "" });
      i += 1;
      continue;
    }
    if (/^([-*_])(\s*\1){2,}$/.test(trimmed)) {
      blocks.push({ kind: "rule" });
      i += 1;
      continue;
    }
    const admonition = /^(!!!|\?\?\?\+?)\s+(\w+)(?:\s+"([^"]*)")?/.exec(trimmed);
    if (admonition) {
      const body: string[] = [];
      i += 1;
      while (
        i < lines.length &&
        (!(lines[i] ?? "").trim() || indentOf(lines[i] ?? "") > indentOf(line))
      ) {
        body.push(lines[i++] ?? "");
      }
      const type = admonition[2] ?? "note";
      blocks.push({
        kind: "admonition",
        type,
        title: admonition[3] ?? type.charAt(0).toUpperCase() + type.slice(1),
        body: dedent(body),
      });
      continue;
    }
    if (trimmed.startsWith("|") && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1] ?? "")) {
      const header = cells(trimmed);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && (lines[i] ?? "").trim().startsWith("|"))
        rows.push(cells(lines[i++] ?? ""));
      blocks.push({ kind: "table", header, rows });
      continue;
    }
    if (trimmed.startsWith(">")) {
      const body: string[] = [];
      while (i < lines.length && (lines[i] ?? "").trim().startsWith(">")) {
        body.push((lines[i++] ?? "").trim().replace(/^>\s?/, ""));
      }
      blocks.push({ kind: "quote", text: body.join("\n") });
      continue;
    }
    const item = LIST.exec(line);
    if (item) {
      const base = indentOf(line);
      const ordered = /\d/.test(item[2] ?? "");
      const items: string[] = [];
      while (i < lines.length) {
        const current = LIST.exec(lines[i] ?? "");
        if (!current || indentOf(lines[i] ?? "") !== base) break;
        const body = [current[3] ?? ""];
        i += 1;
        // The item's continuation and nested lines: indented deeper, or blank lines followed by them.
        while (i < lines.length) {
          const next = lines[i] ?? "";
          if (next.trim() && indentOf(next) > base) {
            body.push(next);
            i += 1;
          } else if (
            !next.trim() &&
            indentOf(lines[i + 1] ?? "") > base &&
            (lines[i + 1] ?? "").trim()
          ) {
            body.push("");
            i += 1;
          } else break;
        }
        items.push(body[0] + (body.length > 1 ? "\n" + dedent(body.slice(1)) : ""));
      }
      blocks.push({ kind: "list", ordered, items });
      continue;
    }
    const paragraph: string[] = [];
    while (i < lines.length) {
      const next = lines[i] ?? "";
      const t = next.trim();
      if (
        !t ||
        t.startsWith("```") ||
        /^#{1,6}\s/.test(t) ||
        (t.startsWith("|") && paragraph.length === 0) ||
        LIST.test(next) ||
        t.startsWith(">") ||
        /^!!!\s/.test(t)
      ) {
        break;
      }
      paragraph.push(t);
      i += 1;
    }
    if (paragraph.length === 0) {
      paragraph.push(trimmed);
      i += 1;
    }
    blocks.push({ kind: "paragraph", text: paragraph.join(" ") });
  }
  return blocks;
}

const INLINE =
  /(`+)([\s\S]*?[^`])\1(?!`)|\*\*([\s\S]+?)\*\*|\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)|(?<![\w*])\*(?![\s*])([^*]+?)\*(?![\w*])|(?<!\w)_(?![\s_])([^_]+?)_(?!\w)|<?(https?:\/\/[^\s<>)]+[^\s<>).,;:])>?/g;

function Inline({ text, props, prefix }: { text: string; props: MarkdownProps; prefix: string }) {
  const out: ReactNode[] = [];
  const plain = (value: string, key: string) =>
    props.renderText ? props.renderText(value, key) : value;
  let last = 0;
  let n = 0;
  for (const match of text.matchAll(INLINE)) {
    const start = match.index;
    if (start > last)
      out.push(
        <Fragment key={`${prefix}t${String(n)}`}>
          {plain(text.slice(last, start), `${prefix}t${String(n)}`)}
        </Fragment>,
      );
    const key = `${prefix}m${String(n++)}`;
    if (match[2] !== undefined) {
      out.push(
        <code key={key} className="rounded bg-gray-100 px-1 py-0.5 font-mono text-[0.85em]">
          {match[2].trim()}
        </code>,
      );
    } else if (match[3] !== undefined) {
      out.push(
        <strong key={key}>
          <Inline text={match[3]} props={props} prefix={`${key}.`} />
        </strong>,
      );
    } else if (match[4] !== undefined && match[5] !== undefined) {
      out.push(<LinkTo key={key} href={match[5]} label={match[4]} props={props} prefix={key} />);
    } else if (match[6] !== undefined || match[7] !== undefined) {
      out.push(
        <em key={key}>
          <Inline text={match[6] ?? match[7] ?? ""} props={props} prefix={`${key}.`} />
        </em>,
      );
    } else if (match[8] !== undefined) {
      out.push(<LinkTo key={key} href={match[8]} label={match[8]} props={props} prefix={key} />);
    }
    last = start + match[0].length;
  }
  if (last < text.length)
    out.push(<Fragment key={`${prefix}end`}>{plain(text.slice(last), `${prefix}end`)}</Fragment>);
  return <>{out}</>;
}

function LinkTo({
  href,
  label,
  props,
  prefix,
}: {
  href: string;
  label: string;
  props: MarkdownProps;
  prefix: string;
}) {
  const content = (
    <Inline text={label} props={{ ...props, renderText: undefined }} prefix={`${prefix}.`} />
  );
  if (/^https?:\/\//i.test(href)) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-brand-blue underline"
      >
        {content}
      </a>
    );
  }
  const action = props.onLink?.(href) ?? null;
  if (action) {
    return (
      <button type="button" className="text-brand-blue underline" onClick={action}>
        {content}
      </button>
    );
  }
  return <span>{content}</span>;
}

const ADMONITION_STYLES: Record<string, string> = {
  warning: "border-amber-300 bg-amber-50",
  caution: "border-amber-300 bg-amber-50",
  danger: "border-red-300 bg-red-50",
  tip: "border-green-300 bg-green-50",
};

function Blocks({
  blocks,
  props,
  prefix,
}: {
  blocks: Block[];
  props: MarkdownProps;
  prefix: string;
}) {
  return (
    <>
      {blocks.map((block, index) => {
        const key = `${prefix}${String(index)}`;
        switch (block.kind) {
          case "heading": {
            const size = ["text-xl", "text-lg", "text-base", "text-sm", "text-sm", "text-sm"][
              block.level - 1
            ];
            const Tag = `h${String(Math.min(6, block.level + 1))}` as "h2";
            return (
              <Tag key={key} className={`${size ?? "text-sm"} mt-2 font-semibold`}>
                <Inline text={block.text} props={props} prefix={`${key}.`} />
              </Tag>
            );
          }
          case "paragraph":
            return (
              <p key={key}>
                <Inline text={block.text} props={props} prefix={`${key}.`} />
              </p>
            );
          case "list": {
            const Tag = block.ordered ? "ol" : "ul";
            return (
              <Tag
                key={key}
                className={`${block.ordered ? "list-decimal" : "list-disc"} space-y-1 pl-6`}
              >
                {block.items.map((item, n) => {
                  const [first, ...rest] = item.split("\n");
                  return (
                    <li key={`${key}.${String(n)}`}>
                      <Inline text={first ?? ""} props={props} prefix={`${key}.${String(n)}.`} />
                      {rest.length ? (
                        <div className="mt-1 flex flex-col gap-1">
                          <Blocks
                            blocks={parseBlocks(rest.join("\n"))}
                            props={props}
                            prefix={`${key}.${String(n)}.`}
                          />
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </Tag>
            );
          }
          case "code":
            return (
              <pre
                key={key}
                className="overflow-x-auto rounded-md bg-gray-100 p-3 font-mono text-xs"
              >
                {block.text}
              </pre>
            );
          case "table":
            return (
              <div key={key} className="overflow-x-auto">
                <table className="min-w-full border-collapse text-left text-sm">
                  <thead>
                    <tr>
                      {block.header.map((cell, n) => (
                        <th key={n} className="border-b border-gray-300 px-2 py-1 font-semibold">
                          <Inline text={cell} props={props} prefix={`${key}.h${String(n)}.`} />
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, r) => (
                      <tr key={r} className="align-top">
                        {row.map((cell, n) => (
                          <td key={n} className="border-b border-gray-100 px-2 py-1">
                            <Inline
                              text={cell}
                              props={props}
                              prefix={`${key}.${String(r)}.${String(n)}.`}
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          case "quote":
            return (
              <blockquote key={key} className="border-l-4 border-gray-200 pl-3 text-gray-700">
                <Blocks blocks={parseBlocks(block.text)} props={props} prefix={`${key}.`} />
              </blockquote>
            );
          case "admonition":
            return (
              <div
                key={key}
                className={`flex flex-col gap-1 rounded-md border px-3 py-2 ${ADMONITION_STYLES[block.type] ?? "border-blue-200 bg-blue-50"}`}
              >
                <p className="font-semibold">{block.title}</p>
                <Blocks blocks={parseBlocks(block.body)} props={props} prefix={`${key}.`} />
              </div>
            );
          case "rule":
            return <hr key={key} className="border-gray-200" />;
        }
      })}
    </>
  );
}

export function Markdown(props: MarkdownProps) {
  return (
    <div className={props.className ?? "flex flex-col gap-2 text-sm leading-relaxed"}>
      <Blocks blocks={parseBlocks(props.text)} props={props} prefix="b" />
    </div>
  );
}
