"""The PDF export of a distillation-fidelity run (the Benchmarking tab's Export results).

User, 2026-11-23: "can we add an 'export results' button to create and download a PDF of the results for the
distillation fidelity exercise? and can that also include the Table 1 ... separately for enrollment and soc evals?"

``build_report`` draws the PDF from what the run page shows (``Benchmarks.fidelity``: metrics.json, design notes,
receipts) and, for a real run, the demographic tables of ``matchminer_ai.evaluation.demographics`` (aggregates,
already suppressed). Sections: a title page (the eval, the models, the denominators), retrieval and the decision model
with the page's tables and charts, Table 1, the metrics by demographic group, a methods note, counts and design, and
the stages. Charts follow the page's (one brand-blue series, grey hairline axes; every value is in the table beside
it). Built with reportlab's own fonts and drawing, so no plotting library is needed.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import time
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (CondPageBreak, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

RANKERS = (("embeddinggemma2", "(a) EmbeddingGemma 2 (base)"),
           ("trialspace_round2", "(b) TrialSpace round 2"),
           ("trialspace_round2_decision", "(c) TrialSpace round 2 top K + decision reasonableness"))
SHORT = {"embeddinggemma2": "(a)", "trialspace_round2": "(b)", "trialspace_round2_decision": "(c)"}
WORKFLOWS = (("patient_centric", "Trials for a patient (patient-centric)"),
             ("trial_centric", "Patients for a trial (trial-centric)"))
DIFFERENCES = (("trialspace_round2", "embeddinggemma2"), ("trialspace_round2_decision", "trialspace_round2"),
               ("trialspace_round2_decision", "embeddinggemma2"))
FAMILY_LABELS = {"reasonableness": "Reasonableness", "cancer_type": "Cancer type", "burden": "Burden",
                 "treatment_history": "Treatment history", "biomarker": "Biomarker", "boilerplate": "Boilerplate",
                 "criterion (all lines)": "Criterion lines: all",
                 "criterion (inclusion lines)": "Criterion lines: inclusion",
                 "criterion (exclusion lines)": "Criterion lines: exclusion"}
SCOPES = (("pooled", "Both workflows pooled"), ("patient_centric", "Trials for a patient"),
          ("trial_centric", "Patients for a trial"))
BANDS = ("1-20", "21-50", "51-80", "81-100")
BAND_COLORS = ("#0d366b", "#1c5cab", "#3987e5", "#86b6ef")  # the page's ordinal ramp, nearest ranks darkest
STAGE_LABELS = {"extraction": "Cohort extraction", "patients": "Patients", "cohort_table": "Cohort table",
                "corpus": "Corpus and query samples", "retrieve": "Retrieval", "lists_ab": "Lists (a) and (b)",
                "lists_c": "Lists (c): decision rerank", "metrics": "Metrics"}
RESEARCH_USE = ("For research use. MatchMiner-AI is a research prioritization tool: its scores are screening signals "
                "for human review, and the reference here is a teacher model's labels, not a clinician's review.")

MARK = colors.HexColor("#00629b")
GRID = colors.HexColor("#e5e7eb")
AXIS = colors.HexColor("#d1d5db")
MUTED = colors.HexColor("#6b7280")
INK = colors.HexColor("#111827")
SURFACE = colors.white
PAGE = landscape(letter)
MARGIN = 40
WIDTH = PAGE[0] - 2 * MARGIN

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12, textColor=INK)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=9.5, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.5, leading=9)
CELL_RIGHT = ParagraphStyle("cell_right", parent=CELL, alignment=TA_RIGHT)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Helvetica-Bold", textColor=MUTED)
HEAD_RIGHT = ParagraphStyle("head_right", parent=HEAD, alignment=TA_RIGHT)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Helvetica-Bold", fontSize=22, leading=27, spaceAfter=4)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontSize=13, leading=17, textColor=MUTED, spaceAfter=14)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Helvetica-Bold", fontSize=14, leading=18, spaceBefore=4,
                    spaceAfter=6)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=8,
                    spaceAfter=4)
H3 = ParagraphStyle("h3", parent=BODY, fontName="Helvetica-Bold", fontSize=9, leading=12, spaceBefore=6,
                    spaceAfter=3, textColor=colors.HexColor("#374151"))
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)


# --------------------------------------------------------------------------- formatting


def num(value, digits=3) -> str:
    if value is None or not isinstance(value, (int, float)) or value != value:
        return "–"
    return f"{value:.{digits}f}"


def signed(value, digits=3) -> str:
    if value is None or value != value:
        return "–"
    return f"{'+' if value >= 0 else '-'}{abs(value):.{digits}f}"


def count(value) -> str:
    return "–" if value is None else f"{int(value):,}"


def interval(low, high, digits=3, fmt=None) -> str:
    if low is None or high is None:
        return ""
    fmt = fmt or (lambda v, d: f"{v:.{d}f}")
    return f"[{fmt(low, digits)}, {fmt(high, digits)}]"


def words(value: str) -> str:
    return value.replace("_", " ")


def stamp(value) -> str:
    """Seconds since the epoch or an ISO string, in the server's local time, as '2026-11-23 06:26 EST'."""
    from datetime import datetime
    if value in (None, ""):
        return ""
    try:
        moment = (datetime.fromtimestamp(value) if isinstance(value, (int, float))
                  else datetime.fromisoformat(str(value)))
    except ValueError:
        return str(value)
    if moment.tzinfo is not None:
        moment = moment.astimezone()
    return moment.strftime("%Y-%m-%d %H:%M ") + (time.strftime("%Z") if moment.tzinfo is None else moment.strftime("%Z"))


def p(text: str, style=CELL) -> Paragraph:
    return Paragraph(text, style)


def estimate(value, low=None, high=None, digits=3, sign=False, base=None) -> Paragraph:
    """A value with its interval (and a chance baseline) under it, as the page's metric cells."""
    fmt = signed if sign else num
    lines = [f"<b>{fmt(value, digits)}</b>"]
    ci = interval(low, high, digits, (lambda v, d: signed(v, d)) if sign else None)
    if ci:
        lines.append(f"<font size=6.5 color='#6b7280'>{ci}</font>")
    if base is not None:
        lines.append(f"<font size=6.5 color='#6b7280'>base {num(base)}</font>")
    return p("<br/>".join(lines))


def headers(names, right=()) -> list:
    return [p(name, HEAD_RIGHT if i in right else HEAD) for i, name in enumerate(names)]


def table(rows, widths, *, header=1, right=(), spans=(), zebra=False, section_rows=()) -> Table:
    out = Table(rows, colWidths=widths, repeatRows=header, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
             ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
             ("LINEBELOW", (0, header - 1), (-1, header - 1), 0.75, AXIS),
             ("LINEBELOW", (0, header), (-1, -1), 0.4, GRID)]
    if header:
        style += [("FONTNAME", (0, 0), (-1, header - 1), "Helvetica-Bold"),
                  ("TEXTCOLOR", (0, 0), (-1, header - 1), MUTED)]
    for column in right:
        style.append(("ALIGN", (column, 0), (column, -1), "RIGHT"))
    for span in spans:
        style.append(("SPAN", *span))
    for row in section_rows:  # a section's title stays with its first row
        style += [("FONTNAME", (0, row), (-1, row), "Helvetica-Bold"), ("TOPPADDING", (0, row), (-1, row), 6),
                  ("NOSPLIT", (0, row), (-1, min(row + 1, len(rows) - 1)))]
    if zebra:
        style.append(("ROWBACKGROUNDS", (0, header), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]))
    out.setStyle(TableStyle(style))
    return out


# --------------------------------------------------------------------------- charts (the page's, as drawings)


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    import math
    span = hi - lo
    if not span > 0:
        return [lo]
    raw = span / n
    power = 10 ** math.floor(math.log10(raw))
    step = next((m for m in (1, 2, 2.5, 5, 10) if m * power >= raw), 10) * power
    out, v = [], math.ceil(lo / step - 1e-9) * step
    while v <= hi + 1e-9:
        out.append(0.0 if abs(v) < 1e-12 else float(f"{v:.12g}"))
        v += step
    return out


def domain_of(values, include=()) -> tuple[float, float]:
    import math
    finite = [v for v in (*values, *include) if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return (0.0, 1.0)
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    ticks = nice_ticks(lo, hi, 4)
    step = ticks[1] - ticks[0] if len(ticks) > 1 else (hi - lo) / 4
    return (math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step)


def scale(domain, a, b):
    lo, hi = domain
    return lambda v: a + (min(max(v, lo), hi) - lo) / ((hi - lo) or 1) * (b - a)


def tick_text(v: float, step: float) -> str:
    digits = 0
    while digits < 4 and abs(round(step * 10 ** digits) - step * 10 ** digits) > 1e-6:
        digits += 1
    return f"{v:.{digits}f}"


PAD = 6


def interval_mark(value, low, high, domain, width=84, reference=None) -> Drawing:
    """A forest-plot row: hairline ticks, an optional reference line, the interval and the estimate."""
    height = 13
    drawing = Drawing(width, height)
    x = scale(domain, PAD, width - PAD)
    for t in nice_ticks(*domain, n=max(2, min(4, int((width - 2 * PAD) // 28)))):
        drawing.add(Line(x(t), 1.5, x(t), height - 1.5, strokeColor=GRID, strokeWidth=0.5))
    if reference is not None:
        drawing.add(Line(x(reference), 0, x(reference), height, strokeColor=MUTED, strokeWidth=0.6))
    if value is not None and low is not None and high is not None:
        drawing.add(Line(x(low), height / 2, x(high), height / 2, strokeColor=MARK, strokeWidth=1.6, strokeLineCap=1))
    if value is not None:
        drawing.add(Circle(x(value), height / 2, 2.6, fillColor=MARK, strokeColor=SURFACE, strokeWidth=1))
    return drawing


def interval_axis(domain, width=84) -> Drawing:
    drawing = Drawing(width, 11)
    x = scale(domain, PAD, width - PAD)
    ticks = nice_ticks(*domain, n=max(2, min(4, int((width - 2 * PAD) // 28))))
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1
    drawing.add(Line(PAD, 10, width - PAD, 10, strokeColor=AXIS, strokeWidth=0.5))
    for t in ticks:
        drawing.add(String(x(t), 2, tick_text(t, step), fontName="Helvetica", fontSize=6, fillColor=MUTED,
                           textAnchor="middle"))
    return drawing


def reliability_plot(bins, size=150) -> Drawing:
    """Observed yes rate against mean P(yes) per equal-mass bin, with the diagonal of perfect calibration."""
    left, right, top, bottom = 26, 6, 6, 22
    drawing = Drawing(size, size)
    x = scale((0, 1), left, size - right)
    y = scale((0, 1), bottom, size - top)
    for t in (0, 0.25, 0.5, 0.75, 1):
        drawing.add(Line(x(t), y(0), x(t), y(1), strokeColor=GRID, strokeWidth=0.5))
        drawing.add(Line(x(0), y(t), x(1), y(t), strokeColor=GRID, strokeWidth=0.5))
        label = str(t) if t in (0, 1) else f"{t:.2f}".rstrip("0")
        drawing.add(String(x(t), bottom - 9, label, fontName="Helvetica", fontSize=6, fillColor=MUTED,
                           textAnchor="middle"))
        drawing.add(String(left - 3, y(t) - 2, label, fontName="Helvetica", fontSize=6, fillColor=MUTED,
                           textAnchor="end"))
    drawing.add(Line(x(0), y(0), x(1), y(0), strokeColor=AXIS, strokeWidth=0.6))
    drawing.add(Line(x(0), y(0), x(0), y(1), strokeColor=AXIS, strokeWidth=0.6))
    drawing.add(Line(x(0), y(0), x(1), y(1), strokeColor=MUTED, strokeWidth=0.6))
    drawing.add(String((x(0) + x(1)) / 2, 2, "Mean P(yes)", fontName="Helvetica", fontSize=6.5, fillColor=MUTED,
                       textAnchor="middle"))
    label = String(0, 0, "Observed yes", fontName="Helvetica", fontSize=6.5, fillColor=MUTED, textAnchor="middle")
    from reportlab.graphics.shapes import Group
    rotated = Group(label)
    rotated.translate(7, (y(0) + y(1)) / 2)
    rotated.rotate(90)
    drawing.add(rotated)
    points = [(x(b["mean_predicted"]), y(b["observed"])) for b in bins]
    if len(points) > 1:
        drawing.add(PolyLine([c for pt in points for c in pt], strokeColor=MARK, strokeWidth=1.5, strokeLineJoin=1,
                             strokeLineCap=1))
    for px, py in points:
        drawing.add(Circle(px, py, 2.6, fillColor=MARK, strokeColor=SURFACE, strokeWidth=1))
    return drawing


def segmented_bar(parts, width=360) -> Drawing:
    """A thin bar split into parts with 2pt gaps, and its legend (names and shares) under it."""
    height, gap = 8, 2
    drawing = Drawing(width, 26)
    total = sum(v for _, v, _ in parts) or 1
    usable = width - gap * (len(parts) - 1)
    at = 0.0
    for _, value, color in parts:
        w = usable * value / total
        if w > 0:
            drawing.add(Rect(at, 16, w, height, fillColor=colors.HexColor(color), strokeColor=None))
        at += w + gap
    at = 0.0
    for label, value, color in parts:
        drawing.add(Rect(at, 3, 7, 7, fillColor=colors.HexColor(color), strokeColor=None))
        text = f"{label}  {value * 100:.1f}%"
        drawing.add(String(at + 10, 4, text, fontName="Helvetica", fontSize=7, fillColor=INK))
        at += 10 + stringWidth(text, "Helvetica", 7) + 14
    return drawing


# --------------------------------------------------------------------------- sections


def receipt(directory: Path, name: str) -> dict:
    try:
        return json.loads((directory / name / "stage_receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def eval_of(directory: Path) -> dict:
    """The eval a run is (from its patients stage receipt, no patient data): synthetic, enrollment or SOC."""
    identity = receipt(directory, "patients").get("identity") or {}
    if identity.get("stage") == "synthetic_patients":
        split = identity.get("source_split")
        return dict(kind="synthetic", label="Synthetic patients" + (f" ({split} split)" if split else ""))
    argv = list((receipt(directory, "patients").get("provenance") or {}).get("argv") or [])
    name = Path(argv[argv.index("--denominator") + 1]).name if "--denominator" in argv[:-1] else ""
    from matchminer_ai.evaluation.demographics import EVAL_KINDS
    kind, label = EVAL_KINDS.get(name, ("other", "Retrospective cohort"))
    return dict(kind=kind, label=label)


def models(directory: Path) -> list[tuple[str, str]]:
    """The rankers' models as the receipts identify them (names, revisions and digests; no paths)."""
    def encoder(embedder):
        enc = (receipt(directory, f"embeddings/{embedder}/patients").get("identity") or {}).get("encoder") or {}
        model, revision = str(enc.get("model") or ""), enc.get("revision")
        if "/" in model and len(model) < 80:
            return model + (f" @ {str(revision)[:8]}" if revision else "")
        return f"model digest {model[:12]}" if model else "–"
    decide = receipt(directory, "decide/match").get("identity") or {}
    lists = receipt(directory, "lists_c").get("content") or {}
    decision = f"fingerprint {str(decide.get('model_fingerprint'))[:12]}" if decide.get("model_fingerprint") else "–"
    k = lists.get("k")
    return [(RANKERS[0][1], encoder("embeddinggemma2")), (RANKERS[1][1], encoder("trialspace_round2")),
            (RANKERS[2][1], f"TrialSpace round 2 top {k if k else 'K'}, reranked by the decision model ({decision})")]


def title_page(run: dict, directory: Path, info: dict, mode: str, demographics: dict | None) -> list:
    stages = run.get("stages") or {}
    patients = (stages.get("patients") or {}).get("content") or {}
    corpus = (stages.get("corpus") or {}).get("content") or {}
    counts = (run.get("metrics") or {}).get("counts") or {}
    label = info["label"]
    splits = (demographics or {}).get("eval", {}).get("splits") or []
    if len(splits) == 1:
        label += f" ({splits[0]} split)"
    out = [Spacer(1, 30), p("Distillation fidelity", TITLE), p(escape(run.get("name") or ""), SUBTITLE)]
    about = [("Eval", label),
             ("Patients", "Real patients: aggregates only, small cells suppressed" if mode == "real"
              else "Synthetic patients (no PHI)"),
             ("Report built", time.strftime("%Y-%m-%d %H:%M %Z")),
             ("Run status updated", stamp(run.get("updated"))),
             ("Teacher labeling", "finished" if run.get("released") else "not finished"),
             ("Metrics computed", stamp((stages.get("metrics") or {}).get("finished_at")))]
    out += [table([[p(f"<b>{escape(k)}</b>"), p(escape(v))] for k, v in about], [130, 520], header=0), Spacer(1, 12)]
    out += [p("Models", H2), table([[p("<b>Ranker</b>"), p("<b>Model</b>")]]
                                   + [[p(escape(a)), p(escape(b))] for a, b in models(directory)], [260, 450])]
    denominators = [
        *([] if info["kind"] == "synthetic" else [("Index events (treatment starts)", patients.get("events"))]),
        ("Patients", patients.get("persons")),
        ("Cohorts", corpus.get("cohorts")),
        ("Patient-centric queries", corpus.get("patient_queries")),
        ("Trial-centric queries", corpus.get("cohort_queries")),
        ("Comparison subset for rankers (a) and (b): patient queries", corpus.get("comparison_patient_queries")),
        ("Comparison subset for rankers (a) and (b): cohort queries", corpus.get("comparison_cohort_queries")),
        ("Pairs on the merged top-20 lists with teacher labels", counts.get("union_pairs_with_match_labels")),
        ("Boilerplate combinations with teacher labels", counts.get("boilerplate_combinations_labeled")),
    ]
    rows = [[p(escape(k)), p(count(v), CELL_RIGHT)] for k, v in denominators if v is not None]
    out += [Spacer(1, 10), p("Denominators", H2), table([[p("<b>Count</b>"), p("<b>n</b>", HEAD_RIGHT)]] + rows,
                                                         [400, 90])]
    out += [Spacer(1, 16), p(escape(RESEARCH_USE), SMALL), PageBreak()]
    return out


def retrieval_section(metrics: dict, k_binding: dict | None) -> list:
    retrieval = metrics.get("retrieval")
    if not retrieval:
        return []
    top = retrieval["top"]
    full = metrics.get("full_pipeline")
    out = [p(f"Retrieval against the teacher: P@{top}", H1),
           p(f"P@{top}: for each query, the share of its top {top} that the teacher judged a reasonable match, over the "
             "list's labeled pairs (unlabeled pairs are counted beside it). Mean score: the teacher's 0-6 match score "
             "of the list. 95% intervals bootstrap the queries; paired differences resample the same queries."
             + (" These ranker comparisons cover the seeded comparison queries; the full pipeline's table covers every "
                "query." if full else ""), BODY)]
    rows = list(retrieval["table"].values())
    score_domain = domain_of([v for r in rows for v in (r["mean_score"]["low"], r["mean_score"]["high"])], [0])
    diffs = retrieval.get("paired_differences") or {}

    def diff_domain(metric):
        return domain_of([v for k, d in diffs.items() if f"/{metric}/" in k for v in (d.get("low"), d.get("high"))], [0])

    domains = dict(precision=diff_domain("precision"), mean_score=diff_domain("mean_score"))
    for workflow, title in WORKFLOWS:
        present = [(r, label) for r, label in RANKERS if f"{workflow}/{r}" in retrieval["table"]]
        block = [p(title, H2)]
        if not present:
            out += block + [p("No queries.", SMALL)]
            continue
        head = ["Ranker", "Queries", f"P@{top} [95% CI]", "", "Mean 0-6 score [95% CI]", "", f"Lists &lt; {top}",
                "Unlabeled pairs"]
        data = [headers(head, right=(1, 6, 7))]
        for ranker, label in present:
            m = retrieval["table"][f"{workflow}/{ranker}"]
            pr, s = m["precision_at_top"], m["mean_score"]
            data.append([p(escape(label)), count(m["queries"]), estimate(pr["mean"], pr["low"], pr["high"]),
                         interval_mark(pr["mean"], pr["low"], pr["high"], (0, 1)),
                         estimate(s["mean"], s["low"], s["high"], 2),
                         interval_mark(s["mean"], s["low"], s["high"], score_domain),
                         count(m["queries_with_fewer_than_top"]), count(m["unlabeled_pairs"])])
        data.append(["", "", "", interval_axis((0, 1)), "", interval_axis(score_domain), "", ""])
        widths = [210, 50, 80, 90, 85, 90, 50, 57]
        block.append(table(data, widths, right=(1, 6, 7)))
        diff_rows = [(a, b) for a, b in DIFFERENCES
                     if (diffs.get(f"{workflow}/precision/{a}-minus-{b}") or {}).get("mean") is not None]
        if diff_rows:
            data = [["Difference", f"P@{top} [95% CI]", "", "Mean score [95% CI]", "", "Queries"]]
            for a, b in diff_rows:
                d1 = diffs.get(f"{workflow}/precision/{a}-minus-{b}") or {}
                d2 = diffs.get(f"{workflow}/mean_score/{a}-minus-{b}") or {}
                data.append([f"{SHORT[a]} - {SHORT[b]}", estimate(d1.get("mean"), d1.get("low"), d1.get("high"), sign=True),
                             interval_mark(d1.get("mean"), d1.get("low"), d1.get("high"), domains["precision"], reference=0),
                             estimate(d2.get("mean"), d2.get("low"), d2.get("high"), 2, sign=True),
                             interval_mark(d2.get("mean"), d2.get("low"), d2.get("high"), domains["mean_score"],
                                           reference=0),
                             count(d1.get("n") or d2.get("n"))])
            data.append(["", "", interval_axis(domains["precision"]), "", interval_axis(domains["mean_score"]), ""])
            block += [p("Paired differences", H3), table(data, [120, 90, 90, 90, 90, 50], right=(5,))]
        if full and (m := full["table"].get(f"{workflow}/{full['ranker']}")) and m["precision_at_top"]["mean"] is not None:
            cluster = full["clusters"][workflow]
            pr, s = m["precision_at_top"], m["mean_score"]
            data = [headers(["Ranker", "Queries", f"{cluster.capitalize()}s", f"P@{top} [95% CI]", "",
                             "Mean 0-6 score [95% CI]", "", f"Lists &lt; {top}", "Unlabeled pairs"], right=(1, 2, 7, 8)),
                    [p(escape(dict(RANKERS)[full["ranker"]])), count(m["queries"]), count(m["clusters"]),
                     estimate(pr["mean"], pr["low"], pr["high"]), interval_mark(pr["mean"], pr["low"], pr["high"], (0, 1)),
                     estimate(s["mean"], s["low"], s["high"], 2),
                     interval_mark(s["mean"], s["low"], s["high"], score_domain),
                     count(m["queries_with_fewer_than_top"]), count(m["unlabeled_pairs"])],
                    ["", "", "", "", interval_axis((0, 1)), "", interval_axis(score_domain), "", ""]]
            block += [p(f"Full pipeline over every query: P@{top} of (c)", H3),
                      p(f"Every query of this workflow; 95% intervals bootstrap {cluster}s (all of a {cluster}'s "
                        "queries together).", SMALL),
                      table(data, [170, 50, 50, 80, 90, 85, 90, 45, 52], right=(1, 2, 7, 8))]
        values = (k_binding or {}).get(workflow)
        if values:
            bands = [b for b in BANDS if b in values] + [b for b in values if "-" in b and b not in BANDS]
            parts = [(f"ranks {b.replace('-', '–')}", float(values[b]), BAND_COLORS[min(i, len(BAND_COLORS) - 1)])
                     for i, b in enumerate(bands)]
            beyond = values.get("queries_with_a_pick_beyond_80")
            block += [p(f"Where (c)'s top {top} sit in TrialSpace's ranking", H3), segmented_bar(parts, 520)]
            if beyond is not None:
                block.append(p(f"{count(beyond)} queries with a pick beyond rank 80.", SMALL))
        out += [CondPageBreak(220), *block, Spacer(1, 6)]
    return out


def decision_section(metrics: dict) -> list:
    decision = metrics.get("decision")
    if not decision:
        return []
    out = [PageBreak(), p("Decision model against the teacher", H1),
           p("On the merged top-20 pairs of the three rankers: one row per unique pair (boilerplate: per unique "
             "combination of the two boilerplate texts; criterion lines: a stratified sample). Yes means potentially "
             "eligible or not clearly failing; for the cancer type, burden, treatment history and biomarker questions, "
             "yes means the cohort specifically requires it and the patient is documented to meet it. AUPRC (no) "
             "scores 1 - P(yes) against the no class; its chance level is the no prevalence (base). 95% intervals "
             "resample patients (pooled) or queries (per workflow).", BODY)]
    for scope, title in SCOPES:
        tables = decision.get(scope)
        if not tables:
            continue
        families = list(tables.items())
        domain = domain_of([v for _, m in families for v in ((m.get("auroc") or {}).get("low"),
                                                            (m.get("auroc") or {}).get("high"))], [1])
        head = ["Question", "n (clusters)", "AUROC", "", "AUPRC (yes)", "AUPRC (no)", "Brier", "ECE",
                "Cal. intercept", "Cal. slope", "Mean P(yes) / observed"]
        data = [[p(h, HEAD if i != 1 else HEAD_RIGHT) for i, h in enumerate(head)]]
        for family, m in families:
            label = FAMILY_LABELS.get(family, family)
            if not m.get("n"):
                data.append([label, "0", "no rows"] + [""] * 8)
                continue

            def cell(key, base=None, sign=False):
                v = m.get(key) or {}
                return estimate(v.get("value"), v.get("low"), v.get("high"), sign=sign, base=base)
            a = m.get("auroc") or {}
            data.append([p(f"<b>{escape(label)}</b>"),
                         p(f"{count(m['n'])}<br/><font size=6.5 color='#6b7280'>({count(m.get('clusters'))})</font>",
                           CELL_RIGHT),
                         cell("auroc"), interval_mark(a.get("value"), a.get("low"), a.get("high"), domain, width=72),
                         cell("auprc_yes", m.get("prevalence_yes")), cell("auprc_no", m.get("prevalence_no")),
                         cell("brier"), cell("ece"), cell("calibration_intercept", sign=True), cell("calibration_slope"),
                         p(f"{num(m.get('mean_predicted'))}<br/><font size=6.5 color='#6b7280'>"
                           f"{num(m.get('observed_rate'))}</font>", CELL_RIGHT)])
        data.append(["", "", "", interval_axis(domain, 72), "", "", "", "", "", "", ""])
        out += [CondPageBreak(200), p(title, H2),
                table(data, [96, 56, 66, 76, 66, 66, 56, 56, 62, 56, 56], right=(1, 10))]
    pooled = decision.get("pooled") or {}
    plots = [(FAMILY_LABELS.get(f, f), m) for f, m in pooled.items() if m.get("n") and m.get("reliability")]
    if plots:
        cells = []
        for label, m in plots:
            caption = p(f"<b>{escape(label)}</b><br/><font size=6.5 color='#6b7280'>ECE "
                        f"{num((m.get('ece') or {}).get('value'))} · n {count(m['n'])}</font>")
            cells.append([caption, reliability_plot(m["reliability"], size=132)])
        per_row = 5
        grid_rows = [cells[i:i + per_row] + [""] * (per_row - len(cells[i:i + per_row]))
                     for i in range(0, len(cells), per_row)]
        grid = Table(grid_rows, colWidths=[WIDTH / per_row] * per_row, hAlign="LEFT")
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        out += [CondPageBreak(260), p("Reliability (both workflows pooled)", H2),
                p("Observed yes rate against the mean P(yes) in equal-mass bins of P(yes); the diagonal is perfect "
                  "calibration.", SMALL), grid]
    return out


def demographic_cell(cell: dict) -> str:
    if cell.get("text"):
        return cell["text"]
    if not cell.get("n"):
        return "0"
    return f"{cell['n']:,} ({cell['pct']:.1f}%)"


def linkage_text(value: dict | None) -> str:
    if not value:
        return "–"
    return value["text"] if value.get("text") else f"{value['pct']:.1f}%"


def table1_section(demographics: dict, info: dict) -> list:
    out = [PageBreak(), p("Table 1: patients and treatment starts", H1)]
    t1 = demographics["table1"]
    columns = t1["columns"]
    out.append(p(f"{escape(info['label'])}: the eval's patients and their treatment starts (index events). Patients: "
                 "race, ethnicity and sex, with age and cancer type at the patient's first treatment start in this "
                 "eval. Treatment starts: every index event, with its own age and cancer type.", BODY))
    head = [p("Characteristic", HEAD)] + [p(f"{escape(c['label'])}<br/>(N = {c['n']:,})", HEAD_RIGHT) for c in columns]
    data, sections = [head], []
    ages = []
    for c in columns:
        a = t1["age"][c["key"]]
        ages.append("–" if a["median"] is None else f"{a['median']:.0f} ({a['q1']:.0f}–{a['q3']:.0f})")
    data.append(["Age at treatment start, years, median (IQR)", *ages])
    for section in t1["sections"]:
        sections.append(len(data))
        data.append([section["label"] + ", n (%)", *[""] * len(columns)])
        for row in section["rows"]:
            data.append(["    " + row["category"], *[demographic_cell(row["cells"][c["key"]]) for c in columns]])
    out.append(table(data, [300, 150, 150], right=(1, 2), section_rows=sections))
    link = demographics.get("linkage") or {}
    out += [Spacer(1, 6),
            p("Age is (index date - birth date) / 365.25, among treatment starts with a recorded birth date. The index "
              "date is the start of the treatment plan that defines the event: the first research chemotherapy plan "
              "on the trial for an enrollment, the standard chemotherapy plan for a SOC start. Cancer type groups the "
              "ICD-10 diagnosis codes of the index plan by site (a primary cancer code first, then in situ, then an "
              "unknown primary or secondary site). Race: the registration record's derived race (several races: "
              "Multiple); ethnicity: its Hispanic indicator; sex: its gender, else the sex assigned at birth.", SMALL),
            p("Linkage to the registration records: "
              f"{linkage_text(link.get('registration'))} of treatment starts; recorded birth date "
              f"{linkage_text(link.get('birth_date'))}; index plan diagnosis code "
              f"{linkage_text(link.get('plan_diagnosis'))}.", SMALL),
            p("<b>Small cells.</b> " + escape(demographics.get("rule") or ""), SMALL)]
    return out


def groups_section(demographics: dict) -> list:
    out = [CondPageBreak(300), p("Fidelity by demographic group", H1)]
    groups = demographics["groups"]
    out.append(p("Headline metrics by group: the full pipeline's (ranker (c)) P@20 and mean teacher 0-6 score over the "
                 "patient-centric queries (each a treatment start) of the group, and the decision model's reasonableness "
                 "AUROC over the teacher-labeled pairs of the group's patients (both workflows pooled). Trial-centric "
                 "queries are cohorts, so trial-centric retrieval is not split by patient group. 95% intervals bootstrap "
                 "patients (all of a patient's treatment starts together). The interval marks are zoomed to the "
                 "groups' range; the axis under each column gives it.", BODY))
    rows = [groups["overall"], *[r for s in groups["sections"] for r in s["rows"]]]
    # zoomed to the groups' intervals (the axis under each column gives the range)
    p_domain, s_domain, a_domain = (domain_of([(r.get(key) or {}).get(k) for r in rows for k in ("low", "high")])
                                    for key in ("precision", "mean_score", "auroc"))

    def metric(row, key, domain, digits=3):
        value = row.get(key)
        if value is None:
            hidden = (row.get("hidden") or {}).get("auroc" if key == "auroc" else "precision")
            return [p("n&lt;11" if hidden else "–"), ""]
        return [estimate(value["value"], value["low"], value["high"], digits),
                interval_mark(value["value"], value["low"], value["high"], domain, width=80)]

    head = ["Group", "Treatment starts", "P@20 (c) [95% CI]", "", "Mean 0-6 score (c) [95% CI]", "",
            "Reasonableness AUROC [95% CI]", ""]
    data, sections = [[p(h, HEAD if i != 1 else HEAD_RIGHT) for i, h in enumerate(head)]], []

    def line(label, row):
        return [label, demographic_cell(row["queries"]).split(" (")[0], *metric(row, "precision", p_domain),
                *metric(row, "mean_score", s_domain, 2), *metric(row, "auroc", a_domain)]

    data.append(line("All patients", groups["overall"]))
    for section in groups["sections"]:
        sections.append(len(data))
        data.append([section["label"], "", "", "", "", "", "", ""])
        for row in section["rows"]:
            data.append(line("    " + row["category"], row))
    data.append(["", "", "", interval_axis(p_domain, 80), "", interval_axis(s_domain, 80), "",
                 interval_axis(a_domain, 80)])
    out += [table(data, [150, 62, 80, 86, 80, 86, 82, 86], right=(1,), section_rows=sections), Spacer(1, 6),
            p("<b>Small cells.</b> " + escape(demographics.get("rule") or ""), SMALL)]
    return out


def methods_section(metrics: dict, demographics: bool) -> list:
    full = bool(metrics.get("full_pipeline"))
    items = [
        "<b>Reference.</b> The teacher model's labels (one model's judgments; the design notes name it) are the "
        "reference throughout. The numbers measure how closely the distilled models reproduce the teacher on these "
        "patients, not clinical accuracy or eligibility.",
        "<b>P@20.</b> For each query, the share of a ranker's top 20 that the teacher judged a reasonable match (its "
        "reasonableness answer is yes), over the list's labeled pairs; the mean of that share over queries. Mean score: "
        "the teacher's 0-6 match score (0 unless reasonable, else 1 + cancer type + burden + treatment history + 2 x "
        "biomarker), averaged over the list and then over queries.",
        "<b>Rankers.</b> (a) base EmbeddingGemma 2 and (b) TrialSpace round 2: cosine top 20. (c) the full pipeline: "
        "TrialSpace's top K reranked by the decision model's reasonableness P(yes), top 20.",
        "<b>Decision model.</b> AUROC, AUPRC, Brier score, expected calibration error (equal-mass bins), calibration "
        "intercept and slope of its P(yes) against the teacher's answers, on the pairs of the merged top-20 lists; "
        "selection-conditioned on retrieval.",
        "<b>Intervals.</b> Percentile bootstraps: queries for the ranker comparison"
        + ("; persons or trials for the full pipeline over every query" if full else "")
        + "; patients (pooled) or queries (per workflow) for the decision model.",
    ]
    if demographics:
        items += [
            "<b>Age at treatment start.</b> The patient's age on the index date, the start of the treatment plan that "
            "defines the event (patient summaries were made from notes up to 14 days before it).",
            "<b>Groups.</b> Metrics by group use the same definitions within the group's patients; groups with fewer "
            "than 11 patients are not shown, and small counts are suppressed (Table 1's footnote).",
        ]
    return [CondPageBreak(200), p("Methods note", H1),
            *[Paragraph(text, BULLET, bulletText="•") for text in items], Spacer(1, 6), p(escape(RESEARCH_USE), SMALL)]


def notes_section(run: dict) -> list:
    counts = (run.get("metrics") or {}).get("counts") or {}
    notes = [n if isinstance(n, str) else json.dumps(n) for n in run.get("design_notes") or []]
    out = [CondPageBreak(200), p("Counts and design", H1)]
    if counts:
        out.append(table([[p("<b>Count</b>"), p("<b>n</b>", HEAD_RIGHT)]]
                         + [[p(escape(words(k).capitalize())), p(count(v), CELL_RIGHT)] for k, v in counts.items()],
                         [400, 90]))
    if notes:
        out += [Spacer(1, 6), *[Paragraph(escape(n), BULLET, bulletText="•") for n in notes]]
    return out


def stage_label(job: str) -> str:
    import re
    if job in STAGE_LABELS:
        return STAGE_LABELS[job]
    if m := re.match(r"^embed_(.+)_(patients|cohorts)$", job):
        name = "EmbeddingGemma 2" if m[1] == "embeddinggemma2" else words(m[1]).replace("trialspace round2",
                                                                                         "TrialSpace round 2")
        return f"Embed {m[2]}: {name}"
    if m := re.match(r"^decide_(.+)$", job):
        return f"Decision model: {words(m[1])}"
    if m := re.match(r"^label_(.+)_(ab|c)$", job):
        return f"Teacher labels: {words(m[1])}, lists {'(a)(b)' if m[2] == 'ab' else '(c)'}"
    return words(job)


def receipt_key(job: str) -> str:
    import re
    if job == "retrieve":
        return "retrieval"
    if m := re.match(r"^(decide|label)_(.+)$", job):
        return f"{'labels' if m[1] == 'label' else 'decide'}/{m[2]}"
    return job


def stages_section(run: dict) -> list:
    jobs = run.get("jobs") or {}
    if not jobs:
        return []
    data = [["Stage", "State", "Finished", "Commit"]]
    for name, job in jobs.items():
        r = (run.get("stages") or {}).get(receipt_key(name)) or {}
        data.append([stage_label(name), job.get("state", ""), stamp(r.get("finished_at")), (r.get("commit") or "")[:7]])
    return [CondPageBreak(200), p("Stages", H1), table(data, [260, 70, 120, 70], zebra=True)]


def build_report(run: dict, directory: Path, *, mode: str, demographics: dict | None = None,
                 note: str | None = None) -> bytes:
    """The PDF of a fidelity run: ``run`` is ``Benchmarks.fidelity``'s answer, ``directory`` the run directory (for
    its receipts' model identities), ``demographics`` ``evaluation.demographics.run_demographics``' result (real
    mode) or None with ``note`` saying why there is none."""
    metrics = run.get("metrics") or {}
    info = (demographics or {}).get("eval") or eval_of(Path(directory))
    name = run.get("name") or ""
    buffer = BytesIO()

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, PAGE[1] - 24, f"Distillation fidelity · {name}")
        canvas.drawRightString(PAGE[0] - MARGIN, PAGE[1] - 24, info["label"])
        canvas.drawString(MARGIN, 20, "For research use; scores are screening signals for human review.")
        canvas.drawRightString(PAGE[0] - MARGIN, 20, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buffer, pagesize=PAGE, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=36,
                            bottomMargin=34, title=f"Distillation fidelity: {name}", author="MatchMiner-AI",
                            subject=info["label"], creator="MatchMiner-AI dashboard")
    story = title_page(run, Path(directory), info, mode, demographics)
    if metrics:
        story += retrieval_section(metrics, run.get("k_binding"))
        story += decision_section(metrics)
    else:
        story.append(p("The results appear here when the metrics stage has run.", BODY))
    if demographics is not None:
        story += table1_section(demographics, info)
        story += groups_section(demographics)
    else:
        story += [CondPageBreak(120), p("Table 1 and fidelity by demographic group", H1),
                  p(escape(note or "This run has no demographic data."), BODY)]
    story += methods_section(metrics, demographics is not None)
    story += notes_section(run)
    story += stages_section(run)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()


__all__ = ["build_report", "eval_of", "models"]
