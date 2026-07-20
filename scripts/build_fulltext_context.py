#!/usr/bin/env python3
"""Build a full-text clustering input CSV from locally downloaded PDFs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


TARGET_TERMS = [
    "design knowledge",
    "design expertise",
    "design rationale",
    "design pattern",
    "design patterns",
    "design principle",
    "design principles",
    "design guideline",
    "design guidelines",
    "design heuristic",
    "design heuristics",
    "design procedure",
    "design procedures",
    "design rule",
    "design rules",
    "design framework",
    "design frameworks",
    "design method",
    "design methods",
    "design theory",
]

DISCUSSION_HEADINGS = [
    "discussion",
    "discussions",
    "findings and discussion",
    "results and discussion",
    "discussion and conclusion",
    "discussion and conclusions",
]


def normalize_space(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(path: Path, max_pages: int | None = None) -> tuple[str, str]:
    reader = PdfReader(str(path))
    page_texts = []
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    for page in pages:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception:
            continue
    return normalize_space("\n\n".join(page_texts)), "pypdf"


def split_paragraphs(text: str) -> list[str]:
    rough = re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z][a-z])", text)
    paragraphs = []
    for paragraph in rough:
        cleaned = normalize_space(paragraph)
        if len(cleaned) >= 80:
            paragraphs.append(cleaned)
    return paragraphs


def score_paragraph(paragraph: str, terms: list[str]) -> int:
    low = paragraph.lower()
    score = 0
    for term in terms:
        count = len(re.findall(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low))
        score += count * (4 if " " in term else 1)
    return score


def select_context(paragraphs: list[str], terms: list[str], max_paragraphs: int, max_chars: int) -> tuple[str, int]:
    scored = [(score_paragraph(paragraph, terms), len(paragraph), paragraph) for paragraph in paragraphs]
    matches = [(score, length, paragraph) for score, length, paragraph in scored if score > 0]
    matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [paragraph for _, _, paragraph in matches[:max_paragraphs]]
    if not selected:
        selected = paragraphs[:max_paragraphs]

    clipped = []
    for paragraph in selected:
        clipped.append(paragraph[:max_chars].strip())
    return "\n\n".join(clipped), len(matches)


def extract_discussion(paragraphs: list[str], max_paragraphs: int, max_chars: int) -> tuple[bool, int, str, str, str]:
    heading_pattern = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?(" + "|".join(re.escape(h) for h in DISCUSSION_HEADINGS) + r")\s*$",
        re.IGNORECASE,
    )
    start = None
    for index, paragraph in enumerate(paragraphs):
        first_line = paragraph.split("\n", 1)[0].strip()
        if heading_pattern.match(first_line):
            start = index
            break
        if re.match(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?discussion\b", first_line, re.IGNORECASE):
            start = index
            break

    if start is None:
        return False, 0, "", "", ""

    selected = paragraphs[start : start + max_paragraphs]
    discussion_full = "\n\n".join(p[:max_chars].strip() for p in selected)
    excerpt = selected[0][:max_chars].strip() if selected else ""
    summary_items = []
    for paragraph in selected[:3]:
        sentence = re.split(r"(?<=[.!?])\s+", paragraph.strip())[0]
        if len(sentence) >= 40:
            summary_items.append(f"- {sentence[:260].strip()}")
    return True, len(selected), "\n".join(summary_items), excerpt, discussion_full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/final_advancing_list.csv")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--confirmed-list", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-paragraphs", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    input_df = pd.read_csv(args.input).fillna("")
    if args.confirmed_list:
        confirmed = pd.read_csv(args.confirmed_list).fillna("")
        confirmed_ids = set(confirmed["paper_id"].astype(str))
    else:
        confirmed_ids = {path.stem for path in Path(args.pdf_dir).glob("*.pdf")}

    pdf_dir = Path(args.pdf_dir)
    rows = []
    failures = []
    for _, row in input_df.iterrows():
        paper_id = str(row["paper_id"])
        if paper_id not in confirmed_ids:
            continue
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        record = row.to_dict()
        if not pdf_path.exists():
            record.update(
                {
                    "pdf_found": False,
                    "pdf_path": "",
                    "pdf_extract_method": "",
                    "full_text_chars": 0,
                    "paragraph_count": 0,
                    "matched_paragraph_count": 0,
                    "extracted_context": "",
                    "discussion_found": False,
                    "discussion_paragraph_count": 0,
                    "discussion_summary": "",
                    "discussion_excerpt": "",
                    "discussion_full": "",
                    "extraction_terms": "; ".join(TARGET_TERMS),
                }
            )
            rows.append(record)
            failures.append((paper_id, "missing_pdf"))
            continue

        try:
            full_text, method = extract_pdf_text(pdf_path, args.max_pages)
            paragraphs = split_paragraphs(full_text)
            context, matched_count = select_context(paragraphs, TARGET_TERMS, args.max_paragraphs, args.max_chars)
            discussion_found, discussion_count, discussion_summary, discussion_excerpt, discussion_full = extract_discussion(
                paragraphs, args.max_paragraphs, args.max_chars
            )
            record.update(
                {
                    "pdf_found": True,
                    "pdf_path": str(pdf_path),
                    "pdf_extract_method": method,
                    "full_text_chars": len(full_text),
                    "paragraph_count": len(paragraphs),
                    "matched_paragraph_count": matched_count,
                    "extracted_context": context,
                    "discussion_found": discussion_found,
                    "discussion_paragraph_count": discussion_count,
                    "discussion_summary": discussion_summary,
                    "discussion_excerpt": discussion_excerpt,
                    "discussion_full": discussion_full,
                    "extraction_terms": "; ".join(TARGET_TERMS),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "pdf_found": False,
                    "pdf_path": str(pdf_path),
                    "pdf_extract_method": "failed",
                    "full_text_chars": 0,
                    "paragraph_count": 0,
                    "matched_paragraph_count": 0,
                    "extracted_context": "",
                    "discussion_found": False,
                    "discussion_paragraph_count": 0,
                    "discussion_summary": "",
                    "discussion_excerpt": "",
                    "discussion_full": "",
                    "extraction_terms": "; ".join(TARGET_TERMS),
                }
            )
            failures.append((paper_id, repr(exc)))
        rows.append(record)

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    found = out_df["pdf_found"].astype(str).str.lower().eq("true").sum() if len(out_df) else 0
    matched = (pd.to_numeric(out_df.get("matched_paragraph_count", pd.Series(dtype=int)), errors="coerce").fillna(0) > 0).sum()
    discussion = out_df["discussion_found"].astype(str).str.lower().eq("true").sum() if len(out_df) else 0
    report = [
        "# Full-Text Context Extraction",
        "",
        f"- Input CSV: `{args.input}`",
        f"- PDF folder: `{args.pdf_dir}`",
        f"- Output CSV: `{args.output}`",
        f"- Confirmed rows: {len(out_df)}",
        f"- PDFs extracted: {found}/{len(out_df)}",
        f"- Papers with matched paragraphs: {matched}/{len(out_df)}",
        f"- Papers with discussion sections: {discussion}/{len(out_df)}",
        f"- Max paragraphs per paper: {args.max_paragraphs}",
        f"- Max chunk characters: {args.max_chars}",
        "",
        "## Target Terms",
        "",
        ", ".join(TARGET_TERMS),
    ]
    if failures:
        report.extend(["", "## Extraction Failures", ""])
        report.extend(f"- `{paper_id}`: {reason}" for paper_id, reason in failures[:50])
        if len(failures) > 50:
            report.append(f"- ... {len(failures) - 50} more")
    Path(args.report).write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    print(f"Extracted {found}/{len(out_df)} PDFs; matched context for {matched}/{len(out_df)} papers.")


if __name__ == "__main__":
    main()
