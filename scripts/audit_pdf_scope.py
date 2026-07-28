#!/usr/bin/env python3
"""Audit whether each paper PDF is a single paper or a larger container.

The audit is intentionally non-destructive. It records candidate page ranges for
manual review but never rewrites or replaces source PDFs.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "toward",
    "towards",
    "with",
}


@dataclass(frozen=True)
class OutlineEntry:
    title: str
    page: int


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_tokens(value: Any) -> list[str]:
    return [
        token
        for token in normalize_text(value).split()
        if len(token) > 1 and token not in STOPWORDS
    ]


def title_similarity(expected: str, observed: str) -> float:
    """Score a short observed title against the expected paper title."""
    expected_normalized = normalize_text(expected)
    observed_normalized = normalize_text(observed)
    if not expected_normalized or not observed_normalized:
        return 0.0
    if expected_normalized in observed_normalized or observed_normalized in expected_normalized:
        return 1.0

    expected_set = set(title_tokens(expected))
    observed_set = set(title_tokens(observed))
    if not expected_set or not observed_set:
        return 0.0
    coverage = len(expected_set & observed_set) / len(expected_set)
    sequence = SequenceMatcher(None, expected_normalized, observed_normalized).ratio()
    return round((0.7 * coverage) + (0.3 * sequence), 4)


def page_title_coverage(expected: str, page_text: str) -> float:
    expected_set = set(title_tokens(expected))
    page_set = set(title_tokens(page_text))
    if not expected_set:
        return 0.0
    return round(len(expected_set & page_set) / len(expected_set), 4)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_outline_entries(reader: PdfReader) -> list[OutlineEntry]:
    """Return top-level outline destinations in document order."""
    entries: list[OutlineEntry] = []
    try:
        outline = reader.outline
    except Exception:
        return entries

    for item in outline:
        if isinstance(item, list):
            continue
        title = getattr(item, "title", "")
        if not title:
            continue
        try:
            page = reader.get_destination_page_number(item) + 1
        except Exception:
            continue
        entries.append(OutlineEntry(str(title), page))

    # Some PDFs repeat a destination in the top-level outline.
    deduplicated: list[OutlineEntry] = []
    seen: set[tuple[str, int]] = set()
    for entry in sorted(entries, key=lambda item: item.page):
        key = (normalize_text(entry.title), entry.page)
        if key not in seen:
            deduplicated.append(entry)
            seen.add(key)
    return deduplicated


def best_outline_match(
    title: str, entries: list[OutlineEntry], page_count: int
) -> tuple[float, str, int | None, int | None]:
    if not entries:
        return 0.0, "", None, None
    scored = [(title_similarity(title, entry.title), index, entry) for index, entry in enumerate(entries)]
    score, index, entry = max(scored, key=lambda item: item[0])

    next_page = None
    for later in entries[index + 1 :]:
        if later.page > entry.page:
            next_page = later.page
            break
    candidate_end = (next_page - 1) if next_page else page_count
    return score, entry.title, entry.page, candidate_end


def extract_page_texts(reader: PdfReader) -> tuple[list[str], int]:
    texts: list[str] = []
    failures = 0
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
            failures += 1
    return texts, failures


def text_extraction_quality(page_texts: list[str]) -> tuple[str, float]:
    combined = "\n".join(page_texts)
    if not combined.strip():
        return "empty", 1.0
    control_count = sum(
        1
        for char in combined
        if unicodedata.category(char) == "Cc" and char not in {"\n", "\r", "\t"}
    )
    control_ratio = control_count / max(len(combined), 1)
    if control_ratio >= 0.01 or "(cid:" in combined:
        return "suspect_encoding", round(control_ratio, 5)
    return "usable", round(control_ratio, 5)


def best_page_match(title: str, page_texts: list[str]) -> tuple[int | None, float]:
    if not page_texts:
        return None, 0.0
    scored = [
        (page_title_coverage(title, text[:6000]), index + 1)
        for index, text in enumerate(page_texts)
    ]
    score, page = max(scored, key=lambda item: item[0])
    return page, score


def page_contains_doi(doi: str, page_texts: list[str]) -> list[int]:
    normalized_doi = normalize_text(doi)
    if not normalized_doi:
        return []
    compact_doi = normalized_doi.replace(" ", "")
    matches = []
    for index, text in enumerate(page_texts):
        compact_page = normalize_text(text).replace(" ", "")
        if compact_doi and compact_doi in compact_page:
            matches.append(index + 1)
    return matches


def classify_scope(
    page_count: int,
    outline_count: int,
    outline_score: float,
    outline_start: int | None,
    outline_end: int | None,
    title_page: int | None,
    title_score: float,
) -> tuple[str, int | None, int | None, str]:
    container_signal = page_count >= 80 or outline_count >= 8
    usable_outline_range = (
        outline_score >= 0.68
        and outline_start is not None
        and outline_end is not None
        and outline_end >= outline_start
        and (outline_end - outline_start + 1) < page_count
    )

    # Short journal/conference papers often expose section headings as top-level
    # bookmarks. A strong title hit on the opening pages is better evidence that
    # these are single papers than bookmark count is evidence of a container.
    if page_count < 80 and title_page is not None and title_page <= 5 and title_score >= 0.6:
        return (
            "single_paper_likely",
            1,
            page_count,
            "Target title appears near the beginning and the PDF has no strong page-count container signal.",
        )
    if container_signal and usable_outline_range:
        return (
            "chapter_candidate",
            outline_start,
            outline_end,
            "Large/container PDF; title matched a bookmark and the next bookmark defines the candidate end.",
        )
    if container_signal:
        return (
            "container_needs_manual_boundary",
            outline_start or title_page,
            outline_end,
            "Large/container PDF without a sufficiently reliable complete bookmark range.",
        )
    if outline_score >= 0.68 and usable_outline_range:
        return (
            "chapter_candidate",
            outline_start,
            outline_end,
            "Target title matched a bookmark range in a multi-item PDF.",
        )
    return (
        "needs_manual_review",
        title_page,
        None,
        "Title or document-boundary evidence is insufficient for automatic classification.",
    )


def audit_record(row: pd.Series, pdf_dir: Path) -> dict[str, Any]:
    record = row.to_dict()
    paper_id = str(record.get("paper_id", ""))
    pdf_path = pdf_dir / f"{paper_id}.pdf"
    record.update(
        {
            "pdf_path": str(pdf_path),
            "pdf_exists": pdf_path.exists(),
            "pdf_sha256": "",
            "page_count": 0,
            "outline_entry_count": 0,
            "outline_match_title": "",
            "outline_match_score": 0.0,
            "outline_start_page": "",
            "outline_candidate_end_page": "",
            "best_title_page": "",
            "best_title_page_coverage": 0.0,
            "doi_hit_pages": "",
            "page_extraction_failures": 0,
            "text_extraction_status": "",
            "text_control_character_ratio": "",
            "scope_status": "missing_pdf",
            "candidate_start_page": "",
            "candidate_end_page": "",
            "candidate_page_count": "",
            "scope_reason": "No PDF named by paper_id was found.",
            "review_decision": "",
            "reviewed_start_page": "",
            "reviewed_end_page": "",
            "review_notes": "",
        }
    )
    if not pdf_path.exists():
        return record

    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        outline_entries = extract_outline_entries(reader)
        outline_score, outline_title, outline_start, outline_end = best_outline_match(
            str(record.get("title", "")), outline_entries, page_count
        )
        page_texts, extraction_failures = extract_page_texts(reader)
        extraction_status, control_ratio = text_extraction_quality(page_texts)
        title_page, title_score = best_page_match(str(record.get("title", "")), page_texts)
        doi_pages = page_contains_doi(str(record.get("doi", "")), page_texts)
        status, candidate_start, candidate_end, reason = classify_scope(
            page_count=page_count,
            outline_count=len(outline_entries),
            outline_score=outline_score,
            outline_start=outline_start,
            outline_end=outline_end,
            title_page=title_page,
            title_score=title_score,
        )
        candidate_count = (
            candidate_end - candidate_start + 1
            if candidate_start is not None and candidate_end is not None
            else ""
        )
        record.update(
            {
                "pdf_sha256": sha256_file(pdf_path),
                "page_count": page_count,
                "outline_entry_count": len(outline_entries),
                "outline_match_title": outline_title,
                "outline_match_score": outline_score,
                "outline_start_page": outline_start or "",
                "outline_candidate_end_page": outline_end or "",
                "best_title_page": title_page or "",
                "best_title_page_coverage": title_score,
                "doi_hit_pages": ";".join(str(page) for page in doi_pages),
                "page_extraction_failures": extraction_failures,
                "text_extraction_status": extraction_status,
                "text_control_character_ratio": control_ratio,
                "scope_status": status,
                "candidate_start_page": candidate_start or "",
                "candidate_end_page": candidate_end or "",
                "candidate_page_count": candidate_count,
                "scope_reason": reason,
            }
        )
    except Exception as exc:
        record.update(
            {
                "scope_status": "pdf_read_error",
                "scope_reason": repr(exc),
            }
        )
    return record


def write_report(
    audit_df: pd.DataFrame,
    output_csv: Path,
    report_path: Path,
    input_path: Path,
    group_label: str,
    expected_count: int | None,
) -> None:
    counts = audit_df["scope_status"].value_counts().to_dict()
    lines = [
        "# PDF Scope Audit",
        "",
        f"- Input: `{input_path}`",
        f"- Corpus selection: `{group_label}`",
        f"- Records in group: {len(audit_df)}",
        f"- Expected count supplied: {expected_count if expected_count is not None else 'not specified'}",
        f"- PDFs found: {int(audit_df['pdf_exists'].sum())}/{len(audit_df)}",
        f"- Output: `{output_csv}`",
        "",
        "## Status summary",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: {count}")

    if expected_count is not None and expected_count != len(audit_df):
        lines.extend(
            [
                "",
                "## Count mismatch",
                "",
                f"- The input contains {len(audit_df)} records, not the expected {expected_count}.",
                "- Resolve the corpus definition before freezing downstream results.",
            ]
        )

    review_df = audit_df[
        audit_df["scope_status"].isin(
            ["chapter_candidate", "container_needs_manual_boundary", "needs_manual_review", "pdf_read_error"]
        )
    ]
    lines.extend(["", "## Cases requiring review", ""])
    if review_df.empty:
        lines.append("- None")
    else:
        for _, row in review_df.iterrows():
            page_range = (
                f"{row['candidate_start_page']}-{row['candidate_end_page']}"
                if row["candidate_start_page"] and row["candidate_end_page"]
                else "unresolved"
            )
            lines.append(
                f"- `{row['paper_id']}` - {row['title']} "
                f"({row['page_count']} PDF pages; candidate {page_range}; `{row['scope_status']}`)"
            )

    lines.extend(
        [
            "",
            "## Review protocol",
            "",
            "1. Inspect the candidate first page, last page, and adjacent pages visually.",
            "2. Confirm that the first page contains the target title and authors.",
            "3. Confirm that the last page belongs to the target paper, normally ending with references.",
            "4. Confirm that the following page is another chapter/article or outside the target paper.",
            "5. Fill `review_decision`, `reviewed_start_page`, `reviewed_end_page`, and `review_notes` in the CSV.",
            "6. Do not cut or replace source PDFs until the range is approved.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/final_advancing_list.csv")
    parser.add_argument("--keyword", default=None)
    parser.add_argument(
        "--confirmed-list",
        default=None,
        help="Optional CSV whose paper_id column defines the exact corpus.",
    )
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    pdf_dir = Path(args.pdf_dir)

    input_df = pd.read_csv(input_path).fillna("")
    group_df = input_df.copy()
    group_labels = []
    if args.confirmed_list:
        confirmed_df = pd.read_csv(args.confirmed_list).fillna("")
        confirmed_ids = set(confirmed_df["paper_id"].astype(str))
        group_df = group_df[group_df["paper_id"].astype(str).isin(confirmed_ids)].copy()
        group_labels.append(f"confirmed IDs from {args.confirmed_list}")
    if args.keyword:
        keyword_mask = group_df["keyword"].astype(str).str.casefold() == args.keyword.casefold()
        group_df = group_df[keyword_mask].copy()
        group_labels.append(f"keyword={args.keyword}")
    if not group_labels:
        group_labels.append("all input records")
    group_label = "; ".join(group_labels)
    audited = [audit_record(row, pdf_dir) for _, row in group_df.iterrows()]
    audit_df = pd.DataFrame(audited)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False)
    write_report(
        audit_df,
        output_path,
        report_path,
        input_path,
        group_label,
        args.expected_count,
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(f"Audited {len(audit_df)} records; found {int(audit_df['pdf_exists'].sum())} PDFs.")
    print(audit_df["scope_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
