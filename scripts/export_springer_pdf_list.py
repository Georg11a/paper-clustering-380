#!/usr/bin/env python3
"""Export Springer Nature-family PDFs from a completed PDF scope audit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SPRINGER_DOI_PREFIXES = (
    "10.1007/",
    "10.1057/",
    "10.1038/",
    "10.1186/",
)

SPRINGER_URL_MARKERS = (
    "link.springer.com",
    "springer.com",
    "springernature.com",
    "nature.com",
    "palgrave.com",
    "biomedcentral.com",
)


def is_springer_family(doi: str, url: str) -> bool:
    doi_value = str(doi or "").strip().casefold()
    url_value = str(url or "").strip().casefold()
    return doi_value.startswith(SPRINGER_DOI_PREFIXES) or any(
        marker in url_value for marker in SPRINGER_URL_MARKERS
    )


def is_springer_book_chapter_doi(doi: str) -> bool:
    value = str(doi or "").strip().casefold()
    return bool(re.match(r"^10\.1007/(?:978|3)-.+_\d+$", value))


def cutting_priority(row: pd.Series) -> tuple[str, str, str]:
    page_count = int(float(row.get("page_count", 0) or 0))
    status = str(row.get("scope_status", ""))
    doi = str(row.get("doi", ""))

    if status == "chapter_candidate":
        return (
            "high",
            "cut_candidate",
            "Container PDF with a candidate target range; manually verify and cut the target paper.",
        )
    if page_count >= 80 or status == "container_needs_manual_boundary":
        return (
            "high",
            "scope_decision_required",
            "Large/container PDF without a complete reliable range; decide the intended corpus unit before cutting.",
        )
    if is_springer_book_chapter_doi(doi) and status == "single_paper_likely":
        return (
            "medium",
            "verify_existing_chapter",
            "Book-chapter DOI, but the downloaded file already looks like a standalone chapter; verify only.",
        )
    if status != "single_paper_likely" or page_count >= 50:
        return (
            "medium",
            "boundary_review",
            "Boundary evidence is not fully conclusive; visually verify before text extraction.",
        )
    return (
        "low",
        "no_cut_expected",
        "Likely a normal single-paper PDF; no cut is expected unless visual review disagrees.",
    )


def write_markdown(rows: pd.DataFrame, output_path: Path, csv_path: Path) -> None:
    lines = [
        "# Springer Nature PDF Review List",
        "",
        f"- Machine-readable list: `{csv_path}`",
        f"- Total Springer Nature-family candidates: {len(rows)}",
        f"- High-priority manual cutting review: {(rows['manual_cut_priority'] == 'high').sum()}",
        f"- Medium-priority boundary review: {(rows['manual_cut_priority'] == 'medium').sum()}",
        f"- Low-priority likely single papers: {(rows['manual_cut_priority'] == 'low').sum()}",
        "",
        "Only high- and medium-priority files need boundary attention first. "
        "The original PDFs must remain unchanged.",
    ]
    for priority in ("high", "medium", "low"):
        subset = rows[rows["manual_cut_priority"] == priority]
        lines.extend(["", f"## {priority.title()} priority ({len(subset)})", ""])
        if subset.empty:
            lines.append("- None")
            continue
        for _, row in subset.iterrows():
            candidate = (
                f"{row['candidate_start_page']}-{row['candidate_end_page']}"
                if row["candidate_start_page"] and row["candidate_end_page"]
                else "unresolved"
            )
            lines.append(
                f"- `{row['paper_id']}` - {row['title']} "
                f"({row['page_count']} pages; `{row['scope_status']}`; candidate {candidate})"
            )
            lines.append(f"  - DOI: `{row['doi'] or 'missing'}`")
            lines.append(f"  - PDF: `{row['pdf_path']}`")
            lines.append(f"  - Action: `{row['manual_action']}` - {row['manual_cut_reason']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    audit_path = Path(args.audit)
    output_path = Path(args.output)
    report_path = Path(args.report)
    audit_df = pd.read_csv(audit_path).fillna("")
    mask = audit_df.apply(
        lambda row: is_springer_family(row.get("doi", ""), row.get("url", "")),
        axis=1,
    )
    springer_df = audit_df[mask].copy()
    priorities = springer_df.apply(cutting_priority, axis=1)
    springer_df["manual_cut_priority"] = [item[0] for item in priorities]
    springer_df["manual_action"] = [item[1] for item in priorities]
    springer_df["manual_cut_reason"] = [item[2] for item in priorities]
    priority_order = pd.Categorical(
        springer_df["manual_cut_priority"],
        categories=["high", "medium", "low"],
        ordered=True,
    )
    springer_df = (
        springer_df.assign(_priority_order=priority_order)
        .sort_values(["_priority_order", "page_count"], ascending=[True, False])
        .drop(columns=["_priority_order"])
    )

    preferred_columns = [
        "manual_cut_priority",
        "manual_action",
        "paper_id",
        "keyword",
        "title",
        "doi",
        "url",
        "venue",
        "pdf_path",
        "page_count",
        "scope_status",
        "candidate_start_page",
        "candidate_end_page",
        "candidate_page_count",
        "outline_match_title",
        "outline_match_score",
        "best_title_page",
        "best_title_page_coverage",
        "text_extraction_status",
        "manual_cut_reason",
        "review_decision",
        "reviewed_start_page",
        "reviewed_end_page",
        "review_notes",
    ]
    available_columns = [column for column in preferred_columns if column in springer_df.columns]
    springer_df = springer_df[available_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    springer_df.to_csv(output_path, index=False)
    write_markdown(springer_df, report_path, output_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(f"Springer Nature-family candidates: {len(springer_df)}")
    print(springer_df["manual_cut_priority"].value_counts().to_string())


if __name__ == "__main__":
    main()
