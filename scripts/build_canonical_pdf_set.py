#!/usr/bin/env python3
"""Build a non-destructive canonical PDF set from a scope audit."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from pypdf import PdfReader, PdfWriter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_page_range(source: Path, destination: Path, start_page: int, end_page: int) -> None:
    reader = PdfReader(str(source))
    if start_page < 1 or end_page > len(reader.pages) or end_page < start_page:
        raise ValueError(
            f"Invalid page range {start_page}-{end_page} for {source} ({len(reader.pages)} pages)"
        )
    writer = PdfWriter()
    for page_number in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_number])
    with destination.open("wb") as stream:
        writer.write(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    audit_df = pd.read_csv(args.audit).fillna("")
    review_df = pd.read_csv(args.reviews).fillna("")
    review_map = review_df.set_index("paper_id").to_dict("index") if len(review_df) else {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for _, row in audit_df.iterrows():
        paper_id = str(row["paper_id"])
        source = Path(str(row["pdf_path"]))
        manifest_row = {
            "paper_id": paper_id,
            "title": row["title"],
            "source_pdf_path": str(source),
            "source_pdf_sha256": row["pdf_sha256"],
            "source_page_count": row["page_count"],
            "scope_status": row["scope_status"],
            "review_decision": "",
            "canonical_start_page": "",
            "canonical_end_page": "",
            "canonical_page_count": "",
            "canonical_pdf_path": "",
            "canonical_pdf_sha256": "",
            "canonical_method": "",
            "build_status": "",
            "build_notes": "",
        }
        if not source.exists():
            manifest_row.update(
                {
                    "build_status": "missing_source_pdf",
                    "build_notes": "No source PDF was available.",
                }
            )
            manifest_rows.append(manifest_row)
            continue

        review = review_map.get(paper_id, {})
        decision = str(review.get("review_decision", ""))
        if decision.startswith(("excluded_", "replacement_required_")):
            destination = output_dir / f"{paper_id}.pdf"
            if destination.exists():
                destination.unlink()
            manifest_row.update(
                {
                    "review_decision": decision,
                    "build_status": decision,
                    "build_notes": str(review.get("review_notes", "")),
                }
            )
            manifest_rows.append(manifest_row)
            continue
        if decision:
            start_page = int(review["reviewed_start_page"])
            end_page = int(review["reviewed_end_page"])
            notes = str(review.get("review_notes", ""))
        elif row["scope_status"] == "single_paper_likely":
            decision = "auto_single_candidate"
            start_page = 1
            end_page = int(row["page_count"])
            notes = "Accepted from conservative single-paper audit; still subject to corpus QA."
        else:
            manifest_row.update(
                {
                    "build_status": "awaiting_scope_review",
                    "build_notes": "A manual boundary decision is required.",
                }
            )
            manifest_rows.append(manifest_row)
            continue

        destination = output_dir / f"{paper_id}.pdf"
        source_page_count = int(row["page_count"])
        reviewed_page_count = end_page - start_page + 1
        source_is_pretrimmed = (
            decision == "approved_chapter"
            and source_page_count == reviewed_page_count
            and not (start_page == 1 and end_page == source_page_count)
        )
        if source_is_pretrimmed:
            shutil.copy2(source, destination)
            method = "pretrimmed_source_copy"
            start_page = 1
            end_page = source_page_count
            notes = (
                f"{notes} The current source PDF is already the verified "
                "single-paper page range, so it was copied without a second extraction."
            )
        elif start_page == 1 and end_page == source_page_count:
            shutil.copy2(source, destination)
            method = "source_copy"
        else:
            extract_page_range(source, destination, start_page, end_page)
            method = "page_range_extract"

        canonical_reader = PdfReader(str(destination))
        expected_pages = end_page - start_page + 1
        if len(canonical_reader.pages) != expected_pages:
            raise RuntimeError(
                f"Canonical page-count mismatch for {paper_id}: "
                f"{len(canonical_reader.pages)} != {expected_pages}"
            )
        manifest_row.update(
            {
                "review_decision": decision,
                "canonical_start_page": start_page,
                "canonical_end_page": end_page,
                "canonical_page_count": expected_pages,
                "canonical_pdf_path": str(destination),
                "canonical_pdf_sha256": sha256_file(destination),
                "canonical_method": method,
                "build_status": "built",
                "build_notes": notes,
            }
        )
        manifest_rows.append(manifest_row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path}")
    print(f"Built {(manifest_df['build_status'] == 'built').sum()}/{len(manifest_df)} canonical PDFs.")
    print(manifest_df["build_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
