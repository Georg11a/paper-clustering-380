#!/usr/bin/env python3
"""Materialize one analysis-unit PDF per new Page 3 record.

Container PDFs are sliced to the reviewed chapter boundary.  A scanned
single-paper PDF is OCRed separately by ``ocrmypdf`` after this script runs.
The Drive-synced originals are never modified.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from pypdf import PdfReader, PdfWriter


# Inclusive, one-based boundaries verified against title pages and end matter.
BOUNDARIES: dict[str, tuple[int, int]] = {
    "01276f979d0b": (307, 319),
    "7f559689c730": (472, 477),
    "c55894a018e8": (133, 151),
    "31314745e651": (40, 47),
    "d98d3e1b6edc": (117, 140),
    "2d47b6d1002a": (145, 163),
    "0c97b40952ed": (259, 267),
}


def extract_pages(source: Path, target: Path, start: int, end: int) -> None:
    reader = PdfReader(str(source))
    if not (1 <= start <= end <= len(reader.pages)):
        raise ValueError(f"Invalid page range {start}-{end} for {source}")
    writer = PdfWriter()
    for page_index in range(start - 1, end):
        writer.add_page(reader.pages[page_index])
    with target.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Candidate metadata CSV")
    parser.add_argument("--source-dir", required=True, help="Drive-synced originals")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    candidates = pd.read_csv(args.input).fillna("")
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        paper_id = str(row["paper_id"])
        source = source_dir / f"{paper_id}.pdf"
        target = output_dir / f"{paper_id}.pdf"
        if paper_id in BOUNDARIES:
            start, end = BOUNDARIES[paper_id]
            extract_pages(source, target, start, end)
            mode = "chapter_extracted"
        else:
            shutil.copy2(source, target)
            start, end = 1, len(PdfReader(str(source)).pages)
            mode = "single_pdf_copied"
        rows.append(
            {
                "paper_id": paper_id,
                "source_pdf": str(source),
                "analysis_pdf": str(target),
                "materialization": mode,
                "source_start_page": start,
                "source_end_page": end,
                "analysis_page_count": len(PdfReader(str(target)).pages),
                "ocr_required": paper_id == "acbc78e3036a",
            }
        )

    manifest = pd.DataFrame(rows).sort_values("paper_id")
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(manifest["materialization"].value_counts().to_string())
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
