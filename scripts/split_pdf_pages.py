#!/usr/bin/env python3
"""Extract an inclusive, 1-based page range into a new PDF.

The source PDF is never modified. The command refuses to overwrite an existing
output file unless --overwrite is supplied explicitly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def split_pdf(
    input_path: Path,
    output_path: Path,
    start_page: int,
    end_page: int,
    overwrite: bool = False,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite only after checking the target."
        )

    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    if start_page < 1:
        raise ValueError("start-page must be at least 1")
    if end_page < start_page:
        raise ValueError("end-page must be greater than or equal to start-page")
    if end_page > total_pages:
        raise ValueError(
            f"end-page {end_page} exceeds the source PDF page count ({total_pages})"
        )

    writer = PdfWriter()
    for page_index in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_index])
    if reader.metadata:
        metadata = {
            str(key): str(value)
            for key, value in reader.metadata.items()
            if key and value is not None
        }
        writer.add_metadata(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("wb") as stream:
            writer.write(stream)
        written_reader = PdfReader(str(temporary_path))
        expected_pages = end_page - start_page + 1
        if len(written_reader.pages) != expected_pages:
            raise RuntimeError(
                f"Written PDF has {len(written_reader.pages)} pages; expected {expected_pages}"
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return end_page - start_page + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save an inclusive 1-based page range as a new PDF."
    )
    parser.add_argument("--input", required=True, help="Source PDF path")
    parser.add_argument("--output", required=True, help="New PDF path")
    parser.add_argument("--start-page", type=int, required=True)
    parser.add_argument("--end-page", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    page_count = split_pdf(
        input_path=Path(args.input),
        output_path=Path(args.output),
        start_page=args.start_page,
        end_page=args.end_page,
        overwrite=args.overwrite,
    )
    print(f"Wrote {args.output} ({page_count} pages).")
    print(f"Source preserved: {args.input}")


if __name__ == "__main__":
    main()
