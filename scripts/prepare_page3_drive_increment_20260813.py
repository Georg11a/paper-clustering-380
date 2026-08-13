#!/usr/bin/env python3
"""Stage the newly found PDFs for the coauthor Drive and Page 3 processing.

The source CSV is the reviewed missing-PDF tracker.  Exact ``Found == Y``
records are analysis candidates.  Non-English downloads are archived to Drive
but intentionally excluded from the English Page 3 corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_pdf(paper_id: str, downloads: Path) -> Path:
    candidates = [
        downloads / f"{paper_id}.pdf",
        downloads / "lit-review-papers" / f"{paper_id}.pdf",
        downloads / "lit-review-papers" / f"x{paper_id}.pdf",
        downloads / f"{paper_id}_Russian.pdf",
    ]
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(f"No PDF found for {paper_id}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--downloads", default="/Users/baiyixin/Downloads")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    tracker_path = Path(args.tracker)
    downloads = Path(args.downloads)
    outdir = Path(args.outdir)
    upload_dir = outdir / "upload_ready"
    analysis_dir = outdir / "analysis_candidates_local_preupload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    tracker = pd.read_csv(tracker_path).fillna("")
    found = tracker["Found"].astype(str).str.strip()
    exact_y = tracker[found.eq("Y")].copy()
    non_english = tracker[found.str.startswith("Y (")].copy()

    rows: list[dict[str, object]] = []
    for corpus_role, frame in [
        ("page3_analysis_candidate", exact_y),
        ("drive_archive_non_english", non_english),
    ]:
        for _, row in frame.iterrows():
            paper_id = str(row["paper_id"])
            source = resolve_pdf(paper_id, downloads)
            target = upload_dir / f"{paper_id}.pdf"
            shutil.copy2(source, target)
            if corpus_role == "page3_analysis_candidate":
                shutil.copy2(source, analysis_dir / f"{paper_id}.pdf")
            try:
                pages = len(PdfReader(str(source)).pages)
            except Exception:
                pages = ""
            rows.append(
                {
                    "paper_id": paper_id,
                    "title": row.get("title", ""),
                    "found_status": row.get("Found", ""),
                    "corpus_role": corpus_role,
                    "source_path": str(source),
                    "upload_filename": target.name,
                    "bytes": target.stat().st_size,
                    "page_count": pages,
                    "sha256": sha256(target),
                    "drive_file_id": "",
                    "drive_download_url": "",
                }
            )

    upload_manifest = pd.DataFrame(rows).sort_values(
        ["corpus_role", "paper_id"]
    )
    upload_manifest.to_csv(outdir / "drive_upload_manifest.csv", index=False)
    exact_y.to_csv(outdir / "page3_analysis_candidates.csv", index=False)
    non_english.to_csv(outdir / "drive_archive_non_english.csv", index=False)

    print(f"Page 3 analysis candidates: {len(exact_y)}")
    print(f"Drive-only non-English archive: {len(non_english)}")
    print(f"Upload files staged: {len(upload_manifest)}")
    print(f"Upload bytes: {int(upload_manifest['bytes'].sum())}")
    print(f"Wrote {outdir / 'drive_upload_manifest.csv'}")


if __name__ == "__main__":
    main()
