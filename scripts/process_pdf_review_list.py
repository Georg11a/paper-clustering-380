#!/usr/bin/env python3
"""Process a PDF review list into a verified paper-ID-named folder."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from split_pdf_pages import split_pdf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_page(value: object) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError("Missing required candidate page")
    return int(float(value))


def write_readme(manifest: pd.DataFrame, output_path: Path, input_path: Path) -> None:
    action_counts = manifest["processing_action"].value_counts().to_dict()
    lines = [
        "# Processed Springer Nature PDFs",
        "",
        f"- Source review list: `{input_path}`",
        f"- Total records: {len(manifest)}",
        f"- Successfully written: {(manifest['status'] == 'success').sum()}",
        f"- Analysis ready: {manifest['analysis_ready'].astype(bool).sum()}",
        f"- Scope unresolved: {(manifest['analysis_ready'] == False).sum()}",
        "",
        "## Processing actions",
        "",
    ]
    for action, count in sorted(action_counts.items()):
        lines.append(f"- `{action}`: {count}")
    unresolved = manifest[manifest["analysis_ready"] == False]
    lines.extend(["", "## Excluded pending scope decision", ""])
    if unresolved.empty:
        lines.append("- None")
    else:
        for _, row in unresolved.iterrows():
            lines.append(f"- `{row['paper_id']}` - {row['title']}: {row['notes']}")
    lines.extend(
        [
            "",
            "All output PDFs are named `<paper_id>.pdf`. Source PDFs were preserved.",
            "Do not include rows with `analysis_ready=false` in text extraction or embeddings.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manual-cuts-dir", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    manual_cuts_dir = Path(args.manual_cuts_dir) if args.manual_cuts_dir else None
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    review_df = pd.read_csv(input_path).fillna("")
    if review_df["paper_id"].astype(str).duplicated().any():
        duplicates = review_df.loc[
            review_df["paper_id"].astype(str).duplicated(), "paper_id"
        ].tolist()
        raise ValueError(f"Duplicate paper IDs in review list: {duplicates}")

    manifest_rows = []
    for _, row in review_df.iterrows():
        paper_id = str(row["paper_id"])
        source_path = Path(str(row["pdf_path"]))
        output_path = output_dir / f"{paper_id}.pdf"
        manual_path = manual_cuts_dir / f"{paper_id}.pdf" if manual_cuts_dir else None
        requested_action = str(row.get("manual_action", ""))
        analysis_ready = requested_action != "scope_decision_required"
        notes = ""

        if manual_path and manual_path.exists():
            shutil.copy2(manual_path, output_path)
            processing_action = "verified_manual_cut"
            source_used = manual_path
        elif requested_action == "cut_candidate":
            start_page = integer_page(row.get("candidate_start_page"))
            end_page = integer_page(row.get("candidate_end_page"))
            split_pdf(source_path, output_path, start_page, end_page)
            processing_action = "candidate_range_extract"
            source_used = source_path
        elif requested_action == "scope_decision_required":
            shutil.copy2(source_path, output_path)
            processing_action = "full_container_copy_pending_scope"
            source_used = source_path
            notes = (
                "The corpus record appears to represent a full book. "
                "Retained intact but excluded from downstream analysis pending a corpus-unit decision."
            )
        else:
            shutil.copy2(source_path, output_path)
            processing_action = "verified_source_copy"
            source_used = source_path

        reader = PdfReader(str(output_path))
        output_pages = len(reader.pages)
        if output_pages < 1:
            raise RuntimeError(f"Written PDF is empty: {output_path}")
        if processing_action in {"verified_manual_cut", "candidate_range_extract"}:
            expected_pages = integer_page(row.get("candidate_end_page")) - integer_page(
                row.get("candidate_start_page")
            ) + 1
            if output_pages != expected_pages:
                raise RuntimeError(
                    f"{paper_id}: output has {output_pages} pages; expected {expected_pages}"
                )

        manifest_rows.append(
            {
                "paper_id": paper_id,
                "keyword": row.get("keyword", ""),
                "title": row.get("title", ""),
                "doi": row.get("doi", ""),
                "requested_action": requested_action,
                "processing_action": processing_action,
                "source_used": str(source_used),
                "output_pdf": str(output_path),
                "output_pages": output_pages,
                "output_sha256": sha256_file(output_path),
                "analysis_ready": analysis_ready,
                "status": "success",
                "notes": notes,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "processing_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    write_readme(manifest_df, output_dir / "README.md", input_path)
    print(f"Wrote {len(manifest_df)} PDFs to {output_dir}")
    print(f"Analysis ready: {manifest_df['analysis_ready'].astype(bool).sum()}/{len(manifest_df)}")
    print(manifest_df["processing_action"].value_counts().to_string())


if __name__ == "__main__":
    main()
