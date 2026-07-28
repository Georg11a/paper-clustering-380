#!/usr/bin/env python3
"""Freeze a global PDF/text manifest without modifying source PDFs."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pypdf_text(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return normalize_text("\n\n".join(pages)), len(reader.pages)


def paragraph_count(text: str) -> int:
    rough = re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z][a-z])", text)
    return sum(1 for value in rough if len(normalize_text(value)) >= 80)


def load_index(path: str | None, key: str = "paper_id") -> dict[str, dict[str, object]]:
    if not path:
        return {}
    frame = pd.read_csv(path).fillna("")
    return {str(row[key]): row.to_dict() for _, row in frame.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/final_advancing_list.csv")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--reviews", default="data/pdf_scope_reviews.csv")
    parser.add_argument("--overrides", default="data/canonical_metadata_overrides.csv")
    parser.add_argument("--ocr-pdf-dir", default="outputs/batch1/ocr_pdfs")
    parser.add_argument("--ocr-text-dir", default="outputs/batch1/ocr_text")
    parser.add_argument("--text-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ready-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-ready", type=int, default=None)
    args = parser.parse_args()

    source = pd.read_csv(args.input).fillna("")
    if source["paper_id"].astype(str).duplicated().any():
        duplicates = source.loc[source["paper_id"].astype(str).duplicated(), "paper_id"].tolist()
        raise ValueError(f"Duplicate paper IDs in input: {duplicates[:20]}")

    reviews = load_index(args.reviews)
    overrides = load_index(args.overrides)
    pdf_dir = Path(args.pdf_dir)
    ocr_pdf_dir = Path(args.ocr_pdf_dir)
    ocr_text_dir = Path(args.ocr_text_dir)
    text_dir = Path(args.text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    ready_rows: list[dict[str, object]] = []
    for _, source_row in source.iterrows():
        record = source_row.to_dict()
        paper_id = str(record["paper_id"])
        review = reviews.get(paper_id, {})
        decision = str(review.get("review_decision", "")).strip()
        override = overrides.get(paper_id, {})
        canonical_title = str(override.get("canonical_title", "")).strip() or str(record.get("title", "")).strip()
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        manifest = {
            "paper_id": paper_id,
            "source_title": record.get("title", ""),
            "canonical_title": canonical_title,
            "keyword": record.get("keyword", ""),
            "review_decision": decision,
            "analysis_unit_type": "book" if paper_id == "edde6dc75b19" else "paper",
            "source_pdf_path": str(pdf_path) if pdf_path.exists() else "",
            "source_pdf_sha256": "",
            "canonical_pdf_path": "",
            "canonical_pdf_sha256": "",
            "page_count": "",
            "text_source": "",
            "canonical_text_path": "",
            "canonical_text_sha256": "",
            "canonical_text_chars": 0,
            "canonical_paragraph_count": 0,
            "abstract_present": bool(str(record.get("abstract", "")).strip()),
            "analysis_ready": False,
            "status": "",
            "notes": str(review.get("review_notes", "")).strip(),
        }

        if decision.startswith(("excluded_", "replacement_required_")):
            manifest["status"] = decision
            manifest_rows.append(manifest)
            continue
        if not pdf_path.exists():
            manifest["status"] = "missing_pdf"
            manifest_rows.append(manifest)
            continue

        source_hash = sha256_file(pdf_path)
        ocr_text_path = ocr_text_dir / f"{paper_id}.txt"
        ocr_pdf_path = ocr_pdf_dir / f"{paper_id}.pdf"
        if ocr_text_path.exists() and ocr_pdf_path.exists():
            text = normalize_text(ocr_text_path.read_text(encoding="utf-8", errors="replace"))
            canonical_pdf = ocr_pdf_path
            text_source = "ocrmypdf-tesseract-eng"
            page_count_value = len(PdfReader(str(ocr_pdf_path)).pages)
        else:
            text, page_count_value = extract_pypdf_text(pdf_path)
            canonical_pdf = pdf_path
            text_source = "pypdf"

        count = paragraph_count(text)
        if len(text) < 500 or count == 0:
            manifest.update(
                {
                    "source_pdf_sha256": source_hash,
                    "canonical_pdf_path": str(canonical_pdf),
                    "canonical_pdf_sha256": sha256_file(canonical_pdf),
                    "page_count": page_count_value,
                    "text_source": text_source,
                    "canonical_text_chars": len(text),
                    "canonical_paragraph_count": count,
                    "status": "text_qa_failed",
                }
            )
            manifest_rows.append(manifest)
            continue

        output_text = text_dir / f"{paper_id}.txt"
        output_text.write_text(text + "\n", encoding="utf-8")
        text_hash = sha256_bytes((text + "\n").encode("utf-8"))
        manifest.update(
            {
                "source_pdf_sha256": source_hash,
                "canonical_pdf_path": str(canonical_pdf),
                "canonical_pdf_sha256": sha256_file(canonical_pdf),
                "page_count": page_count_value,
                "text_source": text_source,
                "canonical_text_path": str(output_text),
                "canonical_text_sha256": text_hash,
                "canonical_text_chars": len(text),
                "canonical_paragraph_count": count,
                "analysis_ready": True,
                "status": "analysis_ready",
            }
        )
        ready_record = dict(record)
        ready_record.update(manifest)
        ready_rows.append(ready_record)
        manifest_rows.append(manifest)

    manifest_frame = pd.DataFrame(manifest_rows)
    ready_frame = pd.DataFrame(ready_rows)
    if args.expected_ready is not None and len(ready_frame) != args.expected_ready:
        counts = manifest_frame["status"].value_counts().to_dict()
        raise RuntimeError(
            f"Expected {args.expected_ready} analysis-ready documents, found {len(ready_frame)}; "
            f"status counts: {counts}"
        )

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    manifest_frame.to_csv(args.manifest, index=False)
    ready_frame.to_csv(args.ready_output, index=False)
    counts = manifest_frame["status"].value_counts()
    report_lines = [
        "# Global Canonical Text Freeze",
        "",
        f"- Source records: {len(source)}",
        f"- Analysis-ready documents: {len(ready_frame)}",
        f"- Unique ready paper IDs: {ready_frame['paper_id'].nunique() if len(ready_frame) else 0}",
        f"- Global manifest: `{args.manifest}`",
        f"- Frozen input: `{args.ready_output}`",
        f"- Canonical text directory: `{args.text_dir}`",
        "",
        "## Status counts",
        "",
    ]
    report_lines.extend(f"- `{status}`: {count}" for status, count in counts.items())
    report_lines.extend(
        [
            "",
            "## Text-source counts",
            "",
        ]
    )
    if len(ready_frame):
        report_lines.extend(
            f"- `{source_name}`: {count}"
            for source_name, count in ready_frame["text_source"].value_counts().items()
        )
    Path(args.report).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.manifest}")
    print(f"Wrote {args.ready_output}")
    print(f"Wrote {args.report}")
    print(f"Analysis-ready: {len(ready_frame)}/{len(source)}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
