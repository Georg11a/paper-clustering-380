#!/usr/bin/env python3
"""Match unrenamed PDFs in Downloads against missing accepted-paper records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


ID_RE = re.compile(r"^[0-9a-f]{12}$", re.I)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
STOP = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
    "of", "on", "or", "the", "to", "toward", "towards", "using", "with",
}


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(value.split())


def compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", norm(value))


def clean_doi(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    return value.rstrip(".,;)]}")


def inspect_pdf(path_string: str) -> dict[str, object]:
    path = Path(path_string)
    record: dict[str, object] = {
        "path": str(path),
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": "",
        "metadata_title": "",
        "first_pages_text": "",
        "dois": "",
        "error": "",
    }
    try:
        reader = PdfReader(str(path), strict=False)
        meta = reader.metadata or {}
        record["metadata_title"] = str(meta.get("/Title", "") or "")
        pages = []
        for page in reader.pages[:3]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        record["first_pages_text"] = " ".join(text.split())[:50000]
        record["dois"] = ";".join(sorted({clean_doi(x) for x in DOI_RE.findall(text)}))
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        record["sha256"] = h.hexdigest()
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", default="/Users/baiyixin/Downloads")
    parser.add_argument("--accepted", default="/Users/baiyixin/Downloads/accepted_papers.csv")
    parser.add_argument("--current", default="data/final_advancing_list_expanded_459.csv")
    parser.add_argument("--outdir", default="tmp/pdf_inventory_20260811")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--recent-days",
        type=float,
        default=3,
        help=(
            "Always scan top-level Downloads and the paper corpus folders; "
            "for other nested folders, include only files modified within "
            "this many days. Use 0 to scan every nested PDF."
        ),
    )
    args = parser.parse_args()

    downloads = Path(args.downloads)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    inventory_path = outdir / "downloads_pdf_inventory.csv"
    cutoff = time.time() - args.recent_days * 86400
    paths = sorted(
        p
        for p in downloads.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".pdf"
        and (
            args.recent_days <= 0
            or any(part.startswith("pdf_386") for part in p.parts)
            or p.stat().st_mtime >= cutoff
        )
    )
    cached: dict[str, dict[str, object]] = {}
    if inventory_path.exists():
        old = pd.read_csv(inventory_path, dtype=str).fillna("")
        cached = {row["path"]: row.to_dict() for _, row in old.iterrows()}
    todo = [p for p in paths if str(p) not in cached]
    print(f"PDFs={len(paths)} cached={len(paths)-len(todo)} inspect={len(todo)}", flush=True)
    if todo:
        # Thread workers avoid macOS sandbox semaphore restrictions while still
        # overlapping PDF I/O and decompression effectively for this audit.
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(inspect_pdf, str(path)): path for path in todo}
            for i, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                cached[record["path"]] = record
                if i % 100 == 0 or i == len(todo):
                    pd.DataFrame(cached.values()).to_csv(inventory_path, index=False)
                    print(f"inspected {i}/{len(todo)}", flush=True)
    inventory = pd.DataFrame(cached.values()).fillna("")
    inventory.to_csv(inventory_path, index=False)

    accepted = pd.read_csv(args.accepted, dtype=str).fillna("")
    accepted = accepted[accepted["paper_id"].map(lambda x: bool(ID_RE.fullmatch(x)))].copy()
    accepted = accepted.drop_duplicates("paper_id")
    current_ids = set(pd.read_csv(args.current, dtype=str)["paper_id"].astype(str))
    missing = accepted.loc[~accepted["paper_id"].isin(current_ids)].copy()

    pdf_evidence = []
    for _, pdf in inventory.iterrows():
        filename = str(pdf["filename"])
        raw = " ".join(
            [Path(filename).stem, str(pdf["metadata_title"]), str(pdf["first_pages_text"])[:12000]]
        )
        raw_norm = norm(raw)
        pdf_evidence.append(
            (
                pdf,
                filename,
                raw_norm,
                re.sub(r"[^a-z0-9]", "", raw_norm),
                {clean_doi(x) for x in str(pdf["dois"]).split(";") if x},
            )
        )

    records = []
    for _, paper in missing.iterrows():
        pid = str(paper["paper_id"])
        title = str(paper["title"])
        title_norm = norm(title)
        title_compact = compact(title)
        tokens = [token for token in title_norm.split() if token not in STOP and len(token) > 2]
        target_doi = clean_doi(paper.get("doi", ""))
        for pdf, filename, evidence_norm, evidence_compact, pdf_dois in pdf_evidence:
            if filename.lower() == f"{pid}.pdf":
                score, reason = 1.0, "exact_paper_id_filename"
            else:
                if target_doi and target_doi in pdf_dois:
                    score, reason = 1.0, "exact_doi"
                elif len(title_compact) >= 18 and title_compact in evidence_compact:
                    score, reason = 0.99, "exact_normalized_title"
                elif tokens:
                    present = sum(token in evidence_norm for token in tokens)
                    coverage = present / len(tokens)
                    early_present = present / len(tokens)
                    score = 0.7 * early_present + 0.3 * coverage
                    reason = "high_title_token_coverage"
                    if score < 0.84 or present < min(4, len(tokens)):
                        continue
                else:
                    continue
            records.append(
                {
                    "paper_id": pid,
                    "title": title,
                    "doi": paper.get("doi", ""),
                    "match_score": round(float(score), 4),
                    "match_reason": reason,
                    "pdf_path": pdf["path"],
                    "pdf_metadata_title": pdf["metadata_title"],
                    "pdf_dois": pdf["dois"],
                    "pdf_sha256": pdf["sha256"],
                }
            )
    matches = pd.DataFrame(records)
    if len(matches):
        matches = matches.sort_values(["paper_id", "match_score"], ascending=[True, False])
    matches.to_csv(outdir / "missing_pdf_match_candidates.csv", index=False)
    missing.to_csv(outdir / "missing_before_rescan.csv", index=False)
    print(f"missing={len(missing)} candidate_rows={len(matches)} candidate_ids={matches['paper_id'].nunique() if len(matches) else 0}")


if __name__ == "__main__":
    main()
