#!/usr/bin/env python3
"""Finalize the Downloads rescan and build the 464-PDF / 459-analysis corpus."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


REPO = Path(__file__).resolve().parents[1]
ACCEPTED = Path("/Users/baiyixin/Downloads/accepted_papers.csv")
CURRENT = REPO / "data/final_advancing_list_expanded_459.csv"
CURRENT_MANIFEST = REPO / "data/expanded_corpus_manifest_459_20260811.csv"
REVIEWS = REPO / "data/pdf_scope_reviews.csv"
CORPUS = Path(
    "/Users/baiyixin/Documents/Survey - design knowledge/"
    "expanded_pdf_corpus_464_20260811"
)
PIPELINE = Path(
    "/Users/baiyixin/Documents/Survey - design knowledge/"
    "expanded_pipeline_464_20260811"
)
OUTPUT = REPO / "data/final_advancing_list_expanded_464.csv"
READY = REPO / "data/expanded_analysis_ready_459_rescanned_20260811.csv"
MANIFEST = REPO / "data/expanded_corpus_manifest_464_20260811.csv"

NEW_SOURCES = {
    "1aecb7940a63": Path("/Users/baiyixin/Downloads/brazier1997.pdf"),
    "38b608e72e2d": Path(
        "/Users/baiyixin/Downloads/"
        "A_BOUNDARY_OBJECT_FOR_MAPPING_COMPARING_AND_INTEGR.pdf"
    ),
    "781628bf9a48": Path("/Users/baiyixin/Downloads/347642.347652.pdf"),
    "a285258b5545": Path(
        "/Users/baiyixin/Downloads/Extended_General_Design_Theory_1st_Report.pdf"
    ),
    "55dc4ef842a6": Path(
        "/Users/baiyixin/Downloads/"
        "IMPLEMENTATION_OF_THE_BASIC_PRINCIPLES_OF_FORMING_.pdf"
    ),
}
ID_RE = re.compile(r"^[0-9a-f]{12}$", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_best_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("paper_id", sort=False):
        scores = group.apply(
            lambda row: sum(bool(str(value).strip()) for value in row), axis=1
        )
        best = group.loc[scores.idxmax()].copy()
        for column in group.columns:
            if not str(best[column]).strip():
                values = [str(value).strip() for value in group[column] if str(value).strip()]
                if values:
                    best[column] = values[0]
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def extract_text(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path), strict=False)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages).replace("\x00", " ").strip()
    return text, len(reader.pages)


def ensure_link(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Refusing to replace existing corpus path: {destination}")
    destination.symlink_to(source)


def main() -> None:
    accepted = pd.read_csv(ACCEPTED, dtype=str).fillna("")
    accepted = accepted[accepted["paper_id"].map(lambda value: bool(ID_RE.fullmatch(value)))]
    accepted = select_best_metadata(accepted).set_index("paper_id", drop=False)
    current = pd.read_csv(CURRENT, dtype=str).fillna("")
    current_manifest = pd.read_csv(CURRENT_MANIFEST, dtype=str).fillna("")
    current_source = current_manifest.set_index("paper_id")["source_pdf_path"].to_dict()

    metadata = pd.concat(
        [current, accepted.loc[list(NEW_SOURCES)].reset_index(drop=True)],
        ignore_index=True,
    )
    if len(metadata) != 464 or metadata["paper_id"].nunique() != 464:
        raise RuntimeError("Rescanned metadata must contain 464 unique accepted records.")

    reviews = pd.read_csv(REVIEWS, dtype=str).fillna("")
    decisions = reviews.drop_duplicates("paper_id", keep="last").set_index("paper_id")[
        "review_decision"
    ].to_dict()
    excluded = {
        paper_id
        for paper_id, decision in decisions.items()
        if decision.startswith("excluded_") or decision.startswith("replacement_required_")
    }
    expected_excluded = {
        "12029fe2805e",
        "c1db52f69aae",
        "c3b41755b43d",
        "a285258b5545",
        "55dc4ef842a6",
    }
    active_excluded = set(metadata["paper_id"]) & excluded
    if active_excluded != expected_excluded:
        raise RuntimeError(
            f"Unexpected active exclusions: {sorted(active_excluded)}; "
            f"expected {sorted(expected_excluded)}"
        )

    corpus_rows = []
    canonical_dir = PIPELINE / "canonical_text"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    old_canonical = Path(
        "/Users/baiyixin/Documents/Survey - design knowledge/"
        "expanded_pipeline_459_20260811/canonical_text"
    )
    for paper_id in metadata["paper_id"].astype(str):
        source = (
            NEW_SOURCES[paper_id]
            if paper_id in NEW_SOURCES
            else Path(current_source[paper_id])
        )
        if not source.exists():
            raise RuntimeError(f"Missing source PDF: {source}")
        destination = CORPUS / f"{paper_id}.pdf"
        ensure_link(destination, source)
        if paper_id in NEW_SOURCES:
            text, page_count = extract_text(source)
            (canonical_dir / f"{paper_id}.txt").write_text(text, encoding="utf-8")
        else:
            old_text = old_canonical / f"{paper_id}.txt"
            if old_text.exists():
                ensure_link(canonical_dir / f"{paper_id}.txt", old_text)
            else:
                # Excluded source documents were intentionally absent from the
                # prior English canonical-text set; preserve an audit extract.
                text, _ = extract_text(source)
                (canonical_dir / f"{paper_id}.txt").write_text(text, encoding="utf-8")
            page_count = int(
                current_manifest.loc[
                    current_manifest["paper_id"].eq(paper_id), "page_count"
                ].iloc[0]
            )
        corpus_rows.append(
            {
                "paper_id": paper_id,
                "source_pdf_path": str(source),
                "corpus_pdf_path": str(destination),
                "canonical_text_path": str(canonical_dir / f"{paper_id}.txt"),
                "pdf_sha256": sha256(source),
                "page_count": page_count,
                "review_decision": decisions.get(paper_id, "included_prior_review"),
                "analysis_eligible": paper_id not in active_excluded,
                "corpus_role": (
                    "rescan_increment_20260811"
                    if paper_id in NEW_SOURCES
                    else "prior_459"
                ),
            }
        )

    metadata["fulltext_flag"] = "TRUE"
    ready = metadata.loc[~metadata["paper_id"].isin(active_excluded)].copy()
    if len(ready) != 459 or ready["paper_id"].nunique() != 459:
        raise RuntimeError("Analysis-ready corpus must contain 459 unique record IDs.")
    path_map = {row["paper_id"]: row["canonical_text_path"] for row in corpus_rows}
    ready["canonical_text_path"] = ready["paper_id"].map(path_map)
    new_ready = ready.loc[ready["paper_id"].isin({"1aecb7940a63", "38b608e72e2d", "781628bf9a48"})].copy()
    if len(new_ready) != 3:
        raise RuntimeError("Expected exactly three newly eligible English papers.")

    metadata.to_csv(OUTPUT, index=False)
    ready.to_csv(READY, index=False)
    new_ready.to_csv(PIPELINE / "new_analysis_ready_3.csv", index=False)
    pd.DataFrame(corpus_rows).to_csv(MANIFEST, index=False)
    print(f"Wrote {OUTPUT}: {len(metadata)} PDF-backed records")
    print(f"Wrote {READY}: {len(ready)} English analysis records")
    print(f"Wrote {PIPELINE / 'new_analysis_ready_3.csv'}")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
