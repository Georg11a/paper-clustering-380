#!/usr/bin/env python3
"""Prepare a clean Page 3 replacement record for a PDF text-encoding defect."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


PAPER_ID = "d24e6cf8ae6a"
CLEAN_TITLE = (
    "Physical Design Cognition: A Non-symbolic Formalization for Performing "
    "Design Knowledge"
)
CLEAN_ABSTRACT = (
    "This paper frames design knowledge as formalizable physical action for "
    "developing embodied computational design skills that can more fully exploit "
    "current and future digital fabrication prototyping methods. Digitally "
    "integrated prototyping tools reveal the physicality of cognition in "
    "computational design activity; however, because current theories of design "
    "knowledge define cognition as a mental process, physical computation design "
    "skills remain underdeveloped. We identify symbolic formalization as the root "
    "of this problem. We present a non-symbolic action-based notation drawing from "
    "embodied cognition as an alternative model for design cognition. Designerly "
    "knowledge is discussed in terms of reflective action and epistemic action."
)


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00ad", "").replace("\ufffd", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, dtype=str).fillna("")
    chunks = pd.read_csv(args.chunks, dtype=str).fillna("")

    selected_metadata = metadata.loc[metadata["paper_id"] == PAPER_ID].copy()
    selected_chunks = chunks.loc[chunks["paper_id"] == PAPER_ID].copy()
    if len(selected_metadata) != 1:
        raise ValueError(f"Expected one metadata row for {PAPER_ID}")
    if len(selected_chunks) != 13:
        raise ValueError(f"Expected 13 chunks for {PAPER_ID}")

    selected_metadata.loc[:, "title"] = CLEAN_TITLE
    selected_metadata.loc[:, "canonical_title"] = CLEAN_TITLE
    selected_metadata.loc[:, "abstract"] = CLEAN_ABSTRACT
    selected_metadata.loc[:, "notes"] = (
        "Title and abstract normalized from the visually verified source PDF; "
        "replacement characters, non-breaking spaces, and soft hyphens removed."
    )

    old_title = str(selected_chunks.iloc[0]["chunk"]).split(" [SEP] ", 1)[0]
    repaired_chunks = []
    for _, row in selected_chunks.iterrows():
        chunk_index = int(row["chunk_index"])
        chunk = str(row["chunk"])
        if " [SEP] " in chunk:
            _, body = chunk.split(" [SEP] ", 1)
        else:
            body = chunk
        body = CLEAN_ABSTRACT if chunk_index == 0 else clean_text(body)
        repaired_chunks.append(
            {"paper_id": PAPER_ID, "chunk_index": chunk_index, "chunk": f"{CLEAN_TITLE} [SEP] {body}"}
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metadata_path = outdir / "metadata_1.csv"
    chunks_path = outdir / "R_cent_neutral_chunks_1.csv"
    order_path = outdir / "paper_order_1.csv"
    selected_metadata.to_csv(metadata_path, index=False)
    pd.DataFrame(repaired_chunks).to_csv(chunks_path, index=False)
    pd.DataFrame({"row_index": [0], "paper_id": [PAPER_ID]}).to_csv(order_path, index=False)

    checks = {
        "paper_id": PAPER_ID,
        "old_title": old_title,
        "clean_title": CLEAN_TITLE,
        "chunk_count": len(repaired_chunks),
        "metadata_replacement_characters": int(
            selected_metadata.astype(str).apply(lambda col: col.str.count("\ufffd").sum()).sum()
        ),
        "chunk_replacement_characters": sum(item["chunk"].count("\ufffd") for item in repaired_chunks),
        "chunk_non_breaking_spaces": sum(item["chunk"].count("\u00a0") for item in repaired_chunks),
        "chunk_soft_hyphens": sum(item["chunk"].count("\u00ad") for item in repaired_chunks),
    }
    if any(checks[key] for key in checks if key.endswith(("characters", "spaces", "hyphens"))):
        raise ValueError(f"Encoding repair validation failed: {checks}")
    (outdir / "repair_manifest.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
