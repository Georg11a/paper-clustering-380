#!/usr/bin/env python3
"""Replace selected records in a frozen Page 3 neutral corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-chunks", required=True)
    parser.add_argument("--base-vectors", required=True)
    parser.add_argument("--base-metadata", required=True)
    parser.add_argument("--replacement-chunks", required=True)
    parser.add_argument("--replacement-vectors", required=True)
    parser.add_argument("--replacement-metadata", required=True)
    parser.add_argument("--expected-papers", type=int, required=True)
    parser.add_argument("--expected-replacements", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    base_chunks = pd.read_csv(args.base_chunks).fillna("")
    replacement_chunks = pd.read_csv(args.replacement_chunks).fillna("")
    order = base_chunks["paper_id"].astype(str).drop_duplicates().tolist()
    replacement_order = (
        replacement_chunks["paper_id"].astype(str).drop_duplicates().tolist()
    )
    if len(order) != args.expected_papers:
        raise ValueError(f"Expected {args.expected_papers} papers, found {len(order)}")
    if len(replacement_order) != args.expected_replacements:
        raise ValueError(
            f"Expected {args.expected_replacements} replacements, "
            f"found {len(replacement_order)}"
        )
    missing = sorted(set(replacement_order) - set(order))
    if missing:
        raise ValueError(f"Replacement paper IDs missing from base corpus: {missing}")

    base_vectors = np.load(args.base_vectors)
    replacement_vectors = np.load(args.replacement_vectors)
    if base_vectors.shape[0] != len(order):
        raise ValueError(f"Base vector/order mismatch: {base_vectors.shape} vs {len(order)}")
    if replacement_vectors.shape != (len(replacement_order), base_vectors.shape[1]):
        raise ValueError(
            f"Unexpected replacement vector shape {replacement_vectors.shape}"
        )

    replacement_index = {paper_id: i for i, paper_id in enumerate(replacement_order)}
    combined_vectors = base_vectors.copy()
    for row_index, paper_id in enumerate(order):
        if paper_id in replacement_index:
            combined_vectors[row_index] = replacement_vectors[replacement_index[paper_id]]

    required = ["paper_id", "chunk_index", "chunk"]
    replacement_chunks = replacement_chunks[required].copy()
    replacement_chunks["paper_id"] = replacement_chunks["paper_id"].astype(str)
    base_chunks = base_chunks[required].copy()
    base_chunks["paper_id"] = base_chunks["paper_id"].astype(str)
    chunks_by_id = {
        paper_id: frame.copy()
        for paper_id, frame in base_chunks.groupby("paper_id", sort=False)
    }
    for paper_id, frame in replacement_chunks.groupby("paper_id", sort=False):
        chunks_by_id[paper_id] = frame.copy()
    combined_chunks = pd.concat([chunks_by_id[paper_id] for paper_id in order], ignore_index=True)
    if combined_chunks["paper_id"].drop_duplicates().tolist() != order:
        raise ValueError("Combined chunks no longer match the frozen paper order")

    base_metadata = pd.read_csv(args.base_metadata, dtype=str).fillna("")
    replacement_metadata = pd.read_csv(args.replacement_metadata, dtype=str).fillna("")
    base_metadata = base_metadata.drop_duplicates("paper_id", keep="last")
    replacement_metadata = replacement_metadata.drop_duplicates("paper_id", keep="last")
    metadata = base_metadata.set_index("paper_id")
    for _, row in replacement_metadata.iterrows():
        paper_id = str(row["paper_id"])
        for column, value in row.items():
            if column in metadata.columns:
                metadata.loc[paper_id, column] = value
    metadata = metadata.loc[order].reset_index()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_path = outdir / "R_cent_neutral_chunks.csv"
    vector_path = outdir / "R_cent.npy"
    metadata_path = outdir / f"page3_metadata_{len(order)}.csv"
    order_path = outdir / "paper_order.csv"
    combined_chunks.to_csv(chunk_path, index=False)
    np.save(vector_path, combined_vectors.astype(np.float32))
    metadata.to_csv(metadata_path, index=False)
    pd.DataFrame({"row_index": range(len(order)), "paper_id": order}).to_csv(
        order_path, index=False
    )

    manifest = {
        "method": "replace corrected records in the frozen Page 3 corpus",
        "paper_count": len(order),
        "replacement_count": len(replacement_order),
        "replacement_paper_ids": replacement_order,
        "chunk_count": len(combined_chunks),
        "embedding_shape": list(combined_vectors.shape),
        "base_chunks_sha256": sha256(Path(args.base_chunks)),
        "base_vectors_sha256": sha256(Path(args.base_vectors)),
        "replacement_chunks_sha256": sha256(Path(args.replacement_chunks)),
        "replacement_vectors_sha256": sha256(Path(args.replacement_vectors)),
        "combined_chunks_sha256": sha256(chunk_path),
        "combined_vectors_sha256": sha256(vector_path),
        "combined_metadata_sha256": sha256(metadata_path),
    }
    (outdir / "assembly_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(combined_chunks)} chunks, {combined_vectors.shape} vectors, "
        f"and {len(metadata)} metadata rows"
    )


if __name__ == "__main__":
    main()
