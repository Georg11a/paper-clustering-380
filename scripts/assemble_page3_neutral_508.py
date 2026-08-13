#!/usr/bin/env python3
"""Append 49 Drive-synced papers to the frozen 459-paper Page 3 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-chunks", required=True)
    parser.add_argument("--old-vectors", required=True)
    parser.add_argument("--old-metadata", required=True)
    parser.add_argument("--new-chunks", required=True)
    parser.add_argument("--new-vectors", required=True)
    parser.add_argument("--new-metadata", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    old_chunks = pd.read_csv(args.old_chunks).fillna("")
    new_chunks = pd.read_csv(args.new_chunks).fillna("")
    old_order = old_chunks["paper_id"].astype(str).drop_duplicates().tolist()
    new_order = new_chunks["paper_id"].astype(str).drop_duplicates().tolist()
    if len(old_order) != 459 or len(new_order) != 49:
        raise ValueError(f"Expected 459 + 49 papers, got {len(old_order)} + {len(new_order)}")
    if set(old_order) & set(new_order):
        raise ValueError("Old and new Page 3 inputs overlap")

    old_vectors = np.load(args.old_vectors)
    new_vectors = np.load(args.new_vectors)
    if old_vectors.shape != (459, 1024) or new_vectors.shape != (49, 1024):
        raise ValueError(f"Unexpected vector shapes {old_vectors.shape}, {new_vectors.shape}")

    required = ["paper_id", "chunk_index", "chunk"]
    combined_chunks = pd.concat(
        [old_chunks[required], new_chunks[required]], ignore_index=True
    )
    combined_vectors = np.vstack([old_vectors, new_vectors]).astype(np.float32)
    order = old_order + new_order
    if combined_chunks["paper_id"].astype(str).drop_duplicates().tolist() != order:
        raise ValueError("Combined chunk and vector orders are not aligned")

    old_metadata = pd.read_csv(args.old_metadata).fillna("")
    new_metadata = pd.read_csv(args.new_metadata).fillna("")
    combined_metadata = pd.concat([old_metadata, new_metadata], ignore_index=True)
    combined_metadata = combined_metadata.drop_duplicates("paper_id", keep="last")
    combined_metadata = combined_metadata.set_index(
        combined_metadata["paper_id"].astype(str)
    ).loc[order].reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_path = outdir / "R_cent_neutral_chunks.csv"
    vector_path = outdir / "R_cent.npy"
    metadata_path = outdir / "page3_metadata_508.csv"
    combined_chunks.to_csv(chunk_path, index=False)
    np.save(vector_path, combined_vectors)
    combined_metadata.to_csv(metadata_path, index=False)
    pd.DataFrame({"row_index": range(508), "paper_id": order}).to_csv(
        outdir / "paper_order.csv", index=False
    )
    (outdir / "assembly_manifest.json").write_text(
        json.dumps(
            {
                "method": "reuse frozen 459-paper Page 3 vectors; append 49 Drive-synced papers processed with the identical R_cent + BGE-M3 pipeline",
                "paper_count": 508,
                "old_paper_count": 459,
                "new_paper_count": 49,
                "chunk_count": len(combined_chunks),
                "embedding_shape": list(combined_vectors.shape),
                "combined_chunks_sha256": sha256(chunk_path),
                "combined_metadata_sha256": sha256(metadata_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(combined_chunks)} chunks / {combined_vectors.shape} / {len(combined_metadata)} metadata rows")


if __name__ == "__main__":
    main()
