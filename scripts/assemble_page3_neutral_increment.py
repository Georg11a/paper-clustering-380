#!/usr/bin/env python3
"""Append a Drive-synced Page 3 increment to an existing frozen corpus."""

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
    parser.add_argument("--expected-old", type=int, required=True)
    parser.add_argument("--expected-new", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    old_chunks = pd.read_csv(args.old_chunks).fillna("")
    new_chunks = pd.read_csv(args.new_chunks).fillna("")
    old_order = old_chunks["paper_id"].astype(str).drop_duplicates().tolist()
    new_order = new_chunks["paper_id"].astype(str).drop_duplicates().tolist()
    if len(old_order) != args.expected_old or len(new_order) != args.expected_new:
        raise ValueError(
            f"Expected {args.expected_old} + {args.expected_new} papers, "
            f"got {len(old_order)} + {len(new_order)}"
        )
    if set(old_order) & set(new_order):
        raise ValueError("Old and new Page 3 inputs overlap")

    old_vectors = np.load(args.old_vectors)
    new_vectors = np.load(args.new_vectors)
    expected_dimension = old_vectors.shape[1]
    if old_vectors.shape != (args.expected_old, expected_dimension):
        raise ValueError(f"Unexpected old vector shape {old_vectors.shape}")
    if new_vectors.shape != (args.expected_new, expected_dimension):
        raise ValueError(f"Unexpected new vector shape {new_vectors.shape}")

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
    combined_metadata.index = combined_metadata["paper_id"].astype(str)
    combined_metadata = combined_metadata.loc[order].reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chunk_path = outdir / "R_cent_neutral_chunks.csv"
    vector_path = outdir / "R_cent.npy"
    metadata_path = outdir / f"page3_metadata_{len(order)}.csv"
    combined_chunks.to_csv(chunk_path, index=False)
    np.save(vector_path, combined_vectors)
    combined_metadata.to_csv(metadata_path, index=False)
    pd.DataFrame({"row_index": range(len(order)), "paper_id": order}).to_csv(
        outdir / "paper_order.csv", index=False
    )
    (outdir / "assembly_manifest.json").write_text(
        json.dumps(
            {
                "method": (
                    f"reuse frozen {args.expected_old}-paper Page 3 vectors; "
                    f"append {args.expected_new} Drive-synced papers processed "
                    "with the identical R_cent + BGE-M3 pipeline"
                ),
                "paper_count": len(order),
                "old_paper_count": args.expected_old,
                "new_paper_count": args.expected_new,
                "chunk_count": len(combined_chunks),
                "embedding_shape": list(combined_vectors.shape),
                "combined_chunks_sha256": sha256(chunk_path),
                "combined_metadata_sha256": sha256(metadata_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(combined_chunks)} chunks / {combined_vectors.shape} / "
        f"{len(combined_metadata)} metadata rows"
    )


if __name__ == "__main__":
    main()
