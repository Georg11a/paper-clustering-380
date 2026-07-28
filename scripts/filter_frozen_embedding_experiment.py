#!/usr/bin/env python3
"""Create an auditable filtered copy of a frozen embedding experiment."""

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
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--chunk-manifest")
    parser.add_argument("--exclude-paper-ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--output-input-name")
    parser.add_argument("--output-embedding-name")
    parser.add_argument("--output-chunk-name")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    embedding_path = Path(args.embeddings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    embeddings = np.load(embedding_path)
    if len(frame) != len(embeddings):
        raise ValueError(
            f"Input has {len(frame)} rows but embeddings have "
            f"{len(embeddings)} rows."
        )
    excluded = {
        value.strip()
        for value in args.exclude_paper_ids.split(",")
        if value.strip()
    }
    present = excluded & set(frame["paper_id"].astype(str))
    missing = excluded - present
    if missing:
        raise ValueError(f"Excluded IDs absent from input: {sorted(missing)}")

    keep = ~frame["paper_id"].astype(str).isin(excluded)
    filtered_frame = frame.loc[keep].reset_index(drop=True)
    filtered_embeddings = embeddings[keep.to_numpy()]
    if len(filtered_frame) != len(filtered_embeddings):
        raise RuntimeError("Filtered CSV and embedding matrix lost alignment.")

    output_input = out / (args.output_input_name or input_path.name)
    output_embeddings = out / (
        args.output_embedding_name or embedding_path.name
    )
    filtered_frame.to_csv(output_input, index=False, encoding="utf-8-sig")
    np.save(output_embeddings, filtered_embeddings)

    chunk_output = None
    if args.chunk_manifest:
        chunk_path = Path(args.chunk_manifest)
        chunks = pd.read_csv(chunk_path)
        chunks = chunks[
            ~chunks["paper_id"].astype(str).isin(excluded)
        ].reset_index(drop=True)
        chunk_output = out / (args.output_chunk_name or chunk_path.name)
        chunks.to_csv(chunk_output, index=False, encoding="utf-8-sig")

    metadata = {
        "operation": "filter frozen embedding experiment",
        "excluded_paper_ids": sorted(excluded),
        "source_input": str(input_path),
        "source_input_sha256": sha256(input_path),
        "source_embeddings": str(embedding_path),
        "source_embeddings_sha256": sha256(embedding_path),
        "source_rows": len(frame),
        "output_rows": len(filtered_frame),
        "embedding_shape": list(filtered_embeddings.shape),
        "embedding_recomputed": False,
        "justification": (
            "Embeddings are document-independent. Removing a scope-excluded "
            "document does not change the frozen vectors of retained papers."
        ),
        "output_input": str(output_input),
        "output_input_sha256": sha256(output_input),
        "output_embeddings": str(output_embeddings),
        "output_embeddings_sha256": sha256(output_embeddings),
        "output_chunk_manifest": str(chunk_output) if chunk_output else "",
    }
    (out / "filter_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Filtered {len(frame)} -> {len(filtered_frame)} rows; "
        f"wrote {out}"
    )


if __name__ == "__main__":
    main()
