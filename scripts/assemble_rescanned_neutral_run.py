#!/usr/bin/env python3
"""Assemble the corrected 459-paper R_cent representation after Downloads rescan."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/baiyixin/Documents/Survey - design knowledge")
OLD_CHUNKS = ROOT / "expanded_pipeline_459_20260811/neutral_457/R_cent_neutral_chunks.csv"
OLD_VECTORS = ROOT / "expanded_pipeline_459_20260811/neutral_457/R_cent.npy"
NEW_CHUNKS = ROOT / "expanded_pipeline_464_20260811/chunks_new_3/R_cent_chunks.csv"
NEW_VECTORS = ROOT / "expanded_pipeline_464_20260811/neutral_new_3/emb/R_cent.npy"
OUT = ROOT / "expanded_pipeline_464_20260811/neutral_459"
EXCLUDED = "c3b41755b43d"


def main() -> None:
    old_chunks = pd.read_csv(OLD_CHUNKS).fillna("")
    old_order = old_chunks["paper_id"].drop_duplicates().astype(str).tolist()
    old_vectors = np.load(OLD_VECTORS)
    if len(old_order) != 457 or old_vectors.shape != (457, 1024):
        raise RuntimeError("Unexpected prior R_cent inputs.")
    old_map = {paper_id: old_vectors[index] for index, paper_id in enumerate(old_order)}
    old_keep = [paper_id for paper_id in old_order if paper_id != EXCLUDED]

    new_chunks = pd.read_csv(NEW_CHUNKS).fillna("")
    new_order = new_chunks["paper_id"].drop_duplicates().astype(str).tolist()
    new_vectors = np.load(NEW_VECTORS)
    if len(new_order) != 3 or new_vectors.shape != (3, 1024):
        raise RuntimeError("Unexpected new R_cent inputs.")
    overlap = set(old_keep) & set(new_order)
    if overlap:
        raise RuntimeError(f"Old/new overlap: {sorted(overlap)}")

    combined_order = old_keep + new_order
    vectors = np.vstack(
        [old_map[paper_id] for paper_id in old_keep] + list(new_vectors)
    ).astype(np.float32)
    chunks = pd.concat(
        [
            old_chunks.loc[~old_chunks["paper_id"].astype(str).eq(EXCLUDED)],
            new_chunks.drop(columns=["keyword", "representation"], errors="ignore"),
        ],
        ignore_index=True,
    )
    chunk_order = chunks["paper_id"].drop_duplicates().astype(str).tolist()
    if chunk_order != combined_order or len(combined_order) != 459:
        raise RuntimeError("Combined chunk/vector order is not 459-paper aligned.")

    OUT.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(OUT / "R_cent_neutral_chunks.csv", index=False)
    np.save(OUT / "R_cent.npy", vectors)
    pd.DataFrame(
        {"row_index": range(len(combined_order)), "paper_id": combined_order}
    ).to_csv(OUT / "paper_order.csv", index=False)
    print(f"Wrote {len(chunks)} chunks / {len(combined_order)} papers / {vectors.shape}")


if __name__ == "__main__":
    main()
