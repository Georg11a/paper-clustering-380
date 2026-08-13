#!/usr/bin/env python3
"""Embed a neutral chunk CSV and mean-pool to one BGE-M3 vector per paper."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from add_bge_m3_to_experiments import ollama_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paper-order", required=True)
    parser.add_argument("--batch-cache", required=True)
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    chunks = pd.read_csv(args.chunks).fillna("")
    forbidden = {"keyword", "cluster", "label", "query", "search_keyword"}
    if forbidden & set(chunks.columns):
        chunks = chunks.drop(columns=sorted(forbidden & set(chunks.columns)))
    order = chunks["paper_id"].astype(str).drop_duplicates().tolist()
    texts = chunks["chunk"].astype(str).tolist()

    cache_dir = Path(args.batch_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedded: list[np.ndarray] = []
    for start in range(0, len(texts), args.batch_size):
        stop = min(start + args.batch_size, len(texts))
        values = texts[start:stop]
        digest = hashlib.sha256("\n\0".join(values).encode("utf-8")).hexdigest()[:16]
        cache = cache_dir / f"batch_{start:06d}_{stop:06d}_{digest}.npy"
        if cache.exists():
            vectors = np.load(cache)
        else:
            vectors, _ = ollama_embeddings(
                values, args.model, args.host, args.batch_size, args.timeout
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            np.save(cache, vectors)
        embedded.append(vectors)
        print(f"embedded {stop}/{len(texts)}", flush=True)

    chunk_vectors = np.vstack(embedded)
    indices = chunks.groupby("paper_id", sort=False).indices
    pooled = np.vstack(
        [chunk_vectors[np.asarray(indices[paper_id])].mean(axis=0) for paper_id in order]
    )
    pooled = normalize(pooled, norm="l2").astype(np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, pooled)
    pd.DataFrame({"row_index": range(len(order)), "paper_id": order}).to_csv(
        args.paper_order, index=False
    )
    print(f"Wrote {output}: {pooled.shape}")


if __name__ == "__main__":
    main()
