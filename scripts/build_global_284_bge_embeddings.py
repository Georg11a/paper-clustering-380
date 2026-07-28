#!/usr/bin/env python3
"""Build the global 284-paper K12 representation and BGE-M3 embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from add_bge_m3_to_experiments import ollama_embeddings
from build_paragraph_budget_pilot import (
    TOKEN_RE,
    bm25_score,
    jaccard,
    keyword_query,
    select_passages,
    split_passages,
)


def clean(value: object) -> str:
    return " ".join(str(value if value is not None else "").split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="outputs/batch1/canonical_analysis_input_284.csv"
    )
    parser.add_argument(
        "--out", default="outputs/batch2/global_284_bge_m3_contextual"
    )
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = repo / args.out
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(repo / args.input).fillna("")
    if len(frame) != 284 or frame["paper_id"].nunique() != 284:
        raise RuntimeError(
            f"Expected 284 unique papers, found {len(frame)} rows and "
            f"{frame['paper_id'].nunique()} IDs."
        )

    input_rows: list[dict[str, object]] = []
    flat_chunks: list[str] = []
    owners: list[int] = []
    chunk_rows: list[dict[str, object]] = []

    for paper_index, row in frame.sort_values("paper_id").reset_index(drop=True).iterrows():
        title = clean(row["canonical_title"] or row["title"])
        abstract = clean(row["abstract"])
        text_path = Path(str(row["canonical_text_path"]))
        if not text_path.is_absolute():
            text_path = repo / text_path
        passages = split_passages(
            str(row["paper_id"]),
            text_path.read_text(encoding="utf-8", errors="replace"),
        )
        passages = [
            passage
            for passage in passages
            if not (abstract and jaccard(passage.text, abstract) >= 0.38)
        ]
        bm25_score(passages, keyword_query(str(row["keyword"])))
        selected = select_passages(
            passages,
            budget=args.budget,
            facet_cap=0,
            coverage_first=False,
        )
        metadata_chunk = f"{title} [SEP] {abstract}"
        paper_chunks = [metadata_chunk]
        paper_chunks.extend(
            f"{title} [SEP] {clean(passage.text)}" for passage in selected
        )
        for chunk_index, chunk in enumerate(paper_chunks):
            owners.append(paper_index)
            flat_chunks.append(chunk)
            chunk_rows.append(
                {
                    "paper_id": row["paper_id"],
                    "keyword": row["keyword"],
                    "chunk_index": chunk_index,
                    "chunk_type": (
                        "title_abstract" if chunk_index == 0 else "title_pdf_passage"
                    ),
                    "characters": len(chunk),
                    "words": len(TOKEN_RE.findall(chunk)),
                }
            )
        input_rows.append(
            {
                "paper_id": row["paper_id"],
                "keyword": row["keyword"],
                "title": title,
                "abstract": abstract,
                "budget": args.budget,
                "selected_count": len(selected),
                "passage_ids": ";".join(passage.passage_id for passage in selected),
                "subdocument": "\n\n".join(passage.text for passage in selected),
                "canonical_text_path": row["canonical_text_path"],
            }
        )

    input_frame = pd.DataFrame(input_rows)
    chunk_frame = pd.DataFrame(chunk_rows)
    input_frame.to_csv(out / "global_contextual_input_284.csv", index=False)
    chunk_frame.to_csv(out / "global_chunk_manifest_284.csv", index=False)

    chunk_embeddings, timing = ollama_embeddings(
        flat_chunks,
        args.model,
        args.host,
        args.batch_size,
        args.timeout,
    )
    pooled: list[np.ndarray] = []
    counts: list[int] = []
    for paper_index in range(len(input_frame)):
        indices = np.flatnonzero(np.asarray(owners) == paper_index)
        pooled.append(chunk_embeddings[indices].mean(axis=0))
        counts.append(len(indices))
    embeddings = normalize(np.vstack(pooled), norm="l2").astype(np.float32)
    np.save(out / "embeddings_bge_m3_global_k12.npy", embeddings)

    metadata = {
        "paper_count": len(input_frame),
        "keyword_counts": input_frame["keyword"].value_counts().to_dict(),
        "budget": args.budget,
        "selection": (
            "BM25 keyword relevance + MMR redundancy penalty + soft facet "
            "diversity; no hard facet cap and no coverage-first initialization"
        ),
        "chunk_design": [
            "Title [SEP] Abstract",
            "Title [SEP] each selected canonical-PDF passage",
        ],
        "pooling": "mean of L2-normalized chunk embeddings, then L2 normalize",
        "minimum_chunks": min(counts),
        "maximum_chunks": max(counts),
        "mean_chunks": float(np.mean(counts)),
        "embedding_shape": list(embeddings.shape),
        "embedding_runtime": timing,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(input_frame)} papers, {len(flat_chunks)} chunks, "
        f"and embeddings {embeddings.shape} to {out}"
    )


if __name__ == "__main__":
    main()
