#!/usr/bin/env python3
"""Expand the Title+Abstract+K12 BGE-M3 representation while reusing old rows."""

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
    parser.add_argument("--input", required=True)
    parser.add_argument("--old-input", required=True)
    parser.add_argument("--old-embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input).fillna("").sort_values("paper_id").reset_index(drop=True)
    old_input = pd.read_csv(args.old_input).fillna("")
    old_vectors = normalize(np.load(args.old_embeddings), norm="l2")
    if len(old_input) != len(old_vectors):
        raise ValueError("Old input and embeddings do not align")
    old_rows = old_input.set_index("paper_id").to_dict("index")
    old_vector_map = {
        str(paper_id): old_vectors[index]
        for index, paper_id in enumerate(old_input["paper_id"].astype(str))
    }

    input_rows: list[dict[str, object]] = []
    new_chunks: list[str] = []
    new_chunk_owners: list[str] = []
    chunk_rows: list[dict[str, object]] = []
    reused = 0

    for _, row in frame.iterrows():
        paper_id = str(row["paper_id"])
        if paper_id in old_rows:
            prior = dict(old_rows[paper_id])
            prior["paper_id"] = paper_id
            input_rows.append(prior)
            reused += 1
            continue

        title = clean(row.get("title", ""))
        abstract = clean(row.get("abstract", ""))
        text_path = Path(str(row["canonical_text_path"]))
        passages = split_passages(
            paper_id, text_path.read_text(encoding="utf-8", errors="replace")
        )
        passages = [
            passage
            for passage in passages
            if not (abstract and jaccard(passage.text, abstract) >= 0.38)
        ]
        bm25_score(passages, keyword_query(str(row["keyword"])))
        selected = select_passages(
            passages, budget=args.budget, facet_cap=0, coverage_first=False
        )
        chunks = [f"{title} [SEP] {abstract}"] + [
            f"{title} [SEP] {clean(passage.text)}" for passage in selected
        ]
        for chunk_index, chunk in enumerate(chunks):
            new_chunks.append(chunk)
            new_chunk_owners.append(paper_id)
            chunk_rows.append(
                {
                    "paper_id": paper_id,
                    "keyword": row["keyword"],
                    "chunk_index": chunk_index,
                    "chunk_type": "title_abstract" if chunk_index == 0 else "title_pdf_passage",
                    "characters": len(chunk),
                    "words": len(TOKEN_RE.findall(chunk)),
                }
            )
        input_rows.append(
            {
                "paper_id": paper_id,
                "keyword": row["keyword"],
                "title": title,
                "abstract": abstract,
                "budget": args.budget,
                "selected_count": len(selected),
                "passage_ids": ";".join(p.passage_id for p in selected),
                "subdocument": "\n\n".join(p.text for p in selected),
                "canonical_text_path": row["canonical_text_path"],
            }
        )

    timing: dict[str, object] = {"reused_only": True}
    new_vector_map: dict[str, np.ndarray] = {}
    if new_chunks:
        chunk_vectors, timing = ollama_embeddings(
            new_chunks, args.model, args.host, args.batch_size, args.timeout
        )
        np.save(out / "new_chunk_embeddings.npy", chunk_vectors)
        owner_array = np.asarray(new_chunk_owners)
        for paper_id in sorted(set(new_chunk_owners)):
            indices = np.flatnonzero(owner_array == paper_id)
            new_vector_map[paper_id] = normalize(
                chunk_vectors[indices].mean(axis=0, keepdims=True), norm="l2"
            )[0]

    input_frame = pd.DataFrame(input_rows).sort_values("paper_id").reset_index(drop=True)
    ordered_vectors = [
        old_vector_map.get(pid, new_vector_map.get(pid))
        for pid in input_frame["paper_id"].astype(str)
    ]
    if any(vector is None for vector in ordered_vectors):
        raise RuntimeError("At least one vector is missing")
    vectors = np.vstack(ordered_vectors).astype(np.float32)
    vectors = normalize(vectors, norm="l2").astype(np.float32)

    count = len(input_frame)
    input_frame.to_csv(out / f"global_contextual_input_{count}.csv", index=False)
    pd.DataFrame(chunk_rows).to_csv(out / f"new_chunk_manifest_{count}.csv", index=False)
    np.save(out / f"embeddings_bge_m3_global_{count}_k12.npy", vectors)
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "paper_count": count,
                "reused_papers": reused,
                "new_papers_embedded": count - reused,
                "new_chunks_embedded": len(new_chunks),
                "representation": "BGE-M3 Title+Abstract+K12",
                "embedding_runtime": timing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {count} papers: reused {reused}, embedded {count-reused} new papers / {len(new_chunks)} chunks")


if __name__ == "__main__":
    main()
