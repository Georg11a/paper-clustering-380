#!/usr/bin/env python3
"""Clean frozen contextual text, exclude reviewed versions, and refresh vectors.

Unchanged papers retain their frozen document embeddings. Papers whose title,
abstract, or subdocument changes after encoding repair are re-embedded with
the same BGE-M3 chunk-pooling strategy used by the contextual experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ftfy import fix_encoding
from sklearn.preprocessing import normalize

from add_bge_m3_to_experiments import ollama_embeddings
from build_paragraph_budget_pilot import TOKEN_RE


TEXT_COLUMNS = ("title", "abstract", "subdocument")
KNOWN_MOJIBAKE_REPLACEMENTS = {
    "‚Äâ√ó‚Äâ": " × ",
    "‚Äê": "-",
    "√§": "ä",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_encoding(value: object) -> str:
    """Repair encoding only; do not normalize typography or wording."""
    cleaned = fix_encoding(str(value if value is not None else ""))
    for broken, replacement in KNOWN_MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(broken, replacement)
    return cleaned


def paper_chunks(row: pd.Series) -> list[str]:
    title = " ".join(str(row["title"]).split())
    abstract = " ".join(str(row["abstract"]).split())
    passages = [
        " ".join(value.split())
        for value in str(row["subdocument"]).split("\n\n")
        if value.strip()
    ]
    chunks = [f"{title} [SEP] {abstract}"]
    chunks.extend(f"{title} [SEP] {passage}" for passage in passages)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--exclude-paper-ids", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    embedding_path = Path(args.embeddings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path).fillna("")
    frozen_embeddings = np.load(embedding_path)
    if len(frame) != len(frozen_embeddings):
        raise ValueError("Input rows and frozen embeddings do not align.")
    if frame["paper_id"].duplicated().any():
        raise ValueError("Input contains duplicate paper IDs.")

    excluded = {
        value.strip()
        for value in args.exclude_paper_ids.split(",")
        if value.strip()
    }
    missing = excluded - set(frame["paper_id"].astype(str))
    if missing:
        raise ValueError(f"Excluded IDs absent from input: {sorted(missing)}")

    keep = ~frame["paper_id"].astype(str).isin(excluded)
    cleaned = frame.loc[keep].reset_index(drop=True)
    refreshed = frozen_embeddings[keep.to_numpy()].astype(np.float32)
    before = cleaned.loc[:, TEXT_COLUMNS].copy()
    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].map(clean_encoding)

    changed_mask = np.zeros(len(cleaned), dtype=bool)
    changed_cells: dict[str, int] = {}
    for column in TEXT_COLUMNS:
        column_changed = before[column].astype(str) != cleaned[column].astype(str)
        changed_cells[column] = int(column_changed.sum())
        changed_mask |= column_changed.to_numpy()

    changed_indices = np.flatnonzero(changed_mask)
    flat_chunks: list[str] = []
    owners: list[int] = []
    for paper_index in changed_indices:
        for chunk in paper_chunks(cleaned.iloc[paper_index]):
            owners.append(int(paper_index))
            flat_chunks.append(chunk)

    timing: dict[str, object] = {
        "model": args.model,
        "provider": "Ollama local /api/embed",
        "reembedded_paper_count": 0,
        "reembedded_chunk_count": 0,
    }
    if flat_chunks:
        chunk_embeddings, timing = ollama_embeddings(
            flat_chunks,
            args.model,
            args.host,
            args.batch_size,
            args.timeout,
        )
        owner_array = np.asarray(owners)
        for paper_index in changed_indices:
            indices = np.flatnonzero(owner_array == paper_index)
            refreshed[paper_index] = normalize(
                chunk_embeddings[indices].mean(axis=0, keepdims=True),
                norm="l2",
            )[0]
        timing["reembedded_paper_count"] = int(len(changed_indices))
        timing["reembedded_chunk_count"] = int(len(flat_chunks))

    refreshed = normalize(refreshed, norm="l2").astype(np.float32)
    count = len(cleaned)
    output_input = out / f"global_contextual_input_{count}.csv"
    output_embeddings = out / f"embeddings_bge_m3_global_{count}_k12.npy"
    output_chunks = out / f"global_chunk_manifest_{count}.csv"
    cleaned.to_csv(output_input, index=False, encoding="utf-8-sig")
    np.save(output_embeddings, refreshed)

    chunk_rows: list[dict[str, object]] = []
    for _, row in cleaned.iterrows():
        for chunk_index, chunk in enumerate(paper_chunks(row)):
            chunk_rows.append(
                {
                    "paper_id": row["paper_id"],
                    "keyword": row["keyword"],
                    "chunk_index": chunk_index,
                    "chunk_type": (
                        "title_abstract"
                        if chunk_index == 0
                        else "title_pdf_passage"
                    ),
                    "characters": len(chunk),
                    "words": len(TOKEN_RE.findall(chunk)),
                }
            )
    pd.DataFrame(chunk_rows).to_csv(
        output_chunks, index=False, encoding="utf-8-sig"
    )

    metadata = {
        "operation": "clean and refresh frozen contextual embeddings",
        "excluded_paper_ids": sorted(excluded),
        "source_rows": int(len(frame)),
        "output_rows": int(count),
        "encoding_repair": "ftfy.fix_encoding only",
        "changed_cells_by_column": changed_cells,
        "reembedded_paper_ids": cleaned.loc[
            changed_indices, "paper_id"
        ].astype(str).tolist(),
        "unchanged_embeddings_reused": int(count - len(changed_indices)),
        "embedding_shape": list(refreshed.shape),
        "embedding_timing": timing,
        "source_input": str(input_path),
        "source_input_sha256": sha256(input_path),
        "source_embeddings": str(embedding_path),
        "source_embeddings_sha256": sha256(embedding_path),
        "output_input": str(output_input),
        "output_input_sha256": sha256(output_input),
        "output_embeddings": str(output_embeddings),
        "output_embeddings_sha256": sha256(output_embeddings),
        "output_chunk_manifest": str(output_chunks),
    }
    (out / "refresh_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Cleaned {len(frame)} -> {count} papers; repaired "
        f"{sum(changed_cells.values())} cells and re-embedded "
        f"{len(changed_indices)} papers."
    )


if __name__ == "__main__":
    main()
