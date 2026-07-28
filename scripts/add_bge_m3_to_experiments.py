#!/usr/bin/env python3
"""Add BGE-M3 to the already completed Experiment A and B outputs."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from run_experiment_a import neighbor_rows, probe_representation


def ollama_embeddings(
    texts: list[str],
    model: str,
    host: str,
    batch_size: int,
    timeout: int,
) -> tuple[np.ndarray, dict[str, object]]:
    vectors: list[list[float]] = []
    total_duration_ns = 0
    load_duration_ns = 0
    truncated_inputs = 0
    started = time.monotonic()
    endpoint = f"{host.rstrip('/')}/api/embed"

    def request_batch(batch: list[str], truncate: bool = False) -> list[list[float]]:
        nonlocal total_duration_ns, load_duration_ns, truncated_inputs
        payload = json.dumps(
            {
                "model": model,
                "input": batch,
                "truncate": truncate,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 400 and len(batch) > 1:
                midpoint = len(batch) // 2
                return request_batch(batch[:midpoint]) + request_batch(batch[midpoint:])
            if error.code == 400 and len(batch) == 1 and not truncate:
                truncated_inputs += 1
                return request_batch(batch, truncate=True)
            raise
        returned = result.get("embeddings", [])
        if len(returned) != len(batch):
            raise RuntimeError(
                f"Ollama returned {len(returned)} vectors for {len(batch)} texts"
            )
        total_duration_ns += int(result.get("total_duration", 0))
        load_duration_ns += int(result.get("load_duration", 0))
        return returned

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(request_batch(batch))
        print(
            f"Embedded {min(start + batch_size, len(texts))}/{len(texts)} "
            f"with {model}",
            flush=True,
        )
    embeddings = normalize(np.asarray(vectors, dtype=np.float32), norm="l2")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("BGE-M3 produced non-finite values")
    metadata = {
        "model": model,
        "provider": "Ollama local /api/embed",
        "host": host,
        "dimensions": int(embeddings.shape[1]),
        "batch_size": batch_size,
        "truncate": (
            "False by default; True only for isolated over-context inputs"
        ),
        "truncated_input_count": truncated_inputs,
        "wall_seconds": time.monotonic() - started,
        "ollama_total_duration_seconds": total_duration_ns / 1e9,
        "ollama_load_duration_seconds": load_duration_ns / 1e9,
    }
    return embeddings, metadata


def ollama_chunk_pooling(
    rows: pd.DataFrame,
    model: str,
    host: str,
    batch_size: int,
    timeout: int,
) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    flat_chunks: list[str] = []
    owners: list[int] = []
    chunk_rows: list[dict] = []
    for paper_index, row in rows.iterrows():
        passages = [
            " ".join(value.split())
            for value in str(row["subdocument"]).split("\n\n")
            if value.strip()
        ]
        chunks = [str(row["experiment_a_text"])]
        chunks.extend(
            f"{row['title']} [SEP] {passage}" for passage in passages
        )
        for chunk_index, chunk in enumerate(chunks):
            owners.append(paper_index)
            flat_chunks.append(chunk)
            chunk_rows.append(
                {
                    "paper_id": row["paper_id"],
                    "chunk_index": chunk_index,
                    "chunk_type": (
                        "title_abstract" if chunk_index == 0 else "title_passage"
                    ),
                    "characters": len(chunk),
                }
            )
    chunk_embeddings, timing = ollama_embeddings(
        flat_chunks, model, host, batch_size, timeout
    )
    pooled: list[np.ndarray] = []
    counts: list[int] = []
    for paper_index in range(len(rows)):
        indices = [
            index for index, owner in enumerate(owners) if owner == paper_index
        ]
        counts.append(len(indices))
        pooled.append(chunk_embeddings[indices].mean(axis=0))
    embeddings = normalize(np.vstack(pooled), norm="l2").astype(np.float32)
    timing.update(
        {
            "strategy": "chunk_pooling",
            "chunk_design": [
                "shared capped Title [SEP] Abstract",
                "Title [SEP] each selected passage",
            ],
            "pooling": (
                "mean of L2-normalized chunk embeddings, then L2 normalize"
            ),
            "minimum_chunks": min(counts),
            "maximum_chunks": max(counts),
            "mean_chunks": float(np.mean(counts)),
        }
    )
    return embeddings, timing, pd.DataFrame(chunk_rows)


def replace_model_rows(path: Path, new_rows: pd.DataFrame, model: str) -> None:
    if path.exists():
        existing = pd.read_csv(path)
        if "model" in existing:
            existing = existing[existing["model"] != model]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def add_to_experiment(
    output: Path,
    text_column: str,
    model_label: str,
    embedding_filename: str,
    args: argparse.Namespace,
    chunk_pooling: bool = False,
) -> dict[str, object]:
    input_files = sorted(output.glob("experiment_*_input.csv"))
    if len(input_files) != 1:
        raise RuntimeError(f"Expected one experiment input in {output}")
    input_frame = pd.read_csv(input_files[0]).fillna("")
    if text_column not in input_frame:
        raise RuntimeError(f"{input_files[0]} has no {text_column!r} column")

    if chunk_pooling:
        embeddings, metadata, chunk_manifest = ollama_chunk_pooling(
            input_frame,
            args.model,
            args.host,
            args.batch_size,
            args.timeout,
        )
        chunk_manifest.to_csv(
            output / "bge_m3_chunk_manifest.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        embeddings, metadata = ollama_embeddings(
            input_frame[text_column].astype(str).tolist(),
            args.model,
            args.host,
            args.batch_size,
            args.timeout,
        )
    np.save(output / embedding_filename, embeddings)

    runs, summary, selection = probe_representation(
        model_label,
        embeddings,
        range(args.k_min, args.k_max + 1),
        args.seed,
        args.seed_runs,
    )
    neighbors = neighbor_rows(model_label, embeddings, input_frame)
    replace_model_rows(output / "probe_runs.csv", pd.DataFrame(runs), model_label)
    replace_model_rows(
        output / "probe_summary.csv", pd.DataFrame(summary), model_label
    )
    replace_model_rows(
        output / "nearest_neighbors_top5.csv",
        pd.DataFrame(neighbors),
        model_label,
    )

    assignments_path = output / "probe_cluster_assignments.csv"
    assignments = pd.read_csv(assignments_path)
    labels = selection.pop("labels")
    assignments[f"{model_label}_probe_cluster"] = labels
    assignments.to_csv(assignments_path, index=False, encoding="utf-8-sig")

    metadata["ollama_model"] = args.model
    metadata["model"] = model_label
    metadata["input_file"] = str(input_files[0])
    metadata["input_text_column"] = text_column
    metadata["probe_selection"] = selection
    metadata_path = output / "run_metadata.json"
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_metadata["models"] = [
        item
        for item in run_metadata.get("models", [])
        if item.get("model") not in {args.model, model_label}
    ]
    run_metadata["models"].append(metadata)
    run_metadata["status"] = (
        "embedding comparison complete for TF-IDF–SVD, SPECTER2, and BGE-M3; "
        "probe clusters are not final"
    )
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    return {
        "output": str(output),
        "rows": len(input_frame),
        "shape": list(embeddings.shape),
        "selection": selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--experiment-a", default="outputs/batch2/experiment_a_design_theory"
    )
    parser.add_argument(
        "--experiment-b", default="outputs/batch2/experiment_b_design_theory"
    )
    args = parser.parse_args()

    results = [
        add_to_experiment(
            Path(args.experiment_a),
            "experiment_a_text",
            "bge_m3",
            "embeddings_bge_m3.npy",
            args,
        ),
        add_to_experiment(
            Path(args.experiment_b),
            "contextual_text",
            "bge_m3_chunk_pooling",
            "embeddings_bge_m3_chunk_pooling.npy",
            args,
            chunk_pooling=True,
        ),
    ]
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
