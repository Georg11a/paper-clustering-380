#!/usr/bin/env python3
"""Run contextual Experiment B on Title + Abstract + final K12 passages."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import adapters
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from adapters import AutoAdapterModel
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

from run_experiment_a import (
    neighbor_rows,
    probe_representation,
    tfidf_svd_embeddings,
    write_csv,
)


def contextual_input(
    experiment_a_input: Path,
    subdocuments: Path,
) -> pd.DataFrame:
    base = pd.read_csv(experiment_a_input).fillna("")
    if "experiment_a_text" not in base:
        raise RuntimeError(
            f"{experiment_a_input} has no shared capped experiment_a_text"
        )
    subdocs = pd.read_csv(subdocuments).fillna("")
    required = {"paper_id", "budget", "selected_count", "passage_ids", "subdocument"}
    missing = required - set(subdocs)
    if missing:
        raise RuntimeError(f"Sub-document CSV is missing {sorted(missing)}")
    if len(subdocs) != subdocs["paper_id"].nunique():
        raise RuntimeError("Expected exactly one final sub-document per paper")
    rows = base.merge(
        subdocs[
            ["paper_id", "budget", "selected_count", "passage_ids", "subdocument"]
        ],
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    if rows["subdocument"].isna().any():
        raise RuntimeError("Some Experiment A papers have no final sub-document")
    rows["contextual_text"] = rows.apply(
        lambda row: (
            f"{row['experiment_a_text']}\n\n{row['subdocument']}"
        ),
        axis=1,
    )
    return rows


def specter2_chunk_pooling(
    rows: pd.DataFrame,
    base_path: Path,
    adapter_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object], list[dict]]:
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    model = AutoAdapterModel.from_pretrained(str(base_path), local_files_only=True)
    adapter_name = model.load_adapter(
        str(adapter_path), load_as="proximity", set_active=True
    )
    model.set_active_adapters(adapter_name)
    model.eval()

    flat_chunks: list[str] = []
    ownership: list[int] = []
    chunk_rows: list[dict] = []
    for paper_index, row in rows.iterrows():
        passages = [
            " ".join(value.split())
            for value in str(row["subdocument"]).split("\n\n")
            if value.strip()
        ]
        chunks = [str(row["experiment_a_text"])]
        chunks.extend(
            f"{row['title']}{tokenizer.sep_token}{passage}"
            for passage in passages
        )
        for chunk_index, chunk in enumerate(chunks):
            ownership.append(paper_index)
            flat_chunks.append(chunk)
            chunk_rows.append(
                {
                    "paper_id": row["paper_id"],
                    "chunk_index": chunk_index,
                    "chunk_type": "title_abstract" if chunk_index == 0 else "title_passage",
                    "characters": len(chunk),
                }
            )

    vectors: list[np.ndarray] = []
    for start in range(0, len(flat_chunks), batch_size):
        inputs = tokenizer(
            flat_chunks[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        with torch.inference_mode():
            output = model(**inputs).last_hidden_state[:, 0, :]
        vectors.append(output.detach().cpu().numpy())
    chunk_embeddings = normalize(np.vstack(vectors), norm="l2")

    pooled: list[np.ndarray] = []
    chunk_counts: list[int] = []
    for paper_index in range(len(rows)):
        indices = [
            index for index, owner in enumerate(ownership) if owner == paper_index
        ]
        chunk_counts.append(len(indices))
        pooled.append(chunk_embeddings[indices].mean(axis=0))
    embeddings = normalize(np.vstack(pooled), norm="l2").astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise RuntimeError("SPECTER2 chunk pooling produced non-finite values")
    metadata = {
        "model": "specter2_chunk_pooling",
        "base_path": str(base_path),
        "adapter_path": str(adapter_path),
        "adapter": str(adapter_name),
        "adapter_active": str(model.active_adapters),
        "dimensions": int(embeddings.shape[1]),
        "max_length_per_chunk": 512,
        "chunk_design": [
            "Title [SEP] Abstract",
            "Title [SEP] each selected passage",
        ],
        "pooling": "mean of L2-normalized CLS chunk embeddings, then L2 normalize",
        "minimum_chunks": min(chunk_counts),
        "maximum_chunks": max(chunk_counts),
        "mean_chunks": float(np.mean(chunk_counts)),
        "batch_size": batch_size,
    }
    return embeddings, metadata, chunk_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-a-input",
        default=(
            "outputs/batch2/experiment_a_design_theory/"
            "experiment_a_input.csv"
        ),
    )
    parser.add_argument(
        "--subdocuments",
        default=(
            "outputs/batch2/final_subdocuments_design_theory/"
            "subdocuments_k12_design_theory.csv"
        ),
    )
    parser.add_argument("--keyword", default="Design Theory")
    parser.add_argument(
        "--output", default="outputs/batch2/experiment_b_design_theory"
    )
    specter_cache = Path.home() / ".cache" / "specter2"
    parser.add_argument("--specter-base", default=str(specter_cache / "base"))
    parser.add_argument(
        "--specter-adapter",
        default=str(specter_cache / "adapters" / "proximity"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = contextual_input(
        Path(args.experiment_a_input), Path(args.subdocuments)
    )
    input_path = output / "experiment_b_input.csv"
    rows.to_csv(input_path, index=False, encoding="utf-8-sig")

    model_outputs: list[tuple[str, np.ndarray, dict[str, object]]] = []
    print(f"Encoding {len(rows)} papers with contextual TF-IDF–SVD", flush=True)
    tfidf_embeddings, tfidf_metadata = tfidf_svd_embeddings(
        rows["contextual_text"].tolist(), args.seed
    )
    model_outputs.append(("tfidf_svd_contextual", tfidf_embeddings, tfidf_metadata))

    print(
        f"Encoding {len(rows)} papers with SPECTER2 chunk pooling", flush=True
    )
    specter_embeddings, specter_metadata, chunk_rows = specter2_chunk_pooling(
        rows,
        Path(args.specter_base),
        Path(args.specter_adapter),
        args.batch_size,
    )
    model_outputs.append(
        ("specter2_chunk_pooling", specter_embeddings, specter_metadata)
    )
    write_csv(output / "specter2_chunk_manifest.csv", chunk_rows)

    run_rows: list[dict] = []
    summary_rows: list[dict] = []
    neighbors: list[dict] = []
    assignments: dict[str, object] = {
        "paper_id": rows["paper_id"],
        "title": rows["title"],
    }
    metadata_models: list[dict] = []
    for name, embeddings, metadata in model_outputs:
        np.save(output / f"embeddings_{name}.npy", embeddings)
        model_runs, model_summary, selection = probe_representation(
            name,
            embeddings,
            range(args.k_min, args.k_max + 1),
            args.seed,
            args.seed_runs,
        )
        run_rows.extend(model_runs)
        summary_rows.extend(model_summary)
        neighbors.extend(neighbor_rows(name, embeddings, rows))
        assignments[f"{name}_probe_cluster"] = selection.pop("labels")
        metadata["probe_selection"] = selection
        metadata_models.append(metadata)

    write_csv(output / "probe_runs.csv", run_rows)
    write_csv(output / "probe_summary.csv", summary_rows)
    write_csv(output / "nearest_neighbors_top5.csv", neighbors)
    pd.DataFrame(assignments).to_csv(
        output / "probe_cluster_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "experiment": "B",
        "status": (
            "contextual embedding comparison; probe clusters are not final; "
            "BGE-M3 pending"
        ),
        "keyword": args.keyword,
        "paper_count": int(len(rows)),
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "input_representation": "Title + Abstract + final K12 sub-document",
        "shared_metadata_input": args.experiment_a_input,
        "subdocument_source": args.subdocuments,
        "seed": args.seed,
        "seed_runs": args.seed_runs,
        "k_range": [args.k_min, args.k_max],
        "models": metadata_models,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "adapters": adapters.__version__,
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote Experiment B outputs to {output}", flush=True)


if __name__ == "__main__":
    main()
