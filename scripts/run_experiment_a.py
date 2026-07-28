#!/usr/bin/env python3
"""Run the controlled Title + Abstract embedding comparison.

Experiment A keeps the input identical across TF-IDF–SVD and SPECTER2. The
K-means runs are representation probes only; they do not freeze final clusters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
from itertools import combinations
from pathlib import Path

import adapters
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from adapters import AutoAdapterModel
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def normalized_text(value: object) -> str:
    return " ".join(str(value if value is not None else "").split())


def build_input(manifest: Path, keyword: str) -> pd.DataFrame:
    frame = pd.read_csv(manifest).fillna("")
    rows = frame[frame["keyword"].str.casefold() == keyword.casefold()].copy()
    if rows.empty:
        raise RuntimeError(f"No papers found for keyword {keyword!r}")
    rows["title"] = rows["canonical_title"].map(normalized_text)
    rows["abstract"] = rows["abstract"].map(normalized_text)
    rows["experiment_a_text"] = rows.apply(
        lambda row: f"{row['title']} [SEP] {row['abstract']}", axis=1
    )
    missing = rows[(rows["title"] == "") | (rows["abstract"] == "")]
    if not missing.empty:
        raise RuntimeError(
            f"Experiment A requires Title + Abstract for every paper; "
            f"missing for {missing['paper_id'].tolist()}"
        )
    columns = ["paper_id", "keyword", "title", "abstract", "experiment_a_text"]
    return rows[columns].sort_values("paper_id").reset_index(drop=True)


def apply_shared_token_cap(
    frame: pd.DataFrame,
    tokenizer_path: Path,
    max_tokens: int,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True
    )
    rows = frame.copy()
    capped_texts: list[str] = []
    original_counts: list[int] = []
    capped_counts: list[int] = []
    truncated: list[bool] = []
    for text in rows["experiment_a_text"].astype(str):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        kept_ids = token_ids[:max_tokens]
        capped_texts.append(
            tokenizer.decode(
                kept_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
        )
        original_counts.append(len(token_ids))
        capped_counts.append(len(kept_ids))
        truncated.append(len(token_ids) > max_tokens)
    rows["experiment_a_text_original"] = rows["experiment_a_text"]
    rows["experiment_a_text"] = capped_texts
    rows["original_shared_token_count"] = original_counts
    rows["shared_token_count"] = capped_counts
    rows["shared_truncated"] = truncated
    return rows


def tfidf_svd_embeddings(
    texts: list[str], seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=8000,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    dimensions = min(50, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if dimensions < 2:
        raise RuntimeError(f"TF-IDF matrix is too small for SVD: {matrix.shape}")
    svd = TruncatedSVD(n_components=dimensions, random_state=seed)
    embeddings = normalize(svd.fit_transform(matrix), norm="l2")
    metadata = {
        "model": "tfidf_svd",
        "vocabulary_size": int(matrix.shape[1]),
        "dimensions": int(dimensions),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "parameters": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 8000,
            "sublinear_tf": True,
            "stop_words": "english",
        },
    }
    return embeddings.astype(np.float32), metadata


def specter2_embeddings(
    texts: list[str],
    base_path: Path,
    adapter_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True)
    model = AutoAdapterModel.from_pretrained(str(base_path), local_files_only=True)
    adapter_name = model.load_adapter(
        str(adapter_path), load_as="proximity", set_active=True
    )
    model.set_active_adapters(adapter_name)
    model.eval()

    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        with torch.inference_mode():
            output = model(**inputs).last_hidden_state[:, 0, :]
        vectors.append(output.detach().cpu().numpy())
    embeddings = normalize(np.vstack(vectors), norm="l2").astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise RuntimeError("SPECTER2 produced non-finite values")
    metadata = {
        "model": "specter2",
        "base_path": str(base_path),
        "adapter_path": str(adapter_path),
        "adapter": str(adapter_name),
        "adapter_active": str(model.active_adapters),
        "dimensions": int(embeddings.shape[1]),
        "max_length": 512,
        "pooling": "last_hidden_state[:, 0, :] (CLS)",
        "batch_size": batch_size,
    }
    return embeddings, metadata


def probe_representation(
    model_name: str,
    embeddings: np.ndarray,
    k_values: range,
    seed: int,
    seed_runs: int,
) -> tuple[list[dict], list[dict], dict]:
    run_rows: list[dict] = []
    summary_rows: list[dict] = []
    labels_by_k: dict[int, list[np.ndarray]] = {}
    for k in k_values:
        labels_by_k[k] = []
        silhouettes = []
        for offset in range(seed_runs):
            run_seed = seed + offset
            labels = KMeans(
                n_clusters=k,
                random_state=run_seed,
                n_init=20,
                max_iter=500,
            ).fit_predict(embeddings)
            labels_by_k[k].append(labels)
            score = float(silhouette_score(embeddings, labels, metric="cosine"))
            silhouettes.append(score)
            run_rows.append(
                {
                    "model": model_name,
                    "k": k,
                    "seed": run_seed,
                    "silhouette_cosine": score,
                    "smallest_cluster": int(np.bincount(labels).min()),
                    "largest_cluster": int(np.bincount(labels).max()),
                }
            )
        pairwise_ari = [
            adjusted_rand_score(a, b)
            for a, b in combinations(labels_by_k[k], 2)
        ]
        summary_rows.append(
            {
                "model": model_name,
                "k": k,
                "mean_silhouette_cosine": float(np.mean(silhouettes)),
                "sd_silhouette_cosine": float(np.std(silhouettes)),
                "mean_seed_ari": float(np.mean(pairwise_ari)),
                "minimum_seed_ari": float(np.min(pairwise_ari)),
            }
        )
    selected = max(
        summary_rows,
        key=lambda row: (row["mean_silhouette_cosine"], row["mean_seed_ari"]),
    )
    selected_k = int(selected["k"])
    final_labels = KMeans(
        n_clusters=selected_k,
        random_state=seed,
        n_init=50,
        max_iter=500,
    ).fit_predict(embeddings)
    selection = {
        "model": model_name,
        "selected_probe_k": selected_k,
        "selection_rule": "maximum mean cosine silhouette; seed ARI breaks ties",
        "mean_silhouette_cosine": selected["mean_silhouette_cosine"],
        "mean_seed_ari": selected["mean_seed_ari"],
        "labels": final_labels,
    }
    return run_rows, summary_rows, selection


def neighbor_rows(
    model_name: str,
    embeddings: np.ndarray,
    input_frame: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    similarities = cosine_similarity(embeddings)
    rows: list[dict] = []
    for index in range(len(input_frame)):
        order = np.argsort(-similarities[index])
        order = [candidate for candidate in order if candidate != index][:top_n]
        for rank, candidate in enumerate(order, start=1):
            rows.append(
                {
                    "model": model_name,
                    "paper_id": input_frame.iloc[index]["paper_id"],
                    "title": input_frame.iloc[index]["title"],
                    "neighbor_rank": rank,
                    "neighbor_paper_id": input_frame.iloc[candidate]["paper_id"],
                    "neighbor_title": input_frame.iloc[candidate]["title"],
                    "cosine_similarity": float(similarities[index, candidate]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="outputs/batch1/canonical_analysis_input_284.csv"
    )
    parser.add_argument("--keyword", default="Design Theory")
    parser.add_argument(
        "--output", default="outputs/batch2/experiment_a_design_theory"
    )
    specter_cache = Path.home() / ".cache" / "specter2"
    parser.add_argument("--specter-base", default=str(specter_cache / "base"))
    parser.add_argument(
        "--specter-adapter",
        default=str(specter_cache / "adapters" / "proximity"),
    )
    parser.add_argument(
        "--shared-tokenizer", default=str(specter_cache / "base")
    )
    parser.add_argument("--max-shared-tokens", type=int, default=510)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tfidf_svd", "specter2"],
        default=["tfidf_svd", "specter2"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest = Path(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    input_frame = apply_shared_token_cap(
        build_input(manifest, args.keyword),
        Path(args.shared_tokenizer),
        args.max_shared_tokens,
    )
    input_path = output / "experiment_a_input.csv"
    input_frame.to_csv(input_path, index=False, encoding="utf-8-sig")

    all_run_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    all_neighbor_rows: list[dict] = []
    assignment_columns: dict[str, object] = {
        "paper_id": input_frame["paper_id"],
        "title": input_frame["title"],
    }
    model_metadata: list[dict] = []

    for model_name in args.models:
        print(f"Encoding {len(input_frame)} papers with {model_name}", flush=True)
        if model_name == "tfidf_svd":
            embeddings, metadata = tfidf_svd_embeddings(
                input_frame["experiment_a_text"].tolist(), args.seed
            )
        else:
            embeddings, metadata = specter2_embeddings(
                input_frame["experiment_a_text"].tolist(),
                Path(args.specter_base),
                Path(args.specter_adapter),
                args.batch_size,
            )
        np.save(output / f"embeddings_{model_name}.npy", embeddings)
        run_rows, summary_rows, selection = probe_representation(
            model_name,
            embeddings,
            range(args.k_min, args.k_max + 1),
            args.seed,
            args.seed_runs,
        )
        all_run_rows.extend(run_rows)
        all_summary_rows.extend(summary_rows)
        all_neighbor_rows.extend(
            neighbor_rows(model_name, embeddings, input_frame)
        )
        assignment_columns[f"{model_name}_probe_cluster"] = selection.pop("labels")
        metadata["probe_selection"] = selection
        model_metadata.append(metadata)

    write_csv(output / "probe_runs.csv", all_run_rows)
    write_csv(output / "probe_summary.csv", all_summary_rows)
    write_csv(output / "nearest_neighbors_top5.csv", all_neighbor_rows)
    pd.DataFrame(assignment_columns).to_csv(
        output / "probe_cluster_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    metadata = {
        "experiment": "A",
        "status": "embedding comparison; probe clusters are not final",
        "keyword": args.keyword,
        "paper_count": int(len(input_frame)),
        "input": str(input_path),
        "input_sha256": input_hash,
        "input_representation": "Title + Abstract",
        "shared_input_cap": {
            "tokenizer": args.shared_tokenizer,
            "maximum_tokens_without_model_special_tokens": args.max_shared_tokens,
            "truncated_papers": int(input_frame["shared_truncated"].sum()),
        },
        "seed": args.seed,
        "seed_runs": args.seed_runs,
        "k_range": [args.k_min, args.k_max],
        "models": model_metadata,
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
    print(f"Wrote Experiment A outputs to {output}", flush=True)


if __name__ == "__main__":
    main()
