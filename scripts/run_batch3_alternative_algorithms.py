#!/usr/bin/env python3
"""Compare non-K-means clustering families on frozen Batch 2 embeddings."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
    OPTICS,
    SpectralClustering,
)
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score
from sklearn.preprocessing import normalize


FitFunction = Callable[[np.ndarray, int], np.ndarray]


def valid_partition(labels: np.ndarray) -> bool:
    kept = labels[labels >= 0]
    return len(kept) >= 3 and 2 <= len(np.unique(kept)) < len(kept)


def partition_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    kept = labels >= 0
    kept_labels = labels[kept]
    counts = (
        np.unique(kept_labels, return_counts=True)[1]
        if len(kept_labels)
        else np.array([])
    )
    return {
        "cluster_count": int(len(counts)),
        "silhouette_cosine": (
            float(silhouette_score(x[kept], kept_labels, metric="cosine"))
            if valid_partition(labels)
            else float("nan")
        ),
        "noise_fraction": float(np.mean(labels < 0)),
        "smallest_cluster": int(counts.min()) if len(counts) else 0,
        "largest_cluster": int(counts.max()) if len(counts) else 0,
    }


def overlap_stability(
    x: np.ndarray,
    fit: FitFunction,
    repeats: int,
    fraction: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample_size = int(round(len(x) * fraction))
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(repeats):
        indices = np.sort(rng.choice(len(x), size=sample_size, replace=False))
        runs.append((indices, fit(x[indices], seed + repeat)))
    scores: list[float] = []
    for (left_i, left_y), (right_i, right_y) in combinations(runs, 2):
        common, left_pos, right_pos = np.intersect1d(
            left_i, right_i, return_indices=True
        )
        if len(common) < 3:
            continue
        left_common, right_common = left_y[left_pos], right_y[right_pos]
        if not valid_partition(left_common) or not valid_partition(right_common):
            continue
        scores.append(adjusted_rand_score(left_common, right_common))
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.min(scores))


def repeat_stability(label_runs: list[np.ndarray]) -> tuple[float, float]:
    scores = [
        adjusted_rand_score(left, right)
        for left, right in combinations(label_runs, 2)
        if valid_partition(left) and valid_partition(right)
    ]
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.min(scores))


def medoid_run(label_runs: list[np.ndarray]) -> np.ndarray:
    if len(label_runs) == 1:
        return label_runs[0]
    score_matrix = np.zeros((len(label_runs), len(label_runs)))
    for left, right in combinations(range(len(label_runs)), 2):
        score = adjusted_rand_score(label_runs[left], label_runs[right])
        score_matrix[left, right] = score_matrix[right, left] = score
    return label_runs[int(np.argmax(score_matrix.mean(axis=1)))]


def add_result(
    rows: list[dict[str, object]],
    assignments: dict[str, np.ndarray],
    x: np.ndarray,
    family: str,
    config: str,
    labels: np.ndarray,
    bootstrap: tuple[float, float],
    repeat: tuple[float, float] = (float("nan"), float("nan")),
) -> None:
    assignments[config] = labels
    rows.append(
        {
            "algorithm": family,
            "config": config,
            **partition_metrics(x, labels),
            "repeat_ari_mean": repeat[0],
            "repeat_ari_min": repeat[1],
            "bootstrap_ari_mean": bootstrap[0],
            "bootstrap_ari_min": bootstrap[1],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=15)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--min-eligible-cluster-size", type=int, default=3)
    parser.add_argument("--max-noise-fraction", type=float, default=0.35)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    papers = pd.read_csv(args.input)
    x = normalize(np.load(args.embeddings), norm="l2")
    if len(papers) != len(x):
        raise ValueError("Paper and embedding counts do not match.")

    rows: list[dict[str, object]] = []
    assignments: dict[str, np.ndarray] = {}

    # Cosine hierarchical clustering: average is the natural non-Euclidean
    # linkage; complete is retained as a sensitivity alternative.
    for linkage in ["average", "complete"]:
        for k in range(2, args.k_max + 1):
            fit = lambda values, _seed, k=k, linkage=linkage: AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage=linkage
            ).fit_predict(values)
            labels = fit(x, args.seed)
            bootstrap = overlap_stability(
                x, fit, args.bootstrap_repeats, args.bootstrap_fraction, args.seed + k
            )
            add_result(
                rows,
                assignments,
                x,
                "agglomerative",
                f"agg_{linkage}_k{k}",
                labels,
                bootstrap,
            )

    # DBSCAN eps values are derived from the observed cosine k-distance range.
    density_min_samples = [3, 5, 8, 10] if len(x) > 100 else [2, 3, 4, 5, 6]
    for min_samples in density_min_samples:
        for eps in np.arange(0.10, 0.31, 0.01):
            fit = lambda values, _seed, eps=eps, ms=min_samples: DBSCAN(
                eps=float(eps), min_samples=ms, metric="cosine"
            ).fit_predict(values)
            labels = fit(x, args.seed)
            bootstrap = overlap_stability(
                x,
                fit,
                args.bootstrap_repeats,
                args.bootstrap_fraction,
                args.seed + min_samples,
            )
            add_result(
                rows,
                assignments,
                x,
                "dbscan",
                f"dbscan_eps{eps:.2f}_ms{min_samples}",
                labels,
                bootstrap,
            )

    # OPTICS explores multiple density radii rather than fixing one eps.
    for min_samples in density_min_samples:
        for xi in [0.03, 0.05, 0.08, 0.10]:
            fit = lambda values, _seed, ms=min_samples, xi=xi: OPTICS(
                min_samples=ms,
                xi=xi,
                min_cluster_size=3,
                metric="cosine",
                cluster_method="xi",
            ).fit_predict(values)
            labels = fit(x, args.seed)
            bootstrap = overlap_stability(
                x,
                fit,
                args.bootstrap_repeats,
                args.bootstrap_fraction,
                args.seed + min_samples,
            )
            add_result(
                rows,
                assignments,
                x,
                "optics",
                f"optics_xi{xi:.2f}_ms{min_samples}",
                labels,
                bootstrap,
            )

    # Spectral clustering checks community structure in a kNN affinity graph.
    spectral_seeds = [11, 23, 37, 53, 71]
    for neighbors in [5, 8, 10, 15]:
        for k in range(2, args.k_max + 1):
            def spectral_fit(
                values: np.ndarray,
                seed: int,
                k: int = k,
                neighbors: int = neighbors,
            ) -> np.ndarray:
                return SpectralClustering(
                    n_clusters=k,
                    affinity="nearest_neighbors",
                    n_neighbors=min(neighbors, len(values) - 1),
                    assign_labels="kmeans",
                    n_init=20,
                    random_state=seed,
                ).fit_predict(values)

            label_runs = [spectral_fit(x, seed) for seed in spectral_seeds]
            labels = medoid_run(label_runs)
            bootstrap = overlap_stability(
                x,
                spectral_fit,
                args.bootstrap_repeats,
                args.bootstrap_fraction,
                args.seed + neighbors + k,
            )
            add_result(
                rows,
                assignments,
                x,
                "spectral",
                f"spectral_knn{neighbors}_k{k}",
                labels,
                bootstrap,
                repeat_stability(label_runs),
            )

    # BERTopic-style neural topic pipeline: UMAP to 5D, then HDBSCAN.
    umap_seeds = [11, 23, 37, 53, 71]
    umap_cluster_sizes = (
        [5, 8, 10, 12, 15, 20]
        if len(x) > 100
        else [3, 4, 5, 6, 8, 10]
    )
    for neighbors in [5, 10, 15]:
        for min_cluster_size in umap_cluster_sizes:
            def umap_hdbscan_fit(
                values: np.ndarray,
                seed: int,
                neighbors: int = neighbors,
                mcs: int = min_cluster_size,
            ) -> np.ndarray:
                reduced = umap.UMAP(
                    n_neighbors=min(neighbors, len(values) - 1),
                    n_components=5,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=seed,
                    n_jobs=1,
                ).fit_transform(values)
                return HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=max(1, mcs // 2),
                    metric="euclidean",
                    copy=True,
                ).fit_predict(reduced)

            label_runs = [umap_hdbscan_fit(x, seed) for seed in umap_seeds]
            labels = medoid_run(label_runs)
            bootstrap = overlap_stability(
                x,
                umap_hdbscan_fit,
                args.bootstrap_repeats,
                args.bootstrap_fraction,
                args.seed + neighbors + min_cluster_size,
            )
            add_result(
                rows,
                assignments,
                x,
                "umap_hdbscan",
                f"umap{neighbors}_hdbscan_mcs{min_cluster_size}",
                labels,
                bootstrap,
                repeat_stability(label_runs),
            )

    metrics = pd.DataFrame(rows)
    metrics["eligible"] = (
        metrics["cluster_count"].between(2, args.k_max)
        & (metrics["smallest_cluster"] >= args.min_eligible_cluster_size)
        & (metrics["noise_fraction"] <= args.max_noise_fraction)
        & metrics["silhouette_cosine"].notna()
        & metrics["bootstrap_ari_mean"].notna()
    )
    eligible = metrics["eligible"]
    metrics.loc[eligible, "silhouette_percentile"] = metrics.loc[
        eligible, "silhouette_cosine"
    ].rank(pct=True)
    metrics.loc[eligible, "stability_percentile"] = metrics.loc[
        eligible, "bootstrap_ari_mean"
    ].rank(pct=True)
    metrics.loc[eligible, "selection_score"] = (
        0.4 * metrics.loc[eligible, "silhouette_percentile"]
        + 0.6 * metrics.loc[eligible, "stability_percentile"]
    )
    metrics = metrics.sort_values(
        ["eligible", "selection_score", "silhouette_cosine"],
        ascending=[False, False, False],
        na_position="last",
    )
    metrics.to_csv(out / "alternative_configuration_metrics.csv", index=False)

    top_configs = metrics.loc[metrics["eligible"], "config"].head(10).tolist()
    membership_rows: list[dict[str, object]] = []
    for config in top_configs:
        for index, label in enumerate(assignments[config]):
            membership_rows.append(
                {
                    "config": config,
                    "paper_id": papers.iloc[index]["paper_id"],
                    "title": papers.iloc[index]["title"],
                    "cluster": int(label),
                    "is_noise": bool(label < 0),
                }
            )
    pd.DataFrame(membership_rows).to_csv(
        out / "alternative_top_memberships.csv", index=False
    )
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "paper_count": len(papers),
                "embedding": str(Path(args.embeddings)),
                "families": [
                    "agglomerative",
                    "dbscan",
                    "optics",
                    "spectral",
                    "umap_hdbscan",
                ],
                "top_configs": top_configs,
                "eligibility": (
                    f"2-{args.k_max} clusters; smallest cluster >="
                    f"{args.min_eligible_cluster_size}; noise <="
                    f"{args.max_noise_fraction:.0%}; valid silhouette and "
                    "bootstrap stability"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.head(20).to_string(index=False))
    print(f"Wrote alternative-algorithm comparison to {out}")


if __name__ == "__main__":
    main()
