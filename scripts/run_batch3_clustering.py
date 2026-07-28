#!/usr/bin/env python3
"""Evaluate and export Batch 3 clustering configurations.

The script consumes a frozen, precomputed embedding matrix. It does not
regenerate text or embeddings, so the Batch 2 representation remains fixed.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def valid_partition(labels: np.ndarray, ignore_noise: bool = False) -> bool:
    if ignore_noise:
        labels = labels[labels >= 0]
    return len(labels) >= 3 and 2 <= len(np.unique(labels)) < len(labels)


def cosine_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    mask = labels >= 0
    kept_labels = labels[mask]
    if not valid_partition(kept_labels):
        return float("nan")
    return float(silhouette_score(embeddings[mask], kept_labels, metric="cosine"))


def mean_pairwise_ari(label_runs: list[np.ndarray]) -> tuple[float, float]:
    scores = [
        adjusted_rand_score(left, right)
        for left, right in combinations(label_runs, 2)
    ]
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.min(scores))


def consensus_partition(label_runs: list[np.ndarray], n_clusters: int) -> np.ndarray:
    """Create a reproducible partition from repeated K-means co-assignment."""
    n = len(label_runs[0])
    coassignment = np.zeros((n, n), dtype=float)
    for labels in label_runs:
        coassignment += labels[:, None] == labels[None, :]
    coassignment /= len(label_runs)
    distance = 1.0 - coassignment
    np.fill_diagonal(distance, 0.0)
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance)


def bootstrap_kmeans(
    embeddings: np.ndarray,
    k: int,
    repeats: int,
    fraction: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    sample_size = max(k + 2, int(round(n * fraction)))
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(repeats):
        indices = np.sort(rng.choice(n, size=sample_size, replace=False))
        labels = KMeans(
            n_clusters=k,
            n_init=50,
            random_state=seed + repeat,
        ).fit_predict(embeddings[indices])
        runs.append((indices, labels))
    return overlap_ari(runs)


def bootstrap_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    repeats: int,
    fraction: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    sample_size = max(min_cluster_size * 2, int(round(n * fraction)))
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(repeats):
        indices = np.sort(rng.choice(n, size=sample_size, replace=False))
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            copy=True,
        ).fit_predict(embeddings[indices])
        runs.append((indices, labels))
    return overlap_ari(runs)


def overlap_ari(runs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    scores: list[float] = []
    for (left_indices, left_labels), (right_indices, right_labels) in combinations(runs, 2):
        common, left_pos, right_pos = np.intersect1d(
            left_indices, right_indices, return_indices=True
        )
        if len(common) < 3:
            continue
        left_common = left_labels[left_pos]
        right_common = right_labels[right_pos]
        if not valid_partition(left_common) or not valid_partition(right_common):
            continue
        scores.append(adjusted_rand_score(left_common, right_common))
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.min(scores))


def cluster_sizes(labels: np.ndarray) -> tuple[int, int, int, float]:
    kept = labels[labels >= 0]
    counts = np.unique(kept, return_counts=True)[1] if len(kept) else np.array([])
    cluster_count = int(len(counts))
    smallest = int(counts.min()) if len(counts) else 0
    largest = int(counts.max()) if len(counts) else 0
    noise_fraction = float(np.mean(labels < 0))
    return cluster_count, smallest, largest, noise_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k-values", default="2,3,4,5,6,7,8")
    parser.add_argument("--seeds", default="11,23,37,53,71,89,107,131,151,173")
    parser.add_argument("--min-cluster-sizes", default="3,4,5,6,8,10,12")
    parser.add_argument("--min-samples-values", default="1,2,3,4,5")
    parser.add_argument("--bootstrap-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument("--max-eligible-clusters", type=int, default=8)
    parser.add_argument("--min-eligible-cluster-size", type=int, default=3)
    parser.add_argument(
        "--exclude-paper-ids",
        default="",
        help="Comma-separated paper IDs for a documented sensitivity run.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    papers = pd.read_csv(args.input)
    embeddings = np.load(args.embeddings)
    if len(papers) != len(embeddings):
        raise ValueError(
            f"Input has {len(papers)} papers but embeddings have {len(embeddings)} rows."
        )
    excluded_ids = {
        item.strip() for item in args.exclude_paper_ids.split(",") if item.strip()
    }
    if excluded_ids:
        keep = ~papers["paper_id"].astype(str).isin(excluded_ids)
        embeddings = embeddings[keep.to_numpy()]
        papers = papers.loc[keep].reset_index(drop=True)
    embeddings = normalize(embeddings, norm="l2")

    seeds = parse_ints(args.seeds)
    metric_rows: list[dict[str, object]] = []
    assignment_columns: dict[str, np.ndarray] = {}

    for k in parse_ints(args.k_values):
        label_runs = [
            KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(embeddings)
            for seed in seeds
        ]
        seed_ari_mean, seed_ari_min = mean_pairwise_ari(label_runs)
        bootstrap_mean, bootstrap_min = bootstrap_kmeans(
            embeddings,
            k,
            args.bootstrap_repeats,
            args.bootstrap_fraction,
            args.bootstrap_seed + k,
        )
        labels = consensus_partition(label_runs, k)
        cluster_count, smallest, largest, noise_fraction = cluster_sizes(labels)
        config = f"kmeans_k{k}"
        assignment_columns[config] = labels
        metric_rows.append(
            {
                "algorithm": "kmeans",
                "config": config,
                "requested_k": k,
                "min_cluster_size": "",
                "min_samples": "",
                "cluster_count": cluster_count,
                "silhouette_cosine": cosine_silhouette(embeddings, labels),
                "seed_ari_mean": seed_ari_mean,
                "seed_ari_min": seed_ari_min,
                "bootstrap_ari_mean": bootstrap_mean,
                "bootstrap_ari_min": bootstrap_min,
                "noise_fraction": noise_fraction,
                "smallest_cluster": smallest,
                "largest_cluster": largest,
            }
        )

    for min_cluster_size in parse_ints(args.min_cluster_sizes):
        for min_samples in parse_ints(args.min_samples_values):
            if min_samples > min_cluster_size:
                continue
            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                copy=True,
            ).fit_predict(embeddings)
            bootstrap_mean, bootstrap_min = bootstrap_hdbscan(
                embeddings,
                min_cluster_size,
                min_samples,
                args.bootstrap_repeats,
                args.bootstrap_fraction,
                args.bootstrap_seed + 100 * min_cluster_size + min_samples,
            )
            cluster_count, smallest, largest, noise_fraction = cluster_sizes(labels)
            config = f"hdbscan_mcs{min_cluster_size}_ms{min_samples}"
            assignment_columns[config] = labels
            metric_rows.append(
                {
                    "algorithm": "hdbscan",
                    "config": config,
                    "requested_k": "",
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "cluster_count": cluster_count,
                    "silhouette_cosine": cosine_silhouette(embeddings, labels),
                    "seed_ari_mean": "",
                    "seed_ari_min": "",
                    "bootstrap_ari_mean": bootstrap_mean,
                    "bootstrap_ari_min": bootstrap_min,
                    "noise_fraction": noise_fraction,
                    "smallest_cluster": smallest,
                    "largest_cluster": largest,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics["eligible"] = (
        metrics["cluster_count"].between(2, args.max_eligible_clusters)
        & (metrics["smallest_cluster"] >= args.min_eligible_cluster_size)
        & (metrics["noise_fraction"] <= 0.35)
        & metrics["silhouette_cosine"].notna()
        & metrics["bootstrap_ari_mean"].notna()
    )
    eligible = metrics["eligible"]
    for column in ["silhouette_cosine", "bootstrap_ari_mean"]:
        ranked = metrics.loc[eligible, column].rank(pct=True, method="average")
        metrics.loc[eligible, f"{column}_percentile"] = ranked
    metrics["selection_score"] = np.nan
    metrics.loc[eligible, "selection_score"] = (
        0.4 * metrics.loc[eligible, "silhouette_cosine_percentile"]
        + 0.6 * metrics.loc[eligible, "bootstrap_ari_mean_percentile"]
    )
    metrics = metrics.sort_values(
        ["eligible", "selection_score", "silhouette_cosine"],
        ascending=[False, False, False],
        na_position="last",
    )

    assignments = papers[["paper_id", "title"]].copy()
    for config, labels in assignment_columns.items():
        assignments[config] = labels
    metrics.to_csv(out / "configuration_metrics.csv", index=False)
    assignments.to_csv(out / "all_cluster_assignments.csv", index=False)

    top_configs = metrics.loc[metrics["eligible"], "config"].head(5).tolist()
    candidate_rows = []
    profile_rows = []
    text_column = "contextual_text"
    if text_column not in papers.columns:
        required_text = {"title", "abstract", "subdocument"}
        missing_text = required_text - set(papers.columns)
        if missing_text:
            raise ValueError(
                f"Input CSV lacks {text_column!r} and fallback columns "
                f"{sorted(missing_text)}."
            )
        papers[text_column] = (
            papers["title"].fillna("")
            + " [SEP] "
            + papers["abstract"].fillna("")
            + " [SEP] "
            + papers["subdocument"].fillna("")
        )
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        max_features=5000,
    )
    text_vectors = vectorizer.fit_transform(papers[text_column].fillna(""))
    feature_names = vectorizer.get_feature_names_out()
    for config in top_configs:
        labels = assignment_columns[config]
        representative_ranks: dict[int, dict[int, int]] = {}
        for label in sorted(set(labels) - {-1}):
            member_indices = np.flatnonzero(labels == label)
            centroid = normalize(
                embeddings[member_indices].mean(axis=0, keepdims=True), norm="l2"
            )[0]
            similarities = embeddings[member_indices] @ centroid
            ranked_positions = np.argsort(-similarities)
            representative_ranks[label] = {
                int(member_indices[position]): rank + 1
                for rank, position in enumerate(ranked_positions[:5])
            }
            mean_terms = np.asarray(text_vectors[member_indices].mean(axis=0)).ravel()
            top_term_indices = np.argsort(-mean_terms)[:12]
            representative_titles = [
                str(papers.iloc[member_indices[position]]["title"])
                for position in ranked_positions[:5]
            ]
            profile_rows.append(
                {
                    "config": config,
                    "cluster": int(label),
                    "paper_count": int(len(member_indices)),
                    "top_terms": " | ".join(feature_names[top_term_indices]),
                    "representative_titles": " | ".join(representative_titles),
                }
            )
        for index, label in enumerate(labels):
            candidate_rows.append(
                {
                    "config": config,
                    "paper_id": papers.iloc[index]["paper_id"],
                    "title": papers.iloc[index]["title"],
                    "cluster": int(label),
                    "is_noise": bool(label < 0),
                    "representative_rank": representative_ranks.get(
                        int(label), {}
                    ).get(index, ""),
                }
            )
    pd.DataFrame(candidate_rows).to_csv(out / "top_candidate_memberships.csv", index=False)
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(out / "candidate_cluster_profiles.csv", index=False)

    if top_configs:
        best_config = top_configs[0]
        best_labels = assignment_columns[best_config]
        best_memberships = pd.DataFrame(
            {
                "paper_id": papers["paper_id"],
                "title": papers["title"],
                "abstract": papers.get("abstract", ""),
                "passage_ids": papers.get("passage_ids", ""),
                "cluster": best_labels,
            }
        )
        best_profiles = profiles.loc[
            profiles["config"] == best_config,
            ["cluster", "paper_count", "top_terms", "representative_titles"],
        ]
        audit = best_memberships.merge(best_profiles, on="cluster", how="left")
        audit["reviewer_membership"] = ""
        audit["reviewer_suggested_cluster"] = ""
        audit["reviewer_notes"] = ""
        audit.to_csv(out / "human_audit_shortlist.csv", index=False)

    metadata = {
        "input": str(Path(args.input)),
        "embeddings": str(Path(args.embeddings)),
        "paper_count": int(len(papers)),
        "embedding_dimensions": int(embeddings.shape[1]),
        "embedding_normalization": "L2",
        "excluded_paper_ids": sorted(excluded_ids),
        "k_values": parse_ints(args.k_values),
        "seeds": seeds,
        "min_cluster_sizes": parse_ints(args.min_cluster_sizes),
        "min_samples_values": parse_ints(args.min_samples_values),
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_fraction": args.bootstrap_fraction,
        "selection_score": (
            "0.4 * silhouette percentile + 0.6 * bootstrap ARI percentile; "
            "diagnostic shortlist only, pending human coherence audit"
        ),
        "kmeans_assignment": (
            "Consensus partition from the co-assignment matrix across all listed seeds"
        ),
        "top_configs": top_configs,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(metrics.head(12).to_string(index=False))
    print(f"Wrote Batch 3 diagnostics to {out}")


if __name__ == "__main__":
    main()
