#!/usr/bin/env python3
"""Confirmatory clustering benchmark on frozen Design Theory embeddings.

The script does not regenerate text or embeddings. It compares clustering
inductive biases on the same L2-normalized BGE-M3 representation and produces:

* fixed-k configuration metrics and paper assignments;
* an ARI agreement matrix at the primary k;
* Cohesion Ratio estimates with fixed-size permutation null distributions;
* full-partition and per-cluster subsample stability;
* blinded 4+1 paper-intruder tasks with a separate private answer key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    HDBSCAN,
    KMeans,
    SpectralClustering,
)
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize


FitFunction = Callable[[np.ndarray, int], np.ndarray]


def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters by first appearance while preserving noise as -1."""
    result = np.full(len(labels), -1, dtype=int)
    mapping: dict[int, int] = {}
    next_label = 0
    for index, raw_label in enumerate(labels):
        label = int(raw_label)
        if label < 0:
            continue
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        result[index] = mapping[label]
    return result


def ward_fit(values: np.ndarray, _seed: int, k: int) -> np.ndarray:
    # Values are L2-normalized. Ward itself optimizes Euclidean variance.
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(values)


def average_cosine_fit(values: np.ndarray, _seed: int, k: int) -> np.ndarray:
    return AgglomerativeClustering(
        n_clusters=k, metric="cosine", linkage="average"
    ).fit_predict(values)


def kmeans_fit(values: np.ndarray, seed: int, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(values)


def spectral_fit(
    values: np.ndarray,
    seed: int,
    k: int,
    neighbors: int,
) -> np.ndarray:
    return SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=min(neighbors, len(values) - 1),
        assign_labels="kmeans",
        n_init=20,
        random_state=seed,
    ).fit_predict(values)


def raw_hdbscan_fit(values: np.ndarray, _seed: int) -> np.ndarray:
    return HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="euclidean",
        copy=True,
    ).fit_predict(values)


def configuration_functions(k: int) -> dict[str, FitFunction]:
    return {
        f"ward_k{k}": lambda values, seed: ward_fit(values, seed, k),
        f"average_cosine_k{k}": (
            lambda values, seed: average_cosine_fit(values, seed, k)
        ),
        f"kmeans_k{k}": lambda values, seed: kmeans_fit(values, seed, k),
        f"spectral_knn10_k{k}": (
            lambda values, seed: spectral_fit(values, seed, k, 10)
        ),
        f"spectral_knn15_k{k}": (
            lambda values, seed: spectral_fit(values, seed, k, 15)
        ),
    }


def pairwise_cosine_similarity(values: np.ndarray) -> np.ndarray:
    similarities = np.clip(values @ values.T, 0.0, 1.0)
    np.fill_diagonal(similarities, np.nan)
    return similarities


def cohesion_ratio(
    similarities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, int]:
    """Return rho, within mean, global mean, and within-pair count.

    Following Neveditsin et al., singleton clusters contribute one virtual pair
    with similarity equal to the global mean. HDBSCAN noise points are treated
    as separate singleton clusters rather than as one shared noise cluster.
    """
    if len(labels) < 2:
        return float("nan"), float("nan"), float("nan"), 0

    global_mean = float(np.nanmean(similarities))
    within_values: list[np.ndarray] = []
    singleton_count = int(np.sum(labels < 0))
    for cluster in np.unique(labels[labels >= 0]):
        indices = np.flatnonzero(labels == cluster)
        if len(indices) < 2:
            singleton_count += 1
            continue
        block = similarities[np.ix_(indices, indices)]
        within_values.append(block[np.triu_indices(len(indices), k=1)])
    joined = (
        np.concatenate(within_values)
        if within_values
        else np.array([], dtype=float)
    )
    if singleton_count:
        joined = np.concatenate(
            [joined, np.full(singleton_count, global_mean, dtype=float)]
        )
    if not len(joined):
        return float("nan"), float("nan"), global_mean, 0
    within_mean = float(np.mean(joined))
    ratio = within_mean / global_mean if global_mean > 0 else float("nan")
    return ratio, within_mean, global_mean, int(len(joined))


def permutation_cohesion(
    similarities: np.ndarray,
    labels: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    observed, within_mean, global_mean, pair_count = cohesion_ratio(
        similarities, labels
    )
    rng = np.random.default_rng(seed)
    null_values = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        permuted = rng.permutation(labels)
        null_values[repeat] = cohesion_ratio(similarities, permuted)[0]
    null_mean = float(np.nanmean(null_values))
    null_sd = float(np.nanstd(null_values, ddof=1))
    z_score = (
        float((observed - null_mean) / null_sd)
        if np.isfinite(observed) and null_sd > 0
        else float("nan")
    )
    p_value = float(
        (1 + np.sum(null_values >= observed)) / (repeats + 1)
    )
    return {
        "cohesion_ratio": observed,
        "within_cosine_mean": within_mean,
        "global_cosine_mean": global_mean,
        "within_pair_count": pair_count,
        "permutation_rho_mean": null_mean,
        "permutation_rho_sd": null_sd,
        "permutation_adjusted_z_rho": z_score,
        "permutation_p_greater_equal": p_value,
    }


def best_cluster_jaccard(
    reference_labels: np.ndarray,
    sample_indices: np.ndarray,
    sample_labels: np.ndarray,
    reference_cluster: int,
) -> float:
    reference_members = set(
        sample_indices[reference_labels[sample_indices] == reference_cluster]
    )
    if not reference_members:
        return float("nan")
    scores: list[float] = []
    for sample_cluster in np.unique(sample_labels[sample_labels >= 0]):
        candidate_members = set(sample_indices[sample_labels == sample_cluster])
        union = reference_members | candidate_members
        scores.append(
            len(reference_members & candidate_members) / len(union)
            if union
            else 0.0
        )
    return max(scores, default=0.0)


def subsample_stability(
    values: np.ndarray,
    reference_labels: np.ndarray,
    fit: FitFunction,
    repeats: int,
    fraction: float,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    sample_size = max(3, int(round(len(values) * fraction)))
    runs: list[tuple[np.ndarray, np.ndarray]] = []
    reference_ari_scores: list[float] = []
    per_cluster: dict[int, list[float]] = {
        int(cluster): [] for cluster in np.unique(reference_labels)
        if cluster >= 0
    }
    for repeat in range(repeats):
        indices = np.sort(rng.choice(len(values), sample_size, replace=False))
        sample_labels = canonicalize_labels(fit(values[indices], seed + repeat))
        runs.append((indices, sample_labels))
        reference_on_sample = reference_labels[indices]
        reference_ari_scores.append(
            adjusted_rand_score(reference_on_sample, sample_labels)
        )
        for cluster in per_cluster:
            per_cluster[cluster].append(
                best_cluster_jaccard(
                    reference_labels, indices, sample_labels, cluster
                )
            )

    pairwise_ari_scores: list[float] = []
    for (left_indices, left_labels), (right_indices, right_labels) in combinations(
        runs, 2
    ):
        common, left_positions, right_positions = np.intersect1d(
            left_indices,
            right_indices,
            return_indices=True,
        )
        if len(common) < 3:
            continue
        pairwise_ari_scores.append(
            adjusted_rand_score(
                left_labels[left_positions],
                right_labels[right_positions],
            )
        )
    summary = {
        "bootstrap_pairwise_ari_mean": float(np.mean(pairwise_ari_scores)),
        "bootstrap_pairwise_ari_min": float(np.min(pairwise_ari_scores)),
        "bootstrap_pairwise_ari_sd": float(
            np.std(pairwise_ari_scores, ddof=1)
        ),
        "bootstrap_reference_ari_mean": float(
            np.mean(reference_ari_scores)
        ),
        "bootstrap_reference_ari_min": float(np.min(reference_ari_scores)),
        "bootstrap_reference_ari_sd": float(
            np.std(reference_ari_scores, ddof=1)
        ),
    }
    cluster_rows = []
    for cluster, scores in per_cluster.items():
        cluster_rows.append(
            {
                "cluster": cluster,
                "bootstrap_best_jaccard_mean": float(np.nanmean(scores)),
                "bootstrap_best_jaccard_min": float(np.nanmin(scores)),
                "bootstrap_best_jaccard_sd": float(np.nanstd(scores, ddof=1)),
            }
        )
    return summary, cluster_rows


def ari_matrix(assignments: dict[str, np.ndarray]) -> pd.DataFrame:
    configs = list(assignments)
    matrix = np.eye(len(configs), dtype=float)
    for left in range(len(configs)):
        for right in range(left + 1, len(configs)):
            score = adjusted_rand_score(
                assignments[configs[left]], assignments[configs[right]]
            )
            matrix[left, right] = matrix[right, left] = score
    return pd.DataFrame(matrix, index=configs, columns=configs)


def make_intruder_tasks(
    papers: pd.DataFrame,
    values: np.ndarray,
    assignments: dict[str, np.ndarray],
    tasks_per_cluster: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    public_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    position_labels = list("ABCDE")
    for config_index, (config, labels) in enumerate(assignments.items()):
        condition_hash = hashlib.sha256(config.encode("utf-8")).hexdigest()[:8]
        for cluster in np.unique(labels):
            if cluster < 0:
                continue
            member_indices = np.flatnonzero(labels == cluster)
            outside_indices = np.flatnonzero(labels != cluster)
            if len(member_indices) < 4 or not len(outside_indices):
                continue
            centroid = normalize(
                values[member_indices].mean(axis=0, keepdims=True)
            )[0]
            member_order = member_indices[
                np.argsort(-(values[member_indices] @ centroid))
            ]
            # Hard intruders are the nearest papers assigned outside this cluster.
            intruder_order = outside_indices[
                np.argsort(-(values[outside_indices] @ centroid))
            ]
            for task_number in range(tasks_per_cluster):
                offset = task_number % max(1, len(member_order) - 3)
                selected_members = np.roll(member_order, -offset)[:4]
                intruder = intruder_order[task_number % len(intruder_order)]
                items = [*selected_members.tolist(), int(intruder)]
                rng = np.random.default_rng(
                    seed + config_index * 10_000 + int(cluster) * 100 + task_number
                )
                rng.shuffle(items)
                task_id = (
                    f"IT-{condition_hash}-C{int(cluster):02d}-"
                    f"{task_number + 1:02d}"
                )
                public: dict[str, object] = {
                    "task_id": task_id,
                    "blind_condition": condition_hash,
                    "reviewer_intruder_position": "",
                    "reviewer_confidence_1_to_5": "",
                    "reviewer_notes": "",
                }
                for position, paper_index in zip(position_labels, items):
                    public[f"paper_{position}_title"] = papers.iloc[paper_index][
                        "title"
                    ]
                answer_position = position_labels[items.index(int(intruder))]
                public_rows.append(public)
                key_rows.append(
                    {
                        "task_id": task_id,
                        "config": config,
                        "cluster": int(cluster),
                        "correct_intruder_position": answer_position,
                        "intruder_paper_id": papers.iloc[intruder]["paper_id"],
                        "intruder_title": papers.iloc[intruder]["title"],
                    }
                )
    return pd.DataFrame(public_rows), pd.DataFrame(key_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--primary-k", type=int, default=3)
    parser.add_argument("--k-values", default="2,3,4")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--intruder-tasks-per-cluster", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    papers = pd.read_csv(args.input).reset_index(drop=True)
    values = normalize(np.load(args.embeddings), norm="l2")
    if len(papers) != len(values):
        raise ValueError(
            f"Input contains {len(papers)} papers, embeddings contain "
            f"{len(values)} rows."
        )
    if papers["paper_id"].astype(str).duplicated().any():
        raise ValueError("paper_id must be unique.")

    similarities = pairwise_cosine_similarity(values)
    k_values = sorted({int(value) for value in args.k_values.split(",")})
    all_assignments: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []

    for k in k_values:
        for config_index, (config, fit) in enumerate(
            configuration_functions(k).items()
        ):
            labels = canonicalize_labels(fit(values, args.seed))
            all_assignments[config] = labels
            counts = np.unique(labels[labels >= 0], return_counts=True)[1]
            cohesion = permutation_cohesion(
                similarities,
                labels,
                args.permutations,
                args.seed + 1000 * k + config_index,
            )
            stability, cluster_stability = subsample_stability(
                values,
                labels,
                fit,
                args.bootstrap_repeats,
                args.bootstrap_fraction,
                args.seed + 10_000 * k + config_index,
            )
            coverage = float(np.mean(labels >= 0))
            metric_rows.append(
                {
                    "config": config,
                    "algorithm": config.rsplit("_k", 1)[0],
                    "k": k,
                    "coverage": coverage,
                    "cluster_count": int(len(counts)),
                    "smallest_cluster": int(counts.min()),
                    "largest_cluster": int(counts.max()),
                    "silhouette_cosine": float(
                        silhouette_score(values, labels, metric="cosine")
                    ),
                    **cohesion,
                    **stability,
                }
            )
            for row in cluster_stability:
                cluster = int(row["cluster"])
                stability_rows.append(
                    {
                        "config": config,
                        "k": k,
                        "cluster": cluster,
                        "full_cluster_size": int(np.sum(labels == cluster)),
                        **{key: value for key, value in row.items()
                           if key != "cluster"},
                    }
                )

    # Run raw HDBSCAN once as the documented density-based negative control.
    hdbscan_labels = canonicalize_labels(raw_hdbscan_fit(values, args.seed))
    hdbscan_counts = np.unique(
        hdbscan_labels[hdbscan_labels >= 0], return_counts=True
    )[1]
    metric_rows.append(
        {
            "config": "raw_hdbscan_mcs5_ms3",
            "algorithm": "raw_hdbscan",
            "k": float("nan"),
            "coverage": float(np.mean(hdbscan_labels >= 0)),
            "cluster_count": int(len(hdbscan_counts)),
            "smallest_cluster": (
                int(hdbscan_counts.min()) if len(hdbscan_counts) else 0
            ),
            "largest_cluster": (
                int(hdbscan_counts.max()) if len(hdbscan_counts) else 0
            ),
            "silhouette_cosine": (
                float(
                    silhouette_score(
                        values[hdbscan_labels >= 0],
                        hdbscan_labels[hdbscan_labels >= 0],
                        metric="cosine",
                    )
                )
                if len(hdbscan_counts) >= 2
                else float("nan")
            ),
            **permutation_cohesion(
                similarities,
                hdbscan_labels,
                args.permutations,
                args.seed + 999_999,
            ),
            "bootstrap_pairwise_ari_mean": float("nan"),
            "bootstrap_pairwise_ari_min": float("nan"),
            "bootstrap_pairwise_ari_sd": float("nan"),
            "bootstrap_reference_ari_mean": float("nan"),
            "bootstrap_reference_ari_min": float("nan"),
            "bootstrap_reference_ari_sd": float("nan"),
        }
    )

    metrics = pd.DataFrame(metric_rows)
    metrics["passes_frozen_numeric_rule"] = (
        (metrics["coverage"] >= 0.95)
        & (metrics["smallest_cluster"] >= 8)
        & (metrics["permutation_adjusted_z_rho"] > 2)
        & (metrics["bootstrap_pairwise_ari_mean"] >= 0.6)
    )
    metrics.sort_values(
        ["k", "config"], na_position="last"
    ).to_csv(out / "configuration_metrics.csv", index=False)
    pd.DataFrame(stability_rows).sort_values(
        ["k", "config", "cluster"]
    ).to_csv(out / "per_cluster_bootstrap_stability.csv", index=False)

    assignment_frame = papers[["paper_id", "title"]].copy()
    for config, labels in all_assignments.items():
        assignment_frame[config] = labels
    assignment_frame["raw_hdbscan_mcs5_ms3"] = hdbscan_labels
    assignment_frame.to_csv(out / "paper_cluster_assignments.csv", index=False)

    primary_assignments = {
        config: labels
        for config, labels in all_assignments.items()
        if config.endswith(f"_k{args.primary_k}")
    }
    agreement = ari_matrix(primary_assignments)
    agreement.to_csv(out / f"ari_matrix_k{args.primary_k}.csv")

    disagreement_count = np.zeros(len(papers), dtype=int)
    primary_configs = list(primary_assignments)
    for index in range(len(papers)):
        labels = [
            int(primary_assignments[config][index])
            for config in primary_configs
        ]
        # Count pairwise disagreements; label identities do not align across
        # methods, so use co-membership disagreements with all other papers.
        for left in range(len(primary_configs)):
            for right in range(left + 1, len(primary_configs)):
                left_membership = (
                    primary_assignments[primary_configs[left]]
                    == primary_assignments[primary_configs[left]][index]
                )
                right_membership = (
                    primary_assignments[primary_configs[right]]
                    == primary_assignments[primary_configs[right]][index]
                )
                disagreement_count[index] += int(
                    np.sum(left_membership != right_membership)
                )
    disagreements = papers[["paper_id", "title"]].copy()
    disagreements["pairwise_comembership_disagreements"] = disagreement_count
    disagreements = disagreements.sort_values(
        ["pairwise_comembership_disagreements", "paper_id"],
        ascending=[False, True],
    )
    disagreements.to_csv(
        out / f"paper_disagreement_shortlist_k{args.primary_k}.csv", index=False
    )

    passing_configs = set(
        metrics.loc[metrics["passes_frozen_numeric_rule"], "config"]
    )
    intruder_assignments = {
        config: labels
        for config, labels in all_assignments.items()
        if config in passing_configs
    }
    intruder_public, intruder_key = make_intruder_tasks(
        papers,
        values,
        intruder_assignments,
        args.intruder_tasks_per_cluster,
        args.seed,
    )
    intruder_public.to_csv(out / "intruder_task_blinded.csv", index=False)
    intruder_key.to_csv(out / "intruder_task_private_key.csv", index=False)

    k_decisions = (
        metrics[
            metrics["algorithm"].isin(
                [
                    "ward",
                    "average_cosine",
                    "kmeans",
                    "spectral_knn10",
                    "spectral_knn15",
                ]
            )
        ]
        .groupby("k", as_index=False)
        .agg(
            passing_configurations=("passes_frozen_numeric_rule", "sum"),
            tested_configurations=("config", "count"),
            minimum_bootstrap_ari=("bootstrap_pairwise_ari_mean", "min"),
            minimum_cluster_size=("smallest_cluster", "min"),
            minimum_z_rho=("permutation_adjusted_z_rho", "min"),
        )
    )
    k_decisions["all_configurations_pass"] = (
        k_decisions["passing_configurations"]
        == k_decisions["tested_configurations"]
    )
    k_decisions.to_csv(out / "k_rule_summary.csv", index=False)

    metadata = {
        "date_frozen": "2026-07-28",
        "status": (
            "Prospectively frozen confirmation rule after exploratory "
            "diagnostics; not a preregistration."
        ),
        "paper_count": len(papers),
        "embedding_shape": list(values.shape),
        "embedding_file": str(Path(args.embeddings)),
        "input_file": str(Path(args.input)),
        "primary_k": args.primary_k,
        "tested_k_values": k_values,
        "numeric_rule": {
            "coverage_minimum": 0.95,
            "minimum_cluster_size": 8,
            "permutation_adjusted_z_rho_strictly_greater_than": 2,
            "bootstrap_pairwise_subsample_ari_minimum": 0.6,
            "tie_break": "Choose the smallest k among passing candidates.",
            "human_review": "Veto only; cannot promote a failing configuration.",
        },
        "cohesion_permutations": args.permutations,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_fraction": args.bootstrap_fraction,
        "intruder_tasks_per_cluster": args.intruder_tasks_per_cluster,
        "random_seed": args.seed,
        "notes": [
            "Ward is applied to L2-normalized vectors with Euclidean geometry.",
            "Cohesion Ratio is treated as a structure test, not a standalone "
            "selector of k.",
            "Raw HDBSCAN is a one-run negative control and is not used for "
            "fixed-k comparison.",
        ],
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(metrics.sort_values(["k", "config"], na_position="last").to_string(
        index=False
    ))
    print(f"\nARI agreement at k={args.primary_k}")
    print(agreement.to_string())
    print(f"\nWrote confirmatory benchmark to {out}")


if __name__ == "__main__":
    main()
