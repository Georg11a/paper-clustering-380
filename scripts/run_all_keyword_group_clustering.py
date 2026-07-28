#!/usr/bin/env python3
"""Build complete keyword-conditioned assignments before topic modeling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from run_batch3_confirmatory_benchmark import (
    canonicalize_labels,
    pairwise_cosine_similarity,
    permutation_cohesion,
    spectral_fit,
    subsample_stability,
)


KEYWORD_CODES = {
    "design knowledge": "DK",
    "design theory": "DT",
    "design patterns": "DPAT",
    "design methods": "DM",
    "design guidelines": "DG",
    "design principles": "DPRI",
    "design rationale": "DRAT",
    "design rules": "DRUL",
    "design heuristics": "DH",
    "design frameworks": "DF",
    "design procedures": "DPROC",
}


def elbow_k(values: np.ndarray, maximum_k: int, seed: int) -> tuple[int, list[dict]]:
    """Estimate k by maximum distance from the endpoint chord of inertia."""
    rows = []
    for k in range(1, maximum_k + 1):
        model = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(values)
        rows.append({"k": k, "kmeans_inertia": float(model.inertia_)})
    if maximum_k <= 2:
        selected = maximum_k
    else:
        x = np.arange(1, maximum_k + 1, dtype=float)
        y = np.asarray([row["kmeans_inertia"] for row in rows], dtype=float)
        x = (x - x.min()) / max(x.max() - x.min(), 1e-12)
        y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
        start = np.array([x[0], y[0]])
        end = np.array([x[-1], y[-1]])
        line = end - start
        distances = np.abs(
            line[0] * (start[1] - y) - (start[0] - x) * line[1]
        ) / max(np.linalg.norm(line), 1e-12)
        distances[[0, -1]] = -np.inf
        selected = int(np.argmax(distances) + 1)
        for row, distance in zip(rows, distances):
            row["elbow_chord_distance"] = (
                float(distance) if np.isfinite(distance) else 0.0
            )
    for row in rows:
        row["elbow_selected"] = row["k"] == selected
    return selected, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--design-theory-assignment", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--minimum-cluster-size", type=int, default=3)
    parser.add_argument("--minimum-cluster-fraction", type=float, default=0.15)
    parser.add_argument("--maximum-k", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--expected-paper-count", type=int)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    papers = pd.read_csv(args.input).reset_index(drop=True)
    values = normalize(np.load(args.embeddings), norm="l2")
    if len(papers) != len(values):
        raise ValueError("Input and embedding rows do not match.")
    expected_count = args.expected_paper_count or len(papers)
    if (
        len(papers) != expected_count
        or papers["paper_id"].nunique() != expected_count
    ):
        raise ValueError(
            f"Expected exactly {expected_count} unique papers, found "
            f"{len(papers)} rows and {papers['paper_id'].nunique()} IDs."
        )

    frozen_dt = pd.read_csv(args.design_theory_assignment)
    frozen_dt_map = frozen_dt.set_index("paper_id")["cluster_index"].to_dict()
    assignment_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    elbow_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    for group_number, (keyword, group_frame) in enumerate(
        papers.groupby("keyword", sort=True)
    ):
        indices = group_frame.index.to_numpy()
        group_values = values[indices]
        normalized_keyword = str(keyword).casefold()
        code = KEYWORD_CODES.get(
            normalized_keyword,
            "".join(word[0] for word in normalized_keyword.split()).upper(),
        )
        n = len(group_frame)
        group_minimum_size = max(
            args.minimum_cluster_size,
            int(math.ceil(args.minimum_cluster_fraction * n)),
        )

        if normalized_keyword == "design theory":
            missing = set(group_frame["paper_id"]) - set(frozen_dt_map)
            if missing:
                raise ValueError(
                    f"Frozen Design Theory assignment misses {sorted(missing)}"
                )
            labels = np.asarray(
                [frozen_dt_map[paper_id] for paper_id in group_frame["paper_id"]],
                dtype=int,
            )
            selected_k = int(len(np.unique(labels)))
            status = "frozen_prior_pilot"
            method = "spectral_knn10"
            elbow_estimate = None
        elif n < 2 * group_minimum_size:
            labels = np.zeros(n, dtype=int)
            selected_k = 1
            status = "single_group_below_clustering_threshold"
            method = "no_subclustering"
            elbow_estimate = None
        else:
            maximum_k = min(args.maximum_k, n // group_minimum_size)
            elbow_estimate, group_elbow = elbow_k(
                group_values, maximum_k, args.seed + group_number
            )
            for row in group_elbow:
                elbow_rows.append({"keyword": keyword, "paper_count": n, **row})

            candidates: list[dict[str, object]] = []
            candidate_labels: dict[int, np.ndarray] = {}
            similarities = pairwise_cosine_similarity(group_values)
            for k in range(2, maximum_k + 1):
                fit = lambda sample, seed, k=k: spectral_fit(
                    sample, seed, k, args.neighbors
                )
                labels_k = canonicalize_labels(
                    fit(group_values, args.seed + group_number)
                )
                candidate_labels[k] = labels_k
                counts = np.unique(labels_k, return_counts=True)[1]
                cohesion = permutation_cohesion(
                    similarities,
                    labels_k,
                    args.permutations,
                    args.seed + group_number * 1000 + k,
                )
                stability, _ = subsample_stability(
                    group_values,
                    labels_k,
                    fit,
                    args.bootstrap_repeats,
                    args.bootstrap_fraction,
                    args.seed + group_number * 10_000 + k,
                )
                row = {
                    "keyword": keyword,
                    "paper_count": n,
                    "config": f"spectral_knn{args.neighbors}_k{k}",
                    "k": k,
                    "elbow_estimated_k": elbow_estimate,
                    "smallest_cluster": int(counts.min()),
                    "largest_cluster": int(counts.max()),
                    "silhouette_cosine": float(
                        silhouette_score(group_values, labels_k, metric="cosine")
                    ),
                    **cohesion,
                    **stability,
                }
                row["passes_numeric_rule"] = (
                    row["smallest_cluster"] >= group_minimum_size
                    and row["permutation_adjusted_z_rho"] > 2
                    and row["bootstrap_pairwise_ari_mean"] >= 0.6
                )
                candidates.append(row)
                metric_rows.append(row)

            passing = [row for row in candidates if row["passes_numeric_rule"]]
            if passing:
                chosen = min(
                    passing,
                    key=lambda row: (
                        abs(int(row["k"]) - int(elbow_estimate)),
                        -float(row["bootstrap_pairwise_ari_mean"]),
                    ),
                )
                selected_k = int(chosen["k"])
                status = "numerically_selected_pending_human_veto"
            else:
                # Preserve the computationally estimated k rather than
                # silently substituting the most stable post-hoc candidate.
                selected_k = int(elbow_estimate)
                status = "numeric_rule_failed_requires_review"
            labels = candidate_labels[selected_k]
            method = f"spectral_knn{args.neighbors}"

        counts = pd.Series(labels).value_counts().sort_index().to_dict()
        group_rows.append(
            {
                "keyword": keyword,
                "paper_count": n,
                "minimum_cluster_size": group_minimum_size,
                "elbow_estimated_k": elbow_estimate,
                "selected_k": selected_k,
                "assignment_method": method,
                "assignment_status": status,
                "cluster_sizes": "/".join(str(counts[key]) for key in sorted(counts)),
            }
        )
        for local_position, (_, paper) in enumerate(group_frame.iterrows()):
            cluster_index = int(labels[local_position])
            assignment_rows.append(
                {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "keyword": keyword,
                    "cluster_id": f"{code}-C{cluster_index + 1:02d}",
                    "cluster_index": cluster_index,
                    "group_k": selected_k,
                    "assignment_method": method,
                    "assignment_status": status,
                }
            )

    assignments = pd.DataFrame(assignment_rows)
    if (
        len(assignments) != expected_count
        or assignments["paper_id"].nunique() != expected_count
    ):
        raise RuntimeError("Complete assignment is not one row per paper.")
    assignment_name = (
        f"all_{expected_count}_keyword_conditioned_assignments.csv"
    )
    assignments.sort_values(["keyword", "cluster_id", "paper_id"]).to_csv(
        out / assignment_name,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(group_rows).sort_values(
        "paper_count", ascending=False
    ).to_csv(out / "keyword_group_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(
        out / "eligible_group_candidate_metrics.csv", index=False
    )
    pd.DataFrame(elbow_rows).to_csv(
        out / "keyword_group_elbow_curves.csv", index=False
    )
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "paper_count": expected_count,
                "representation": (
                    "BGE-M3 Title+Abstract+K12; K12 selected using each "
                    "paper's predefined keyword"
                ),
                "clustering_scope": "within predefined keyword group only",
                "cluster_method": f"spectral_knn{args.neighbors}",
                "k_estimation": (
                    "K-means inertia elbow by maximum endpoint-chord "
                    "distance; candidate must pass frozen numeric validation"
                ),
                "small_group_rule": (
                    "A group receives k=1 only when it cannot form two "
                    "clusters satisfying max(absolute minimum "
                    f"{args.minimum_cluster_size}, ceil("
                    f"{args.minimum_cluster_fraction} * n))."
                ),
                "topic_modeling_started": False,
                "random_seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(group_rows).sort_values(
        "paper_count", ascending=False
    ).to_string(index=False))
    print(
        f"Wrote complete {expected_count}-paper assignment to "
        f"{out / assignment_name}"
    )


if __name__ == "__main__":
    main()
