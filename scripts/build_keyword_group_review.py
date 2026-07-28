#!/usr/bin/env python3
"""Build human-review materials for keyword-conditioned cluster candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--representatives-per-cluster", type=int, default=3)
    parser.add_argument("--boundary-per-cluster", type=int, default=3)
    args = parser.parse_args()

    papers = pd.read_csv(args.input).fillna("")
    assignments = pd.read_csv(args.assignments).fillna("")
    values = normalize(np.load(args.embeddings), norm="l2")
    if len(papers) != len(values):
        raise ValueError("Input and embedding rows do not match.")
    if papers["paper_id"].duplicated().any():
        raise ValueError("Input contains duplicate paper IDs.")
    if assignments["paper_id"].duplicated().any():
        raise ValueError("Assignments contain duplicate paper IDs.")

    frame = papers.merge(
        assignments[
            [
                "paper_id",
                "cluster_id",
                "cluster_index",
                "group_k",
                "assignment_method",
                "assignment_status",
            ]
        ],
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    if frame["cluster_id"].eq("").any() or frame["cluster_id"].isna().any():
        raise ValueError("At least one paper is missing a cluster assignment.")

    embedding_by_id = {
        paper_id: values[index]
        for index, paper_id in enumerate(papers["paper_id"])
    }
    ordered_values = np.vstack(
        [embedding_by_id[paper_id] for paper_id in frame["paper_id"]]
    )
    full_text = (
        frame["title"].astype(str)
        + " [SEP] "
        + frame["abstract"].astype(str)
        + " [SEP] "
        + frame["subdocument"].astype(str)
    )

    profile_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for keyword, group in frame.groupby("keyword", sort=True):
        group_indices = group.index.to_numpy()
        cluster_ids = sorted(group["cluster_id"].unique())
        centroids: dict[str, np.ndarray] = {}
        for cluster_id in cluster_ids:
            member_indices = group.index[group["cluster_id"] == cluster_id].to_numpy()
            centroids[cluster_id] = normalize(
                ordered_values[member_indices].mean(axis=0, keepdims=True),
                norm="l2",
            )[0]

        min_df = 1 if len(group) < 10 else 2
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=1.0,
            max_features=5000,
        )
        text_vectors = vectorizer.fit_transform(full_text.loc[group_indices])
        terms = vectorizer.get_feature_names_out()
        local_position = {
            global_index: position
            for position, global_index in enumerate(group_indices)
        }

        for cluster_id in cluster_ids:
            member_indices = group.index[group["cluster_id"] == cluster_id].to_numpy()
            centroid = centroids[cluster_id]
            assignment_status = str(group.iloc[0]["assignment_status"])
            requires_human_review = assignment_status in {
                "numeric_rule_failed_requires_review",
                "numerically_selected_pending_human_veto",
            }
            own_similarity = ordered_values[member_indices] @ centroid
            representative_order = member_indices[np.argsort(-own_similarity)]

            local_members = [local_position[index] for index in member_indices]
            mean_terms = np.asarray(
                text_vectors[local_members].mean(axis=0)
            ).ravel()
            top_terms = terms[np.argsort(-mean_terms)[:15]]
            representative_titles = frame.loc[
                representative_order[:5], "title"
            ].tolist()

            if len(cluster_ids) > 1:
                other_centroids = np.vstack(
                    [
                        centroids[other_id]
                        for other_id in cluster_ids
                        if other_id != cluster_id
                    ]
                )
                nearest_other = (
                    ordered_values[member_indices] @ other_centroids.T
                ).max(axis=1)
                margins = own_similarity - nearest_other
            else:
                nearest_other = np.full(len(member_indices), np.nan)
                margins = np.full(len(member_indices), np.nan)

            profile_rows.append(
                {
                    "keyword": keyword,
                    "cluster_id": cluster_id,
                    "paper_count": int(len(member_indices)),
                    "assignment_status": assignment_status,
                    "requires_human_review": requires_human_review,
                    "descriptive_tfidf_terms_not_final_topics": " | ".join(top_terms),
                    "representative_titles": " | ".join(representative_titles),
                    "mean_centroid_similarity": float(own_similarity.mean()),
                    "minimum_assignment_margin": (
                        float(np.nanmin(margins))
                        if np.isfinite(margins).any()
                        else ""
                    ),
                }
            )

            representative_set = set(
                representative_order[: args.representatives_per_cluster]
            )
            boundary_set = set()
            if len(cluster_ids) > 1:
                boundary_set = set(
                    member_indices[np.argsort(margins)[: args.boundary_per_cluster]]
                )
            selected = representative_set | boundary_set
            if not requires_human_review:
                continue
            for member_index, own, other, margin in zip(
                member_indices, own_similarity, nearest_other, margins
            ):
                if member_index not in selected:
                    continue
                roles = []
                if member_index in representative_set:
                    roles.append("representative")
                if member_index in boundary_set:
                    roles.append("boundary")
                row = frame.loc[member_index]
                review_rows.append(
                    {
                        "keyword": keyword,
                        "cluster_id": cluster_id,
                        "paper_id": row["paper_id"],
                        "title": row["title"],
                        "review_role": "+".join(roles),
                        "own_centroid_similarity": float(own),
                        "nearest_other_centroid_similarity": (
                            float(other) if np.isfinite(other) else ""
                        ),
                        "assignment_margin": (
                            float(margin) if np.isfinite(margin) else ""
                        ),
                        "assignment_status": row["assignment_status"],
                        "reviewer_accept_membership_yes_no": "",
                        "reviewer_suggested_cluster_id": "",
                        "reviewer_cluster_coherent_yes_no": "",
                        "reviewer_granularity_too_coarse_ok_too_fine": "",
                        "reviewer_notes": "",
                    }
                )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profiles = pd.DataFrame(profile_rows).sort_values(["keyword", "cluster_id"])
    review = pd.DataFrame(review_rows).sort_values(
        ["keyword", "cluster_id", "review_role", "assignment_margin"],
        na_position="last",
    )
    cluster_review = profiles.loc[profiles["requires_human_review"]].copy()
    cluster_review["reviewer_cluster_coherent_yes_no"] = ""
    cluster_review["reviewer_granularity_too_coarse_ok_too_fine"] = ""
    cluster_review["reviewer_accept_current_membership_yes_no"] = ""
    cluster_review["reviewer_accept_numeric_exception_yes_no"] = ""
    cluster_review["reviewer_notes"] = ""
    profiles.to_csv(out / "keyword_cluster_profiles_for_review.csv", index=False)
    cluster_review.to_csv(out / "human_review_clusters.csv", index=False)
    review.to_csv(out / "human_review_shortlist.csv", index=False)
    print(
        f"Wrote {len(profiles)} cluster profiles and "
        f"{len(cluster_review)} cluster-level plus {len(review)} paper-level "
        f"review rows to {out}"
    )


if __name__ == "__main__":
    main()
